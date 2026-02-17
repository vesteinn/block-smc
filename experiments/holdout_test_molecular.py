"""Held-out evaluation: token-level baseline vs Block SMC vs Block SMC + twist
for the Molecular Synthesis domain.

Key differences from Goal Inference:
  - PartialSMILES potential gives real negative signal at intermediate boundaries
  - max_tokens=40 (short sequences)
  - EOS = newline tokens
  - QED score (0-1 continuous) as evaluation metric
  - Same potential for all instances (no instance-specific setup)

Usage:
    # Quick test (5 instances, subset of methods)
    conda run -n blocksmc python experiments/holdout_test_molecular.py --quick

    # Full training + eval
    conda run -n blocksmc python experiments/holdout_test_molecular.py

    # Load pre-trained twist
    conda run -n blocksmc python experiments/holdout_test_molecular.py --load-twist twist_weights_mol.pt
"""

import asyncio
import argparse
import copy
import json
import time
import torch
import numpy as np
from pathlib import Path

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.constant import EndOfSequence
from genlm.control.sampler.unit import MultiTokenUnitSampler, flatten_units
from genlm.control.sampler.token import DirectTokenSampler
from genlm.eval.domains.molecular_synthesis import (
    MolecularSynthesisDataset,
    PartialSMILES,
    MolecularSynthesisEvaluator,
    default_prompt_formatter,
)

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import SMILESBoundary, FixedIntervalBoundary
from block_smc.twist import (
    TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data,
    adapt_twist_online, restore_twist_weights,
)
from block_smc.hidden_states import HiddenStateExtractor

# Paths
GRAMMAR_PATH = Path(__file__).parent / "../../control-iclr-2025/experiments/molecular_synthesis/smiles.lark"
SMILES_PATH = Path(__file__).parent / "../../control-iclr-2025/experiments/molecular_synthesis/GDB17.50000000.smi"


def decoded_posterior_nested(sequences):
    """Decode posterior from unit-level SMC where contexts are nested lists."""
    posterior = {}
    for sequence, w in zip(sequences.contexts, np.exp(sequences.log_weights)):
        flat = [t for t in flatten_units(sequence) if not isinstance(t, EndOfSequence)]
        try:
            s = b"".join(flat).decode("utf-8")
            posterior[s] = posterior.get(s, 0) + w
        except (UnicodeDecodeError, TypeError):
            pass
    total = sum(posterior.values())
    if total > 0:
        return {k: v / total for k, v in posterior.items()}
    return {}


def save_incremental(output_path, data):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def compute_method_summary(method_name, per_instance):
    scores = [r["qed"] for r in per_instance]
    valid = [r["valid"] for r in per_instance]
    logZs = [r["log_ml"] for r in per_instance]
    finite_logZs = [z for z in logZs if np.isfinite(z)]
    times = [r["wall_time"] for r in per_instance]

    n = len(scores)
    n_valid = sum(valid)
    mean_qed = float(np.mean(scores))

    return {
        "method": method_name,
        "mean_qed": mean_qed,
        "n_valid": n_valid,
        "n_total": n,
        "valid_rate": n_valid / n if n > 0 else 0,
        "mean_log_ml": float(np.mean(finite_logZs)) if finite_logZs else float("-inf"),
        "var_log_ml": float(np.var(finite_logZs)) if len(finite_logZs) > 1 else 0.0,
        "n_finite_logZ": len(finite_logZs),
        "mean_ess": float(np.mean([r["ess"] for r in per_instance])),
        "mean_wall_time": float(np.mean(times)),
        "per_instance": per_instance,
    }


async def collect_training_data(
    train_instances, llm, grammar, hse,
    n_particles, max_tokens, block_size, n_train_rounds,
    n_gramfree_rounds=0, boundary_type="smiles",
):
    """Collect twist training data from train instances."""
    buffer = TwistTrainingBuffer()
    n_guided = n_train_rounds // 2
    n_explore = n_train_rounds - n_guided
    num_blocks_est = max(max_tokens // block_size, 1)

    expensive = PartialSMILES()

    for inst_idx, instance in enumerate(train_instances):
        print(f"  Collecting from train instance {inst_idx+1}/{len(train_instances)}...")
        use_chat = "instruct" in llm.model.name.lower() if hasattr(llm.model, 'name') else False
        prompt_ids = default_prompt_formatter(llm.model.tokenizer, instance, use_chat_format=use_chat)
        llm.prompt_ids = prompt_ids

        eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"\n" in t]
        local_llm = llm.spawn_new_eos(eos_tokens)
        token_sampler = eager_token_sampler(local_llm, grammar)
        coerced = expensive.coerce(local_llm, f=b"".join)

        boundary = SMILESBoundary(min_tokens=2) if boundary_type == "smiles" else FixedIntervalBoundary(block_size)

        # Guided rounds: grammar + potential
        for r in range(n_guided):
            np.random.seed(r + inst_idx * 100)
            smc_collect = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=boundary,
                expensive_potential=expensive, llm=local_llm, oracle_twist=False,
            )
            seq = await run_block_smc(smc_collect, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
            await collect_twist_training_data(seq, hse, coerced, buffer, num_blocks_est=num_blocks_est)
            buffer.advance_age()

        # Explore rounds: grammar, no potential
        for r in range(n_explore):
            np.random.seed(1000 + r + inst_idx * 100)
            unit_sampler = MultiTokenUnitSampler(
                token_sampler, boundary, max_subunits_per_unit=100
            )
            explore_smc = SMC(unit_sampler)
            seq = await explore_smc(n_particles=n_particles, ess_threshold=0.5, max_tokens=max_tokens)
            await collect_twist_training_data(seq, hse, coerced, buffer, num_blocks_est=num_blocks_est)
            buffer.advance_age()

        # Grammar-free rounds
        for r in range(n_gramfree_rounds):
            np.random.seed(2000 + r + inst_idx * 100)
            unconstrained_sampler = DirectTokenSampler(local_llm)
            unit_sampler_gf = MultiTokenUnitSampler(
                unconstrained_sampler, FixedIntervalBoundary(block_size),
                max_subunits_per_unit=100,
            )
            gf_smc = SMC(unit_sampler_gf)
            seq = await gf_smc(n_particles=n_particles, ess_threshold=0.5, max_tokens=max_tokens)
            await collect_twist_training_data(seq, hse, coerced, buffer, num_blocks_est=num_blocks_est)
            buffer.advance_age()

    n_pos = sum(1 for l in buffer.labels if l > 0.5)
    n_neg = sum(1 for l in buffer.labels if l <= 0.5)
    print(f"  Buffer: {len(buffer)} samples ({n_pos}+ / {n_neg}-)")
    if n_gramfree_rounds > 0:
        print(f"    ({n_guided} guided + {n_explore} explore + {n_gramfree_rounds} grammar-free rounds/instance)")
    return buffer


async def eval_single_instance(
    method_name, instance, llm, grammar, hse, evaluator,
    n_particles, max_tokens, block_size, boundary_type="smiles",
    twist_head=None, twist_scale=1.0, seed=42,
    online_config=None,
):
    """Evaluate a single method on a single instance. Returns result dict."""
    num_blocks_est = max(max_tokens // block_size, 1)

    use_chat = "instruct" in llm.model.name.lower() if hasattr(llm.model, 'name') else False
    prompt_ids = default_prompt_formatter(llm.model.tokenizer, instance, use_chat_format=use_chat)
    llm.prompt_ids = prompt_ids

    eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"\n" in t]
    local_llm = llm.spawn_new_eos(eos_tokens)
    token_sampler = eager_token_sampler(local_llm, grammar)

    expensive = PartialSMILES()
    coerced = expensive.coerce(local_llm, f=b"".join)

    boundary = SMILESBoundary(min_tokens=2) if boundary_type == "smiles" else FixedIntervalBoundary(block_size)

    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()

    if method_name == "baseline":
        # Token-level SMC (paper's full-smc)
        smc = SMC(token_sampler, critic=coerced)
        seq = await smc(n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        try:
            posterior = dict(seq.decoded_posterior)
        except Exception:
            posterior = {}

    elif online_config is not None and online_config.get("from_scratch"):
        scratch_cfg = online_config

        if scratch_cfg.get("grammar_free"):
            unconstrained_sampler = DirectTokenSampler(local_llm)
            unit_sampler_p1 = MultiTokenUnitSampler(
                unconstrained_sampler, FixedIntervalBoundary(block_size),
                max_subunits_per_unit=100,
            )
            smc_explore = SMC(unit_sampler_p1)
            seq_p1 = await smc_explore(n_particles=n_particles, ess_threshold=0.5, max_tokens=max_tokens)
        else:
            smc_vanilla = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=boundary,
                expensive_potential=expensive, llm=local_llm,
            )
            seq_p1 = await run_block_smc(smc_vanilla, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)

        adapt_buffer = TwistTrainingBuffer()
        collect_stats = await collect_twist_training_data(
            seq_p1, hse, coerced, adapt_buffer, num_blocks_est=num_blocks_est,
        )
        adapt_buffer.boundary_fracs = [0.0] * len(adapt_buffer)
        n_pos = collect_stats["n_positive"]
        n_neg = collect_stats["n_examples"] - n_pos
        gf_tag = " [grammar-free]" if scratch_cfg.get("grammar_free") else ""
        print(f"      {method_name} P1 data: {collect_stats['n_examples']} samples ({n_pos}+ / {n_neg}-){gf_tag}")

        scratch_twist = TwistHead(
            d_model=hse.hidden_dim,
            hidden_dim=scratch_cfg.get("hidden_dim", 256),
            logit_clamp=scratch_cfg.get("logit_clamp", 3.0),
        ).to(hse.device)
        opt = torch.optim.Adam(scratch_twist.parameters(), lr=scratch_cfg.get("lr", 1e-3))
        train_res = train_twist_step(
            scratch_twist, adapt_buffer, opt, hse.device,
            n_epochs=scratch_cfg.get("n_epochs", 10),
            batch_size=64, class_balance=True, uniform_weights=True,
        )
        scratch_twist.eval()
        print(f"      {method_name} twist: loss={train_res['loss']:.4f} acc={train_res['accuracy']:.3f}")

        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=boundary,
            expensive_potential=expensive, llm=local_llm,
            twist_head=scratch_twist, hidden_state_extractor=hse,
            num_blocks=num_blocks_est, twist_scale=twist_scale,
        )
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        posterior = decode_block_sequences(seq)

    elif online_config is not None and twist_head is not None:
        pretrained_state = online_config["pretrained_state"]

        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=boundary,
            expensive_potential=expensive, llm=local_llm,
            twist_head=twist_head, hidden_state_extractor=hse,
            num_blocks=num_blocks_est, twist_scale=twist_scale,
        )
        seq_p1 = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)

        adapt_buffer = TwistTrainingBuffer()
        collect_stats = await collect_twist_training_data(
            seq_p1, hse, coerced, adapt_buffer, num_blocks_est=num_blocks_est,
        )
        n_pos = collect_stats["n_positive"]
        n_neg = collect_stats["n_examples"] - n_pos
        print(f"      {method_name} P1 data: {collect_stats['n_examples']} samples ({n_pos}+ / {n_neg}-)")

        adapt_twist_online(
            twist_head, adapt_buffer, device=hse.device,
            n_epochs=online_config.get("n_epochs", 3),
            lr=online_config.get("lr", 1e-4),
            l2_weight=online_config.get("l2_weight", 0.1),
        )

        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        posterior = decode_block_sequences(seq)
        restore_twist_weights(twist_head, pretrained_state)

    elif twist_head is not None:
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=boundary,
            expensive_potential=expensive, llm=local_llm,
            twist_head=twist_head, hidden_state_extractor=hse,
            num_blocks=num_blocks_est, twist_scale=twist_scale,
        )
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        posterior = decode_block_sequences(seq)

    else:
        # Block SMC vanilla (no twist)
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=boundary,
            expensive_potential=expensive, llm=local_llm,
        )
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        posterior = decode_block_sequences(seq)

    dt = time.time() - t0

    # Evaluate: QED score of best sample
    qed = 0.0
    valid = False
    best_smiles = ""
    if posterior:
        best = max(posterior, key=posterior.get)
        result = evaluator.evaluate_sample(instance, best)
        qed = result.score
        valid = result.desc == "valid"
        best_smiles = best

    return {
        "qed": qed,
        "valid": valid,
        "best_smiles": best_smiles,
        "log_ml": float(seq.log_ml),
        "ess": float(seq.ess),
        "wall_time": dt,
    }


def print_summary_table(all_results, n_test, model_name):
    print(f"\n{'='*90}")
    print(f"MOLECULAR SYNTHESIS  |  {n_test} test instances  |  {model_name}")
    print(f"{'='*90}")
    header = f"{'Method':<20} {'QED':>6} {'Valid':>8} {'E[logZ]':>8} {'Var[logZ]':>10} {'ESS':>5} {'Time':>6}"
    print(header)
    print("-" * 90)
    for r in all_results:
        print(f"{r['method']:<20} {r['mean_qed']:>6.3f} {r['n_valid']:>3}/{r['n_total']:<3}  "
              f"{r['mean_log_ml']:>8.2f} {r['var_log_ml']:>10.2f} "
              f"{r['mean_ess']:>5.1f} {r['mean_wall_time']:>6.1f}s")


async def main(args):
    output_path = Path(__file__).parent / "holdout_results_molecular.json"

    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")

    grammar_text = GRAMMAR_PATH.read_text()
    grammar = BoolCFG.from_lark(grammar_text)

    evaluator = MolecularSynthesisEvaluator()
    hse = HiddenStateExtractor(llm)

    # Load dataset
    total_needed = args.n_train_instances + args.n_test_instances
    print(f"Loading dataset: {total_needed} instances from {SMILES_PATH}")
    ds = MolecularSynthesisDataset.from_smiles(
        str(SMILES_PATH), n_molecules=20, n_instances=total_needed, seed=1234,
    )
    all_instances = list(ds)
    print(f"Got {len(all_instances)} instances total")

    train_instances = all_instances[:args.n_train_instances]
    test_instances = all_instances[args.n_train_instances:]
    n_test = min(args.n_test_instances, len(test_instances))
    test_instances = test_instances[:n_test]

    print(f"Train: {len(train_instances)} instances")
    print(f"Test:  {n_test} instances")

    # ---- Train or load twist ----
    twist_config = {}
    if args.load_twist:
        print(f"\n{'='*60}")
        print(f"Loading twist weights from {args.load_twist}")
        print(f"{'='*60}")
        checkpoint = torch.load(args.load_twist, map_location=hse.device)
        lc = args.logit_clamp
        twist = TwistHead(d_model=hse.hidden_dim, hidden_dim=checkpoint.get("hidden_dim", 256), logit_clamp=lc).to(hse.device)
        twist.load_state_dict(checkpoint["state_dict"])
        twist.eval()
        for p in twist.parameters():
            p.requires_grad_(False)
        twist_linear = None
        if "linear_state_dict" in checkpoint:
            twist_linear = TwistHead(
                d_model=hse.hidden_dim, hidden_dim=0, dropout=0.3, logit_clamp=lc,
            ).to(hse.device)
            twist_linear.load_state_dict(checkpoint["linear_state_dict"])
            twist_linear.eval()
            for p in twist_linear.parameters():
                p.requires_grad_(False)
        twist_config = checkpoint.get("train_config", {})
        print(f"  Config: {twist_config}")
    else:
        print(f"\n{'='*60}")
        print(f"Phase 1: Collecting training data from {len(train_instances)} instances")
        print(f"{'='*60}")
        t0 = time.time()
        buffer = await collect_training_data(
            train_instances, llm, grammar, hse,
            n_particles=args.n_particles, max_tokens=args.max_tokens,
            block_size=args.block_size, n_train_rounds=args.n_train_rounds,
            n_gramfree_rounds=args.n_gramfree_rounds,
            boundary_type=args.boundary_type,
        )
        data_time = time.time() - t0
        print(f"Data collection: {data_time:.0f}s")

        if len(buffer) == 0:
            print("ERROR: Empty buffer")
            await llm.cleanup()
            return

        print(f"\n{'='*60}")
        print("Phase 2: Training twist")
        print(f"{'='*60}")

        buf_train = copy.deepcopy(buffer)
        buf_train.boundary_fracs = [0.0] * len(buf_train)

        lc = args.logit_clamp
        twist = TwistHead(d_model=hse.hidden_dim, hidden_dim=256, dropout=0.0, logit_clamp=lc).to(hse.device)
        opt = torch.optim.Adam(twist.parameters(), lr=5e-3)
        res = train_twist_step(twist, buf_train, opt, hse.device,
                               n_epochs=50, batch_size=64,
                               class_balance=True, uniform_weights=True)
        twist.eval()
        for p in twist.parameters():
            p.requires_grad_(False)
        print(f"    MLP: loss={res['loss']:.4f} acc={res['accuracy']:.3f} (n_pos={res['n_pos']}, n_neg={res['n_neg']})")

        twist_linear = None  # Skip linear probe for now

        twist_config = {
            "lr": 5e-3, "ep": 50, "uniform_weights": True,
            "mlp_acc": res["accuracy"], "mlp_loss": res["loss"],
            "n_train": len(train_instances),
            "n_train_rounds": args.n_train_rounds,
            "n_gramfree_rounds": args.n_gramfree_rounds,
            "buffer_size": len(buffer), "n_pos": res["n_pos"], "n_neg": res["n_neg"],
            "data_collection_time": data_time,
        }

        save_path = args.save_twist or str(Path(__file__).parent / "twist_weights_mol.pt")
        torch.save({
            "state_dict": twist.state_dict(),
            "hidden_dim": 256,
            "d_model": hse.hidden_dim,
            "train_config": twist_config,
        }, save_path)
        print(f"  Saved twist weights to {save_path}")

    # ---- Evaluate ----
    n_seeds = args.n_seeds
    print(f"\n{'='*60}")
    print(f"Phase 3: Evaluating on {n_test} test instances ({n_seeds} seed{'s' if n_seeds > 1 else ''} each)")
    print(f"{'='*60}")

    online_cfg = {
        "n_epochs": args.online_epochs,
        "lr": args.online_lr,
        "l2_weight": args.online_l2,
    }
    pretrained_mlp_state = {k: v.clone() for k, v in twist.state_dict().items()}
    online_cfg_mlp = {**online_cfg, "pretrained_state": pretrained_mlp_state}

    ts = args.twist_scale

    scratch_cfg = {
        "from_scratch": True, "n_epochs": 10, "lr": 1e-3,
        "hidden_dim": 256, "logit_clamp": args.logit_clamp,
    }

    methods = [
        ("baseline", None, 0.0, None),
        ("vanilla", None, 0.0, None),
        (f"twist_mlp_s{ts}", twist, ts, None),
        (f"online_mlp_s{ts}", twist, ts, online_cfg_mlp),
        (f"scratch_s{ts}", None, ts, scratch_cfg),
    ]

    if args.methods:
        all_method_names = [m[0] for m in methods]
        requested = set(args.methods.split(","))
        methods = [m for m in methods if m[0] in requested]
        if not methods:
            print(f"ERROR: No matching methods. Available: {','.join(all_method_names)}")
            await llm.cleanup()
            return

    # Resume support
    completed_methods = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            prev = json.load(f)
        for m_name, m_results in prev.get("methods", {}).items():
            if len(m_results) >= n_test:
                completed_methods[m_name] = m_results
                print(f"  Resuming: skipping {m_name} ({len(m_results)} done)")

    output = {
        "model": args.model,
        "domain": "molecular_synthesis",
        "n_train": len(train_instances),
        "n_test": n_test,
        "n_seeds": n_seeds,
        "n_particles": args.n_particles,
        "max_tokens": args.max_tokens,
        "block_size": args.block_size,
        "boundary_type": args.boundary_type,
        "twist_scale": args.twist_scale,
        "logit_clamp": args.logit_clamp,
        "twist_config": twist_config,
        "method_names": [m[0] for m in methods],
        "methods": {},
        "status": "running",
    }
    save_incremental(output_path, output)

    all_results = []
    for method_name, tw, m_ts, m_online in methods:
        if method_name in completed_methods:
            per_instance = completed_methods[method_name]
            summary = compute_method_summary(method_name, per_instance)
            all_results.append(summary)
            output["methods"][method_name] = per_instance
            save_incremental(output_path, output)
            print(f"\n  --- {method_name}: SKIPPED (resumed) => QED={summary['mean_qed']:.3f} valid={summary['n_valid']}/{summary['n_total']} ---")
            continue

        print(f"\n  --- {method_name} ---")
        per_instance = []

        for inst_idx, instance in enumerate(test_instances):
            seed_results = []
            for s in range(n_seeds):
                seed = 42 + inst_idx * n_seeds + s
                result = await eval_single_instance(
                    method_name, instance, llm, grammar, hse, evaluator,
                    n_particles=args.n_particles, max_tokens=args.max_tokens,
                    block_size=args.block_size, boundary_type=args.boundary_type,
                    twist_head=tw, twist_scale=m_ts, seed=seed,
                    online_config=m_online,
                )
                seed_results.append(result)

            # Aggregate across seeds
            mean_qed = float(np.mean([r["qed"] for r in seed_results]))
            any_valid = any(r["valid"] for r in seed_results)
            finite_logZs = [r["log_ml"] for r in seed_results if np.isfinite(r["log_ml"])]
            aggregated = {
                "qed": mean_qed,
                "valid": any_valid,
                "best_smiles": max(seed_results, key=lambda r: r["qed"])["best_smiles"],
                "log_ml": float(np.mean(finite_logZs)) if finite_logZs else float("-inf"),
                "ess": float(np.mean([r["ess"] for r in seed_results])),
                "wall_time": float(np.sum([r["wall_time"] for r in seed_results])),
            }
            per_instance.append(aggregated)

            status = f"QED={aggregated['qed']:.3f}" if aggregated['valid'] else "invalid"
            logz_str = f"{aggregated['log_ml']:.2f}" if np.isfinite(aggregated['log_ml']) else "-inf"
            print(f"    [{inst_idx+1:3d}/{n_test}] {status}  log_Z={logz_str}  "
                  f"ESS={aggregated['ess']:.1f}  time={aggregated['wall_time']:.1f}s")

            output["methods"][method_name] = per_instance.copy()
            save_incremental(output_path, output)

        summary = compute_method_summary(method_name, per_instance)
        all_results.append(summary)
        print(f"  => {method_name}: QED={summary['mean_qed']:.3f}  "
              f"valid={summary['n_valid']}/{summary['n_total']}  "
              f"E[logZ]={summary['mean_log_ml']:.2f}  ESS={summary['mean_ess']:.1f}  "
              f"time={summary['mean_wall_time']:.1f}s")

    output["results"] = all_results
    output["status"] = "complete"
    save_incremental(output_path, output)

    print_summary_table(all_results, n_test, args.model)
    print(f"\nSaved to {output_path}")
    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecular Synthesis: Block SMC + twist evaluation")
    parser.add_argument("--n-train-instances", type=int, default=10)
    parser.add_argument("--n-test-instances", type=int, default=50)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--n-train-rounds", type=int, default=8)
    parser.add_argument("--n-gramfree-rounds", type=int, default=0)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--save-twist", type=str, default=None)
    parser.add_argument("--load-twist", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--boundary-type", type=str, default="smiles",
                        choices=["smiles", "fixed"], help="Block boundary type")

    # Twist params
    parser.add_argument("--twist-scale", type=float, default=0.1)
    parser.add_argument("--logit-clamp", type=float, default=3.0)

    # Online params
    parser.add_argument("--online-epochs", type=int, default=3)
    parser.add_argument("--online-lr", type=float, default=1e-4)
    parser.add_argument("--online-l2", type=float, default=0.1)

    # Method selection
    parser.add_argument("--methods", type=str, default=None)

    # Quick mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 5 test instances, subset of methods")

    args = parser.parse_args()

    if args.quick:
        args.n_test_instances = min(args.n_test_instances, 5)
        args.n_train_instances = min(args.n_train_instances, 5)
        args.n_seeds = 1
        if args.methods is None:
            args.methods = f"baseline,vanilla,twist_mlp_s{args.twist_scale},scratch_s{args.twist_scale}"

    asyncio.run(main(args))
