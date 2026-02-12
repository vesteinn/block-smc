"""Held-out evaluation: token-level baseline vs Block SMC vs Block SMC + twist.

Trains twist on N instances, saves weights, then evaluates all three methods
on fresh held-out instances. Results are saved incrementally after each instance
to survive crashes/segfaults.

Usage:
    # Full run: train twist on 20 instances, evaluate on 100+ test instances
    conda run -n blocksmc python experiments/holdout_test.py --max-objects 9 --n-train-instances 20

    # Load pre-trained twist weights (skip training phase)
    conda run -n blocksmc python experiments/holdout_test.py --load-twist twist_weights.pt --max-objects 9
"""

import asyncio
import argparse
import copy
import json
import time
import torch
import numpy as np
from pathlib import Path

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC, EOS
from genlm.control.constant import EndOfSequence
from genlm.control.sampler.unit import MultiTokenUnitSampler, TokenSetBoundary, flatten_units
from genlm.eval.domains.goal_inference import (
    GoalInferenceDataset,
    GoalInferenceVALPotential,
)

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor

from run_goal_inference import (
    GoalInferenceEvaluatorWithTools,
    make_prompt,
    GRAMMAR_PATH, DOMAIN_PATH, FAST_DOWNWARD_CMD, VAL_CMD, CACHE_ROOT,
)


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
    """Save results to JSON, overwriting previous save."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def compute_method_summary(method_name, per_instance):
    """Compute summary stats from per-instance results."""
    accs = [r["accuracy"] for r in per_instance]
    logZs = [r["log_ml"] for r in per_instance]
    finite_logZs = [z for z in logZs if np.isfinite(z)]
    times = [r["wall_time"] for r in per_instance]

    n = len(accs)
    k = sum(1 for a in accs if a > 0)
    acc, ci_lo, ci_hi = wilson_ci(k, n)

    return {
        "method": method_name,
        "accuracy": float(acc),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_solved": k,
        "n_total": n,
        "mean_log_ml": float(np.mean(finite_logZs)) if finite_logZs else float("-inf"),
        "var_log_ml": float(np.var(finite_logZs)) if len(finite_logZs) > 1 else 0.0,
        "std_log_ml": float(np.std(finite_logZs)) if len(finite_logZs) > 1 else 0.0,
        "n_finite_logZ": len(finite_logZs),
        "n_inf_logZ": n - len(finite_logZs),
        "mean_ess": float(np.mean([r["ess"] for r in per_instance])),
        "mean_wall_time": float(np.mean(times)),
        "total_wall_time": float(np.sum(times)),
        "per_instance": per_instance,
    }


def mcnemar_test(per_a, per_b):
    """McNemar's test on paired binary outcomes. Returns (a_wins, b_wins, p_value)."""
    a_wins = sum(1 for a, b in zip(per_a, per_b) if a["accuracy"] > 0 and b["accuracy"] == 0)
    b_wins = sum(1 for a, b in zip(per_a, per_b) if a["accuracy"] == 0 and b["accuracy"] > 0)
    n_discord = a_wins + b_wins
    if n_discord == 0:
        return a_wins, b_wins, 1.0
    try:
        from scipy.stats import binomtest
        result = binomtest(b_wins, n_discord, 0.5)
        return a_wins, b_wins, float(result.pvalue)
    except (ImportError, AttributeError):
        from math import comb
        k = min(a_wins, b_wins)
        p = sum(comb(n_discord, i) for i in range(k + 1)) / 2**n_discord * 2
        return a_wins, b_wins, min(float(p), 1.0)


async def collect_training_data(
    train_instances, llm, grammar, hse,
    n_particles, max_tokens, block_size, n_train_rounds,
):
    """Collect twist training data from train instances."""
    buffer = TwistTrainingBuffer()
    n_guided = n_train_rounds // 2
    n_explore = n_train_rounds - n_guided
    num_blocks_est = max(max_tokens // block_size, 1)
    domain_text = DOMAIN_PATH.read_text()

    for inst_idx, instance in enumerate(train_instances):
        print(f"  Collecting from train instance {inst_idx+1}/{len(train_instances)}...")
        use_chat = "instruct" in llm.model.name.lower() if hasattr(llm.model, 'name') else False
        prompt_ids = make_prompt(llm.model.tokenizer, instance, use_chat_format=use_chat)
        llm.prompt_ids = prompt_ids

        eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"))" in t]
        local_llm = llm.spawn_new_eos(eos_tokens)
        token_sampler = eager_token_sampler(local_llm, grammar)

        expensive = GoalInferenceVALPotential(
            domain_pddl_text=domain_text,
            problem_pddl_text=instance.problem_text,
            fast_downward_cmd=FAST_DOWNWARD_CMD,
            val_cmd=VAL_CMD,
            cache_root=str(CACHE_ROOT),
        )
        coerced = expensive.coerce(local_llm, f=b"".join)

        for r in range(n_guided):
            np.random.seed(r + inst_idx * 100)
            smc_collect = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=FixedIntervalBoundary(block_size),
                expensive_potential=expensive, llm=local_llm, oracle_twist=False,
            )
            seq = await run_block_smc(smc_collect, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
            await collect_twist_training_data(seq, hse, coerced, buffer, num_blocks_est=num_blocks_est)
            buffer.advance_age()

        for r in range(n_explore):
            np.random.seed(1000 + r + inst_idx * 100)
            unit_sampler = MultiTokenUnitSampler(
                token_sampler, FixedIntervalBoundary(block_size), max_subunits_per_unit=100
            )
            explore_smc = SMC(unit_sampler)
            seq = await explore_smc(n_particles=n_particles, ess_threshold=0.5, max_tokens=max_tokens)
            await collect_twist_training_data(seq, hse, coerced, buffer, num_blocks_est=num_blocks_est)
            buffer.advance_age()

    n_pos = sum(1 for l in buffer.labels if l > 0.5)
    n_neg = sum(1 for l in buffer.labels if l <= 0.5)
    print(f"  Buffer: {len(buffer)} samples ({n_pos}+ / {n_neg}-)")
    return buffer


async def eval_single_instance(
    method_name, instance, llm, grammar, hse, evaluator,
    n_particles, max_tokens, block_size, twist_head=None, twist_scale=1.0, seed=42,
):
    """Evaluate a single method on a single instance. Returns result dict."""
    domain_text = DOMAIN_PATH.read_text()
    num_blocks_est = max(max_tokens // block_size, 1)

    use_chat = "instruct" in llm.model.name.lower() if hasattr(llm.model, 'name') else False
    prompt_ids = make_prompt(llm.model.tokenizer, instance, use_chat_format=use_chat)
    llm.prompt_ids = prompt_ids

    eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"))" in t]
    local_llm = llm.spawn_new_eos(eos_tokens)
    token_sampler = eager_token_sampler(local_llm, grammar)

    expensive = GoalInferenceVALPotential(
        domain_pddl_text=domain_text,
        problem_pddl_text=instance.problem_text,
        fast_downward_cmd=FAST_DOWNWARD_CMD,
        val_cmd=VAL_CMD,
        cache_root=str(CACHE_ROOT),
    )
    coerced = expensive.coerce(local_llm, f=b"".join)

    # Coerce for unit-level methods (nested context needs flatten_units)
    coerced_nested = expensive.coerce(
        local_llm,
        f=lambda ctx: b"".join(t for t in flatten_units(ctx) if not isinstance(t, EndOfSequence)),
    )

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
    elif method_name == "bracket":
        # Unit-level SMC with bracket (goal expression) boundaries
        bracket_boundary = TokenSetBoundary(set(v for v in local_llm.vocab if isinstance(v, bytes) and v.endswith(b")")))
        unit_sampler = MultiTokenUnitSampler(
            subunit_sampler=token_sampler,
            boundary_predicate=bracket_boundary,
            max_subunits_per_unit=50,
        )
        smc = SMC(unit_sampler, critic=coerced_nested)
        seq = await smc(n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        posterior = decoded_posterior_nested(seq)
    elif twist_head is not None:
        # Block SMC with twist
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(block_size),
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
            boundary_predicate=FixedIntervalBoundary(block_size),
            expensive_potential=expensive, llm=local_llm,
        )
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        posterior = decode_block_sequences(seq)

    dt = time.time() - t0

    acc = 0.0
    if posterior:
        best = max(posterior, key=posterior.get)
        acc = evaluator.evaluate_sample(instance, best).score

    return {
        "accuracy": acc,
        "log_ml": float(seq.log_ml),
        "ess": float(seq.ess),
        "wall_time": dt,
    }


def print_summary_table(all_results, n_test, model_name, max_objects):
    """Print a formatted summary table."""
    print(f"\n{'='*90}")
    print(f"HELD-OUT RESULTS  |  {n_test} test instances  |  {model_name}  |  max_objects={max_objects}")
    print(f"{'='*90}")
    header = f"{'Method':<20} {'Acc':>6} {'95% CI':>14} {'Solved':>8} {'E[logZ]':>8} {'Var[logZ]':>10} {'ESS':>5} {'Time':>6}"
    print(header)
    print("-" * 90)
    for r in all_results:
        print(f"{r['method']:<20} {r['accuracy']:>6.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] "
              f"{r['n_solved']:>3}/{r['n_total']:<3}  {r['mean_log_ml']:>8.2f} {r['var_log_ml']:>10.2f} "
              f"{r['mean_ess']:>5.1f} {r['mean_wall_time']:>6.1f}s")

    # Paired comparisons
    print(f"\n{'='*90}")
    print("PAIRED COMPARISONS (McNemar's test)")
    print(f"{'='*90}")
    for i, ra in enumerate(all_results):
        for j, rb in enumerate(all_results):
            if j <= i:
                continue
            a_wins, b_wins, p = mcnemar_test(ra["per_instance"], rb["per_instance"])
            sig = " *" if p < 0.05 else " **" if p < 0.01 else ""
            print(f"  {ra['method']:<20} vs {rb['method']:<20}: "
                  f"{ra['method'].split('_')[0]}_wins={a_wins}, {rb['method'].split('_')[0]}_wins={b_wins}  "
                  f"p={p:.4f}{sig}")


async def main(args):
    output_path = Path(__file__).parent / f"holdout_results_obj{args.max_objects}.json"

    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")

    grammar_text = GRAMMAR_PATH.read_text()
    grammar = BoolCFG.from_lark(grammar_text)

    evaluator = GoalInferenceEvaluatorWithTools()
    hse = HiddenStateExtractor(llm)

    # Load dataset
    total_needed = args.n_train_instances + args.n_test_instances
    print(f"Loading dataset: requesting {total_needed} instances (max_objects={args.max_objects})")
    ds = GoalInferenceDataset.from_hf_planetarium(
        n_examples=total_needed, max_objects=args.max_objects, domains=["blocksworld"]
    )
    all_instances = list(ds)
    print(f"Got {len(all_instances)} instances total")

    if len(all_instances) <= args.n_train_instances:
        print(f"ERROR: Need more than {args.n_train_instances} instances but only got {len(all_instances)}.")
        await llm.cleanup()
        return

    # Split: first N for training, rest for testing
    train_instances = all_instances[:args.n_train_instances]
    test_instances = all_instances[args.n_train_instances:]
    n_test = min(args.n_test_instances, len(test_instances))
    test_instances = test_instances[:n_test]

    print(f"Train: {len(train_instances)} instances (indices 0-{args.n_train_instances-1})")
    print(f"Test:  {n_test} instances (indices {args.n_train_instances}-{args.n_train_instances + n_test - 1})")

    # ---- Phase 1+2: Train or load twist ----
    twist_config = {}
    if args.load_twist:
        print(f"\n{'='*60}")
        print(f"Loading twist weights from {args.load_twist}")
        print(f"{'='*60}")
        checkpoint = torch.load(args.load_twist, map_location=hse.device)
        twist = TwistHead(d_model=hse.hidden_dim, hidden_dim=checkpoint.get("hidden_dim", 256)).to(hse.device)
        twist.load_state_dict(checkpoint["state_dict"])
        twist.eval()
        for p in twist.parameters():
            p.requires_grad_(False)
        twist_config = checkpoint.get("train_config", {})
        print(f"  Loaded. Config: {twist_config}")
    else:
        print(f"\n{'='*60}")
        print(f"Phase 1: Collecting training data from {len(train_instances)} instances")
        print(f"{'='*60}")
        t0 = time.time()
        buffer = await collect_training_data(
            train_instances, llm, grammar, hse,
            n_particles=args.n_particles, max_tokens=args.max_tokens,
            block_size=args.block_size, n_train_rounds=args.n_train_rounds,
        )
        data_time = time.time() - t0
        print(f"Data collection: {data_time:.0f}s")

        if len(buffer) == 0:
            print("ERROR: Empty buffer")
            await llm.cleanup()
            return

        print(f"\n{'='*60}")
        print("Phase 2: Training twist (uniform weights + class balance + stable BCE)")
        print(f"{'='*60}")

        buf_train = copy.deepcopy(buffer)
        buf_train.boundary_fracs = [0.0] * len(buf_train)  # bf=False
        twist = TwistHead(d_model=hse.hidden_dim, hidden_dim=256).to(hse.device)
        opt = torch.optim.Adam(twist.parameters(), lr=5e-3)
        res = train_twist_step(twist, buf_train, opt, hse.device,
                                n_epochs=50, batch_size=64,
                                class_balance=True, uniform_weights=True)
        twist.eval()
        for p in twist.parameters():
            p.requires_grad_(False)
        print(f"  Twist: loss={res['loss']:.4f} acc={res['accuracy']:.3f} (n_pos={res['n_pos']}, n_neg={res['n_neg']})")

        twist_config = {
            "bf": False, "cb": True, "lr": 5e-3, "ep": 50,
            "uniform_weights": True, "twist_acc": res["accuracy"],
            "twist_loss": res["loss"], "n_train": len(train_instances),
            "buffer_size": len(buffer), "n_pos": res["n_pos"], "n_neg": res["n_neg"],
            "data_collection_time": data_time,
        }

        # Save twist weights
        save_path = args.save_twist or str(Path(__file__).parent / f"twist_weights_obj{args.max_objects}.pt")
        torch.save({
            "state_dict": twist.state_dict(),
            "hidden_dim": 256,
            "d_model": hse.hidden_dim,
            "train_config": twist_config,
        }, save_path)
        print(f"  Saved twist weights to {save_path}")

    # ---- Phase 3: Evaluate on fresh test set ----
    print(f"\n{'='*60}")
    print(f"Phase 3: Evaluating on {n_test} FRESH test instances")
    print(f"{'='*60}")

    # Methods to evaluate
    methods = [
        ("baseline", None, 0.0),        # token-level SMC
        ("bracket", None, 0.0),          # unit-level SMC, bracket boundaries
        ("vanilla", None, 0.0),          # block SMC, fixed-interval boundaries
        ("twist_s0.1", twist, 0.1),      # block SMC + learned twist
    ]

    # Initialize incremental output
    output = {
        "model": args.model,
        "n_train": len(train_instances),
        "n_test": n_test,
        "max_objects": args.max_objects,
        "n_particles": args.n_particles,
        "max_tokens": args.max_tokens,
        "block_size": args.block_size,
        "twist_config": twist_config,
        "methods": {},
        "status": "running",
    }
    save_incremental(output_path, output)

    all_results = []
    for method_name, tw, ts in methods:
        print(f"\n  --- {method_name} ---")
        per_instance = []

        for inst_idx, instance in enumerate(test_instances):
            seed = 42 + inst_idx

            result = await eval_single_instance(
                method_name, instance, llm, grammar, hse, evaluator,
                n_particles=args.n_particles, max_tokens=args.max_tokens,
                block_size=args.block_size, twist_head=tw, twist_scale=ts, seed=seed,
            )
            per_instance.append(result)

            status = "OK" if result["accuracy"] > 0 else "FAIL"
            logz_str = f"{result['log_ml']:.2f}" if np.isfinite(result['log_ml']) else "-inf"
            print(f"    [{inst_idx+1:3d}/{n_test}] {status}  log_Z={logz_str}  "
                  f"ESS={result['ess']:.1f}  time={result['wall_time']:.1f}s")

            # Incremental save after every instance
            output["methods"][method_name] = per_instance.copy()
            save_incremental(output_path, output)

        summary = compute_method_summary(method_name, per_instance)
        all_results.append(summary)
        print(f"  => {method_name}: {summary['accuracy']:.3f} [{summary['ci_lo']:.3f}, {summary['ci_hi']:.3f}]  "
              f"({summary['n_solved']}/{summary['n_total']})  "
              f"E[logZ]={summary['mean_log_ml']:.2f}  Var[logZ]={summary['var_log_ml']:.2f}  "
              f"ESS={summary['mean_ess']:.1f}  time={summary['mean_wall_time']:.1f}s")

    # Final save with summaries
    output["results"] = all_results
    output["status"] = "complete"
    save_incremental(output_path, output)

    # Print summary table
    print_summary_table(all_results, n_test, args.model, args.max_objects)

    print(f"\nSaved to {output_path}")
    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-out evaluation of Block SMC + twist")
    parser.add_argument("--n-train-instances", type=int, default=20)
    parser.add_argument("--n-test-instances", type=int, default=150)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-train-rounds", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=9)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--save-twist", type=str, default=None, help="Path to save twist weights (default: auto)")
    parser.add_argument("--load-twist", type=str, default=None, help="Path to load pre-trained twist weights")
    args = parser.parse_args()

    asyncio.run(main(args))
