"""Systematic comparison: train twist once, then evaluate all methods.

Two-phase protocol:
    Phase 1 (Training): Collect data from multiple SMC sweeps, train twist head.
                         This runs once and produces a frozen twist.
    Phase 2 (Evaluation): Run each method N_seeds times with the frozen twist.
                          Report metrics averaged across seeds.

Methods evaluated:
    1. Baseline: token-level SMC (paper's full-smc)
    2. Vanilla block SMC: MultiTokenUnitSampler + coerced critic (genlm-control built-in)
    3. Block SMC, oracle twist
    4. Block SMC, frozen learned twist (trained in Phase 1)

Usage:
    conda run -n blocksmc python experiments/compare_methods_v2.py [--n-seeds 5] [--n-particles 10]
"""

import asyncio
import argparse
import json
import time
import torch
import numpy as np

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.sampler.unit import MultiTokenUnitSampler, flatten_units
from genlm.eval.domains.molecular_synthesis import PartialSMILES

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary, SMILESBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor
from block_smc.critic import _flatten_context


EXAMPLE_SMILES = ["CCO", "CC(C)C", "C1=CC=CC=C1", "CC(=O)O", "CC(C)O"]
GRAMMAR_PATH = "/home/vesteinn/Projects/BLOCK_SMC/code/control-iclr-2025/experiments/molecular_synthesis/smiles.lark"


def make_prompt(tokenizer):
    examples = "\n".join(EXAMPLE_SMILES)
    text = (
        f"Generate a valid SMILES molecular string.\n\n"
        f"Examples:\n{examples}\n\n"
        f"New molecule:"
    )
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def check_smiles_validity(smiles_str: str) -> bool:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles_str)
        return mol is not None
    except ImportError:
        return len(smiles_str) > 0 and smiles_str.isascii()


def compute_validity(posterior: dict) -> float:
    if not posterior:
        return 0.0
    total_weight = sum(posterior.values())
    valid_weight = sum(w for s, w in posterior.items() if check_smiles_validity(s))
    return valid_weight / total_weight if total_weight > 0 else 0.0


def summarize(results_by_seed: list[dict]) -> dict:
    metrics = ["log_ml", "ess", "n_unique", "validity", "wall_time"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results_by_seed
                if np.isfinite(r.get(m, float("nan")))]
        if vals:
            summary[m] = {"mean": np.mean(vals), "std": np.std(vals), "n": len(vals)}
        else:
            summary[m] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return summary


async def main(args):
    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")
    llm.prompt_ids = make_prompt(llm.model.tokenizer)

    with open(GRAMMAR_PATH) as f:
        grammar_string = f.read()
    grammar = BoolCFG.from_lark(grammar_string)
    expensive = PartialSMILES()

    eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"\n" in t]
    llm = llm.spawn_new_eos(eos_tokens)
    token_sampler = eager_token_sampler(llm, grammar)
    coerced = expensive.coerce(llm, f=b"".join)

    hse = HiddenStateExtractor(llm)
    N = args.n_particles
    T = args.max_tokens

    # =========================================================================
    # PHASE 1: TRAINING (run once)
    # =========================================================================
    n_guided = args.n_train_rounds // 2
    n_explore = args.n_train_rounds - n_guided
    print(f"\n{'='*70}")
    print(f"PHASE 1: TRAINING ({n_guided} guided + {n_explore} exploration rounds, N={N})")
    print(f"{'='*70}")

    torch.manual_seed(0)
    np.random.seed(0)

    buffer = TwistTrainingBuffer()
    train_t0 = time.time()

    # --- Guided collection: SMC with expensive potential as critic ---
    # Produces mostly positive examples (survivors of resampling)
    print(f"\n  Guided collection ({n_guided} rounds):")
    for r in range(n_guided):
        smc_collect = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(5),
            expensive_potential=expensive,
            llm=llm,
            oracle_twist=False,
        )
        seq = await run_block_smc(smc_collect, n_particles=N, ess_threshold=0.9, max_tokens=T)
        stats = await collect_twist_training_data(seq, hse, coerced, buffer)
        posterior = decode_block_sequences(seq)
        print(f"    Round {r+1}: examples={stats['n_examples']} pos={stats['n_positive']} "
              f"log_ml={seq.log_ml:.4f} unique={len(posterior)}")
        buffer.advance_age()

    # --- Exploration collection: SMC without expensive potential ---
    # Generates diverse particles (no constraint-based resampling), some will fail Φ.
    # Labels are computed by evaluating each particle's complete sequence against Φ.
    print(f"\n  Exploration collection ({n_explore} rounds, no critic):")
    for r in range(n_explore):
        unit_sampler = MultiTokenUnitSampler(
            token_sampler, FixedIntervalBoundary(5), max_subunits_per_unit=100
        )
        explore_smc = SMC(unit_sampler)  # No critic → no potential-based resampling
        seq = await explore_smc(n_particles=N, ess_threshold=0.5, max_tokens=T)
        stats = await collect_twist_training_data(seq, hse, coerced, buffer)
        posterior = decode_block_sequences(seq)
        print(f"    Round {r+1}: examples={stats['n_examples']} pos={stats['n_positive']} "
              f"neg={stats['n_examples'] - stats['n_positive']} unique={len(posterior)}")
        buffer.advance_age()

    n_pos = sum(1 for l in buffer.labels if l > 0.5)
    n_neg = sum(1 for l in buffer.labels if l <= 0.5)
    print(f"\n  Buffer: {len(buffer)} examples, {n_pos} positive, {n_neg} negative")

    if n_neg == 0:
        print("  WARNING: Still 0 negative examples. Twist cannot learn discrimination.")

    # Train twist head
    d_model = hse.hidden_dim
    device = hse.device
    twist_head = TwistHead(d_model=d_model, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=1e-3)

    train_result = train_twist_step(
        twist_head, buffer, optimizer, device,
        n_epochs=20, batch_size=64,
    )
    train_time = time.time() - train_t0

    print(f"\n  Twist training: loss={train_result['loss']:.4f} accuracy={train_result['accuracy']:.4f}")
    print(f"  Total training time: {train_time:.1f}s")

    # Freeze twist
    twist_head.eval()
    for p in twist_head.parameters():
        p.requires_grad_(False)

    # =========================================================================
    # PHASE 2: EVALUATION (multiple seeds, frozen twist)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: EVALUATION ({args.n_seeds} seeds, N={N})")
    print(f"{'='*70}")

    seeds = list(range(100, 100 + args.n_seeds))  # Different from training seeds
    all_results = {}

    # --- Method 1: Baseline token-level SMC ---
    print(f"\n--- Baseline: token-level SMC ---")
    results = []
    for seed in seeds:
        np.random.seed(seed)
        baseline_smc = SMC(token_sampler, critic=coerced)
        t0 = time.time()
        seq = await baseline_smc(n_particles=N, ess_threshold=0.9, max_tokens=T)
        dt = time.time() - t0
        try:
            posterior = dict(seq.decoded_posterior)
        except Exception:
            posterior = {}
        r = {"log_ml": float(seq.log_ml), "ess": float(seq.ess),
             "n_unique": len(posterior), "validity": compute_validity(posterior), "wall_time": dt}
        results.append(r)
        print(f"  seed={seed}: log_Z={r['log_ml']:.3f}  ESS={r['ess']:.1f}  uniq={r['n_unique']}  valid={r['validity']:.2f}  t={dt:.1f}s")
    all_results["baseline_token_smc"] = results

    # --- Method 2: Vanilla block SMC (genlm-control MultiTokenUnitSampler + coerced) ---
    print(f"\n--- Vanilla block SMC (MultiTokenUnit + coerced) ---")
    coerced_block = expensive.coerce(llm, f=lambda ctx: b"".join(flatten_units(ctx)))
    results = []
    for seed in seeds:
        np.random.seed(seed)
        unit_sampler = MultiTokenUnitSampler(token_sampler, FixedIntervalBoundary(5), max_subunits_per_unit=100)
        block_smc = SMC(unit_sampler, critic=coerced_block)
        t0 = time.time()
        seq = await block_smc(n_particles=N, ess_threshold=0.9, max_tokens=T)
        dt = time.time() - t0
        posterior = decode_block_sequences(seq)
        r = {"log_ml": float(seq.log_ml), "ess": float(seq.ess),
             "n_unique": len(posterior), "validity": compute_validity(posterior), "wall_time": dt}
        results.append(r)
        print(f"  seed={seed}: log_Z={r['log_ml']:.3f}  ESS={r['ess']:.1f}  uniq={r['n_unique']}  valid={r['validity']:.2f}  t={dt:.1f}s")
    all_results["block_vanilla"] = results

    # --- Method 3: Block SMC, oracle twist ---
    print(f"\n--- Block SMC, oracle twist ---")
    results = []
    for seed in seeds:
        np.random.seed(seed)
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(5),
            expensive_potential=expensive, llm=llm, oracle_twist=True,
        )
        t0 = time.time()
        seq = await run_block_smc(smc, n_particles=N, ess_threshold=0.9, max_tokens=T)
        dt = time.time() - t0
        posterior = decode_block_sequences(seq)
        r = {"log_ml": float(seq.log_ml), "ess": float(seq.ess),
             "n_unique": len(posterior), "validity": compute_validity(posterior), "wall_time": dt}
        results.append(r)
        print(f"  seed={seed}: log_Z={r['log_ml']:.3f}  ESS={r['ess']:.1f}  uniq={r['n_unique']}  valid={r['validity']:.2f}  t={dt:.1f}s")
    all_results["block_oracle_twist"] = results

    # --- Method 4: Block SMC, frozen learned twist ---
    print(f"\n--- Block SMC, learned twist (frozen, trained in Phase 1) ---")
    results = []
    for seed in seeds:
        np.random.seed(seed)
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(5),
            expensive_potential=expensive, llm=llm,
            twist_head=twist_head, hidden_state_extractor=hse, num_blocks=8,
        )
        t0 = time.time()
        seq = await run_block_smc(smc, n_particles=N, ess_threshold=0.9, max_tokens=T)
        dt = time.time() - t0
        posterior = decode_block_sequences(seq)
        r = {"log_ml": float(seq.log_ml), "ess": float(seq.ess),
             "n_unique": len(posterior), "validity": compute_validity(posterior), "wall_time": dt}
        results.append(r)
        print(f"  seed={seed}: log_Z={r['log_ml']:.3f}  ESS={r['ess']:.1f}  uniq={r['n_unique']}  valid={r['validity']:.2f}  t={dt:.1f}s")
    all_results["block_learned_twist"] = results

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"RESULTS (mean ± std over {args.n_seeds} eval seeds)")
    print(f"Training: {n_guided} guided + {n_explore} explore rounds, {len(buffer)} examples ({n_pos}+/{n_neg}-), {train_time:.1f}s")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'log Ẑ':>14} {'ESS':>10} {'Unique':>10} {'Valid':>10} {'Time':>10}")
    print("-" * 89)

    for name, results in all_results.items():
        s = summarize(results)
        n_fail = sum(1 for r in results if not np.isfinite(r.get("log_ml", float("nan"))))
        fail_str = f" ({n_fail} fail)" if n_fail > 0 else ""
        print(f"{name:<35} "
              f"{s['log_ml']['mean']:>6.2f}±{s['log_ml']['std']:>4.2f} "
              f"{s['ess']['mean']:>5.1f}±{s['ess']['std']:>3.1f} "
              f"{s['n_unique']['mean']:>5.1f}±{s['n_unique']['std']:>3.1f} "
              f"{s['validity']['mean']:>5.2f}±{s['validity']['std']:>3.2f} "
              f"{s['wall_time']['mean']:>5.1f}±{s['wall_time']['std']:>3.1f}"
              f"{fail_str}")

    # Save
    output = {
        "config": {"n_particles": N, "max_tokens": T, "n_seeds": args.n_seeds,
                    "n_train_rounds": args.n_train_rounds, "model": args.model},
        "training": {"buffer_size": len(buffer), "twist_loss": train_result["loss"],
                      "twist_accuracy": train_result["accuracy"], "train_time": train_time,
                      "n_positive": n_pos, "n_negative": n_neg,
                      "n_guided_rounds": n_guided, "n_explore_rounds": n_explore},
        "results": {name: [{k: v for k, v in r.items()} for r in rs]
                    for name, rs in all_results.items()},
        "summary": {name: summarize(rs) for name, rs in all_results.items()},
    }
    output_path = "experiments/comparison_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--n-train-rounds", type=int, default=10,
                        help="SMC sweeps for twist training data collection")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
