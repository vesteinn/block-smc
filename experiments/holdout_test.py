"""Clean held-out test of pre-selected twist configs.

Trains twist on 5 instances (same as sweep, max_objects=4, indices 0-4),
then tests on FRESH instances never seen during hyperparameter selection.

Only tests 2 pre-selected configs from the 1B sweep + vanilla baseline.
No multiple comparisons issue since configs were chosen a priori.

Usage:
    conda run -n blocksmc python experiments/holdout_test.py
    conda run -n blocksmc python experiments/holdout_test.py --n-test-instances 50 --max-objects 8
"""

import asyncio
import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.sampler.unit import MultiTokenUnitSampler
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

# The sweep used max_objects=4 which gives exactly 20 instances (indices 0-19).
# Indices 0-4 were train, 5-19 were val. Any index >= 20 is fresh.
SWEEP_USED_COUNT = 20


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


async def eval_method(
    method_name, test_instances, llm, grammar, hse, evaluator,
    n_particles, max_tokens, block_size, twist_head=None, twist_scale=1.0, seed=42,
):
    """Evaluate a single method on all test instances."""
    domain_text = DOMAIN_PATH.read_text()
    num_blocks_est = max(max_tokens // block_size, 1)
    results = []

    for inst_idx, instance in enumerate(test_instances):
        np.random.seed(seed + inst_idx)
        torch.manual_seed(seed + inst_idx)

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

        if twist_head is not None:
            smc = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=FixedIntervalBoundary(block_size),
                expensive_potential=expensive, llm=local_llm,
                twist_head=twist_head, hidden_state_extractor=hse,
                num_blocks=num_blocks_est, twist_scale=twist_scale,
            )
        else:
            smc = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=FixedIntervalBoundary(block_size),
                expensive_potential=expensive, llm=local_llm,
            )

        t0 = time.time()
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        dt = time.time() - t0

        posterior = decode_block_sequences(seq)
        acc = 0.0
        if posterior:
            best = max(posterior, key=posterior.get)
            acc = evaluator.evaluate_sample(instance, best).score

        results.append({
            "accuracy": acc,
            "log_ml": float(seq.log_ml),
            "ess": float(seq.ess),
            "wall_time": dt,
        })

        status = "OK" if acc > 0 else "FAIL"
        print(f"    [{inst_idx+1:2d}/{len(test_instances)}] {status}  log_Z={seq.log_ml:.2f}  time={dt:.1f}s")

    accs = [r["accuracy"] for r in results]
    mean_acc = np.mean(accs)
    # Wilson score 95% CI for binomial proportion
    n = len(accs)
    k = sum(accs)
    z = 1.96
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    ci_lo, ci_hi = center - margin, center + margin

    return {
        "method": method_name,
        "accuracy": float(mean_acc),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_solved": int(k),
        "n_total": n,
        "mean_log_ml": float(np.mean([r["log_ml"] for r in results if np.isfinite(r["log_ml"])])) if any(np.isfinite(r["log_ml"]) for r in results) else float("-inf"),
        "mean_wall_time": float(np.mean([r["wall_time"] for r in results])),
        "per_instance": results,
    }


async def main(args):
    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")

    grammar_text = GRAMMAR_PATH.read_text()
    grammar = BoolCFG.from_lark(grammar_text)

    evaluator = GoalInferenceEvaluatorWithTools()
    hse = HiddenStateExtractor(llm)

    # Load dataset — enough to cover both train (reuse) and fresh test
    total_needed = SWEEP_USED_COUNT + args.n_test_instances
    print(f"Loading dataset: requesting {total_needed} instances (max_objects={args.max_objects})")
    ds = GoalInferenceDataset.from_hf_planetarium(
        n_examples=total_needed, max_objects=args.max_objects, domains=["blocksworld"]
    )
    all_instances = list(ds)
    print(f"Got {len(all_instances)} instances total")

    if len(all_instances) <= SWEEP_USED_COUNT:
        print(f"ERROR: Need more than {SWEEP_USED_COUNT} instances but only got {len(all_instances)}.")
        print(f"Try increasing --max-objects (currently {args.max_objects}).")
        await llm.cleanup()
        return

    # Split: train from sweep's train set, test from FRESH instances only
    train_instances = all_instances[:args.n_train_instances]  # same 5 as sweep
    test_instances = all_instances[SWEEP_USED_COUNT:]  # skip ALL sweep instances
    n_test = min(args.n_test_instances, len(test_instances))
    test_instances = test_instances[:n_test]

    print(f"Train: {len(train_instances)} instances (indices 0-{args.n_train_instances-1}, same as sweep)")
    print(f"Test:  {n_test} FRESH instances (indices {SWEEP_USED_COUNT}-{SWEEP_USED_COUNT + n_test - 1})")
    print(f"       (sweep used indices 0-{SWEEP_USED_COUNT-1}, all skipped)")

    # ---- Phase 1: Collect training data ----
    print(f"\n{'='*60}")
    print("Phase 1: Collecting training data (same 5 train instances as sweep)")
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

    # ---- Phase 2: Train twist with fixed settings ----
    print(f"\n{'='*60}")
    print("Phase 2: Training twist (uniform weights + class balance + stable BCE)")
    print(f"{'='*60}")

    import copy
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

    # ---- Phase 3: Evaluate on fresh test set ----
    print(f"\n{'='*60}")
    print(f"Phase 3: Evaluating on {n_test} FRESH test instances")
    print(f"{'='*60}")

    methods = [
        ("vanilla", None, 0.0),
        ("twist_s0.1", twist, 0.1),
        ("twist_s0.3", twist, 0.3),
        ("twist_s0.5", twist, 0.5),
        ("twist_s1.0", twist, 1.0),
    ]

    all_results = []
    for method_name, tw, ts in methods:
        print(f"\n  --- {method_name} ---")
        result = await eval_method(
            method_name, test_instances, llm, grammar, hse, evaluator,
            n_particles=args.n_particles, max_tokens=args.max_tokens,
            block_size=args.block_size, twist_head=tw, twist_scale=ts,
        )
        all_results.append(result)
        print(f"  => {method_name}: {result['accuracy']:.3f} [{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]  "
              f"({result['n_solved']}/{result['n_total']})  "
              f"E[logZ]={result['mean_log_ml']:.2f}  time={result['mean_wall_time']:.1f}s")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"HELD-OUT TEST RESULTS ({n_test} fresh instances, 1B model)")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Acc':>6} {'95% CI':>14} {'Solved':>8} {'E[logZ]':>8} {'Time':>6}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['method']:<20} {r['accuracy']:>6.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] "
              f"{r['n_solved']:>3}/{r['n_total']:<3}  {r['mean_log_ml']:>8.2f} {r['mean_wall_time']:>6.1f}s")

    # Paired comparisons vs vanilla
    vanilla_per = all_results[0]["per_instance"]
    for twist_result in all_results[1:]:
        twist_per = twist_result["per_instance"]
        twist_wins = sum(1 for v, t in zip(vanilla_per, twist_per) if v["accuracy"] == 0 and t["accuracy"] == 1)
        vanilla_wins = sum(1 for v, t in zip(vanilla_per, twist_per) if v["accuracy"] == 1 and t["accuracy"] == 0)
        n_discord = twist_wins + vanilla_wins
        print(f"\n  {twist_result['method']} vs vanilla: twist_wins={twist_wins}, vanilla_wins={vanilla_wins}", end="")
        if n_discord > 0:
            try:
                from scipy.stats import binomtest
                result = binomtest(twist_wins, n_discord, 0.5)
                print(f"  p={result.pvalue:.4f}")
            except (ImportError, AttributeError):
                # Fallback: manual two-sided binomial test
                from math import comb
                p = sum(comb(n_discord, k) for k in range(min(twist_wins, vanilla_wins) + 1)) / 2**n_discord * 2
                print(f"  p={min(p, 1.0):.4f}")
        else:
            print("  (no discordant pairs)")

    # Save results
    output = {
        "model": args.model,
        "n_train": len(train_instances),
        "n_test": n_test,
        "test_indices": f"{SWEEP_USED_COUNT}-{SWEEP_USED_COUNT + n_test - 1}",
        "max_objects": args.max_objects,
        "n_particles": args.n_particles,
        "max_tokens": args.max_tokens,
        "block_size": args.block_size,
        "data_collection_time": data_time,
        "twist_config": {"bf": False, "cb": True, "lr": 5e-3, "ep": 50,
                         "uniform_weights": True, "twist_acc": res["accuracy"],
                         "twist_loss": res["loss"]},
        "results": all_results,
    }
    output_path = Path(__file__).parent / "holdout_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train-instances", type=int, default=5)
    parser.add_argument("--n-test-instances", type=int, default=50)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-train-rounds", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
