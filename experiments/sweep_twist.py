"""Hyperparameter sweep for twist function training.

Collects training data ONCE, then trains and evaluates across all configs.
This is much faster than re-collecting data for each config.

Usage:
    conda run -n blocksmc python experiments/sweep_twist.py
"""

import asyncio
import argparse
import copy
import json
import time
import torch
import numpy as np
from pathlib import Path
from itertools import product

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.sampler.unit import MultiTokenUnitSampler
from genlm.eval.domains.goal_inference import (
    GoalInferenceDataset,
    GoalInferenceVALPotential,
    goal_default_prompt_formatter,
)

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor

# Import evaluator from the main experiment
from run_goal_inference import (
    GoalInferenceEvaluatorWithTools,
    make_prompt,
    GRAMMAR_PATH, DOMAIN_PATH, FAST_DOWNWARD_CMD, VAL_CMD, CACHE_ROOT,
)


async def collect_training_data(
    train_instances, llm, grammar, domain_text, hse,
    n_particles, max_tokens, block_size, n_train_rounds,
):
    """Collect twist training data once, reused across all sweep configs."""
    buffer = TwistTrainingBuffer()
    n_guided = n_train_rounds // 2
    n_explore = n_train_rounds - n_guided
    num_blocks_est = max(max_tokens // block_size, 1)

    for inst_idx, instance in enumerate(train_instances):
        print(f"  Collecting data from instance {inst_idx+1}/{len(train_instances)}...")
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
    return buffer, n_pos, n_neg


def train_twist_from_buffer(
    buffer, hse, twist_lr, twist_epochs, twist_hidden_dim, class_balance, use_boundary_frac,
):
    """Train a twist head from an existing buffer."""
    # Clone buffer so we don't mutate the original
    buf = copy.deepcopy(buffer)

    if not use_boundary_frac:
        buf.boundary_fracs = [0.0] * len(buf)

    d_model = hse.hidden_dim
    device = hse.device

    twist_head = TwistHead(d_model=d_model, hidden_dim=twist_hidden_dim).to(device)
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=twist_lr)
    train_result = train_twist_step(
        twist_head, buf, optimizer, device,
        n_epochs=twist_epochs, batch_size=64,
        class_balance=class_balance,
    )

    twist_head.eval()
    for p in twist_head.parameters():
        p.requires_grad_(False)

    return twist_head, train_result


async def evaluate_twist(
    eval_instances, llm, grammar, domain_text, hse, evaluator,
    twist_head, n_particles, max_tokens, block_size, twist_scale, seed=42,
):
    """Evaluate a trained twist head on held-out instances."""
    num_blocks_est = max(max_tokens // block_size, 1)
    results = []

    for inst_idx, instance in enumerate(eval_instances):
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

        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(block_size),
            expensive_potential=expensive, llm=local_llm,
            twist_head=twist_head, hidden_state_extractor=hse,
            num_blocks=num_blocks_est,
            twist_scale=twist_scale,
        )

        t0 = time.time()
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        dt = time.time() - t0

        posterior = decode_block_sequences(seq)
        accuracy = 0.0
        if posterior:
            best_response = max(posterior, key=posterior.get)
            eval_result = evaluator.evaluate_sample(instance, best_response)
            accuracy = eval_result.score

        results.append({
            "accuracy": accuracy,
            "log_ml": float(seq.log_ml),
            "ess": float(seq.ess),
            "wall_time": dt,
        })

    accuracies = [r["accuracy"] for r in results]
    log_mls = [r["log_ml"] for r in results if np.isfinite(r["log_ml"])]
    return {
        "accuracy": float(np.mean(accuracies)),
        "e_log_z": float(np.mean(log_mls)) if log_mls else float("-inf"),
        "runtime": float(np.mean([r["wall_time"] for r in results])),
        "per_instance": results,
    }


async def main(args):
    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")

    grammar_text = GRAMMAR_PATH.read_text()
    grammar = BoolCFG.from_lark(grammar_text)
    domain_text = DOMAIN_PATH.read_text()

    total_needed = args.n_train_instances + args.n_eval_instances
    ds = GoalInferenceDataset.from_hf_planetarium(
        n_examples=total_needed, max_objects=args.max_objects, domains=["blocksworld"]
    )
    all_instances = list(ds)
    train_instances = all_instances[:args.n_train_instances]
    eval_instances = all_instances[args.n_train_instances:]
    print(f"Split: {len(train_instances)} train, {len(eval_instances)} eval")

    evaluator = GoalInferenceEvaluatorWithTools()
    hse = HiddenStateExtractor(llm)

    # ---- Phase 1: Collect training data ONCE ----
    print(f"\n{'='*60}")
    print("Phase 1: Collecting training data")
    print(f"{'='*60}")
    t0 = time.time()
    buffer, n_pos, n_neg = await collect_training_data(
        train_instances, llm, grammar, domain_text, hse,
        n_particles=args.n_particles, max_tokens=args.max_tokens,
        block_size=args.block_size, n_train_rounds=args.n_train_rounds,
    )
    data_time = time.time() - t0
    print(f"Data collection took {data_time:.0f}s")

    if len(buffer) == 0:
        print("ERROR: Empty buffer, aborting sweep")
        await llm.cleanup()
        return

    # ---- Phase 2: Evaluate vanilla baseline once ----
    print(f"\n{'='*60}")
    print("Phase 2: Vanilla baseline (no twist)")
    print(f"{'='*60}")
    vanilla_results = []
    num_blocks_est = max(args.max_tokens // args.block_size, 1)
    for inst_idx, instance in enumerate(eval_instances):
        np.random.seed(42 + inst_idx)
        torch.manual_seed(42 + inst_idx)
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
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(args.block_size),
            expensive_potential=expensive, llm=local_llm,
        )
        t0 = time.time()
        seq = await run_block_smc(smc, n_particles=args.n_particles, ess_threshold=0.9, max_tokens=args.max_tokens)
        dt = time.time() - t0
        posterior = decode_block_sequences(seq)
        acc = 0.0
        if posterior:
            best = max(posterior, key=posterior.get)
            acc = evaluator.evaluate_sample(instance, best).score
        vanilla_results.append({"accuracy": acc, "log_ml": float(seq.log_ml), "wall_time": dt})

    vanilla_acc = float(np.mean([r["accuracy"] for r in vanilla_results]))
    vanilla_logz = [r["log_ml"] for r in vanilla_results if np.isfinite(r["log_ml"])]
    vanilla_mean_logz = float(np.mean(vanilla_logz)) if vanilla_logz else float("-inf")
    print(f"Vanilla: acc={vanilla_acc:.3f}  E[logZ]={vanilla_mean_logz:.2f}")

    # ---- Phase 3: Train + Eval sweep ----
    print(f"\n{'='*60}")
    print("Phase 3: Training + Evaluation sweep")
    print(f"{'='*60}")

    # Build config grid
    train_configs = []
    for bf, cb, lr, ep in product(
        [True, False],               # use_boundary_frac
        [True, False],               # class_balance
        [1e-3, 5e-3],               # twist_lr
        [20, 50],                    # twist_epochs
    ):
        train_configs.append({
            "use_boundary_frac": bf,
            "class_balance": cb,
            "twist_lr": lr,
            "twist_epochs": ep,
            "twist_hidden_dim": 256,
        })

    twist_scales = [0.05, 0.1, 0.3, 1.0]

    print(f"Training configs: {len(train_configs)}")
    print(f"Twist scales: {twist_scales}")
    print(f"Total evaluations: {len(train_configs) * len(twist_scales)}")
    print(f"Eval instances: {len(eval_instances)}")

    all_results = []
    best_acc = -1
    best_config = None

    for i, tc in enumerate(train_configs):
        print(f"\n--- Train config {i+1}/{len(train_configs)}: "
              f"bf={tc['use_boundary_frac']}, cb={tc['class_balance']}, "
              f"lr={tc['twist_lr']}, ep={tc['twist_epochs']} ---")

        # Train twist (fast — just MLP training on existing buffer)
        t0 = time.time()
        twist_head, train_result = train_twist_from_buffer(
            buffer, hse,
            twist_lr=tc["twist_lr"],
            twist_epochs=tc["twist_epochs"],
            twist_hidden_dim=tc["twist_hidden_dim"],
            class_balance=tc["class_balance"],
            use_boundary_frac=tc["use_boundary_frac"],
        )
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.1f}s  twist_acc={train_result['accuracy']:.3f}  loss={train_result['loss']:.4f}")

        # Evaluate at each twist scale
        for ts in twist_scales:
            t0 = time.time()
            eval_result = await evaluate_twist(
                eval_instances, llm, grammar, domain_text, hse, evaluator,
                twist_head, n_particles=args.n_particles, max_tokens=args.max_tokens,
                block_size=args.block_size, twist_scale=ts,
            )
            eval_time = time.time() - t0

            full_config = {**tc, "twist_scale": ts}
            result = {
                **eval_result,
                "config": full_config,
                "twist_acc": train_result["accuracy"],
                "twist_loss": train_result["loss"],
                "n_pos": n_pos,
                "n_neg": n_neg,
                "buffer_size": len(buffer),
                "train_time": train_time,
                "eval_time": eval_time,
            }
            all_results.append(result)

            marker = ""
            if result["accuracy"] > best_acc:
                best_acc = result["accuracy"]
                best_config = full_config
                marker = " *** NEW BEST ***"

            print(f"    ts={ts:.2f}: acc={result['accuracy']:.3f}  E[logZ]={result['e_log_z']:.2f}  "
                  f"time={eval_time:.0f}s{marker}")

        # Save incrementally
        output_path = Path(__file__).parent / "sweep_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "vanilla": {"accuracy": vanilla_acc, "e_log_z": vanilla_mean_logz},
                "buffer": {"size": len(buffer), "n_pos": n_pos, "n_neg": n_neg},
                "data_collection_time": data_time,
                "results": all_results,
            }, f, indent=2, default=str)

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY ({len(all_results)} configs, {len(eval_instances)} eval instances)")
    print(f"Vanilla baseline: acc={vanilla_acc:.3f}  E[logZ]={vanilla_mean_logz:.2f}")
    print(f"{'='*70}")

    ranked = sorted(all_results, key=lambda r: (r["accuracy"], r["e_log_z"]), reverse=True)

    print(f"\nTop 15 configs:")
    print(f"{'Rank':>4}  {'Acc':>6}  {'E[logZ]':>8}  {'T.Acc':>6}  {'BF':>5}  {'CB':>5}  {'LR':>8}  {'Ep':>4}  {'TS':>5}")
    print("-" * 75)
    for rank, r in enumerate(ranked[:15], 1):
        c = r["config"]
        delta = r["accuracy"] - vanilla_acc
        sign = "+" if delta >= 0 else ""
        print(f"{rank:>4}  {r['accuracy']:>6.3f}  {r['e_log_z']:>8.2f}  "
              f"{r['twist_acc']:>6.3f}  {str(c['use_boundary_frac']):>5}  "
              f"{str(c['class_balance']):>5}  {c['twist_lr']:>8.4f}  "
              f"{c['twist_epochs']:>4}  {c['twist_scale']:>5.2f}  ({sign}{delta:.3f})")

    # Analyze which factors matter most
    print(f"\n--- Factor Analysis ---")
    for factor in ["use_boundary_frac", "class_balance", "twist_lr", "twist_epochs", "twist_scale"]:
        values = sorted(set(r["config"][factor] for r in all_results))
        print(f"\n{factor}:")
        for val in values:
            subset = [r for r in all_results if r["config"][factor] == val]
            mean_acc = np.mean([r["accuracy"] for r in subset])
            mean_logz = np.mean([r["e_log_z"] for r in subset if np.isfinite(r["e_log_z"])])
            print(f"  {val}: acc={mean_acc:.3f}  E[logZ]={mean_logz:.2f}  (n={len(subset)})")

    print(f"\nSaved to {output_path}")
    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train-instances", type=int, default=5)
    parser.add_argument("--n-eval-instances", type=int, default=15)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-train-rounds", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
