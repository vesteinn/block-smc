"""Goal Inference / Planetarium experiment.

Compares baseline token-level SMC vs Block SMC (vanilla and with learned twist)
on PDDL goal inference from the BatsResearch/planetarium dataset.

The expensive potential (GoalInferenceVALPotential) uses Fast-Downward for plan
generation and VAL for plan validation, making it genuinely expensive (~seconds
per evaluation). This is exactly the setting where Block SMC should shine:
fewer expensive evaluations with twist compensation.

Methods:
    1. Baseline: token-level SMC (paper's full-smc)
    2. Block SMC, no twist (vanilla)
    3. Block SMC, learned twist

Metrics (with bootstrap CIs):
    Accuracy, Runtime, E[log Z], log E[Z], Var(log Z), CV(w), ESS

Usage:
    conda run -n blocksmc python experiments/run_goal_inference.py [--n-instances 10]
"""

import asyncio
import argparse
import json
import time
import torch
import numpy as np
import planetarium
from pathlib import Path
from functools import partial

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.sampler.unit import MultiTokenUnitSampler
from genlm.eval.domains.goal_inference import (
    GoalInferenceDataset,
    GoalInferenceEvaluator,
    GoalInferenceVALPotential,
    GoalInferenceInstance,
    goal_default_prompt_formatter,
)
from genlm.eval.core import EvaluationResult

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor


# Paths
GRAMMAR_PATH = Path(__file__).parent / "../../control-iclr-2025/experiments/goal_inference/grammars/goal_inference.lark"
DOMAIN_PATH = Path(__file__).parent / "../../control-iclr-2025/experiments/goal_inference/pddl_domains/blocksworld.pddl"
FAST_DOWNWARD_CMD = str(Path(__file__).parent / "fast-downward.sh")
VAL_CMD = str(Path(__file__).parent / "validate.sh")
CACHE_ROOT = Path(__file__).parent / "../cache/goal_inference"


class GoalInferenceEvaluatorWithTools(GoalInferenceEvaluator):
    """Evaluator that passes correct tool paths to planetarium."""

    def __init__(self, val_cmd=VAL_CMD, fast_downward_cmd=FAST_DOWNWARD_CMD):
        super().__init__()
        self.val_cmd = val_cmd
        self.fast_downward_cmd = fast_downward_cmd

    def evaluate_sample(
        self, instance: GoalInferenceInstance, response: str
    ) -> EvaluationResult:
        masked = instance.masked_pddl
        full_pddl = instance.problem_text
        if not masked or not full_pddl:
            return EvaluationResult(score=0.0, desc="missing_problem_or_masked")
        if "[BLANK]" not in masked:
            return EvaluationResult(score=0.0, desc="no_blank_marker")

        pred = response.strip() if response is not None else ""
        generated_pddl = masked.replace("[BLANK]", pred + ")")

        try:
            ok = planetarium.evaluate(
                full_pddl, generated_pddl,
                val=self.val_cmd,
                fast_downward=self.fast_downward_cmd,
            )[2]
        except (ValueError, AttributeError):
            return EvaluationResult(score=0.0, desc="planetarium_error")

        return EvaluationResult(
            score=1.0 if ok else 0.0,
            desc="equiv" if ok else "not_equiv",
            metadata={"candidate": generated_pddl},
        )


def make_prompt(tokenizer, instance, use_chat_format=False):
    """Create prompt for goal inference instance."""
    return goal_default_prompt_formatter(
        tokenizer=tokenizer,
        instance=instance,
        use_chat_format=use_chat_format,
    )


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    values = np.array([v for v in values if np.isfinite(v)])
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[int((1 - alpha) * n_bootstrap)]
    return float(np.mean(values)), float(lo), float(hi)


def compute_metrics(all_instance_results):
    """Compute aggregate metrics across instances with bootstrap CIs.

    all_instance_results: list of dicts, one per instance, each with:
        log_ml, ess, wall_time, accuracy (0 or 1), log_weights (list)
    """
    n = len(all_instance_results)

    # Accuracy: fraction of instances with correct output
    accuracies = [r["accuracy"] for r in all_instance_results]
    acc_mean, acc_lo, acc_hi = bootstrap_ci(accuracies)

    # Runtime
    runtimes = [r["wall_time"] for r in all_instance_results]
    rt_mean, rt_lo, rt_hi = bootstrap_ci(runtimes)

    # E[log Z] - average of log Z across instances
    log_zs = [r["log_ml"] for r in all_instance_results]
    elz_mean, elz_lo, elz_hi = bootstrap_ci(log_zs)

    # log E[Z] - log of average of exp(log Z)
    finite_log_zs = [z for z in log_zs if np.isfinite(z)]
    if finite_log_zs:
        log_ez = float(np.log(np.mean(np.exp(finite_log_zs))))
        # Bootstrap log E[Z]
        rng = np.random.RandomState(42)
        boot_log_ez = []
        arr = np.array(finite_log_zs)
        for _ in range(1000):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_log_ez.append(np.log(np.mean(np.exp(sample))))
        boot_log_ez = np.sort(boot_log_ez)
        lez_lo = float(boot_log_ez[25])
        lez_hi = float(boot_log_ez[975])
    else:
        log_ez, lez_lo, lez_hi = float("nan"), float("nan"), float("nan")

    # Var(log Z)
    if len(finite_log_zs) > 1:
        var_lz = float(np.var(finite_log_zs))
        rng = np.random.RandomState(42)
        boot_var = []
        arr = np.array(finite_log_zs)
        for _ in range(1000):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_var.append(np.var(sample))
        boot_var = np.sort(boot_var)
        vlz_lo = float(boot_var[25])
        vlz_hi = float(boot_var[975])
    else:
        var_lz, vlz_lo, vlz_hi = float("nan"), float("nan"), float("nan")

    # CV(w) - coefficient of variation of normalized weights, averaged across instances
    cvs = []
    for r in all_instance_results:
        lw = np.array(r.get("log_weights", []))
        if len(lw) > 0:
            lw = lw - np.max(lw)  # for numerical stability
            w = np.exp(lw)
            w = w / w.sum()
            cv = float(np.std(w) / np.mean(w)) if np.mean(w) > 0 else float("nan")
            cvs.append(cv)
    cv_mean, cv_lo, cv_hi = bootstrap_ci(cvs)

    # ESS
    ess_vals = [r["ess"] for r in all_instance_results]
    ess_mean, ess_lo, ess_hi = bootstrap_ci(ess_vals)

    return {
        "Accuracy": {"mean": acc_mean, "ci": [acc_lo, acc_hi]},
        "Runtime (s)": {"mean": rt_mean, "ci": [rt_lo, rt_hi]},
        "E[log Z]": {"mean": elz_mean, "ci": [elz_lo, elz_hi]},
        "log E[Z]": {"mean": log_ez, "ci": [lez_lo, lez_hi]},
        "Var(log Z)": {"mean": var_lz, "ci": [vlz_lo, vlz_hi]},
        "CV(w)": {"mean": cv_mean, "ci": [cv_lo, cv_hi]},
        "ESS": {"mean": ess_mean, "ci": [ess_lo, ess_hi]},
    }


def format_metric(m):
    """Format a metric dict as 'mean [lo, hi]'."""
    if np.isnan(m["mean"]):
        return "N/A"
    return f"{m['mean']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"


async def run_single_instance(
    method_name, instance, llm, grammar, domain_text, evaluator,
    n_particles, max_tokens, ess_threshold, block_size,
    twist_head=None, hse=None, seed=42,
):
    """Run a single method on a single instance."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_chat = "instruct" in llm.model.name.lower() if hasattr(llm.model, 'name') else False
    prompt_ids = make_prompt(llm.model.tokenizer, instance, use_chat_format=use_chat)
    llm.prompt_ids = prompt_ids

    # EOS tokens: stop on "))""
    eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"))" in t]
    local_llm = llm.spawn_new_eos(eos_tokens)

    # Token sampler with grammar
    token_sampler = eager_token_sampler(local_llm, grammar)

    # Expensive potential
    expensive = GoalInferenceVALPotential(
        domain_pddl_text=domain_text,
        problem_pddl_text=instance.problem_text,
        fast_downward_cmd=FAST_DOWNWARD_CMD,
        val_cmd=VAL_CMD,
        cache_root=str(CACHE_ROOT),
    )
    coerced = expensive.coerce(local_llm, f=b"".join)

    t0 = time.time()

    if method_name == "baseline":
        # Token-level SMC
        smc = SMC(token_sampler, critic=coerced)
        seq = await smc(n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=max_tokens)
        try:
            posterior = dict(seq.decoded_posterior)
        except Exception:
            posterior = {}
    elif method_name == "block_vanilla":
        # Block SMC, no twist
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(block_size),
            expensive_potential=expensive, llm=local_llm, oracle_twist=False,
        )
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=max_tokens)
        posterior = decode_block_sequences(seq)
    elif method_name == "block_twist":
        # Block SMC, learned twist
        num_blocks_est = max(max_tokens // block_size, 1)
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(block_size),
            expensive_potential=expensive, llm=local_llm,
            twist_head=twist_head, hidden_state_extractor=hse,
            num_blocks=num_blocks_est,
        )
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=max_tokens)
        posterior = decode_block_sequences(seq)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    dt = time.time() - t0

    # Evaluate accuracy
    accuracy = 0.0
    best_response = ""
    if posterior:
        best_response = max(posterior, key=posterior.get)
        eval_result = evaluator.evaluate_sample(instance, best_response)
        accuracy = eval_result.score

    result = {
        "log_ml": float(seq.log_ml),
        "ess": float(seq.ess),
        "wall_time": dt,
        "accuracy": accuracy,
        "n_unique": len(posterior),
        "best_response": best_response,
        "log_weights": [float(w) for w in seq.log_weights],
    }
    return result


async def train_twist_for_domain(
    instances, llm, grammar, domain_text, hse,
    n_particles, max_tokens, block_size, n_train_rounds,
):
    """Train twist head using a few instances from the dataset."""
    buffer = TwistTrainingBuffer()
    n_guided = n_train_rounds // 2
    n_explore = n_train_rounds - n_guided

    # Use first 5 instances (or fewer) for training
    train_instances = instances[:min(5, len(instances))]

    for inst_idx, instance in enumerate(train_instances):
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

        # Guided collection
        for r in range(n_guided):
            np.random.seed(r + inst_idx * 100)
            smc_collect = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=FixedIntervalBoundary(block_size),
                expensive_potential=expensive, llm=local_llm, oracle_twist=False,
            )
            seq = await run_block_smc(smc_collect, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
            stats = await collect_twist_training_data(seq, hse, coerced, buffer)
            print(f"  Inst {inst_idx} Guided {r+1}: ex={stats['n_examples']} pos={stats['n_positive']}")
            buffer.advance_age()

        # Exploration collection (no critic)
        for r in range(n_explore):
            np.random.seed(1000 + r + inst_idx * 100)
            unit_sampler = MultiTokenUnitSampler(
                token_sampler, FixedIntervalBoundary(block_size), max_subunits_per_unit=100
            )
            explore_smc = SMC(unit_sampler)
            seq = await explore_smc(n_particles=n_particles, ess_threshold=0.5, max_tokens=max_tokens)
            stats = await collect_twist_training_data(seq, hse, coerced, buffer)
            print(f"  Inst {inst_idx} Explore {r+1}: ex={stats['n_examples']} pos={stats['n_positive']} neg={stats['n_examples'] - stats['n_positive']}")
            buffer.advance_age()

    n_pos = sum(1 for l in buffer.labels if l > 0.5)
    n_neg = sum(1 for l in buffer.labels if l <= 0.5)
    print(f"\n  Buffer: {len(buffer)} examples, {n_pos}+, {n_neg}-")

    # Train
    d_model = hse.hidden_dim
    device = hse.device
    twist_head = TwistHead(d_model=d_model, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=1e-3)
    train_result = train_twist_step(twist_head, buffer, optimizer, device, n_epochs=20, batch_size=64)
    print(f"  Twist: loss={train_result['loss']:.4f} acc={train_result['accuracy']:.4f}")

    twist_head.eval()
    for p in twist_head.parameters():
        p.requires_grad_(False)

    return twist_head, train_result, {"n_pos": n_pos, "n_neg": n_neg, "buffer_size": len(buffer)}


async def main(args):
    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")

    grammar_text = GRAMMAR_PATH.read_text()
    grammar = BoolCFG.from_lark(grammar_text)
    domain_text = DOMAIN_PATH.read_text()

    print(f"Loading dataset: {args.n_instances} instances from Planetarium")
    ds = GoalInferenceDataset.from_hf_planetarium(
        n_examples=args.n_instances, max_objects=args.max_objects, domains=["blocksworld"]
    )
    instances = list(ds)
    print(f"Loaded {len(instances)} instances")

    evaluator = GoalInferenceEvaluatorWithTools()
    hse = HiddenStateExtractor(llm)

    N = args.n_particles
    T = args.max_tokens
    BS = args.block_size

    # =========================================================================
    # PHASE 1: TRAIN TWIST
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: TRAINING TWIST (block_size={BS}, N={N})")
    print(f"{'='*70}")

    train_t0 = time.time()
    twist_head, train_result, train_stats = await train_twist_for_domain(
        instances, llm, grammar, domain_text, hse,
        n_particles=N, max_tokens=T, block_size=BS,
        n_train_rounds=args.n_train_rounds,
    )
    train_time = time.time() - train_t0
    print(f"  Training time: {train_time:.1f}s")

    # =========================================================================
    # PHASE 2: EVALUATE
    # =========================================================================
    methods = ["baseline", "block_vanilla", "block_twist"]
    all_results = {m: [] for m in methods}

    print(f"\n{'='*70}")
    print(f"PHASE 2: EVALUATION ({len(instances)} instances, N={N}, T={T}, BS={BS})")
    print(f"{'='*70}")

    for inst_idx, instance in enumerate(instances):
        print(f"\n--- Instance {inst_idx} (id={instance.instance_id}): {instance.nl_goal[:80]}... ---")

        for method in methods:
            try:
                result = await run_single_instance(
                    method_name=method,
                    instance=instance,
                    llm=llm, grammar=grammar, domain_text=domain_text,
                    evaluator=evaluator,
                    n_particles=N, max_tokens=T, ess_threshold=0.9,
                    block_size=BS,
                    twist_head=twist_head if method == "block_twist" else None,
                    hse=hse if method == "block_twist" else None,
                    seed=42 + inst_idx,
                )
                all_results[method].append(result)
                print(f"  {method:<15} acc={result['accuracy']:.0f}  "
                      f"log_Z={result['log_ml']:.2f}  "
                      f"ess={result['ess']:.1f}  "
                      f"time={result['wall_time']:.1f}s  "
                      f"uniq={result['n_unique']}  "
                      f"best='{result['best_response'][:50]}'")
            except Exception as e:
                print(f"  {method:<15} FAILED: {e}")
                all_results[method].append({
                    "log_ml": float("-inf"), "ess": 0.0,
                    "wall_time": 0.0, "accuracy": 0.0,
                    "n_unique": 0, "best_response": "",
                    "log_weights": [],
                })

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY (Goal Inference / Planetarium, {len(instances)} instances, N={N}, T={T}, BS={BS})")
    print(f"Training: {train_stats['buffer_size']} examples ({train_stats['n_pos']}+/{train_stats['n_neg']}-), {train_time:.1f}s")
    print(f"{'='*70}")

    method_labels = {
        "baseline": "Baseline (token-level)",
        "block_vanilla": f"Block SMC (vanilla, BS={BS})",
        "block_twist": f"Block SMC (twist, BS={BS})",
    }

    # Compute metrics
    metrics_by_method = {}
    for method in methods:
        metrics_by_method[method] = compute_metrics(all_results[method])

    # Print table
    metric_names = ["Accuracy", "Runtime (s)", "E[log Z]", "log E[Z]", "Var(log Z)", "CV(w)", "ESS"]
    header = f"{'Metric':<15}" + "".join(f"  {method_labels[m]:>30}" for m in methods)
    print(header)
    print("-" * len(header))
    for metric in metric_names:
        row = f"{metric:<15}"
        for method in methods:
            row += f"  {format_metric(metrics_by_method[method][metric]):>30}"
        print(row)

    # Save
    output = {
        "config": {
            "n_instances": len(instances),
            "n_particles": N, "max_tokens": T, "block_size": BS,
            "model": args.model, "n_train_rounds": args.n_train_rounds,
        },
        "training": {
            "buffer_size": train_stats["buffer_size"],
            "n_positive": train_stats["n_pos"],
            "n_negative": train_stats["n_neg"],
            "twist_loss": train_result["loss"],
            "twist_accuracy": train_result["accuracy"],
            "train_time": train_time,
        },
        "results": {m: rs for m, rs in all_results.items()},
        "metrics": {m: metrics_by_method[m] for m in methods},
    }
    output_path = Path(__file__).parent / "goal_inference_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instances", type=int, default=10)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-train-rounds", type=int, default=8,
                        help="Training rounds per instance (half guided, half explore)")
    parser.add_argument("--max-objects", type=int, default=4,
                        help="Maximum number of objects per instance")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
