"""Twist function diagnostics: Properties B through F.

Tests whether the learned twist generalizes, is calibrated, matches
the inference distribution, is directionally correct, and preserves
particle diversity.

Usage:
    conda run -n blocksmc python experiments/diagnose_twist.py \
        --load-twist experiments/twist_weights_obj9.pt \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --n-test-instances 10
"""

import asyncio
import argparse
import copy
import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.constant import EndOfSequence
from genlm.control.sampler.unit import MultiTokenUnitSampler, flatten_units

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor
from block_smc.critic import TwistedBlockCritic, _flatten_context

from run_goal_inference import (
    GoalInferenceEvaluatorWithTools,
    make_prompt,
    GRAMMAR_PATH, DOMAIN_PATH, FAST_DOWNWARD_CMD, VAL_CMD, CACHE_ROOT,
)
from genlm.eval.domains.goal_inference import (
    GoalInferenceDataset,
    GoalInferenceVALPotential,
)


# ============================================================
# Property B: Generalization
# ============================================================

def evaluate_twist_accuracy(twist_head, buffer, device):
    """Evaluate twist classification accuracy on a buffer of (h, label) pairs."""
    if len(buffer) == 0:
        return {"accuracy": float("nan"), "n": 0}

    h, w, y, bf = buffer.to_tensors(device, discount=1.0)
    with torch.no_grad():
        psi = twist_head(h, bf)
        preds = (psi > 0.5).float()
        correct = (preds == y).float()
        acc = correct.mean().item()

        # Per-class
        pos_mask = y > 0.5
        neg_mask = y <= 0.5
        tpr = correct[pos_mask].mean().item() if pos_mask.any() else float("nan")
        tnr = correct[neg_mask].mean().item() if neg_mask.any() else float("nan")

        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

    return {
        "accuracy": acc,
        "tpr": tpr,  # true positive rate (sensitivity)
        "tnr": tnr,  # true negative rate (specificity)
        "n": len(buffer),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ============================================================
# Property C: Calibration
# ============================================================

def compute_calibration(twist_head, buffer, device, n_bins=10):
    """Compute reliability diagram and ECE."""
    if len(buffer) == 0:
        return {"ece": float("nan"), "bins": []}

    h, w, y, bf = buffer.to_tensors(device, discount=1.0)
    with torch.no_grad():
        psi = twist_head(h, bf).cpu().numpy()
    labels = y.cpu().numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (psi >= lo) & (psi < hi) if i < n_bins - 1 else (psi >= lo) & (psi <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "mean_psi": 0, "actual_pos_rate": 0})
            continue

        mean_psi = psi[mask].mean()
        actual_pos_rate = labels[mask].mean()
        ece += (n_in_bin / total) * abs(actual_pos_rate - mean_psi)
        bins.append({
            "lo": float(lo), "hi": float(hi), "n": int(n_in_bin),
            "mean_psi": float(mean_psi), "actual_pos_rate": float(actual_pos_rate),
        })

    return {"ece": float(ece), "bins": bins}


# ============================================================
# Property D: Distribution match
# ============================================================

def compute_distribution_stats(twist_head, buffer, device):
    """Compute logit and psi distribution statistics."""
    if len(buffer) == 0:
        return {}

    h, w, y, bf = buffer.to_tensors(device, discount=1.0)
    with torch.no_grad():
        psi = twist_head(h, bf).cpu().numpy()
        # Get raw logits
        if bf.dim() < h.dim():
            bf = bf.unsqueeze(-1)
        x = torch.cat([h, bf], dim=-1)
        logits = twist_head.net(x).squeeze(-1).cpu().numpy()

    return {
        "logit_mean": float(np.mean(logits)),
        "logit_std": float(np.std(logits)),
        "logit_min": float(np.min(logits)),
        "logit_max": float(np.max(logits)),
        "psi_mean": float(np.mean(psi)),
        "psi_std": float(np.std(psi)),
        "psi_lt_001": float((psi < 0.01).mean()),
        "psi_gt_099": float((psi > 0.99).mean()),
        "psi_lt_010": float((psi < 0.10).mean()),
        "psi_gt_090": float((psi > 0.90).mean()),
    }


# ============================================================
# Property E: Directional correctness
# ============================================================

def compute_directional_correctness(per_particle_data):
    """Compute Spearman correlation between psi and outcome per instance."""
    correlations = []
    for inst in per_particle_data:
        psi_vals = np.array(inst["psi_values"])
        outcomes = np.array(inst["outcomes"])

        if len(np.unique(outcomes)) < 2 or len(np.unique(psi_vals)) < 2:
            continue

        corr, pval = spearmanr(psi_vals, outcomes)
        correlations.append({"corr": float(corr), "pval": float(pval), "n": len(psi_vals)})

    if not correlations:
        return {"mean_corr": float("nan"), "n_instances": 0, "n_positive_corr": 0, "per_instance": []}

    mean_corr = np.mean([c["corr"] for c in correlations])
    n_positive = sum(1 for c in correlations if c["corr"] > 0)
    return {
        "mean_corr": float(mean_corr),
        "n_instances": len(correlations),
        "n_positive_corr": n_positive,
        "per_instance": correlations,
    }


# ============================================================
# Main data collection
# ============================================================

async def collect_data_from_instances(
    instances, llm, grammar, hse, evaluator, twist_mlp,
    n_particles, max_tokens, block_size, label="",
):
    """Run vanilla Block SMC on instances, collect hidden states + labels + per-particle psi/outcomes."""
    buffer = TwistTrainingBuffer()
    per_particle_data = []
    num_blocks_est = max(max_tokens // block_size, 1)
    domain_text = DOMAIN_PATH.read_text()
    device = hse.device

    for inst_idx, instance in enumerate(instances):
        print(f"  {label} instance {inst_idx+1}/{len(instances)}...")
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

        np.random.seed(42 + inst_idx)
        torch.manual_seed(42 + inst_idx)

        # Run vanilla Block SMC (no twist)
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(block_size),
            expensive_potential=expensive, llm=local_llm,
        )
        seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)

        # Collect (hidden_state, label) pairs
        await collect_twist_training_data(seq, hse, coerced, buffer, num_blocks_est=num_blocks_est)

        # Per-particle psi and outcome (for Property E)
        inst_psi_values = []
        inst_outcomes = []

        for p_idx in range(len(seq.contexts)):
            ctx = seq.contexts[p_idx]
            units = [u for u in ctx if isinstance(u, list)]
            if not units:
                continue

            # Flatten all tokens
            cumulative_tokens = []
            for unit in units:
                cumulative_tokens.extend(unit)
            byte_toks = [t for t in cumulative_tokens if isinstance(t, bytes)]
            if not byte_toks:
                continue

            token_ids = hse.token_ids_from_bytes(byte_toks)
            if not token_ids:
                continue

            h = hse.extract(token_ids, position=-1)
            bf_tensor = torch.tensor([0.0], device=device, dtype=h.dtype)

            with torch.no_grad():
                psi_val = twist_mlp(h.unsqueeze(0), bf_tensor).item()

            # Evaluate outcome
            try:
                decoded = b"".join(byte_toks).decode("utf-8")
                score = evaluator.evaluate_sample(instance, decoded).score
                outcome = 1.0 if score > 0 else 0.0
            except Exception:
                outcome = 0.0

            inst_psi_values.append(psi_val)
            inst_outcomes.append(outcome)

        per_particle_data.append({
            "instance_idx": inst_idx,
            "psi_values": inst_psi_values,
            "outcomes": inst_outcomes,
        })

        n_ok = sum(inst_outcomes)
        print(f"    {n_ok}/{len(inst_outcomes)} particles OK, buffer: {len(buffer)}")

    return buffer, per_particle_data


async def run_diversity_comparison(
    instances, llm, grammar, hse,
    twist_mlp, n_particles, max_tokens, block_size,
):
    """Property F: Compare particle diversity with and without twist."""
    num_blocks_est = max(max_tokens // block_size, 1)
    domain_text = DOMAIN_PATH.read_text()
    results = []

    for inst_idx, instance in enumerate(instances):
        print(f"  Instance {inst_idx+1}/{len(instances)}...")
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

        inst_result = {"instance_idx": inst_idx}

        for mode, tw, ts in [("vanilla", None, 0.0), ("twist_s0.1", twist_mlp, 0.1)]:
            np.random.seed(42 + inst_idx)
            torch.manual_seed(42 + inst_idx)

            smc = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=FixedIntervalBoundary(block_size),
                expensive_potential=expensive, llm=local_llm,
                twist_head=tw, hidden_state_extractor=hse if tw else None,
                num_blocks=num_blocks_est if tw else None,
                twist_scale=ts,
            )
            seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)

            # Count unique final particles
            final_strs = set()
            for ctx in seq.contexts:
                flat = _flatten_context(ctx)
                byte_toks = [t for t in flat if isinstance(t, bytes)]
                try:
                    final_strs.add(b"".join(byte_toks))
                except Exception:
                    pass

            inst_result[mode] = {
                "n_unique_final": len(final_strs),
                "ess": float(seq.ess),
                "log_ml": float(seq.log_ml),
            }
            print(f"    {mode}: {len(final_strs)} unique, ESS={seq.ess:.1f}")

        results.append(inst_result)
    return results


# ============================================================
# Report
# ============================================================

def print_report(results):
    """Print the diagnostic report."""
    print(f"\n{'='*70}")
    print("TWIST DIAGNOSTIC REPORT")
    print(f"{'='*70}")

    # Property B
    print(f"\n--- Property B: Generalization ---")
    for name in ["mlp", "linear"]:
        train = results["B"].get(f"train_{name}", {})
        test = results["B"].get(f"test_{name}", {})
        if not train or not test or np.isnan(train.get("accuracy", float("nan"))):
            continue
        drop = train["accuracy"] - test["accuracy"]
        status = "PASS" if test["accuracy"] > 0.80 else ("WARN" if test["accuracy"] > 0.60 else "FAIL")
        print(f"  {name.upper():>7s} train: {train['accuracy']:.3f}  test: {test['accuracy']:.3f}  "
              f"(drop: {drop:+.3f})  -> {status}")
        print(f"          TPR: {test['tpr']:.3f}  TNR: {test['tnr']:.3f}  "
              f"(n={test['n']}, {test['n_pos']}+ / {test['n_neg']}-)")

    # Property C
    print(f"\n--- Property C: Calibration ---")
    for name in ["mlp", "linear"]:
        for split in ["train", "test"]:
            cal = results["C"].get(f"{split}_{name}", {})
            if not cal or np.isnan(cal.get("ece", float("nan"))):
                continue
            status = "PASS" if cal["ece"] < 0.10 else ("WARN" if cal["ece"] < 0.20 else "FAIL")
            print(f"  {name.upper():>7s} {split:>5s}: ECE={cal['ece']:.4f}  -> {status}")

        cal = results["C"].get(f"test_{name}", {})
        if cal and cal.get("bins"):
            print(f"          Reliability bins (test):")
            for b in cal["bins"]:
                if b["n"] > 0:
                    bar = "#" * int(b["actual_pos_rate"] * 20)
                    print(f"            [{b['lo']:.1f}-{b['hi']:.1f}] n={b['n']:4d}  "
                          f"pred={b['mean_psi']:.3f}  actual={b['actual_pos_rate']:.3f}  {bar}")

    # Property D
    print(f"\n--- Property D: Distribution Match ---")
    for name in ["mlp", "linear"]:
        train_d = results["D"].get(f"train_{name}", {})
        test_d = results["D"].get(f"test_{name}", {})
        if not train_d or not test_d:
            continue
        print(f"  {name.upper():>7s}  {'':>6s} {'logit_m':>8s} {'logit_s':>8s} "
              f"{'psi<.01':>8s} {'psi>.99':>8s} {'psi<.10':>8s} {'psi>.90':>8s}")
        print(f"          train: {train_d['logit_mean']:>8.2f} {train_d['logit_std']:>8.2f} "
              f"{train_d['psi_lt_001']:>8.1%} {train_d['psi_gt_099']:>8.1%} "
              f"{train_d['psi_lt_010']:>8.1%} {train_d['psi_gt_090']:>8.1%}")
        print(f"          test:  {test_d['logit_mean']:>8.2f} {test_d['logit_std']:>8.2f} "
              f"{test_d['psi_lt_001']:>8.1%} {test_d['psi_gt_099']:>8.1%} "
              f"{test_d['psi_lt_010']:>8.1%} {test_d['psi_gt_090']:>8.1%}")

    # Property E
    print(f"\n--- Property E: Directional Correctness ---")
    e = results["E"]
    if e["n_instances"] > 0:
        status = "PASS" if e["mean_corr"] > 0.2 else ("WARN" if e["mean_corr"] > 0 else "FAIL")
        print(f"  Mean Spearman corr: {e['mean_corr']:.3f}  -> {status}")
        print(f"  Positive corr: {e['n_positive_corr']}/{e['n_instances']} instances")
        for i, c in enumerate(e["per_instance"]):
            print(f"    Instance {i}: corr={c['corr']:.3f}  p={c['pval']:.3f}  n={c['n']}")
    else:
        print(f"  SKIP: No instances with variance in both psi and outcome")

    # Property F
    print(f"\n--- Property F: Particle Diversity ---")
    if results["F"]:
        van_unique = [r["vanilla"]["n_unique_final"] for r in results["F"]]
        tw_unique = [r["twist_s0.1"]["n_unique_final"] for r in results["F"]]
        van_ess = [r["vanilla"]["ess"] for r in results["F"]]
        tw_ess = [r["twist_s0.1"]["ess"] for r in results["F"]]
        ratio = np.mean(tw_unique) / max(np.mean(van_unique), 1e-8)
        status = "PASS" if ratio >= 0.7 else ("WARN" if ratio >= 0.5 else "FAIL")
        print(f"  Unique particles -- vanilla: {np.mean(van_unique):.1f}  "
              f"twist: {np.mean(tw_unique):.1f}  ratio: {ratio:.2f}  -> {status}")
        print(f"  ESS             -- vanilla: {np.mean(van_ess):.1f}  twist: {np.mean(tw_ess):.1f}")
        for r in results["F"]:
            print(f"    Instance {r['instance_idx']}: "
                  f"vanilla={r['vanilla']['n_unique_final']} unique, ESS={r['vanilla']['ess']:.1f}  |  "
                  f"twist={r['twist_s0.1']['n_unique_final']} unique, ESS={r['twist_s0.1']['ess']:.1f}")

    print(f"\n{'='*70}")


# ============================================================
# Main
# ============================================================

async def main(args):
    output_path = Path(__file__).parent / "twist_diagnostics.json"

    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")
    grammar = BoolCFG.from_lark(GRAMMAR_PATH.read_text())
    evaluator = GoalInferenceEvaluatorWithTools()
    hse = HiddenStateExtractor(llm)
    device = hse.device

    # Load twist weights
    print(f"Loading twist weights from {args.load_twist}")
    checkpoint = torch.load(args.load_twist, map_location=device)
    twist_mlp = TwistHead(d_model=hse.hidden_dim, hidden_dim=checkpoint.get("hidden_dim", 256)).to(device)
    twist_mlp.load_state_dict(checkpoint["state_dict"])
    twist_mlp.eval()

    twist_linear = None
    if "linear_state_dict" in checkpoint:
        twist_linear = TwistHead(
            d_model=hse.hidden_dim,
            hidden_dim=checkpoint.get("linear_hidden_dim", 0),
            dropout=checkpoint.get("linear_dropout", 0.3),
        ).to(device)
        twist_linear.load_state_dict(checkpoint["linear_state_dict"])
        twist_linear.eval()
        print("  Loaded MLP + linear probe twists.")
    else:
        print("  Loaded MLP twist only.")

    # Load dataset (same split as holdout_test.py)
    total_needed = args.n_train_instances + args.n_test_instances
    print(f"Loading dataset: {total_needed} instances (max_objects={args.max_objects})")
    ds = GoalInferenceDataset.from_hf_planetarium(
        n_examples=total_needed, max_objects=args.max_objects, domains=["blocksworld"]
    )
    all_instances = list(ds)
    train_instances = all_instances[:args.n_train_instances]
    test_instances = all_instances[args.n_train_instances:args.n_train_instances + args.n_test_instances]
    print(f"Train: {len(train_instances)}, Test: {len(test_instances)}")

    # ---- Phase 0: Collect training data for comparison ----
    print(f"\n{'='*60}")
    print("Phase 0: Collecting TRAINING data (5 instances for comparison)")
    print(f"{'='*60}")
    train_buffer, _ = await collect_data_from_instances(
        train_instances[:5], llm, grammar, hse, evaluator, twist_mlp,
        n_particles=args.n_particles, max_tokens=args.max_tokens,
        block_size=args.block_size, label="Train",
    )
    n_pos = sum(1 for l in train_buffer.labels if l > 0.5)
    n_neg = sum(1 for l in train_buffer.labels if l <= 0.5)
    print(f"  Train buffer: {len(train_buffer)} ({n_pos}+ / {n_neg}-)")

    # ---- Phase 1: Collect test data ----
    print(f"\n{'='*60}")
    print(f"Phase 1: Collecting TEST data ({len(test_instances)} instances)")
    print(f"{'='*60}")
    test_buffer, per_particle_data = await collect_data_from_instances(
        test_instances, llm, grammar, hse, evaluator, twist_mlp,
        n_particles=args.n_particles, max_tokens=args.max_tokens,
        block_size=args.block_size, label="Test",
    )
    n_pos = sum(1 for l in test_buffer.labels if l > 0.5)
    n_neg = sum(1 for l in test_buffer.labels if l <= 0.5)
    print(f"  Test buffer: {len(test_buffer)} ({n_pos}+ / {n_neg}-)")

    # Zero boundary fracs (bf=False, matching training config)
    train_buf_eval = copy.deepcopy(train_buffer)
    train_buf_eval.boundary_fracs = [0.0] * len(train_buf_eval)
    test_buf_eval = copy.deepcopy(test_buffer)
    test_buf_eval.boundary_fracs = [0.0] * len(test_buf_eval)

    # ---- Phase 2: Property B ----
    print(f"\n{'='*60}")
    print("Phase 2: Property B -- Generalization")
    print(f"{'='*60}")
    results_B = {
        "train_mlp": evaluate_twist_accuracy(twist_mlp, train_buf_eval, device),
        "test_mlp": evaluate_twist_accuracy(twist_mlp, test_buf_eval, device),
    }
    print(f"  MLP    train: {results_B['train_mlp']['accuracy']:.3f}  test: {results_B['test_mlp']['accuracy']:.3f}")
    if twist_linear:
        results_B["train_linear"] = evaluate_twist_accuracy(twist_linear, train_buf_eval, device)
        results_B["test_linear"] = evaluate_twist_accuracy(twist_linear, test_buf_eval, device)
        print(f"  Linear train: {results_B['train_linear']['accuracy']:.3f}  test: {results_B['test_linear']['accuracy']:.3f}")

    # ---- Phase 3: Property C ----
    print(f"\n{'='*60}")
    print("Phase 3: Property C -- Calibration")
    print(f"{'='*60}")
    results_C = {
        "train_mlp": compute_calibration(twist_mlp, train_buf_eval, device),
        "test_mlp": compute_calibration(twist_mlp, test_buf_eval, device),
    }
    print(f"  MLP    ECE train: {results_C['train_mlp']['ece']:.4f}  test: {results_C['test_mlp']['ece']:.4f}")
    if twist_linear:
        results_C["train_linear"] = compute_calibration(twist_linear, train_buf_eval, device)
        results_C["test_linear"] = compute_calibration(twist_linear, test_buf_eval, device)
        print(f"  Linear ECE train: {results_C['train_linear']['ece']:.4f}  test: {results_C['test_linear']['ece']:.4f}")

    # ---- Phase 4: Property D ----
    print(f"\n{'='*60}")
    print("Phase 4: Property D -- Distribution Match")
    print(f"{'='*60}")
    results_D = {
        "train_mlp": compute_distribution_stats(twist_mlp, train_buf_eval, device),
        "test_mlp": compute_distribution_stats(twist_mlp, test_buf_eval, device),
    }
    if twist_linear:
        results_D["train_linear"] = compute_distribution_stats(twist_linear, train_buf_eval, device)
        results_D["test_linear"] = compute_distribution_stats(twist_linear, test_buf_eval, device)

    # ---- Phase 5: Property E ----
    print(f"\n{'='*60}")
    print("Phase 5: Property E -- Directional Correctness")
    print(f"{'='*60}")
    results_E = compute_directional_correctness(per_particle_data)
    if results_E["n_instances"] > 0:
        print(f"  Mean Spearman corr: {results_E['mean_corr']:.3f} ({results_E['n_instances']} instances)")
    else:
        print(f"  No instances with variance in both psi and outcome")

    # ---- Phase 6: Property F ----
    print(f"\n{'='*60}")
    print("Phase 6: Property F -- Particle Diversity")
    print(f"{'='*60}")
    results_F = await run_diversity_comparison(
        test_instances[:5], llm, grammar, hse,
        twist_mlp, n_particles=args.n_particles, max_tokens=args.max_tokens,
        block_size=args.block_size,
    )

    # ---- Full report ----
    all_results = {"B": results_B, "C": results_C, "D": results_D, "E": results_E, "F": results_F}
    print_report(all_results)

    # Save
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=make_serializable)
    print(f"\nSaved raw data to {output_path}")
    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twist function diagnostics")
    parser.add_argument("--load-twist", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n-train-instances", type=int, default=20)
    parser.add_argument("--n-test-instances", type=int, default=10)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--max-objects", type=int, default=9)
    args = parser.parse_args()

    asyncio.run(main(args))
