"""Diagnose why twist training produces degenerate ψ=1.0 for everything.

Key hypothesis: SMC importance weights are extremely skewed, so training
is dominated by a handful of data points.

This script:
1. Collects training data (same as inspect_twist.py Phase 1)
2. Analyzes the weight distribution
3. Trains twist with 3 weight schemes: original, uniform, unit-weight
4. Tests discrimination (AUC) for each
5. Checks hidden state separability (can ANY linear model separate +/−?)

Usage:
    /home/vesteinn/miniconda3/envs/blocksmc/bin/python -u experiments/diagnose_twist.py
"""

import asyncio
import argparse
import copy
import json
import torch
import numpy as np
from pathlib import Path

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.sampler.unit import MultiTokenUnitSampler
from genlm.control.constant import EndOfSequence

from genlm.eval.domains.goal_inference import (
    GoalInferenceDataset,
    GoalInferenceVALPotential,
)

from block_smc.sampler import make_block_smc, run_block_smc
from block_smc.boundary import FixedIntervalBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor

from run_goal_inference import (
    make_prompt,
    GRAMMAR_PATH, DOMAIN_PATH, FAST_DOWNWARD_CMD, VAL_CMD, CACHE_ROOT,
)


def compute_auc(labels, scores):
    """Wilcoxon-Mann-Whitney AUC."""
    pos = [s for s, l in zip(scores, labels) if l > 0.5]
    neg = [s for s, l in zip(scores, labels) if l <= 0.5]
    if not pos or not neg:
        return float("nan")
    n_concordant = sum(1 for p in pos for n in neg if p > n)
    n_tied = sum(1 for p in pos for n in neg if p == n)
    return (n_concordant + 0.5 * n_tied) / (len(pos) * len(neg))


def analyze_weights(buffer):
    """Analyze the distribution of importance weights in the buffer."""
    weights = np.array(buffer.weights)
    labels = np.array(buffer.labels)

    print("\n=== Weight Distribution ===")
    print(f"  N = {len(weights)}")
    print(f"  min={weights.min():.6f}  max={weights.max():.6f}")
    print(f"  mean={weights.mean():.6f}  median={np.median(weights):.6f}")
    print(f"  std={weights.std():.6f}")

    # How concentrated are the weights?
    sorted_w = np.sort(weights)[::-1]
    cumsum = np.cumsum(sorted_w) / sorted_w.sum()
    for frac in [0.5, 0.9, 0.99]:
        n_needed = np.searchsorted(cumsum, frac) + 1
        print(f"  Top {n_needed} samples ({n_needed/len(weights)*100:.1f}%) account for {frac*100:.0f}% of total weight")

    # Effective sample size
    ess = (weights.sum() ** 2) / (weights ** 2).sum()
    print(f"  Effective sample size (ESS): {ess:.1f} / {len(weights)}")

    # Weight distribution by label
    pos_w = weights[labels > 0.5]
    neg_w = weights[labels <= 0.5]
    print(f"\n  By label:")
    print(f"    Positives (n={len(pos_w)}): mean_w={pos_w.mean():.6f}  sum_w={pos_w.sum():.4f}")
    if len(neg_w) > 0:
        print(f"    Negatives (n={len(neg_w)}): mean_w={neg_w.mean():.6f}  sum_w={neg_w.sum():.4f}")
        print(f"    Weight ratio (pos_sum/neg_sum): {pos_w.sum()/max(neg_w.sum(), 1e-12):.1f}x")
    else:
        print(f"    Negatives: NONE")

    # Top-10 heaviest samples
    top_idx = np.argsort(weights)[-10:][::-1]
    print(f"\n  Top-10 heaviest samples:")
    for rank, idx in enumerate(top_idx):
        print(f"    #{rank+1}: w={weights[idx]:.6f}  label={labels[idx]:.0f}  bf={buffer.boundary_fracs[idx]:.3f}")


def analyze_hidden_states(buffer, device):
    """Check if hidden states are separable at all (logistic regression)."""
    labels = np.array(buffer.labels)
    n_pos = (labels > 0.5).sum()
    n_neg = (labels <= 0.5).sum()

    if n_pos == 0 or n_neg == 0:
        print("\n=== Hidden State Separability: SKIPPED (only one class) ===")
        return

    h_all = torch.stack(buffer.hidden_states).to(device)
    y_all = torch.tensor(labels, device=device, dtype=torch.float32)

    # Simple logistic regression (no hidden layers) — can we linearly separate?
    d = h_all.shape[1]
    linear = torch.nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(linear.parameters(), lr=1e-3)

    n = len(y_all)
    for epoch in range(100):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, 256):
            idx = perm[start:start+256]
            logits = linear(h_all[idx]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        logits = linear(h_all).squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y_all).float().mean().item()

    probs_np = probs.cpu().numpy()
    auc = compute_auc(labels.tolist(), probs_np.tolist())

    print(f"\n=== Hidden State Separability (logistic regression, UNIFORM weights) ===")
    print(f"  Linear model accuracy: {acc:.3f}")
    print(f"  Linear model AUC: {auc:.3f}")
    print(f"  (Majority baseline: {max(n_pos, n_neg)/len(labels):.3f})")

    # Hidden state diversity
    h_np = h_all.cpu().numpy()
    norms = np.linalg.norm(h_np, axis=1)
    print(f"\n  Hidden state norms: mean={norms.mean():.2f}  std={norms.std():.2f}")

    # Cosine similarity between random pairs
    n_pairs = min(500, n * (n-1) // 2)
    cos_sims = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        cos = np.dot(h_np[i], h_np[j]) / (norms[i] * norms[j] + 1e-8)
        cos_sims.append(cos)
    print(f"  Pairwise cosine similarity: mean={np.mean(cos_sims):.4f}  std={np.std(cos_sims):.4f}")

    # Within vs between class cosine
    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    within_pos, within_neg, between = [], [], []
    for _ in range(200):
        if len(pos_idx) >= 2:
            i, j = np.random.choice(pos_idx, 2, replace=False)
            within_pos.append(np.dot(h_np[i], h_np[j]) / (norms[i] * norms[j] + 1e-8))
        if len(neg_idx) >= 2:
            i, j = np.random.choice(neg_idx, 2, replace=False)
            within_neg.append(np.dot(h_np[i], h_np[j]) / (norms[i] * norms[j] + 1e-8))
        i = np.random.choice(pos_idx)
        j = np.random.choice(neg_idx)
        between.append(np.dot(h_np[i], h_np[j]) / (norms[i] * norms[j] + 1e-8))

    if within_pos and within_neg:
        print(f"  Within-pos cosine: {np.mean(within_pos):.4f}")
        print(f"  Within-neg cosine: {np.mean(within_neg):.4f}")
        print(f"  Between cosine:    {np.mean(between):.4f}")


def train_and_eval(buffer, device, weight_scheme, class_balance=False, label=""):
    """Train twist head and compute AUC on training data."""
    buf = copy.deepcopy(buffer)

    # Override boundary fracs (bf=False was best)
    buf.boundary_fracs = [0.0] * len(buf)

    # Override weights based on scheme
    if weight_scheme == "uniform":
        buf.weights = [1.0] * len(buf)

    d_model = buf.hidden_states[0].shape[0]
    twist_head = TwistHead(d_model=d_model, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=5e-3)
    result = train_twist_step(twist_head, buf, optimizer, device,
                              n_epochs=50, batch_size=64, class_balance=class_balance,
                              discount=1.0)

    # Compute AUC
    h, _, y, bf = buf.to_tensors(device, discount=1.0)
    labels_np = y.cpu().numpy()
    with torch.no_grad():
        psi = twist_head(h, bf).cpu().numpy()

    n_pos = (labels_np > 0.5).sum()
    n_neg = (labels_np <= 0.5).sum()
    auc = compute_auc(labels_np.tolist(), psi.tolist()) if n_pos > 0 and n_neg > 0 else float("nan")

    pos_psi = psi[labels_np > 0.5]
    neg_psi = psi[labels_np <= 0.5]

    cb_str = "+cb" if class_balance else ""
    print(f"\n  [{label}] weights={weight_scheme}{cb_str}")
    print(f"    Loss: {result['loss']:.4f}  Acc: {result['accuracy']:.3f}  AUC: {auc:.3f}")
    print(f"    ψ positives: mean={pos_psi.mean():.4f}  std={pos_psi.std():.4f}")
    if len(neg_psi) > 0:
        print(f"    ψ negatives: mean={neg_psi.mean():.4f}  std={neg_psi.std():.4f}")
    print(f"    ψ range: [{psi.min():.4f}, {psi.max():.4f}]")

    return {"weight_scheme": weight_scheme, "class_balance": class_balance,
            "loss": result["loss"], "accuracy": result["accuracy"], "auc": auc,
            "psi_pos_mean": float(pos_psi.mean()), "psi_neg_mean": float(neg_psi.mean()) if len(neg_psi) > 0 else None}


async def main(args):
    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")

    grammar_text = GRAMMAR_PATH.read_text()
    grammar = BoolCFG.from_lark(grammar_text)
    domain_text = DOMAIN_PATH.read_text()

    hse = HiddenStateExtractor(llm)
    device = hse.device

    ds = GoalInferenceDataset.from_hf_planetarium(
        n_examples=args.n_train_instances, max_objects=args.max_objects,
        domains=["blocksworld"]
    )
    train_instances = list(ds)
    print(f"Training on {len(train_instances)} instances")

    num_blocks_est = max(args.max_tokens // args.block_size, 1)

    # ==================================================================
    # Phase 1: Collect training data
    # ==================================================================
    print(f"\n{'='*60}")
    print("Phase 1: Collecting training data")
    print(f"{'='*60}")

    buffer = TwistTrainingBuffer()

    for inst_idx, instance in enumerate(train_instances):
        print(f"  Instance {inst_idx+1}/{len(train_instances)}...")
        model_name = getattr(llm.model, 'name', getattr(llm.model, 'model_name', args.model))
        use_chat = "instruct" in model_name.lower()
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

        # Guided rounds (with constraint potential)
        for r in range(4):
            np.random.seed(r + inst_idx * 100)
            smc_collect = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=FixedIntervalBoundary(args.block_size),
                expensive_potential=expensive, llm=local_llm, oracle_twist=False,
            )
            seq = await run_block_smc(smc_collect, n_particles=args.n_particles,
                                       ess_threshold=0.9, max_tokens=args.max_tokens)
            stats = await collect_twist_training_data(seq, hse, coerced, buffer,
                                                      num_blocks_est=num_blocks_est)
            print(f"    Guided {r}: {stats['n_examples']} ex, {stats['n_positive']}+, {stats['n_examples']-stats['n_positive']}-")
            buffer.advance_age()

        # Exploration rounds (no constraint → diverse negatives)
        for r in range(4):
            np.random.seed(1000 + r + inst_idx * 100)
            unit_sampler = MultiTokenUnitSampler(
                token_sampler, FixedIntervalBoundary(args.block_size),
                max_subunits_per_unit=100
            )
            explore_smc = SMC(unit_sampler)
            seq = await explore_smc(n_particles=args.n_particles,
                                     ess_threshold=0.5, max_tokens=args.max_tokens)
            stats = await collect_twist_training_data(seq, hse, coerced, buffer,
                                                      num_blocks_est=num_blocks_est)
            print(f"    Explore {r}: {stats['n_examples']} ex, {stats['n_positive']}+, {stats['n_examples']-stats['n_positive']}-")
            buffer.advance_age()

    n_pos = sum(1 for l in buffer.labels if l > 0.5)
    n_neg = sum(1 for l in buffer.labels if l <= 0.5)
    print(f"\n  Buffer total: {len(buffer)} ({n_pos}+ / {n_neg}-)")

    # ==================================================================
    # Phase 2: Weight distribution analysis
    # ==================================================================
    print(f"\n{'='*60}")
    print("Phase 2: Weight distribution analysis")
    print(f"{'='*60}")

    analyze_weights(buffer)

    # ==================================================================
    # Phase 3: Hidden state analysis
    # ==================================================================
    print(f"\n{'='*60}")
    print("Phase 3: Hidden state separability")
    print(f"{'='*60}")

    analyze_hidden_states(buffer, device)

    # ==================================================================
    # Phase 4: Train twist with different weight schemes
    # ==================================================================
    print(f"\n{'='*60}")
    print("Phase 4: Training with different weight schemes")
    print(f"{'='*60}")

    results = []
    results.append(train_and_eval(buffer, device, "original", class_balance=False, label="A"))
    results.append(train_and_eval(buffer, device, "uniform", class_balance=False, label="B"))
    results.append(train_and_eval(buffer, device, "uniform", class_balance=True, label="C"))
    results.append(train_and_eval(buffer, device, "original", class_balance=True, label="D"))

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  {'Scheme':<35} {'Loss':>8} {'Acc':>8} {'AUC':>8} {'ψ_pos':>8} {'ψ_neg':>8}")
    print(f"  {'-'*77}")
    for r in results:
        cb_str = "+cb" if r["class_balance"] else ""
        name = f"{r['weight_scheme']}{cb_str}"
        neg_str = f"{r['psi_neg_mean']:.4f}" if r['psi_neg_mean'] is not None else "n/a"
        print(f"  {name:<35} {r['loss']:>8.4f} {r['accuracy']:>8.3f} {r['auc']:>8.3f} {r['psi_pos_mean']:>8.4f} {neg_str:>8}")

    # Save
    output_path = Path(__file__).parent / "diagnose_results.json"
    output = {
        "buffer_size": len(buffer),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train-instances", type=int, default=5)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
