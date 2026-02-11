"""Inspect twist predictions in isolation: do they make sense?

Test A — Prefix inspection:
    For each instance, generate particles via block SMC (no twist).
    At each block boundary, show the actual text + ψ prediction.
    Compare ψ for particles that ultimately succeed vs fail.

Test B — Discrimination (AUC):
    From the training buffer, compute ψ for all stored hidden states.
    Measure AUC: does ψ rank positives (label=1) higher than negatives (label=0)?
    Break down by boundary position.

Usage:
    conda run -n blocksmc python experiments/inspect_twist.py
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
from genlm.control.sampler.unit import MultiTokenUnitSampler
from genlm.control.constant import EndOfSequence
from genlm.eval.domains.goal_inference import (
    GoalInferenceDataset,
    GoalInferenceVALPotential,
)

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor
from block_smc.critic import _flatten_context

from run_goal_inference import (
    GoalInferenceEvaluatorWithTools,
    make_prompt,
    GRAMMAR_PATH, DOMAIN_PATH, FAST_DOWNWARD_CMD, VAL_CMD, CACHE_ROOT,
)


def compute_auc(labels, scores):
    """Compute AUC from labels and scores."""
    pos = [s for s, l in zip(scores, labels) if l > 0.5]
    neg = [s for s, l in zip(scores, labels) if l <= 0.5]
    if not pos or not neg:
        return float("nan")
    # Wilcoxon-Mann-Whitney statistic
    n_concordant = sum(1 for p in pos for n in neg if p > n)
    n_tied = sum(1 for p in pos for n in neg if p == n)
    return (n_concordant + 0.5 * n_tied) / (len(pos) * len(neg))


async def inspect_instance(
    instance, llm, grammar, hse, twist_head, evaluator,
    n_particles, max_tokens, block_size, num_blocks_est,
):
    """Run vanilla block SMC on one instance and inspect ψ at each boundary."""
    domain_text = DOMAIN_PATH.read_text()
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

    # Run vanilla block SMC (no twist) so particles are unbiased
    smc = make_block_smc(
        token_sampler=token_sampler,
        boundary_predicate=FixedIntervalBoundary(block_size),
        expensive_potential=expensive, llm=local_llm,
    )
    seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)

    # Now inspect each particle's trajectory
    particle_data = []
    contexts = seq.contexts
    norm_weights = seq.normalized_weights

    for i in range(len(contexts)):
        ctx = contexts[i]
        w_i = float(norm_weights[i])

        units = [u for u in ctx if isinstance(u, list)]
        if not units:
            continue

        flat_all = _flatten_context(ctx)
        has_eos = any(isinstance(t, EndOfSequence) for t in flat_all)
        all_byte_tokens = [t for t in flat_all if isinstance(t, bytes)]

        # Get final outcome
        if has_eos and all_byte_tokens:
            complete_score = await coerced.complete(all_byte_tokens)
            final_success = complete_score > float("-inf")
        else:
            final_success = False

        full_text = b"".join(all_byte_tokens).decode("utf-8", errors="replace")

        # Walk boundaries
        boundaries = []
        cumulative_tokens = []
        for k, unit in enumerate(units):
            cumulative_tokens.extend(unit)
            byte_toks = [t for t in cumulative_tokens if isinstance(t, bytes)]
            if not byte_toks:
                continue

            token_ids = hse.token_ids_from_bytes(byte_toks)
            if not token_ids:
                continue

            # Get text at this boundary
            text_so_far = b"".join(byte_toks).decode("utf-8", errors="replace")
            # Just the new block's text
            block_bytes = [t for t in unit if isinstance(t, bytes)]
            block_text = b"".join(block_bytes).decode("utf-8", errors="replace")

            # Get twist prediction
            h = hse.extract(token_ids, position=-1)
            boundary_frac = (k + 1) / num_blocks_est
            with torch.no_grad():
                bf_tensor = torch.tensor([boundary_frac], device=h.device, dtype=h.dtype)
                log_psi = twist_head.log_psi(h.unsqueeze(0), bf_tensor).item()
                psi = np.exp(log_psi)

            # Get prefix label (is prefix still viable?)
            prefix_score = await coerced.prefix(byte_toks)
            prefix_ok = prefix_score > float("-inf")

            boundaries.append({
                "k": k,
                "boundary_frac": boundary_frac,
                "block_text": block_text,
                "text_so_far": text_so_far,
                "psi": psi,
                "log_psi": log_psi,
                "prefix_ok": prefix_ok,
            })

        particle_data.append({
            "particle_id": i,
            "weight": w_i,
            "final_success": final_success,
            "full_text": full_text,
            "n_blocks": len(units),
            "boundaries": boundaries,
        })

    return particle_data


async def main(args):
    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")

    grammar_text = GRAMMAR_PATH.read_text()
    grammar = BoolCFG.from_lark(grammar_text)
    domain_text = DOMAIN_PATH.read_text()

    evaluator = GoalInferenceEvaluatorWithTools()
    hse = HiddenStateExtractor(llm)

    n_train = args.n_train_instances
    n_inspect = args.n_inspect_instances
    total = n_train + n_inspect
    ds = GoalInferenceDataset.from_hf_planetarium(
        n_examples=total, max_objects=args.max_objects, domains=["blocksworld"]
    )
    all_instances = list(ds)
    train_instances = all_instances[:n_train]
    inspect_instances = all_instances[n_train:n_train + n_inspect]
    print(f"Train: {len(train_instances)}, Inspect: {len(inspect_instances)}")

    num_blocks_est = max(args.max_tokens // args.block_size, 1)

    # ================================================================
    # Phase 1: Collect training data and train twist
    # ================================================================
    print(f"\n{'='*60}")
    print("Phase 1: Training twist")
    print(f"{'='*60}")

    buffer = TwistTrainingBuffer()
    n_guided = args.n_train_rounds // 2
    n_explore = args.n_train_rounds - n_guided

    for inst_idx, instance in enumerate(train_instances):
        print(f"  Instance {inst_idx+1}/{len(train_instances)}...")
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
                boundary_predicate=FixedIntervalBoundary(args.block_size),
                expensive_potential=expensive, llm=local_llm, oracle_twist=False,
            )
            seq = await run_block_smc(smc_collect, n_particles=args.n_particles, ess_threshold=0.9, max_tokens=args.max_tokens)
            await collect_twist_training_data(seq, hse, coerced, buffer, num_blocks_est=num_blocks_est)
            buffer.advance_age()

        for r in range(n_explore):
            np.random.seed(1000 + r + inst_idx * 100)
            unit_sampler = MultiTokenUnitSampler(
                token_sampler, FixedIntervalBoundary(args.block_size), max_subunits_per_unit=100
            )
            explore_smc = SMC(unit_sampler)
            seq = await explore_smc(n_particles=args.n_particles, ess_threshold=0.5, max_tokens=args.max_tokens)
            await collect_twist_training_data(seq, hse, coerced, buffer, num_blocks_est=num_blocks_est)
            buffer.advance_age()

    n_pos = sum(1 for l in buffer.labels if l > 0.5)
    n_neg = sum(1 for l in buffer.labels if l <= 0.5)
    print(f"  Buffer: {len(buffer)} ({n_pos}+ / {n_neg}-)")

    # Train with uniform weights + class balance (fixes ψ=1.0 collapse)
    buf_train = copy.deepcopy(buffer)
    buf_train.boundary_fracs = [0.0] * len(buf_train)  # bf=False
    twist_head = TwistHead(d_model=hse.hidden_dim, hidden_dim=256).to(hse.device)
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=5e-3)
    train_result = train_twist_step(twist_head, buf_train, optimizer, hse.device,
                                     n_epochs=50, batch_size=64,
                                     class_balance=True, uniform_weights=True)
    print(f"  Twist: loss={train_result['loss']:.4f} acc={train_result['accuracy']:.3f}")
    twist_head.eval()
    for p in twist_head.parameters():
        p.requires_grad_(False)

    # ================================================================
    # Test B: AUC on training buffer
    # ================================================================
    print(f"\n{'='*60}")
    print("Test B: Discrimination (AUC) on training data")
    print(f"{'='*60}")

    all_psi = []
    all_labels = []
    with torch.no_grad():
        for i in range(len(buffer)):
            h = buffer.hidden_states[i].unsqueeze(0).to(hse.device)
            # Use boundary_frac=0 since we trained with bf=False
            bf = torch.tensor([0.0], device=hse.device, dtype=h.dtype)
            log_psi = twist_head.log_psi(h, bf).item()
            all_psi.append(np.exp(log_psi))
            all_labels.append(buffer.labels[i])

    overall_auc = compute_auc(all_labels, all_psi)
    print(f"\n  Overall AUC: {overall_auc:.3f}")

    # AUC by boundary fraction bucket
    print(f"\n  AUC by boundary position:")
    bf_values = sorted(set(round(bf, 3) for bf in buffer.boundary_fracs))
    for bf_val in bf_values[:10]:  # first 10 positions
        mask = [abs(buffer.boundary_fracs[i] - bf_val) < 0.01 for i in range(len(buffer))]
        sub_labels = [l for l, m in zip(all_labels, mask) if m]
        sub_psi = [p for p, m in zip(all_psi, mask) if m]
        n_p = sum(1 for l in sub_labels if l > 0.5)
        n_n = sum(1 for l in sub_labels if l <= 0.5)
        if n_p > 0 and n_n > 0:
            auc = compute_auc(sub_labels, sub_psi)
            print(f"    bf={bf_val:.3f}: AUC={auc:.3f}  ({n_p}+ / {n_n}-)")
        else:
            print(f"    bf={bf_val:.3f}: n/a  ({n_p}+ / {n_n}-)")

    # ψ distribution for positives vs negatives
    pos_psi = [p for p, l in zip(all_psi, all_labels) if l > 0.5]
    neg_psi = [p for p, l in zip(all_psi, all_labels) if l <= 0.5]
    print(f"\n  ψ statistics:")
    print(f"    Positives (n={len(pos_psi)}): mean={np.mean(pos_psi):.4f}  median={np.median(pos_psi):.4f}  std={np.std(pos_psi):.4f}")
    print(f"    Negatives (n={len(neg_psi)}): mean={np.mean(neg_psi):.4f}  median={np.median(neg_psi):.4f}  std={np.std(neg_psi):.4f}")
    print(f"    Separation: pos_mean/neg_mean = {np.mean(pos_psi)/max(np.mean(neg_psi), 1e-8):.1f}x")

    # ================================================================
    # Test A: Prefix inspection on held-out instances
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Test A: Prefix inspection ({len(inspect_instances)} instances)")
    print(f"{'='*60}")

    for inst_idx, instance in enumerate(inspect_instances):
        # Show the goal
        goal_text = instance.problem_text
        # Extract just the goal section
        goal_start = goal_text.find("(:goal")
        goal_section = goal_text[goal_start:goal_start+200] if goal_start >= 0 else "?"

        print(f"\n{'─'*70}")
        print(f"Instance {inst_idx} — Goal: {goal_section[:120]}...")
        print(f"{'─'*70}")

        np.random.seed(42 + inst_idx)
        torch.manual_seed(42 + inst_idx)

        particle_data = await inspect_instance(
            instance, llm, grammar, hse, twist_head, evaluator,
            n_particles=args.n_particles, max_tokens=args.max_tokens,
            block_size=args.block_size, num_blocks_est=num_blocks_est,
        )

        # Show a few particles: 1-2 that succeeded, 1-2 that failed
        succeeded = [p for p in particle_data if p["final_success"]]
        failed = [p for p in particle_data if not p["final_success"]]

        print(f"  {len(succeeded)} succeeded, {len(failed)} failed out of {len(particle_data)} particles")

        def show_particle(p, label):
            print(f"\n  [{label}] Particle {p['particle_id']} (w={p['weight']:.3f}, {p['n_blocks']} blocks)")
            for b in p["boundaries"]:
                prefix_marker = "+" if b["prefix_ok"] else "X"
                block_display = b["block_text"].replace("\n", " ")[:50]
                print(f"    k={b['k']:2d}  ψ={b['psi']:.4f}  log(ψ)={b['log_psi']:7.2f}  "
                      f"[{prefix_marker}]  \"{block_display}\"")
            outcome = "PASS" if p["final_success"] else "FAIL"
            final_display = p["full_text"].replace("\n", " ")[:80]
            print(f"    => {outcome}: \"{final_display}...\"")

        for p in succeeded[:2]:
            show_particle(p, "SUCCESS")
        for p in failed[:2]:
            show_particle(p, "FAIL")

        # Summary: mean ψ at each boundary for succeeded vs failed
        if succeeded and failed:
            max_k = max(len(p["boundaries"]) for p in particle_data)
            print(f"\n  Mean ψ by boundary position:")
            print(f"  {'k':>4}  {'ψ(succeed)':>12}  {'ψ(fail)':>12}  {'ratio':>8}")
            for k in range(min(max_k, 15)):
                s_psi = [p["boundaries"][k]["psi"] for p in succeeded if k < len(p["boundaries"])]
                f_psi = [p["boundaries"][k]["psi"] for p in failed if k < len(p["boundaries"])]
                if s_psi and f_psi:
                    ratio = np.mean(s_psi) / max(np.mean(f_psi), 1e-8)
                    print(f"  {k:>4}  {np.mean(s_psi):>12.4f}  {np.mean(f_psi):>12.4f}  {ratio:>7.1f}x")

    # Save
    output_path = Path(__file__).parent / "inspect_results.json"
    output = {
        "model": args.model,
        "auc": overall_auc,
        "train_twist_acc": train_result["accuracy"],
        "train_twist_loss": train_result["loss"],
        "buffer_size": len(buffer),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train-instances", type=int, default=5)
    parser.add_argument("--n-inspect-instances", type=int, default=3)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-train-rounds", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
