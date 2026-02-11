"""Block size scaling experiment: how does twist value change with block size?

As blocks get larger, more tokens are generated between expensive potential
evaluations. The twist must compensate for this lost information. This experiment
measures the twist's value as a function of block size.

Methods per block size:
    1. Baseline: token-level SMC (paper's full-smc, independent of block size)
    2. Block SMC, no twist
    3. Block SMC, learned twist (frozen, trained once at block_size=5)

Usage:
    conda run -n blocksmc python experiments/block_size_scaling.py [--n-seeds 20]
"""

import asyncio
import argparse
import json
import time
import torch
import numpy as np

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.sampler.unit import MultiTokenUnitSampler, flatten_units
from genlm.control.sampler.token import DirectTokenSampler
from genlm.eval.domains.molecular_synthesis import PartialSMILES

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor


GRAMMAR_PATH = "/home/vesteinn/Projects/BLOCK_SMC/code/control-iclr-2025/experiments/molecular_synthesis/smiles.lark"


def make_prompt(tokenizer):
    examples = ["CCO", "CC(C)C", "C1=CC=CC=C1", "CC(=O)O", "CC(C)O"]
    text = (
        f"Generate a valid SMILES molecular string.\n\n"
        f"Examples:\n" + "\n".join(examples) + "\n\n"
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
            summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        else:
            summary[m] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    n_fail = sum(1 for r in results_by_seed if not np.isfinite(r.get("log_ml", float("nan"))))
    summary["n_fail"] = n_fail
    return summary


async def main(args):
    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")
    llm.prompt_ids = make_prompt(llm.model.tokenizer)

    with open(GRAMMAR_PATH) as f:
        grammar_string = f.read()

    use_grammar = not args.no_grammar
    if use_grammar:
        grammar = BoolCFG.from_lark(grammar_string)
        print("Using grammar enforcement (BoolCFG + eager_token_sampler)")
    else:
        print("NO grammar enforcement (DirectTokenSampler)")

    expensive = PartialSMILES()

    eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"\n" in t]
    llm = llm.spawn_new_eos(eos_tokens)

    if use_grammar:
        token_sampler = eager_token_sampler(llm, grammar)
    else:
        token_sampler = DirectTokenSampler(llm)

    coerced = expensive.coerce(llm, f=b"".join)
    hse = HiddenStateExtractor(llm)
    N = args.n_particles
    T = args.max_tokens
    block_sizes = [int(b) for b in args.block_sizes.split(",")]

    # =========================================================================
    # PHASE 1: TRAIN TWIST (once, at block_size=5)
    # =========================================================================
    train_block_size = 5
    n_guided = args.n_train_rounds // 2
    n_explore = args.n_train_rounds - n_guided
    print(f"\n{'='*70}")
    print(f"PHASE 1: TRAINING (block_size={train_block_size}, {n_guided} guided + {n_explore} explore, N={N})")
    print(f"{'='*70}")

    torch.manual_seed(0)
    np.random.seed(0)
    buffer = TwistTrainingBuffer()
    train_t0 = time.time()

    # Guided collection
    for r in range(n_guided):
        smc_collect = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(train_block_size),
            expensive_potential=expensive, llm=llm, oracle_twist=False,
        )
        seq = await run_block_smc(smc_collect, n_particles=N, ess_threshold=0.9, max_tokens=T)
        stats = await collect_twist_training_data(seq, hse, coerced, buffer)
        print(f"  Guided {r+1}: ex={stats['n_examples']} pos={stats['n_positive']}")
        buffer.advance_age()

    # Exploration collection (no critic)
    for r in range(n_explore):
        unit_sampler = MultiTokenUnitSampler(
            token_sampler, FixedIntervalBoundary(train_block_size), max_subunits_per_unit=100
        )
        explore_smc = SMC(unit_sampler)
        seq = await explore_smc(n_particles=N, ess_threshold=0.5, max_tokens=T)
        stats = await collect_twist_training_data(seq, hse, coerced, buffer)
        print(f"  Explore {r+1}: ex={stats['n_examples']} pos={stats['n_positive']} neg={stats['n_examples'] - stats['n_positive']}")
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
    train_time = time.time() - train_t0
    print(f"  Twist: loss={train_result['loss']:.4f} acc={train_result['accuracy']:.4f} time={train_time:.1f}s")

    twist_head.eval()
    for p in twist_head.parameters():
        p.requires_grad_(False)

    # =========================================================================
    # PHASE 2: EVALUATE across block sizes
    # =========================================================================
    seeds = list(range(100, 100 + args.n_seeds))
    all_results = {}

    # --- Token-level baseline (run once, same for all block sizes) ---
    print(f"\n{'='*70}")
    print(f"PHASE 2: EVALUATION ({args.n_seeds} seeds, N={N}, block_sizes={block_sizes})")
    print(f"{'='*70}")

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
    all_results["baseline_token_smc"] = results
    s = summarize(results)
    print(f"  log_Z={s['log_ml']['mean']:.2f}±{s['log_ml']['std']:.2f}  "
          f"valid={s['validity']['mean']:.2f}±{s['validity']['std']:.2f}  "
          f"uniq={s['n_unique']['mean']:.1f}  fail={s['n_fail']}")

    # --- Per block size ---
    for bs in block_sizes:
        print(f"\n--- Block size {bs} ---")

        # No twist
        results_no = []
        for seed in seeds:
            np.random.seed(seed)
            smc = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=FixedIntervalBoundary(bs),
                expensive_potential=expensive, llm=llm, oracle_twist=False,
            )
            t0 = time.time()
            seq = await run_block_smc(smc, n_particles=N, ess_threshold=0.9, max_tokens=T)
            dt = time.time() - t0
            posterior = decode_block_sequences(seq)
            r = {"log_ml": float(seq.log_ml), "ess": float(seq.ess),
                 "n_unique": len(posterior), "validity": compute_validity(posterior), "wall_time": dt}
            results_no.append(r)
        all_results[f"block_{bs}_no_twist"] = results_no
        s = summarize(results_no)
        print(f"  no_twist:      log_Z={s['log_ml']['mean']:.2f}±{s['log_ml']['std']:.2f}  "
              f"valid={s['validity']['mean']:.2f}±{s['validity']['std']:.2f}  "
              f"uniq={s['n_unique']['mean']:.1f}  fail={s['n_fail']}")

        # Learned twist
        num_blocks_est = max(T // bs, 1)
        results_tw = []
        for seed in seeds:
            np.random.seed(seed)
            smc = make_block_smc(
                token_sampler=token_sampler,
                boundary_predicate=FixedIntervalBoundary(bs),
                expensive_potential=expensive, llm=llm,
                twist_head=twist_head, hidden_state_extractor=hse,
                num_blocks=num_blocks_est,
            )
            t0 = time.time()
            seq = await run_block_smc(smc, n_particles=N, ess_threshold=0.9, max_tokens=T)
            dt = time.time() - t0
            posterior = decode_block_sequences(seq)
            r = {"log_ml": float(seq.log_ml), "ess": float(seq.ess),
                 "n_unique": len(posterior), "validity": compute_validity(posterior), "wall_time": dt}
            results_tw.append(r)
        all_results[f"block_{bs}_twist"] = results_tw
        s = summarize(results_tw)
        print(f"  learned_twist: log_Z={s['log_ml']['mean']:.2f}±{s['log_ml']['std']:.2f}  "
              f"valid={s['validity']['mean']:.2f}±{s['validity']['std']:.2f}  "
              f"uniq={s['n_unique']['mean']:.1f}  fail={s['n_fail']}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    grammar_label = "with_grammar" if use_grammar else "no_grammar"
    print(f"\n{'='*70}")
    print(f"SUMMARY ({grammar_label}, N={N}, T={T}, {args.n_seeds} seeds)")
    print(f"Training: {len(buffer)} examples ({n_pos}+/{n_neg}-), {train_time:.1f}s")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'log Ẑ':>12} {'Valid':>10} {'Unique':>8} {'Fail':>6} {'Time':>8}")
    print("-" * 74)

    for name in ["baseline_token_smc"] + [f"block_{bs}_{t}" for bs in block_sizes for t in ["no_twist", "twist"]]:
        if name not in all_results:
            continue
        s = summarize(all_results[name])
        print(f"{name:<30} "
              f"{s['log_ml']['mean']:>5.1f}±{s['log_ml']['std']:>4.1f} "
              f"{s['validity']['mean']:>5.2f}±{s['validity']['std']:>3.2f} "
              f"{s['n_unique']['mean']:>5.1f} "
              f"{s['n_fail']:>5d} "
              f"{s['wall_time']['mean']:>5.1f}s")

    # Save
    output = {
        "config": {"n_particles": N, "max_tokens": T, "n_seeds": args.n_seeds,
                    "model": args.model, "block_sizes": block_sizes,
                    "grammar": use_grammar, "n_train_rounds": args.n_train_rounds},
        "training": {"buffer_size": len(buffer), "twist_loss": train_result["loss"],
                      "twist_accuracy": train_result["accuracy"], "train_time": train_time,
                      "n_positive": n_pos, "n_negative": n_neg},
        "results": {name: rs for name, rs in all_results.items()},
        "summary": {name: summarize(rs) for name, rs in all_results.items()},
    }
    output_path = f"experiments/block_size_scaling_{grammar_label}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--n-train-rounds", type=int, default=10)
    parser.add_argument("--block-sizes", type=str, default="3,5,10,20",
                        help="Comma-separated block sizes to test")
    parser.add_argument("--no-grammar", action="store_true",
                        help="Disable grammar enforcement (harder task)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
