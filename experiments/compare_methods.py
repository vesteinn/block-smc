"""Systematic comparison of all Block SMC methods.

Runs each method multiple times with different seeds and reports
averaged metrics with standard deviations.

Methods:
    1. Baseline: token-level SMC (paper's full-smc)
    2. Block SMC, Φ_exp only (no twist)
    3. Block SMC, oracle twist
    4. Block SMC, learned twist (3 collection rounds)
    5. Block SMC + EM (3 EM iterations)

Metrics per run:
    - log_ml: log marginal likelihood estimate
    - ESS: effective sample size
    - n_unique: number of unique sequences in posterior
    - validity: fraction of weighted posterior that decodes to valid SMILES
    - wall_time: seconds

Usage:
    conda run -n blocksmc python experiments/compare_methods.py [--n-seeds 5] [--n-particles 10]
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
from block_smc.em import run_em, EMConfig
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
    """Check if a string is a valid SMILES molecule."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles_str)
        return mol is not None
    except ImportError:
        # Fallback: just check it's non-empty ASCII
        return len(smiles_str) > 0 and smiles_str.isascii()


def compute_validity(posterior: dict) -> float:
    """Compute weighted validity rate from posterior."""
    if not posterior:
        return 0.0
    total_weight = sum(posterior.values())
    valid_weight = sum(w for s, w in posterior.items() if check_smiles_validity(s))
    return valid_weight / total_weight if total_weight > 0 else 0.0


async def run_baseline(token_sampler, coerced_potential, n_particles, max_tokens, seed):
    """Token-level SMC baseline (paper's full-smc)."""
    np.random.seed(seed)
    baseline_smc = SMC(token_sampler, critic=coerced_potential)
    t0 = time.time()
    seq = await baseline_smc(n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
    dt = time.time() - t0

    try:
        posterior = dict(seq.decoded_posterior)
    except Exception:
        posterior = {}

    return {
        "log_ml": float(seq.log_ml),
        "ess": float(seq.ess),
        "n_unique": len(posterior),
        "validity": compute_validity(posterior),
        "wall_time": dt,
        "top_sequences": list(posterior.items())[:5],
    }


async def run_block_smc_method(
    token_sampler, expensive, llm, boundary, n_particles, max_tokens,
    oracle_twist=False, twist_head=None, hse=None, num_blocks=None, seed=42
):
    """Run a single Block SMC configuration."""
    np.random.seed(seed)
    smc = make_block_smc(
        token_sampler=token_sampler,
        boundary_predicate=boundary,
        expensive_potential=expensive,
        llm=llm,
        oracle_twist=oracle_twist,
        twist_head=twist_head,
        hidden_state_extractor=hse,
        num_blocks=num_blocks,
    )
    t0 = time.time()
    seq = await run_block_smc(smc, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
    dt = time.time() - t0
    posterior = decode_block_sequences(seq)

    return {
        "log_ml": float(seq.log_ml),
        "ess": float(seq.ess),
        "n_unique": len(posterior),
        "validity": compute_validity(posterior),
        "wall_time": dt,
        "top_sequences": list(posterior.items())[:5],
    }, seq


async def train_twist_and_run(
    token_sampler, expensive, llm, coerced, hse,
    n_particles, max_tokens, n_collection_rounds, seed
):
    """Collect data, train twist, run with learned twist."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    d_model = hse.hidden_dim
    device = hse.device
    buffer = TwistTrainingBuffer()

    # Collect training data
    t0 = time.time()
    for _ in range(n_collection_rounds):
        smc_collect = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(5),
            expensive_potential=expensive,
            llm=llm,
            oracle_twist=False,
        )
        seq = await run_block_smc(smc_collect, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens)
        await collect_twist_training_data(seq, hse, coerced, buffer)
        buffer.advance_age()

    # Train twist
    twist_head = TwistHead(d_model=d_model, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=1e-3)
    train_result = train_twist_step(
        twist_head, buffer, optimizer, device, n_epochs=10, batch_size=64
    )
    collection_time = time.time() - t0

    # Run with learned twist
    result, _ = await run_block_smc_method(
        token_sampler, expensive, llm, FixedIntervalBoundary(5),
        n_particles, max_tokens,
        twist_head=twist_head, hse=hse, num_blocks=8, seed=seed
    )
    result["wall_time"] += collection_time  # Include training cost
    result["twist_accuracy"] = train_result["accuracy"]
    result["buffer_size"] = len(buffer)
    return result


def summarize(results_by_seed: list[dict]) -> dict:
    """Compute mean ± std for each metric."""
    metrics = ["log_ml", "ess", "n_unique", "validity", "wall_time"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results_by_seed if not np.isnan(r.get(m, float("nan")))]
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
    seeds = list(range(args.n_seeds))
    N = args.n_particles
    T = args.max_tokens

    all_results = {}

    # =========================================================================
    # Method 1: Baseline token-level SMC
    # =========================================================================
    print(f"\n--- Method 1: Baseline (token-level SMC, N={N}) ---")
    results = []
    for seed in seeds:
        r = await run_baseline(token_sampler, coerced, N, T, seed)
        results.append(r)
        print(f"  seed={seed}: log_ml={r['log_ml']:.4f} ESS={r['ess']:.1f} unique={r['n_unique']} valid={r['validity']:.2f} t={r['wall_time']:.1f}s")
    all_results["baseline_token_smc"] = results

    # =========================================================================
    # Method 1b: Vanilla MultiTokenUnitSampler + coerced critic (genlm-control built-in)
    # =========================================================================
    print(f"\n--- Method 1b: MultiTokenUnitSampler + coerced critic, FixedInterval(5) ---")
    # This is the "natural" way to do block-level SMC using genlm-control directly:
    # coerce the expensive potential with flatten_units, then use it as critic.
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
        r = {
            "log_ml": float(seq.log_ml),
            "ess": float(seq.ess),
            "n_unique": len(posterior),
            "validity": compute_validity(posterior),
            "wall_time": dt,
        }
        results.append(r)
        print(f"  seed={seed}: log_ml={r['log_ml']:.4f} ESS={r['ess']:.1f} unique={r['n_unique']} valid={r['validity']:.2f} t={r['wall_time']:.1f}s")
    all_results["block_vanilla_coerced"] = results

    # =========================================================================
    # Method 2: Block SMC, Φ_exp only (no twist), fixed interval
    # =========================================================================
    print(f"\n--- Method 2: Block SMC, no twist, FixedInterval(5) ---")
    results = []
    for seed in seeds:
        r, _ = await run_block_smc_method(
            token_sampler, expensive, llm, FixedIntervalBoundary(5),
            N, T, seed=seed
        )
        results.append(r)
        print(f"  seed={seed}: log_ml={r['log_ml']:.4f} ESS={r['ess']:.1f} unique={r['n_unique']} valid={r['validity']:.2f} t={r['wall_time']:.1f}s")
    all_results["block_no_twist"] = results

    # =========================================================================
    # Method 3: Block SMC, oracle twist, fixed interval
    # =========================================================================
    print(f"\n--- Method 3: Block SMC, oracle twist, FixedInterval(5) ---")
    results = []
    for seed in seeds:
        r, _ = await run_block_smc_method(
            token_sampler, expensive, llm, FixedIntervalBoundary(5),
            N, T, oracle_twist=True, seed=seed
        )
        results.append(r)
        print(f"  seed={seed}: log_ml={r['log_ml']:.4f} ESS={r['ess']:.1f} unique={r['n_unique']} valid={r['validity']:.2f} t={r['wall_time']:.1f}s")
    all_results["block_oracle_twist"] = results

    # =========================================================================
    # Method 4: Block SMC, SMILES boundaries, oracle twist
    # =========================================================================
    print(f"\n--- Method 4: Block SMC, oracle twist, SMILESBoundary ---")
    results = []
    for seed in seeds:
        r, _ = await run_block_smc_method(
            token_sampler, expensive, llm, SMILESBoundary(min_tokens=2),
            N, T, oracle_twist=True, seed=seed
        )
        results.append(r)
        print(f"  seed={seed}: log_ml={r['log_ml']:.4f} ESS={r['ess']:.1f} unique={r['n_unique']} valid={r['validity']:.2f} t={r['wall_time']:.1f}s")
    all_results["block_oracle_smiles"] = results

    # =========================================================================
    # Method 5: Block SMC, learned twist
    # =========================================================================
    print(f"\n--- Method 5: Block SMC, learned twist (3 collection rounds) ---")
    results = []
    for seed in seeds:
        r = await train_twist_and_run(
            token_sampler, expensive, llm, coerced, hse,
            N, T, n_collection_rounds=3, seed=seed
        )
        results.append(r)
        print(f"  seed={seed}: log_ml={r['log_ml']:.4f} ESS={r['ess']:.1f} unique={r['n_unique']} valid={r['validity']:.2f} t={r['wall_time']:.1f}s buf={r['buffer_size']}")
    all_results["block_learned_twist"] = results

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (mean ± std across seeds)")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'log_ml':>14} {'ESS':>10} {'Unique':>10} {'Valid':>10} {'Time(s)':>10}")
    print("-" * 84)

    for method_name, results in all_results.items():
        s = summarize(results)
        lml = s['log_ml']
        ess = s['ess']
        uniq = s['n_unique']
        val = s['validity']
        t = s['wall_time']
        print(f"{method_name:<30} "
              f"{lml['mean']:>6.2f}±{lml['std']:>4.2f} "
              f"{ess['mean']:>5.1f}±{ess['std']:>3.1f} "
              f"{uniq['mean']:>5.1f}±{uniq['std']:>3.1f} "
              f"{val['mean']:>5.2f}±{val['std']:>3.2f} "
              f"{t['mean']:>5.1f}±{t['std']:>3.1f}")

    # Save raw results
    output = {
        "config": {
            "n_particles": N,
            "max_tokens": T,
            "n_seeds": args.n_seeds,
            "model": args.model,
        },
        "results": {},
        "summary": {},
    }
    for method_name, results in all_results.items():
        output["results"][method_name] = [
            {k: v for k, v in r.items() if k != "top_sequences"} for r in results
        ]
        output["summary"][method_name] = summarize(results)

    output_path = "experiments/comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Systematic method comparison")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
