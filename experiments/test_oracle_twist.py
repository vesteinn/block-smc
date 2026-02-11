"""Integration test: Block SMC with oracle twist on Molecular Synthesis.

Validates the full pipeline wiring:
    token sampler → MultiTokenUnitSampler → SMC with OracleTwistCritic

Uses the oracle twist (Φ_exp as ψ) so we don't need a trained twist head.
Compares Block SMC against the token-level baseline SMC.

Usage:
    conda run -n blocksmc python experiments/test_oracle_twist.py [--n-particles 5] [--max-tokens 40]
"""

import asyncio
import argparse
import time

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.eval.domains.molecular_synthesis import PartialSMILES

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary, SMILESBoundary


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


async def run_test(n_particles: int, max_tokens: int, model_name: str):
    print(f"Loading LLM: {model_name}")
    llm = PromptedLLM.from_name(model_name, backend="hf")
    llm.prompt_ids = make_prompt(llm.model.tokenizer)
    print(f"Prompt length: {len(llm.prompt_ids)} tokens")

    # Load SMILES grammar
    with open(GRAMMAR_PATH) as f:
        grammar_string = f.read()
    grammar = BoolCFG.from_lark(grammar_string)
    expensive = PartialSMILES()

    # Add newline EOS (same as baseline)
    eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"\n" in t]
    llm = llm.spawn_new_eos(eos_tokens)
    token_sampler = eager_token_sampler(llm, grammar)

    results = {}

    # --- Test 1: Baseline token-level SMC ---
    print(f"\n{'='*60}")
    print(f"Test 1: Baseline (token-level SMC, N={n_particles})")
    print(f"{'='*60}")
    coerced = expensive.coerce(llm, f=b"".join)
    baseline_smc = SMC(token_sampler, critic=coerced)
    t0 = time.time()
    seq_baseline = await baseline_smc(
        n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens
    )
    dt = time.time() - t0
    print(f"  Time: {dt:.1f}s")
    print(f"  log_ml: {seq_baseline.log_ml:.4f}")
    print(f"  ESS: {seq_baseline.ess:.1f}")
    try:
        posterior = seq_baseline.decoded_posterior
        print(f"  Unique sequences: {len(posterior)}")
        for s, p in list(posterior.items())[:5]:
            print(f"    {p:.3f}  {s}")
    except Exception as e:
        print(f"  decoded_posterior error: {e}")
    results["baseline"] = (seq_baseline.log_ml, seq_baseline.ess, dt)

    # --- Test 2: Block SMC with oracle twist + fixed-interval ---
    print(f"\n{'='*60}")
    print(f"Test 2: Block SMC oracle twist + FixedInterval(5)")
    print(f"{'='*60}")
    smc_fixed = make_block_smc(
        token_sampler=token_sampler,
        boundary_predicate=FixedIntervalBoundary(5),
        expensive_potential=expensive,
        llm=llm,
        oracle_twist=True,
    )
    t0 = time.time()
    seq_fixed = await run_block_smc(
        smc_fixed, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens
    )
    dt = time.time() - t0
    posterior_fixed = decode_block_sequences(seq_fixed)
    print(f"  Time: {dt:.1f}s")
    print(f"  log_ml: {seq_fixed.log_ml:.4f}")
    print(f"  ESS: {seq_fixed.ess:.1f}")
    print(f"  Unique sequences: {len(posterior_fixed)}")
    for s, p in list(posterior_fixed.items())[:5]:
        print(f"    {p:.3f}  {s}")
    results["block_fixed5"] = (seq_fixed.log_ml, seq_fixed.ess, dt)

    # --- Test 3: Block SMC with oracle twist + SMILES boundaries ---
    print(f"\n{'='*60}")
    print(f"Test 3: Block SMC oracle twist + SMILESBoundary")
    print(f"{'='*60}")
    smc_smiles = make_block_smc(
        token_sampler=token_sampler,
        boundary_predicate=SMILESBoundary(min_tokens=2),
        expensive_potential=expensive,
        llm=llm,
        oracle_twist=True,
    )
    t0 = time.time()
    seq_smiles = await run_block_smc(
        smc_smiles, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens
    )
    dt = time.time() - t0
    posterior_smiles = decode_block_sequences(seq_smiles)
    print(f"  Time: {dt:.1f}s")
    print(f"  log_ml: {seq_smiles.log_ml:.4f}")
    print(f"  ESS: {seq_smiles.ess:.1f}")
    print(f"  Unique sequences: {len(posterior_smiles)}")
    for s, p in list(posterior_smiles.items())[:5]:
        print(f"    {p:.3f}  {s}")
    results["block_smiles"] = (seq_smiles.log_ml, seq_smiles.ess, dt)

    # --- Test 4: Block SMC with NO critic (proposal only) ---
    print(f"\n{'='*60}")
    print(f"Test 4: Block SMC no critic (proposal only)")
    print(f"{'='*60}")
    smc_nocrit = make_block_smc(
        token_sampler=token_sampler,
        boundary_predicate=FixedIntervalBoundary(5),
        expensive_potential=expensive,
        llm=llm,
        oracle_twist=False,  # TwistedBlockCritic with no twist head = Φ_exp only
    )
    t0 = time.time()
    seq_nocrit = await run_block_smc(
        smc_nocrit, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens
    )
    dt = time.time() - t0
    posterior_nocrit = decode_block_sequences(seq_nocrit)
    print(f"  Time: {dt:.1f}s")
    print(f"  log_ml: {seq_nocrit.log_ml:.4f}")
    print(f"  ESS: {seq_nocrit.ess:.1f}")
    print(f"  Unique sequences: {len(posterior_nocrit)}")
    for s, p in list(posterior_nocrit.items())[:5]:
        print(f"    {p:.3f}  {s}")
    results["block_phi_only"] = (seq_nocrit.log_ml, seq_nocrit.ess, dt)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'log_ml':>10} {'ESS':>6} {'Time':>6}")
    print("-" * 55)
    for name, (lml, ess, dt) in results.items():
        print(f"{name:<30} {lml:>10.4f} {ess:>6.1f} {dt:>5.1f}s")

    await llm.cleanup()
    print("\nAll tests completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle twist integration test")
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(run_test(args.n_particles, args.max_tokens, args.model))
