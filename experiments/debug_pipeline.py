"""Debug script to trace where Block SMC particles die."""

import asyncio
import numpy as np
from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler, SMC
from genlm.control.sampler.unit import MultiTokenUnitSampler, flatten_units
from genlm.control.constant import EOS
from genlm.eval.domains.molecular_synthesis import PartialSMILES

from block_smc.boundary import FixedIntervalBoundary
from block_smc.critic import OracleTwistCritic, _flatten_context

EXAMPLE_SMILES = ["CCO", "CC(C)C", "C1=CC=CC=C1", "CC(=O)O", "CC(C)O"]


def make_prompt(tokenizer):
    examples = "\n".join(EXAMPLE_SMILES)
    text = f"Generate a valid SMILES molecular string.\n\nExamples:\n{examples}\n\nNew molecule:"
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


async def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading {model_name}...")
    llm = PromptedLLM.from_name(model_name, backend="hf")
    llm.prompt_ids = make_prompt(llm.model.tokenizer)

    grammar_path = "/home/vesteinn/Projects/BLOCK_SMC/code/control-iclr-2025/experiments/molecular_synthesis/smiles.lark"
    with open(grammar_path) as f:
        grammar_string = f.read()
    grammar = BoolCFG.from_lark(grammar_string)
    expensive = PartialSMILES()

    # Add newline EOS (same as baseline)
    eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"\n" in t]
    print(f"EOS tokens: {len(eos_tokens)} tokens")
    llm = llm.spawn_new_eos(eos_tokens)
    token_sampler = eager_token_sampler(llm, grammar)

    # --- Step 1: Sample a few tokens (stop at EOS) ---
    print("\n--- Step 1: Token sampling (stop at EOS) ---")
    flat_ctx = []
    for i in range(20):
        tok, logw, logp = await token_sampler.sample(flat_ctx)
        print(f"  Token {i}: {tok!r}, logw={logw:.4f}")
        if tok is EOS:
            print("  -> EOS reached")
            break
        flat_ctx.append(tok)
    text = b"".join(t for t in flat_ctx if isinstance(t, bytes))
    print(f"  Generated: {text!r}")

    # --- Step 2: Baseline token-level SMC ---
    print("\n--- Step 2: Baseline token-level SMC (5 particles, 40 tokens) ---")
    coerced = expensive.coerce(llm, f=b"".join)
    baseline_smc = SMC(token_sampler, critic=coerced)
    result = await baseline_smc(n_particles=5, ess_threshold=0.9, max_tokens=40)
    print(f"  log_ml: {result.log_ml:.4f}")
    print(f"  ESS: {result.ess:.1f}")
    for seq, w in zip(result.contexts[:3], result.log_weights[:3]):
        flat = [t for t in seq if isinstance(t, bytes)]
        text = b"".join(flat)
        print(f"  logw={w:.4f} text={text!r}")

    # --- Step 3: Block SMC without critic ---
    print("\n--- Step 3: Block SMC without critic (MultiTokenUnitSampler only) ---")
    boundary = FixedIntervalBoundary(5)
    unit_sampler = MultiTokenUnitSampler(token_sampler, boundary, max_subunits_per_unit=50)
    smc_nc = SMC(unit_sampler, critic=None)
    result_nc = await smc_nc(n_particles=5, ess_threshold=0.5, max_tokens=40)
    print(f"  log_ml: {result_nc.log_ml:.4f}")
    print(f"  ESS: {result_nc.ess:.1f}")
    for i, (seq, w) in enumerate(zip(result_nc.contexts[:3], result_nc.log_weights[:3])):
        flat = _flatten_context(seq)
        text = b"".join(t for t in flat if isinstance(t, bytes))
        print(f"  [{i}] logw={w:.4f} units={len(seq)} text={text!r}")

    # --- Step 4: Test OracleTwistCritic scoring ---
    print("\n--- Step 4: OracleTwistCritic scoring ---")
    oracle_critic = OracleTwistCritic(coerced)
    print(f"  critic token_type: {oracle_critic.token_type}")
    print(f"  unit_sampler token_type: {unit_sampler.token_type}")
    # Score a sample nested context from step 3
    if result_nc.contexts:
        for i, seq in enumerate(result_nc.contexts[:2]):
            # Take first 2 units (before EOS)
            units_only = [u for u in seq if isinstance(u, list)]
            if units_only:
                score = await oracle_critic.score(units_only)
                flat = _flatten_context(units_only)
                text = b"".join(t for t in flat if isinstance(t, bytes))
                print(f"  [{i}] score={score:.4f} text={text!r}")

    # --- Step 5: Block SMC with oracle critic ---
    print("\n--- Step 5: Block SMC with OracleTwistCritic ---")
    unit_sampler2 = MultiTokenUnitSampler(token_sampler, FixedIntervalBoundary(5), max_subunits_per_unit=50)
    smc_oracle = SMC(unit_sampler2, critic=oracle_critic)
    result_oracle = await smc_oracle(n_particles=5, ess_threshold=0.9, max_tokens=40)
    print(f"  log_ml: {result_oracle.log_ml:.4f}")
    print(f"  ESS: {result_oracle.ess:.1f}")
    for i, (seq, w) in enumerate(zip(result_oracle.contexts[:3], result_oracle.log_weights[:3])):
        flat = _flatten_context(seq)
        text = b"".join(t for t in flat if isinstance(t, bytes))
        print(f"  [{i}] logw={w:.4f} text={text!r}")

    await llm.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
