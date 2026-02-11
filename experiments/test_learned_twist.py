"""End-to-end test: Train a twist head from Block SMC trajectories.

Pipeline:
    1. Run Block SMC with Φ_exp-only critic → collect trajectories
    2. Extract (h, w, y, k/K) training data from trajectories
    3. Train TwistHead via weighted BCE
    4. Run Block SMC with learned TwistedBlockCritic
    5. Compare: no-twist vs learned-twist vs oracle-twist

Usage:
    conda run -n blocksmc python experiments/test_learned_twist.py [--n-particles 10] [--max-tokens 40]
"""

import asyncio
import argparse
import time
import torch

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler
from genlm.eval.domains.molecular_synthesis import PartialSMILES

from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary import FixedIntervalBoundary
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data
from block_smc.hidden_states import HiddenStateExtractor


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


async def run_test(n_particles: int, max_tokens: int, n_collection_rounds: int, model_name: str):
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

    # Coerce expensive potential to LLM token type
    coerced = expensive.coerce(llm, f=b"".join)

    # Create hidden state extractor
    hse = HiddenStateExtractor(llm)
    d_model = hse.hidden_dim
    print(f"Hidden dim: {d_model}")

    boundary = FixedIntervalBoundary(5)
    results = {}

    # =========================================================================
    # Step 1: Run Block SMC with Φ_exp only (no twist) — collect trajectories
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Step 1: Collecting training data ({n_collection_rounds} rounds, N={n_particles})")
    print(f"{'='*60}")

    buffer = TwistTrainingBuffer()
    total_stats = {"n_examples": 0, "n_positive": 0, "n_particles": 0}

    for round_idx in range(n_collection_rounds):
        smc_collect = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=FixedIntervalBoundary(5),
            expensive_potential=expensive,
            llm=llm,
            oracle_twist=False,
        )
        t0 = time.time()
        seq = await run_block_smc(
            smc_collect, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens
        )
        dt = time.time() - t0

        # Collect training data from this sweep
        stats = await collect_twist_training_data(
            sequences=seq,
            hidden_state_extractor=hse,
            expensive_potential=coerced,
            buffer=buffer,
        )

        total_stats["n_examples"] += stats["n_examples"]
        total_stats["n_positive"] += stats["n_positive"]
        total_stats["n_particles"] += stats["n_particles"]

        posterior = decode_block_sequences(seq)
        print(f"  Round {round_idx+1}: log_ml={seq.log_ml:.4f}, ESS={seq.ess:.1f}, "
              f"examples={stats['n_examples']}, positive={stats['n_positive']}, "
              f"time={dt:.1f}s")
        if posterior:
            top = list(posterior.items())[:3]
            for s, p in top:
                print(f"    {p:.3f}  {s}")

        buffer.advance_age()

    print(f"\n  Total buffer: {len(buffer)} examples, "
          f"{total_stats['n_positive']} positive, "
          f"{total_stats['n_particles']} particles")

    if len(buffer) == 0:
        print("  ERROR: No training data collected! All particles may have died.")
        print("  Try increasing --n-particles or --max-tokens.")
        await llm.cleanup()
        return

    # Save no-twist result from last round for comparison
    results["no_twist"] = (seq.log_ml, seq.ess, dt)

    # =========================================================================
    # Step 2: Train the TwistHead
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Step 2: Training TwistHead (d_model={d_model})")
    print(f"{'='*60}")

    device = hse.device
    twist_head = TwistHead(d_model=d_model, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=1e-3)

    train_result = train_twist_step(
        twist_head=twist_head,
        buffer=buffer,
        optimizer=optimizer,
        device=device,
        n_epochs=10,
        batch_size=64,
        discount=0.9,
    )
    print(f"  Training loss: {train_result['loss']:.4f}")
    print(f"  Training accuracy: {train_result['accuracy']:.4f}")

    # =========================================================================
    # Step 3: Run Block SMC with learned twist
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Step 3: Block SMC with learned twist (N={n_particles})")
    print(f"{'='*60}")

    smc_learned = make_block_smc(
        token_sampler=token_sampler,
        boundary_predicate=FixedIntervalBoundary(5),
        expensive_potential=expensive,
        llm=llm,
        twist_head=twist_head,
        hidden_state_extractor=hse,
        num_blocks=8,  # Approximate expected blocks for max_tokens/5
    )
    t0 = time.time()
    seq_learned = await run_block_smc(
        smc_learned, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens
    )
    dt_learned = time.time() - t0
    posterior_learned = decode_block_sequences(seq_learned)
    print(f"  Time: {dt_learned:.1f}s")
    print(f"  log_ml: {seq_learned.log_ml:.4f}")
    print(f"  ESS: {seq_learned.ess:.1f}")
    print(f"  Unique sequences: {len(posterior_learned)}")
    for s, p in list(posterior_learned.items())[:5]:
        print(f"    {p:.3f}  {s}")
    results["learned_twist"] = (seq_learned.log_ml, seq_learned.ess, dt_learned)

    # =========================================================================
    # Step 4: Oracle twist baseline for comparison
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Step 4: Block SMC with oracle twist (N={n_particles})")
    print(f"{'='*60}")

    smc_oracle = make_block_smc(
        token_sampler=token_sampler,
        boundary_predicate=FixedIntervalBoundary(5),
        expensive_potential=expensive,
        llm=llm,
        oracle_twist=True,
    )
    t0 = time.time()
    seq_oracle = await run_block_smc(
        smc_oracle, n_particles=n_particles, ess_threshold=0.9, max_tokens=max_tokens
    )
    dt_oracle = time.time() - t0
    posterior_oracle = decode_block_sequences(seq_oracle)
    print(f"  Time: {dt_oracle:.1f}s")
    print(f"  log_ml: {seq_oracle.log_ml:.4f}")
    print(f"  ESS: {seq_oracle.ess:.1f}")
    print(f"  Unique sequences: {len(posterior_oracle)}")
    for s, p in list(posterior_oracle.items())[:5]:
        print(f"    {p:.3f}  {s}")
    results["oracle_twist"] = (seq_oracle.log_ml, seq_oracle.ess, dt_oracle)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training data: {len(buffer)} examples, {total_stats['n_positive']} positive")
    print(f"Twist accuracy: {train_result['accuracy']:.4f}")
    print()
    print(f"{'Method':<25} {'log_ml':>10} {'ESS':>6} {'Time':>6}")
    print("-" * 50)
    for name, (lml, ess, dt) in results.items():
        print(f"{name:<25} {lml:>10.4f} {ess:>6.1f} {dt:>5.1f}s")

    await llm.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learned twist end-to-end test")
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--n-collection-rounds", type=int, default=3,
                        help="Number of Block SMC sweeps for collecting training data")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(run_test(args.n_particles, args.max_tokens, args.n_collection_rounds, args.model))
