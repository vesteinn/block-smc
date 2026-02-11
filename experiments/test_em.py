"""Integration test: EM boundary learning on Molecular Synthesis.

Runs a small EM loop (3 iterations) to validate the full pipeline:
    1. Block SMC with initial fixed boundaries
    2. Twist training from trajectories
    3. Local search to refine boundaries
    4. Repeat

Usage:
    conda run -n blocksmc python experiments/test_em.py [--n-em-iters 3] [--n-particles 10]
"""

import asyncio
import argparse
import json

from genlm.control import PromptedLLM, BoolCFG, eager_token_sampler
from genlm.eval.domains.molecular_synthesis import PartialSMILES

from block_smc.em import run_em, EMConfig


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


async def main(args):
    print(f"Loading LLM: {args.model}")
    llm = PromptedLLM.from_name(args.model, backend="hf")
    llm.prompt_ids = make_prompt(llm.model.tokenizer)
    print(f"Prompt length: {len(llm.prompt_ids)} tokens")

    with open(GRAMMAR_PATH) as f:
        grammar_string = f.read()
    grammar = BoolCFG.from_lark(grammar_string)
    expensive = PartialSMILES()

    eos_tokens = [t for t in llm.vocab if isinstance(t, bytes) and b"\n" in t]
    llm = llm.spawn_new_eos(eos_tokens)
    token_sampler = eager_token_sampler(llm, grammar)

    config = EMConfig(
        n_particles=args.n_particles,
        ess_threshold=0.9,
        max_tokens=args.max_tokens,
        min_block=2,
        max_block=args.max_tokens,
        initial_interval=5,
        n_em_iters=args.n_em_iters,
        n_search_steps=args.n_search_steps,
        search_n_runs=1,
        search_n_particles=args.search_particles,
        twist_hidden_dim=256,
        twist_lr=1e-3,
        twist_epochs=10,
        twist_batch_size=64,
        buffer_discount=0.9,
        seed=42,
        verbose=True,
    )

    history, twist_head, final_boundaries = await run_em(
        token_sampler=token_sampler,
        expensive_potential=expensive,
        llm=llm,
        config=config,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("EM HISTORY SUMMARY")
    print(f"{'='*60}")
    for i in range(len(history.log_ml)):
        print(f"Iter {i+1}: log_ml={history.log_ml[i]:.4f}  "
              f"ESS={history.ess[i]:.1f}  "
              f"unique={history.n_unique[i]}  "
              f"twist_acc={history.twist_accuracy[i]:.4f}  "
              f"sizes={history.block_sizes[i]}")

    # Save results
    results = {
        "config": {k: v for k, v in config.__dict__.items()},
        "final_boundaries": final_boundaries,
        "history": {
            "boundaries": history.boundaries,
            "block_sizes": history.block_sizes,
            "log_ml": history.log_ml,
            "ess": history.ess,
            "n_unique": history.n_unique,
            "twist_loss": history.twist_loss,
            "twist_accuracy": history.twist_accuracy,
            "wall_time": history.wall_time,
        }
    }
    output_path = f"experiments/em_results_seed{config.seed}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    await llm.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EM boundary learning test")
    parser.add_argument("--n-em-iters", type=int, default=3)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--n-search-steps", type=int, default=10,
                        help="Local search steps per EM iteration (keep small for testing)")
    parser.add_argument("--search-particles", type=int, default=5)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    asyncio.run(main(args))
