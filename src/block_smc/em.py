"""Full EM loop: alternates boundary optimisation (E-step) and twist training (M-step).

Pipeline per iteration:
    1. Run Block SMC with current (boundaries, twist) → particles + trajectories
    2. Collect twist training data from trajectories
    3. M-step: train twist head on buffered data
    4. E-step: local search to refine boundaries using Block SMC feedback
    5. Log diagnostics

Components:
    EMConfig   — configuration for the EM loop
    EMHistory  — per-iteration diagnostics
    run_em     — outer EM loop
"""

import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from block_smc.boundary import PositionListBoundary
from block_smc.boundary_learning import (
    local_search,
    make_initial_boundaries,
    get_block_sizes,
    is_valid_segmentation,
)
from block_smc.twist import (
    TwistHead,
    TwistTrainingBuffer,
    train_twist_step,
    collect_twist_training_data,
)
from block_smc.hidden_states import HiddenStateExtractor
from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences


@dataclass
class EMConfig:
    """Configuration for the EM loop."""

    # SMC parameters
    n_particles: int = 10
    ess_threshold: float = 0.9
    max_tokens: int = 40

    # Boundary constraints
    min_block: int = 2
    max_block: int = 15
    initial_interval: int = 5

    # EM iterations
    n_em_iters: int = 10

    # Local search (E-step)
    n_search_steps: int = 50
    search_n_runs: int = 2
    search_n_particles: int = 5

    # Twist training (M-step)
    twist_hidden_dim: int = 256
    twist_lr: float = 1e-3
    twist_epochs: int = 10
    twist_batch_size: int = 64
    buffer_discount: float = 0.9

    # General
    seed: int = 42
    verbose: bool = True


@dataclass
class EMHistory:
    """Per-iteration diagnostics from the EM loop."""

    boundaries: list[list[int]] = field(default_factory=list)
    block_sizes: list[list[int]] = field(default_factory=list)
    log_ml: list[float] = field(default_factory=list)
    ess: list[float] = field(default_factory=list)
    n_unique: list[int] = field(default_factory=list)
    twist_loss: list[float] = field(default_factory=list)
    twist_accuracy: list[float] = field(default_factory=list)
    buffer_size: list[int] = field(default_factory=list)
    search_score: list[float] = field(default_factory=list)
    wall_time: list[float] = field(default_factory=list)
    posteriors: list[dict] = field(default_factory=list)


async def run_em(
    token_sampler,
    expensive_potential,
    llm,
    config: EMConfig = EMConfig(),
    initial_boundaries: Optional[list[int]] = None,
) -> tuple[EMHistory, TwistHead, list[int]]:
    """Run the full EM algorithm.

    Args:
        token_sampler: Token-level sampler (e.g., eager_token_sampler(llm, grammar)).
        expensive_potential: Byte-level constraint potential (e.g., PartialSMILES).
        llm: PromptedLLM instance.
        config: EM configuration.
        initial_boundaries: Starting boundary positions. If None, uses
            evenly-spaced boundaries.

    Returns:
        (history, trained_twist_head, final_boundaries)
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Set up hidden state extractor and twist head
    hse = HiddenStateExtractor(llm)
    d_model = hse.hidden_dim
    device = hse.device
    twist_head = TwistHead(d_model=d_model, hidden_dim=config.twist_hidden_dim).to(device)
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=config.twist_lr)
    buffer = TwistTrainingBuffer()

    # Coerce expensive potential to LLM token type
    coerced_potential = expensive_potential
    if hasattr(expensive_potential, 'token_type') and expensive_potential.token_type != token_sampler.token_type:
        coerced_potential = expensive_potential.coerce(llm, f=b"".join)

    # Initialize boundaries
    T = config.max_tokens
    if initial_boundaries is not None:
        boundaries = initial_boundaries.copy()
    else:
        boundaries = make_initial_boundaries(
            T, interval=config.initial_interval,
            min_block=config.min_block, max_block=config.max_block,
        )

    history = EMHistory()

    if config.verbose:
        print(f"EM starting: T={T}, initial boundaries={boundaries}")
        print(f"  Block sizes: {get_block_sizes(boundaries)}")
        print(f"  Particles: {config.n_particles}, ESS threshold: {config.ess_threshold}")
        print()

    for it in range(config.n_em_iters):
        t0 = time.time()

        # =====================================================================
        # Step 1: Run Block SMC with current boundaries + twist
        # =====================================================================
        num_blocks = len(boundaries)
        smc = make_block_smc(
            token_sampler=token_sampler,
            boundary_predicate=PositionListBoundary(boundaries),
            expensive_potential=expensive_potential,
            llm=llm,
            twist_head=twist_head if it > 0 else None,  # No twist on first iter
            hidden_state_extractor=hse if it > 0 else None,
            num_blocks=num_blocks,
        )

        seq = await run_block_smc(
            smc,
            n_particles=config.n_particles,
            ess_threshold=config.ess_threshold,
            max_tokens=config.max_tokens,
        )

        posterior = decode_block_sequences(seq)
        n_unique = len(posterior)

        # =====================================================================
        # Step 2: Collect twist training data
        # =====================================================================
        stats = await collect_twist_training_data(
            sequences=seq,
            hidden_state_extractor=hse,
            expensive_potential=coerced_potential,
            buffer=buffer,
        )

        # =====================================================================
        # Step 3: M-step — train twist head
        # =====================================================================
        train_result = train_twist_step(
            twist_head=twist_head,
            buffer=buffer,
            optimizer=optimizer,
            device=device,
            n_epochs=config.twist_epochs,
            batch_size=config.twist_batch_size,
            discount=config.buffer_discount,
        )

        buffer.advance_age()

        # =====================================================================
        # Step 4: E-step — local search to refine boundaries
        # =====================================================================
        async def quick_eval(candidate_boundaries: list[int]) -> float:
            """Evaluate a candidate segmentation by running Block SMC."""
            if not is_valid_segmentation(
                candidate_boundaries, T, config.min_block, config.max_block
            ):
                return -1e6

            total_log_ml = 0.0
            for _ in range(config.search_n_runs):
                smc_eval = make_block_smc(
                    token_sampler=token_sampler,
                    boundary_predicate=PositionListBoundary(candidate_boundaries),
                    expensive_potential=expensive_potential,
                    llm=llm,
                    twist_head=twist_head,
                    hidden_state_extractor=hse,
                    num_blocks=len(candidate_boundaries),
                )
                seq_eval = await run_block_smc(
                    smc_eval,
                    n_particles=config.search_n_particles,
                    ess_threshold=config.ess_threshold,
                    max_tokens=config.max_tokens,
                )
                total_log_ml += seq_eval.log_ml

            return total_log_ml / config.search_n_runs

        new_boundaries, search_score = await local_search(
            boundaries=boundaries,
            eval_fn=quick_eval,
            T=T,
            min_block=config.min_block,
            max_block=config.max_block,
            n_steps=config.n_search_steps,
            seed=config.seed + it,
        )

        dt = time.time() - t0

        # =====================================================================
        # Record diagnostics
        # =====================================================================
        history.boundaries.append(boundaries.copy())
        history.block_sizes.append(get_block_sizes(boundaries))
        history.log_ml.append(seq.log_ml)
        history.ess.append(seq.ess)
        history.n_unique.append(n_unique)
        history.twist_loss.append(train_result["loss"])
        history.twist_accuracy.append(train_result["accuracy"])
        history.buffer_size.append(len(buffer))
        history.search_score.append(search_score)
        history.wall_time.append(dt)
        history.posteriors.append(posterior)

        if config.verbose:
            print(f"Iter {it+1:2d} | boundaries={boundaries}")
            print(f"         sizes={get_block_sizes(boundaries)}")
            print(f"         log_ml={seq.log_ml:.4f}  ESS={seq.ess:.1f}  "
                  f"unique={n_unique}")
            print(f"         twist_loss={train_result['loss']:.4f}  "
                  f"twist_acc={train_result['accuracy']:.4f}  "
                  f"buffer={len(buffer)}")
            print(f"         new_boundaries={new_boundaries}  "
                  f"search_score={search_score:.4f}")
            print(f"         time={dt:.1f}s")
            if posterior:
                top = list(posterior.items())[:3]
                for s, p in top:
                    print(f"         {p:.3f}  {s}")
            print()

        # Check convergence: boundaries didn't change
        if new_boundaries == boundaries and it > 0:
            if config.verbose:
                print(f"  Boundaries converged at iteration {it+1}")
            boundaries = new_boundaries
            break

        boundaries = new_boundaries

    # Record final state
    history.boundaries.append(boundaries.copy())
    history.block_sizes.append(get_block_sizes(boundaries))

    if config.verbose:
        print(f"{'='*60}")
        print(f"EM complete after {len(history.log_ml)} iterations")
        print(f"Final boundaries: {boundaries}")
        print(f"Final block sizes: {get_block_sizes(boundaries)}")
        print(f"Final twist accuracy: {history.twist_accuracy[-1]:.4f}")
        print(f"Total wall time: {sum(history.wall_time):.1f}s")

    return history, twist_head, boundaries
