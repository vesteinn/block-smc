"""Factory functions composing the full Block SMC pipeline.

Wires together:
    LLM * grammar_potential → token-level product
    → MultiTokenUnitSampler with BoundaryPredicate → block-level generation
    → SMC with TwistedBlockCritic → shaped potential reweighting + resampling

Components:
    make_block_smc — create a configured Block SMC sampler for a given domain
"""

from typing import Optional

from genlm.control.potential.base import Potential
from genlm.control.sampler.sequence import SMC
from genlm.control.sampler.unit import MultiTokenUnitSampler, BoundaryPredicate

from block_smc.twist import TwistHead
from block_smc.critic import TwistedBlockCritic, OracleTwistCritic
from block_smc.hidden_states import HiddenStateExtractor


def make_block_smc(
    token_sampler,
    boundary_predicate: BoundaryPredicate,
    expensive_potential: Potential,
    twist_head: Optional[TwistHead] = None,
    hidden_state_extractor: Optional[HiddenStateExtractor] = None,
    num_blocks: Optional[int] = None,
    max_subunits_per_unit: int = 100,
    oracle_twist: bool = False,
) -> SMC:
    """Create a Block SMC sampler with optional learned twist.

    This is the main entry point for constructing the Block SMC pipeline.

    Args:
        token_sampler: Token-level sampler (e.g., AWRS(llm * grammar)).
            Must be a TokenSampler instance.
        boundary_predicate: Determines block boundaries.
        expensive_potential: The Φ_exp potential for constraint checking.
        twist_head: Learned TwistHead, or None for Φ_exp-only critic.
        hidden_state_extractor: For extracting hidden states. Required if twist_head is provided.
        num_blocks: Expected number of blocks K. If None, inferred from context length.
        max_subunits_per_unit: Max tokens per block before timeout.
        oracle_twist: If True, use the expensive potential as the twist (for validation).
            Overrides twist_head.

    Returns:
        SMC instance ready to call with (n_particles, ess_threshold, max_tokens).
    """
    # Wrap token sampler in multi-token unit sampler
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=token_sampler,
        boundary_predicate=boundary_predicate,
        max_subunits_per_unit=max_subunits_per_unit,
    )

    # Build critic
    if oracle_twist:
        critic = OracleTwistCritic(expensive_potential)
    else:
        critic = TwistedBlockCritic(
            expensive_potential=expensive_potential,
            twist_head=twist_head,
            hidden_state_extractor=hidden_state_extractor,
            num_blocks=num_blocks,
        )

    # Coerce critic to match the token sampler's token type if needed
    if unit_sampler.token_type != critic.token_type:
        critic = critic.coerce(unit_sampler.target, f=lambda ctx: ctx)

    return SMC(unit_sampler=unit_sampler, critic=critic)


async def run_block_smc(
    smc: SMC,
    n_particles: int = 10,
    ess_threshold: float = 0.9,
    max_tokens: int = 100,
    **kwargs,
):
    """Run Block SMC inference and return results.

    Convenience wrapper around SMC.__call__ with sensible defaults.

    Args:
        smc: SMC instance from make_block_smc().
        n_particles: Number of particles. Default 10.
        ess_threshold: ESS threshold for resampling (fraction of n_particles). Default 0.9.
        max_tokens: Maximum tokens per sequence. Default 100.
        **kwargs: Additional kwargs passed to smc_standard.

    Returns:
        Sequences object with contexts, log_weights, posterior, etc.
    """
    return await smc(
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        **kwargs,
    )
