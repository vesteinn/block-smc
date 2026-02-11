"""Factory functions composing the full Block SMC pipeline.

Wires together:
    LLM * grammar_potential → token-level product
    → MultiTokenUnitSampler with BoundaryPredicate → block-level generation
    → SMC with TwistedBlockCritic → shaped potential reweighting + resampling

Components:
    make_block_smc — create a configured Block SMC sampler for a given domain
    decode_block_sequences — flatten nested unit contexts for decoding
"""

import numpy as np
from typing import Optional

from genlm.grammar import Float
from genlm.control.potential.base import Potential
from genlm.control.constant import EOS, EndOfSequence
from genlm.control.sampler.sequence import SMC, Sequences
from genlm.control.sampler.unit import MultiTokenUnitSampler, BoundaryPredicate

from block_smc.twist import TwistHead
from block_smc.critic import TwistedBlockCritic, OracleTwistCritic, _flatten_context
from block_smc.hidden_states import HiddenStateExtractor


def make_block_smc(
    token_sampler,
    boundary_predicate: BoundaryPredicate,
    expensive_potential: Potential,
    llm=None,
    twist_head: Optional[TwistHead] = None,
    hidden_state_extractor: Optional[HiddenStateExtractor] = None,
    num_blocks: Optional[int] = None,
    max_subunits_per_unit: int = 100,
    oracle_twist: bool = False,
    twist_scale: float = 1.0,
) -> SMC:
    """Create a Block SMC sampler with optional learned twist.

    This is the main entry point for constructing the Block SMC pipeline.

    Args:
        token_sampler: Token-level sampler (e.g., eager_token_sampler(llm, grammar)).
            Must be a TokenSampler instance.
        boundary_predicate: Determines block boundaries.
        expensive_potential: The Φ_exp potential for constraint checking.
            Should be a byte-level potential (e.g., PartialSMILES). Will be
            coerced to the token sampler's token type if llm is provided.
        llm: PromptedLLM instance. Required for coercing byte-level potentials
            to the LLM's token type. Also used by HiddenStateExtractor.
        twist_head: Learned TwistHead, or None for Φ_exp-only critic.
        hidden_state_extractor: For extracting hidden states. Required if twist_head is provided.
        num_blocks: Expected number of blocks K. If None, inferred from context length.
        max_subunits_per_unit: Max tokens per block before timeout.
        oracle_twist: If True, use the expensive potential as the twist (for validation).
            Overrides twist_head.
        twist_scale: Scaling factor α for log(ψ). Default 1.0 (no scaling).
            Values < 1 temper the twist's influence on resampling weights.

    Returns:
        SMC instance ready to call with (n_particles, ess_threshold, max_tokens).
    """
    # Wrap token sampler in multi-token unit sampler
    unit_sampler = MultiTokenUnitSampler(
        subunit_sampler=token_sampler,
        boundary_predicate=boundary_predicate,
        max_subunits_per_unit=max_subunits_per_unit,
    )

    # Coerce the expensive potential to match the token sampler's token type.
    # The baseline does: critic = critic.coerce(llm, f=b"".join)
    # This converts byte-level potentials to operate on LLM tokens.
    coerced_potential = expensive_potential
    if llm is not None and expensive_potential.token_type != unit_sampler.token_type:
        coerced_potential = expensive_potential.coerce(llm, f=b"".join)

    # Build critic
    if oracle_twist:
        critic = OracleTwistCritic(coerced_potential)
    else:
        critic = TwistedBlockCritic(
            expensive_potential=coerced_potential,
            twist_head=twist_head,
            hidden_state_extractor=hidden_state_extractor,
            num_blocks=num_blocks,
            twist_scale=twist_scale,
        )

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
    sequences = await smc(
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        **kwargs,
    )
    return sequences


def decode_block_sequences(sequences: Sequences) -> dict[str, float]:
    """Decode Block SMC output, flattening nested unit contexts.

    The standard Sequences.decoded_posterior fails with MultiTokenUnitSampler
    because contexts contain nested lists of tokens instead of flat lists.
    This function handles the flattening.

    Args:
        sequences: Sequences object from Block SMC.

    Returns:
        Dict mapping decoded strings to their posterior probabilities.
    """
    posterior = Float.chart()
    for sequence, w in zip(sequences.contexts, np.exp(sequences.log_weights)):
        flat = _flatten_context(sequence)
        # Check for EOS at the end
        has_eos = flat and isinstance(flat[-1], EndOfSequence)
        byte_tokens = [t for t in flat if isinstance(t, bytes)]
        if has_eos or byte_tokens:
            try:
                text = b"".join(byte_tokens).decode("utf-8")
                posterior[text] += w
            except (UnicodeDecodeError, TypeError):
                pass
    if posterior:
        return dict(posterior.normalize().sort_descending())
    return {}
