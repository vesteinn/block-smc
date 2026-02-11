"""TwistedBlockCritic: Potential wrapping Φ_exp + learned twist ψ.

The shaped potential G̃_k = G_k · ψ_k / ψ_{k-1} is implemented by returning
log(Φ_exp_prefix(ctx)) + log(ψ_k) from prefix(), and log(Φ_exp_complete(ctx))
from complete(). The twist/untwist mechanism in llamppl automatically computes
the incremental G̃_k as differences between consecutive twist amounts.

Integration with genlm-control's SMC:
    SequenceModel.step() calls critic.score(token_ctx) after each unit.
    - During generation: score() dispatches to prefix() → returns log(Φ_prefix) + log(ψ)
    - At completion:   score() dispatches to complete() → returns log(Φ_complete)
    The twist mechanism (add at step k, subtract at step k+1) gives us:
        G̃_k = exp(twist_k - twist_{k-1}) = (Φ_prefix_k · ψ_k) / (Φ_prefix_{k-1} · ψ_{k-1})
"""

import torch
from typing import Optional

from genlm.control.potential.base import Potential

from block_smc.twist import TwistHead
from block_smc.hidden_states import HiddenStateExtractor


class TwistedBlockCritic(Potential):
    """Critic combining expensive potential with learned twist function.

    At each block boundary, returns:
        prefix:   log Φ_exp(prefix) + log ψ_k(h, k/K)
        complete: log Φ_exp(complete)

    The twist/untwist lifecycle in llamppl ensures ψ influences resampling
    but telescopes out of the final importance weight.

    Args:
        expensive_potential: The Φ_exp potential (e.g., PartialSMILES).
        twist_head: Learned TwistHead (or None for Φ_exp-only mode).
        hidden_state_extractor: For extracting h from the LM.
        num_blocks: Expected number of blocks K (for normalising k/K).
            If None, uses the count of completed units as k.
        twist_scale: Scaling factor α for log(ψ). Default 1.0 (no scaling).
            Values < 1 temper the twist's influence on resampling weights.
    """

    def __init__(
        self,
        expensive_potential: Potential,
        twist_head: Optional[TwistHead] = None,
        hidden_state_extractor: Optional[HiddenStateExtractor] = None,
        num_blocks: Optional[int] = None,
        twist_scale: float = 1.0,
    ):
        # Inherit vocabulary from the expensive potential
        super().__init__(
            vocabulary=expensive_potential.vocab,
            token_type=expensive_potential.token_type,
            eos=expensive_potential.eos,
        )
        self.expensive_potential = expensive_potential
        self.twist_head = twist_head
        self.hidden_state_extractor = hidden_state_extractor
        self.num_blocks = num_blocks
        self.twist_scale = twist_scale

        # Track boundary index and hidden states for training data collection
        self._last_hidden_states: dict[int, list[torch.Tensor]] = {}

    async def prefix(self, context) -> float:
        """Score a prefix: log Φ_exp(prefix) + log ψ_k(h, k/K).

        Args:
            context: Token context (may be nested list of units from MultiTokenUnitSampler).

        Returns:
            Log-weight for the twist mechanism.
        """
        flat_ctx = _flatten_context(context)

        # Expensive potential prefix score
        phi_score = await self.expensive_potential.prefix(flat_ctx)

        if phi_score == float("-inf"):
            return float("-inf")

        # Add twist if available
        if self.twist_head is not None and self.hidden_state_extractor is not None:
            twist_score = self._compute_twist(context)
            return phi_score + twist_score

        return phi_score

    async def complete(self, context) -> float:
        """Score a complete sequence: log Φ_exp(complete) only.

        No twist at completion — the twist telescopes out via untwist().

        Args:
            context: Token context (may be nested).

        Returns:
            Log-weight of the expensive potential on the complete sequence.
        """
        flat_ctx = _flatten_context(context)
        return await self.expensive_potential.complete(flat_ctx)

    def _compute_twist(self, context) -> float:
        """Compute log ψ_k(h, k/K) for the current block boundary.

        Args:
            context: Token context (nested list of units).

        Returns:
            log ψ value as a float.
        """
        # Determine boundary index k
        # context is a list of units; the number of units = block index k
        k = len(context)
        K = self.num_blocks if self.num_blocks is not None else max(k, 1)
        boundary_frac = k / K

        # Extract hidden state at the last token of the current prefix
        flat_ctx = _flatten_context(context)
        token_ids = self.hidden_state_extractor.token_ids_from_bytes(flat_ctx)
        if not token_ids:
            return 0.0

        h = self.hidden_state_extractor.extract(token_ids, position=-1)

        # Compute log ψ, scaled by twist_scale to temper influence on resampling
        with torch.no_grad():
            bf_tensor = torch.tensor([boundary_frac], device=h.device, dtype=h.dtype)
            log_psi = self.twist_head.log_psi(h.unsqueeze(0), bf_tensor)
            return self.twist_scale * log_psi.item()

    def get_last_hidden_state(self, context) -> Optional[torch.Tensor]:
        """Extract and return the hidden state for the current context.

        Utility for training data collection — avoids recomputation
        when the caller also needs the hidden state.
        """
        if self.hidden_state_extractor is None:
            return None

        flat_ctx = _flatten_context(context)
        token_ids = self.hidden_state_extractor.token_ids_from_bytes(flat_ctx)
        if not token_ids:
            return None

        return self.hidden_state_extractor.extract(token_ids, position=-1)

    async def cleanup(self):
        await self.expensive_potential.cleanup()


class OracleTwistCritic(Potential):
    """Critic using the expensive potential itself as the twist function.

    This is a validation tool: by using Φ_exp as ψ, we test the shaped
    potential wiring without a learned twist. The twist should equal the
    potential's prefix score, giving shaped potential:
        G̃_k = Φ_prefix(x_{1..k}) / Φ_prefix(x_{1..k-1})

    This is mathematically equivalent to the baseline's critic but with
    block-level resampling instead of token-level.

    Args:
        expensive_potential: The Φ_exp potential.
    """

    def __init__(self, expensive_potential: Potential):
        super().__init__(
            vocabulary=expensive_potential.vocab,
            token_type=expensive_potential.token_type,
            eos=expensive_potential.eos,
        )
        self.expensive_potential = expensive_potential

    async def prefix(self, context) -> float:
        flat_ctx = _flatten_context(context)
        return await self.expensive_potential.prefix(flat_ctx)

    async def complete(self, context) -> float:
        flat_ctx = _flatten_context(context)
        return await self.expensive_potential.complete(flat_ctx)

    async def cleanup(self):
        await self.expensive_potential.cleanup()


def _flatten_context(context) -> list:
    """Flatten nested unit context to flat token list.

    Handles both flat ([tok1, tok2, ...]) and nested ([[tok1, tok2], [tok3], ...])
    contexts from MultiTokenUnitSampler.
    """
    flattened = []
    for item in context:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened
