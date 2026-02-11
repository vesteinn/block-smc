"""Block SMC with learned twist functions and EM boundary learning."""

from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data, adapt_twist_online, restore_twist_weights
from block_smc.critic import TwistedBlockCritic, OracleTwistCritic
from block_smc.boundary import (
    PositionListBoundary,
    FixedIntervalBoundary,
    SMILESBoundary,
    SQLClauseBoundary,
)
from block_smc.hidden_states import HiddenStateExtractor, HiddenStateCache
from block_smc.sampler import make_block_smc, run_block_smc, decode_block_sequences
from block_smc.boundary_learning import (
    optimize_boundaries_dp,
    local_search,
    make_initial_boundaries,
    is_valid_segmentation,
    get_block_sizes,
)
from block_smc.em import EMConfig, EMHistory, run_em

__all__ = [
    # Twist
    "TwistHead",
    "TwistTrainingBuffer",
    "train_twist_step",
    "collect_twist_training_data",
    "adapt_twist_online",
    "restore_twist_weights",
    # Critics
    "TwistedBlockCritic",
    "OracleTwistCritic",
    # Boundaries
    "PositionListBoundary",
    "FixedIntervalBoundary",
    "SMILESBoundary",
    "SQLClauseBoundary",
    # Hidden states
    "HiddenStateExtractor",
    "HiddenStateCache",
    # Pipeline
    "make_block_smc",
    "run_block_smc",
    "decode_block_sequences",
    # Boundary learning
    "optimize_boundaries_dp",
    "local_search",
    "make_initial_boundaries",
    "is_valid_segmentation",
    "get_block_sizes",
    # EM
    "EMConfig",
    "EMHistory",
    "run_em",
]
