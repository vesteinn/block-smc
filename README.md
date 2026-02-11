# Block SMC with Learned Twist Functions

Block-level Sequential Monte Carlo with learned twist functions for constrained text generation. Built on top of [genlm-control](https://github.com/genlm/genlm-control).

**Status**: Early-stage research prototype.

## Overview

Standard token-level SMC evaluates constraint potentials at every token, which is prohibitively expensive when potentials involve external tools (plan validators, code executors, parsers). Block SMC evaluates potentials only at block boundaries, using a learned twist function to compensate for the reduced feedback.

Key components:
- **Block-level SMC**: Resampling at block boundaries instead of every token
- **Learned twist function**: MLP head on LM hidden states, trained via weighted BCE from SMC particle trajectories
- **Shaped potential**: G̃_k = G_k · ψ_k / ψ_{k-1} (telescoping preserves the target distribution)
- **EM boundary learning**: Optimize block boundary positions via DP + local search

## Installation

Requires [genlm-control](https://github.com/genlm/genlm-control) and Python 3.11+.

```bash
pip install -e .
```

## Package Structure

```
src/block_smc/
├── twist.py             # TwistHead (MLP), TwistTrainingBuffer, training loop
├── critic.py            # TwistedBlockCritic, OracleTwistCritic (Potential wrappers)
├── boundary.py          # Boundary predicates (fixed interval, SMILES, SQL clause)
├── hidden_states.py     # HiddenStateExtractor for LM hidden states
├── sampler.py           # Factory functions: make_block_smc, run_block_smc
├── boundary_learning.py # DP + local search for boundary optimization
└── em.py                # Full EM loop (E-step: boundaries, M-step: twist)
```

## Usage

```python
from block_smc import (
    make_block_smc, run_block_smc, decode_block_sequences,
    TwistHead, HiddenStateExtractor, FixedIntervalBoundary,
)

# Create block SMC sampler with learned twist
smc = make_block_smc(
    token_sampler=token_sampler,           # from genlm-control
    boundary_predicate=FixedIntervalBoundary(block_size=10),
    expensive_potential=potential,          # evaluated at boundaries only
    llm=llm,
    twist_head=twist_head,                 # learned MLP
    hidden_state_extractor=hse,
    num_blocks=15,
)

# Run inference
sequences = await run_block_smc(smc, n_particles=10, max_tokens=150)
posterior = decode_block_sequences(sequences)
```

## Experiments

Experiment scripts are in `experiments/`. Each domain has its own entry point:

```bash
# Molecular synthesis (SMILES)
python experiments/compare_methods.py

# Goal inference (PDDL / Planetarium)
python experiments/run_goal_inference.py --n-instances 20 --max-objects 5
```

## License

MIT
