# Block SMC with Learned Twist Functions

> **Work in progress** -- This is an active research prototype. APIs, experiment scripts, and results are subject to change. Expect rough edges.

Block-level Sequential Monte Carlo with learned twist functions for constrained text generation. Built on top of [genlm-control](https://github.com/genlm/genlm-control).

## Overview

Standard token-level SMC evaluates constraint potentials at every token, which is prohibitively expensive when potentials involve external tools (plan validators, code executors, parsers). Block SMC evaluates potentials only at block boundaries, using a learned twist function to compensate for the reduced feedback.

Key components:
- **Block-level SMC**: Resampling at block boundaries instead of every token
- **Learned twist function**: MLP head on LM hidden states, trained via weighted BCE from SMC particle trajectories
- **Shaped potential**: G̃_k = G_k · ψ_k / ψ_{k-1} (telescoping preserves the target distribution)
- **Online adaptation**: Per-instance twist fine-tuning with L2 regularization toward pretrained weights
- **EM boundary learning**: Optimize block boundary positions via DP + local search

## Installation

Requires [genlm-control](https://github.com/genlm/genlm-control), [control-iclr-2025](https://github.com/genlm/control-iclr-2025), and Python 3.11+.

```bash
pip install -e ./code/genlm-control -e ./code/control-iclr-2025 -e ./code/block-smc
```

## Package Structure

```
src/block_smc/
├── twist.py             # TwistHead (MLP), TwistTrainingBuffer, training loop, online adaptation
├── critic.py            # TwistedBlockCritic, OracleTwistCritic (Potential wrappers)
├── boundary.py          # Boundary predicates (fixed interval, SMILES, SQL clause)
├── hidden_states.py     # HiddenStateExtractor for LM hidden states
├── sampler.py           # Factory functions: make_block_smc, run_block_smc
├── boundary_learning.py # DP + local search for boundary optimization
└── em.py                # Full EM loop (E-step: boundaries, M-step: twist)
```

## Experiments

Experiment scripts are in `experiments/`. The two main evaluation scripts are:

### Goal Inference (PDDL / Planetarium)

```bash
# Quick test (5 instances, fast iteration)
python experiments/holdout_test.py --quick

# Full evaluation (10 train, 50 test, all methods)
python experiments/holdout_test.py

# Load a pre-trained twist
python experiments/holdout_test.py --load-twist experiments/twist_weights_obj9.pt

# Run specific methods only
python experiments/holdout_test.py --methods vanilla,twist_mlp_s0.1,online_mlp_s0.1

# With grammar-free pretraining rounds (better negative labels)
python experiments/holdout_test.py --n-gramfree-rounds 4
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `Llama-3.2-1B-Instruct` | HuggingFace model name |
| `--n-train-instances` | 10 | Instances for twist training |
| `--n-test-instances` | 50 | Instances for evaluation |
| `--n-particles` | 10 | SMC particles |
| `--max-tokens` | 150 | Max tokens per sample |
| `--block-size` | 10 | Tokens per block (fixed interval) |
| `--max-objects` | 9 | PDDL problem complexity |
| `--n-train-rounds` | 8 | SMC sweeps per train instance |
| `--n-gramfree-rounds` | 0 | Grammar-free exploration rounds (produces negative labels) |
| `--n-seeds` | 1 | Seeds per test instance |
| `--twist-scale` | 0.1 | Scale α for log(ψ) in shaped potential |
| `--logit-clamp` | 3.0 | Clamp twist logits to [-c, c] (0=off) |
| `--online-epochs` | 3 | Gradient steps for online adaptation |
| `--online-lr` | 1e-4 | Learning rate for online adaptation |
| `--online-l2` | 0.1 | L2 penalty toward pretrained weights |
| `--methods` | all | Comma-separated method filter |
| `--load-twist` | | Path to pre-trained twist checkpoint |
| `--save-twist` | auto | Path to save twist checkpoint |
| `--resume` | | Resume from existing results file |
| `--quick` | | 5 instances, 1 seed, subset of methods |

**Methods:** `baseline` (token-level SMC), `bracket` (bracket boundaries), `vanilla` (block SMC, no twist), `twist_mlp_s{α}` (pretrained twist), `online_mlp_s{α}` (online adaptation), `scratch_s{α}` (train from scratch per instance), `gramfree_s{α}` (grammar-free scratch)

### Molecular Synthesis (SMILES)

```bash
# Quick test
python experiments/holdout_test_molecular.py --quick

# Full evaluation
python experiments/holdout_test_molecular.py

# With SMILES-aware boundaries (default) or fixed-interval
python experiments/holdout_test_molecular.py --boundary-type smiles
python experiments/holdout_test_molecular.py --boundary-type fixed
```

**Additional arguments** (beyond those shared with Goal Inference):

| Argument | Default | Description |
|---|---|---|
| `--boundary-type` | `smiles` | Block boundary type (`smiles` or `fixed`) |
| `--max-tokens` | 40 | Shorter sequences for SMILES |
| `--block-size` | 5 | Smaller blocks for SMILES |

**Methods:** `baseline`, `vanilla`, `twist_mlp_s{α}`, `online_mlp_s{α}`, `scratch_s{α}`

### Other scripts

| Script | Description |
|---|---|
| `diagnose_twist.py` | Twist function diagnostics (generalization, calibration, OOD detection) |
| `run_goal_inference.py` | Older single-method Goal Inference runner |
| `compare_methods.py` | Early method comparison (superseded by holdout_test) |
| `sweep_twist.py` | Hyperparameter sweep for twist training |
| `test_oracle_twist.py` | Oracle twist upper bound experiments |

## License

MIT
