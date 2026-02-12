#!/bin/bash
# Wrapper script to run holdout test with unbuffered output
export PYTHONUNBUFFERED=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate blocksmc
cd /home/vesteinn/Projects/BLOCK_SMC/code/block-smc
exec python -u experiments/holdout_test.py "$@"
