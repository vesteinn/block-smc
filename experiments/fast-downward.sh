#!/bin/bash
# Wrapper for Fast-Downward that mimics the .sif singularity interface
exec python /home/vesteinn/tools/fast-downward/fast-downward.py "$@"
