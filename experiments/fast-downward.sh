#!/bin/bash
# Wrapper for Fast-Downward that mimics the .sif singularity interface
exec python /tmp/fast-downward/fast-downward.py "$@"
