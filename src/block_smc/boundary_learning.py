"""Boundary learning: discrimination scores, DP initialisation, local search.

Components:
    compute_discrimination     — LM-based score variance at candidate boundary positions
    compute_all_discriminations — sweep all valid (start, end) pairs
    optimize_boundaries_dp     — DP shortest-path for optimal segmentation
    local_search               — shift/merge/split/swap moves with SMC feedback
"""
