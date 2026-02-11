"""Tests for boundary learning: DP, local search, helpers."""

import pytest
import numpy as np
from block_smc.boundary_learning import (
    is_valid_segmentation,
    get_block_sizes,
    optimize_boundaries_dp,
    local_search,
    make_initial_boundaries,
)


class TestHelpers:
    def test_valid_segmentation(self):
        assert is_valid_segmentation([5, 10], T=10, min_block=2, max_block=10)

    def test_invalid_no_end_at_T(self):
        assert not is_valid_segmentation([5, 8], T=10, min_block=2, max_block=10)

    def test_invalid_block_too_small(self):
        assert not is_valid_segmentation([1, 10], T=10, min_block=2, max_block=10)

    def test_invalid_block_too_large(self):
        assert not is_valid_segmentation([20], T=20, min_block=2, max_block=10)

    def test_invalid_empty(self):
        assert not is_valid_segmentation([], T=10, min_block=2, max_block=10)

    def test_get_block_sizes(self):
        assert get_block_sizes([5, 12, 20]) == [5, 7, 8]

    def test_get_block_sizes_single(self):
        assert get_block_sizes([10]) == [10]


class TestDP:
    def test_uniform_discrimination(self):
        """With uniform disc scores, DP should produce evenly-sized blocks."""
        T = 12
        disc_cache = {}
        for s in range(T):
            for e in range(s + 2, min(s + 7, T + 1)):
                disc_cache[(s, e)] = 1.0  # uniform score

        boundaries = optimize_boundaries_dp(
            disc_cache, T=T, min_block=2, max_block=6, lam=0.0
        )
        assert boundaries[-1] == T
        assert is_valid_segmentation(boundaries, T, min_block=2, max_block=6)

    def test_prefers_high_discrimination(self):
        """DP should prefer segmentations using high-disc blocks."""
        T = 10
        disc_cache = {}
        for s in range(T):
            for e in range(s + 2, min(s + 6, T + 1)):
                disc_cache[(s, e)] = 0.1

        # Make one segmentation much better: [0,5] [5,10]
        disc_cache[(0, 5)] = 10.0
        disc_cache[(5, 10)] = 10.0

        boundaries = optimize_boundaries_dp(
            disc_cache, T=T, min_block=2, max_block=5, lam=0.0
        )
        assert boundaries == [5, 10]

    def test_lambda_reduces_boundaries(self):
        """Higher lambda should produce fewer boundaries."""
        T = 20
        disc_cache = {}
        for s in range(T):
            for e in range(s + 2, min(s + 11, T + 1)):
                disc_cache[(s, e)] = 1.0

        b_low = optimize_boundaries_dp(
            disc_cache, T=T, min_block=2, max_block=10, lam=0.01
        )
        b_high = optimize_boundaries_dp(
            disc_cache, T=T, min_block=2, max_block=10, lam=5.0
        )
        # Higher lambda → fewer boundaries
        assert len(b_high) <= len(b_low)

    def test_no_valid_path(self):
        """Returns empty list when no valid segmentation exists."""
        boundaries = optimize_boundaries_dp(
            {}, T=10, min_block=2, max_block=3
        )
        assert boundaries == []


class TestLocalSearch:
    @pytest.mark.asyncio
    async def test_accepts_improvement(self):
        """Local search should improve or maintain the score."""
        T = 20
        calls = []

        async def eval_fn(boundaries):
            # Score inversely proportional to size variance
            sizes = get_block_sizes(boundaries)
            variance = np.var(sizes)
            score = -variance
            calls.append(boundaries.copy())
            return score

        init = [5, 10, 15, 20]  # even blocks of 5
        best, best_score = await local_search(
            init, eval_fn, T=T, min_block=2, max_block=10, n_steps=20, seed=42
        )
        assert best[-1] == T
        assert is_valid_segmentation(best, T, min_block=2, max_block=10)
        # Should have evaluated initial solution
        assert len(calls) > 0

    @pytest.mark.asyncio
    async def test_respects_constraints(self):
        """All evaluated candidates should be valid segmentations."""
        T = 15
        evaluated = []

        async def eval_fn(boundaries):
            evaluated.append(boundaries.copy())
            return float(-len(boundaries))

        init = [5, 10, 15]
        await local_search(
            init, eval_fn, T=T, min_block=2, max_block=8, n_steps=30, seed=42
        )

        for b in evaluated:
            assert is_valid_segmentation(b, T, min_block=2, max_block=8), f"Invalid: {b}"


class TestMakeInitialBoundaries:
    def test_basic(self):
        b = make_initial_boundaries(T=20, interval=5)
        assert b == [5, 10, 15, 20]

    def test_uneven(self):
        b = make_initial_boundaries(T=22, interval=5)
        assert b[-1] == 22
        sizes = get_block_sizes(b)
        assert all(s >= 2 for s in sizes)

    def test_merges_tiny_last_block(self):
        b = make_initial_boundaries(T=11, interval=5, min_block=2)
        # Without merge: [5, 10, 11] → last block size 1 < min_block
        # Should merge: [5, 11] → block sizes [5, 6]
        assert b[-1] == 11
        sizes = get_block_sizes(b)
        assert all(s >= 2 for s in sizes)
