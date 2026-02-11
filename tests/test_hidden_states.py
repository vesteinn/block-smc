"""Tests for HiddenStateCache (extractor tests require a model)."""

import torch
import pytest
from block_smc.hidden_states import HiddenStateCache


class TestHiddenStateCache:
    def test_store_and_get(self):
        cache = HiddenStateCache()
        h = torch.randn(64)
        cache.store(particle_id=0, boundary_idx=1, hidden_state=h)
        retrieved = cache.get(0, 1)
        assert retrieved is not None
        torch.testing.assert_close(retrieved, h)

    def test_get_missing(self):
        cache = HiddenStateCache()
        assert cache.get(0, 0) is None

    def test_len(self):
        cache = HiddenStateCache()
        assert len(cache) == 0
        cache.store(0, 0, torch.randn(64))
        cache.store(0, 1, torch.randn(64))
        cache.store(1, 0, torch.randn(64))
        assert len(cache) == 3

    def test_clear(self):
        cache = HiddenStateCache()
        cache.store(0, 0, torch.randn(64))
        cache.clear()
        assert len(cache) == 0

    def test_remap_ancestors(self):
        cache = HiddenStateCache()
        h0 = torch.randn(64)
        h1 = torch.randn(64)
        cache.store(0, 0, h0)
        cache.store(1, 0, h1)

        # After resampling: particles 0,1,2 all descend from ancestor 0
        cache.remap_ancestors({0: 0, 1: 0, 2: 0})

        # All new particles should have ancestor 0's hidden state
        assert cache.get(0, 0) is not None
        assert cache.get(1, 0) is not None
        assert cache.get(2, 0) is not None
        torch.testing.assert_close(cache.get(0, 0), h0)
        torch.testing.assert_close(cache.get(1, 0), h0)
        torch.testing.assert_close(cache.get(2, 0), h0)
