"""Tests for TwistHead, TwistTrainingBuffer, and collect_twist_training_data."""

import torch
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step, collect_twist_training_data


class TestTwistHead:
    def test_output_shape_single(self):
        head = TwistHead(d_model=64, hidden_dim=32)
        h = torch.randn(64)
        bf = torch.tensor(0.5)
        psi = head(h, bf)
        assert psi.shape == ()
        assert 0 < psi.item() < 1

    def test_output_shape_batch(self):
        head = TwistHead(d_model=64, hidden_dim=32)
        h = torch.randn(8, 64)
        bf = torch.full((8,), 0.5)
        psi = head(h, bf)
        assert psi.shape == (8,)
        assert (psi > 0).all() and (psi < 1).all()

    def test_log_psi_consistency(self):
        head = TwistHead(d_model=64, hidden_dim=32)
        h = torch.randn(4, 64)
        bf = torch.full((4,), 0.3)
        psi = head(h, bf)
        log_psi = head.log_psi(h, bf)
        torch.testing.assert_close(torch.log(psi), log_psi, atol=1e-5, rtol=1e-5)

    def test_log_psi_no_nan(self):
        """log_psi should never produce NaN even for extreme inputs."""
        head = TwistHead(d_model=64, hidden_dim=32)
        h = torch.randn(16, 64) * 100  # large inputs
        bf = torch.full((16,), 0.5)
        log_psi = head.log_psi(h, bf)
        assert not torch.isnan(log_psi).any()
        assert (log_psi <= 0).all()  # log of value in (0,1) is negative

    def test_boundary_frac_affects_output(self):
        """Different boundary fractions should generally give different outputs."""
        head = TwistHead(d_model=64, hidden_dim=32)
        h = torch.randn(64)
        psi_0 = head(h, torch.tensor(0.0))
        psi_1 = head(h, torch.tensor(1.0))
        # Not guaranteed to be different, but very unlikely to be exactly equal
        # after random init
        assert psi_0.item() != psi_1.item()


class TestTwistTrainingBuffer:
    def test_add_and_len(self):
        buf = TwistTrainingBuffer()
        assert len(buf) == 0
        buf.add(torch.randn(64), weight=0.5, label=1.0, boundary_frac=0.25)
        assert len(buf) == 1

    def test_add_batch(self):
        buf = TwistTrainingBuffer()
        n = 10
        buf.add_batch(
            hidden_states=torch.randn(n, 64),
            weights=np.ones(n) / n,
            labels=np.array([1.0] * 5 + [0.0] * 5),
            boundary_frac=0.5,
        )
        assert len(buf) == n

    def test_to_tensors_shapes(self):
        buf = TwistTrainingBuffer()
        d = 64
        for i in range(20):
            buf.add(torch.randn(d), weight=1.0, label=float(i % 2), boundary_frac=i / 20)
        h, w, y, bf = buf.to_tensors(torch.device("cpu"))
        assert h.shape == (20, d)
        assert w.shape == (20,)
        assert y.shape == (20,)
        assert bf.shape == (20,)

    def test_age_discounting(self):
        buf = TwistTrainingBuffer()
        buf.add(torch.randn(8), weight=1.0, label=1.0, boundary_frac=0.5)
        buf.advance_age()
        buf.add(torch.randn(8), weight=1.0, label=0.0, boundary_frac=0.5)

        _, w, _, _ = buf.to_tensors(torch.device("cpu"), discount=0.5)
        # First entry is age 1, discount = 0.5^1 = 0.5
        # Second entry is age 0, discount = 0.5^0 = 1.0
        assert abs(w[0].item() - 0.5) < 1e-6
        assert abs(w[1].item() - 1.0) < 1e-6

    def test_clear(self):
        buf = TwistTrainingBuffer()
        buf.add(torch.randn(8), weight=1.0, label=1.0, boundary_frac=0.5)
        buf.clear()
        assert len(buf) == 0


class TestTrainTwistStep:
    def test_training_reduces_loss(self):
        """Training on a simple signal should reduce loss."""
        d = 32
        head = TwistHead(d_model=d, hidden_dim=16)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        buf = TwistTrainingBuffer()

        # Create data with a simple signal: positive label when first dim > 0
        torch.manual_seed(42)
        for _ in range(100):
            h = torch.randn(d)
            label = 1.0 if h[0] > 0 else 0.0
            buf.add(h, weight=1.0, label=label, boundary_frac=0.5)

        result1 = train_twist_step(head, buf, optimizer, torch.device("cpu"), n_epochs=5)
        result2 = train_twist_step(head, buf, optimizer, torch.device("cpu"), n_epochs=20)

        assert result2["loss"] < result1["loss"] or result2["accuracy"] > 0.6

    def test_empty_buffer(self):
        head = TwistHead(d_model=32, hidden_dim=16)
        optimizer = torch.optim.Adam(head.parameters())
        buf = TwistTrainingBuffer()
        result = train_twist_step(head, buf, optimizer, torch.device("cpu"))
        assert np.isnan(result["loss"])


class TestCollectTwistTrainingData:
    """Tests for collect_twist_training_data using mock objects."""

    def _make_mock_sequences(self, n_particles=3, n_units=2):
        """Create mock Sequences with nested unit contexts."""
        from genlm.control.constant import EndOfSequence

        contexts = []
        for i in range(n_particles):
            # Each context: [[unit1_tokens], [unit2_tokens], EndOfSequence]
            units = [[b"tok%d_%d" % (i, k)] for k in range(n_units)]
            units.append(EndOfSequence())
            contexts.append(units)

        log_weights = np.array([0.0] * n_particles)  # uniform weights
        norm_weights = np.exp(log_weights - np.log(np.sum(np.exp(log_weights))))

        seq = MagicMock()
        seq.contexts = contexts
        seq.normalized_weights = norm_weights
        return seq

    def _make_mock_hse(self, d_model=32):
        """Create mock HiddenStateExtractor."""
        hse = MagicMock()
        hse.token_ids_from_bytes = MagicMock(return_value=[1, 2, 3])
        hse.extract = MagicMock(return_value=torch.randn(d_model))
        return hse

    def _make_mock_potential(self, complete_score=0.0, prefix_score=0.0):
        """Create mock expensive potential."""
        pot = AsyncMock()
        pot.complete = AsyncMock(return_value=complete_score)
        pot.prefix = AsyncMock(return_value=prefix_score)
        return pot

    @pytest.mark.asyncio
    async def test_basic_collection(self):
        """Collects examples from all particles at all boundaries."""
        seq = self._make_mock_sequences(n_particles=3, n_units=2)
        hse = self._make_mock_hse(d_model=32)
        pot = self._make_mock_potential(complete_score=0.0)
        buffer = TwistTrainingBuffer()

        stats = await collect_twist_training_data(seq, hse, pot, buffer)

        # 3 particles × 2 boundaries = 6 examples
        assert stats["n_examples"] == 6
        assert stats["n_particles"] == 3
        assert stats["n_positive"] == 3  # complete_score=0.0 > -inf → label=1
        assert len(buffer) == 6

    @pytest.mark.asyncio
    async def test_negative_labels(self):
        """Particles failing constraint get label=0."""
        seq = self._make_mock_sequences(n_particles=2, n_units=2)
        hse = self._make_mock_hse()
        pot = self._make_mock_potential(complete_score=float("-inf"))
        buffer = TwistTrainingBuffer()

        stats = await collect_twist_training_data(seq, hse, pot, buffer)

        assert stats["n_positive"] == 0
        assert all(label == 0.0 for label in buffer.labels)

    @pytest.mark.asyncio
    async def test_skips_zero_weight_particles(self):
        """Particles with zero weight are skipped."""
        seq = self._make_mock_sequences(n_particles=3, n_units=2)
        seq.normalized_weights = np.array([0.5, 0.0, 0.5])
        hse = self._make_mock_hse()
        pot = self._make_mock_potential(complete_score=0.0)
        buffer = TwistTrainingBuffer()

        stats = await collect_twist_training_data(seq, hse, pot, buffer)

        # Only 2 particles contribute (indices 0 and 2), 2 boundaries each
        assert stats["n_examples"] == 4

    @pytest.mark.asyncio
    async def test_boundary_fracs(self):
        """Boundary fractions are correctly computed as (k+1)/K."""
        seq = self._make_mock_sequences(n_particles=1, n_units=4)
        hse = self._make_mock_hse()
        pot = self._make_mock_potential(complete_score=0.0)
        buffer = TwistTrainingBuffer()

        await collect_twist_training_data(seq, hse, pot, buffer)

        # 4 boundaries: fracs should be 0.25, 0.5, 0.75, 1.0
        expected = [0.25, 0.5, 0.75, 1.0]
        for actual, exp in zip(buffer.boundary_fracs, expected):
            assert abs(actual - exp) < 1e-6

    @pytest.mark.asyncio
    async def test_empty_sequences(self):
        """Handles empty contexts gracefully."""
        seq = MagicMock()
        seq.contexts = [[], []]
        seq.normalized_weights = np.array([0.5, 0.5])
        hse = self._make_mock_hse()
        pot = self._make_mock_potential()
        buffer = TwistTrainingBuffer()

        stats = await collect_twist_training_data(seq, hse, pot, buffer)
        assert stats["n_examples"] == 0
