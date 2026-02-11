"""Tests for TwistHead and TwistTrainingBuffer."""

import torch
import numpy as np
import pytest
from block_smc.twist import TwistHead, TwistTrainingBuffer, train_twist_step


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
