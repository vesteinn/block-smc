"""Learned twist function: MLP head on LM hidden states.

Components:
    TwistHead         — nn.Module predicting P(constraint satisfied | prefix)
    TwistTrainingBuffer — replay buffer collecting (h, w, y) from SMC trajectories
    train_twist_step  — weighted BCE training loop
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field


class TwistHead(nn.Module):
    """MLP predicting P(constraint satisfied | LM hidden state at boundary).

    Architecture:
        [h; k/K] → Linear(d_model+1, hidden) → ReLU → Linear(hidden, hidden) → ReLU → Linear(hidden, 1) → Sigmoid

    The input is the LM last-layer hidden state h concatenated with the
    normalised boundary fraction k/K. Output is ψ ∈ (0, 1).

    Args:
        d_model: Hidden dimension of the language model.
        hidden_dim: Width of the MLP hidden layers. Default 256.
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        # +1 for boundary fraction k/K
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, boundary_frac: torch.Tensor) -> torch.Tensor:
        """Compute twist value ψ(h, k/K).

        Args:
            h: Hidden state(s), shape (..., d_model).
            boundary_frac: Normalised boundary position k/K, shape (..., 1) or (...,).

        Returns:
            ψ values in (0, 1), same batch shape as input.
        """
        if boundary_frac.dim() < h.dim():
            boundary_frac = boundary_frac.unsqueeze(-1)
        x = torch.cat([h, boundary_frac], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)

    def log_psi(self, h: torch.Tensor, boundary_frac: torch.Tensor) -> torch.Tensor:
        """Compute log ψ(h, k/K) with numerical stability.

        Uses log-sigmoid to avoid log(0).
        """
        if boundary_frac.dim() < h.dim():
            boundary_frac = boundary_frac.unsqueeze(-1)
        x = torch.cat([h, boundary_frac], dim=-1)
        logits = self.net(x).squeeze(-1)
        return nn.functional.logsigmoid(logits)


@dataclass
class TwistTrainingBuffer:
    """Replay buffer collecting training data from SMC trajectories.

    After each Block SMC sweep, we store per boundary k, per particle i:
        (h^(i)_{τ_k}, w^(i), y^(i), k/K)
    where:
        h = hidden state at boundary position
        w = self-normalised importance weight
        y = 1 if particle survived constraint (Φ > 0), else 0
        k/K = normalised boundary index

    Supports age-based discounting: older data weighted by γ^age.
    """

    hidden_states: list[torch.Tensor] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    boundary_fracs: list[float] = field(default_factory=list)
    ages: list[int] = field(default_factory=list)
    _current_age: int = 0

    def add(
        self,
        hidden_state: torch.Tensor,
        weight: float,
        label: float,
        boundary_frac: float,
    ):
        """Add a single training example."""
        self.hidden_states.append(hidden_state.detach().cpu())
        self.weights.append(weight)
        self.labels.append(label)
        self.boundary_fracs.append(boundary_frac)
        self.ages.append(self._current_age)

    def add_batch(
        self,
        hidden_states: torch.Tensor,
        weights: np.ndarray,
        labels: np.ndarray,
        boundary_frac: float,
    ):
        """Add a batch of examples from one boundary of one SMC sweep.

        Args:
            hidden_states: (n_particles, d_model)
            weights: (n_particles,) self-normalised importance weights
            labels: (n_particles,) binary labels (survived constraint)
            boundary_frac: k/K for this boundary
        """
        for i in range(hidden_states.shape[0]):
            self.add(hidden_states[i], float(weights[i]), float(labels[i]), boundary_frac)

    def advance_age(self):
        """Call after each EM iteration to age existing data."""
        self._current_age += 1

    def to_tensors(
        self, device: torch.device, discount: float = 0.9
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert buffer to tensors with age-based discounting.

        Args:
            device: Target device.
            discount: γ for age discounting. Older data weighted by γ^(current_age - entry_age).

        Returns:
            (hidden_states, weights, labels, boundary_fracs) as tensors.
        """
        if not self.hidden_states:
            d = self.hidden_states[0].shape[-1] if self.hidden_states else 1
            return (
                torch.empty(0, d, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device),
            )

        h = torch.stack(self.hidden_states).to(device)
        w = torch.tensor(self.weights, device=device)
        y = torch.tensor(self.labels, device=device)
        bf = torch.tensor(self.boundary_fracs, device=device)

        # Apply age discount
        age_offsets = torch.tensor(
            [self._current_age - a for a in self.ages], device=device, dtype=torch.float
        )
        age_discount = discount ** age_offsets
        w = w * age_discount

        return h, w, y, bf

    def clear(self):
        self.hidden_states.clear()
        self.weights.clear()
        self.labels.clear()
        self.boundary_fracs.clear()
        self.ages.clear()
        self._current_age = 0

    def __len__(self) -> int:
        return len(self.hidden_states)


def train_twist_step(
    twist_head: TwistHead,
    buffer: TwistTrainingBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 5,
    batch_size: int = 256,
    discount: float = 0.9,
) -> dict:
    """Train the twist head on buffered data via weighted BCE.

    Loss: L = -Σ_i w_i [y_i log ψ(h_i, k_i) + (1-y_i) log(1 - ψ(h_i, k_i))]

    Args:
        twist_head: The TwistHead module.
        buffer: Training data buffer.
        optimizer: Optimizer for twist_head parameters.
        device: Compute device.
        n_epochs: Number of passes over the buffer.
        batch_size: Mini-batch size.
        discount: Age discount factor.

    Returns:
        Dict with 'loss' (final epoch average) and 'accuracy' (on training set).
    """
    if len(buffer) == 0:
        return {"loss": float("nan"), "accuracy": float("nan")}

    h, w, y, bf = buffer.to_tensors(device, discount=discount)
    n = h.shape[0]

    twist_head.train()
    total_loss = 0.0
    n_batches = 0

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            h_batch = h[idx]
            w_batch = w[idx]
            y_batch = y[idx]
            bf_batch = bf[idx]

            psi = twist_head(h_batch, bf_batch)
            # Weighted BCE
            eps = 1e-7
            bce = -(
                y_batch * torch.log(psi + eps) + (1 - y_batch) * torch.log(1 - psi + eps)
            )
            loss = (w_batch * bce).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    twist_head.eval()

    # Compute accuracy on full buffer
    with torch.no_grad():
        psi_all = twist_head(h, bf)
        predictions = (psi_all > 0.5).float()
        accuracy = (predictions == y).float().mean().item()
        avg_loss = total_loss / max(n_batches, 1)

    return {"loss": avg_loss, "accuracy": accuracy}
