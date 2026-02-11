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
            return (
                torch.empty(0, 1, device=device),
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


async def collect_twist_training_data(
    sequences,
    hidden_state_extractor,
    expensive_potential,
    buffer: TwistTrainingBuffer,
    boundary_level_labels: bool = True,
):
    """Collect twist training data from completed Block SMC results.

    For each particle, at each block boundary:
        - Extract h from the LM at the boundary position
        - Record (h, normalised_weight, label, k/K)

    When boundary_level_labels=True (default), labels are computed per-boundary
    by evaluating expensive_potential.prefix() at each boundary position. This
    gives direct supervision: "at boundary k, is the prefix still viable?"

    When boundary_level_labels=False, a single global label is computed for the
    full sequence and applied to all boundaries (legacy behavior).

    Args:
        sequences: Sequences object from Block SMC run (contexts are nested units).
        hidden_state_extractor: HiddenStateExtractor for the LM.
        expensive_potential: Coerced Φ_exp potential (operates on flat LLM tokens).
        buffer: TwistTrainingBuffer to populate.
        boundary_level_labels: If True, evaluate prefix at each boundary for
            per-boundary labels. If False, use global sequence label.

    Returns:
        Dict with collection stats: n_examples, n_positive, n_particles.
    """
    from block_smc.critic import _flatten_context
    from genlm.control.constant import EndOfSequence

    contexts = sequences.contexts
    norm_weights = sequences.normalized_weights
    n_particles = len(contexts)
    n_examples = 0
    n_positive = 0

    for i in range(n_particles):
        ctx = contexts[i]
        w_i = float(norm_weights[i])

        if w_i == 0.0 or np.isnan(w_i):
            continue

        # Separate units from EOS
        units = [u for u in ctx if isinstance(u, list)]
        if not units:
            continue

        # Pre-compute global label (used when boundary_level_labels=False,
        # and for the final boundary when boundary_level_labels=True)
        flat_all = _flatten_context(ctx)
        has_eos = any(isinstance(t, EndOfSequence) for t in flat_all)
        all_byte_tokens = [t for t in flat_all if isinstance(t, bytes)]

        global_label = None
        if not boundary_level_labels:
            if has_eos and all_byte_tokens:
                complete_score = await expensive_potential.complete(all_byte_tokens)
                global_label = 1.0 if complete_score > float("-inf") else 0.0
            else:
                prefix_score = await expensive_potential.prefix(all_byte_tokens)
                global_label = 1.0 if prefix_score > float("-inf") else 0.0

        K = len(units)

        # Extract hidden state at each block boundary
        cumulative_tokens = []
        for k, unit in enumerate(units):
            cumulative_tokens.extend(unit)
            byte_toks = [t for t in cumulative_tokens if isinstance(t, bytes)]
            if not byte_toks:
                continue

            token_ids = hidden_state_extractor.token_ids_from_bytes(byte_toks)
            if not token_ids:
                continue

            # Compute label for this boundary
            if boundary_level_labels:
                is_final = (k == K - 1)
                if is_final and has_eos:
                    # Final boundary with EOS: use complete()
                    score = await expensive_potential.complete(byte_toks)
                else:
                    # Intermediate boundary (or final without EOS): use prefix()
                    score = await expensive_potential.prefix(byte_toks)
                label = 1.0 if score > float("-inf") else 0.0
            else:
                label = global_label

            h = hidden_state_extractor.extract(token_ids, position=-1)
            boundary_frac = (k + 1) / K

            buffer.add(
                hidden_state=h,
                weight=w_i,
                label=label,
                boundary_frac=boundary_frac,
            )
            n_examples += 1
            n_positive += int(label)

    return {
        "n_examples": n_examples,
        "n_positive": n_positive,
        "n_particles": n_particles,
    }


def train_twist_step(
    twist_head: TwistHead,
    buffer: TwistTrainingBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 5,
    batch_size: int = 256,
    discount: float = 0.9,
    class_balance: bool = True,
) -> dict:
    """Train the twist head on buffered data via weighted BCE.

    Loss: L = Σ_i w_i · bce_i / Σ_i w_i

    With class_balance=True, positive-class BCE terms are scaled by n_neg/n_pos
    to compensate for class imbalance in the training buffer.

    Args:
        twist_head: The TwistHead module.
        buffer: Training data buffer.
        optimizer: Optimizer for twist_head parameters.
        device: Compute device.
        n_epochs: Number of passes over the buffer.
        batch_size: Mini-batch size.
        discount: Age discount factor.
        class_balance: If True, apply pos_weight = n_neg / n_pos to balance classes.

    Returns:
        Dict with 'loss' (final epoch average) and 'accuracy' (on training set).
    """
    if len(buffer) == 0:
        return {"loss": float("nan"), "accuracy": float("nan")}

    h, w, y, bf = buffer.to_tensors(device, discount=discount)
    n = h.shape[0]

    # Compute class balance weight
    n_pos = (y > 0.5).sum().item()
    n_neg = (y <= 0.5).sum().item()
    if class_balance and n_pos > 0 and n_neg > 0:
        pos_weight = n_neg / n_pos
    else:
        pos_weight = 1.0

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
            # Weighted BCE with class balancing
            eps = 1e-7
            bce_pos = -y_batch * torch.log(psi + eps)
            bce_neg = -(1 - y_batch) * torch.log(1 - psi + eps)
            bce = pos_weight * bce_pos + bce_neg

            # Proper normalization: weighted sum / weight sum
            loss = (w_batch * bce).sum() / (w_batch.sum() + 1e-8)

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

    return {"loss": avg_loss, "accuracy": accuracy, "n_pos": n_pos, "n_neg": n_neg, "pos_weight": pos_weight}


def adapt_twist_online(
    twist_head: TwistHead,
    buffer: TwistTrainingBuffer,
    device: torch.device,
    n_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    l2_weight: float = 0.1,
    discount: float = 1.0,
) -> dict:
    """Adapt twist head to a specific instance using particles from one SMC sweep.

    Two-phase inference: run SMC once with pretrained twist, collect particles,
    adapt twist with a few gradient steps + L2 penalty back to pretrained weights,
    then run SMC again with the adapted twist.

    The L2 penalty prevents overconfidence/diversity collapse by anchoring the
    adapted twist near the pretrained parameters.

    Args:
        twist_head: The TwistHead module (modified in-place).
        buffer: Training data from one SMC sweep on this instance.
        device: Compute device.
        n_epochs: Number of passes over the buffer (keep small: 2-5).
        batch_size: Mini-batch size.
        lr: Learning rate (smaller than pretraining LR).
        l2_weight: Weight on L2 penalty ||θ - θ_pretrained||².
        discount: Age discount (1.0 = no discount, appropriate for single-sweep data).

    Returns:
        Dict with adaptation stats.
    """
    if len(buffer) == 0:
        return {"loss": float("nan"), "adapted": False, "n_examples": 0}

    # Save pretrained weights as anchor
    anchor = {k: v.clone() for k, v in twist_head.state_dict().items()}
    optimizer = torch.optim.Adam(twist_head.parameters(), lr=lr)

    twist_head.train()
    for p in twist_head.parameters():
        p.requires_grad_(True)

    h, w, y, bf = buffer.to_tensors(device, discount=discount)
    n = h.shape[0]

    # Class balancing
    n_pos = (y > 0.5).sum().item()
    n_neg = (y <= 0.5).sum().item()
    pos_weight = n_neg / n_pos if n_pos > 0 and n_neg > 0 else 1.0

    total_loss = 0.0
    n_batches = 0
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            h_batch, w_batch, y_batch, bf_batch = h[idx], w[idx], y[idx], bf[idx]

            psi = twist_head(h_batch, bf_batch)
            eps = 1e-7
            bce_pos = -y_batch * torch.log(psi + eps)
            bce_neg = -(1 - y_batch) * torch.log(1 - psi + eps)
            bce = pos_weight * bce_pos + bce_neg
            bce_loss = (w_batch * bce).sum() / (w_batch.sum() + 1e-8)

            # L2 penalty: ||θ - θ_pretrained||²
            l2_loss = sum(
                ((p - anchor[k].to(device)) ** 2).sum()
                for k, p in twist_head.named_parameters()
            )
            loss = bce_loss + l2_weight * l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    twist_head.eval()
    for p in twist_head.parameters():
        p.requires_grad_(False)

    return {
        "loss": total_loss / max(n_batches, 1),
        "n_examples": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "adapted": True,
    }


def restore_twist_weights(twist_head: TwistHead, state_dict: dict):
    """Restore twist head to pretrained weights (undo online adaptation)."""
    twist_head.load_state_dict(state_dict)
    twist_head.eval()
    for p in twist_head.parameters():
        p.requires_grad_(False)
