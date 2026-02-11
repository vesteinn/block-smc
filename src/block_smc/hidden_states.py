"""Hidden state extraction from LM at boundary positions.

Components:
    HiddenStateExtractor — extracts h ∈ R^{d_model} from LM forward pass
    HiddenStateCache     — per-particle cache of hidden states at boundaries
"""

import torch
import numpy as np
from typing import Optional


class HiddenStateExtractor:
    """Extracts last-layer hidden states from a language model.

    Works with genlm-control's PromptedLLM. Accesses the underlying
    HuggingFace model via llm.model.model and runs a forward pass
    with output_hidden_states=True.

    Args:
        llm: A PromptedLLM instance (genlm-control).
    """

    def __init__(self, llm):
        self.llm = llm
        self._hf_model = None
        self._tokenizer = None
        self._device = None
        self._hidden_dim = None

    @property
    def hf_model(self):
        if self._hf_model is None:
            self._hf_model = self.llm.model.model
        return self._hf_model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.llm.model.tokenizer
        return self._tokenizer

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.hf_model.parameters()).device
        return self._device

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            self._hidden_dim = self.hf_model.config.hidden_size
        return self._hidden_dim

    def extract(self, token_ids: list[int], position: int = -1) -> torch.Tensor:
        """Extract hidden state at a given position from a token sequence.

        Args:
            token_ids: List of token IDs (integers).
            position: Position index to extract from. Default -1 (last token).

        Returns:
            Tensor of shape (hidden_dim,) — the last-layer hidden state.
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        with torch.no_grad():
            outputs = self.hf_model(input_ids, output_hidden_states=True)
        # Last layer hidden state at the specified position, cast to float32
        # (LM may use bfloat16 but twist head trains in float32)
        return outputs.hidden_states[-1][0, position, :].detach().float()

    def extract_batch(
        self, token_ids_list: list[list[int]], position: int = -1
    ) -> torch.Tensor:
        """Extract hidden states for a batch of token sequences.

        All sequences are padded to the same length (left-padded).

        Args:
            token_ids_list: List of token ID lists.
            position: Position index to extract from. Default -1 (last token).

        Returns:
            Tensor of shape (batch_size, hidden_dim).
        """
        if not token_ids_list:
            return torch.empty(0, self.hidden_dim, device=self.device)

        max_len = max(len(ids) for ids in token_ids_list)
        pad_id = self.tokenizer.pad_token_id or 0

        # Left-pad so position=-1 always refers to the last real token
        padded = []
        attention_masks = []
        for ids in token_ids_list:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)

        with torch.no_grad():
            outputs = self.hf_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        return outputs.hidden_states[-1][:, position, :].detach().float()

    def token_ids_from_bytes(self, byte_tokens: list[bytes]) -> list[int]:
        """Convert byte tokens (from genlm-control) to token IDs.

        genlm-control works with byte tokens. This converts them back
        to integer IDs for the HF model forward pass.

        Args:
            byte_tokens: Flat list of byte tokens from genlm-control.

        Returns:
            List of integer token IDs.
        """
        text = b"".join(byte_tokens).decode("utf-8", errors="replace")
        return self.tokenizer.encode(text, add_special_tokens=False)


class HiddenStateCache:
    """Cache of hidden states indexed by (particle_id, boundary_index).

    Used during a single Block SMC sweep to avoid recomputing hidden states.
    Cleared or remapped after resampling.
    """

    def __init__(self):
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def store(self, particle_id: int, boundary_idx: int, hidden_state: torch.Tensor):
        self._cache[(particle_id, boundary_idx)] = hidden_state

    def get(
        self, particle_id: int, boundary_idx: int
    ) -> Optional[torch.Tensor]:
        return self._cache.get((particle_id, boundary_idx))

    def remap_ancestors(self, ancestor_map: dict[int, int]):
        """After resampling, remap cache entries from ancestor particles.

        Args:
            ancestor_map: {new_particle_id: ancestor_particle_id}
        """
        new_cache = {}
        for (pid, bidx), h in self._cache.items():
            # Find all new particles that descended from this one
            for new_pid, ancestor_pid in ancestor_map.items():
                if ancestor_pid == pid:
                    new_cache[(new_pid, bidx)] = h
        self._cache = new_cache

    def clear(self):
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
