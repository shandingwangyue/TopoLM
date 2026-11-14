"""Causal domain routing with explicit padding validity."""

import torch
from torch import nn


class DomainRouter(nn.Module):
    """Pointwise routing with a fixed initial-prefix fallback.

    Global top-k can leak future information by displacing earlier keys.
    Prefix fallback keeps each key's selection invariant under continuation.
    """

    def __init__(self, threshold: float = 1e-4, min_active_k: int = 4):
        super().__init__()
        if not 0 <= threshold <= 1 or min_active_k < 1:
            raise ValueError("threshold must be in [0, 1] and min_active_k >= 1")
        self.threshold = threshold
        self.min_active_k = min_active_k

    @torch.no_grad()
    def route_active_tokens(self, semantic_x, topo_space, active_domain_idx,
                            *, return_mask=False):
        batch_size, seq_len, _ = semantic_x.shape
        if batch_size == 0 or seq_len == 0:
            raise ValueError("Cannot route an empty batch or sequence")
        probs = topo_space.membership_prob(semantic_x, active_domain_idx)
        selected = probs >= self.threshold
        selected[:, :min(self.min_active_k, seq_len)] = True
        indices = [row.nonzero(as_tuple=True)[0] for row in selected]
        max_k = max(idx.numel() for idx in indices)
        padded = torch.zeros(batch_size, max_k, dtype=torch.long, device=semantic_x.device)
        valid = torch.zeros_like(padded, dtype=torch.bool)
        for batch, idx in enumerate(indices):
            padded[batch, :idx.numel()] = idx
            valid[batch, :idx.numel()] = True
        if return_mask:
            return padded, probs, valid
        # Original two-value API remains available to older callers.
        for batch, idx in enumerate(indices):
            padded[batch, idx.numel():] = idx[-1]
        return padded, probs
