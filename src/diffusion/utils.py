from __future__ import annotations

import torch


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract a[t] for a batch of indices t and reshape to broadcast over x."""
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))
