from __future__ import annotations

import torch

from .schedules import BetaSchedule


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: BetaSchedule, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Sample x_t ~ q(x_t | x_0) for a batch of timesteps t.

    x0: [B, ...]
    t:  [B] integer timesteps in [0, T-1]
    """
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bars = schedule.alpha_bars.to(device=x0.device)
    a_bar_t = alpha_bars.gather(0, t).view(-1, *([1] * (x0.ndim - 1)))

    return torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * noise
