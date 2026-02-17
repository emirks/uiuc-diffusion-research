from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BetaSchedule:
    betas: torch.Tensor  # [T]

    @property
    def alphas(self) -> torch.Tensor:
        return 1.0 - self.betas

    @property
    def alpha_bars(self) -> torch.Tensor:
        return torch.cumprod(self.alphas, dim=0)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2, device: torch.device | None = None) -> BetaSchedule:
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=torch.float32)
    return BetaSchedule(betas=betas)
