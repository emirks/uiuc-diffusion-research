import torch

from diffusion.schedules import linear_beta_schedule


def test_linear_schedule_shapes() -> None:
    sched = linear_beta_schedule(100)
    assert sched.betas.shape == (100,)
    assert sched.alpha_bars.shape == (100,)
    assert torch.all(sched.betas > 0)
