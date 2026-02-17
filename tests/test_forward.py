import torch

from diffusion.forward import q_sample
from diffusion.schedules import linear_beta_schedule


def test_q_sample_shape() -> None:
    sched = linear_beta_schedule(10)
    x0 = torch.randn(4, 3, 8, 8)
    t = torch.randint(0, 10, (4,))
    xt = q_sample(x0=x0, t=t, schedule=sched)
    assert xt.shape == x0.shape
