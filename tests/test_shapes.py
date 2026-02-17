import torch


def test_sanity_shape() -> None:
    x = torch.randn(4, 3, 32, 32)
    assert x.shape == (4, 3, 32, 32)
