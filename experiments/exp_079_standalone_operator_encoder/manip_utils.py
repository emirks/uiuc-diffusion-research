"""Content-identical temporal manipulations for exp_079 "SupCon-T".

A clip's own time-reversal / time-warp shares byte-identical *content* (same pixels, same scene)
but differs only on the *operator axis* (direction, timing/easing). These are exact
same-content / different-operator pairs, manufactured for free — the one confound-valid contrastive
signal available before the factorial dataset lands. A code that separates a clip from its own
reversal provably carries information no content-only code can.

Manipulations act in PIXEL space (the LTX VAE is causal; reversing latents is NOT reversing video),
keep the fixed 121-frame length, and use nearest-frame resampling (no interpolation artifacts).

  * warp  — source_idx(i) = round((i/(F-1))**gamma * (F-1)); FIXES the endpoints (t=0->0, t=1->1)
            and bends only the interior timing. gamma=1 is the exact identity. gamma>1 = ease-in
            (slow start), gamma<1 = ease-out (fast start).
  * reverse — full frame flip. Swaps the endpoints by construction; that swap IS the direction
            operator, so reverse is deliberately NOT endpoint-preserving.
"""

import torch

# name -> (mode, gamma)
MANIPULATIONS = {
    "identity":     ("warp", 1.0),
    "reverse":      ("reverse", None),
    "ease_in_g2":   ("warp", 2.0),
    "ease_out_g05": ("warp", 0.5),
    # held-out — generalization probes only, NEVER trained on
    "warp_g3":      ("warp", 3.0),
    "warp_g033":    ("warp", 1.0 / 3.0),
    "warp_g15":     ("warp", 1.5),
    "warp_g067":    ("warp", 2.0 / 3.0),
}
TRAIN_MANIPS = ["identity", "reverse", "ease_in_g2", "ease_out_g05"]
HELDOUT_MANIPS = ["warp_g3", "warp_g033", "warp_g15", "warp_g067"]


def warp_source_indices(num_frames: int, gamma: float) -> torch.Tensor:
    """Nearest-frame source indices for a gamma time-warp; endpoints fixed."""
    t = torch.arange(num_frames, dtype=torch.float64) / max(num_frames - 1, 1)
    return torch.round(t.pow(gamma) * (num_frames - 1)).long().clamp(0, num_frames - 1)


def manipulate(frames: torch.Tensor, name: str) -> torch.Tensor:
    """frames [F, C, H, W] -> manipulated [F, C, H, W] (same length)."""
    if name not in MANIPULATIONS:
        raise KeyError(f"unknown manipulation {name!r}; known: {list(MANIPULATIONS)}")
    mode, gamma = MANIPULATIONS[name]
    if mode == "reverse":
        return torch.flip(frames, dims=[0])
    src = warp_source_indices(frames.shape[0], gamma).to(frames.device)
    return frames.index_select(0, src)


if __name__ == "__main__":
    # self-check: identity is a true no-op; reverse is an involution; warps fix endpoints.
    x = torch.randn(121, 3, 8, 6)
    assert torch.equal(manipulate(x, "identity"), x), "identity must be exact no-op"
    assert torch.equal(manipulate(manipulate(x, "reverse"), "reverse"), x), "reverse must be involution"
    for name, (mode, g) in MANIPULATIONS.items():
        y = manipulate(x, name)
        assert y.shape == x.shape, f"{name}: shape changed"
        if mode == "warp":
            assert torch.equal(y[0], x[0]) and torch.equal(y[-1], x[-1]), f"{name}: endpoints not fixed"
    # reverse swaps endpoints
    r = manipulate(x, "reverse")
    assert torch.equal(r[0], x[-1]) and torch.equal(r[-1], x[0]), "reverse must swap endpoints"
    print("[manip_utils] all self-checks PASS:", list(MANIPULATIONS))
