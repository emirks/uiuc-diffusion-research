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
  * segment_swap / block_reversal — PROBE-ONLY permutations (see below).

PERMUTATION vs RESAMPLING — this distinction decides which probes are content-controlled, and it
is the reason the pre-registered bars were revised before the first read:

  * `reverse`, `segment_swap`, `block_reversal` are PERMUTATIONS of the frame multiset. Any
    order-invariant (permutation-invariant) encoder is EXACTLY invariant to them, so ANY nonzero
    response is order information by construction — a content-only code cannot fake it.
  * `warp` RESAMPLES WITH REPETITION: it changes the frame multiset, so appearance statistics move
    and an order-blind code responds (measured: a mean-over-frames encoder scores gamma-monotonicity
    rho=0.946). The gamma probes are therefore NOT content-controlled and are report-only.
"""

import torch

# name -> (mode, param)
MANIPULATIONS = {
    "identity":     ("warp", 1.0),
    "reverse":      ("reverse", None),
    "ease_in_g2":   ("warp", 2.0),
    "ease_out_g05": ("warp", 0.5),
    # held-out gamma warps — REPORT-ONLY diagnostics (not content-controlled), never trained on
    "warp_g3":      ("warp", 3.0),
    "warp_g033":    ("warp", 1.0 / 3.0),
    "warp_g15":     ("warp", 1.5),
    "warp_g067":    ("warp", 2.0 / 3.0),
    # held-out PERMUTATIONS — the generalization GATE. Probe-only, never trained on; multiset
    # preserved exactly, so they inherit `reverse`'s content-control by construction.
    "segment_swap":   ("segment_swap", 2),
    "block_reversal": ("block_reversal", 4),
}
TRAIN_MANIPS = ["identity", "reverse", "ease_in_g2", "ease_out_g05"]
GAMMA_MANIPS = ["warp_g3", "warp_g033", "warp_g15", "warp_g067"]        # report-only
PERM_MANIPS = ["segment_swap", "block_reversal"]                          # the held-out GATE
HELDOUT_MANIPS = GAMMA_MANIPS + PERM_MANIPS


def warp_source_indices(num_frames: int, gamma: float) -> torch.Tensor:
    """Nearest-frame source indices for a gamma time-warp; endpoints fixed."""
    t = torch.arange(num_frames, dtype=torch.float64) / max(num_frames - 1, 1)
    return torch.round(t.pow(gamma) * (num_frames - 1)).long().clamp(0, num_frames - 1)


def permutation_indices(num_frames: int, name: str) -> torch.Tensor:
    """Source indices for a PURE PERMUTATION manipulation (frame multiset exactly preserved)."""
    mode, param = MANIPULATIONS[name]
    if mode == "reverse":
        return torch.arange(num_frames - 1, -1, -1)
    if mode == "segment_swap":
        # split into `param` contiguous segments and rotate their order: [S1,S0] for param=2.
        sizes = [num_frames // param + (1 if i < num_frames % param else 0) for i in range(param)]
        segs, start = [], 0
        for s in sizes:
            segs.append(torch.arange(start, start + s)); start += s
        return torch.cat(segs[1:] + segs[:1])
    if mode == "block_reversal":
        # split into `param` contiguous blocks and reverse the frames WITHIN each block.
        sizes = [num_frames // param + (1 if i < num_frames % param else 0) for i in range(param)]
        out, start = [], 0
        for s in sizes:
            out.append(torch.arange(start + s - 1, start - 1, -1)); start += s
        return torch.cat(out)
    raise ValueError(f"{name} is not a permutation manipulation")


def manipulate(frames: torch.Tensor, name: str) -> torch.Tensor:
    """frames [F, C, H, W] -> manipulated [F, C, H, W] (same length)."""
    if name not in MANIPULATIONS:
        raise KeyError(f"unknown manipulation {name!r}; known: {list(MANIPULATIONS)}")
    mode, param = MANIPULATIONS[name]
    if mode == "warp":
        src = warp_source_indices(frames.shape[0], param)
    else:
        src = permutation_indices(frames.shape[0], name)
    return frames.index_select(0, src.to(frames.device))


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

    # THE load-bearing property: reverse + the probe permutations preserve the frame multiset
    # EXACTLY (so an order-invariant encoder cannot respond to them), while gamma warps do NOT.
    ref = torch.arange(121)
    for name in ["reverse"] + PERM_MANIPS:
        idx = permutation_indices(121, name)
        assert torch.equal(idx.sort().values, ref), f"{name} must be a PERMUTATION of the frames"
        assert not torch.equal(idx, ref), f"{name} must actually reorder the frames"
        # an order-invariant statistic (mean over frames) must be exactly unchanged
        assert torch.allclose(manipulate(x, name).mean(0), x.mean(0), atol=1e-6), \
            f"{name}: order-invariant statistic moved — not a pure permutation"
    for name in GAMMA_MANIPS:
        idx = warp_source_indices(121, MANIPULATIONS[name][1])
        assert not torch.equal(idx.sort().values, ref), \
            f"{name} is expected to RESAMPLE (change the multiset) — that is why it is report-only"
    print("[manip_utils] all self-checks PASS")
    print(f"  train        {TRAIN_MANIPS}")
    print(f"  gamma (report-only, NOT content-controlled) {GAMMA_MANIPS}")
    print(f"  permutations (held-out GATE, multiset-preserving) {PERM_MANIPS}")
