"""One-off probe: how is the `luma` aux texture oriented w.r.t. the output frame?

`glrunner` writes the from/to textures top-down and flips y inside
`getFromColor`, then flips the framebuffer read at the end. The aux sampler is
read with a bare `texture2D(luma, uv)` — no flip — so the map may or may not
line up with the image. Content-aware mattes (edge-following draws) need this
answered, not guessed.

Run:  python experiments/exp_084_luma_matte_viewer/probe_orientation.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp_075_procedural_transition_engine"))

from engine.glrunner import GLRunner  # noqa: E402

LUMA = (REPO_ROOT.parent / "misc" / "gl-transitions" / "transitions" / "luma.glsl").read_text()

H, W = 64, 48


def main() -> None:
    r = GLRunner(W, H)
    prog = r.program("luma", LUMA)

    # from = red everywhere, to = green everywhere
    a = np.zeros((H, W, 3), np.uint8); a[..., 0] = 255
    b = np.zeros((H, W, 3), np.uint8); b[..., 1] = 255

    # aux array: row 0 (TOP of the array) = 0.0, bottom half = 1.0
    m = np.zeros((H, W, 3), np.uint8)
    m[H // 2:] = 255
    r.set_aux_map(m)

    # luma.glsl: mix(to, from, step(progress, luma)) -> luma < progress shows `to`
    out = r.render(prog, a, b, 0.5, {}, "luma")
    top = out[: H // 2].reshape(-1, 3).mean(0)
    bot = out[H // 2:].reshape(-1, 3).mean(0)
    print(f"aux array top half = 0.0, bottom half = 1.0")
    print(f"  output TOP    mean RGB = {top}  -> {'TO(green)' if top[1] > top[0] else 'FROM(red)'}")
    print(f"  output BOTTOM mean RGB = {bot}  -> {'TO(green)' if bot[1] > bot[0] else 'FROM(red)'}")
    # array row 0 is `to` (green) at the output top  => aux is NOT flipped
    aligned = top[1] > top[0]
    print(f"\nVERDICT: aux map row 0 lands at the {'TOP' if aligned else 'BOTTOM'} of the "
          f"output image -> upload arrays {'AS IS' if aligned else 'FLIPPED (arr[::-1])'}")

    # also verify the image orientation itself: from = vertical gradient
    g = np.zeros((H, W, 3), np.uint8)
    g[:, :, 0] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    out2 = r.render(prog, g, b, 0.0, {}, "luma")   # progress 0 -> pure `from`
    print(f"image identity at p=0: max|out-from| = {np.abs(out2.astype(int) - g.astype(int)).max()}")


if __name__ == "__main__":
    main()
