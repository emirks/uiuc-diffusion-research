"""3D-plausible transition operators: camera path × atmosphere × focus × blend.

Every element here is chosen because it is *optically motivated* rather than
decorative, which is what separates this family from the 2D shader bank:

* **parallax** comes from real per-pixel depth, so foreground and background
  separate correctly under camera motion;
* **fog** is exponential extinction along the view ray — the physically correct
  law, and the reason a depth-ramped haze reads as real distance;
* **defocus** blurs by circle-of-confusion, which depends on depth and focus
  distance, so a rack focus lands on the right plane;
* **handheld jitter** exists because a perfectly rigid virtual camera reads as CGI.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import cv2
import numpy as np

from . import cameras
from .depth import to_view_depth

EASINGS = {
    "linear":       lambda u: u,
    "smoothstep":   lambda u: u * u * (3 - 2 * u),
    "smootherstep": lambda u: u ** 3 * (u * (6 * u - 15) + 10),
    "in_cubic":     lambda u: u ** 3,
    "out_cubic":    lambda u: 1 - (1 - u) ** 3,
    "in_out_cubic": lambda u: np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2),
    "in_out_sine":  lambda u: -(np.cos(np.pi * u) - 1) / 2,
    "in_out_quart": lambda u: np.where(u < 0.5, 8 * u ** 4, 1 - (-2 * u + 2) ** 4 / 2),
    "accel":        lambda u: u ** 2 * (3 - 2 * u) ** 0.5,
}

# The CAMERA path may only use easings with **zero velocity at both endpoints**.
# The middle samples the open interval, so an easing that is still moving at u=1
# leaves the camera off-rest on the last rendered frame and the join to the end
# bucket becomes a jump cut. Measured: in_out_cubic gave a seam ratio of 0.46
# while linear/accel/in_cubic ran 20-160x worse. This is also just correct
# cinematography — real cameras do not start or stop instantaneously.
PATH_EASINGS = ("smootherstep", "in_out_cubic", "in_out_sine", "in_out_quart")


@dataclasses.dataclass
class Operator3D:
    path: str
    amplitude: float
    sign: int
    pivot: float
    fovy: float
    axis: int
    turns: float
    easing: str             # camera path — restricted to PATH_EASINGS
    blend_easing: str       # blend timing — unrestricted
    handheld: float
    dolly_zoom: float       # 0 = off; 1 = full Vertigo (subject size held constant)
    motion_blur: int        # sub-frames accumulated per output frame (1 = off)
    dissolve: str           # none | fbm | worley | plane | sphere
    dissolve_freq: float
    dissolve_edge: float
    blend: str              # crossfade | depth_wipe
    blend_window: float
    wipe_dir: int           # +1 near-to-far, -1 far-to-near
    wipe_band: float
    fog: float              # peak extinction density (0 = off)
    fog_color: tuple
    focus: float            # peak circle-of-confusion in px (0 = off)
    depth_near: float
    depth_far: float
    depth_gamma: float
    seed: int

    def describe(self) -> str:
        bits = [f"{self.path}{'+' if self.sign > 0 else '-'}",
                f"amp={self.amplitude:.2f}", f"fov={np.degrees(self.fovy):.0f}°",
                f"depth={self.depth_near:.1f}-{self.depth_far:.1f}",
                f"ease={self.easing}", self.blend]
        if self.path in ("orbit", "crane", "spiral"):
            bits.append(f"pivot={self.pivot:.1f}")
        if self.handheld > 0:
            bits.append(f"handheld={self.handheld:.2f}")
        if self.dolly_zoom > 0:
            bits.append(f"dollyzoom={self.dolly_zoom:.2f}")
        if self.motion_blur > 1:
            bits.append(f"mblur={self.motion_blur}x")
        if self.dissolve != "none":
            bits.append(f"dissolve={self.dissolve}@{self.dissolve_freq:.1f}")
        if self.fog > 0:
            bits.append(f"fog={self.fog:.2f}")
        if self.focus > 0:
            bits.append(f"rackfocus={self.focus:.0f}px")
        return "  ".join(bits)

    def short(self) -> str:
        s = f"{self.path}_{self.blend}"
        if self.dissolve != "none":
            s += f"_{self.dissolve}"
        if self.dolly_zoom > 0:
            s += "_dz"
        if self.fog > 0:
            s += "_fog"
        if self.focus > 0:
            s += "_focus"
        return s


def sample_operator(rng: random.Random) -> Operator3D:
    path = rng.choice(sorted(cameras.PATHS))
    fog = rng.choice([0.0, 0.0, 0.0, rng.uniform(0.5, 2.4)])
    focus = rng.choice([0.0, 0.0, 0.0, rng.uniform(4.0, 16.0)])
    return Operator3D(
        path=path,
        amplitude=rng.uniform(0.55, 1.6),
        sign=rng.choice([-1, 1]),
        pivot=rng.uniform(1.4, 3.2),
        fovy=np.radians(rng.uniform(35.0, 75.0)),
        axis=rng.randrange(2),
        turns=rng.uniform(0.5, 1.8),
        easing=rng.choice(PATH_EASINGS),
        blend_easing=rng.choice(sorted(EASINGS)),
        handheld=rng.choice([0.0, rng.uniform(0.2, 1.0)]),
        dolly_zoom=(rng.choice([0.0, rng.uniform(0.5, 1.0)])
                    if path in ("dolly", "spiral") else 0.0),
        motion_blur=rng.choice([1, 1, 3, 4]),
        dissolve=rng.choice(["none", "none", "fbm", "worley", "plane", "sphere"]),
        dissolve_freq=rng.uniform(0.6, 3.0),
        dissolve_edge=rng.uniform(0.04, 0.22),
        blend=rng.choice(["crossfade", "crossfade", "depth_wipe"]),
        blend_window=rng.uniform(0.18, 0.75),
        wipe_dir=rng.choice([-1, 1]),
        wipe_band=rng.uniform(0.08, 0.35),
        fog=fog,
        fog_color=tuple(round(rng.uniform(0.55, 1.0), 3) for _ in range(3)),
        focus=focus,
        depth_near=1.0,
        depth_far=rng.uniform(2.2, 5.5),
        depth_gamma=rng.uniform(0.6, 1.4),
        seed=rng.randrange(1 << 30),
    )


# --------------------------------------------------------------------------
# Optical pre-render modifiers (applied to the layer texture, i.e. to the scene)
# --------------------------------------------------------------------------

def apply_fog(frame: np.ndarray, view_z: np.ndarray, density: float,
              color: tuple) -> np.ndarray:
    """Beer-Lambert extinction along the view ray: T = exp(-density * z)."""
    if density <= 0:
        return frame
    t = np.exp(-density * (view_z - view_z.min()))[..., None]
    fog_rgb = np.array(color, dtype=np.float32)[None, None, :] * 255.0
    return np.clip(frame.astype(np.float32) * t + fog_rgb * (1.0 - t), 0, 255
                   ).astype(np.uint8)


def apply_defocus(frame: np.ndarray, view_z: np.ndarray, focus_z: float,
                  max_coc: float) -> np.ndarray:
    """Depth-of-field by circle of confusion, approximated with 4 blur bands."""
    if max_coc <= 0.5:
        return frame
    coc = np.abs(1.0 / np.maximum(view_z, 1e-3) - 1.0 / max(focus_z, 1e-3))
    coc = coc / (coc.max() + 1e-8) * max_coc
    out = frame.astype(np.float32)
    levels = [max_coc * f for f in (0.35, 0.65, 1.0)]
    prev = 0.0
    for lv in levels:
        k = int(max(3, round(lv))) | 1
        blurred = cv2.GaussianBlur(frame, (k, k), 0).astype(np.float32)
        m = np.clip((coc - prev) / max(lv - prev, 1e-6), 0, 1)[..., None]
        out = out * (1 - m) + blurred * m
        prev = lv
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Compositing
# --------------------------------------------------------------------------

def _fill_holes(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Normalised convolution (push-pull): spread valid colour into disocclusions.

    Disocclusions are where the camera move revealed geometry neither layer has.
    Blurring numerator and denominator together and dividing pulls surrounding
    valid colour inward at progressively coarser scales — cheap, and adequate
    here because the incoming layer already covers most of the hole.
    """
    den2 = den.reshape(den.shape[0], den.shape[1])          # (H, W)
    out = np.zeros_like(num)
    ok = den2 > 1e-3
    np.divide(num, den2[..., None], out=out, where=ok[..., None])
    if ok.all():
        return out
    for k in (9, 31, 81):
        nb = cv2.GaussianBlur(num, (k, k), 0)
        db = cv2.GaussianBlur(den2, (k, k), 0)
        need = (~ok) & (db > 1e-4)
        if need.any():
            out[need] = nb[need] / db[need][:, None]
        ok = ok | need
        if ok.all():
            break
    return out


def composite(rgba_a: np.ndarray, rgba_b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Blend two rendered layers by weight `w` (0 = A, 1 = B), honouring alpha."""
    a_rgb = rgba_a[..., :3].astype(np.float32)
    b_rgb = rgba_b[..., :3].astype(np.float32)
    a_a = (rgba_a[..., 3:4].astype(np.float32) / 255.0) * (1.0 - w)
    b_a = (rgba_b[..., 3:4].astype(np.float32) / 255.0) * w
    num = a_rgb * a_a + b_rgb * b_a
    den = a_a + b_a
    return np.clip(_fill_holes(num, den), 0, 255).astype(np.uint8)


def world_positions(view_z: np.ndarray, fovy: float) -> np.ndarray:
    """Unproject a depth map to scene-space (H, W, 3) positions."""
    h, w = view_z.shape
    t = np.tan(fovy / 2.0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs / (w - 1)) * 2 - 1
    ny = (ys / (h - 1)) * 2 - 1
    return np.stack([nx * t * (w / h) * view_z, ny * t * view_z, -view_z], -1)


def dissolve_field(kind: str, P: np.ndarray, freq: float, seed: int) -> np.ndarray:
    """Scalar field in [0,1] evaluated at **scene-space** positions.

    This is the cheapest operator on the list that is genuinely three-dimensional.
    Because the field is sampled at world positions rather than screen UVs, the
    dissolve pattern sticks to surfaces: it parallaxes with the camera, foreshortens
    on oblique surfaces, and stays put on objects. A screen-space noise dissolve
    slides over the image and instantly reads as an overlay.
    """
    rng = np.random.default_rng(seed)
    if kind == "plane":                      # a real plane sweeping through the scene
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        f = P @ n.astype(np.float32)
    elif kind == "sphere":                   # an expanding shell about a scene point
        c = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1),
                      -rng.uniform(1.2, 3.0)], dtype=np.float32)
        f = np.linalg.norm(P - c, axis=-1)
    else:                                    # fbm / worley, on a precomputed volume
        res = 48
        vol = rng.random((res, res, res)).astype(np.float32)
        if kind == "worley":
            pts = rng.random((24, 3)).astype(np.float32) * res
            gg = np.stack(np.meshgrid(*[np.arange(res, dtype=np.float32)] * 3,
                                      indexing="ij"), -1)
            vol = np.min(np.linalg.norm(gg[None] - pts[:, None, None, None], axis=-1),
                         axis=0)
        else:
            for _ in range(3):               # cheap smoothing -> value-noise fBm
                vol = (vol + np.roll(vol, 1, 0) + np.roll(vol, -1, 0)
                       + np.roll(vol, 1, 1) + np.roll(vol, -1, 1)
                       + np.roll(vol, 1, 2) + np.roll(vol, -1, 2)) / 7.0
        from scipy.ndimage import map_coordinates
        c = (P * freq * 6.0) % res
        f = map_coordinates(vol, [c[..., 0].ravel(), c[..., 1].ravel(),
                                  c[..., 2].ravel()], order=1, mode="grid-wrap")
        f = f.reshape(P.shape[:2])
    f = f - f.min()
    return (f / (f.max() + 1e-8)).astype(np.float32)


def blend_weight(op: Operator3D, u: float, disp_b: np.ndarray) -> np.ndarray:
    """Per-pixel weight of layer B at transition progress `u`."""
    if op.blend == "depth_wipe":
        # Reveal B in depth order: near surfaces first, or far surfaces first.
        d = disp_b if op.wipe_dir > 0 else (1.0 - disp_b)
        thr = u * (1.0 + 2 * op.wipe_band) - op.wipe_band
        return np.clip((d - (1.0 - thr - op.wipe_band)) / (2 * op.wipe_band + 1e-6),
                       0, 1)[..., None].astype(np.float32)
    hw = max(op.blend_window, 1e-3) / 2.0
    return np.float32(np.clip((u - 0.5 + hw) / (2 * hw), 0.0, 1.0)).reshape(1, 1, 1)


# --------------------------------------------------------------------------
# The renderer driver
# --------------------------------------------------------------------------

def render_transition(renderer, op: Operator3D, start9: np.ndarray,
                      end9: np.ndarray, disp_a: np.ndarray, disp_b: np.ndarray,
                      n_middle: int) -> np.ndarray:
    """Assemble start9 + procedurally rendered middle + end9.

    The two layers ride one continuous camera trajectory: layer A leaves from
    rest, layer B arrives at rest, and the excursion is offset by the full travel
    so the two halves join into a single move.
    """
    za = to_view_depth(disp_a, op.depth_near, op.depth_far, op.depth_gamma)
    zb = to_view_depth(disp_b, op.depth_near, op.depth_far, op.depth_gamma)

    frame_a, frame_b = start9[-1], end9[0]

    path = cameras.PATHS[op.path]
    travel = op.amplitude * cameras.PATH_SCALE[op.path] * op.sign
    ease = EASINGS[op.easing]
    bease = EASINGS[op.blend_easing]
    jitter = (cameras.handheld(op.seed, n_middle, op.handheld)
              if op.handheld > 0 else None)

    # World-space dissolve fields, evaluated once at the scene positions.
    fld_a = fld_b = None
    if op.dissolve != "none":
        fld_a = dissolve_field(op.dissolve, world_positions(za, op.fovy),
                               op.dissolve_freq, op.seed)
        fld_b = dissolve_field(op.dissolve, world_positions(zb, op.fovy),
                               op.dissolve_freq, op.seed + 1)

    # Subject depth for the dolly-zoom constraint: hold the central subject's
    # projected size constant while translating, so the background rushes and the
    # subject does not. Geometrically impossible under any 2D warp, which is
    # exactly why it reads as unmistakably three-dimensional.
    h, w = za.shape
    z_subj = float(np.median(za[h // 3: 2 * h // 3, w // 3: 2 * w // 3]))

    mb = max(1, op.motion_blur)
    du = 1.0 / (n_middle + 1)

    def layer(frame, z, fld, s, e, eb, jit, is_a):
        """Render one layer at camera excursion `s` (camera clock `e`, blend clock `eb`)."""
        v, sh = path(s, pivot=op.pivot, axis=op.axis, turns=op.turns)
        if jit is not None:
            v = cameras.apply_handheld(v, jit)
        fov = op.fovy
        if op.dolly_zoom > 0 and op.path in ("dolly", "spiral"):
            ratio = max((z_subj - s) / max(z_subj, 1e-3), 0.15)
            fov = 2.0 * np.arctan(np.tan(op.fovy / 2.0)
                                  * (1 - op.dolly_zoom + op.dolly_zoom * ratio))
        f = frame
        if op.fog > 0:
            # The fog must RAMP, not sit at full density: the outgoing layer hazes
            # over as it recedes and the incoming layer clears as it arrives, so
            # both buckets are reached fog-free. Applying it at constant density
            # (as this first did) puts a fully fogged frame next to an unfogged
            # bucket — a ~170 MAE jump cut, and the single worst seam in the run.
            f = apply_fog(f, z, op.fog * (eb if is_a else 1.0 - eb), op.fog_color)
        if op.focus > 0:
            span = op.depth_far - op.depth_near
            f = (apply_defocus(f, z, op.depth_near + e * span, op.focus * e) if is_a
                 else apply_defocus(f, z, op.depth_far - e * span, op.focus * (1 - e)))
        if fld is not None:
            # A erodes away as the transition runs; B accretes back in.
            # The threshold must sweep the FULL field range plus the edge band, or
            # the layer is already partly dissolved on the first rendered frame and
            # the join to the bucket becomes a jump. (Measured: getting this wrong
            # cost a seam ratio of ~190 vs ~1.8 without a dissolve.) It rides the
            # BLEND clock, which closes exactly inside the rendered range.
            ed = op.dissolve_edge
            tau = -ed + (eb if is_a else 1.0 - eb) * (1.0 + 2 * ed)
            m = np.clip((fld - tau) / (2 * ed) + 0.5, 0.0, 1.0)
            f = np.concatenate([f, (m * 255).astype(np.uint8)[..., None]], axis=2)
        return renderer.render(f, z, v, fovy=fov, shear=sh, shear_ref=op.pivot)

    mid = []
    for k in range(n_middle):
        # Camera samples the open interval; the BLEND is normalised to close fully
        # inside the rendered range, so the last middle frame is pure B and the
        # first is pure A. Without this the join carries whatever fraction of the
        # outgoing layer the blend had not yet retired.
        u = (k + 1) * du
        vb_ = k / max(n_middle - 1, 1)
        acc = None
        for j in range(mb):
            # 180-degree shutter: accumulate sub-frames across half the frame
            # interval and average. The largest realism gain available on a camera
            # move, and it also softens disocclusion smear.
            uj = u + (j / mb - 0.5) * du * 0.5
            e = float(ease(np.clip(np.float64(uj), 0.0, 1.0)))
            eb = float(bease(np.float64(vb_)))
            jit = jitter[k] if jitter is not None else None
            ra = layer(frame_a, za, fld_a, e * travel, e, eb, jit, True)
            rb = layer(frame_b, zb, fld_b, (e - 1.0) * travel, e, eb, jit, False)
            f = composite(ra, rb, blend_weight(op, eb, disp_b)).astype(np.float32)
            acc = f if acc is None else acc + f
        mid.append((acc / mb).astype(np.uint8))

    return np.concatenate([start9, np.stack(mid), end9], axis=0)


def _mean_frame_delta(block: np.ndarray) -> float:
    return float(np.abs(np.diff(block.astype(np.float32), axis=0)).mean())


def seam_error(clip: np.ndarray, n_start: int, n_middle: int
               ) -> tuple[float, float, float, float]:
    """How smoothly the rendered middle joins the two given buckets.

    Endpoint *identity* is trivially exact here (the buckets are copied through
    verbatim), so the quantity that matters is continuity at the joins.

    Raw MAE is not comparable across clips — a near-static bucket has an internal
    frame-to-frame delta of ~1.7 while a fast one runs ~25, so the same absolute
    seam is invisible in one and a jump cut in the other. We therefore report the
    ratio of the seam step to the bucket's own internal motion: **≈1 means the
    join is as smooth as the content's natural motion**, and >>1 is a visible cut.
    """
    i, j = n_start - 1, n_start + n_middle
    d0 = float(np.abs(clip[i].astype(np.float32) - clip[i + 1].astype(np.float32)).mean())
    d1 = float(np.abs(clip[j].astype(np.float32) - clip[j - 1].astype(np.float32)).mean())
    ref0 = max(_mean_frame_delta(clip[:n_start]), 1e-3)
    ref1 = max(_mean_frame_delta(clip[j:]), 1e-3)
    return d0, d1, d0 / ref0, d1 / ref1
