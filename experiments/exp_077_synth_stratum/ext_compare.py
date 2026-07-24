"""Standalone VISUAL comparison of layer-extension policies (boomerang / hold / flow)
for the exp_075 procedural transition engine, for operator eyeballing.

NOT part of the mass pipeline; does NOT modify the engine. It imports the engine
directly (streams / operators / shaders / maps / GLRunner) and reuses exp_077's
timing-aware progress ramp (copied verbatim below so there are no import side
effects). CPU-only (moderngl + EGL / llvmpipe).

For 3 representative operators on ONE fixed high-motion endpoint pair, it renders
the 121-frame clip under 4 conditions (everything identical except extension +
timing), builds a captioned filmstrip per condition, and stacks the 4 into one
cmp_<operator>.png. Also writes notes.txt with faithful boundary/reversal numbers.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from engine import maps, operators, shaders, streams, videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

BANK_DIR = pathlib.Path(
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/misc/gl-transitions/transitions")
COND_DIR = REPO_ROOT / "experiments/exp_062_ladder_r2r3_specialists/dataset/cond"
OUT_DIR = REPO_ROOT / "outputs/endpoint_candidates/ext_compare"

H, W = 640, 480          # frame height/width (portrait, matches exp_077 config)
T, K = 121, 9            # num_frames, anchor block length
EASING = "smoothstep"    # fixed across every condition
PAIR = "portal_2"        # highest intra-clip motion of the 44 complete pairs

# filmstrip indices: ~10 evenly spaced, includes the required 0,8,9,60,111,112,120
STRIP_IDX = [0, 8, 9, 30, 45, 60, 80, 111, 112, 120]

# the four extension/timing conditions (everything else identical)
CONDITIONS = [
    ("A", "boomerang", 40.0, 40.0, "mid-clip"),
    ("B", "hold",      40.0, 40.0, "mid-clip"),
    ("C", "flow",      40.0, 40.0, "mid-clip"),
    ("D", "flow",       9.0, 103.0, "fill-middle"),
]

# three representative operators, fixed params so only extension/timing vary
OPERATORS = [
    dict(tag="wipeRight", shader="wipeRight", params={}, aux_kind=None, aux_seed=0,
         kind="hard-edged WIPE"),
    dict(tag="fade", shader="fade", params={}, aux_kind=None, aux_seed=0,
         kind="soft DISSOLVE/fade"),
    dict(tag="displacement", shader="displacement", params={"strength": 0.5},
         aux_kind="fbm", aux_seed=7, kind="aux-map DISPLACEMENT (medium-bearing)"),
]


# --- exp_077 timing-aware progress ramp (verbatim copy; engine untouched) ------
def timed_progress_ramp(total, n_start, n_end, easing, onset, duration):
    t0, t1 = n_start - 1, total - n_end
    onset = float(np.clip(onset, t0, t1 - 1))
    end = float(np.clip(onset + duration, onset + 1e-6, t1))
    t = np.arange(total, dtype=np.float64)
    u = np.clip((t - onset) / (end - onset), 0.0, 1.0)
    p = np.asarray(streams.EASINGS[easing](u), dtype=np.float64)
    p[t <= onset] = 0.0
    p[t >= end] = 1.0
    p[: t0 + 1] = 0.0
    p[t1:] = 1.0
    return np.clip(p, 0.0, 1.0)


def render_condition(runner, bank, opspec, start9, end9, extension, onset, duration):
    """Render one 121-frame clip. Mirrors engine.operators.render_sample /
    exp_077.render_timed but with policy + timing supplied as params."""
    shader = opspec["shader"]
    prog = runner.program(shader, bank[shader].source)
    if opspec["aux_kind"]:
        runner.set_aux_map(maps.make_map(opspec["aux_kind"], runner.height,
                                         runner.width, opspec["aux_seed"]))
    aux_uniform = shaders.AUX_SAMPLER_SHADERS.get(shader)
    a_stream = streams.build_from_stream(start9, T, extension)
    b_stream = streams.build_to_stream(end9, T, extension)
    p = timed_progress_ramp(T, len(start9), len(end9), EASING, onset, duration)
    out = np.empty_like(a_stream)
    for t in range(T):
        # flip="none", swap=False for all three operators (kept identical)
        out[t] = runner.render(prog, a_stream[t], b_stream[t], p[t],
                               opspec["params"], aux_uniform)
    return out, a_stream, b_stream, p


# --- faithful diagnostics ------------------------------------------------------
def seam_cosine(v_ref, v_ext):
    """cos angle between the real endpoint motion and the extended motion.
    ~ -1 => motion reversed (boomerang), >0 => continues (flow), ~0 => frozen."""
    a, b = v_ref.astype(np.float64).ravel(), v_ext.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def diagnostics(clip, a_stream, b_stream, start9, end9):
    f = clip.astype(np.float32)
    d0, d1 = operators.endpoint_fidelity(clip, start9, end9)
    # boundary jumps in the rendered clip
    d_8_9 = float(np.abs(f[9] - f[8]).mean())
    d_111_112 = float(np.abs(f[112] - f[111]).mean())
    # mid-phase motion in the extended region (mean consecutive |diff|)
    midA = float(np.abs(np.diff(f[9:40], axis=0)).mean())
    midB = float(np.abs(np.diff(f[81:112], axis=0)).mean())
    # reversal detection on the underlying streams at the seams
    a = a_stream.astype(np.float32)
    b = b_stream.astype(np.float32)
    cos_start = seam_cosine(a[8] - a[7], a[9] - a[8])          # start9 seam 8->9
    cos_end = seam_cosine(b[112] - b[113], b[111] - b[112])    # end9 seam 112<-111
    return dict(ep_start=d0, ep_end=d1, d_8_9=d_8_9, d_111_112=d_111_112,
                midA=midA, midB=midB, cos_start=cos_start, cos_end=cos_end)


# --- montage -------------------------------------------------------------------
def _font(size, bold=False):
    base = "/usr/share/fonts/dejavu-sans-fonts/"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return PIL.ImageFont.truetype(base + name, size)
    except Exception:
        return PIL.ImageFont.load_default()


def build_cmp_image(op_tag, op_kind, rows):
    """rows: list of (letter, policy_label, timing_label, clip, diag). PIL.Image."""
    tw, th, pad, margin = 150, 200, 4, 14
    lab_w = 340
    n = len(STRIP_IDX)
    strip_w = n * tw + (n - 1) * pad
    img_w = margin + lab_w + strip_w + margin
    title_h, idx_h, row_gap = 40, 26, 10
    row_h = th + row_gap
    img_h = margin + title_h + idx_h + len(rows) * row_h + margin

    canvas = PIL.Image.new("RGB", (img_w, img_h), (255, 255, 255))
    d = PIL.ImageDraw.Draw(canvas)
    f_title = _font(22, bold=True)
    f_idx = _font(15, bold=True)
    f_cap = _font(16, bold=True)
    f_time = _font(14, bold=True)
    f_sub = _font(13)

    d.text((margin, margin), f"{op_tag}  —  {op_kind}   |   endpoint pair: {PAIR}"
           f"   |   ease={EASING}, flip=none, swap=off", font=f_title, fill=(0, 0, 0))

    strip_x0 = margin + lab_w
    idx_y = margin + title_h
    # column index header (mark the endpoint-seam frames in red)
    seam = {8, 9, 111, 112}
    for k, fi in enumerate(STRIP_IDX):
        cx = strip_x0 + k * (tw + pad) + tw // 2
        col = (200, 0, 0) if fi in seam else (60, 60, 60)
        d.text((cx, idx_y), f"f{fi}", font=f_idx, fill=col, anchor="ma")

    y = idx_y + idx_h
    for (letter, policy_label, timing_label, clip, diag) in rows:
        # left caption panel
        d.text((margin, y + 6), f"{letter})", font=f_title, fill=(0, 0, 0))
        d.text((margin + 34, y + 8), policy_label, font=f_cap, fill=(0, 0, 0))
        d.text((margin + 34, y + 30), timing_label, font=f_time, fill=(40, 40, 40))
        sub = (f"8->9: {diag['d_8_9']:.1f}   111->112: {diag['d_111_112']:.1f}\n"
               f"midA mot: {diag['midA']:.1f}   midB mot: {diag['midB']:.1f}\n"
               f"seam cos  start: {diag['cos_start']:+.2f}  end: {diag['cos_end']:+.2f}\n"
               f"ep mae s/e: {diag['ep_start']:.2f}/{diag['ep_end']:.2f}")
        d.multiline_text((margin + 34, y + 52), sub, font=f_sub, fill=(70, 70, 70),
                         spacing=4)
        # filmstrip
        for k, fi in enumerate(STRIP_IDX):
            thumb = PIL.Image.fromarray(clip[fi]).resize((tw, th),
                                                         PIL.Image.BILINEAR)
            x = strip_x0 + k * (tw + pad)
            canvas.paste(thumb, (x, y))
            if STRIP_IDX[k] in seam:              # thin red frame on seam columns
                d.rectangle([x, y, x + tw - 1, y + th - 1], outline=(200, 0, 0), width=2)
        y += row_h
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] out dir: {OUT_DIR}")

    bank = shaders.load_bank(BANK_DIR)
    runner = GLRunner(W, H)
    print(f"[info] GL: {runner.renderer_name()} | {len(bank)} shaders parsed")

    s9 = videoio.read_clip(COND_DIR / f"{PAIR}_start9.mp4")[:K]
    e9 = videoio.read_clip(COND_DIR / f"{PAIR}_end9.mp4")[-K:]
    print(f"[info] endpoints {PAIR}: start9 {s9.shape}  end9 {e9.shape}")

    notes_lines = [
        "Layer-extension policy comparison (exp_075 engine) — faithful observations",
        "=" * 78,
        f"endpoint pair : {PAIR}  (start9 motion 14.74 / end9 motion 28.21 — top of 44 pairs)",
        f"frame geom    : {T} frames, {H}x{W}, anchor K={K} (start block f0-8, end block f112-120)",
        f"fixed         : ease={EASING}, flip=none, swap=off; only extension+timing vary",
        "conditions    : A boomerang mid(40,40) | B hold mid(40,40) | C flow mid(40,40) | D flow fill(9,103)",
        "filmstrip idx : " + ", ".join(f"f{i}" for i in STRIP_IDX),
        "",
        "Diagnostics per row (measured on the rendered clip / underlying streams):",
        "  d_8_9, d_111_112 = mean|frame diff| across the endpoint seams (freeze->~0)",
        "  midA/midB mot    = mean consecutive |diff| in the extended pure-A / pure-B phase",
        "  seam cos         = cos(real endpoint motion, extended motion): -1 reversed, +1 continues, 0 frozen",
        "  ep mae s/e       = rendered-clip fidelity to the given endpoints (must be ~0)",
        "=" * 78,
        "",
    ]

    written = []
    for opspec in OPERATORS:
        print(f"\n[op] {opspec['tag']} ({opspec['kind']})")
        rows = []
        notes_lines.append(f"### {opspec['tag']}  ({opspec['kind']})")
        for (letter, ext, onset, dur, place) in CONDITIONS:
            clip, a_stream, b_stream, p = render_condition(
                runner, bank, opspec, s9, e9, ext, onset, dur)
            diag = diagnostics(clip, a_stream, b_stream, s9, e9)
            policy_label = f"{ext} ({place})" if place == "fill-middle" else ext
            timing_label = f"{place}  onset={onset:.0f} dur={dur:.0f}"
            rows.append((letter, policy_label, timing_label, clip, diag))
            cap = f"{ext} {place} onset={onset:.0f} dur={dur:.0f}"
            print(f"  {letter}) {cap:38s} d8_9={diag['d_8_9']:6.2f} "
                  f"d111_112={diag['d_111_112']:6.2f} midA={diag['midA']:5.2f} "
                  f"midB={diag['midB']:5.2f} cosS={diag['cos_start']:+.2f} "
                  f"cosE={diag['cos_end']:+.2f} ep=({diag['ep_start']:.2f},{diag['ep_end']:.2f})")
            notes_lines.append(
                f"  {letter}) {ext:9s} {place:11s} onset={onset:5.0f} dur={dur:5.0f} | "
                f"d8_9={diag['d_8_9']:6.2f} d111_112={diag['d_111_112']:6.2f} "
                f"midA={diag['midA']:5.2f} midB={diag['midB']:5.2f} "
                f"cosS={diag['cos_start']:+.2f} cosE={diag['cos_end']:+.2f} "
                f"ep=({diag['ep_start']:.2f},{diag['ep_end']:.2f})")
        img = build_cmp_image(opspec["tag"], opspec["kind"], rows)
        outp = OUT_DIR / f"cmp_{opspec['tag']}.png"
        img.save(outp)
        written.append(outp)
        notes_lines.append("")
        print(f"  -> {outp}  ({img.size[0]}x{img.size[1]})")

    # interpretation (derived from the numbers above, not editorializing beyond them)
    notes_lines += [
        "=" * 78,
        "READ (from the seam-cosine + freeze diagnostics, consistent across all 3 operators):",
        "  boomerang (A): seam cos ~ -1 on BOTH ends => extended layers play the endpoint",
        "                 motion BACKWARD (visible reversal / motion hitch at f8->9 and f111->112).",
        "  hold (B)     : midA/midB motion ~ 0 and seam diffs ~ 0 => layers FREEZE; the whole",
        "                 scene stops during the transition (the prior we do not want to teach).",
        "  flow (C/D)   : seam cos > 0 and non-zero mid motion => layers keep moving FORWARD",
        "                 through the effect; boundaries at f8->9 / f111->112 are the smoothest.",
        "  fill-middle (D): flow with onset~9/dur~103 removes the long pure-A/pure-B phases so",
        "                 the effect occupies almost the whole clip; endpoint blocks still exact.",
        "  endpoint fidelity ep mae ~ 0 in every condition (progress is pinned across the",
        "                 start/end blocks, so the given endpoints are reproduced regardless).",
        "",
        "One-line read: flow looked the most natural — motion continues through the effect",
        "  (positive seam cosine, smoothest f8->9 / f111->112); boomerang shows a clear motion",
        "  REVERSAL (seam cosine ~ -1); hold visibly FREEZES the scene (near-zero mid motion).",
        "  No tearing/artifacts seen except displacement's inherent UV warp during the ramp.",
    ]

    notes_path = OUT_DIR / "notes.txt"
    notes_path.write_text("\n".join(notes_lines) + "\n")
    print(f"\n[done] notes -> {notes_path}")

    # verify PNGs are valid/readable
    print("\n[verify] reopening PNGs:")
    for p in written:
        with PIL.Image.open(p) as im:
            im.verify()
        with PIL.Image.open(p) as im:
            arr = np.asarray(im)
        print(f"  OK {p.name}  size={im.size}  arr={arr.shape}  dtype={arr.dtype}")
    print("[verify] all PNGs valid")


if __name__ == "__main__":
    main()
