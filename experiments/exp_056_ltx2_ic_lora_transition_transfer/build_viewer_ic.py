#!/usr/bin/env python
"""exp_056 fork of exp_055/build_viewer.py: adds condition_reference manifest
passthrough -> videos.reference (the actual in-context reference clip of each
quadruple), rendered by viewer_template_ic.html. Everything else identical.

Original docstring:
exp_055 — build a self-contained, serveable interactive HTML viewer bundle
for a transition-eval-harness result (exp_052/exp_053 and any future run).

Pure front-end + data-packaging. NO GPU, NO DINO/torch: per-frame DINO features
are read from the harness's existing `outputs/eval/cache/dino_arr_*.npz` cache
(content-keyed exactly as `features.file_key` produces them), and the core-frame
labelling reuses the harness's OWN `morph_profile`/`core_mask` (loaded directly
from `src/diffusion/transition_eval/morph.py`, bypassing the heavy
`diffusion` package __init__ which drags in the diffusers/torch video stack).

Emits under <out>/:
  data.json            — everything the UI needs (schema documented in README)
  assets/videos/...    — RELATIVE symlinks to gen/ref/cond/lerp mp4s (big; not copied)
  assets/filmstrips/*.jpg — labelled core-frame filmstrips (endpoints vs core)
  assets/figures/*.png — copies of the harness figures (portable)
  index.html           — copied verbatim from viewer_template_ic.html

Usage (example — the exp_053 build; see run.py for the wired-up preset):
  python build_viewer.py \
    --validation   outputs/eval/exp_052/validation/run_0001 \
    --items        outputs/eval/exp_052/ladder/run_0001/items.jsonl \
    --manifest     experiments/exp_052_transition_eval_harness/manifest_exp051.json \
    --ceilings     outputs/eval/exp_052/ladder/run_0001/ceilings.json \
    --judge-summary outputs/eval/exp_053/judge_gemini_ladder/run_0004/judge_summary.json \
    --judge-results outputs/eval/exp_053/judge_gemini_ladder/run_0004/judge_results.json \
    --checks       outputs/eval/exp_053/checks/run_0001 \
    --report       outputs/eval/exp_053/ladder_v2/run_0002/report.md \
    --figures-dir  outputs/eval/exp_053/pair_examples \
    --figures-dir  outputs/eval/exp_053/dedup \
    --dedup        outputs/eval/exp_053/dedup/duplicates.json \
    --controls     outputs/eval/exp_052/controls \
    --transitions-root data/processed/transitions \
    --exclude higgsfield \
    --out          outputs/eval/exp_053/viewer \
    --label        "exp_053 — ladder_v2 (24 items)"
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DINO_MODEL = "facebook/dinov2-base"
SHORT_SIDE = "256"  # feature-cache key component (matches config features.short_side)


# --- reuse the harness math WITHOUT importing the diffusion package ----------
def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


morph = _load_module("te_morph", "src/diffusion/transition_eval/morph.py")
report = _load_module("te_report", "src/diffusion/transition_eval/report.py")
rubric = _load_module("te_rubric", "src/diffusion/transition_eval/rubric.py")
video_io = _load_module("te_video_io", "src/diffusion/transition_eval/video_io.py")


# --- small helpers -----------------------------------------------------------
def resolve(p) -> pathlib.Path:
    p = os.path.expandvars(str(p))
    return pathlib.Path(p) if os.path.isabs(p) else REPO_ROOT / p


def file_key(path: pathlib.Path, *parts: str) -> str:
    st = pathlib.Path(path).stat()
    raw = "|".join([str(pathlib.Path(path).resolve()), str(st.st_mtime_ns),
                    str(st.st_size), *parts])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def feats_for(path: pathlib.Path, cache_dir: pathlib.Path):
    """Return cached L2-normalized DINO features [T,D] for a video file, or None.
    Mirrors features.array_features naming for pipeline-processed videos."""
    try:
        k = file_key(path, DINO_MODEL, SHORT_SIDE)
    except FileNotFoundError:
        return None
    cache = cache_dir / f"dino_arr_{hashlib.sha1(k.encode()).hexdigest()[:16]}.npz"
    if cache.exists():
        return np.load(cache)["feats"]
    # fall-back: the video_features() naming (src=path) — scan is cheap only if needed
    alt = cache_dir / f"dino_{k}.npz"
    if alt.exists():
        return np.load(alt)["feats"]
    return None


def rel_symlink(target: pathlib.Path, link: pathlib.Path) -> bool:
    """Create a RELATIVE symlink link -> target (repo-portable). Returns success."""
    target = target.resolve()
    if not target.exists():
        return False
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    rel = os.path.relpath(target, link.parent)
    os.symlink(rel, link)
    return True


def r(x, n=4):
    """JSON-safe round: NaN/inf -> None, numpy scalars -> python."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(xf):
        return None
    return round(xf, n)


def rlist(a, n=4):
    return [r(v, n) for v in np.asarray(a).tolist()]


# --- fonts for filmstrip labels ----------------------------------------------
def _font(size: int):
    for cand in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            pass
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


ROLE_COLORS = {  # from the validated dataviz categorical palette
    "endpointA": (42, 120, 214),   # blue  — endpoint A (video's own start scene)
    "endpointB": (74, 58, 167),    # violet— endpoint B (video's own end scene)
    "core":      (227, 73, 72),    # red   — core / effect-medium frames
    "transition": (137, 135, 129), # muted — in-between (approaching/leaving effect)
}
ROLE_LABEL = {"endpointA": "endpoint A", "endpointB": "endpoint B",
              "core": "core / effect", "transition": "transition"}


def frame_role(idx: int, T: int, n_pre: int, n_suf: int, core_set: set) -> str:
    if idx < n_pre:
        return "endpointA"
    if idx >= T - n_suf:
        return "endpointB"
    if idx in core_set:
        return "core"
    return "transition"


def pick_columns(T: int, n_pre: int, n_suf: int, core_idx, k_target: int = 12):
    """Curated, temporally-ordered set of frame indices that always shows both
    endpoints and (up to 5) core/effect frames, padded with transition frames."""
    core_idx = list(core_idx)
    picks = {0, max(0, n_pre - 1), T - n_suf, T - 1}
    if core_idx:
        sel = np.linspace(0, len(core_idx) - 1, min(5, len(core_idx)))
        picks |= {int(core_idx[int(round(s))]) for s in sel}
    # fill remaining budget with a uniform sweep so the progression stays legible
    for f in np.linspace(0, T - 1, k_target):
        if len(picks) >= k_target:
            break
        picks.add(int(round(f)))
    return sorted(i for i in picks if 0 <= i < T)


def build_filmstrip(video_path: pathlib.Path, feats: np.ndarray, n_pre: int,
                    n_suf: int, out_path: pathlib.Path,
                    cell_h: int = 104, cell_w: int = 150) -> dict | None:
    """Decode the mp4, compute the harness morph profile + core mask, and write a
    labelled filmstrip JPG (endpoints vs core clearly marked). Returns the
    profile summary (curves, core indices, scalars, column roles) for data.json,
    or None if the clip is too short / unreadable."""
    try:
        prof = morph.morph_profile(feats, n_prefix=n_pre, n_suffix=n_suf)
    except ValueError:
        return None
    core = morph.core_mask(prof)
    core_idx = [int(i) for i in np.flatnonzero(core)]
    scal = morph.derived_scalars(prof)
    T = len(feats)

    frames, _fps = video_io.load_frames(video_path, short_side=max(cell_h, 220))
    T = min(T, len(frames))
    cols = [c for c in pick_columns(T, n_pre, n_suf, core_idx) if c < T]
    core_set = set(core_idx)

    pad, lab_h, top_h = 6, 30, 22
    n = len(cols)
    W = pad + n * (cell_w + pad)
    H = top_h + pad + cell_h + lab_h + pad
    bg = (24, 24, 25)
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    f_small = _font(12)
    f_tiny = _font(11)

    # legend row
    lx = pad
    for role in ("endpointA", "endpointB", "core", "transition"):
        draw.rectangle([lx, 6, lx + 12, 18], fill=ROLE_COLORS[role])
        draw.text((lx + 16, 5), ROLE_LABEL[role], fill=(210, 210, 205), font=f_tiny)
        lx += 20 + int(draw.textlength(ROLE_LABEL[role], font=f_tiny)) + 18

    col_roles = []
    x = pad
    for idx in cols:
        role = frame_role(idx, T, n_pre, n_suf, core_set)
        col_roles.append({"idx": idx, "role": role})
        color = ROLE_COLORS[role]
        # letterbox the frame into the cell
        im = Image.fromarray(frames[idx])
        im.thumbnail((cell_w, cell_h), Image.BILINEAR)
        cell = Image.new("RGB", (cell_w, cell_h), (12, 12, 12))
        cell.paste(im, ((cell_w - im.width) // 2, (cell_h - im.height) // 2))
        y = top_h + pad
        canvas.paste(cell, (x, y))
        # colored role bar on top of the cell + captions under it
        draw.rectangle([x, y - 4, x + cell_w, y - 1], fill=color)
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=color, width=2)
        cap = f"f{idx}"
        draw.text((x + 3, y + cell_h + 2), cap, fill=(200, 200, 195), font=f_small)
        rl = ROLE_LABEL[role].split(" / ")[0]
        tw = int(draw.textlength(rl, font=f_tiny))
        draw.text((x + cell_w - tw - 3, y + cell_h + 3), rl, fill=color, font=f_tiny)
        x += cell_w + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=84)

    b_hat = prof["b_hat"]
    return {
        "n_frames": T,
        "n_prefix": n_pre,
        "n_suffix": n_suf,
        "cross": r(prof["cross"], 4),
        "cross_high": bool(prof["cross_high"]),
        "a_hat": rlist(prof["a_hat"][:T], 3),
        "b_hat": rlist(b_hat[:T], 3) if b_hat is not None else None,
        "core_idx": core_idx,
        "scalars": {k: r(v, 4) for k, v in scal.items()},
        "columns": col_roles,
    }


# --- exam / retrieval --------------------------------------------------------
def exam_from_validation(val_dir: pathlib.Path) -> dict:
    results = json.loads((val_dir / "results.json").read_text())
    z = np.load(val_dir / "distance_matrices.npz", allow_pickle=True)
    names = [str(x) for x in z["names"]]
    styles = [str(x) for x in z["styles"]]
    classes = sorted(set(styles))

    # metric name -> distance-matrix key
    mat_key = {"morph_dtw": "morph", "effect_appearance": "appearance",
               "motion_fidelity": "motion"}
    metrics = {}
    matrices = {}
    retrieval_examples = {}
    for mname, rd in results["retrieval"].items():
        key = mat_key.get(mname, mname)
        if key not in z.files:
            continue
        D = np.asarray(z[key], dtype=float)
        matrices[mname] = [rlist(row, 4) for row in D]
        # recompute Wilson from the confusion matrix (exact k / n) — faithful
        conf = rd["confusion"]
        k_correct = int(sum(conf[c].get(c, 0) for c in classes))
        n_total = int(sum(sum(conf[c].values()) for c in classes))
        acc_wilson = report.wilson_interval(k_correct, n_total)
        per_class_wilson = {}
        for c in classes:
            row = conf[c]
            nc = int(sum(row.values()))
            kc = int(row.get(c, 0))
            per_class_wilson[c] = list(report.wilson_interval(kc, nc)) if nc else [None, None]
        # LOO 1-NN examples straight off the matrix (query -> nearest neighbour)
        Dl = D.copy()
        np.fill_diagonal(Dl, np.inf)
        Dl[~np.isfinite(Dl)] = np.inf
        ex = []
        for i in range(len(names)):
            j = int(np.argmin(Dl[i]))
            ex.append({
                "q": names[i], "q_style": styles[i],
                "nn": names[j], "nn_style": styles[j],
                "dist": r(D[i, j], 4),
                "correct": bool(styles[i] == styles[j]),
            })
        retrieval_examples[mname] = ex
        metrics[mname] = {
            "accuracy_1nn": r(rd["accuracy_1nn"], 4),
            "accuracy_wilson95": [r(acc_wilson[0], 4), r(acc_wilson[1], 4)],
            "k_correct": k_correct, "n_total": n_total,
            "chance": r(rd["chance"], 4),
            "coverage": r(rd["coverage"], 4),
            "per_class_recall": {c: r(v, 4) for c, v in rd["per_class_recall"].items()},
            "per_class_wilson": per_class_wilson,
            "per_class_n": {c: int(sum(conf[c].values())) for c in classes},
            "confusion": conf,
            "within_mean": r(rd["within_mean"], 4),
            "cross_mean": r(rd["cross_mean"], 4),
            "separation_cohens_d": r(rd["separation_cohens_d"], 4),
        }
    return {
        "names": names, "styles": styles, "classes": classes,
        "metrics": metrics, "matrices": matrices,
        "retrieval_examples": retrieval_examples,
        "lerp_floor": {k: r(v) if isinstance(v, (int, float)) else v
                       for k, v in results.get("lerp_floor", {}).items()},
        "clip_scalars": {name: {k: r(v, 4) for k, v in sc.items()}
                         for name, sc in results.get("scalars", {}).items()},
    }


# --- score tables (recomputed structured; faithful to report.score_tables) ---
def _cell(rows, col, flagged=False):
    vals = np.array([x for x in (row.get(col) for row in rows)
                     if x is not None and np.isfinite(x)], dtype=float)
    if len(vals) == 0:
        return {"mean": None, "std": None, "n": 0, "flag": bool(flagged)}
    return {"mean": r(vals.mean(), 3),
            "std": r(vals.std(ddof=1) if len(vals) > 1 else 0.0, 3),
            "n": int(len(vals)), "flag": bool(flagged)}


def score_tables(items, trust, judge_by_arm):
    arms = sorted({it["arm"] for it in items})
    trust = trust or {}

    def arm_flag(sub, key):
        return any(not trust.get(s, {}).get(key, True) for s in {r["style"] for r in sub})

    headline, analysis = [], []
    for arm in arms:
        sub = [it for it in items if it["arm"] == arm]
        mflag = arm_flag(sub, "motion_trusted")
        cflag = arm_flag(sub, "ceiling_trusted")
        ep = [it.get("prefix_dino") for it in sub] + [it.get("suffix_dino") for it in sub]
        ep_rows = [{"v": v} for v in ep]
        jp = (judge_by_arm or {}).get(arm, {}).get("all_pass")
        headline.append({
            "arm": arm, "n": len(sub),
            "appearance": _cell(sub, "norm_appearance_best", cflag),
            "motion": _cell(sub, "norm_motion_fidelity_mean", mflag or cflag),
            "judge_pass": r(jp, 3) if jp is not None else None,
            "endpoint_dino": _cell(ep_rows, "v"),
            "max_seam_z": _cell(sub, "max_seam_z"),
            "leak_max_sim": _cell(sub, "leak_max_sim_target"),
        })
        analysis.append({
            "arm": arm,
            "profile_dtw_norm": _cell(sub, "norm_profile_dtw_best", cflag),
            "depth": _cell(sub, "scalar_depth"),
            "depart": _cell(sub, "scalar_depart"),
            "arrive": _cell(sub, "scalar_arrive"),
            "core_frac": _cell(sub, "scalar_core_frac"),
            "leak_excess": _cell(sub, "leak_excess"),
            "cross_high_items": int(sum(1 for it in sub if it.get("scalar_cross_high"))),
        })
    return {"headline": headline, "analysis": analysis, "arms": arms}


# --- glossary (static, harness-faithful) -------------------------------------
GLOSSARY = {
    "metrics": {
        "M1_morph": "Morph Profile (M1): per-frame cosine similarity to the video's OWN "
                    "endpoints -> curves a(t)/b(t), floor-normalized by cross=cos(eA,eB). "
                    "core frames = 'neither endpoint' (env<0.5); the effect medium lives there.",
        "M2_motion": "Motion Fidelity (M2): CoTracker3 tracklet velocity-direction "
                     "cross-correlation, bidirectional mean-of-max. NaN when no moving tracklets.",
        "M3_appearance": "Effect Appearance (M3): symmetric mean-of-max DINO cosine between "
                         "generated core frames and reference core frames.",
        "M4_judge": "Rubric VLM judge (M4): Gemini native-video checklist q1-q5. EXPERIMENTAL / "
                    "advisory until human-validated; measures cleanliness more than transfer.",
        "M5_endpoints": "Endpoint fidelity + seams (M5): LPIPS+DINO of conditioned frames vs the "
                        "condition clips; seam z = robust z-score of temporal LPIPS at the handoff "
                        "boundaries (z<0 = no seam).",
        "M6_leakage": "Leakage (M6): near-duplicate retrieval of gen core frames against the "
                      "target style's frames vs unrelated styles. max_sim_target >= ~0.88 = "
                      "near-copy regime; excess = target max - mean cross-style max.",
    },
    "normalization": "Every reported score is normalized (raw - floor) / (ceiling - floor), "
                     "clipped to [0,1]. floor = lerp-crossfade control; ceiling = real same-style "
                     "clips (leave-one-out). No composite score is ever produced.",
    "trust_flags": {
        "dagger": "† metric not exam-certified for this style (motion recall < 0.5).",
        "double_dagger": "‡ / () ceiling rests on < 4 reference clips.",
    },
    "rubric": rubric.RUBRIC_QUESTIONS,
    "fail_if_true": list(rubric.FAIL_IF_TRUE),
}

FIGURE_CAPTIONS = {
    "confusion_effect_appearance.png": "Exam confusion — effect appearance (M3) LOO 1-NN.",
    "confusion_morph_dtw.png": "Exam confusion — morph-profile DTW (M1) LOO 1-NN.",
    "confusion_motion_fidelity.png": "Exam confusion — motion fidelity (M2) LOO 1-NN.",
    "morph_profiles.png": "Per-style morph profiles a(t)/b(t) over the reference corpus.",
    "depth_hist.png": "Transformation-depth histogram — real clips vs lerp floor.",
    "scatter.png": "Per-item normalized scores by arm (appearance / motion / profile DTW).",
    "leak_scatter.png": "Adversarial leakage per arm with the 0.88 near-copy bar.",
    "ceiling.png": "Pair example — real same-style ceiling pair.",
    "dilution.png": "Pair example — appearance dilution.",
    "negatives.png": "Pair example — cross-style negatives.",
    "flips.png": "Pair example — retrieval flips (nearest neighbour is the wrong style).",
    "duplicates.png": "Dedup montage — near-duplicate reference clips removed from the corpus.",
    "flying_cam_0_vs_1.png": "Dedup — flying_cam 0 vs 1 order-invariant near-duplicate.",
}


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation", required=True, help="validation run dir (results.json + distance_matrices.npz)")
    ap.add_argument("--items", help="ladder items.jsonl (scored items)")
    ap.add_argument("--manifest", help="manifest.json mapping item_id -> videos")
    ap.add_argument("--ceilings", help="ceilings.json (per-style floor/ceiling ref)")
    ap.add_argument("--judge-summary", help="judge_summary.json (by arm)")
    ap.add_argument("--judge-results", help="judge_results.json (per item)")
    ap.add_argument("--checks", help="checks run dir (checks.json + adversarial.jsonl)")
    ap.add_argument("--report", help="report.md (embedded verbatim for reference)")
    ap.add_argument("--figures-dir", action="append", default=[],
                    help="extra dir of PNGs to include (repeatable)")
    ap.add_argument("--dedup", help="dedup duplicates.json")
    ap.add_argument("--controls", help="controls/ dir of lerp floor mp4s")
    ap.add_argument("--transitions-root", default="data/processed/transitions")
    ap.add_argument("--cache-dir", default="outputs/eval/cache")
    ap.add_argument("--exclude", nargs="*", default=["higgsfield"])
    ap.add_argument("--template", default=str(pathlib.Path(__file__).parent / "viewer_template_ic.html"))
    ap.add_argument("--out", required=True, help="bundle output dir")
    ap.add_argument("--label", default="transition-eval-harness")
    args = ap.parse_args()

    out = resolve(args.out)
    (out / "assets" / "videos").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "filmstrips").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "figures").mkdir(parents=True, exist_ok=True)
    cache_dir = resolve(args.cache_dir)
    troot = resolve(args.transitions_root)
    warnings = []

    print(f"[build] out={out}")

    # ---- corpus (current on-disk truth) ----
    styles = []
    style_dirs = sorted([d for d in troot.iterdir()
                         if d.is_dir() and d.name not in args.exclude])
    for d in style_dirs:
        clips = sorted(d.glob("*.mp4"))
        dups = sorted((d / "_dup").glob("*.mp4")) if (d / "_dup").is_dir() else []
        styles.append({"style": d.name, "n_clips": len(clips),
                       "n_dup": len(dups),
                       "clips": [c.stem for c in clips]})
    total_clips = sum(s["n_clips"] for s in styles)
    print(f"[build] corpus: {len(styles)} styles, {total_clips} clips on disk")

    # ---- exam ----
    exam = exam_from_validation(resolve(args.validation))
    exam_styles = set(exam["styles"])

    # ---- trust flags (report.trust_flags, exactly as run_score.load_trust) ----
    val_results = json.loads((resolve(args.validation) / "results.json").read_text())
    ref_counts = {s["style"]: s["n_clips"] for s in styles}
    trust = report.trust_flags(val_results, ref_counts)
    trust = {k: {kk: (r(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()}
             for k, v in trust.items()}

    # ---- filmstrips for the exam reference clips (best-effort) ----
    def clip_video(name):
        style, clip = name.split("/", 1)
        for c in (troot / style / f"{clip}.mp4", troot / style / "_dup" / f"{clip}.mp4"):
            if c.exists():
                return c
        return None

    clip_records = {}   # name -> {video(rel), filmstrip(rel), profile}
    n_strip = 0
    for name in exam["names"]:
        vp = clip_video(name)
        rec = {"video": None, "filmstrip": None, "profile": None}
        if vp is not None:
            link = out / "assets" / "videos" / "ref" / f"{name}.mp4"
            if rel_symlink(vp, link):
                rec["video"] = os.path.relpath(link, out)
            fe = feats_for(vp, cache_dir)
            if fe is not None:
                strip = out / "assets" / "filmstrips" / f"ref__{name.replace('/', '__')}.jpg"
                prof = build_filmstrip(vp, fe, 9, 8, strip)
                if prof is not None:
                    rec["filmstrip"] = os.path.relpath(strip, out)
                    rec["profile"] = prof
                    n_strip += 1
        clip_records[name] = rec
    print(f"[build] reference filmstrips: {n_strip}/{len(exam['names'])}")

    # ---- items (scored) ----
    manifest = {}
    if args.manifest:
        for m in json.loads(resolve(args.manifest).read_text()):
            manifest[m["item_id"]] = m

    judge_results = {}
    if args.judge_results and resolve(args.judge_results).exists():
        judge_results = json.loads(resolve(args.judge_results).read_text())
    judge_summary = {}
    if args.judge_summary and resolve(args.judge_summary).exists():
        judge_summary = json.loads(resolve(args.judge_summary).read_text())

    ceilings = {}
    if args.ceilings and resolve(args.ceilings).exists():
        ceilings = json.loads(resolve(args.ceilings).read_text())

    items = []
    raw_items = []
    if args.items:
        raw_items = [json.loads(l) for l in resolve(args.items).read_text().splitlines() if l.strip()]

    n_item_strip = 0
    for row in raw_items:
        iid = row["item_id"]
        m = manifest.get(iid, {})
        vids = {"generated": None, "prefix": None, "suffix": None, "reference": None}
        if m.get("generated_video"):
            gp = resolve(m["generated_video"])
            link = out / "assets" / "videos" / "gen" / f"{iid}.mp4"
            if rel_symlink(gp, link):
                vids["generated"] = os.path.relpath(link, out)
        for side, key in (("prefix", "condition_prefix"), ("suffix", "condition_suffix"),
                          ("reference", "condition_reference")):
            cond = m.get(key)
            if cond and cond.get("video"):
                cp = resolve(cond["video"])
                link = out / "assets" / "videos" / "cond" / f"{iid}_{side}.mp4"
                if rel_symlink(cp, link):
                    vids[side] = os.path.relpath(link, out)

        # filmstrip + profile for the generated video
        profile = None
        filmstrip = None
        n_pre = (m.get("condition_prefix") or {}).get("num_frames", 9)
        n_suf = (m.get("condition_suffix") or {}).get("num_frames", 8)
        if m.get("generated_video"):
            gp = resolve(m["generated_video"])
            fe = feats_for(gp, cache_dir)
            if fe is not None and gp.exists():
                strip = out / "assets" / "filmstrips" / f"gen__{iid}.jpg"
                profile = build_filmstrip(gp, fe, n_pre, n_suf, strip)
                if profile is not None:
                    filmstrip = os.path.relpath(strip, out)
                    n_item_strip += 1
        if m.get("generated_video") and profile is None:
            warnings.append(f"no filmstrip for item {iid} (missing feats/video cache)")

        jr = judge_results.get(iid)
        judge = None
        if jr:
            judge = {q: {"answer": jr.get(q, {}).get("answer"),
                         "evidence": jr.get(q, {}).get("evidence")}
                     for q in rubric.RUBRIC_QUESTIONS}
            judge["all_pass"] = rubric.item_pass(jr)
            judge["model_version"] = jr.get("_model_version")

        # every scalar/metric from the row, JSON-cleaned
        metrics = {k: (r(v, 5) if isinstance(v, (int, float)) else v)
                   for k, v in row.items()
                   if k not in ("item_id", "arm", "style", "n_endpoints")}

        items.append({
            "item_id": iid, "arm": row["arm"], "style": row["style"],
            "n_endpoints": row.get("n_endpoints", 2),
            "notes": m.get("notes", ""),
            "videos": vids, "filmstrip": filmstrip, "profile": profile,
            "ref_clips": [n for n in exam["names"] if n.split("/", 1)[0] == row["style"]],
            "ceiling": ceilings.get(row["style"]),
            "metrics": metrics,
            "judge": judge,
        })
    print(f"[build] items: {len(items)}, item filmstrips: {n_item_strip}")

    # ---- score tables ----
    tables = score_tables(raw_items, trust, judge_summary) if raw_items else None

    # ---- checks / adversarial ----
    checks = None
    adversarial = []
    if args.checks:
        cdir = resolve(args.checks)
        if (cdir / "checks.json").exists():
            checks = json.loads((cdir / "checks.json").read_text())
        aj = cdir / "adversarial.jsonl"
        if aj.exists():
            for l in aj.read_text().splitlines():
                if l.strip():
                    a = json.loads(l)
                    adversarial.append({k: (r(v, 5) if isinstance(v, (int, float)) else v)
                                        for k, v in a.items()})
        au = cdir / "checkC_audit.json"
        if au.exists() and checks is not None:
            checks["checkC_audit"] = json.loads(au.read_text())

    # ---- dedup ----
    dedup = None
    if args.dedup and resolve(args.dedup).exists():
        dedup = json.loads(resolve(args.dedup).read_text())

    # ---- lerp floor controls (per style, symlinked) ----
    controls = {}
    if args.controls and resolve(args.controls).exists():
        croot = resolve(args.controls)
        for sd in sorted(croot.iterdir()):
            if sd.is_dir():
                vs = []
                for v in sorted(sd.glob("*.mp4")):
                    link = out / "assets" / "videos" / "lerp" / sd.name / v.name
                    if rel_symlink(v, link):
                        vs.append(os.path.relpath(link, out))
                if vs:
                    controls[sd.name] = vs

    # ---- figures (copied — portable) ----
    figures = []
    fig_dirs = [resolve(args.validation)]
    if args.checks:
        fig_dirs.append(resolve(args.checks))
    if args.report:
        fig_dirs.append(resolve(args.report).parent)
    fig_dirs += [resolve(fd) for fd in args.figures_dir]
    seen = set()
    for fd in fig_dirs:
        if not fd.exists():
            continue
        for png in sorted(fd.glob("*.png")):
            if png.name in seen:
                continue
            seen.add(png.name)
            dst = out / "assets" / "figures" / png.name
            shutil.copy2(png, dst)
            figures.append({"name": png.name,
                            "path": os.path.relpath(dst, out),
                            "caption": FIGURE_CAPTIONS.get(png.name, png.stem.replace("_", " ")),
                            "group": ("exam" if "confusion" in png.name or "morph_prof" in png.name
                                      or "depth" in png.name else
                                      "checks" if "leak" in png.name else
                                      "dedup" if "dup" in png.name or "flying_cam_0" in png.name else
                                      "pairs" if png.name in ("ceiling.png", "dilution.png",
                                                              "negatives.png", "flips.png") else
                                      "ladder")})
    print(f"[build] figures copied: {len(figures)}")

    # ---- report.md verbatim ----
    report_md = None
    if args.report and resolve(args.report).exists():
        report_md = resolve(args.report).read_text()

    data = {
        "meta": {
            "label": args.label,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "generator": "experiments/exp_055_eval_viewer/build_viewer.py",
            "sources": {
                "validation": args.validation, "items": args.items,
                "manifest": args.manifest, "ceilings": args.ceilings,
                "judge_summary": args.judge_summary, "judge_results": args.judge_results,
                "checks": args.checks, "report": args.report, "dedup": args.dedup,
            },
            "n_items": len(items),
            "arms": sorted({it["arm"] for it in items}) if items else [],
            "styles_in_items": sorted({it["style"] for it in items}) if items else [],
        },
        "glossary": GLOSSARY,
        "corpus": {
            "styles": [{**s, "in_exam": s["style"] in exam_styles,
                        "trust": trust.get(s["style"])} for s in styles],
            "total_clips": total_clips,
            "exam_n_clips": len(exam["names"]),
            "controls": controls,
        },
        "exam": {**exam, "clip_records": clip_records},
        "score_tables": tables,
        "judge": {"summary": judge_summary,
                  "rubric": rubric.RUBRIC_QUESTIONS,
                  "fail_if_true": list(rubric.FAIL_IF_TRUE)},
        "checks": checks,
        "adversarial": adversarial,
        "dedup": dedup,
        "figures": figures,
        "report_md": report_md,
        "items": items,
    }

    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False))
    size_mb = (out / "data.json").stat().st_size / 1e6
    print(f"[build] data.json written ({size_mb:.2f} MB)")

    # ---- index.html from template ----
    tpl = resolve(args.template)
    shutil.copy2(tpl, out / "index.html")
    print(f"[build] index.html copied from {tpl.name}")

    if warnings:
        print(f"[build] {len(warnings)} warnings:")
        for w in warnings[:20]:
            print("   -", w)

    print(f"\n[done] bundle -> {out}")
    print(f"       serve:  cd {out} && python -m http.server 8000")
    print(f"       open:   http://localhost:8000/")


if __name__ == "__main__":
    main()
