"""Adversarial ground-truth probes (SPEC §6.2) — synthetic items whose correct
score is KNOWN, so the integrity metrics are graded against truth, not vibes.

copy splices  (M2a): take an existing honest generation, replace its middle
    segment with the REFERENCE's middle segment (resize-cover-crop to match).
    Ground truth: contains literal reference content -> copy_max must clear
    tau_copy. The splice-min / honest-max gap recalibrates tau_copy
    (exp_053 check-C pattern, now reference-local instead of corpus-wide).

cross-label   (M2b): rescore existing generations under a deliberately WRONG
    reference class from a different texture family. Ground truth: the margin
    must go negative and the argmax must name the class the item actually
    imitates.

controls      (floors/flags): lerp (two-sided) and static-hold (one-sided)
    arms must land at the transfer floor and trip core_degenerate/timing flags.

All builders emit standard v3 eval manifests + probe videos — probes are scored
by the SAME score.py as real items (a probe path through special code would
certify the special code, not the instrument).

STATUS: builders implemented; execution is part of the certification run
(needs GPU featurization for spliced videos).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from ..video_io import load_frames, resize_cover_crop, write_video


def build_copy_splice(gen_path: pathlib.Path, ref_path: pathlib.Path,
                      out_path: pathlib.Path, n_prefix: int = 9, n_suffix: int = 8,
                      short_side: int = 256) -> pathlib.Path:
    """Honest generation with its mid-segment replaced by the reference's —
    a ground-truth copy that keeps the original conditioned windows intact."""
    gen, fps = load_frames(gen_path, short_side=short_side)
    ref, _ = load_frames(ref_path, short_side=short_side)
    T = len(gen)
    mid_lo, mid_hi = n_prefix, T - n_suffix
    n_mid = mid_hi - mid_lo
    r_lo = max(0, (len(ref) - n_mid) // 2)
    seg = ref[r_lo:r_lo + n_mid]
    if len(seg) < n_mid:  # short reference: loop-pad, still literal ref content
        seg = np.concatenate([seg, seg[: n_mid - len(seg)]])
    seg = resize_cover_crop(seg, gen.shape[1], gen.shape[2])
    spliced = np.concatenate([gen[:mid_lo], seg, gen[mid_hi:]])
    write_video(spliced, out_path, fps=fps)
    return out_path


def build_cross_label_manifest(items: list[dict], corpus: dict,
                               texture_families: dict[str, str],
                               n_items: int) -> list[dict]:
    """Relabel existing items with a wrong-reference class from a DIFFERENT
    texture family (cousin classes would blur the ground truth — the exp_057
    lesson that cousins legitimately share texture). Emits standard manifest
    rows with `notes` recording the true class for grading."""
    out = []
    for it in items[:n_items]:
        true_cls = it["style"]
        fam = texture_families.get(true_cls)
        wrong = next((c for c in sorted(corpus["classes"])
                      if c != true_cls and texture_families.get(c) not in (fam, None)
                      and corpus["classes"][c]["n_clips"] >= 2), None)
        if wrong is None:
            continue
        ref = f"{wrong}/{sorted(k.split('/')[1] for k in corpus['clips'] if k.startswith(wrong + '/'))[0]}"
        out.append({**it,
                    "item_id": f"xlabel__{it['item_id']}",
                    "style": wrong,
                    "reference_video": f"{corpus['corpus_root']}/{ref}",
                    "notes": f"cross-label probe; true_class={true_cls}"})
    return out


def grade_copy_probes(rows: list[dict], honest_rows: list[dict], tau: float) -> dict:
    splice_scores = [r["copy_max"] for r in rows if np.isfinite(r.get("copy_max", np.nan))]
    honest_scores = [r["copy_max"] for r in honest_rows if np.isfinite(r.get("copy_max", np.nan))]
    return {
        "splice_min": min(splice_scores) if splice_scores else None,
        "honest_max": max(honest_scores) if honest_scores else None,
        "all_splices_flagged": bool(splice_scores and min(splice_scores) >= tau),
        "gap": (min(splice_scores) - max(honest_scores))
               if splice_scores and honest_scores else None,
        "tau_recalibrated": (0.5 * (min(splice_scores) + max(honest_scores)))
                            if splice_scores and honest_scores else None,
    }


def grade_cross_label(rows: list[dict]) -> dict:
    graded = []
    for r in rows:
        true_cls = (r.get("notes") or "").split("true_class=")[-1] or None
        graded.append({
            "item_id": r["item_id"],
            "margin_negative": bool(r.get("margin", 1) < 0),
            "named_true_class": r.get("intruder") == true_cls,
        })
    n = len(graded) or 1
    return {"n": len(graded),
            "frac_negative": sum(g["margin_negative"] for g in graded) / n,
            "frac_named_true": sum(g["named_true_class"] for g in graded) / n,
            "items": graded}
