"""Adversarial ground-truth probes (SPEC §6.2) — synthetic items whose correct
score is KNOWN, so the integrity metrics are graded against truth, not vibes.

copy splices  (M2a): take an existing honest generation, replace its middle
    segment with the REFERENCE's middle segment (resize-cover-crop to match).
    Ground truth: contains literal reference content -> copy_max must clear
    tau_copy. The splice-min / honest-max gap recalibrates tau_copy
    (exp_053 check-C pattern, now reference-local instead of corpus-wide).

cross-label   (M2b): rescore existing generations under a deliberately WRONG
    reference class that is visually FAR from the true class (bottom quartile
    of class-centroid set-similarity — the automatic, pre-registered stand-in
    for hand-labeled texture families; cousins legitimately share texture, the
    exp_057 lesson, and the quartile rule keeps them out by construction).
    Ground truth: the margin must go negative and the argmax must name the
    class the item actually imitates.

endpoint swap (M3a): rescore existing conditioned items against a DIFFERENT
    item's condition clip. Ground truth: the true pairing must beat the
    swapped pairing on prefix_dino for every pair.

hard cut      (M3b): corpus clip A's conditioned prefix + clip B's remainder,
    concatenated with no blend — a ground-truth seam at the handoff index.
    max_seam_z must fire.

controls      (floors/flags): lerp (two-sided) and static-hold (one-sided)
    arms must land at the transfer floor and trip core_degenerate/timing flags.

All builders emit standard v3 eval manifests + probe videos AT THE §2 CONTRACT
resolution — probes are scored by the SAME score.py as real items (a probe
path through special code would certify the special code, not the instrument;
a probe video below contract resolution would be rejected by score.py's input
contract, correctly).

STATUS: builders implemented; execution is part of the certification run
(needs GPU featurization for spliced/hard-cut videos).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from ..video_io import load_frames, resize_cover_crop, write_video


def build_copy_splice(gen_path: pathlib.Path, ref_path: pathlib.Path,
                      out_path: pathlib.Path, n_prefix: int = 9, n_suffix: int = 8,
                      short_side: int | None = None) -> pathlib.Path:
    """Honest generation with its mid-segment replaced by the reference's —
    a ground-truth copy that keeps the original conditioned windows intact.
    Built at native (contract) resolution so score.py's §2 check passes."""
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


def pick_far_class(true_cls: str, class_affinity: dict[str, float],
                   corpus: dict, quartile: float = 0.25) -> str | None:
    """Pre-registered wrong-class rule: from the bottom `quartile` of classes
    by centroid set-similarity to `true_cls` (computed once from the cached
    corpus features), eligible if n_clips >= 2, pick the MEDIAN element —
    deterministic, clearly-different but not cherry-picked-easiest. Cousins
    (high affinity) are excluded by construction."""
    ranked = sorted((s, c) for c, s in class_affinity.items()
                    if c != true_cls and corpus["classes"][c]["n_clips"] >= 2)
    cut = max(1, int(len(ranked) * quartile))
    pool = ranked[:cut]
    return pool[len(pool) // 2][1] if pool else None


def build_cross_label_manifest(items: list[dict], corpus: dict,
                               class_affinity: dict[str, dict[str, float]],
                               n_items: int | None = None) -> list[dict]:
    """Relabel existing items with a wrong-reference class chosen by
    pick_far_class (visually far ⇒ unambiguous ground truth; cousin classes
    would blur it — the exp_057 lesson that cousins legitimately share
    texture). Emits standard manifest rows with `notes` recording the true
    class for grading. class_affinity[true_cls][other] = centroid set-sim."""
    out = []
    for it in (items if n_items is None else items[:n_items]):
        true_cls = it["style"]
        if true_cls not in class_affinity:
            continue
        wrong = pick_far_class(true_cls, class_affinity[true_cls], corpus)
        if wrong is None:
            continue
        ref = f"{wrong}/{sorted(k.split('/')[1] for k in corpus['clips'] if k.startswith(wrong + '/'))[0]}"
        out.append({**it,
                    "item_id": f"xlabel__{it['item_id']}",
                    "style": wrong,
                    "reference_video": f"{corpus['corpus_root']}/{ref}",
                    "notes": f"cross-label probe; true_class={true_cls}"})
    return out


def build_endpoint_swap_manifest(items: list[dict], n_items: int | None = None) -> list[dict]:
    """M3a probe: rescore each conditioned item against the NEXT item's
    condition clips (cyclic pairing across different item_ids). Ground truth:
    the true pairing beats the swapped pairing on prefix_dino, per pair."""
    pool = [it for it in (items if n_items is None else items[:n_items])
            if it.get("condition_prefix")]
    out = []
    for i, it in enumerate(pool):
        donor = pool[(i + 1) % len(pool)]
        if len(pool) < 2 or donor["item_id"] == it["item_id"]:
            continue
        row = {**it, "item_id": f"epswap__{it['item_id']}",
               "condition_prefix": donor["condition_prefix"],
               "notes": f"endpoint-swap probe; true_item={it['item_id']}"}
        if it.get("condition_suffix") and donor.get("condition_suffix"):
            row["condition_suffix"] = donor["condition_suffix"]
        out.append(row)
    return out


def build_hard_cut(clip_a: pathlib.Path, clip_b: pathlib.Path,
                   out_path: pathlib.Path, cut_at: int = 9) -> pathlib.Path:
    """M3b probe: A's first `cut_at` frames + B's remainder, no blend — a
    ground-truth seam exactly at the prefix handoff index. Built at native
    (contract) resolution."""
    a, fps = load_frames(clip_a, short_side=None)
    b, _ = load_frames(clip_b, short_side=None)
    b = resize_cover_crop(b, a.shape[1], a.shape[2])
    frames = np.concatenate([a[:cut_at], b[cut_at:len(a)]])
    write_video(frames, out_path, fps=fps)
    return out_path


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


def grade_endpoint_swap(true_rows: list[dict], swap_rows: list[dict]) -> dict:
    """Every swapped pairing must lose to its true pairing on prefix_dino."""
    true_by_id = {r["item_id"]: r for r in true_rows}
    graded = []
    for r in swap_rows:
        tid = (r.get("notes") or "").split("true_item=")[-1]
        t = true_by_id.get(tid)
        if t is None or not (np.isfinite(r.get("prefix_dino", np.nan))
                             and np.isfinite(t.get("prefix_dino", np.nan))):
            continue
        graded.append({"item_id": r["item_id"],
                       "true_beats_swap": bool(t["prefix_dino"] > r["prefix_dino"]),
                       "gap": float(t["prefix_dino"] - r["prefix_dino"])})
    n = len(graded) or 1
    return {"n": len(graded),
            "frac_true_beats_swap": sum(g["true_beats_swap"] for g in graded) / n,
            "min_gap": min((g["gap"] for g in graded), default=None),
            "items": graded}


def grade_hard_cut(rows: list[dict], z_bar: float = 3.0) -> dict:
    """Every hard-cut probe must trip the seam flag (max_seam_z > z_bar)."""
    zs = [r.get("max_seam_z") for r in rows]
    zs = [z for z in zs if z is not None and np.isfinite(z)]
    n = len(zs) or 1
    return {"n": len(zs),
            "frac_fired": sum(z > z_bar for z in zs) / n,
            "min_z": min(zs, default=None)}
