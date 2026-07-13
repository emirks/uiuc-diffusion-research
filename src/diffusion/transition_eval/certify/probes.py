"""Constructed-truth probes (SPEC §6.2, Block B) — items whose correct score is
KNOWN by construction, so the metrics are graded against truth, not vibes.

Roster (bars.yaml `probes`):
  siblings      — all within-class pairs feed the content-invariance audit via
                  cached-feature statistics (deployed set_similarity, no full
                  scoring); hard bars attach only to the max-endpoint-distance
                  pair per class, which runs through full score.py (bar 2).
  controls      — the sibling items' auto-synthesized lerp/static-hold arms:
                  M1a floor + degeneracy/timing flags (bar 3).
  copy splices  — reference NON-CORE frames into a gen-mid segment (core-frame
                  splices sit outside M2a's comparison pool and would fail
                  spuriously); verbatim + ONE pinned deterministic perturbation
                  (re-rendered-copy proxy); honest set = bar-pair sibling M2a;
                  tau_copy := gap midpoint, set here, tested in Block C (bar 4).
  reversal      — reversed-reference M1b drop on pre-enumerated reversal-
                  sensitive camera bar pairs (bar 5). The reversed reference is
                  intentionally NOT a corpus clip, so this probe is graded
                  through the same pipeline functions (process_video_file +
                  camera_match) rather than score.py's manifest wrapper — the
                  metric code path is identical, only the I/O shim differs.
  m3 panel      — endpoint-swap (true prefix beats a wrong-CLASS prefix) +
                  hard-cut (constructed cut must fire max_seam_z) (bar 6).

All manifest-shaped probes are scored by the SAME score.py as real items — a
probe path through special code would certify the special code, not the
instrument. Every constructed video is written under the certification
artifact dir, never into the corpus.
"""

from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np

from ..appearance import set_similarity
from ..m1_transfer import camera_match
from ..s_structure import core_mask_v3
from ..video_io import load_frames, resize_cover_crop, write_video


# --- endpoint embeddings & sibling pair selection -----------------------------------

def endpoint_vecs(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """eA/eB exactly as S defines them (mean of conditioned-window features)."""
    f = bundle["feats"]
    n_pre, n_suf = bundle["profile"]["n_prefix"], bundle["profile"]["n_suffix"]
    eA = f[:n_pre].mean(axis=0)
    eA /= np.linalg.norm(eA) + 1e-12
    eB = f[-n_suf:].mean(axis=0) if n_suf else eA
    eB /= np.linalg.norm(eB) + 1e-12
    return eA, eB


def endpoint_distance(b1: dict, b2: dict) -> float:
    a1, e1 = endpoint_vecs(b1)
    a2, e2 = endpoint_vecs(b2)
    return float(1.0 - 0.5 * (a1 @ a2 + e1 @ e2))


def sibling_pairs(bundles: dict[str, dict], corpus: dict) -> dict:
    """Per n>=2 class: all within-class pairs + the bar pair (max endpoint
    distance — deterministic from cached features; corpus-only calibration)."""
    by_class: dict[str, list[str]] = {}
    for key in sorted(bundles):
        by_class.setdefault(corpus["clips"][key]["class"], []).append(key)
    out = {}
    for cls, keys in sorted(by_class.items()):
        if len(keys) < 2:
            continue
        pairs = list(itertools.combinations(keys, 2))
        dists = {p: endpoint_distance(bundles[p[0]], bundles[p[1]]) for p in pairs}
        bar_pair = max(sorted(pairs), key=lambda p: dists[p])
        out[cls] = {"pairs": pairs, "distances": {f"{a}|{b}": d for (a, b), d in dists.items()},
                    "bar_pair": list(bar_pair)}
    return out


def content_invariance_audit(bundles: dict[str, dict], corpus: dict,
                             pairs_by_class: dict) -> dict:
    """Required record artifact (SPEC §6.5, non-gating): within-class partial
    correlation of style-similarity (deployed M1a on core frames) vs
    endpoint/content-similarity, pooled after per-class centering. High
    correlation => M1a scores are confounded by content — the trust map would
    be fiction and the record must say so."""
    xs, ys = [], []
    per_class = {}
    for cls, info in pairs_by_class.items():
        side = corpus["classes"][cls]["sidedness"]
        sx, sy = [], []
        for a, b in info["pairs"]:
            ca, _ = core_mask_v3(bundles[a]["profile"], side)
            cb, _ = core_mask_v3(bundles[b]["profile"], side)
            style = set_similarity(bundles[a]["feats"][ca], bundles[b]["feats"][cb])
            content = 1.0 - endpoint_distance(bundles[a], bundles[b])
            sx.append(content)
            sy.append(style)
        per_class[cls] = {"n_pairs": len(sx),
                          "corr": (float(np.corrcoef(sx, sy)[0, 1])
                                   if len(sx) >= 3 and np.std(sx) > 1e-9 and np.std(sy) > 1e-9
                                   else None)}
        if len(sx) >= 2:
            xs.extend(np.asarray(sx) - np.mean(sx))     # per-class centering
            ys.extend(np.asarray(sy) - np.mean(sy))
    pooled = (float(np.corrcoef(xs, ys)[0, 1])
              if len(xs) >= 8 and np.std(xs) > 1e-9 and np.std(ys) > 1e-9 else None)
    return {"pooled_partial_corr": pooled, "n_pairs": len(xs), "per_class": per_class}


# --- manifest builders (scored by score.py) ------------------------------------------

def _clip_item(corpus_root: str, key: str, ref_key: str, cls: str,
               item_id: str, arm: str, n_pre: int = 9, n_suf: int = 8,
               notes: str = "") -> dict:
    """A real clip playing the role of 'generation': conditions are its own
    prefix/suffix windows (score.py slices the condition video itself)."""
    path = f"{corpus_root}/{key}"
    return {"item_id": item_id, "generated_video": path,
            "reference_video": f"{corpus_root}/{ref_key}", "style": cls,
            "n_endpoints": 2 if n_suf else 1,
            "condition_prefix": {"video": path, "num_frames": n_pre},
            "condition_suffix": {"video": path, "num_frames": n_suf} if n_suf else None,
            "arm": arm, "twin_of": None, "notes": notes}


def build_sibling_manifest(pairs_by_class: dict, corpus: dict) -> list[dict]:
    root = corpus["corpus_root"]
    return [_clip_item(root, a, b, cls, f"sib__{cls}", "probe_sibling",
                       notes=f"bar pair; endpoint_dist={info['distances'][f'{a}|{b}']:.4f}")
            for cls, info in pairs_by_class.items()
            for a, b in [tuple(info["bar_pair"])]]


def build_swap_manifest(pairs_by_class: dict, corpus: dict) -> list[dict]:
    """Endpoint-swap (M3a known-answer): same items as siblings but the prefix
    condition points at a wrong-CLASS clip (deterministic cyclic pick). The
    matching sibling item is the 'true' arm of the comparison."""
    root = corpus["corpus_root"]
    classes = sorted(pairs_by_class)
    out = []
    for k, cls in enumerate(classes):
        a, b = pairs_by_class[cls]["bar_pair"]
        wrong_cls = classes[(k + len(classes) // 2) % len(classes)]
        if wrong_cls == cls:
            wrong_cls = classes[(k + 1) % len(classes)]
        wa, _ = pairs_by_class[wrong_cls]["bar_pair"]
        it = _clip_item(root, a, b, cls, f"swap__{cls}", "probe_swap",
                        notes=f"wrong-class prefix from {wa}")
        it["condition_prefix"] = {"video": f"{root}/{wa}", "num_frames": 9}
        out.append(it)
    return out


def build_splice(gen_path: pathlib.Path, ref_path: pathlib.Path,
                 ref_core: np.ndarray, out_path: pathlib.Path,
                 segment_frames: int, n_prefix: int = 9, n_suffix: int = 8,
                 perturb: dict | None = None, short_side: int = 256) -> pathlib.Path:
    """Reference NON-CORE frames spliced into the center of the gen mid window.
    Core-frame splices would sit outside M2a's comparison pool (it scans ref
    non-core only) and fail spuriously — construction pinned by SPEC §6.2.
    `perturb` (bars.probes.copy_splices.perturbation): deterministic center-crop
    + fixed per-channel color gains — the re-rendered-copy proxy."""
    gen, fps = load_frames(gen_path, short_side=short_side)
    ref, _ = load_frames(ref_path, short_side=short_side)
    src_idx = np.flatnonzero(~ref_core)
    if len(src_idx) == 0:
        raise ValueError(f"{ref_path}: reference has no non-core frames")
    T = len(gen)
    mid_lo, mid_hi = n_prefix, T - n_suffix
    seg_len = min(segment_frames, mid_hi - mid_lo)
    start = mid_lo + (mid_hi - mid_lo - seg_len) // 2
    take = src_idx[np.arange(seg_len) % len(src_idx)]
    seg = ref[take]
    seg = resize_cover_crop(seg, gen.shape[1], gen.shape[2])
    if perturb:
        h, w = seg.shape[1:3]
        ch, cw = int(h * perturb["crop_frac"]), int(w * perturb["crop_frac"])
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        seg = resize_cover_crop(seg[:, y0:y0 + ch, x0:x0 + cw], h, w)
        gains = np.array(perturb["color_gains"], dtype=np.float32)
        seg = np.clip(seg.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    out = gen.copy()
    out[start:start + seg_len] = seg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(out, out_path, fps=fps)
    return out_path


def build_splice_manifests(pairs_by_class: dict, corpus: dict, bundles: dict,
                           probe_dir: pathlib.Path, repo_root: pathlib.Path,
                           bars: dict) -> list[dict]:
    root = corpus["corpus_root"]
    seg = bars["probes"]["copy_splices"]["segment_frames"]
    pert = bars["probes"]["copy_splices"]["perturbation"]
    out = []
    for cls, info in sorted(pairs_by_class.items()):
        a, b = info["bar_pair"]
        side = corpus["classes"][cls]["sidedness"]
        ref_core, _ = core_mask_v3(bundles[b]["profile"], side)
        for tag, p in (("verbatim", None), ("perturbed", pert)):
            vp = probe_dir / f"splice_{tag}__{cls}.mp4"
            build_splice(repo_root / root / a, repo_root / root / b, ref_core, vp,
                         segment_frames=seg, perturb=p)
            it = _clip_item(root, a, b, cls, f"splice_{tag}__{cls}", f"probe_splice_{tag}",
                            notes=f"ref non-core segment ({tag}) spliced into {a}")
            it["generated_video"] = str(vp)
            out.append(it)
    return out


def build_hard_cut(a_path: pathlib.Path, b_path: pathlib.Path,
                   out_path: pathlib.Path, n_prefix: int = 9,
                   short_side: int = 256) -> pathlib.Path:
    """Prefix of clip A + remainder of a different clip B: a guaranteed content
    discontinuity exactly at the conditioning handoff (M3b known-answer)."""
    a, fps = load_frames(a_path, short_side=short_side)
    b, _ = load_frames(b_path, short_side=short_side)
    b = resize_cover_crop(b, a.shape[1], a.shape[2])
    cut = np.concatenate([a[:n_prefix], b[n_prefix:len(a)]])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(cut, out_path, fps=fps)
    return out_path


def build_hardcut_manifests(pairs_by_class: dict, corpus: dict,
                            probe_dir: pathlib.Path, repo_root: pathlib.Path) -> list[dict]:
    root = corpus["corpus_root"]
    out = []
    for cls, info in sorted(pairs_by_class.items()):
        a, b = info["bar_pair"]
        vp = probe_dir / f"hardcut__{cls}.mp4"
        build_hard_cut(repo_root / root / a, repo_root / root / b, vp)
        it = _clip_item(root, a, b, cls, f"hardcut__{cls}", "probe_hardcut",
                        n_suf=0, notes=f"prefix of {a} + body of {b}: cut at handoff")
        it["generated_video"] = str(vp)
        out.append(it)
    return out


# --- reversal (Block B bar 5) --------------------------------------------------------

def reversed_cam(cam: dict) -> dict:
    """Camera trajectory of the time-reversed video, derived analytically for
    ENUMERATION only (corpus-only calibration): step order flips, per-step
    motion negates. The probe itself re-tracks real reversed videos."""
    return {"params": -cam["params"][::-1], "Ms": cam["Ms"][::-1],
            "ts": cam["ts"][::-1], "n_points": cam["n_points"][::-1],
            "valid": cam["valid"][::-1]}


def reversal_sensitivity(cam: dict) -> float:
    """Self-reversal distance UNDER THE DEPLOYED STATISTIC. Note the deployed
    M1b z-norms each channel, so any time-antisymmetric velocity profile —
    constant pans and palindromic moves included — is metrically identical to
    its reverse and scores ~0 here. That is the blindness this enumeration
    exists to keep out of bar 5's denominator (SPEC §6.2)."""
    m = camera_match(cam, reversed_cam(cam))
    return m["cam_dtw"] if m["cam_valid"] else float("nan")


def enumerate_reversal_pairs(pairs_by_class: dict, corpus: dict,
                             cams: dict[str, dict], bars: dict) -> list[dict]:
    """Camera-tagged bar pairs where BOTH clips are reversal-sensitive."""
    thresh = bars["probes"]["reversal"]["sensitivity_dtw_min"]
    out = []
    for cls, info in sorted(pairs_by_class.items()):
        if "camera" not in corpus["classes"][cls].get("tags", []):
            continue
        a, b = info["bar_pair"]
        sa, sb = reversal_sensitivity(cams[a]), reversal_sensitivity(cams[b])
        if np.isfinite(sa) and np.isfinite(sb) and sa >= thresh and sb >= thresh:
            out.append({"class": cls, "gen": a, "ref": b,
                        "self_reversal": {a: float(sa), b: float(sb)}})
    return out


def build_reversed_video(src: pathlib.Path, out_path: pathlib.Path,
                         short_side: int = 256) -> pathlib.Path:
    frames, fps = load_frames(src, short_side=short_side)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(frames[::-1].copy(), out_path, fps=fps)
    return out_path


def grade_reversal(pairs: list[dict], cams: dict[str, dict],
                   rev_cams: dict[str, dict], bars: dict) -> dict:
    """Bar 5: cam_corr(gen, ref) must exceed cam_corr(gen, reversed ref) on the
    sensitive pairs. Decision rule (frozen, parameterized by enumerated n):
    one-sided sign test at alpha if n >= n_signtest_min, else every pair must
    drop."""
    from .exam import sign_test_p

    rows = []
    for p in pairs:
        unrev = camera_match(cams[p["gen"]], cams[p["ref"]])
        rev = camera_match(cams[p["gen"]], rev_cams[p["ref"]])
        drop = (unrev["cam_corr"] - rev["cam_corr"]
                if unrev["cam_valid"] and rev["cam_valid"] else float("nan"))
        rows.append({**p, "corr_unreversed": unrev["cam_corr"],
                     "corr_reversed": rev["cam_corr"], "drop": drop})
    graded = [r for r in rows if np.isfinite(r["drop"])]
    wins = sum(1 for r in graded if r["drop"] > 1e-9)
    losses = sum(1 for r in graded if r["drop"] < -1e-9)   # ties excluded (standard)
    n_min = bars["probes"]["reversal"]["bar5"]["n_signtest_min"]
    alpha = bars["probes"]["reversal"]["bar5"]["alpha"]
    if len(graded) >= n_min:
        p = sign_test_p(wins, losses)
        ok = p < alpha
        rule = f"sign test (n={len(graded)}, p={p:.4f}, alpha={alpha})"
    else:
        ok = losses == 0 and wins >= 1
        rule = f"all-must-drop (n={len(graded)} < {n_min})"
    return {"pass": bool(ok), "rule": rule, "wins": wins, "losses": losses,
            "median_drop": float(np.median([r["drop"] for r in graded])) if graded else None,
            "rows": rows}


# --- graders over score.py rows ------------------------------------------------------

def _rows_by_id(items_jsonl: pathlib.Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in
            (json.loads(l) for l in pathlib.Path(items_jsonl).read_text().splitlines() if l.strip())}


def grade_siblings(rows: dict[str, dict], classes: list[str], bars: dict) -> dict:
    """Bar 2: on the bar pair, M1a beats the class's own control arm AND M2a
    stays silent. Count-form; misses documented, never rescued."""
    per = {}
    for cls in classes:
        sib = rows.get(f"sib__{cls}")
        ctrl = next((rows[k] for k in rows if k.endswith(f"__sib__{cls}")
                     and rows[k]["arm"].startswith("control")), None)
        if sib is None or ctrl is None:
            per[cls] = {"pass": False, "reason": "missing sibling or control row"}
            continue
        sep = (np.isfinite(sib.get("app_ref", np.nan)) and np.isfinite(ctrl.get("app_ref", np.nan))
               and sib["app_ref"] > ctrl["app_ref"])
        silent = sib.get("near_copy") is False
        per[cls] = {"pass": bool(sep and silent), "m1a": sib.get("app_ref"),
                    "m1a_control": ctrl.get("app_ref"), "near_copy": sib.get("near_copy"),
                    "copy_max": sib.get("copy_max")}
    n_pass = sum(1 for v in per.values() if v["pass"])
    need = bars["probes"]["siblings"]["bar2"]["min_classes"]
    return {"pass": bool(n_pass >= need), "n_pass": n_pass, "n_classes": len(classes),
            "min_classes": need, "per_class": per,
            "misses": {c: v for c, v in per.items() if not v["pass"]}}


def grade_controls(rows: dict[str, dict], classes: list[str], bars: dict) -> dict:
    """Bar 3 (draft.8): the control arm lands at the floor — M1a below its
    sibling, the floor claim in the same units. Count-form. The draft.7
    core_degenerate conjunct is REMOVED (owner decision, 2026-07-13): it was
    vacuous as a truth claim — trivially true on holds (a static hold has no
    non-endpoint frames by construction) and legitimately false on lerps
    (DINO places pixel blends far from both endpoints, earning a real strict
    core). The flag stays on every row and in the exam's mask-adoption
    criterion, where it does carry information; here it is descriptive."""
    per = {}
    for cls in classes:
        sib = rows.get(f"sib__{cls}")
        ctrl = next((rows[k] for k in rows if k.endswith(f"__sib__{cls}")
                     and rows[k]["arm"].startswith("control")), None)
        if sib is None or ctrl is None:
            per[cls] = {"pass": False, "reason": "missing row"}
            continue
        floored = (np.isfinite(ctrl.get("app_ref", np.nan))
                   and np.isfinite(sib.get("app_ref", np.nan))
                   and ctrl["app_ref"] < sib["app_ref"])
        per[cls] = {"pass": bool(floored),
                    "control_m1a": ctrl.get("app_ref"), "sibling_m1a": sib.get("app_ref"),
                    "core_degenerate": ctrl.get("core_degenerate")}
    n_pass = sum(1 for v in per.values() if v["pass"])
    need = bars["probes"]["controls"]["bar3"]["min_classes"]
    return {"pass": bool(n_pass >= need), "n_pass": n_pass, "n_classes": len(classes),
            "min_classes": need, "per_class": per,
            "misses": {c: v for c, v in per.items() if not v["pass"]}}


def grade_splices(rows: dict[str, dict], classes: list[str], bars: dict) -> dict:
    """Bar 4: every splice (verbatim AND perturbed) >= tau_copy; gap between
    splice min and honest (bar-pair sibling) max >= frozen minimum. tau_copy is
    re-frozen at the gap midpoint — SET here, TESTED in Block C."""
    splices, honest = [], []
    for cls in classes:
        for tag in ("verbatim", "perturbed"):
            r = rows.get(f"splice_{tag}__{cls}")
            if r and np.isfinite(r.get("copy_max", np.nan)):
                splices.append((f"{tag}__{cls}", r["copy_max"]))
        s = rows.get(f"sib__{cls}")
        if s and np.isfinite(s.get("copy_max", np.nan)):
            honest.append((cls, s["copy_max"]))
    if not splices or not honest:
        return {"pass": False, "reason": "no graded splice or honest rows"}
    smin, hmax = min(v for _, v in splices), max(v for _, v in honest)
    tau0 = bars["probes"]["copy_splices"]["tau_copy_initial"]
    gap = smin - hmax
    detect_all = all(v >= tau0 for _, v in splices)
    ok = detect_all and gap >= bars["probes"]["copy_splices"]["bar4"]["min_gap"]
    return {"pass": bool(ok), "detect_all_at_initial_tau": bool(detect_all),
            "splice_min": float(smin), "honest_max": float(hmax), "gap": float(gap),
            "tau_recalibrated": float(0.5 * (smin + hmax)),
            "n_splices": len(splices), "n_honest": len(honest),
            "below_tau": [k for k, v in splices if v < tau0]}


def grade_m3_panel(rows: dict[str, dict], classes: list[str], bars: dict) -> dict:
    """Bar 6: endpoint-swap (true prefix_dino > swapped prefix_dino) and
    hard-cut (max_seam_z > 3). Both count-form."""
    swap_per, cut_per = {}, {}
    for cls in classes:
        sib, sw = rows.get(f"sib__{cls}"), rows.get(f"swap__{cls}")
        if sib and sw and np.isfinite(sib.get("prefix_dino", np.nan)) \
                and np.isfinite(sw.get("prefix_dino", np.nan)):
            swap_per[cls] = {"pass": bool(sib["prefix_dino"] > sw["prefix_dino"]),
                             "true": sib["prefix_dino"], "swapped": sw["prefix_dino"]}
        else:
            swap_per[cls] = {"pass": False, "reason": "missing row"}
        hc = rows.get(f"hardcut__{cls}")
        if hc and np.isfinite(hc.get("max_seam_z", np.nan)):
            cut_per[cls] = {"pass": bool(hc["max_seam_z"] > 3.0), "max_seam_z": hc["max_seam_z"]}
        else:
            cut_per[cls] = {"pass": False, "reason": "missing row"}
    ns, nc = (sum(1 for v in d.values() if v["pass"]) for d in (swap_per, cut_per))
    need_s = bars["probes"]["m3_panel"]["bar6_endpoint_swap"]["min_classes"]
    need_c = bars["probes"]["m3_panel"]["bar6_hard_cut"]["min_classes"]
    return {"pass": bool(ns >= need_s and nc >= need_c),
            "swap": {"n_pass": ns, "min_classes": need_s, "per_class": swap_per},
            "hard_cut": {"n_pass": nc, "min_classes": need_c, "per_class": cut_per}}
