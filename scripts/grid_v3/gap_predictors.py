#!/usr/bin/env python
"""grid v3 — which VIDEO-computable descriptors of a class's reference clips correlate with the DCG-vs-prompt gap?

Question (owner 2026-09-08): before spending GPU on a search over EffectData, is there something computable from the
effect clips THEMSELVES that predicts where the reference (dualforce + DCG) beats prompt-only (base_cond effect)?
Data: the 76 classes of grid v3 (42 Higgsfield-121f + 34 EffectData-81f) with measured per-class levels; per-clip
descriptors from the scorer's OWN cached bundles (DINO features, CoTracker tracks, morph profile) of the corpus clips
of each class — nothing new is extracted (cache warm from scoring), so this runs on a login node.
Output: Spearman correlations per descriptor vs the gaps and levels, a leave-one-out check of the best few, and a CSV.
Exploratory, in-sample, class-level (n=76): a predictor found here must be confirmed on new effects before use.
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WT = REPO / ".claude/worktrees/eval-v4-cert/src"
sys.path.insert(0, str(WT)); sys.path.insert(0, str(REPO / "eval_ladder")); sys.path.insert(0, str(REPO / "scripts/grid_v3"))
import run_eval, closeout  # noqa: E402
from diffusion.transition_eval import versioning  # noqa: E402
from diffusion.transition_eval.features import DinoExtractor  # noqa: E402
from diffusion.transition_eval.motion import Tracker  # noqa: E402
from diffusion.transition_eval.pipeline import process_video_file  # noqa: E402
from diffusion.transition_eval.m1_transfer import camera_trajectory, residual_direction_profile, object_match_from_profiles  # noqa: E402
from diffusion.transition_eval.morph import core_mask  # noqa: E402

CACHE = REPO / "misc/refvfx_baseline/probe/cache"
STD = REPO / "data/processed/transitions_std121"
OUT = REPO / "misc/2026-09-07_eval_grid_v2/eval/gap_predictors.csv"


def class_levels():
    ceil = run_eval.ceilings(); E = closeout.eval_entry()
    def per_class(ha):
        rows = {r["item_id"]: r for r in map(json.loads, filter(str.strip, open(REPO / f"eval_ladder/registry_{ha}.jsonl"))) if r["arm"] == ha}
        x = run_eval.item_pct(run_eval.pool_means(E / ha), rows, ceil); d = collections.defaultdict(list)
        for i, v in x.items():
            d[rows[i]["gt_pool_class"]].append(v)
        return {c: (st.mean(v) * 100, len(v)) for c, v in d.items()}
    out = {}
    for fam in ("v3", "v3ed81"):
        bn, be = per_class(f"base_cond_neutral_{fam}"), per_class(f"base_cond_effect_{fam}")
        gn, ge = per_class(f"dualforce_dcg_w6_neutral_{fam}"), per_class(f"dualforce_dcg_w6_effect_{fam}")
        dn, de = per_class(f"dualforce_control_neutral_{fam}"), per_class(f"dualforce_control_effect_{fam}")
        for c in be:
            out[c] = {"family": "HF" if fam == "v3" else "ED", "n_rows": be[c][1], "ceiling": ceil.get(c),
                      "base_neu": bn[c][0], "base_eff": be[c][0], "dcg_neu": gn[c][0], "dcg_eff": ge[c][0], "ctrl_eff": de[c][0],
                      "gap_eff": ge[c][0] - be[c][0], "gap_neu": gn[c][0] - bn[c][0], "delta_prompt": be[c][0] - bn[c][0]}
    return out


def clip_descriptors(b) -> dict:
    f = np.asarray(b["feats"], dtype=np.float64); T = len(f)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
    cos = lambda a, c: float(np.dot(a, c))
    step = 1.0 - np.einsum("td,td->t", f[:-1], f[1:])
    core = core_mask(b["profile"])
    sc = b["scalars"]
    d = {"endpoint_change": 1.0 - cos(f[:4].mean(0) / np.linalg.norm(f[:4].mean(0)), f[-4:].mean(0) / np.linalg.norm(f[-4:].mean(0))),
         "temporal_energy": float(step.mean()), "temporal_peak": float(step.max()), "temporal_peakiness": float(step.max() / max(step.mean(), 1e-8)),
         "core_frac_mask": float(core.mean()), "depth": float(sc["depth"]), "depart": float(sc["depart"]), "hold": float(sc["hold"]),
         "core_frac": float(sc["core_frac"]), "cross": float(b["profile"]["cross"])}
    tr, vis = b.get("tracks"), b.get("vis")
    if tr is not None and tr.shape[1] > 0:
        cam = camera_trajectory(tr, vis)
        p = cam["params"]; ok = cam["valid"] & np.isfinite(p).all(1)
        d["camera_translation"] = float(np.abs(p[ok, :2]).sum(1).mean()) if ok.any() else 0.0
        d["camera_zoom_rot"] = float((np.abs(p[ok, 2]) + np.abs(p[ok, 3])).mean()) if ok.any() else 0.0
        # raw point speed and camera-residual speed (normalized coords per frame), visible points only
        v = (vis[:-1] > 0.5) & (vis[1:] > 0.5)
        disp = np.linalg.norm(tr[1:] - tr[:-1], axis=2)
        d["point_speed"] = float(disp[v].mean()) if v.any() else 0.0
        res = tr[1:] - (np.einsum("tij,tnj->tni", cam["Ms"], tr[:-1]) + cam["ts"][:, None, :])
        rs = np.linalg.norm(res, axis=2)
        d["residual_speed"] = float(rs[v].mean()) if v.any() else 0.0
        d["moving_frac"] = float((rs[v] > 0.01).mean()) if v.any() else 0.0
        d["_profile"] = residual_direction_profile(tr, vis, cam)
    return d


def spearman(x, y):
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[m], y[m]); return float(r), float(p), int(m.sum())


def main():
    corpus = json.loads((STD / "corpus_manifest.json").read_text())["clips"]
    by_class = collections.defaultdict(list)
    for k in corpus:
        c, clip = k.split("/"); by_class[c].append(clip)
    levels = class_levels()
    print(f"[gap-predictors] {len(levels)} classes with measured levels; corpus classes {len(by_class)}", flush=True)
    device = "cpu"
    extractor = DinoExtractor(versioning.PINS["dino_model"], device=device); tracker = Tracker(device=device)
    short = versioning.PINS["feature_short_side"]
    rows = []
    for c, lv in sorted(levels.items()):
        clips = by_class.get(c, [])
        descs, profiles = [], []
        for clip in clips:
            b, _ = process_video_file(STD / c / clip, CACHE, extractor, tracker, short_side=short, need_frames=False)
            d = clip_descriptors(b); profiles.append(d.pop("_profile", None)); descs.append(d)
        agg = {k: float(np.nanmean([d[k] for d in descs if k in d])) for k in descs[0]}
        agg["desc_sd_temporal_energy"] = float(np.nanstd([d["temporal_energy"] for d in descs]))
        prs = [p for p in profiles if p is not None and len(p)]
        sims = [object_match_from_profiles(prs[i], prs[j]) for i in range(len(prs)) for j in range(len(prs)) if i != j]
        agg["motion_consistency"] = float(np.nanmean(sims)) if sims else float("nan")
        agg["n_clips"] = len(clips)
        rows.append({"class": c, **lv, **agg}); print(f"  {c:36s} clips {len(clips):2d} gap_eff {lv['gap_eff']:+6.1f}", flush=True)
    keys = [k for k in rows[0] if k not in ("class", "family")]
    import csv
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "family"] + keys); w.writeheader(); w.writerows(rows)
    print(f"\n[csv] {OUT.relative_to(REPO)}")
    descs = [k for k in keys if k not in ("n_rows", "base_neu", "base_eff", "dcg_neu", "dcg_eff", "ctrl_eff", "gap_eff", "gap_neu", "delta_prompt")]
    targets = ["gap_eff", "gap_neu", "base_eff", "delta_prompt"]
    for fam_sel, label in (("all", "ALL 76 classes"), ("HF", "Higgsfield 42"), ("ED", "EffectData 34")):
        sub = [r for r in rows if fam_sel == "all" or r["family"] == fam_sel]
        print(f"\n### Spearman ρ of class descriptors vs targets — {label}")
        print("| descriptor | " + " | ".join(f"ρ vs {t}" for t in targets) + " |"); print("|---|" + "---|" * len(targets))
        table = []
        for dname in descs:
            x = np.array([r[dname] for r in sub], dtype=float)
            cells = []
            for t in targets:
                y = np.array([r[t] for r in sub], dtype=float); rho, p, n = spearman(x, y)
                cells.append(f"{rho:+.2f}{'*' if p < 0.05 else ''}")
            rho_eff = spearman(x, np.array([r['gap_eff'] for r in sub], dtype=float))[0]
            table.append((abs(rho_eff) if np.isfinite(rho_eff) else -1, f"| {dname} | " + " | ".join(cells) + " |"))
        for _, line in sorted(table, reverse=True):
            print(line)
    # leave-one-out linear prediction of gap_eff from the top-3 |ρ| descriptors (all classes)
    x_all = {d: np.array([r[d] for r in rows], dtype=float) for d in descs}
    y = np.array([r["gap_eff"] for r in rows], dtype=float)
    ranked = sorted(descs, key=lambda d: -abs(spearman(x_all[d], y)[0]) if np.isfinite(spearman(x_all[d], y)[0]) else 0)
    for k in (1, 2, 3):
        use = ranked[:k]; X = np.column_stack([x_all[d] for d in use] + [np.ones(len(y))])
        m = np.isfinite(X).all(1); Xm, ym = X[m], y[m]; pred = np.full(len(ym), np.nan)
        for i in range(len(ym)):
            idx = np.arange(len(ym)) != i
            beta, *_ = np.linalg.lstsq(Xm[idx], ym[idx], rcond=None); pred[i] = Xm[i] @ beta
        rho = spearman(pred, ym)[0]
        print(f"\nLOO linear fit of gap_eff from {use}: Spearman(pred, actual) = {rho:+.2f} (n={m.sum()})")


if __name__ == "__main__":
    main()
