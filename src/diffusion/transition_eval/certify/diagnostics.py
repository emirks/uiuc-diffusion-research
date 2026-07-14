"""Exam diagnostic state — everything Block A computes but the record strips.

The exam's verdict path keeps only accuracies and adoption checks; this module
persists the rest for inspection after every certification run: full confusion
matrices, per-clip 1-NN predictions with distances and margins, class-pair
distance matrices, R2 per-clip margins with intruder classes, and R1 accuracy
broken out per transition-tag group. Written to <cert_dir>/analysis/ in the
schema `certify.explorer` and `certify.figures` read.

Representation only: consumes finished matrices, never feeds a verdict — the
driver treats any failure here as non-gating.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

TAG_WORDS = ("object", "style", "camera")
COARSE_GROUPS = ("twosided", "onesided", "object", "style", "camera")


def clip_tags(source: str) -> list[str]:
    """twosided_object_camera_air-bending -> [object, camera] (sidedness aside)."""
    parts = pathlib.Path(source).parent.name.split("_")
    return [p for p in parts[1:] if p in TAG_WORDS]


def per_clip_rows(D: np.ndarray, keys: list[str], labels: list[str]) -> list[dict]:
    """Mirror retrieval_eval's masking exactly; add distances and margins."""
    n = len(labels)
    M = D.copy().astype(float)
    np.fill_diagonal(M, np.inf)
    M[np.isnan(M)] = np.inf
    lab = np.array(labels)
    rows = []
    for i in range(n):
        if not np.isfinite(M[i]).any():
            rows.append({"key": keys[i], "label": labels[i], "pred": None})
            continue
        j = int(np.argmin(M[i]))
        same = (lab == labels[i]) & np.isfinite(M[i])
        same[i] = False
        other = (lab != labels[i]) & np.isfinite(M[i])
        d_within = float(M[i][same].min()) if same.any() else None
        d_cross = float(M[i][other].min()) if other.any() else None
        rows.append({
            "key": keys[i], "label": labels[i], "pred": labels[j],
            "nn_key": keys[j], "nn_dist": float(M[i, j]),
            "d_within_min": d_within, "d_cross_min": d_cross,
            "margin": (d_cross - d_within)
                      if d_within is not None and d_cross is not None else None,
        })
    return rows


def class_distance_matrices(D: np.ndarray, labels: list[str]) -> dict:
    """Mean and min inter-clip distance per class pair, finite entries only."""
    classes = sorted(set(labels))
    lab = np.array(labels)
    mean_m, min_m = {}, {}
    for a in classes:
        ia = np.where(lab == a)[0]
        mean_m[a], min_m[a] = {}, {}
        for b in classes:
            ib = np.where(lab == b)[0]
            block = D[np.ix_(ia, ib)].astype(float).copy()
            if a == b:
                block[np.eye(len(ia), dtype=bool)] = np.nan
            vals = block[np.isfinite(block)]
            mean_m[a][b] = float(vals.mean()) if vals.size else None
            min_m[a][b] = float(vals.min()) if vals.size else None
    return {"classes": classes, "mean": mean_m, "min": min_m}


def tag_accuracy(rows_by_metric: dict[str, list[dict]], clips: list[dict]) -> dict:
    """R1 1-NN accuracy per tag group, per metric.

    Two granularities: coarse (each sidedness/tag word pools every clip that
    carries it — groups overlap) and patterns (exact sidedness+tag combination,
    a partition). Ungraded clips (pred None) are excluded from the denominator.
    """
    patterns = sorted({c["sidedness"] + ("_" + "_".join(c["tags"]) if c["tags"] else "")
                       for c in clips})
    out: dict[str, list[dict]] = {"coarse": [], "patterns": []}
    for kind, names in (("coarse", COARSE_GROUPS), ("patterns", patterns)):
        for gname in names:
            if kind == "coarse":
                keys = [c["key"] for c in clips if gname in (c["sidedness"], *c["tags"])]
            else:
                keys = [c["key"] for c in clips
                        if c["sidedness"] + ("_" + "_".join(c["tags"]) if c["tags"] else "") == gname]
            row: dict = {"group": gname, "n": len(keys)}
            for mname, rows in rows_by_metric.items():
                by_key = {r["key"]: r for r in rows}
                graded = [by_key[k] for k in keys if by_key[k]["pred"] is not None]
                row[mname] = (round(float(np.mean([r["pred"] == r["label"] for r in graded])), 4)
                              if graded else None)
            out[kind].append(row)
    return out


def build_analysis(corpus: dict, keys: list[str], labels: list[str],
                   sidedness: list[str], r1: dict, mats: dict[str, np.ndarray],
                   r2: dict, winner: str) -> dict:
    """Assemble the full diagnostic dict (analysis.json schema + by_tag)."""
    clips = [{"key": k, "class": l, "sidedness": s,
              "tags": clip_tags(corpus["clips"][k]["source"])}
             for k, l, s in zip(keys, labels, sidedness)]
    metrics, rows_by_metric = {}, {}
    for name, D in mats.items():
        rows = per_clip_rows(D, keys, labels)
        rows_by_metric[name] = rows
        metrics[name] = {"retrieval": r1[name], "rows": rows,
                         "class_dist": class_distance_matrices(D, labels)}
    return {
        "clips": clips,
        "metrics": metrics,
        "r2": {"winner_mask": winner, **r2,
               "rows": [{**r, "key": k} for r, k in zip(r2["rows"], keys)]},
        "by_tag": tag_accuracy(rows_by_metric, clips),
    }


def write_analysis(out_dir: pathlib.Path, analysis: dict,
                   mats: dict[str, np.ndarray], keys: list[str]) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "distance_matrices.npz",
                        keys=np.array(keys), **mats)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, default=str))
    return out_dir / "analysis.json"
