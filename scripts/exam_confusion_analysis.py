"""Read-only exam deep-dive for a finished certification run.

Recomputes the exact Block-A distance matrices with the DEPLOYED exam code
(appearance_distance_matrix / motion_distance_matrices / retrieval_eval /
pool_margin_exam — imported verbatim, nothing reimplemented) and persists
what exam.json strips: full confusion matrices, per-clip 1-NN predictions
with distances and margins, class-pair distance matrices, R2 per-clip
margins + intruder classes, and per-clip transition tags parsed from the
corpus source paths.

Pure analysis: touches no instrument code, writes only under the cert dir.

    python scripts/exam_confusion_analysis.py \
        --cert-dir outputs/eval/certification/3.0.0-draft.8 \
        --corpus data/processed/transitions_std121/corpus_manifest.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.transition_eval import versioning                      # noqa: E402
from diffusion.transition_eval.manifests_v3 import load_corpus_manifest  # noqa: E402
from diffusion.transition_eval.pipeline import process_video_file     # noqa: E402
from diffusion.transition_eval.report import retrieval_eval           # noqa: E402
from diffusion.transition_eval.certify.exam import (                  # noqa: E402
    CORE_VARIANTS, appearance_distance_matrix, motion_distance_matrices,
    pool_margin_exam)

TAG_WORDS = ("object", "style", "camera")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert-dir", required=True)
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()
    cert_dir = pathlib.Path(args.cert_dir)
    out_dir = cert_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    corpus = load_corpus_manifest(pathlib.Path(args.corpus).resolve())

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[analysis] device={device} (cache-warm: GPU never needed)")
    from diffusion.transition_eval.features import DinoExtractor
    from diffusion.transition_eval.motion import Tracker
    extractor = DinoExtractor(versioning.PINS["dino_model"], device=device)
    tracker = Tracker(device=device)
    cache_dir = REPO_ROOT / "outputs/eval/cache"
    probe_root = corpus["corpus_root"]

    keys = sorted(corpus["clips"])
    bundles = []
    for i, key in enumerate(keys):
        b, _ = process_video_file(REPO_ROOT / probe_root / key, cache_dir,
                                  extractor, tracker,
                                  short_side=versioning.PINS["feature_short_side"])
        bundles.append(b)
        if (i + 1) % 40 == 0:
            print(f"[analysis]   corpus {i + 1}/{len(keys)}")
    labels = [corpus["clips"][k]["class"] for k in keys]
    sidedness = [corpus["classes"][l]["sidedness"] for l in labels]

    result = {
        "clips": [{"key": k, "class": l, "sidedness": s,
                   "tags": clip_tags(corpus["clips"][k]["source"])}
                  for k, l, s in zip(keys, labels, sidedness)],
        "metrics": {},
    }
    mats = {}

    print("[analysis] appearance matrices")
    for v in CORE_VARIANTS:
        D = appearance_distance_matrix(bundles, v, sidedness)
        mats[f"m1a__{v}"] = D
        result["metrics"][f"m1a__{v}"] = {
            "retrieval": retrieval_eval(D, labels),
            "rows": per_clip_rows(D, keys, labels),
            "class_dist": class_distance_matrices(D, labels),
        }
        print(f"[analysis]   m1a__{v} done")

    print("[analysis] motion matrices (DTW pairwise — the slow part)")
    Dm = motion_distance_matrices(bundles)
    for name, D in Dm.items():
        mats[name] = D
        result["metrics"][name] = {
            "retrieval": retrieval_eval(D, labels),
            "rows": per_clip_rows(D, keys, labels),
            "class_dist": class_distance_matrices(D, labels),
        }
        print(f"[analysis]   {name} done")

    exam_json = json.load(open(cert_dir / "exam" / "exam.json"))
    winner = exam_json["mask_adoption"]["winner"]
    print(f"[analysis] R2 pool margins under winner mask ({winner})")
    r2 = pool_margin_exam(bundles, labels, sidedness, winner)
    result["r2"] = {"winner_mask": winner, **r2,
                    "rows": [{**r, "key": k} for r, k in zip(r2["rows"], keys)]}

    np.savez_compressed(out_dir / "distance_matrices.npz",
                        keys=np.array(keys), **mats)
    (out_dir / "analysis.json").write_text(json.dumps(result, default=str))
    print(f"[analysis] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
