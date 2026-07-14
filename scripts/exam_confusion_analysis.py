"""Read-only exam deep-dive for a certification run that predates automatic
diagnostics (the exam now writes analysis/ itself via certify.diagnostics —
this script only exists to backfill older runs).

Recomputes the exact Block-A distance matrices with the DEPLOYED exam code
(appearance_distance_matrix / motion_distance_matrices / retrieval_eval /
pool_margin_exam — imported verbatim, nothing reimplemented) and persists
what exam.json strips, in the same schema certify.diagnostics writes.

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
from diffusion.transition_eval.certify.diagnostics import (           # noqa: E402
    class_distance_matrices, clip_tags, per_clip_rows)


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
                                  short_side=versioning.PINS["feature_short_side"],
                                  need_frames=False)
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
