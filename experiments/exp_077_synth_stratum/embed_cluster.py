"""exp_077 TASK 0e (GPU analysis) — effective-K of the operator bank.

Embeds every rendered audit clip with the v4 metric's frozen DINOv2 encoder
(src/diffusion/transition_eval/features.py, facebook/dinov2-base, per-frame L2-normalized
CLS), reduces each clip to a transition descriptor (mean-pooled features over 3 temporal
thirds of the transition window, endpoints excluded since they are identical across every
operator on the fixed pair), and clusters the descriptors. Reports effective-K (distinguishable
operator clusters) vs the nominal rendered-operator count, at several cosine-distance cuts.

    python embed_cluster.py --run <audit run_dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from engine import videoio  # noqa: E402


def clip_descriptor(feats: np.ndarray, lo: int, hi: int, bins: int = 3) -> np.ndarray:
    """[F,D] per-frame features -> concat of `bins` L2-normalized mean-pooled thirds of [lo,hi]."""
    edges = np.linspace(lo, hi, bins + 1).astype(int)
    parts = []
    for b in range(bins):
        seg = feats[edges[b]:max(edges[b] + 1, edges[b + 1])]
        v = seg.mean(0)
        v = v / (np.linalg.norm(v) + 1e-8)
        parts.append(v)
    d = np.concatenate(parts)
    return d / (np.linalg.norm(d) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="audit run_dir (holds videos/ + rendered_index.json)")
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--anchor", type=int, default=9)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    idx = json.loads((run / "rendered_index.json").read_text())
    vids = [run / "videos" / f"{r['stem']}.mp4" for r in idx]
    print(f"[embed] {len(vids)} rendered clips from {run}")

    from diffusion.transition_eval.features import DinoExtractor
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = DinoExtractor(device=dev)
    print(f"[embed] DinoExtractor {ext.model_name} on {dev}")

    T, K = args.num_frames, args.anchor
    lo, hi = K + 2, T - K - 2                      # transition window, endpoints excluded
    sample_idx = np.linspace(lo, hi, 15).astype(int)

    descs = []
    for i, (r, vp) in enumerate(zip(idx, vids)):
        frames = videoio.read_clip(vp)             # [T,H,W,3] uint8
        sel = frames[sample_idx]
        feats = ext.extract(sel)                   # [15, D]
        descs.append(clip_descriptor(feats, 0, len(sel) - 1, bins=3))
        if (i + 1) % 25 == 0:
            print(f"  embedded {i + 1}/{len(vids)}")
    D = np.stack(descs)                            # [N, 3D]
    print(f"[embed] descriptors {D.shape}")

    # pairwise cosine similarity (descriptors already L2-normalized)
    S = D @ D.T
    iu = np.triu_indices(len(D), k=1)
    off = S[iu]
    sim_stats = {"mean": float(off.mean()), "p10": float(np.percentile(off, 10)),
                 "median": float(np.percentile(off, 50)), "p90": float(np.percentile(off, 90)),
                 "max": float(off.max()), "min": float(off.min())}

    # effective-K via average-linkage agglomerative clustering, cut at cosine-distance thresholds
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - S
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    K_by_cut = {}
    for cut in (0.05, 0.10, 0.15, 0.20, 0.25):
        labels = fcluster(Z, t=cut, criterion="distance")
        K_by_cut[f"{cut:.2f}"] = int(labels.max())

    # silhouette-selected K (numpy silhouette on cosine distance), K in a sensible range
    def silhouette(labels: np.ndarray) -> float:
        uniq = np.unique(labels)
        if len(uniq) < 2 or len(uniq) >= len(labels):
            return -1.0
        sc = []
        for i in range(len(labels)):
            same = labels == labels[i]
            same[i] = False
            a = dist[i, same].mean() if same.any() else 0.0
            b = min(dist[i, labels == c].mean() for c in uniq if c != labels[i])
            sc.append((b - a) / max(a, b, 1e-8))
        return float(np.mean(sc))

    sil = {}
    best = (-1.0, None)
    for k in range(2, min(len(D), 40)):
        labels = fcluster(Z, t=k, criterion="maxclust")
        s = silhouette(labels)
        sil[k] = round(s, 4)
        if s > best[0]:
            best = (s, k)

    out = {
        "n_rendered_operators": int(len(D)),
        "descriptor_dim": int(D.shape[1]),
        "pairwise_cosine_sim": sim_stats,
        "effective_K_by_cosine_distance_cut": K_by_cut,
        "silhouette_by_K": sil,
        "silhouette_best_K": best[1],
        "silhouette_best_score": round(best[0], 4),
    }
    (run / "effective_K.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[done] effective_K.json -> {run}")


if __name__ == "__main__":
    main()
