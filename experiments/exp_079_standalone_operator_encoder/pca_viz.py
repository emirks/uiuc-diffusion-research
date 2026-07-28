"""exp_079 — PCA views of the learned embeddings: does the code cluster the way we claim?

Answers, visually, the two questions the numbers assert:
  1. Do same-class clips cluster?                      (both arms should; it is the easy part)
  2. Is TEMPORAL structure encoded independently of class, and does it survive on classes the
     encoder never saw?                                (this is the certified claim — E1 only)

Panels, per arm (E1 = certified SupCon-T, ablation = plain class-only SupCon):
  A  TRAIN, PCA coloured by CLASS          -> class clustering
  B  TRAIN, PCA coloured by MANIPULATION   -> is the operator axis present at all?
  C  HELD-OUT ZS CLASSES, CLASS-MEAN-REMOVED, coloured by MANIPULATION
       The money panel. Subtracting each class's mean deletes all class/content identity, so any
       remaining structure is temporal-only, measured on classes never trained on. If manipulations
       separate here, the code carries transferable operator structure rather than a class lookup.
  D  HELD-OUT ZS, identity->reverse displacement arrows (one per clip)
       If reversal is encoded as a consistent direction, arrows align rather than scatter.

PCA is plain SVD on centred features (no sklearn in this env).

    python experiments/exp_079_standalone_operator_encoder/pca_viz.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(LAB / "LTX-2-bneck/packages/ltx-trainer/src"))
import probes  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp_079" / "figures"
MANIP_COLOR = {"identity": "#1f77b4", "reverse": "#d62728",
               "ease_in_g2": "#2ca02c", "ease_out_g05": "#ff7f0e",
               "segment_swap": "#9467bd", "block_reversal": "#8c564b"}


def pca(x: np.ndarray, k: int = 2):
    """Plain PCA via SVD. Returns (coords [N,k], explained-variance ratio [k])."""
    xc = x - x.mean(0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / max(len(x) - 1, 1)
    return u[:, :k] * s[:k], var[:k] / max(var.sum(), 1e-12)


def remove_class_means(z: np.ndarray, cls: list[str]) -> np.ndarray:
    """Delete class identity: subtract each class's own mean. What remains is class-independent."""
    out = z.copy()
    for c in set(cls):
        m = np.array([x == c for x in cls])
        out[m] -= out[m].mean(0, keepdims=True)
    return out


def scatter(ax, xy, labels, title, palette=None, legend=False, alpha=0.85):
    keys = sorted(set(labels))
    for k in keys:
        m = np.array([x == k for x in labels])
        col = (palette or {}).get(k)
        ax.scatter(xy[m, 0], xy[m, 1], s=16, alpha=alpha, label=str(k),
                   color=col, edgecolors="none")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(fontsize=6, markerscale=1.2, loc="best", frameon=False, ncol=2)


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    split = json.loads((REPO_ROOT / cfg["data"]["split"]).read_text())
    root = REPO_ROOT / cfg["data"]["manip_latents"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_m = cfg["data"]["train_manips"]
    OUT.mkdir(parents=True, exist_ok=True)

    arms = ["E1_seed42", "ablation_seed42"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    summary = {}

    for row, arm in enumerate(arms):
        ck = REPO_ROOT / cfg["outputs"]["dir"] / arm / "encoder.pt"
        if not ck.exists():
            print(f"[viz] missing {ck}"); continue
        enc, head = probes.load_arm(ck, cfg, device)

        tr = probes.collect(root, split, "train", train_m)
        zs = probes.collect(root, split, "heldout_class", train_m)
        z_tr, _ = probes.embed(enc, head, root, tr, device)
        z_zs, _ = probes.embed(enc, head, root, zs, device)
        print(f"[viz] {arm}: train {z_tr.shape}, zs {z_zs.shape}", flush=True)

        # A: train by class
        xy, ev = pca(z_tr)
        scatter(axes[row, 0], xy, [r["cls"] for r in tr],
                f"{arm}\nA. TRAIN by CLASS (26)  ev={ev[0]:.2f}/{ev[1]:.2f}")
        # B: train by manipulation
        scatter(axes[row, 1], xy, [r["manip"] for r in tr],
                "B. TRAIN by MANIPULATION", palette=MANIP_COLOR, legend=(row == 0))
        # C: held-out zs classes, CLASS-MEAN-REMOVED, by manipulation  <-- the money panel
        zc = remove_class_means(z_zs, [r["cls"] for r in zs])
        xyc, evc = pca(zc)
        scatter(axes[row, 2], xyc, [r["manip"] for r in zs],
                f"C. HELD-OUT ZS, class-mean REMOVED, by MANIP\nev={evc[0]:.2f}/{evc[1]:.2f}",
                palette=MANIP_COLOR, legend=(row == 0))
        # D: identity -> reverse displacement arrows on held-out zs
        xyz, _ = pca(z_zs)
        idx = {(r["clip"], r["manip"]): i for i, r in enumerate(zs)}
        ax = axes[row, 3]
        ax.scatter(xyz[:, 0], xyz[:, 1], s=8, color="#cccccc", edgecolors="none")
        vecs = []
        for clip in sorted({r["clip"] for r in zs}):
            a, b = idx.get((clip, "identity")), idx.get((clip, "reverse"))
            if a is None or b is None:
                continue
            d = xyz[b] - xyz[a]
            vecs.append(d)
            ax.annotate("", xy=xyz[b], xytext=xyz[a],
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8, alpha=0.75))
        vecs = np.array(vecs)
        # alignment: mean resultant length of the displacement directions (1 = perfectly parallel)
        if len(vecs):
            u = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
            align = float(np.linalg.norm(u.mean(0)))
            med_len = float(np.median(np.linalg.norm(vecs, axis=1)))
        else:
            align = med_len = float("nan")
        ax.set_title(f"D. HELD-OUT ZS: identity->reverse\nalignment={align:.2f}  median|Δ|={med_len:.2f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        summary[arm] = {"reverse_arrow_alignment": align, "reverse_median_disp": med_len,
                        "trainC_ev": [float(e) for e in ev], "zs_classfree_ev": [float(e) for e in evc]}

    fig.suptitle("exp_079 — top: E1 (certified SupCon-T, class x manipulation labels)   |   "
                 "bottom: ablation (plain class-only SupCon)\n"
                 "Panel C is the decisive one: class identity is subtracted, so any separation by "
                 "colour is temporal structure on classes the encoder never trained on.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    dst = OUT / "pca_overview.png"
    fig.savefig(dst, dpi=130)
    (OUT / "pca_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[viz] wrote {dst}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
