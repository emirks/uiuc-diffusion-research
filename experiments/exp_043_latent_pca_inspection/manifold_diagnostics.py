"""Phase-1 manifold diagnostics on cached exp_043 z0 latents.

Reads run_0001/latents/{smoke_z0,davis_gen_z0}/*.pt and produces charts that
test whether the shadow_smoke trajectories share a low-dimensional manifold
distinct from clip-specific background/object content.

CPU-only.  No new encodes.  Outputs:

  charts/M1_per_frame_dispersion.png        — σ(t) and ‖μ(t)−μ(0)‖ vs t (smoke vs davis)
  charts/M2_time_explained_variance.png     — R²_time = 1 − V_within / V_total table
  charts/M3_anchored_joint_pca.png          — PCA of (z_k,t − z_k,0) for smoke vs davis
  charts/M4_cross_clip_distance_heatmap.png — 16×16 cross-clip frame-pair distances
  charts/M5_pc1_direction_agreement.png     — pairwise cos sim of per-clip PC1
  charts/M6_shared_direction_projection.png — projection of clip trajectories onto
                                              the mean displacement direction v_smoke

A short JSON summary is written to charts/manifold_summary.json.

Usage:
    python manifold_diagnostics.py --run_dir <path/to/run_NNNN>
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

LATENT_FRAMES = 16
SPATIAL_TOKENS = 19 * 19          # 361
CHANNELS = 128
FRAME_DIM = SPATIAL_TOKENS * CHANNELS   # 46208


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_manifest(run_dir: pathlib.Path) -> dict:
    with (run_dir / "manifest.yaml").open() as f:
        return yaml.safe_load(f)


def load_group(run_dir: pathlib.Path, entries: list[dict]) -> tuple[list[str], np.ndarray]:
    """Returns (names, [G, 16, 46208])."""
    names: list[str] = []
    arrs: list[np.ndarray] = []
    for e in entries:
        t = torch.load(run_dir / e["path"], map_location="cpu", weights_only=False)
        a = t.squeeze(0).float().numpy().reshape(LATENT_FRAMES, FRAME_DIM)
        names.append(e["name"])
        arrs.append(a)
    return names, np.stack(arrs, axis=0)


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def per_frame_dispersion(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sigma_t [16], drift_t [16]).
    sigma_t  = RMS over channels & space of (z_k,t − μ(t)), averaged across clips.
    drift_t  = ‖μ(t) − μ(0)‖_2 (Frobenius), i.e. how far the centroid has moved.
    """
    mu_t = Z.mean(axis=0)                     # [16, FRAME_DIM]
    diff = Z - mu_t[None]                     # [G, 16, FRAME_DIM]
    # RMS over (G, FRAME_DIM) at each t → scalar per t
    sigma_t = np.sqrt((diff ** 2).mean(axis=(0, 2)))   # [16]
    drift_t = np.linalg.norm(mu_t - mu_t[0], axis=1)   # [16]
    return sigma_t, drift_t


def time_explained_variance(Z: np.ndarray) -> dict:
    """R²_time = 1 − V_within / V_total.

    V_total  = Σ_{k,t} ‖z_k,t − μ_global‖²        (μ_global = mean over k,t)
    V_within = Σ_{k,t} ‖z_k,t − μ(t)‖²            (μ(t)      = mean over k at fixed t)
    """
    mu_global = Z.mean(axis=(0, 1))           # [FRAME_DIM]
    mu_t = Z.mean(axis=0)                     # [16, FRAME_DIM]
    V_total = ((Z - mu_global[None, None]) ** 2).sum()
    V_within = ((Z - mu_t[None]) ** 2).sum()
    r2 = 1.0 - V_within / V_total
    # Also: "between-time" variance share = (V_total − V_within) / V_total
    return {"R2_time": float(r2),
            "V_total": float(V_total),
            "V_within": float(V_within),
            "V_between_t": float(V_total - V_within)}


def anchored_pca(Z: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract z_k,0 from each frame, then SVD over all (k, t≥1) deltas.

    Returns (proj [G·15, k], evr [k], components [k, FRAME_DIM]).
    """
    Z0 = Z[:, 0:1]                            # [G, 1, FRAME_DIM]
    delta = Z[:, 1:] - Z0                     # [G, 15, FRAME_DIM]
    X = delta.reshape(-1, FRAME_DIM)          # [G·15, FRAME_DIM]
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = (U * S)[:, :n_components]
    total = (S ** 2).sum()
    evr = (S[:n_components] ** 2) / max(total, 1e-12)
    return proj, evr, Vt[:n_components]


def cross_clip_distance_matrix(Z: np.ndarray) -> np.ndarray:
    """For each (i,j), mean over k!=l of ‖z_k,i − z_l,j‖₂."""
    G, T, D = Z.shape
    M = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            # All G*G pairs, excluding diagonal k==l
            zi = Z[:, i]                      # [G, D]
            zj = Z[:, j]                      # [G, D]
            d2 = ((zi[:, None] - zj[None, :]) ** 2).sum(axis=-1)   # [G, G]
            d  = np.sqrt(d2)
            # Exclude diagonal pairs (k==l)
            mask = ~np.eye(G, dtype=bool)
            M[i, j] = d[mask].mean()
    return M


def per_clip_pc1_direction(Z: np.ndarray) -> np.ndarray:
    """For each clip k, PC1 direction (sign-aligned with z_15 − z_0).
    Returns [G, FRAME_DIM] unit vectors.
    """
    G, T, D = Z.shape
    out = np.zeros((G, D))
    for k in range(G):
        Xc = Z[k] - Z[k].mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        d = Vt[0]
        # Sign-align so projection of (z_k,15 − z_k,0) is positive
        if np.dot(Z[k, -1] - Z[k, 0], d) < 0:
            d = -d
        out[k] = d / max(np.linalg.norm(d), 1e-12)
    return out


def shared_displacement_direction(Z: np.ndarray) -> np.ndarray:
    """v = normalize(mean_k (z_k,15 − z_k,0))."""
    d = (Z[:, -1] - Z[:, 0]).mean(axis=0)
    return d / max(np.linalg.norm(d), 1e-12)


# ─── Charts ──────────────────────────────────────────────────────────────────

def chart_M1_per_frame_dispersion(out, smoke, davis):
    sigma_s, drift_s = per_frame_dispersion(smoke)
    sigma_d, drift_d = per_frame_dispersion(davis)
    ts = np.arange(LATENT_FRAMES)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(ts, sigma_s, "o-", color="tab:red",  label=f"smoke (G={smoke.shape[0]})")
    ax1.plot(ts, sigma_d, "o-", color="tab:blue", label=f"davis A_word (G={davis.shape[0]})")
    ax1.set_xlabel("latent frame index t"); ax1.set_ylabel("σ(t) — RMS dispersion across clips")
    ax1.set_title("Per-frame across-clip dispersion σ(t)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ts, drift_s, "o-", color="tab:red",  label="smoke ‖μ(t)−μ(0)‖")
    ax2.plot(ts, drift_d, "o-", color="tab:blue", label="davis ‖μ(t)−μ(0)‖")
    ax2.set_xlabel("latent frame index t"); ax2.set_ylabel("‖μ(t)−μ(0)‖₂")
    ax2.set_title("Centroid drift (shared trajectory length)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("[M1] Per-frame dispersion & centroid drift — is there a shared time-trajectory?")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "M1_per_frame_dispersion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {"sigma_smoke": sigma_s.tolist(), "sigma_davis": sigma_d.tolist(),
            "drift_smoke": drift_s.tolist(), "drift_davis": drift_d.tolist()}


def chart_M2_time_explained(out, smoke, davis):
    rs = time_explained_variance(smoke)
    rd = time_explained_variance(davis)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = ["smoke", "davis A_word"]
    vals = [rs["R2_time"], rd["R2_time"]]
    bars = ax.bar(labels, vals, color=["tab:red", "tab:blue"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("R²_time = 1 − V_within / V_total")
    ax.set_title("[M2] Variance explained by frame-index alone (higher ⇒ trajectories more shared)")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "M2_time_explained_variance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {"smoke": rs, "davis": rd}


def chart_M3_anchored_pca(out, smoke, davis):
    proj_s, evr_s, comp_s = anchored_pca(smoke, n_components=5)
    proj_d, evr_d, comp_d = anchored_pca(davis, n_components=5)
    G_s, G_d = smoke.shape[0], davis.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Left: smoke anchored trajectories
    ax = axes[0]
    cmap = plt.cm.tab10
    for k in range(G_s):
        coords = proj_s[k * 15:(k + 1) * 15, :2]
        # Prepend origin (frame 0 = 0 by construction)
        coords = np.vstack([[0, 0], coords])
        ax.plot(coords[:, 0], coords[:, 1], "-o", color=cmap(k % 10), lw=1.0, ms=4, alpha=0.8)
        ax.scatter(coords[-1, 0], coords[-1, 1], color=cmap(k % 10), s=80,
                   marker="s", edgecolors="black", linewidths=0.5, zorder=5)
    ax.scatter([0], [0], color="black", s=120, marker="*", zorder=6, label="anchor (t=0)")
    ax.set_title(f"smoke — anchored joint PCA  (PC1 {evr_s[0]*100:.1f}%, PC2 {evr_s[1]*100:.1f}%)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Middle: davis anchored trajectories
    ax = axes[1]
    for k in range(G_d):
        coords = proj_d[k * 15:(k + 1) * 15, :2]
        coords = np.vstack([[0, 0], coords])
        ax.plot(coords[:, 0], coords[:, 1], "-o", color=cmap(k % 10), lw=1.0, ms=4, alpha=0.8)
        ax.scatter(coords[-1, 0], coords[-1, 1], color=cmap(k % 10), s=80,
                   marker="s", edgecolors="black", linewidths=0.5, zorder=5)
    ax.scatter([0], [0], color="black", s=120, marker="*", zorder=6)
    ax.set_title(f"davis A_word — anchored joint PCA  (PC1 {evr_d[0]*100:.1f}%, PC2 {evr_d[1]*100:.1f}%)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)

    # Right: scree of anchored PCA (smoke vs davis)
    ax = axes[2]
    ax.plot(np.arange(1, 6), evr_s, "o-", color="tab:red", label="smoke")
    ax.plot(np.arange(1, 6), evr_d, "o-", color="tab:blue", label="davis")
    ax.set_yscale("log")
    ax.set_xlabel("component"); ax.set_ylabel("explained var ratio (log)")
    ax.set_title("Anchored-PCA scree (top 5)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("[M3] Anchored joint PCA — direction of motion away from t=0 frame")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "M3_anchored_joint_pca.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {"smoke_evr_top5": evr_s.tolist(), "davis_evr_top5": evr_d.tolist()}


def chart_M4_cross_clip_distance(out, smoke, davis):
    M_s = cross_clip_distance_matrix(smoke)
    M_d = cross_clip_distance_matrix(davis)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, M, label in zip(axes, [M_s, M_d], ["smoke (10 clips)", "davis A_word (5 clips)"]):
        im = ax.imshow(M, cmap="viridis", origin="lower")
        plt.colorbar(im, ax=ax, label="mean ‖z_k,i − z_l,j‖₂  (k≠l)")
        ax.set_xlabel("frame j"); ax.set_ylabel("frame i")
        # Annotate the diagonal mean and off-diagonal mean
        diag = np.diag(M).mean()
        off  = (M.sum() - np.trace(M)) / (M.size - M.shape[0])
        ax.set_title(f"{label}\ndiag mean={diag:.1f}  off-diag mean={off:.1f}  "
                     f"ratio={diag/off:.3f}")
    fig.suptitle("[M4] Cross-clip frame-pair distance — small diag ⇒ clips agree at that t")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "M4_cross_clip_distance_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {"smoke_diag_mean": float(np.diag(M_s).mean()),
            "smoke_off_mean":  float((M_s.sum() - np.trace(M_s)) / (M_s.size - M_s.shape[0])),
            "davis_diag_mean": float(np.diag(M_d).mean()),
            "davis_off_mean":  float((M_d.sum() - np.trace(M_d)) / (M_d.size - M_d.shape[0])),
            "smoke_diag_per_t": np.diag(M_s).tolist(),
            "davis_diag_per_t": np.diag(M_d).tolist()}


def chart_M5_direction_agreement(out, smoke, davis):
    D_s = per_clip_pc1_direction(smoke)        # [10, FRAME_DIM]
    D_d = per_clip_pc1_direction(davis)        # [5, FRAME_DIM]
    cos_s = D_s @ D_s.T
    cos_d = D_d @ D_d.T
    # Null expected magnitude for two random unit vectors in R^D
    null_std = 1.0 / np.sqrt(FRAME_DIM)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, M, label, G in zip(axes, [cos_s, cos_d], ["smoke", "davis A_word"],
                                [smoke.shape[0], davis.shape[0]]):
        im = ax.imshow(M, cmap="RdBu_r", origin="lower", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="cosine sim of per-clip PC1 dirs (sign-aligned)")
        off = M[~np.eye(G, dtype=bool)]
        ax.set_title(f"{label}  G={G}\nmean off-diag cos sim = {off.mean():+.3f}  "
                     f"(null σ ≈ {null_std:.4f})")
        ax.set_xlabel("clip l"); ax.set_ylabel("clip k")
    fig.suptitle("[M5] Per-clip PC1-direction cosine similarity — "
                 "off-diag ≫ 0 ⇒ clips share a direction")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "M5_pc1_direction_agreement.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    off_s = cos_s[~np.eye(smoke.shape[0], dtype=bool)]
    off_d = cos_d[~np.eye(davis.shape[0], dtype=bool)]
    return {"smoke_offdiag_mean_cos": float(off_s.mean()),
            "smoke_offdiag_std_cos":  float(off_s.std()),
            "davis_offdiag_mean_cos": float(off_d.mean()),
            "davis_offdiag_std_cos":  float(off_d.std()),
            "null_cos_std": float(null_std)}


def chart_M6_shared_direction_projection(out, smoke, davis):
    v_smoke = shared_displacement_direction(smoke)  # [FRAME_DIM]
    v_davis = shared_displacement_direction(davis)
    # Random control direction, in span of the data:
    rng = np.random.default_rng(42)
    v_rand = rng.standard_normal(FRAME_DIM)
    v_rand /= np.linalg.norm(v_rand)

    ts = np.arange(LATENT_FRAMES)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    def _plot(ax, Z, v, label, color):
        # Project each clip's trajectory onto v (after subtracting per-clip frame 0)
        for k in range(Z.shape[0]):
            d = (Z[k] - Z[k, 0]) @ v          # [16]
            ax.plot(ts, d, "-o", color=color, lw=0.9, ms=3, alpha=0.7)
        # Mean across clips
        mean_d = ((Z - Z[:, 0:1]) @ v).mean(axis=0)
        ax.plot(ts, mean_d, "-", color="black", lw=2.2, label=f"mean ({label})")
        ax.set_xlabel("frame t"); ax.set_ylabel(f"(z_k,t − z_k,0) · v   [{label}]")
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    _plot(axes[0], smoke, v_smoke, "v_smoke (smoke clips)", "tab:red")
    axes[0].set_title("smoke clips projected onto v_smoke")
    _plot(axes[1], davis, v_smoke, "v_smoke applied to davis", "tab:blue")
    axes[1].set_title("davis A_word clips projected onto v_smoke\n(specificity control)")
    _plot(axes[2], smoke, v_rand, "v_random (control)", "tab:gray")
    axes[2].set_title("smoke clips projected onto random v\n(null control)")

    fig.suptitle("[M6] Projection of clip trajectories onto v_smoke = mean(z_15 − z_0)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "M6_shared_direction_projection.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # Final-frame projection statistics
    proj_smoke_final = ((smoke[:, -1] - smoke[:, 0]) @ v_smoke)
    proj_davis_final = ((davis[:, -1] - davis[:, 0]) @ v_smoke)
    proj_rand_final  = ((smoke[:, -1] - smoke[:, 0]) @ v_rand)
    return {"smoke_final_proj_mean_v_smoke": float(proj_smoke_final.mean()),
            "smoke_final_proj_std_v_smoke":  float(proj_smoke_final.std()),
            "davis_final_proj_mean_v_smoke": float(proj_davis_final.mean()),
            "davis_final_proj_std_v_smoke":  float(proj_davis_final.std()),
            "smoke_final_proj_mean_v_rand":  float(proj_rand_final.mean()),
            "smoke_final_proj_std_v_rand":   float(proj_rand_final.std()),
            "v_smoke_vs_v_davis_cos":        float(np.dot(v_smoke, v_davis))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = run_dir / "charts"
    out.mkdir(exist_ok=True)

    manifest = load_manifest(run_dir)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for e in manifest["entries"]:
        by_group[e["group"]].append(e)
    for v in by_group.values():
        v.sort(key=lambda e: e["name"])

    print("[load] smoke z0")
    _, Z_smoke = load_group(run_dir, by_group["smoke_z0"])
    print("[load] davis A_word z0")
    _, Z_davis = load_group(run_dir, by_group["davis_gen_z0"])
    print(f"  smoke {Z_smoke.shape}, davis {Z_davis.shape}")

    summary = {}
    print("[M1] per-frame dispersion")
    summary["M1"] = chart_M1_per_frame_dispersion(out, Z_smoke, Z_davis)
    print("[M2] time-explained variance")
    summary["M2"] = chart_M2_time_explained(out, Z_smoke, Z_davis)
    print("[M3] anchored joint PCA")
    summary["M3"] = chart_M3_anchored_pca(out, Z_smoke, Z_davis)
    print("[M4] cross-clip distance heatmap")
    summary["M4"] = chart_M4_cross_clip_distance(out, Z_smoke, Z_davis)
    print("[M5] per-clip PC1 direction agreement")
    summary["M5"] = chart_M5_direction_agreement(out, Z_smoke, Z_davis)
    print("[M6] shared-direction projection")
    summary["M6"] = chart_M6_shared_direction_projection(out, Z_smoke, Z_davis)

    with (out / "manifold_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote 6 charts + manifold_summary.json → {out}")


if __name__ == "__main__":
    main()
