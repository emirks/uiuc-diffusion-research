"""Render the exp_043 chart matrix from a run_NNNN/ encode directory.

Loads `manifest.yaml` + every `latents/<group>/<name>.pt`, computes the PCAs
described in README.md, and writes PNG figures into `<run_dir>/charts/`.
CPU-only.

Usage:
    python make_charts.py --run_dir <path/to/exp_043 run_NNNN>
"""
from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D


# Packed LTX-2 latent geometry at 608×608 / 121 frames.
LATENT_FRAMES   = 16
SPATIAL_TOKENS  = 19 * 19          # 361
CHANNELS        = 128
TOKENS_PER_FRAME = SPATIAL_TOKENS  # at 608×608 each latent frame = 19×19 tokens
N_TOKENS        = LATENT_FRAMES * SPATIAL_TOKENS  # 5776


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_manifest(run_dir: pathlib.Path) -> dict:
    with (run_dir / "manifest.yaml").open() as f:
        return yaml.safe_load(f)


def load_tensor(run_dir: pathlib.Path, relpath: str) -> np.ndarray:
    """Load a saved bfloat16 tensor as float32 numpy of shape [N_TOKENS, CHANNELS]."""
    t = torch.load(run_dir / relpath, map_location="cpu", weights_only=False)
    # Stored as [1, N, 128] bfloat16; squeeze batch, cast.
    return t.squeeze(0).float().numpy()


def split_by_group(manifest: dict) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for e in manifest["entries"]:
        by_group[e["group"]].append(e)
    for entries in by_group.values():
        entries.sort(key=lambda e: e["name"])
    return dict(by_group)


def load_group(run_dir: pathlib.Path, entries: list[dict]) -> tuple[list[str], np.ndarray]:
    """Returns (names, tensor [G, 16, 361, 128]) — assumes uniform N_TOKENS=5776."""
    names: list[str] = []
    arrs: list[np.ndarray] = []
    for e in entries:
        a = load_tensor(run_dir, e["path"])
        # reshape [5776, 128] → [16, 361, 128]
        a = a.reshape(LATENT_FRAMES, SPATIAL_TOKENS, CHANNELS)
        names.append(e["name"])
        arrs.append(a)
    return names, np.stack(arrs, axis=0)


def load_group_pooled(run_dir: pathlib.Path, entries: list[dict]) -> tuple[list[str], np.ndarray]:
    """For groups with variable token-count (exp_033 z1 used per-clip max_area
    resolutions → 5632 vs 5776 tokens).  Returns the per-frame MEAN-POOLED
    representation: [G, 16, 128] (16 latent frames, 128 channels, averaged
    over spatial tokens).  This is uniform regardless of spatial token count.
    """
    names: list[str] = []
    arrs: list[np.ndarray] = []
    for e in entries:
        a = load_tensor(run_dir, e["path"])           # [N, 128]
        n_tokens = a.shape[0]
        if n_tokens % LATENT_FRAMES != 0:
            raise ValueError(f"{e['name']}: N={n_tokens} not divisible by {LATENT_FRAMES} latent frames")
        spatial = n_tokens // LATENT_FRAMES           # 361 or 352
        a = a.reshape(LATENT_FRAMES, spatial, CHANNELS).mean(axis=1)   # [16, 128]
        names.append(e["name"])
        arrs.append(a)
    return names, np.stack(arrs, axis=0)


# ─── PCA helpers ─────────────────────────────────────────────────────────────

def fit_pca(data: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center + SVD. Returns (proj [N, k], explained_var_ratio [k], components [k, D])."""
    X = data - data.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    proj = (U * S)[:, :n_components]
    total_var = (S ** 2).sum()
    evr = (S[:n_components] ** 2) / max(total_var, 1e-12)
    return proj, evr, Vt[:n_components]


def project_with(data: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (data - mean) @ components.T


def scree(data: np.ndarray, k: int = 20) -> np.ndarray:
    X = data - data.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    total = (S ** 2).sum()
    return (S ** 2 / total)[:k]


# ─── Plot helpers ────────────────────────────────────────────────────────────

def _annotate_evr(ax, evr: np.ndarray) -> None:
    ax.set_xlabel(f"PC1  ({evr[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({evr[1] * 100:.1f}% var)")


def _arrow_traj(ax, coords: np.ndarray, color, label: str | None = None) -> None:
    """Draw a 2D trajectory: line + arrow + start/end markers + frame indices."""
    ax.plot(coords[:, 0], coords[:, 1], "-", color=color, lw=1.0, alpha=0.7)
    # arrow from frame F-2 to F-1
    if len(coords) >= 2:
        ax.annotate("", xy=coords[-1], xytext=coords[-2],
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    ax.scatter(coords[0:1, 0], coords[0:1, 1], color=color, marker="o", s=50,
               edgecolors="black", linewidths=0.6, zorder=3,
               label=label if label is not None else None)
    ax.scatter(coords[-1:, 0], coords[-1:, 1], color=color, marker="s", s=50,
               edgecolors="black", linewidths=0.6, zorder=3)


# ─── Chart functions ─────────────────────────────────────────────────────────

def chart_1_per_sample_trajectory(
    run_dir: pathlib.Path,
    names: list[str],
    Z: np.ndarray,            # [G, 16, 361, 128]
    title: str,
    fname: str,
    cols: int,
) -> None:
    """Per-clip local PCA on its 16 frames (each frame flattened to 361×128)."""
    G = Z.shape[0]
    rows = int(np.ceil(G / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows),
                             squeeze=False)
    for k in range(G):
        ax = axes[k // cols, k % cols]
        frames_flat = Z[k].reshape(LATENT_FRAMES, -1)   # [16, 46208]
        proj, evr, _ = fit_pca(frames_flat, n_components=2)
        # color gradient by frame index for visual continuity
        colors = plt.cm.viridis(np.linspace(0, 1, LATENT_FRAMES))
        ax.plot(proj[:, 0], proj[:, 1], "-", color="gray", lw=0.7, alpha=0.5)
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=24,
                   edgecolors="black", linewidths=0.3)
        for fi in range(LATENT_FRAMES):
            ax.annotate(str(fi), (proj[fi, 0], proj[fi, 1]),
                        fontsize=6, alpha=0.7, xytext=(2, 2),
                        textcoords="offset points")
        ax.set_title(f"{names[k]}\nPC1 {evr[0] * 100:.1f}%, PC2 {evr[1] * 100:.1f}%",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
        ax.axvline(0, color="k", lw=0.3, alpha=0.3)
    # Hide unused panels
    for k in range(G, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_3_smoke_unified_frame(
    run_dir: pathlib.Path, names: list[str], Z: np.ndarray, fname: str,
) -> None:
    """All frames of all smoke clips → one PCA; show twice (by sample, by frame index)."""
    G = Z.shape[0]
    frames_flat = Z.reshape(G * LATENT_FRAMES, -1)        # [G·16, 46208]
    proj, evr, _ = fit_pca(frames_flat, n_components=2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left — color by sample
    cmap = plt.cm.tab10
    for k in range(G):
        coords = proj[k * LATENT_FRAMES:(k + 1) * LATENT_FRAMES]
        color = cmap(k % 10)
        ax_l.plot(coords[:, 0], coords[:, 1], "-", color=color, lw=0.8, alpha=0.6)
        ax_l.scatter(coords[:, 0], coords[:, 1], color=color, s=22,
                     edgecolors="black", linewidths=0.3, label=names[k])
    ax_l.legend(fontsize=6, ncol=2, loc="best")
    ax_l.set_title("Smoke z0 — frames joint PCA, colored by sample")
    _annotate_evr(ax_l, evr)
    ax_l.axhline(0, color="k", lw=0.3, alpha=0.3); ax_l.axvline(0, color="k", lw=0.3, alpha=0.3)

    # Right — color by frame index
    fi = np.tile(np.arange(LATENT_FRAMES), G)
    sc = ax_r.scatter(proj[:, 0], proj[:, 1], c=fi, cmap="viridis", s=22,
                      edgecolors="black", linewidths=0.3)
    plt.colorbar(sc, ax=ax_r, label="frame index")
    ax_r.set_title("Smoke z0 — frames joint PCA, colored by frame index")
    _annotate_evr(ax_r, evr)
    ax_r.axhline(0, color="k", lw=0.3, alpha=0.3); ax_r.axvline(0, color="k", lw=0.3, alpha=0.3)

    fig.suptitle("[3] Smoke unified frame PCA  (each point = one latent frame, 46208-D)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_5_clip_pca_cleans(
    run_dir: pathlib.Path,
    smoke_names: list[str], Z_smoke: np.ndarray,
    davis_names: list[str], Z_davis: np.ndarray,
    fname: str,
) -> None:
    """One point per clip; smoke vs davis-gen."""
    clip_smoke = Z_smoke.reshape(Z_smoke.shape[0], -1)    # [G, 5776*128]
    clip_davis = Z_davis.reshape(Z_davis.shape[0], -1)
    data = np.concatenate([clip_smoke, clip_davis], axis=0)
    proj, evr, _ = fit_pca(data, n_components=2)

    G_s = Z_smoke.shape[0]
    G_d = Z_davis.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[:G_s, 0], proj[:G_s, 1], c="tab:red", s=70,
               edgecolors="black", linewidths=0.5, label=f"smoke ({G_s})")
    for k, nm in enumerate(smoke_names):
        ax.annotate(nm.replace("shadow_smoke_", "ss"), (proj[k, 0], proj[k, 1]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.scatter(proj[G_s:, 0], proj[G_s:, 1], c="tab:blue", s=70,
               edgecolors="black", linewidths=0.5, label=f"davis A_word ({G_d})")
    for k, nm in enumerate(davis_names):
        ax.annotate(nm.split("__")[0], (proj[G_s + k, 0], proj[G_s + k, 1]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.legend(loc="best", fontsize=9)
    ax.set_title("[5] Whole-clip PCA of cleans  (one point = one clip, 5776·128-D)")
    _annotate_evr(ax, evr)
    ax.axhline(0, color="k", lw=0.3, alpha=0.3); ax.axvline(0, color="k", lw=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_6_cross_domain_frame_cleans(
    run_dir: pathlib.Path,
    Z_smoke: np.ndarray,
    Z_davis: np.ndarray,
    fname: str,
) -> None:
    """Joint frame PCA over smoke + davis-gen cleans. The cluster test."""
    smoke_frames = Z_smoke.reshape(-1, SPATIAL_TOKENS * CHANNELS)
    davis_frames = Z_davis.reshape(-1, SPATIAL_TOKENS * CHANNELS)
    data = np.concatenate([smoke_frames, davis_frames], axis=0)
    proj, evr, _ = fit_pca(data, n_components=2)

    n_s = smoke_frames.shape[0]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(proj[:n_s, 0], proj[:n_s, 1], c="tab:red", s=20, alpha=0.7,
               edgecolors="black", linewidths=0.2, label=f"smoke ({Z_smoke.shape[0]} clips × 16 frames)")
    ax.scatter(proj[n_s:, 0], proj[n_s:, 1], c="tab:blue", s=20, alpha=0.7,
               edgecolors="black", linewidths=0.2, label=f"davis A_word ({Z_davis.shape[0]} clips × 16 frames)")

    # Mean cluster locations
    smoke_mu = proj[:n_s].mean(axis=0)
    davis_mu = proj[n_s:].mean(axis=0)
    ax.scatter(*smoke_mu, marker="X", c="darkred", s=200, edgecolors="black",
               linewidths=1.5, zorder=10, label="smoke mean")
    ax.scatter(*davis_mu, marker="X", c="navy", s=200, edgecolors="black",
               linewidths=1.5, zorder=10, label="davis mean")
    ax.legend(loc="best", fontsize=9)
    sep = float(np.linalg.norm(smoke_mu - davis_mu))
    ax.set_title(f"[6] ⭐ Cross-domain frame PCA — cleans  (cluster-mean separation = {sep:.2f})")
    _annotate_evr(ax, evr)
    ax.axhline(0, color="k", lw=0.3, alpha=0.3); ax.axvline(0, color="k", lw=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_7_clip_z1_vs_gaussian(
    run_dir: pathlib.Path, Z_z1_pool: np.ndarray, Z_g_pool: np.ndarray, fname: str,
) -> None:
    """One point per z1 vs per gaussian sample.

    Inputs are MEAN-POOLED over spatial tokens, shape [G, 16, 128].
    Per-clip vector = flattened [16, 128] = 2048-D.  This is the only
    representation that's uniform across z1's that came from different
    encode resolutions (5632 vs 5776 tokens).
    """
    clip_z1 = Z_z1_pool.reshape(Z_z1_pool.shape[0], -1)   # [G, 16*128]
    clip_g  = Z_g_pool.reshape(Z_g_pool.shape[0], -1)
    data = np.concatenate([clip_z1, clip_g], axis=0)
    proj, evr, _ = fit_pca(data, n_components=2)
    n_z1 = clip_z1.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[n_z1:, 0], proj[n_z1:, 1], c="lightgray", s=30, alpha=0.7,
               edgecolors="black", linewidths=0.3, label=f"gaussian ({Z_g_pool.shape[0]})")
    ax.scatter(proj[:n_z1, 0], proj[:n_z1, 1], c="tab:purple", s=80,
               edgecolors="black", linewidths=0.5, label=f"smoke z1 ({n_z1})")

    z1_mu = proj[:n_z1].mean(axis=0)
    g_mu  = proj[n_z1:].mean(axis=0)
    sep = float(np.linalg.norm(z1_mu - g_mu))
    ax.scatter(*z1_mu, marker="X", c="darkmagenta", s=200, edgecolors="black",
               linewidths=1.5, zorder=10, label="z1 mean")
    ax.scatter(*g_mu, marker="X", c="dimgray", s=200, edgecolors="black",
               linewidths=1.5, zorder=10, label="gaussian mean")
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"[7] Whole-clip PCA — smoke z1 vs Gaussian  "
                 f"(spatial-pooled, mean sep = {sep:.2f})")
    _annotate_evr(ax, evr)
    ax.axhline(0, color="k", lw=0.3, alpha=0.3); ax.axvline(0, color="k", lw=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_8_frame_z1_vs_gaussian(
    run_dir: pathlib.Path, Z_z1_pool: np.ndarray, Z_g_pool: np.ndarray, fname: str,
) -> None:
    """Frame-level PCA of z1 frames + gaussian frames (spatial-mean-pooled).

    Inputs are [G, 16, 128]; per-frame vector = 128-D (one point per latent frame).
    """
    z1_frames = Z_z1_pool.reshape(-1, CHANNELS)
    g_frames  = Z_g_pool.reshape(-1, CHANNELS)
    data = np.concatenate([z1_frames, g_frames], axis=0)
    proj, evr, _ = fit_pca(data, n_components=2)
    n_z = z1_frames.shape[0]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(proj[n_z:, 0], proj[n_z:, 1], c="lightgray", s=15, alpha=0.5,
               edgecolors="none", label=f"gaussian frames ({g_frames.shape[0]})")
    ax.scatter(proj[:n_z, 0], proj[:n_z, 1], c="tab:purple", s=20, alpha=0.8,
               edgecolors="black", linewidths=0.2, label=f"smoke z1 frames ({n_z})")

    z1_mu = proj[:n_z].mean(axis=0)
    g_mu  = proj[n_z:].mean(axis=0)
    sep = float(np.linalg.norm(z1_mu - g_mu))
    ax.scatter(*z1_mu, marker="X", c="darkmagenta", s=200, edgecolors="black",
               linewidths=1.5, zorder=10, label="z1 frame mean")
    ax.scatter(*g_mu, marker="X", c="dimgray", s=200, edgecolors="black",
               linewidths=1.5, zorder=10, label="gaussian frame mean")
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"[8] Frame PCA — smoke z1 vs Gaussian  "
                 f"(spatial-pooled, 128-D, mean sep = {sep:.2f})")
    _annotate_evr(ax, evr)
    ax.axhline(0, color="k", lw=0.3, alpha=0.3); ax.axvline(0, color="k", lw=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _inject_role_subset(entries: list[dict], run_dir: pathlib.Path, variant: str
                        ) -> dict[str, np.ndarray]:
    """Return {role -> [16, 361, 128]} for one variant."""
    out = {}
    for e in entries:
        if not e["name"].startswith(variant + "__"):
            continue
        role = e["name"][len(variant) + 2:]
        arr = load_tensor(run_dir, e["path"]).reshape(LATENT_FRAMES, SPATIAL_TOKENS, CHANNELS)
        out[role] = arr
    return out


def chart_9_per_variant_injection(
    run_dir: pathlib.Path,
    variants: list[str],
    variant_dict: dict[str, dict[str, np.ndarray]],
    fname: str,
    cols: int = 4,
) -> None:
    """One panel per variant: PCA over frames of {reference, baseline, inject}."""
    rows = int(np.ceil(len(variants) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.4 * rows), squeeze=False)
    for k, v in enumerate(variants):
        ax = axes[k // cols, k % cols]
        roles = variant_dict[v]
        # Concatenate frames in fixed order: reference, baseline, inject
        ref = roles["reference_recon"].reshape(LATENT_FRAMES, -1)
        bas = roles["perturbed_baseline"].reshape(LATENT_FRAMES, -1)
        inj = roles["perturbed_inject"].reshape(LATENT_FRAMES, -1)
        data = np.concatenate([ref, bas, inj], axis=0)
        proj, evr, _ = fit_pca(data, n_components=2)
        p_ref, p_bas, p_inj = proj[:16], proj[16:32], proj[32:48]
        ax.plot(p_ref[:, 0], p_ref[:, 1], "-", c="tab:green",  lw=1.0, alpha=0.7)
        ax.plot(p_bas[:, 0], p_bas[:, 1], "-", c="tab:red",    lw=1.0, alpha=0.7)
        ax.plot(p_inj[:, 0], p_inj[:, 1], "-", c="tab:purple", lw=1.0, alpha=0.7)
        ax.scatter(*p_ref.T, c="tab:green",  s=24, edgecolors="black",
                   linewidths=0.3, label="reference")
        ax.scatter(*p_bas.T, c="tab:red",    s=24, edgecolors="black",
                   linewidths=0.3, label="baseline")
        ax.scatter(*p_inj.T, c="tab:purple", s=24, edgecolors="black",
                   linewidths=0.3, label="inject")
        # Compute distances in this local PCA: |inject - ref| vs |baseline - ref|
        d_inj = float(np.linalg.norm(p_inj - p_ref, axis=1).mean())
        d_bas = float(np.linalg.norm(p_bas - p_ref, axis=1).mean())
        delta = d_bas - d_inj   # positive ⇒ inject is closer to ref than baseline is
        ax.set_title(f"{v}\nPC1+2={(evr[0]+evr[1])*100:.0f}%  "
                     f"d(bas,ref)={d_bas:.1f}  d(inj,ref)={d_inj:.1f}  Δ={delta:+.1f}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.legend(fontsize=7, loc="best")
        ax.axhline(0, color="k", lw=0.3, alpha=0.3); ax.axvline(0, color="k", lw=0.3, alpha=0.3)
    for k in range(len(variants), rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle("[9] Per-variant injection PCA (local fit per panel)  "
                 "Δ = d(bas,ref) − d(inj,ref), positive ⇒ inject pulled toward reference",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_9b_injection_overlay(
    run_dir: pathlib.Path,
    variants: list[str],
    variant_dict: dict[str, dict[str, np.ndarray]],
    fname: str,
) -> None:
    """One joint PCA over all variants × all roles. Color by role, marker by variant."""
    parts: list[tuple[str, str, np.ndarray]] = []   # (variant, role, [16, D])
    for v in variants:
        roles = variant_dict[v]
        for role in ("reference_recon", "perturbed_baseline", "perturbed_inject"):
            parts.append((v, role, roles[role].reshape(LATENT_FRAMES, -1)))
    data = np.concatenate([p[2] for p in parts], axis=0)
    proj, evr, _ = fit_pca(data, n_components=2)
    role_color = {"reference_recon": "tab:green",
                  "perturbed_baseline": "tab:red",
                  "perturbed_inject": "tab:purple"}

    fig, ax = plt.subplots(figsize=(11, 7))
    n_marker = max(11, len(variants))
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h"][:n_marker]
    var_marker = dict(zip(variants, markers))
    for i, (v, role, _) in enumerate(parts):
        coords = proj[i * LATENT_FRAMES:(i + 1) * LATENT_FRAMES]
        ax.scatter(coords[:, 0], coords[:, 1],
                   color=role_color[role], marker=var_marker[v],
                   s=28, edgecolors="black", linewidths=0.25, alpha=0.7)

    # Legend: two-column (role + variant)
    role_handles = [Line2D([0], [0], marker="o", linestyle="", color=role_color[r],
                           markeredgecolor="black", markersize=9, label=r)
                    for r in ("reference_recon", "perturbed_baseline", "perturbed_inject")]
    variant_handles = [Line2D([0], [0], marker=var_marker[v], linestyle="",
                              color="gray", markeredgecolor="black",
                              markersize=8, label=v) for v in variants]
    leg1 = ax.legend(handles=role_handles, loc="upper left", fontsize=8, title="role")
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, loc="upper right", fontsize=7,
              title="variant", ncol=2)
    ax.set_title("[9b] All-variants injection overlay  (joint PCA, color = role, marker = variant)")
    _annotate_evr(ax, evr)
    ax.axhline(0, color="k", lw=0.3, alpha=0.3); ax.axvline(0, color="k", lw=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Diagnostics ────────────────────────────────────────────────────────────

def chart_z1_per_channel_stats(
    run_dir: pathlib.Path, z1_entries: list[dict], g_entries: list[dict],
    run_dir_for_load: pathlib.Path, fname: str,
) -> None:
    """Per-channel mean / std bar chart: smoke z1 vs Gaussian.  Streams over
    tokens so it works with variable token-count z1's."""
    def _stats(entries):
        sum_x = np.zeros(CHANNELS)
        sum_x2 = np.zeros(CHANNELS)
        n = 0
        for e in entries:
            a = load_tensor(run_dir_for_load, e["path"])    # [N, 128]
            sum_x  += a.sum(axis=0)
            sum_x2 += (a * a).sum(axis=0)
            n      += a.shape[0]
        mu = sum_x / n
        var = sum_x2 / n - mu ** 2
        return mu, np.sqrt(np.maximum(var, 0))

    z1_mean, z1_std = _stats(z1_entries)
    g_mean,  g_std  = _stats(g_entries)

    fig, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    idx = np.arange(CHANNELS)
    ax_m.bar(idx - 0.2, z1_mean, width=0.4, color="tab:purple", label="smoke z1 mean")
    ax_m.bar(idx + 0.2, g_mean,  width=0.4, color="lightgray",  label="gaussian mean")
    ax_m.axhline(0, color="k", lw=0.4)
    ax_m.set_ylabel("per-channel mean"); ax_m.legend(fontsize=8)
    ax_m.set_title("Per-channel first-order stats: smoke z1 vs Gaussian (N=128 channels)")

    ax_s.bar(idx - 0.2, z1_std, width=0.4, color="tab:purple", label="smoke z1 std")
    ax_s.bar(idx + 0.2, g_std,  width=0.4, color="lightgray",  label="gaussian std")
    ax_s.axhline(1.0, color="k", lw=0.4, ls="--")
    ax_s.set_xlabel("channel"); ax_s.set_ylabel("per-channel std"); ax_s.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_scree(
    run_dir: pathlib.Path,
    fname: str,
    series: list[tuple[str, np.ndarray]],
) -> None:
    """Scree plot — each series is (label, data matrix)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    k = 20
    for label, data in series:
        s = scree(data, k=k)
        ax.plot(np.arange(1, len(s) + 1), s, "o-", label=label, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("component"); ax.set_ylabel("explained variance ratio (log)")
    ax.set_title("Scree (top 20 components)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "charts" / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    print(f"[info] run_dir={run_dir}")
    manifest = load_manifest(run_dir)
    by_group = split_by_group(manifest)

    # Load groups with fixed token count [G, 16, 361, 128]
    print("[info] loading tensors …")
    smoke_names, Z_smoke = load_group(run_dir, by_group["smoke_z0"])
    davis_names, Z_davis = load_group(run_dir, by_group["davis_gen_z0"])
    g_names,     Z_g     = load_group(run_dir, by_group["gaussian"])
    # Smoke z1: variable token count (5632 for ss0..3,5..9; 5776 for ss4)
    # → mean-pool over spatial tokens to a uniform [16, 128] per clip.
    z1_names,    Z_z1_pool = load_group_pooled(run_dir, by_group["smoke_z1"])
    # Match: also pool gaussians to [16, 128] so they're comparable.
    _, Z_g_pool = load_group_pooled(run_dir, by_group["gaussian"])
    inject_entries       = by_group["exp_041_inject"]

    print(f"  smoke z0: {Z_smoke.shape}")
    print(f"  smoke z1 (pooled): {Z_z1_pool.shape}")
    print(f"  davis A_word: {Z_davis.shape}")
    print(f"  gaussian: {Z_g.shape}")
    print(f"  gaussian (pooled): {Z_g_pool.shape}")
    print(f"  exp_041 entries: {len(inject_entries)}")

    # ── [1] / [2] Per-sample trajectories ────────────────────────────────
    print("[render] [1] smoke per-sample trajectories")
    chart_1_per_sample_trajectory(
        run_dir, smoke_names, Z_smoke,
        title="[1] Per-sample frame trajectories — shadow_smoke z0 (local PCA per clip)",
        fname="01_smoke_per_sample_trajectories.png",
        cols=4,
    )
    print("[render] [2] davis per-sample trajectories")
    chart_1_per_sample_trajectory(
        run_dir, davis_names, Z_davis,
        title="[2] Per-sample frame trajectories — DAVIS A_word z0 (local PCA per clip)",
        fname="02_davis_per_sample_trajectories.png",
        cols=3,
    )

    # ── [3] Smoke unified frame PCA ──────────────────────────────────────
    print("[render] [3] smoke unified frame PCA")
    chart_3_smoke_unified_frame(run_dir, smoke_names, Z_smoke,
                                 fname="03_smoke_unified_frame.png")

    # ── [5] Whole-clip cleans PCA ────────────────────────────────────────
    print("[render] [5] whole-clip cleans PCA")
    chart_5_clip_pca_cleans(run_dir, smoke_names, Z_smoke, davis_names, Z_davis,
                             fname="05_clip_pca_cleans.png")

    # ── [6] Cross-domain frame PCA (cluster test) ────────────────────────
    print("[render] [6] cross-domain frame PCA cleans")
    chart_6_cross_domain_frame_cleans(run_dir, Z_smoke, Z_davis,
                                       fname="06_cross_domain_frame_cleans.png")

    # ── [7] Clip PCA z1 vs gaussian (spatial-pooled, uniform token count) ──
    print("[render] [7] clip z1 vs gaussian")
    chart_7_clip_z1_vs_gaussian(run_dir, Z_z1_pool, Z_g_pool,
                                 fname="07_clip_z1_vs_gaussian.png")

    # ── [8] Frame PCA z1 vs gaussian (spatial-pooled, 128-D) ─────────────
    print("[render] [8] frame z1 vs gaussian")
    chart_8_frame_z1_vs_gaussian(run_dir, Z_z1_pool, Z_g_pool,
                                  fname="08_frame_z1_vs_gaussian.png")

    # ── [9] / [9b] exp_041 injection PCA ─────────────────────────────────
    print("[render] [9] per-variant injection PCA")
    # Group injection entries by variant
    variants = sorted({e["name"].split("__")[0] for e in inject_entries})
    # Preserve config-style ordering: validation block first, then time-sweep, then cfg32
    variant_order = [
        "all48_alltime", "last47_alltime", "first10", "single10", "first20", "first0",
        "all48_early", "all48_mid", "all48_late",
        "all48_alltime_cfg32", "all48_early_cfg32",
    ]
    variants = [v for v in variant_order if v in variants]
    variant_dict = {v: _inject_role_subset(inject_entries, run_dir, v) for v in variants}
    chart_9_per_variant_injection(run_dir, variants, variant_dict,
                                   fname="09_per_variant_injection.png", cols=4)
    print("[render] [9b] all-variants overlay")
    chart_9b_injection_overlay(run_dir, variants, variant_dict,
                                fname="09b_injection_overlay.png")

    # ── Diagnostics ──────────────────────────────────────────────────────
    print("[render] per-channel z1 stats")
    chart_z1_per_channel_stats(run_dir, by_group["smoke_z1"], by_group["gaussian"],
                                run_dir, fname="10_z1_per_channel_stats.png")

    # Scree for the big joint PCAs.  Noise uses pooled 128-D frame vectors
    # since z1's have variable spatial token count.
    print("[render] scree plots")
    smoke_frames = Z_smoke.reshape(-1, SPATIAL_TOKENS * CHANNELS)
    davis_frames = Z_davis.reshape(-1, SPATIAL_TOKENS * CHANNELS)
    cleans_data  = np.concatenate([smoke_frames, davis_frames], axis=0)
    noise_data   = np.concatenate([Z_z1_pool.reshape(-1, CHANNELS),
                                    Z_g_pool.reshape(-1, CHANNELS)], axis=0)
    chart_scree(run_dir, "11_scree.png", [
        ("cleans frames (smoke+davis, 46208-D)", cleans_data),
        ("noise frames (z1+gauss, pooled 128-D)", noise_data),
    ])

    print(f"[done] charts → {charts_dir}")


if __name__ == "__main__":
    main()
