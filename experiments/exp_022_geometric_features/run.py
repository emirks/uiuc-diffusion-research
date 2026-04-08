"""exp_022 — Geometric Feature Extraction from VC Trajectories.

Loads Stage-1 trajectory files produced by exp_021 and computes a rich set
of geometric features to characterise the "dissolve" artefact that appears
in LTX-2 VC generation when start/end clips are semantically very different.

═══════════════════════════════════════════════════════════════════════════
DISSOLVE HYPOTHESIS
═══════════════════════════════════════════════════════════════════════════
When the VC task connects clips of very different semantic categories (class 6,
class 8), the model cannot smoothly interpolate.  Instead it produces a
dissolve-cut at some frame p* in the free-middle region.

Expected internal signatures:
  • curvature spike at p* in the clean latent z_0 — trajectory makes a
    hard turn in latent space exactly where the dissolve is
  • angular reversal (cos < 0) at p* — consecutive Δ_p vectors flip direction
  • pred_magnitude peak at p* — model exerts maximum correction at that frame
  • step_size spike at p* — per-frame denoising update is largest there

═══════════════════════════════════════════════════════════════════════════
LATENT GEOMETRY (PACKED TOKENS)
═══════════════════════════════════════════════════════════════════════════
The scheduler operates on packed tokens [B, N, C] where:
    N = F' × H' × W'   (temporal-first raster scan)
    C = 128             (VAE channel count)
  e.g. 512×768 Stage-1: F'=16, H'=16, W'=24 → N=6144

This script unpacks them to [C, F', H', W'] before computing spatial features.

═══════════════════════════════════════════════════════════════════════════
WHAT IS COMPUTED
═══════════════════════════════════════════════════════════════════════════
All features are 2-D matrices indexed by (denoising_step τ, frame p).
They are saved as .npz for further analysis.

    In z_t space (VAE latent, at each denoising step τ):
        norm_z      [S, F']    per-frame latent L2 norm  (noise level proxy)
        speed_z     [S, F'-1]  ‖Δ_p z_t(p)‖₂            frame-to-frame speed
        curvature_z [S, F'-2]  ‖Δ²_p z_t(p)‖₂           trajectory bend sharpness
        angular_z   [S, F'-2]  cos(Δ_p z, Δ_{p+1} z)    direction consistency

    In v_pred space (model prediction, at each τ):
        pred_mag    [S, F']    ‖v_θ(z_τ, τ, c)(p)‖₂      prediction intensity
        pred_curv   [S, F'-2]  ‖Δ²_p v_θ(p)‖₂            prediction curvature

    Across denoising time (for fixed frame p):
        step_size_z [S, F']    ‖z_{τ}(p) - z_{τ+1}(p)‖₂  per-frame update size

    At the final clean latent z_0 (most diagnostic for dissolve):
        norm_z0      [F']
        speed_z0     [F'-1]
        curvature_z0 [F'-2]    ← PRIMARY DISSOLVE SIGNAL
        angular_z0   [F'-2]    ← DIRECTION FLIP SIGNAL
        pred_mag0    [F']      (pred_mag at last denoising step)

Derived scalars (per sample):
    dissolve_frame     argmax of curvature_z0
    dissolve_strength  max(curvature_z0) / mean(curvature_z0)   (>1 = spike present)
    angular_min        min(angular_z0)                            (<0 = direction flip)
    dissolve_step_tau  earliest denoising step at which argmax(curvature_z[τ]) stabilises

═══════════════════════════════════════════════════════════════════════════
OUTPUTS
═══════════════════════════════════════════════════════════════════════════
    run_dir/
      features/<sample_id>.npz               feature arrays
      plots/<sample_id>_heatmaps.png          (τ × p) heatmaps for all features
      plots/<sample_id>_dissolve_profile.png  final-step dissolve signals
      plots/comparison_dissolve.png           all samples on one figure ← KEY
      plots/dissolve_frame_evolution.png      how dissolve_frame shifts over τ
      dissolve_table.csv                      per-sample scalars
      summary.yaml
      run.log

How to run:
    source /workspace/miniforge3/etc/profile.d/conda.sh
    conda activate /workspace/envs/diff
    cd /workspace/diffusion-research
    python experiments/exp_022_geometric_features/run.py
    python experiments/exp_022_geometric_features/run.py --traj_dir outputs/videos/exp_021_trajectory_logging/run_XXXX
"""
from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE = 8
LTX_SPATIAL_SCALE  = 32
VIDEO_FPS          = 24.0


def _sec_to_latent(t_s: float) -> float:
    """Convert video time (seconds) to continuous latent frame index.

    LTX-2 uses a **causal** temporal VAE with scale 8:
        p = 0          →  pixel 0  (single anchor frame)
        p ≥ 1          →  pixels (p-1)×8+1 … p×8  (8 pixels per latent)

    Inverse for continuous t:
        px = t × fps
        p  = (px − 1) / 8 + 1   for px > 0,   else 0
    """
    px = t_s * VIDEO_FPS
    if px <= 0.0:
        return 0.0
    return (px - 1.0) / LTX_TEMPORAL_SCALE + 1.0


def _latent_to_sec(p: float) -> float:
    """Convert latent frame index to representative video time (end of window)."""
    if p <= 0:
        return 0.0
    return p * LTX_TEMPORAL_SCALE / VIDEO_FPS

log = logging.getLogger(__name__)

# Class colours for comparison plots
CLASS_COLORS = {"1": "#2196F3", "2": "#4CAF50", "5": "#FF9800", "6": "#F44336", "8": "#9C27B0"}
CLASS_LABELS = {
    "1": "similar ctx / cat / motion",
    "2": "similar ctx / cat / diff motion",
    "5": "diff ctx / similar cat / similar motion",
    "6": "diff ctx / similar cat / diff motion",
    "8": "diff ctx / cat / motion",
}


# ── Latent unpacking ──────────────────────────────────────────────────────────

def unpack_tokens(z: torch.Tensor, F_prime: int, H_prime: int, W_prime: int) -> torch.Tensor:
    """Convert packed token tensor to spatial latent tensor.

    Args:
        z:  [..., N, C]  packed tokens (N = F'×H'×W', C = VAE channels)
    Returns:
        [..., C, F', H', W']  spatial latent
    """
    *leading, N, C = z.shape
    assert N == F_prime * H_prime * W_prime, (
        f"N={N} ≠ F'×H'×W' = {F_prime}×{H_prime}×{W_prime}={F_prime*H_prime*W_prime}. "
        f"Check latent spatial dimensions in config."
    )
    # Token order: temporal-first raster scan → reshape to [*, F', H', W', C]
    z_reshaped = z.reshape(*leading, F_prime, H_prime, W_prime, C)
    # Permute C to front: [..., C, F', H', W']
    ndim = len(leading)
    perm = list(range(ndim)) + [ndim + 3, ndim, ndim + 1, ndim + 2]
    return z_reshaped.permute(*perm).contiguous()


def load_sample(sample_dir: pathlib.Path) -> dict | None:
    """Load trajectory .pt and config_snapshot.yaml from one sample directory."""
    traj_files = list(sample_dir.glob("*_trajectory_stage1.pt"))
    if not traj_files:
        log.warning("No trajectory file in %s — skipping.", sample_dir.name)
        return None

    traj_path   = traj_files[0]
    config_path = sample_dir / "config_snapshot.yaml"

    data = torch.load(traj_path, weights_only=False, map_location="cpu")

    # Extract conditioning geometry from config snapshot
    if not config_path.exists():
        log.warning("No config_snapshot.yaml in %s — using defaults.", sample_dir.name)
        snap = {}
    else:
        with open(config_path) as f:
            snap = yaml.safe_load(f)

    inf = snap.get("inference", {})
    cc  = snap.get("clip_conditioning", {})

    height      = inf.get("height", 512)
    width       = inf.get("width", 768)
    num_frames  = inf.get("num_frames", 121)
    F_prime     = (num_frames - 1) // LTX_TEMPORAL_SCALE + 1
    H_prime     = height // LTX_SPATIAL_SCALE
    W_prime     = width  // LTX_SPATIAL_SCALE
    k_lat       = (cc.get("num_clip_frames", 25) - 1) // LTX_TEMPORAL_SCALE + 1
    end_idx     = cc.get("end_index", F_prime - k_lat)

    # Class from sample_id: "class8__..." → "8"
    sample_id   = data.get("sample_id", sample_dir.name)
    cls         = sample_id.split("__")[0].replace("class", "").strip()

    return {
        "sample_id": sample_id,
        "cls":       cls,
        "F_prime":   F_prime,
        "H_prime":   H_prime,
        "W_prime":   W_prime,
        "k_lat":     k_lat,
        "end_idx":   end_idx,
        "data":      data,
    }


# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(sample: dict) -> dict:
    """Compute all geometric features from the trajectory tensors."""
    d       = sample["data"]
    F, H, W = sample["F_prime"], sample["H_prime"], sample["W_prime"]

    # ── Unpack packed tokens to spatial latents ───────────────────────────────
    # z_t shape: [S, 1, N, C] → unpack → [S, C, F', H', W']
    z_raw = d["z_t"].float().squeeze(1)   # [S, N, C]
    v_raw = d["v_pred"].float().squeeze(1) # [S, N, C]
    S     = z_raw.shape[0]

    z = unpack_tokens(z_raw, F, H, W)    # [S, C, F', H', W']
    v = unpack_tokens(v_raw, F, H, W)    # [S, C, F', H', W']

    # z_final
    zf_raw = d["z_final"].float()         # [1, N, C] or [N, C]
    if zf_raw.ndim == 3:
        zf_raw = zf_raw.squeeze(0)        # [N, C]
    z0 = unpack_tokens(zf_raw, F, H, W)  # [C, F', H', W']

    log.info("  Unpacked: z_t %s → [S=%d, C=%d, F'=%d, H'=%d, W'=%d]",
             d["z_t"].shape, S, z.shape[1], F, H, W)

    # ── Helper: per-frame norm ────────────────────────────────────────────────
    # Flatten C, H', W' → single vector per (step, frame)
    def frame_norm(t: torch.Tensor) -> torch.Tensor:
        # t: [S, C, F', H', W'] → [S, F']
        return t.permute(0, 2, 1, 3, 4).flatten(2).norm(dim=2)

    def frame_norm_1d(t: torch.Tensor) -> torch.Tensor:
        # t: [C, F', H', W'] → [F']
        return t.permute(1, 0, 2, 3).flatten(1).norm(dim=1)

    def spatial_diff(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute speed, curvature, angular consistency along frame axis.
        t: [*, C, F', H', W']
        Returns: speed [*, F'-1], curvature [*, F'-2], angular [*, F'-2]
        """
        dz      = t[..., 1:, :, :] - t[..., :-1, :, :]        # [*, C, F'-1, H', W']
        # Flatten spatial dims, permute frame to axis -2
        if t.ndim == 5:  # [S, C, F', H', W']
            dz_flat = dz.permute(0, 2, 1, 3, 4).flatten(2)     # [S, F'-1, C*H'*W']
        else:            # [C, F', H', W']
            dz_flat = dz.permute(1, 0, 2, 3).flatten(1).unsqueeze(0)  # [1, F'-1, C*H'*W']
        speed = dz_flat.norm(dim=-1).squeeze(0)                 # [S, F'-1] or [F'-1]

        d2z     = dz[..., 1:, :, :] - dz[..., :-1, :, :]
        if t.ndim == 5:
            d2z_flat = d2z.permute(0, 2, 1, 3, 4).flatten(2)
        else:
            d2z_flat = d2z.permute(1, 0, 2, 3).flatten(1).unsqueeze(0)
        curv = d2z_flat.norm(dim=-1).squeeze(0)                 # [S, F'-2] or [F'-2]

        dz_n   = dz_flat / dz_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ang    = (dz_n[..., :-1, :] * dz_n[..., 1:, :]).sum(dim=-1).squeeze(0)  # [S, F'-2] or [F'-2]

        return speed, curv, ang

    # ── Compute z_t features ──────────────────────────────────────────────────
    norm_z           = frame_norm(z)            # [S, F']
    speed_z, curv_z, ang_z = spatial_diff(z)   # [S, F'-1], [S, F'-2], [S, F'-2]

    # ── Compute v_pred features ───────────────────────────────────────────────
    pred_mag         = frame_norm(v)            # [S, F']
    _, pred_curv, _  = spatial_diff(v)          # [S, F'-2]

    # ── Denoising step size per frame: Δ_τ z_p = ‖z_{τ+1}(p) - z_τ(p)‖ ─────
    # z[i] is z_t BEFORE step i.  z[i+1] = z_t before step i+1 = z after step i.
    # For the last step, z_after = z0.
    z_next      = torch.cat([z[1:], z0.unsqueeze(0)], dim=0)  # [S, C, F', H', W']
    step_size_z = frame_norm(z_next - z)        # [S, F']

    # ── Final clean latent z_0 features ──────────────────────────────────────
    norm_z0                  = frame_norm_1d(z0)                   # [F']
    speed_z0, curv_z0, ang_z0 = spatial_diff(z0)                  # [F'-1], [F'-2], [F'-2]
    pred_mag0                = pred_mag[-1]                        # [F']   (last step's v_pred)

    # ── Dissolve scalars ──────────────────────────────────────────────────────
    # Frame indices for curvature (centres of second-order differences, offset +1)
    F_c2 = F - 2  # length of curvature vectors
    curv0_np = curv_z0.numpy()
    dissolve_frame    = int(np.argmax(curv0_np)) + 1   # +1 because curv(p) spans frames p..p+2
    dissolve_strength = float(curv0_np.max() / (curv0_np.mean() + 1e-8))
    angular_min       = float(ang_z0.numpy().min())

    # Dissolve frame estimate across all denoising steps (to see when it stabilises)
    dissolve_frame_by_step = np.argmax(curv_z.numpy(), axis=1) + 1  # [S]

    # Free-middle-restricted dissolve frame.
    # curvature_z0[i] is centred at frame i+1 (discrete Laplacian).
    # Free middle: p = k_lat .. end_idx-1  →  curvature indices i = k_lat-1 .. end_idx-2
    k_lat_   = sample["k_lat"]
    end_idx_ = sample["end_idx"]
    fi_s = k_lat_ - 1          # first curvature index inside free middle (= 3)
    fi_e = end_idx_ - 1        # one past last  (= 11)
    free_curv = curv0_np[fi_s:fi_e]  # length = end_idx - k_lat (= 8)
    if len(free_curv) > 0:
        j = int(np.argmax(free_curv))
        dissolve_frame_free = j + fi_s + 1          # +1 for centring offset
        dissolve_s_free     = _latent_to_sec(dissolve_frame_free)
    else:
        dissolve_frame_free = dissolve_frame
        dissolve_s_free     = _latent_to_sec(dissolve_frame)

    dissolve_s_global = _latent_to_sec(dissolve_frame)

    log.info(
        "  dissolve_frame=%d (%.2fs)  free=%d (%.2fs)  strength=%.2f  angular_min=%.3f",
        dissolve_frame, dissolve_s_global,
        dissolve_frame_free, dissolve_s_free,
        dissolve_strength, angular_min,
    )

    return {
        # 2-D matrices [S, *]
        "norm_z":      norm_z.numpy(),           # [S, F']
        "speed_z":     speed_z.numpy(),          # [S, F'-1]
        "curvature_z": curv_z.numpy(),           # [S, F'-2]
        "angular_z":   ang_z.numpy(),            # [S, F'-2]
        "pred_mag":    pred_mag.numpy(),         # [S, F']
        "pred_curv":   pred_curv.numpy(),        # [S, F'-2]
        "step_size_z": step_size_z.numpy(),      # [S, F']

        # 1-D vectors for the final z_0
        "norm_z0":     norm_z0.numpy(),          # [F']
        "speed_z0":    speed_z0.numpy(),         # [F'-1]
        "curvature_z0": curv_z0.numpy(),         # [F'-2]  ← PRIMARY DISSOLVE SIGNAL
        "angular_z0":  ang_z0.numpy(),           # [F'-2]  ← DIRECTION FLIP SIGNAL
        "pred_mag0":   pred_mag0.numpy(),        # [F']

        # Dissolve scalars — global (argmax over all frames)
        "dissolve_frame":          dissolve_frame,
        "dissolve_s":              dissolve_s_global,
        "dissolve_strength":       dissolve_strength,
        "angular_min":             angular_min,

        # Dissolve scalars — free-middle restricted (argmax within p=k_lat..end_idx-1)
        "dissolve_frame_free":     dissolve_frame_free,
        "dissolve_s_free":         dissolve_s_free,

        "dissolve_frame_by_step":  dissolve_frame_by_step,  # [S]

        # Metadata
        "timesteps": np.array(d["timesteps"]),
        "S": S, "F": F, "H": H, "W": W,
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _cond_shade(ax, F_prime: int, k_lat: int, end_idx: int, alpha: float = 0.12) -> None:
    """Shade conditioned regions and draw boundary lines on ax."""
    ax.axvspan(-0.5, k_lat - 0.5,           alpha=alpha, color="#00BCD4", zorder=0)
    ax.axvspan(end_idx - 0.5, F_prime - 0.5, alpha=alpha, color="#E91E63", zorder=0)
    ax.axvline(k_lat - 0.5,   color="#00BCD4", linewidth=1.2, linestyle="--", alpha=0.9)
    ax.axvline(end_idx - 0.5, color="#E91E63", linewidth=1.2, linestyle="--", alpha=0.9)


def _heatmap(ax, data: np.ndarray, title: str, cmap: str, vmin=None, vmax=None,
             k_lat: int = 0, end_idx: int = 0, diverging: bool = False,
             F_prime: int = 0) -> None:
    """Draw one (τ × p) heatmap panel with conditioning markers.

    Args:
        F_prime: full latent frame count (16).  Used to compute boundary positions
                 for features shorter than F' (speed = F'-1, curvature/angular = F'-2).
                 Pass 0 to use the raw k_lat/end_idx values without adjustment.
    """
    if diverging:
        vmin, vmax = -1.0, 1.0
    else:
        vmin  = vmin if vmin is not None else 0.0
        vmax  = vmax if vmax is not None else float(np.percentile(data, 99))

    S, P = data.shape
    im = ax.imshow(
        data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        origin="upper", interpolation="nearest",
        extent=[-0.5, P - 0.5, S - 0.5, -0.5],   # x=columns, y=steps (0=noisy top)
    )
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("frame  p", fontsize=7)
    # τ=0 (noisy) is the top row; τ increases downward toward clean.
    ax.set_ylabel("denoise step  τ  (0 = noisy)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xticks(range(P))

    # Conditioning boundary lines — adjusted for features shorter than F'.
    # For a feature of length P = F' - offset, column i represents the
    # window centred at frame  i + offset/2 (offset=0 → norm/pred/step,
    # offset=0.5 → speed, offset=1 → curvature/angular).
    # The frame-space boundaries are k_lat-0.5 and end_idx-0.5, so in
    # column space they shift by  -(F'-P)/2 = -offset/2.
    offset = (F_prime - P) if F_prime > 0 else 0   # 0, 1, or 2
    shift  = offset / 2.0                            # 0, 0.5, or 1.0
    for bnd_frame, col in [(k_lat - 0.5, "#00BCD4"), (end_idx - 0.5, "#E91E63")]:
        bnd_col = bnd_frame - shift
        if -0.5 < bnd_col < P - 0.5:
            ax.axvline(bnd_col, color=col, linewidth=1.2, linestyle="--", alpha=0.85)

    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)


# ── Per-sample figures ────────────────────────────────────────────────────────

def plot_heatmaps(feats: dict, sample_id: str, cls: str,
                  k_lat: int, end_idx: int, out_path: pathlib.Path) -> None:
    """6-panel heatmap figure: all features as (τ × p) matrices."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{sample_id}\nClass {cls}: {CLASS_LABELS.get(cls, '?')}  "
        f"|  k_lat={k_lat}  end_idx={end_idx}  "
        f"|  cyan=start, magenta=end conditioning",
        fontsize=10, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    F = feats["F"]

    panels = [
        # (key, title, cmap, diverging)
        ("norm_z",      "‖z_t(p)‖ — noise level per frame",             "PuBu_r",   False),
        ("pred_mag",    "‖v_θ(p)‖ — prediction intensity per frame",     "plasma",    False),
        ("speed_z",     "speed_z  ‖Δ_p z_t‖  — frame-to-frame speed",   "YlOrRd",   False),
        ("step_size_z", "step_size_z  ‖z_{τ+1}(p) - z_τ(p)‖ — update/frame", "magma", False),
        ("curvature_z", "curvature_z  ‖Δ²_p z_t‖  ← DISSOLVE BEND",     "hot",       False),
        ("angular_z",   "angular_z  cos(Δ_p z, Δ_{p+1} z) ← DIRECTION FLIP", "RdYlGn", True),
    ]

    for i, (key, title, cmap, div) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        _heatmap(ax, feats[key], title, cmap, k_lat=k_lat, end_idx=end_idx,
                 diverging=div, F_prime=F)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved heatmaps: %s", out_path.name)


def _draw_dissolve_markers(ax, df_global: int, df_free: int, gt_s: float | None,
                           F: int) -> None:
    """Draw prediction and ground-truth vertical markers on a dissolve panel.

    Yellow solid  = global argmax (over all frames).
    Orange dashed = free-middle argmax (p = k_lat..end_idx-1).
    Green  solid  = visual ground truth annotated from video.
    """
    ax.axvline(df_global, color="#FFC107", linewidth=2.0, linestyle="-",
               label=f"pred global p={df_global}", zorder=5)
    if df_free != df_global:
        ax.axvline(df_free, color="#FF6D00", linewidth=1.8, linestyle="--",
                   label=f"pred free-mid p={df_free}", zorder=5)
    if gt_s is not None:
        gt_p = _sec_to_latent(gt_s)
        ax.axvline(gt_p, color="#4CAF50", linewidth=2.2, linestyle="-",
                   label=f"GT  {gt_s:.2f}s  p≈{gt_p:.1f}", zorder=6)


def plot_dissolve_profile(feats: dict, sample_id: str, cls: str,
                          k_lat: int, end_idx: int, out_path: pathlib.Path,
                          gt_s: float | None = None) -> None:
    """5-panel final-step (z_0) dissolve signals.

    Markers:
      yellow solid  = global argmax prediction
      orange dashed = free-middle restricted prediction
      green  solid  = visual ground truth (if provided)
    """
    F       = feats["F"]
    frames  = np.arange(F)
    f_spd   = np.arange(F - 1) + 0.5       # midpoints for speed
    f_c2    = np.arange(F - 2) + 1.0       # centres for curvature/angular

    curv0   = feats["curvature_z0"]
    ang0    = feats["angular_z0"]
    spd0    = feats["speed_z0"]
    norm0   = feats["norm_z0"]
    pmag0   = feats["pred_mag0"]
    df      = feats["dissolve_frame"]
    df_free = feats["dissolve_frame_free"]

    # GT annotation in title
    gt_str = f"  |  GT = {gt_s:.2f} s  (p≈{_sec_to_latent(gt_s):.1f})" if gt_s is not None else ""
    err_global = f"  err={abs(feats['dissolve_s'] - gt_s):.2f}s" if gt_s is not None else ""
    err_free   = f"  err={abs(feats['dissolve_s_free'] - gt_s):.2f}s" if gt_s is not None else ""

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=False)
    fig.suptitle(
        f"DISSOLVE SIGNALS  (final clean latent z₀)\n"
        f"{sample_id}   [Class {cls}: {CLASS_LABELS.get(cls, '?')}]{gt_str}",
        fontsize=11, fontweight="bold",
    )

    # ── 1. Curvature — primary signal ────────────────────────────────────────
    ax = axes[0]
    ax.bar(f_c2, curv0, width=0.8, color="#FF5722", alpha=0.85)
    _draw_dissolve_markers(ax, df, df_free, gt_s, F)
    ax.set_ylabel("‖Δ²_p z₀‖₂", fontsize=9)
    ax.set_title(
        f"curvature_z0  =  ‖z₀(p+2) − 2·z₀(p+1) + z₀(p)‖₂   "
        f"[strength={feats['dissolve_strength']:.2f}×]"
        f"  global{err_global}  free{err_free}",
        fontsize=8, fontweight="bold",
    )
    ax.legend(fontsize=7.5, loc="upper left")

    # ── 2. Angular consistency — direction flip ───────────────────────────────
    ax = axes[1]
    ax.plot(f_c2, ang0, "o-", color="#2196F3", linewidth=1.5, markersize=4)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.fill_between(f_c2, ang0, 0, where=(ang0 < 0), color="red", alpha=0.25,
                    label="direction reversal")
    _draw_dissolve_markers(ax, df, df_free, gt_s, F)
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("cos(Δ,Δ)", fontsize=9)
    ax.set_title(
        f"angular_z0  =  cos(z₀(p+1)−z₀(p),  z₀(p+2)−z₀(p+1))   "
        f"[min={feats['angular_min']:.3f}]",
        fontsize=8, fontweight="bold",
    )
    ax.legend(fontsize=7.5, loc="upper left")

    # ── 3. Speed ──────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(f_spd, spd0, "s-", color="#9C27B0", linewidth=1.5, markersize=4)
    _draw_dissolve_markers(ax, df, df_free, gt_s, F)
    ax.set_ylabel("‖Δ_p z₀‖₂", fontsize=9)
    ax.set_title(
        "speed_z0  =  ‖z₀(p+1) − z₀(p)‖₂   — frame-to-frame jump magnitude",
        fontsize=8, fontweight="bold",
    )

    # ── 4. Prediction magnitude ───────────────────────────────────────────────
    ax = axes[3]
    ax.bar(frames, pmag0, width=0.8, color="#607D8B", alpha=0.85)
    _draw_dissolve_markers(ax, df, df_free, gt_s, F)
    ax.set_ylabel("‖v_θ(p)‖₂", fontsize=9)
    ax.set_title(
        "pred_mag (final step)  =  ‖v_θ(z_τ, τ, c)(p)‖₂   — where model works hardest",
        fontsize=8, fontweight="bold",
    )

    # ── 5. Per-frame z_0 norm ─────────────────────────────────────────────────
    ax = axes[4]
    ax.bar(frames, norm0, width=0.8, color="#4CAF50", alpha=0.85)
    _draw_dissolve_markers(ax, df, df_free, gt_s, F)
    ax.set_ylabel("‖z₀(p)‖₂", fontsize=9)
    ax.set_title(
        "norm_z0  =  ‖z₀(p)‖₂   — per-frame latent energy",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("frame  p", fontsize=9)

    # Shared formatting
    for ax in axes:
        ax.set_xlim(-0.5, F - 0.5)
        ax.set_xticks(range(F))
        ax.tick_params(labelsize=8)
        _cond_shade(ax, F, k_lat, end_idx)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved dissolve profile: %s", out_path.name)


# ── Cross-sample comparison figures ──────────────────────────────────────────

def plot_comparison(all_samples: list[dict], out_dir: pathlib.Path,
                    gt_dict: dict[str, float | None] | None = None) -> None:
    """Key comparison plot: curvature_z0 and angular_z0 across all samples.

    Thin vertical tick marks show:
      dashed coloured  = model prediction (free-middle argmax)
      solid   black    = visual ground truth (where provided)
    """
    all_samples = sorted(all_samples, key=lambda s: s["cls"])
    gt_dict     = gt_dict or {}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle(
        "Cross-sample Dissolve Signal Comparison  (clean latent z₀)\n"
        "Dashed coloured ticks = free-mid prediction   |   black ticks = visual GT",
        fontsize=12, fontweight="bold",
    )

    F    = all_samples[0]["feats"]["F"]
    f_c2 = np.arange(F - 2) + 1.0

    from matplotlib.lines import Line2D

    for s in all_samples:
        curv  = s["feats"]["curvature_z0"]
        ang   = s["feats"]["angular_z0"]
        cls   = s["cls"]
        sid   = s["sample_id"]
        short = sid.replace(f"class{cls}__", "").replace("__", "→")
        label = f"[C{cls}] {short}"
        color = CLASS_COLORS.get(cls, "#888888")
        df_f  = s["feats"]["dissolve_frame_free"]
        gt_s  = gt_dict.get(sid)

        curv_n = curv / (curv.max() + 1e-8)
        ax1.plot(f_c2, curv_n, "-", color=color, linewidth=1.8, alpha=0.85, label=label)
        ax1.axvline(df_f, color=color, linewidth=1.0, linestyle="--", alpha=0.7)

        ax2.plot(f_c2, ang, "-", color=color, linewidth=1.8, alpha=0.85, label=label)
        ax2.axvline(df_f, color=color, linewidth=1.0, linestyle="--", alpha=0.7)

        if gt_s is not None:
            gt_p = _sec_to_latent(gt_s)
            ax1.axvline(gt_p, color=color, linewidth=2.2, linestyle="-", alpha=0.95)
            ax2.axvline(gt_p, color=color, linewidth=2.2, linestyle="-", alpha=0.95)

    # Legend: sample lines
    ax1.set_title(
        "Normalised curvature_z0  =  ‖z₀(p+2)−2z₀(p+1)+z₀(p)‖₂ / max\n"
        "Free-mid peak = estimated dissolve frame within generated zone",
        fontsize=9,
    )
    ax1.set_ylabel("curvature / max", fontsize=9)
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(fontsize=7, ncol=2, loc="upper left")

    # Extra legend for marker types
    marker_handles = [
        Line2D([0], [0], color="gray", linewidth=2.2, linestyle="-",  label="visual GT (solid)"),
        Line2D([0], [0], color="gray", linewidth=1.0, linestyle="--", label="free-mid pred (dashed)"),
    ]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + marker_handles,
               labels=ax1.get_legend_handles_labels()[1] + [h.get_label() for h in marker_handles],
               fontsize=7, ncol=2, loc="upper left")

    ax2.set_title(
        "angular_z0  =  cos(z₀(p+1)−z₀(p),  z₀(p+2)−z₀(p+1))\n"
        "Negative = direction reversal at dissolve onset",
        fontsize=9,
    )
    ax2.set_ylabel("cos(Δ_p z, Δ_{p+1} z)", fontsize=9)
    ax2.set_ylim(-1.15, 1.15)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax2.legend(fontsize=7, ncol=2, loc="upper left")

    for ax in (ax1, ax2):
        ax.set_xlabel("frame  p", fontsize=9)
        ax.set_xticks(range(F))
        ax.tick_params(labelsize=8)
        ax.set_xlim(-0.5, F - 0.5)

    plt.tight_layout()
    out = out_dir / "comparison_dissolve.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved comparison: %s", out.name)


def plot_dissolve_evolution(all_samples: list[dict], out_dir: pathlib.Path) -> None:
    """Show how the estimated dissolve frame shifts across denoising steps τ.

    This reveals WHEN the model commits to its dissolve location — early (high τ,
    still noisy) or late (low τ, nearly clean).
    """
    all_samples = sorted(all_samples, key=lambda s: s["cls"])

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "Dissolve Frame Estimate  vs.  Denoising Step  τ\n"
        "Shows when the model commits to its dissolve location during generation",
        fontsize=11, fontweight="bold",
    )

    S = all_samples[0]["feats"]["S"]
    steps = np.arange(S)

    for s in all_samples:
        dfe   = s["feats"]["dissolve_frame_by_step"]   # [S]
        cls   = s["cls"]
        sid   = s["sample_id"]
        short = sid.replace(f"class{cls}__", "").replace("__", "→")
        label = f"[C{cls}] {short}"
        color = CLASS_COLORS.get(cls, "#888888")
        ax.plot(steps, dfe, "-", color=color, linewidth=1.8, alpha=0.85, label=label)

    ax.set_xlabel("denoising step  τ  (0 = start/noisy,  39 = end/clean)", fontsize=9)
    ax.set_ylabel("estimated dissolve frame  p*", fontsize=9)
    ax.set_xticks(steps[::5])
    ax.tick_params(labelsize=8)

    # Mark free-middle region
    k_lat   = all_samples[0]["sample"]["k_lat"]
    end_idx = all_samples[0]["sample"]["end_idx"]
    F       = all_samples[0]["feats"]["F"]
    ax.axhspan(k_lat, end_idx - 1, alpha=0.08, color="gold", label=f"free middle p=[{k_lat}..{end_idx-1}]")

    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    out = out_dir / "dissolve_frame_evolution.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved evolution plot: %s", out.name)


def plot_curvature_heatmap_grid(all_samples: list[dict], out_dir: pathlib.Path) -> None:
    """Grid of curvature_z heatmaps for all samples (τ × p), sorted by class."""
    all_samples = sorted(all_samples, key=lambda s: s["cls"])
    n = len(all_samples)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    if rows == 1:
        axes = [axes] if n == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    fig.suptitle(
        "curvature_z  (τ × p)  for all samples\n"
        "Bright = high curvature (sharp turn in latent trajectory)",
        fontsize=11, fontweight="bold",
    )

    global_vmax = max(float(np.percentile(s["feats"]["curvature_z"], 99)) for s in all_samples)

    for ax, s in zip(axes, all_samples):
        F    = s["feats"]["F"]
        k    = s["sample"]["k_lat"]
        e    = s["sample"]["end_idx"]
        cls  = s["cls"]
        sid  = s["sample_id"].replace(f"class{cls}__", "")

        _heatmap(ax, s["feats"]["curvature_z"],
                 f"[C{cls}] {sid}", "hot", vmax=global_vmax, k_lat=k, end_idx=e,
                 F_prime=F)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out = out_dir / "curvature_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved curvature grid: %s", out.name)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="exp_022 — geometric feature extraction")
    parser.add_argument("--config",   type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--traj_dir", type=pathlib.Path, default=None,
                        help="Override trajectory_dir from config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    traj_dir = pathlib.Path(
        args.traj_dir or REPO_ROOT / cfg["trajectory_dir"]
    )
    out_root = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_root)

    feat_dir = run_dir / "features"
    plot_dir = run_dir / "plots"
    feat_dir.mkdir()
    plot_dir.mkdir()

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
            force=True,
        )
        log.info("run_dir    : %s", run_dir)
        log.info("traj_dir   : %s", traj_dir)

        # ── Ground truth annotations ──────────────────────────────────────────
        gt_dict: dict[str, float | None] = {
            k: v for k, v in cfg.get("ground_truth", {}).items()
        }
        n_gt = sum(1 for v in gt_dict.values() if v is not None)
        log.info("Ground truth annotations loaded: %d / %d", n_gt, len(gt_dict))

        # ── Discover samples ──────────────────────────────────────────────────
        sample_dirs = sorted(
            p for p in traj_dir.iterdir()
            if p.is_dir() and (p / "config_snapshot.yaml").exists()
        )
        log.info("Found %d sample directories.", len(sample_dirs))

        all_records: list[dict] = []
        t0_all = time.perf_counter()

        for sample_dir in sample_dirs:
            log.info("─── %s ───", sample_dir.name)
            t0 = time.perf_counter()

            sample = load_sample(sample_dir)
            if sample is None:
                continue

            feats = compute_features(sample)

            # Save .npz
            npz_path = feat_dir / f"{sample['sample_id']}.npz"
            np.savez_compressed(npz_path, **{
                k: v for k, v in feats.items()
                if isinstance(v, np.ndarray)
            })
            log.info("  Features saved: %s  (%.1f KB)", npz_path.name,
                     npz_path.stat().st_size / 1024)

            sid = sample["sample_id"]
            cls = sample["cls"]
            k   = sample["k_lat"]
            e   = sample["end_idx"]
            gt_s = gt_dict.get(sid)

            plot_heatmaps(feats, sid, cls, k, e, plot_dir / f"{sid}_heatmaps.png")
            plot_dissolve_profile(feats, sid, cls, k, e,
                                  plot_dir / f"{sid}_dissolve_profile.png",
                                  gt_s=gt_s)

            all_records.append({
                "sample":    sample,
                "feats":     feats,
                "sample_id": sid,
                "cls":       cls,
            })
            log.info("  Done in %.1fs.", time.perf_counter() - t0)

        if not all_records:
            log.error("No valid samples found — nothing to plot.")
            return

        # ── Cross-sample figures ──────────────────────────────────────────────
        log.info("Generating cross-sample comparison plots …")
        plot_comparison(all_records, plot_dir, gt_dict=gt_dict)
        plot_dissolve_evolution(all_records, plot_dir)
        plot_curvature_heatmap_grid(all_records, plot_dir)

        # ── Dissolve summary table ────────────────────────────────────────────
        def _fmt(v, spec):
            return format(v, spec) if v is not None else ""

        csv_path = run_dir / "dissolve_table.csv"
        fieldnames = [
            "sample_id", "class",
            "pred_global_p", "pred_global_s",
            "pred_free_p",   "pred_free_s",
            "gt_s",          "gt_p_approx",
            "err_global_s",  "err_free_s",
            "dissolve_strength", "angular_min",
            "k_lat", "end_idx",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rec in sorted(all_records, key=lambda r: r["cls"]):
                ft  = rec["feats"]
                sm  = rec["sample"]
                sid = rec["sample_id"]
                gt  = gt_dict.get(sid)
                gt_p = round(_sec_to_latent(gt), 1) if gt is not None else None
                w.writerow({
                    "sample_id":       sid,
                    "class":           rec["cls"],
                    "pred_global_p":   ft["dissolve_frame"],
                    "pred_global_s":   f"{ft['dissolve_s']:.2f}",
                    "pred_free_p":     ft["dissolve_frame_free"],
                    "pred_free_s":     f"{ft['dissolve_s_free']:.2f}",
                    "gt_s":            _fmt(gt, ".2f"),
                    "gt_p_approx":     _fmt(gt_p, ".1f"),
                    "err_global_s":    _fmt(abs(ft["dissolve_s"] - gt), ".2f") if gt else "",
                    "err_free_s":      _fmt(abs(ft["dissolve_s_free"] - gt), ".2f") if gt else "",
                    "dissolve_strength": f"{ft['dissolve_strength']:.3f}",
                    "angular_min":     f"{ft['angular_min']:.4f}",
                    "k_lat":           sm["k_lat"],
                    "end_idx":         sm["end_idx"],
                })
        log.info("Dissolve table: %s", csv_path.name)

        # Print validation table to console
        log.info("")
        log.info("═" * 115)
        log.info(
            "  %-43s  %s  %8s  %8s  %8s  %7s  %7s  %s",
            "sample_id", "cls", "pred_g_s", "pred_f_s", "gt_s",
            "err_g", "err_f", "strength",
        )
        log.info("─" * 115)
        for rec in sorted(all_records, key=lambda r: r["cls"]):
            ft   = rec["feats"]
            gt   = gt_dict.get(rec["sample_id"])
            eg   = _fmt(abs(ft["dissolve_s"] - gt), ".2f") if gt else "  —  "
            ef   = _fmt(abs(ft["dissolve_s_free"] - gt), ".2f") if gt else "  —  "
            log.info(
                "  %-43s  %-3s  %8.2f  %8.2f  %8s  %7s  %7s  %.2f",
                rec["sample_id"], rec["cls"],
                ft["dissolve_s"], ft["dissolve_s_free"],
                _fmt(gt, ".2f") if gt else "  —  ",
                eg, ef, ft["dissolve_strength"],
            )
        log.info("═" * 115)

        # ── summary.yaml ─────────────────────────────────────────────────────
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({
                "run_id":    run_id,
                "traj_dir":  str(traj_dir),
                "n_samples": len(all_records),
                "elapsed_s": round(time.perf_counter() - t0_all, 1),
                "samples": [
                    {
                        "sample_id":           r["sample_id"],
                        "class":               r["cls"],
                        "pred_global_p":        r["feats"]["dissolve_frame"],
                        "pred_global_s":        round(r["feats"]["dissolve_s"], 2),
                        "pred_free_p":          r["feats"]["dissolve_frame_free"],
                        "pred_free_s":          round(r["feats"]["dissolve_s_free"], 2),
                        "gt_s":                gt_dict.get(r["sample_id"]),
                        "dissolve_strength":   round(r["feats"]["dissolve_strength"], 3),
                        "angular_min":         round(r["feats"]["angular_min"], 4),
                    }
                    for r in sorted(all_records, key=lambda x: x["cls"])
                ],
            }, f, sort_keys=False)

        log.info("All done.  run_dir: %s", run_dir)


if __name__ == "__main__":
    main()
