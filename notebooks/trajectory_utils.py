"""
trajectory_utils.py
===================
Shared utilities for the LTX-2 C2V latent trajectory analysis notebooks.

Exports
-------
Constants:
    TRAJ_DIR, LTX_TEMPORAL_SCALE, VIDEO_FPS, C_LATENT, F_PRIME, H_PRIME,
    W_PRIME, S_STEPS, HIDDEN_DIM, K_LAT, END_IDX, GT_CSV
    CLASS_COLORS, CLASS_LABELS
    LAYER_INDICES, STEP_INDICES, PCA_TIMESTEPS

Feature computation:
    frame_norm(t)           → [S, F']
    frame_norm_1d(t)        → [F']
    spatial_diff(t)         → speed, curvature, angular
    compute_features(data)  → dict

Plotting:
    shade_cond(ax, data_len)
    heatmap(ax, data, title, cmap, ...)
    video_html(path, width, label, cls)  → str
    video_grid(records, cols, width)     → HTML

GT helpers:
    load_gt_annotations()          → gt_dict  (sample_id → seconds or None)
    gt_latent(sample_id, gt_dict)  → float | None  (latent frame, sub-frame)
    add_gt_vline(ax, sample_id, gt_dict, ...)

Data loading:
    load_record(sample_dir)  → dict | None
    load_all_records()       → list[dict]

Analysis helpers:
    commit_step(dissolve_by_step, tolerance)  → int

Hidden-state helpers (Level 3):
    unpack_hidden(h)           → [2, F', H', W', D]
    per_frame_norm(h_spatial)  → [F']
    pca_frame_embeddings(h_spatial, n_comp)  → coords, explained_var
    cosine_sim_matrix(h_spatial)  → [F', F']
"""

import pathlib
import base64
import warnings
import gc

import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from IPython.display import HTML, display

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 110, "font.size": 9})


# ── CONFIG ─────────────────────────────────────────────────────────────────────

TRAJ_DIR = pathlib.Path(
    "/workspace/diffusion-research/outputs/videos/exp_021_trajectory_logging/run_0004"
)

GT_CSV = pathlib.Path(
    "/workspace/diffusion-research/outputs/analysis/"
    "exp_022_geometric_features/run_0004/dissolve_table.csv"
)

# LTX-2 architecture constants (from exp_021 config)
LTX_TEMPORAL_SCALE = 8        # pixel frames per latent frame
VIDEO_FPS          = 24.0
C_LATENT           = 128      # VAE channel count
F_PRIME            = 16       # latent temporal frames  (121 pixel frames / 8 ≈ 16)
H_PRIME            = 16       # latent height           (512 px / 32)
W_PRIME            = 24       # latent width            (768 px / 32)
S_STEPS            = 40       # Stage-1 denoising steps
HIDDEN_DIM         = 4096     # transformer hidden dimension

# Conditioning geometry  (num_clip_frames=25, num_frames=121)
#   K_LAT = number of latent frames taken from each conditioning clip
#   END_IDX = first latent frame index of the end-clip region
K_LAT   = (25 - 1) // LTX_TEMPORAL_SCALE + 1   # = 4
END_IDX = F_PRIME - K_LAT                        # = 12

# Semantic class palette (from exp_022)
CLASS_COLORS = {
    "1": "#2196F3",
    "2": "#4CAF50",
    "5": "#FF9800",
    "6": "#F44336",
    "8": "#9C27B0",
}
CLASS_LABELS = {
    "1": "similar ctx / cat / motion",
    "2": "similar ctx / cat / diff motion",
    "5": "diff ctx / similar cat / similar motion",
    "6": "diff ctx / similar cat / diff motion",
    "8": "diff ctx / cat / motion",
}

# Transformer probing config (Level 3)
LAYER_INDICES = [12, 24, 35, 47]        # block indices (of 48) to probe
STEP_INDICES  = [0, 8, 16, 23, 31, 39]  # denoising steps at which hidden states were saved
PCA_TIMESTEPS = [0, 8, 16, 39]          # subset of STEP_INDICES for PCA grids


# ── FEATURE COMPUTATION ────────────────────────────────────────────────────────
#
# All functions operate on tensors already unpacked from the .pt file:
#   z_t      : [S, C, F', H', W']   noisy latent trajectory
#   z_final  : [C, F', H', W']      final clean latent z₀

def frame_norm(t: torch.Tensor) -> torch.Tensor:
    """Per-frame L2 norm across channels and spatial dims.

    Args:
        t: [S, C, F', H', W']
    Returns:
        [S, F'] — noise-level proxy per frame per step.
    """
    return t.pow(2).sum(dim=(1, 3, 4)).sqrt()


def frame_norm_1d(t: torch.Tensor) -> torch.Tensor:
    """Per-frame L2 norm for the final (4-D) clean latent.

    Args:
        t: [C, F', H', W']
    Returns:
        [F']
    """
    return t.pow(2).sum(dim=(0, 2, 3)).sqrt()


def spatial_diff(t: torch.Tensor):
    """Finite differences along the latent-frame axis F'.

    Computes three geometric signals from the frame sequence:
    - **speed**     ‖z(p+1) − z(p)‖       → how much content changes between adjacent frames
    - **curvature** ‖z(p+2) − 2z(p+1) + z(p)‖  → how sharply the trajectory bends at frame p
                    (primary dissolve/cut detector — peaks at a hard transition)
    - **angular**   cos(Δz(p), Δz(p+1))   → whether the trajectory reverses direction
                    (< 0 = direction flip = abrupt content change)

    Args:
        t: [S, C, F', H', W']  or  [C, F', H', W'] (4-D for z₀)
    Returns:
        speed     : [S, F'-1] or [F'-1]
        curvature : [S, F'-2] or [F'-2]
        angular   : [S, F'-2] or [F'-2]
    """
    is_5d = (t.ndim == 5)
    if not is_5d:
        t = t.unsqueeze(0)

    dz    = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]          # [S, C, F'-1, H', W']
    speed = dz.pow(2).sum(dim=(1, 3, 4)).sqrt()              # [S, F'-1]

    d2z  = dz[:, :, 1:, :, :] - dz[:, :, :-1, :, :]         # [S, C, F'-2, H', W']
    curv = d2z.pow(2).sum(dim=(1, 3, 4)).sqrt()              # [S, F'-2]

    dz_flat = dz.permute(0, 2, 1, 3, 4).flatten(2)           # [S, F'-1, C*H'*W']
    dz_unit = dz_flat / dz_flat.norm(dim=2, keepdim=True).clamp(min=1e-8)
    ang = (dz_unit[:, :-1, :] * dz_unit[:, 1:, :]).sum(dim=2)  # [S, F'-2]

    if not is_5d:
        return speed.squeeze(0), curv.squeeze(0), ang.squeeze(0)
    return speed, curv, ang


def compute_features(data: dict) -> dict:
    """Extract all geometric features from one trajectory .pt dict.

    The raw tensors (z_t, v_pred) are ~100 MB each.  Call this once, then
    discard `data` with `del data; gc.collect()`.

    Returns a flat dict of numpy arrays and scalar values.
    """
    z  = data["z_t"].float()      # [S, C, F', H', W']
    v  = data["v_pred"].float()   # [S, C, F', H', W']
    z0 = data["z_final"].float()  # [C, F', H', W']
    S, _, F = z.shape[0], z.shape[1], z.shape[2]
    ts = np.array(data["timesteps"])   # [S] — descending from ~1000 to ~0

    # ── Level 1: VAE latent features ──────────────────────────────────────────
    norm_z                 = frame_norm(z)           # [S, F']
    speed_z, curv_z, ang_z = spatial_diff(z)        # [S,F'-1], [S,F'-2], [S,F'-2]

    # ── Level 2: Velocity field features ──────────────────────────────────────
    pred_mag             = frame_norm(v)             # [S, F']
    _, pred_curv, _      = spatial_diff(v)           # [S, F'-2]

    # Step-size: how much each frame changes between consecutive denoising steps
    z_next      = torch.cat([z[1:], z0.unsqueeze(0)], dim=0)  # [S, C, F', H', W']
    step_size_z = frame_norm(z_next - z)                       # [S, F']

    # ── Final clean-latent z₀ signals ─────────────────────────────────────────
    # These are the most diagnostic for locating the dissolve frame.
    norm_z0                    = frame_norm_1d(z0)        # [F']
    speed_z0, curv_z0, ang_z0 = spatial_diff(z0)         # [F'-1], [F'-2], [F'-2]
    pred_mag0                  = pred_mag[-1]             # [F'] — velocity at last step

    # Curvature-argmax per denoising step — tracks when the model "commits" to
    # a particular transition location as denoising progresses.
    # (curvature is defined on F'-2 positions, index shifted by +1 for centering)
    dissolve_by_step = np.argmax(curv_z.numpy(), axis=1) + 1  # [S]

    return {
        # ── 2-D trajectory matrices [S, *] ────────────────────────────────────
        "norm_z":       norm_z.numpy(),
        "speed_z":      speed_z.numpy(),
        "curvature_z":  curv_z.numpy(),
        "angular_z":    ang_z.numpy(),
        "pred_mag":     pred_mag.numpy(),
        "pred_curv":    pred_curv.numpy(),
        "step_size_z":  step_size_z.numpy(),
        # ── 1-D final-latent vectors [*] ──────────────────────────────────────
        "norm_z0":      norm_z0.numpy(),
        "speed_z0":     speed_z0.numpy(),
        "curvature_z0": curv_z0.numpy(),
        "angular_z0":   ang_z0.numpy(),
        "pred_mag0":    pred_mag0.numpy(),
        # ── Dissolve dynamics ─────────────────────────────────────────────────
        # dissolve_by_step[τ] = argmax curvature at denoising step τ.
        # Tracks when the model "commits" to a particular transition region.
        "dissolve_by_step": dissolve_by_step,
        "timesteps":        ts,
        "S": S, "F": F,
    }


# ── PLOTTING UTILITIES ─────────────────────────────────────────────────────────

def shade_cond(ax, data_len: int, F: int = F_PRIME, k: int = K_LAT,
               e: int = END_IDX, alpha: float = 0.13) -> None:
    """Shade start-clip (cyan) and end-clip (magenta) conditioning regions.

    The conditioning boundaries are in frame-space.  For features shorter
    than F (speed has F-1, curvature/angular have F-2), the boundaries shift
    by offset/2 because each column represents a window between frames.

    Args:
        ax:       Matplotlib axis.
        data_len: Number of data columns (F, F-1, or F-2).
        F:        Total latent frames (default F_PRIME=16).
        k:        Number of start-clip conditioning frames (default K_LAT=4).
        e:        First end-clip frame index (default END_IDX=12).
        alpha:    Shading opacity.
    """
    offset    = F - data_len
    shift     = offset / 2.0
    start_bnd = (k - 0.5) - shift
    end_bnd   = (e - 0.5) - shift
    ax.axvspan(-0.5,       start_bnd,      alpha=alpha, color="#00BCD4", zorder=0)
    ax.axvspan(end_bnd,    data_len - 0.5, alpha=alpha, color="#E91E63", zorder=0)
    if -0.5 < start_bnd < data_len - 0.5:
        ax.axvline(start_bnd, color="#00BCD4", lw=1.3, ls="--", alpha=0.9, zorder=1)
    if -0.5 < end_bnd   < data_len - 0.5:
        ax.axvline(end_bnd,   color="#E91E63", lw=1.3, ls="--", alpha=0.9, zorder=1)


def heatmap(ax, data: np.ndarray, title: str, cmap: str, diverging: bool = False,
            vmin=None, vmax=None, F: int = F_PRIME, k: int = K_LAT,
            e: int = END_IDX) -> None:
    """Draw a (S × F') heatmap with conditioning boundary markers.

    Row 0 = noisiest denoising step (τ=0).  Last row = clean (τ=39).

    Args:
        ax:       Matplotlib axis.
        data:     2-D array [S, P] (S denoising steps, P frame-positions).
        title:    Axis title.
        cmap:     Matplotlib colormap name.
        diverging: If True, fixes vmin=-1 / vmax=+1 (for angular cosine maps).
        vmin/vmax: Override colour scale limits.
        F, k, e:  Geometry constants — passed to shade_cond.
    """
    S_dim, P_dim = data.shape
    if diverging:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = vmin if vmin is not None else 0.0
        vmax = vmax if vmax is not None else float(np.percentile(data, 99))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest",
                   extent=[-0.5, P_dim - 0.5, S_dim - 0.5, -0.5])
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xlabel("frame  p", fontsize=7)
    ax.set_ylabel("step τ  (0=noisy)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xticks(range(P_dim))
    offset = F - P_dim
    shift  = offset / 2.0
    for bnd, col in [(k - 0.5, "#00BCD4"), (e - 0.5, "#E91E63")]:
        bc = bnd - shift
        if -0.5 < bc < P_dim - 0.5:
            ax.axvline(bc, color=col, lw=1.2, ls="--", alpha=0.9)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)


# ── VIDEO EMBEDDING ────────────────────────────────────────────────────────────

def video_html(path, width: int = 380, label: str = "", cls: str = "") -> str:
    """Return an HTML string embedding a video as base64 inline data URI."""
    if path is None or not pathlib.Path(path).exists():
        return f'<div style="width:{width}px;text-align:center;color:#aaa">[no video]</div>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    color    = CLASS_COLORS.get(cls, "#888")
    lbl_html = (f'<p style="font-size:11px;font-weight:bold;margin:2px 0;'
                f'color:{color}">{label}</p>') if label else ""
    return (
        f'<div style="display:inline-block;text-align:center;padding:6px;'
        f'border:2px solid {color};border-radius:6px;margin:4px">'
        f'{lbl_html}'
        f'<video width="{width}" controls>'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
        f'</video></div>'
    )


def video_grid(records: list, cols: int = 5, width: int = 290) -> HTML:
    """Render all generated videos in a labelled grid with class-colour borders."""
    cells = [
        f'<td style="padding:4px;vertical-align:top">'
        f'{video_html(r["video_path"], width=width, label=r["short_id"], cls=r["cls"])}</td>'
        for r in records
    ]
    rows = [
        "<tr>" + "".join(cells[i:i+cols]) + "</tr>"
        for i in range(0, len(cells), cols)
    ]
    legend = "".join(
        f'<span style="background:{v};color:white;padding:2px 7px;border-radius:3px;'
        f'font-size:11px;margin:2px">[C{k}] {CLASS_LABELS[k]}</span>'
        for k, v in CLASS_COLORS.items()
    )
    return HTML(
        f'<div style="margin-bottom:8px">{legend}</div>'
        f'<table style="border-collapse:collapse">{"".join(rows)}</table>'
    )


# ── GT ANNOTATIONS ─────────────────────────────────────────────────────────────

def load_gt_annotations() -> dict:
    """Load ground-truth transition annotations from the exp_022 dissolve CSV.

    Returns:
        gt_dict: {sample_id: float_seconds or None}
                 None = annotation missing or NaN.
    """
    if not GT_CSV.exists():
        print(f"[trajectory_utils] GT CSV not found: {GT_CSV}")
        return {}
    _df = pd.read_csv(GT_CSV)
    gt_dict: dict = {}
    for _, row in _df.iterrows():
        sid = row["sample_id"]
        try:
            gs = float(row["gt_s"])
            gt_dict[sid] = None if (gs != gs) else gs   # NaN check
        except (ValueError, TypeError):
            gt_dict[sid] = None
    return gt_dict


def gt_latent(sample_id: str, gt_dict: dict) -> "float | None":
    """Convert GT transition time (seconds) → latent frame (float).

    Returns None if no annotation exists for this sample.
    """
    gs = gt_dict.get(sample_id)
    if gs is None:
        return None
    return gs * VIDEO_FPS / LTX_TEMPORAL_SCALE


def add_gt_vline(ax, sample_id: str, gt_dict: dict,
                  color: str = "#4CAF50", lw: float = 2.0,
                  ls: str = "-", label: bool = True) -> None:
    """Draw a green GT vertical line on a 1-D frame axis.

    Only drawn if a GT annotation exists for this sample.

    Args:
        ax:        Matplotlib axis.
        sample_id: Sample identifier string.
        gt_dict:   Dict from load_gt_annotations().
        color:     Line colour (default green #4CAF50).
        lw:        Line width.
        ls:        Line style.
        label:     Whether to add a legend label.
    """
    gp = gt_latent(sample_id, gt_dict)
    if gp is not None:
        gs = gt_dict[sample_id]
        lbl = f"GT {gs:.2f}s" if label else None
        ax.axvline(gp, color=color, lw=lw, ls=ls, alpha=0.9,
                   label=lbl, zorder=6)


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_record(sample_dir: pathlib.Path) -> "dict | None":
    """Load one sample directory: extract features, free raw tensors.

    Memory strategy: z_t and v_pred are ~100 MB each in bfloat16.
    hidden_states are ~2.3 GB per sample.  We extract features here
    and discard the raw tensors.  Hidden states are loaded lazily in
    the Level 3 notebook on a per-sample basis.

    Returns:
        Record dict or None if no trajectory file found.
    """
    traj_files  = sorted(sample_dir.glob("*_trajectory_stage1.pt"))
    video_files = sorted(sample_dir.glob("*.mp4"))
    if not traj_files:
        return None

    traj_path = traj_files[0]
    data      = torch.load(traj_path, weights_only=False, map_location="cpu")
    sample_id = data.get("sample_id", sample_dir.name)
    cls       = sample_id.split("__")[0].replace("class", "").strip()
    short_id  = sample_id.replace(f"class{cls}__", "").replace("__", " → ")
    has_hs    = bool(data.get("hidden_states"))

    feats = compute_features(data)
    del data
    gc.collect()

    return {
        "sample_id": sample_id,
        "cls":       cls,
        "short_id":  f"[C{cls}] {short_id}",
        "cls_label": CLASS_LABELS.get(cls, "?"),
        "color":     CLASS_COLORS.get(cls, "#888"),
        "video_path": video_files[0] if video_files else None,
        "traj_path":  traj_path,
        "has_hs":     has_hs,
        "feats":      feats,
    }


def load_all_records(traj_dir: pathlib.Path = TRAJ_DIR) -> list:
    """Load all samples from traj_dir, sorted by class then name.

    Prints a summary table on completion.
    """
    sample_dirs = sorted(
        p for p in traj_dir.iterdir()
        if p.is_dir() and list(p.glob("*_trajectory_stage1.pt"))
    )
    records = [r for d in sample_dirs if (r := load_record(d)) is not None]
    records.sort(key=lambda r: (r["cls"], r["sample_id"]))
    print(f"✓ Loaded {len(records)} samples  (raw tensors freed)")
    print(f"{'sample_id':<50s} {'cls':<3s} {'dis_by_step_final':>18s}")
    print("─" * 73)
    for r in records:
        f = r["feats"]
        p_final = f["dissolve_by_step"][-1]
        print(f"  {r['sample_id']:<48s}  {r['cls']:<3s}  p_commit={p_final:>2d} "
              f"({p_final * LTX_TEMPORAL_SCALE / VIDEO_FPS:.2f}s)")
    return records


# ── ANALYSIS HELPERS ───────────────────────────────────────────────────────────

def commit_step(dissolve_by_step: np.ndarray, tolerance: int = 1) -> int:
    """Find the earliest denoising step at which the curvature-argmax stabilises.

    "Stable" = argmax stays within `tolerance` frames of its final value for
    all subsequent steps.  Lower = more decisive model.

    Args:
        dissolve_by_step: [S] array of argmax(curvature) per denoising step.
        tolerance:        Allowed deviation in frames.
    Returns:
        Earliest step τ where the signal becomes stable.
    """
    final = dissolve_by_step[-1]
    for i in range(len(dissolve_by_step)):
        if np.all(np.abs(dissolve_by_step[i:] - final) <= tolerance):
            return int(i)
    return len(dissolve_by_step) - 1


# ── HIDDEN-STATE HELPERS (Level 3) ─────────────────────────────────────────────

def unpack_hidden(h: torch.Tensor) -> torch.Tensor:
    """Reshape flat token sequence back to spatiotemporal layout.

    Args:
        h: [2, N, D]  where N = F' × H' × W' = 6144
    Returns:
        [2, F', H', W', D]
    """
    B, N, D = h.shape
    assert N == F_PRIME * H_PRIME * W_PRIME, f"Unexpected N={N} (expected {F_PRIME * H_PRIME * W_PRIME})"
    return h.reshape(B, F_PRIME, H_PRIME, W_PRIME, D)


def per_frame_norm(h_spatial: torch.Tensor, cond_idx: int = 1) -> np.ndarray:
    """Mean token-activation norm per latent frame.

    Args:
        h_spatial: [2, F', H', W', D]  (from unpack_hidden)
        cond_idx:  0=unconditional, 1=conditional (default).
    Returns:
        [F'] mean ‖h‖ over the H'×W'=384 spatial tokens.
    """
    h_c   = h_spatial[cond_idx]          # [F', H', W', D]
    norms = h_c.norm(dim=-1)             # [F', H', W']
    return norms.mean(dim=(1, 2)).numpy()  # [F']


def pca_frame_embeddings(h_spatial: torch.Tensor, cond_idx: int = 1,
                          n_comp: int = 2):
    """PCA over frame mean-embeddings.

    Each latent frame is represented by the mean of its H'×W'=384 token
    embeddings.  The 16 such vectors are projected to 2D via SVD.

    Args:
        h_spatial: [2, F', H', W', D]
        cond_idx:  Batch index to use (1=conditional).
        n_comp:    Number of PCA components to return.
    Returns:
        coords      : [F', n_comp]  projected coordinates
        explained   : [n_comp]      fraction of variance explained
    """
    h_c        = h_spatial[cond_idx]                         # [F', H', W', D]
    frame_vecs = h_c.mean(dim=(1, 2)).float().numpy()        # [F', D]
    frame_vecs = frame_vecs - frame_vecs.mean(axis=0)
    U, S, Vt   = np.linalg.svd(frame_vecs, full_matrices=False)
    coords     = U[:, :n_comp] * S[:n_comp]                  # [F', n_comp]
    total_var  = (S**2).sum()
    explained  = (S[:n_comp]**2) / (total_var + 1e-12)
    return coords, explained


def cosine_sim_matrix(h_spatial: torch.Tensor, cond_idx: int = 1) -> np.ndarray:
    """[F' × F'] cosine similarity matrix between frame mean-embeddings.

    Entry (p, q) = cos(h̄^L(p), h̄^L(q)).  A block-diagonal pattern
    (high similarity within each clip half, low across) indicates a hard
    semantic boundary between the two halves — the dissolve frame.

    Args:
        h_spatial: [2, F', H', W', D]
        cond_idx:  Batch index.
    Returns:
        [F', F'] numpy array, values in [-1, 1].
    """
    h_c = h_spatial[cond_idx]
    fv  = h_c.mean(dim=(1, 2)).float()    # [F', D]
    fv  = fv / fv.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (fv @ fv.T).numpy()
