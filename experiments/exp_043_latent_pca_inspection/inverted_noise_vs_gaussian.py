"""exp_043 — Inverted-noise (z1) vs matched-Gaussian deviation analysis.

THE QUESTION
------------
RF-inversion (exp_033 drop1) produces an "inverted noise" latent z1 that, run
back through the sampler, reconstructs a shadow-smoke transition video.  Unlike
a fresh generation (white Gaussian N(0,I)), z1 has the transition "baked into"
the noise.  This script characterizes EXACTLY what in z1 deviates from a matched
white-Gaussian null, and tests whether that deviation is the smoke signature —
localized in the FREE-MIDDLE latent frames.

Downstream (not implemented here): at production we start from Gaussian noise and
want to ADD this signature.  So the deviation is characterized in a way that is
(a) localized (frames / channels / spatial-freq), (b) checked for cross-clip
sharing, and (c) injectable.

GEOMETRY (verified from src + exp_033 run.py, NOT assumed)
----------------------------------------------------------
Packed latent is [1, N, 128], P=P_t=1.  _pack_latents permutes to token order
n = f*(H*W) + h*W + w  (frame-major, then row h, then col w); channel dim is the
raw 128.  So unpack [1,N,128] -> [F,H,W,128] is reshape(F,H,W,128) with the
CORRECT (H,W) for that clip.  F=16 latent frames (121 px frames).
  portrait  (render 704x512) -> H=22, W=16, N=5632 : smoke 0,2,3,6,8
  landscape (render 512x704) -> H=16, W=22, N=5632 : smoke 1,5,7,9
  square    (render 608x608) -> H=19, W=19, N=5776 : smoke 4
Anchors (clamped during inversion): latent frames 0-3 (start clip, k_lat=4) and
13,14,15 (end clip).  end_clip_index = n_lat - k_lat = 16 - 4 = 12; drop1 frees
frame 12.  => FREE-MIDDLE latent frames = 4..12 (9 frames), ANCHORS = 0-3,13-15.

NEVER group by token-count N (portrait & landscape share N=5632 with swapped
H,W).  Cross-sample spatial analysis ONLY within an orientation group.  Cross-
group comparisons use ONLY orientation-invariant scalars (radial power spectrum,
per-channel moments, per-frame stats, isotropic autocorrelation).

NULLS
-----
For every metric we also compute it on (1) white N(0,1) of identical [F,H,W,128]
shape and (2) a "variance-matched" white null scaled to z1's per-(frame,channel)
std, so structure can be isolated from raw variance.  LTX-2's sampling noise is
standard normal, so N(0,I) is the right reference.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import logging

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion.exp_utils import next_run_dir, TeeLogger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)

# ── Orientation groups (geometry-safe) ────────────────────────────────────────
# (H_lat, W_lat) for each clip, from render_HxW / 32.
ORIENTATION = {
    "shadow_smoke_0": ("portrait", 22, 16),
    "shadow_smoke_2": ("portrait", 22, 16),
    "shadow_smoke_3": ("portrait", 22, 16),
    "shadow_smoke_6": ("portrait", 22, 16),
    "shadow_smoke_8": ("portrait", 22, 16),
    "shadow_smoke_1": ("landscape", 16, 22),
    "shadow_smoke_5": ("landscape", 16, 22),
    "shadow_smoke_7": ("landscape", 16, 22),
    "shadow_smoke_9": ("landscape", 16, 22),
    "shadow_smoke_4": ("square", 19, 19),
}
GROUPS = {
    "portrait":  ["shadow_smoke_0", "shadow_smoke_2", "shadow_smoke_3", "shadow_smoke_6", "shadow_smoke_8"],
    "landscape": ["shadow_smoke_1", "shadow_smoke_5", "shadow_smoke_7", "shadow_smoke_9"],
    "square":    ["shadow_smoke_4"],
}
F_LAT = 16
FREE_MIDDLE = list(range(4, 13))      # 4..12 inclusive  (drop1 frees frame 12)
ANCHORS = [0, 1, 2, 3, 13, 14, 15]


# ── IO / geometry ─────────────────────────────────────────────────────────────
def load_unpacked(data_dir: pathlib.Path, sample: str, tensor: str,
                  H: int, W: int) -> np.ndarray:
    """Load packed [1,N,128] -> float32 [F,H,W,128] using the clip's (H,W)."""
    pt = data_dir / sample / f"{tensor}.pt"
    z = torch.load(pt, map_location="cpu", weights_only=False).float()  # [1,N,128]
    N = z.shape[1]
    assert N == F_LAT * H * W, f"{sample}: N={N} != F*H*W={F_LAT*H*W} (H={H},W={W})"
    return z.reshape(F_LAT, H, W, 128).numpy()


def gaussian_null(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).numpy()


def varmatched_null(z: np.ndarray, seed=0):
    """White noise scaled to z's per-(frame,channel) std (mean removed).
    Isolates STRUCTURE: same marginal variance per frame/channel as z, but iid."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*z.shape, generator=g, dtype=torch.float32).numpy()
    std = z.std(axis=(1, 2), keepdims=True)            # [F,1,1,128]
    return w * std


# ── Metrics ───────────────────────────────────────────────────────────────────
def per_channel_moments(z: np.ndarray, frames: list[int]) -> dict:
    """Standardize per channel (over the selected frames' spatial grid), then
    measure mean/var/skew/kurtosis.  Standardizing is correct here: we ask about
    distributional SHAPE, not scale."""
    x = z[frames].reshape(-1, 128)                     # [Fsel*H*W, 128]
    mu = x.mean(0)
    sd = x.std(0) + 1e-8
    xs = (x - mu) / sd
    skew = (xs ** 3).mean(0)
    kurt = (xs ** 4).mean(0) - 3.0                     # excess kurtosis
    return {
        "mean": mu, "std": sd,
        "skew": skew, "kurt": kurt,
        "skew_abs_mean": float(np.abs(skew).mean()),
        "kurt_mean": float(kurt.mean()),
        "kurt_abs_mean": float(np.abs(kurt).mean()),
    }


def radial_power_spectrum(field2d: np.ndarray, nbins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """2D power spectrum, radially averaged.  field2d: [H,W] mean-removed.
    Returns (radial_freq_bin_centers_normalized, mean_power_per_bin)."""
    H, W = field2d.shape
    F = np.fft.fft2(field2d - field2d.mean())
    P = (np.abs(F) ** 2) / (H * W)
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)                     # 0..~0.707
    rmax = r.max()
    edges = np.linspace(0, rmax + 1e-9, nbins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    idx = np.clip(idx, 0, nbins - 1)
    Pf = P.ravel()
    out = np.zeros(nbins)
    cnt = np.zeros(nbins)
    np.add.at(out, idx, Pf)
    np.add.at(cnt, idx, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, out / np.maximum(cnt, 1)


def mean_radial_spectrum(z: np.ndarray, frames: list[int], nbins: int = 16):
    """Average radial power spectrum across selected frames and all 128 channels.
    Orientation-invariant (radial). Returns (centers, mean_power, sem_power)."""
    specs = []
    for f in frames:
        for c in range(128):
            centers, p = radial_power_spectrum(z[f, :, :, c], nbins)
            specs.append(p)
    specs = np.array(specs)
    return centers, specs.mean(0), specs.std(0) / np.sqrt(len(specs))


def isotropic_autocorr(z: np.ndarray, frames: list[int], maxlag: int = 4):
    """Short-range spatial autocorrelation vs radial lag, averaged over frames &
    channels.  White noise -> ~0 at all nonzero lags.  Returns dict lag->corr.
    Mean-removed & variance-normalized per (frame,channel) so it measures
    STRUCTURE (correlation), not scale."""
    acc = {l: [] for l in range(1, maxlag + 1)}
    for f in frames:
        for c in range(128):
            a = z[f, :, :, c]
            a = a - a.mean()
            v = (a ** 2).mean() + 1e-12
            # 4-neighbour shifts at each lag (axis-aligned), normalized.
            for l in range(1, maxlag + 1):
                vals = []
                if a.shape[0] > l:
                    vals.append((a[l:, :] * a[:-l, :]).mean())
                if a.shape[1] > l:
                    vals.append((a[:, l:] * a[:, :-l]).mean())
                acc[l].append(np.mean(vals) / v)
    return {l: float(np.mean(acc[l])) for l in acc}


def temporal_corr(z: np.ndarray, frames: list[int]) -> float:
    """Mean adjacent-frame Pearson correlation across the selected frame band.
    White noise -> ~0.  z[f] flattened over (H,W,C), standardized per frame."""
    flat = z[frames].reshape(len(frames), -1)
    flat = (flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-8)
    n = flat.shape[1]
    cs = [float((flat[i] * flat[i + 1]).mean()) for i in range(len(frames) - 1)]
    return float(np.mean(cs)) if cs else 0.0


def per_frame_energy(z: np.ndarray) -> np.ndarray:
    """RMS per latent frame (orientation-invariant scalar per frame)."""
    return np.sqrt((z ** 2).mean(axis=(1, 2, 3)))      # [F]


def lowfreq_excess_per_frame(z: np.ndarray, null_white: np.ndarray,
                             lowfrac: float = 0.25) -> np.ndarray:
    """Fraction of spectral power in the lowest `lowfrac` radial band, per frame,
    averaged over channels.  Excess over white = low-frequency structure.
    Returns [F]."""
    Fn, H, W, C = z.shape
    out = np.zeros(Fn)
    for f in range(Fn):
        fr = []
        for c in range(C):
            centers, p = radial_power_spectrum(z[f, :, :, c], nbins=12)
            kcut = int(len(centers) * lowfrac)
            fr.append(p[:kcut].sum() / (p.sum() + 1e-12))
        out[f] = np.mean(fr)
    return out


def per_channel_energy(z: np.ndarray, frames: list[int]) -> np.ndarray:
    """Mean power per channel over selected frames. [128]."""
    return (z[frames] ** 2).mean(axis=(0, 1, 2))


# ── Plot helpers ──────────────────────────────────────────────────────────────
def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log.info("  chart -> %s", path.name)


def run_group(group: str, samples: list[str], data_dir: pathlib.Path,
              tensor: str, charts: pathlib.Path, seed: int) -> dict:
    log.info("=== orientation group: %s (%d clips, tensor=%s) ===",
             group, len(samples), tensor)
    _, H, W = ORIENTATION[samples[0]]
    zs = [load_unpacked(data_dir, s, tensor, H, W) for s in samples]
    log.info("  unpacked to [F=%d, H=%d, W=%d, C=128] per clip", F_LAT, H, W)

    # Nulls matched to the FIRST clip's shape (all share shape within group).
    z_white = gaussian_null(zs[0].shape, seed=seed)

    summary = {"group": group, "H": H, "W": W, "n_clips": len(samples),
               "samples": samples, "tensor": tensor,
               "free_middle": FREE_MIDDLE, "anchors": ANCHORS}

    # ── 1. Marginal Gaussianity (per channel, free-middle vs anchors vs null) ──
    mom_free = [per_channel_moments(z, FREE_MIDDLE) for z in zs]
    mom_anch = [per_channel_moments(z, ANCHORS) for z in zs]
    mom_white = per_channel_moments(z_white, FREE_MIDDLE)
    summary["marginal"] = {
        "free_skew_abs_mean": float(np.mean([m["skew_abs_mean"] for m in mom_free])),
        "free_kurt_mean":     float(np.mean([m["kurt_mean"] for m in mom_free])),
        "anch_skew_abs_mean": float(np.mean([m["skew_abs_mean"] for m in mom_anch])),
        "anch_kurt_mean":     float(np.mean([m["kurt_mean"] for m in mom_anch])),
        "white_skew_abs_mean": mom_white["skew_abs_mean"],
        "white_kurt_mean":     mom_white["kurt_mean"],
        "free_std_mean":  float(np.mean([m["std"].mean() for m in mom_free])),
        "anch_std_mean":  float(np.mean([m["std"].mean() for m in mom_anch])),
    }

    # Chart 1: histogram of standardized values (free-middle) z1 vs white null,
    # plus kurtosis-per-channel comparison.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x_all = np.concatenate([((z[FREE_MIDDLE].reshape(-1, 128) -
                              z[FREE_MIDDLE].reshape(-1, 128).mean(0)) /
                             (z[FREE_MIDDLE].reshape(-1, 128).std(0) + 1e-8)).ravel()
                            for z in zs])
    w_all = ((z_white[FREE_MIDDLE].reshape(-1, 128) -
              z_white[FREE_MIDDLE].reshape(-1, 128).mean(0)) /
             (z_white[FREE_MIDDLE].reshape(-1, 128).std(0) + 1e-8)).ravel()
    bins = np.linspace(-6, 6, 120)
    ax[0].hist(x_all, bins=bins, density=True, alpha=0.55, label="z1 free-middle", color="C3")
    ax[0].hist(w_all, bins=bins, density=True, alpha=0.45, label="white N(0,1)", color="C0")
    g = np.exp(-bins ** 2 / 2) / np.sqrt(2 * np.pi)
    ax[0].plot(bins, g, "k--", lw=1, label="ideal Gaussian")
    ax[0].set_yscale("log"); ax[0].set_title(f"[{group}] standardized marginal (free-middle)")
    ax[0].set_xlabel("standardized value"); ax[0].legend()
    kf = np.concatenate([m["kurt"] for m in mom_free])
    ka = np.concatenate([m["kurt"] for m in mom_anch])
    ax[1].hist(kf, bins=40, alpha=0.6, label="z1 free-middle", color="C3", density=True)
    ax[1].hist(ka, bins=40, alpha=0.5, label="z1 anchors", color="C1", density=True)
    ax[1].hist(mom_white["kurt"], bins=40, alpha=0.4, label="white null", color="C0", density=True)
    ax[1].axvline(0, color="k", ls="--", lw=1)
    ax[1].set_title("excess kurtosis per channel"); ax[1].set_xlabel("excess kurtosis"); ax[1].legend()
    savefig(fig, charts / f"01_marginal_{group}.png")

    # ── 2. Spatial structure: radial power spectrum + autocorrelation ──────────
    cen, p_free, sem_free = mean_radial_spectrum(np.mean(zs, 0), FREE_MIDDLE)
    cen, p_anch, _ = mean_radial_spectrum(np.mean(zs, 0), ANCHORS)
    cen, p_white, _ = mean_radial_spectrum(z_white, FREE_MIDDLE)
    z_vm = varmatched_null(zs[0], seed=seed)
    cen, p_vm, _ = mean_radial_spectrum(z_vm, FREE_MIDDLE)
    summary["spectrum"] = {
        "radial_centers": cen.tolist(),
        "z1_free": p_free.tolist(), "z1_anch": p_anch.tolist(),
        "white": p_white.tolist(), "varmatched": p_vm.tolist(),
        "lowfreq_ratio_z1_free": float(p_free[:3].sum() / p_free.sum()),
        "lowfreq_ratio_white":   float(p_white[:3].sum() / p_white.sum()),
    }
    ac_free = isotropic_autocorr(np.mean(zs, 0), FREE_MIDDLE)
    ac_anch = isotropic_autocorr(np.mean(zs, 0), ANCHORS)
    ac_white = isotropic_autocorr(z_white, FREE_MIDDLE)
    summary["autocorr"] = {"z1_free": ac_free, "z1_anch": ac_anch, "white": ac_white}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(cen, p_free, "-o", color="C3", label="z1 free-middle")
    ax[0].fill_between(cen, p_free - sem_free, p_free + sem_free, color="C3", alpha=0.2)
    ax[0].plot(cen, p_anch, "-s", color="C1", label="z1 anchors")
    ax[0].plot(cen, p_white, "-^", color="C0", label="white N(0,1) [flat]")
    ax[0].plot(cen, p_vm, ":", color="C2", label="variance-matched white")
    ax[0].set_xlabel("radial spatial frequency"); ax[0].set_ylabel("mean power")
    ax[0].set_title(f"[{group}] radial power spectrum"); ax[0].legend()
    lags = sorted(ac_free.keys())
    ax[1].plot(lags, [ac_free[l] for l in lags], "-o", color="C3", label="z1 free-middle")
    ax[1].plot(lags, [ac_anch[l] for l in lags], "-s", color="C1", label="z1 anchors")
    ax[1].plot(lags, [ac_white[l] for l in lags], "-^", color="C0", label="white null")
    ax[1].axhline(0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("spatial lag"); ax[1].set_ylabel("autocorrelation")
    ax[1].set_title("isotropic spatial autocorrelation"); ax[1].legend()
    savefig(fig, charts / f"02_spatial_{group}.png")

    # ── 3. Temporal structure ─────────────────────────────────────────────────
    tc_free = float(np.mean([temporal_corr(z, FREE_MIDDLE) for z in zs]))
    tc_anch = float(np.mean([temporal_corr(z, ANCHORS) for z in zs]))
    tc_white = temporal_corr(z_white, FREE_MIDDLE)
    # per-clip adjacent-frame corr across ALL frames for the chart
    full_tc = []
    for z in zs:
        flat = z.reshape(F_LAT, -1)
        flat = (flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-8)
        full_tc.append([float((flat[i] * flat[i + 1]).mean()) for i in range(F_LAT - 1)])
    full_tc = np.array(full_tc)
    summary["temporal"] = {"z1_free_adj_corr": tc_free, "z1_anch_adj_corr": tc_anch,
                           "white_adj_corr": tc_white}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(F_LAT - 1) + 0.5
    ax.plot(xs, full_tc.mean(0), "-o", color="C3", label="z1 adjacent-frame corr")
    ax.fill_between(xs, full_tc.mean(0) - full_tc.std(0), full_tc.mean(0) + full_tc.std(0),
                    color="C3", alpha=0.2)
    ax.axhline(tc_white, color="C0", ls="--", label="white null (~0)")
    for fr in [3.5, 12.5]:
        ax.axvline(fr, color="grey", ls=":", lw=1)
    ax.set_xlabel("between latent frames i,i+1"); ax.set_ylabel("Pearson corr")
    ax.set_title(f"[{group}] temporal coherence (grey = free-middle band edges)")
    ax.legend()
    savefig(fig, charts / f"03_temporal_{group}.png")

    # ── 4. Signature localization: energy & low-freq excess per frame/channel ──
    en = np.array([per_frame_energy(z) for z in zs])          # [n,F]
    lf = np.array([lowfreq_excess_per_frame(z, z_white) for z in zs])  # [n,F]
    lf_white = lowfreq_excess_per_frame(z_white, z_white)
    ch_en_free = np.array([per_channel_energy(z, FREE_MIDDLE) for z in zs]).mean(0)
    ch_en_anch = np.array([per_channel_energy(z, ANCHORS) for z in zs]).mean(0)
    summary["localization"] = {
        "per_frame_energy_mean": en.mean(0).tolist(),
        "per_frame_lowfreq_mean": lf.mean(0).tolist(),
        "white_lowfreq_mean": lf_white.tolist(),
        "free_minus_anchor_lowfreq": float(lf[:, FREE_MIDDLE].mean() - lf[:, ANCHORS].mean()),
        "top_channels_by_freeenergy": np.argsort(-ch_en_free)[:12].tolist(),
    }
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    fx = np.arange(F_LAT)
    ax[0].plot(fx, lf.mean(0), "-o", color="C3", label="z1 low-freq fraction")
    ax[0].fill_between(fx, lf.mean(0) - lf.std(0), lf.mean(0) + lf.std(0), color="C3", alpha=0.2)
    ax[0].plot(fx, lf_white, "-^", color="C0", label="white null")
    for f in FREE_MIDDLE:
        ax[0].axvspan(f - 0.5, f + 0.5, color="yellow", alpha=0.12)
    ax[0].set_xlabel("latent frame"); ax[0].set_ylabel("low-freq power fraction")
    ax[0].set_title(f"[{group}] low-freq structure per frame (yellow=free-middle)")
    ax[0].legend()
    order = np.argsort(-ch_en_free)
    ax[1].plot(ch_en_free[order], color="C3", label="free-middle energy (sorted)")
    ax[1].plot(ch_en_anch[order], color="C1", label="anchor energy (same order)")
    ax[1].axhline(1.0, color="C0", ls="--", label="white null (=1)")
    ax[1].set_xlabel("channel rank"); ax[1].set_ylabel("mean power")
    ax[1].set_title("per-channel energy (free-middle)"); ax[1].legend()
    savefig(fig, charts / f"04_localization_{group}.png")

    # ── 5. Cross-clip shared signature (WITHIN group, spatially aligned) ───────
    # Structured component = per-clip free-middle field minus its per-(frame,chan)
    # mean (removes DC), then averaged over the free band -> [H,W,128] per clip.
    # Cross-clip cosine of these structured maps tests for a SHARED smoke field.
    cross = None
    if len(samples) >= 2:
        structs = []
        for z in zs:
            band = z[FREE_MIDDLE]                          # [9,H,W,128]
            m = band.mean(axis=(1, 2), keepdims=True)      # per (frame,channel) DC
            structs.append((band - m).mean(0).ravel())     # [H*W*128]
        S = np.array(structs)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        C = S @ S.T
        iu = np.triu_indices(len(samples), 1)
        # null: cosine of random structured maps of same dim
        gnull = np.random.RandomState(seed).randn(len(samples), S.shape[1])
        gnull = gnull / np.linalg.norm(gnull, axis=1, keepdims=True)
        Cn = gnull @ gnull.T
        cross = {
            "free_struct_cos_mean": float(C[iu].mean()),
            "free_struct_cos_std": float(C[iu].std()),
            "null_cos_mean": float(Cn[iu].mean()),
            "null_cos_std": float(Cn[iu].std()),
            "matrix": C.tolist(),
        }
        # also: cross-clip cosine of the radial spectra (orientation-invariant)
        specs = []
        for z in zs:
            _, p, _ = mean_radial_spectrum(z, FREE_MIDDLE)
            specs.append(p / (np.linalg.norm(p) + 1e-12))
        Sp = np.array(specs)
        Csp = Sp @ Sp.T
        cross["radial_spectrum_cos_mean"] = float(Csp[iu].mean())
        summary["cross_clip"] = cross

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        im = ax[0].imshow(C, vmin=-0.3, vmax=1.0, cmap="RdBu_r")
        ax[0].set_title(f"[{group}] cross-clip cosine\n(free-middle structured map)")
        ax[0].set_xticks(range(len(samples))); ax[0].set_yticks(range(len(samples)))
        ax[0].set_xticklabels([s.split("_")[-1] for s in samples])
        ax[0].set_yticklabels([s.split("_")[-1] for s in samples])
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].bar(["z1 free\nstruct", "random\nnull"],
                  [C[iu].mean(), Cn[iu].mean()],
                  yerr=[C[iu].std(), Cn[iu].std()], color=["C3", "C0"], capsize=5)
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_title("shared free-middle structure vs null")
        savefig(fig, charts / f"05_crossclip_{group}.png")

    # ── QQ plot for a few high-kurtosis channels (free-middle) ─────────────────
    from scipy import stats
    kurt_mean = np.mean([m["kurt"] for m in mom_free], 0)
    pick = np.argsort(-np.abs(kurt_mean))[:3]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ch in zip(axs, pick):
        vals = np.concatenate([z[FREE_MIDDLE, :, :, ch].ravel() for z in zs])
        vals = (vals - vals.mean()) / (vals.std() + 1e-8)
        stats.probplot(vals, dist="norm", plot=ax)
        ax.set_title(f"ch {ch} (kurt={kurt_mean[ch]:+.2f})")
    fig.suptitle(f"[{group}] QQ-plots, highest-|kurtosis| free-middle channels")
    savefig(fig, charts / f"06_qq_{group}.png")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=pathlib.Path,
                    default=REPO_ROOT / "outputs/videos/exp_033_ltx2_rf_inv_drop1/run_0001")
    ap.add_argument("--tensor", type=str, default="z1", choices=["z1", "z0"])
    ap.add_argument("--groups", type=str, default="portrait,landscape,square")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str,
                    default="outputs/latent_pca/exp_043_inverted_noise_vs_gaussian")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_subdir
    run_id, run_dir = next_run_dir(out_dir)
    charts = run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-8s %(message)s",
                            datefmt="%H:%M:%S", stream=sys.stdout, force=True)
        log.info("[info] run_dir : %s", run_dir)
        log.info("[info] data_dir: %s  tensor=%s", args.data_dir, args.tensor)
        log.info("[info] free-middle latent frames = %s ; anchors = %s",
                 FREE_MIDDLE, ANCHORS)

        results = {"run_id": run_id, "data_dir": str(args.data_dir),
                   "tensor": args.tensor, "seed": args.seed, "groups": {}}
        for g in args.groups.split(","):
            g = g.strip()
            if g not in GROUPS:
                continue
            res = run_group(g, GROUPS[g], args.data_dir, args.tensor, charts, args.seed)
            results["groups"][g] = res

        with (run_dir / "summary.json").open("w") as f:
            json.dump(results, f, indent=2)
        log.info("[done] %s -> summary.json + %d charts",
                 run_id, len(list(charts.glob("*.png"))))


if __name__ == "__main__":
    main()
