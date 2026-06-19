"""Frequency / channel / cluster decomposition of predicted-z0 (z0_recon) smoke latents.

Research question: do the per-sample RECONSTRUCTED z0 latents (z0_recon) of the
shadow_smoke transitions share a common LOW-frequency component (the smoke
"darkening / billowing") that is separable from idiosyncratic per-clip content,
and could therefore be injected CROSS-CLIP?  Which frequency band and which
latent channels carry the shared smoke signal?

Builds on exp_043 manifold diagnostics (M1-M6) — adds the FREQUENCY / CHANNEL /
CLUSTER decomposition they did not do.

Data: outputs/videos/exp_033_ltx2_rf_inv_drop1/run_0001/<sample>/{z0_recon.pt,z0.pt}
Packed [1, N, 128] bf16.  Unpack via the LTX-2 patchifier convention
    b (f h w) (c p1 p2 p3) -> b c f (h p2) (w p3)   with P=1
so token order is frame-major, then latent-row h, then latent-col w; channel = c.
Reshape packed[1,N,128] -> [F, H, W, C] with reshape(F, H, W, C).

For render 704x512: H_lat = 704/32 = 22, W_lat = 512/32 = 16, tokens/frame = 352,
F_lat = (121-1)/8 + 1 = 16, N = 5632.

Anchor latent frames (25-frame clip conditioning): start clip -> latent frames
0..3, end clip -> latent frames 12..15 (clip_conditioning end_index=12, 4 latent
frames each).  FREE-MIDDLE = latent frames 4..11 (default; configurable).

CPU-only (numpy/scipy/sklearn).  No GPU needed.

Usage:
    python predicted_z0_spectra.py \
        --data_dir outputs/videos/exp_033_ltx2_rf_inv_drop1/run_0001 \
        --tensor z0_recon \
        --free_lo 4 --free_hi 11
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

SPATIAL_SCALE = 32
TEMPORAL_SCALE = 8
CHANNELS = 128


# ─── geometry ────────────────────────────────────────────────────────────────

def infer_geometry(N: int, render_h: int, render_w: int) -> tuple[int, int, int]:
    """Return (F_lat, H_lat, W_lat) consistent with N and render resolution."""
    H_lat = render_h // SPATIAL_SCALE
    W_lat = render_w // SPATIAL_SCALE
    tokens_per_frame = H_lat * W_lat
    assert N % tokens_per_frame == 0, (
        f"N={N} not divisible by tokens/frame={tokens_per_frame} "
        f"(render {render_h}x{render_w})")
    F_lat = N // tokens_per_frame
    return F_lat, H_lat, W_lat


def unpack(packed: np.ndarray, F: int, H: int, W: int) -> np.ndarray:
    """[1, N, 128] -> [F, H, W, C].  Token order is (f, h, w), channel last."""
    C = packed.shape[-1]
    return packed.reshape(1, F, H, W, C)[0]


# ─── spectra ─────────────────────────────────────────────────────────────────

def radial_average(power2d: np.ndarray, n_bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Radially average a centered 2-D power spectrum [H, W].

    Returns (radii [n_bins], mean_power [n_bins]).
    """
    H, W = power2d.shape
    cy, cx = H / 2.0, W / 2.0
    y, x = np.indices((H, W))
    r = np.sqrt(((y - cy) / (H / 2.0)) ** 2 + ((x - cx) / (W / 2.0)) ** 2)  # normalized radius 0..~1.41
    r_max = r.max()
    if n_bins is None:
        n_bins = int(min(H, W) // 2)
    bins = np.linspace(0, r_max, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    pw = power2d.ravel()
    sums = np.bincount(idx, weights=pw, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    counts = np.maximum(counts, 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, sums / counts


def spatial_power_spectrum(vol: np.ndarray) -> np.ndarray:
    """vol [F, H, W, C] -> centered 2-D power spectrum [H, W], averaged over F and C."""
    F, H, W, C = vol.shape
    # FFT over spatial H,W axes
    ft = np.fft.fft2(vol, axes=(1, 2))
    power = (np.abs(ft) ** 2)            # [F, H, W, C]
    power = power.mean(axis=(0, 3))      # [H, W]
    return np.fft.fftshift(power)


def freq_band_masks(H: int, W: int, lo_frac: float = 0.25, hi_frac: float = 0.5):
    """Return (low, mid, high) boolean masks on a centered [H, W] freq grid,
    by normalized radius thresholds lo_frac, hi_frac (relative to max radius)."""
    cy, cx = H / 2.0, W / 2.0
    y, x = np.indices((H, W))
    r = np.sqrt(((y - cy) / (H / 2.0)) ** 2 + ((x - cx) / (W / 2.0)) ** 2)
    r_max = r.max()
    rn = r / r_max
    low = rn <= lo_frac
    mid = (rn > lo_frac) & (rn <= hi_frac)
    high = rn > hi_frac
    return low, mid, high


def band_energy_fraction(power_centered: np.ndarray, lo_frac=0.25, hi_frac=0.5) -> dict:
    H, W = power_centered.shape
    low, mid, high = freq_band_masks(H, W, lo_frac, hi_frac)
    tot = power_centered.sum()
    return {
        "low": float(power_centered[low].sum() / tot),
        "mid": float(power_centered[mid].sum() / tot),
        "high": float(power_centered[high].sum() / tot),
    }


# ─── band-pass in freq domain ─────────────────────────────────────────────────

def band_filter(vol: np.ndarray, mask_centered: np.ndarray) -> np.ndarray:
    """Apply a centered spatial-freq boolean mask to vol [F,H,W,C], return filtered vol."""
    F, H, W, C = vol.shape
    ft = np.fft.fftshift(np.fft.fft2(vol, axes=(1, 2)), axes=(1, 2))
    ft = ft * mask_centered[None, :, :, None]
    out = np.fft.ifft2(np.fft.ifftshift(ft, axes=(1, 2)), axes=(1, 2))
    return out.real


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str,
                    default="outputs/videos/exp_033_ltx2_rf_inv_drop1/run_0001")
    ap.add_argument("--tensor", type=str, default="z0_recon",
                    choices=["z0_recon", "z0"])
    ap.add_argument("--free_lo", type=int, default=4, help="first free-middle latent frame (incl)")
    ap.add_argument("--free_hi", type=int, default=11, help="last free-middle latent frame (incl)")
    ap.add_argument("--out_dir", type=str,
                    default="outputs/latent_pca/exp_043_predicted_z0_spectra")
    ap.add_argument("--lo_frac", type=float, default=0.25)
    ap.add_argument("--hi_frac", type=float, default=0.5)
    ap.add_argument("--gaussian_null", choices=["off", "matched", "white"], default="off",
                    help="replace real latents with Gaussian noise. 'matched' = per-channel "
                         "mean/std of the real sample (same marginal stats, no spatial structure); "
                         "'white' = standard normal. Tests whether cross-sample low-freq "
                         "correlation exceeds chance.")
    ap.add_argument("--null_seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    out_root = (REPO_ROOT / args.out_dir).resolve()
    # sequential run dir
    out_root.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("_")[1]) for p in out_root.glob("run_*") if p.name.split("_")[1].isdigit()]
    run_id = max(existing, default=0) + 1
    out = out_root / f"run_{run_id:04d}"
    (out / "charts").mkdir(parents=True, exist_ok=True)
    charts = out / "charts"

    sample_ids = sorted([p.name for p in data_dir.glob("shadow_smoke_*") if p.is_dir()],
                        key=lambda s: int(s.split("_")[-1]))
    print(f"[info] data_dir={data_dir}")
    print(f"[info] tensor={args.tensor}  free-middle latent frames {args.free_lo}..{args.free_hi}")
    print(f"[info] samples: {sample_ids}")

    import yaml
    summary = {"tensor": args.tensor, "free_lo": args.free_lo, "free_hi": args.free_hi,
               "lo_frac": args.lo_frac, "hi_frac": args.hi_frac, "samples": {}}

    vols = {}          # sample_id -> [F,H,W,C] float32
    geoms = {}
    for sid in sample_ids:
        p = data_dir / sid / f"{args.tensor}.pt"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        meta_p = data_dir / sid / "inv_meta.yaml"
        with meta_p.open() as f:
            meta = yaml.safe_load(f)
        rh, rw = meta["render_HxW"]
        t = torch.load(p, map_location="cpu", weights_only=False)
        arr = t.squeeze(0).float().numpy()  # [N, 128]
        N = arr.shape[0]
        F, H, W = infer_geometry(N, rh, rw)
        vol = unpack(arr[None], F, H, W)     # [F,H,W,C]
        vols[sid] = vol
        geoms[sid] = (F, H, W, rh, rw, N)
        print(f"  {sid}: N={N} render={rh}x{rw} -> F={F} H={H} W={W} C={vol.shape[-1]}")
        summary["samples"][sid] = {"N": int(N), "render_HxW": [int(rh), int(rw)],
                                    "F": int(F), "H": int(H), "W": int(W)}

    # Canonicalize orientation: transpose H<->W for landscape volumes so that
    # portrait and landscape renders of the SAME content share one grid.
    # (For radial spectra / channel / cluster analyses, a global spatial
    #  transpose is content-preserving; only the true odd-aspect square differs.)
    from collections import Counter
    raw_geom_counts = Counter((g[0], g[1], g[2]) for g in geoms.values())
    print(f"[info] distinct raw (F,H,W) geometries: {dict(raw_geom_counts)}")
    canon_counts = Counter((g[0], max(g[1], g[2]), min(g[1], g[2])) for g in geoms.values())
    canon_geom = canon_counts.most_common(1)[0][0]   # F, max(H,W), min(H,W)
    Fc, Hc, Wc = canon_geom
    transposed = []
    for sid in list(vols.keys()):
        F, H, W = geoms[sid][0], geoms[sid][1], geoms[sid][2]
        if (F, H, W) == (Fc, Hc, Wc):
            continue
        if (F, W, H) == (Fc, Hc, Wc):
            vols[sid] = np.transpose(vols[sid], (0, 2, 1, 3))  # swap H,W
            geoms[sid] = (Fc, Hc, Wc) + geoms[sid][3:]
            transposed.append(sid)
    common_ids = [sid for sid in vols if (geoms[sid][0], geoms[sid][1], geoms[sid][2]) == (Fc, Hc, Wc)]
    excluded = [sid for sid in vols if sid not in common_ids]
    print(f"[info] canonical geometry F={Fc} H={Hc} W={Wc}; transposed-to-match={transposed}")
    # ── Gaussian null: replace real latents with noise of matched marginals ───
    # Tests whether the cross-sample spatial-frequency correlation structure is
    # real or a chance/marginal-statistics artifact. Matched: each replaced vol
    # keeps the real sample's per-channel mean & std but has i.i.d. spatial noise
    # (so any cross-sample SPATIAL correlation must vanish to ~null).
    if args.gaussian_null != "off":
        rng = np.random.default_rng(args.null_seed)
        for sid in common_ids:
            real = vols[sid]                       # [F,H,W,C]
            if args.gaussian_null == "matched":
                mu = real.mean(axis=(0, 1, 2), keepdims=True)   # [1,1,1,C]
                sd = real.std(axis=(0, 1, 2), keepdims=True)
                vols[sid] = (rng.standard_normal(real.shape) * sd + mu).astype(np.float32)
            else:  # white
                vols[sid] = rng.standard_normal(real.shape).astype(np.float32)
        print(f"[info] GAUSSIAN NULL active: mode={args.gaussian_null} seed={args.null_seed} "
              f"(real latents replaced; cross-sample structure should collapse to ~null)")
    summary["gaussian_null"] = args.gaussian_null
    summary["null_seed"] = args.null_seed

    print(f"[info] cross-sample ids (n={len(common_ids)}): {common_ids}; excluded={excluded}")
    summary["common_geometry"] = {"F": int(Fc), "H": int(Hc), "W": int(Wc)}
    summary["cross_sample_ids"] = common_ids
    summary["excluded_ids"] = excluded
    summary["transposed_to_match"] = transposed

    flo, fhi = args.free_lo, args.free_hi
    anchor_frames = list(range(0, flo)) + list(range(fhi + 1, Fc))
    free_frames = list(range(flo, fhi + 1))
    summary["anchor_frames"] = anchor_frames
    summary["free_frames"] = free_frames

    # ── (1) per-sample radial power spectrum over free-middle ─────────────────
    print("[1] radial power spectra")
    radial_curves = {}
    band_fracs = {}
    for sid in common_ids:
        vol = vols[sid]
        free = vol[free_frames]                 # [nf, H, W, C]
        pw = spatial_power_spectrum(free)        # [H, W] centered
        radii, prof = radial_average(pw)
        radial_curves[sid] = (radii, prof)
        band_fracs[sid] = band_energy_fraction(pw, args.lo_frac, args.hi_frac)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab10
    for i, sid in enumerate(common_ids):
        radii, prof = radial_curves[sid]
        ax.loglog(radii[1:], prof[1:], "-", color=cmap(i % 10), alpha=0.8,
                  label=sid.replace("shadow_smoke_", "s"))
    ax.set_xlabel("normalized spatial frequency (radius)")
    ax.set_ylabel("radially-averaged power (free-middle, mean over F,C)")
    ax.set_title(f"[1] Per-sample 2D spatial power spectrum ({args.tensor}, free frames {flo}-{fhi})")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(charts / "S1_radial_power_spectra.png", dpi=130)
    plt.close(fig)
    summary["S1_band_energy_fraction"] = band_fracs
    lows = np.array([band_fracs[s]["low"] for s in common_ids])
    print(f"    low-freq energy fraction: mean={lows.mean():.3f} range=[{lows.min():.3f},{lows.max():.3f}]")

    # ── (2) shared (mean) vs idiosyncratic (residual) energy per band ─────────
    print("[2] shared vs idiosyncratic per band")
    # per-sample free-middle MEAN over free frames -> [H, W, C]
    M = np.stack([vols[s][free_frames].mean(axis=0) for s in common_ids], axis=0)  # [G,H,W,C]
    mean_lat = M.mean(axis=0)                  # [H,W,C] cross-sample mean
    resid = M - mean_lat[None]                 # [G,H,W,C]
    low, mid, high = freq_band_masks(Hc, Wc, args.lo_frac, args.hi_frac)
    bands = {"low": low, "mid": mid, "high": high}

    def band_energy_of(field):  # field [...,H,W,C] -> scalar energy in band per band
        # field could be [H,W,C] or [G,H,W,C]
        if field.ndim == 3:
            ft = np.fft.fftshift(np.fft.fft2(field, axes=(0, 1)), axes=(0, 1))
            p = (np.abs(ft) ** 2)              # [H,W,C]
            return {b: float(p[m].sum()) for b, m in bands.items()}
        else:
            ft = np.fft.fftshift(np.fft.fft2(field, axes=(1, 2)), axes=(1, 2))
            p = (np.abs(ft) ** 2)              # [G,H,W,C]
            # mean over samples of per-sample band energy
            return {b: float(p[:, m, :].sum() / field.shape[0]) for b, m in bands.items()}

    e_mean = band_energy_of(mean_lat)          # energy of the shared mean per band
    e_resid = band_energy_of(resid)            # mean per-sample residual energy per band
    shared_share = {b: e_mean[b] / (e_mean[b] + e_resid[b] + 1e-12) for b in bands}
    summary["S2_band_energy"] = {"mean_field_energy": e_mean,
                                  "mean_residual_energy": e_resid,
                                  "shared_share": shared_share}
    print(f"    shared-share (||mean||^2 / (||mean||^2+mean||resid||^2)) per band:")
    for b in ["low", "mid", "high"]:
        print(f"      {b}: {shared_share[b]:.3f}  (mean E={e_mean[b]:.3e}, resid E={e_resid[b]:.3e})")

    fig, ax = plt.subplots(figsize=(7, 5))
    xb = np.arange(3); bw = 0.35
    ax.bar(xb - bw/2, [e_mean[b] for b in ["low","mid","high"]], bw, label="shared mean energy", color="tab:green")
    ax.bar(xb + bw/2, [e_resid[b] for b in ["low","mid","high"]], bw, label="mean residual energy", color="tab:red")
    ax.set_yscale("log"); ax.set_xticks(xb); ax.set_xticklabels(["low","mid","high"])
    ax.set_ylabel("band energy"); ax.set_title("[2] Shared (cross-sample mean) vs idiosyncratic (residual) energy")
    for i, b in enumerate(["low","mid","high"]):
        ax.text(i, max(e_mean[b], e_resid[b]) * 1.3, f"share={shared_share[b]:.2f}", ha="center", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(charts / "S2_shared_vs_residual_bands.png", dpi=130)
    plt.close(fig)

    # ── (3) PCA across samples (each sample = one flattened free-middle-mean vector) ──
    print("[3] cross-sample PCA")
    Xpca = M.reshape(len(common_ids), -1)       # [G, H*W*C]
    Xc = Xpca - Xpca.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    # cosine of each centered sample to PC1
    pc1 = Vt[0]
    cos_to_pc1 = [(Xc[i] @ pc1) / (np.linalg.norm(Xc[i]) * np.linalg.norm(pc1) + 1e-12)
                  for i in range(len(common_ids))]
    # Also: how much of the RAW (uncentered) signal is the cross-sample mean vs PC spread
    mean_norm = np.linalg.norm(Xpca.mean(axis=0))
    resid_norms = np.linalg.norm(Xc, axis=1)
    summary["S3_pca"] = {
        "evr_top5": evr[:5].tolist(),
        "cos_to_pc1": {s: float(c) for s, c in zip(common_ids, cos_to_pc1)},
        "cross_sample_mean_norm": float(mean_norm),
        "mean_residual_norm": float(resid_norms.mean()),
        "mean_norm_over_resid": float(mean_norm / (resid_norms.mean() + 1e-12)),
    }
    print(f"    EVR top5: {np.round(evr[:5], 3).tolist()}")
    print(f"    ||cross-sample mean|| = {mean_norm:.1f};  mean||residual|| = {resid_norms.mean():.1f};  ratio = {mean_norm/(resid_norms.mean()+1e-12):.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(np.arange(1, len(evr)+1), evr, "o-")
    ax[0].set_xlabel("PC"); ax[0].set_ylabel("EVR"); ax[0].set_title("[3] cross-sample PCA scree")
    ax[0].grid(alpha=0.3)
    ax[1].bar(range(len(common_ids)), cos_to_pc1)
    ax[1].set_xticks(range(len(common_ids))); ax[1].set_xticklabels([s.replace("shadow_smoke_","s") for s in common_ids], rotation=45)
    ax[1].set_ylabel("cos(sample, PC1)"); ax[1].set_title("per-sample cosine to PC1 (centered)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(charts / "S3_cross_sample_pca.png", dpi=130)
    plt.close(fig)

    # ── (4) band split: low-pass vs high-pass, cross-sample similarity matrices ──
    print("[4] per-band cross-sample similarity")
    band_sim = {}
    for bname, bmask in [("low", low), ("high", high)]:
        # band-filter each sample's free-middle-mean field, then flatten
        feats = []
        for s in common_ids:
            field = vols[s][free_frames].mean(axis=0)   # [H,W,C]
            f4 = field[None]                             # [1,H,W,C]
            filt = band_filter(f4, bmask)[0]             # [H,W,C]
            feats.append(filt.ravel())
        F_ = np.stack(feats, 0)                          # [G, D] (D = H*W*C)
        # cosine (uncentered) — dominated by shared per-channel DC at low band
        Fn = F_ / (np.linalg.norm(F_, axis=1, keepdims=True) + 1e-12)
        cos = Fn @ Fn.T
        # correlation (global mean removed)
        Fc_ = F_ - F_.mean(axis=1, keepdims=True)
        Fcn = Fc_ / (np.linalg.norm(Fc_, axis=1, keepdims=True) + 1e-12)
        corr = Fcn @ Fcn.T
        # DC-removed: subtract each channel's spatial mean (kills the constant
        # per-channel offset that survives low-pass), isolating SPATIAL structure.
        feats_nodc = []
        for s in common_ids:
            field = vols[s][free_frames].mean(axis=0)    # [H,W,C]
            filt = band_filter(field[None], bmask)[0]    # [H,W,C]
            filt = filt - filt.mean(axis=(0, 1), keepdims=True)  # per-channel spatial DC removed
            feats_nodc.append(filt.ravel())
        Fnodc = np.stack(feats_nodc, 0)
        Fnodc = Fnodc / (np.linalg.norm(Fnodc, axis=1, keepdims=True) + 1e-12)
        cos_nodc = Fnodc @ Fnodc.T
        G = len(common_ids)
        offmask = ~np.eye(G, dtype=bool)
        band_sim[bname] = {
            "cos_offdiag_mean": float(cos[offmask].mean()),
            "cos_offdiag_std": float(cos[offmask].std()),
            "corr_offdiag_mean": float(corr[offmask].mean()),
            "corr_offdiag_std": float(corr[offmask].std()),
            "cos_nodc_offdiag_mean": float(cos_nodc[offmask].mean()),
            "cos_nodc_offdiag_std": float(cos_nodc[offmask].std()),
            "cos_matrix": cos.tolist(),
            "cos_nodc_matrix": cos_nodc.tolist(),
        }
        print(f"    {bname}-band: cos={cos[offmask].mean():+.3f}±{cos[offmask].std():.3f}  "
              f"cos_noDC={cos_nodc[offmask].mean():+.3f}±{cos_nodc[offmask].std():.3f}  "
              f"corr={corr[offmask].mean():+.3f}")
    summary["S4_band_similarity"] = band_sim

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    panels = [("low", "cos_matrix", "cos"), ("high", "cos_matrix", "cos"),
              ("low", "cos_nodc_matrix", "cos_noDC"), ("high", "cos_nodc_matrix", "cos_noDC")]
    for ax_, (bname, key, lab) in zip(axes.ravel(), panels):
        M_ = np.array(band_sim[bname][key])
        im = ax_.imshow(M_, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        plt.colorbar(im, ax=ax_)
        offm = band_sim[bname][("cos_offdiag_mean" if lab == "cos" else "cos_nodc_offdiag_mean")]
        ax_.set_title(f"{bname}-band {lab}  off-diag={offm:+.3f}")
        ax_.set_xticks(range(len(common_ids))); ax_.set_yticks(range(len(common_ids)))
        ax_.set_xticklabels([s.replace("shadow_smoke_","s") for s in common_ids], rotation=45, fontsize=7)
        ax_.set_yticklabels([s.replace("shadow_smoke_","s") for s in common_ids], fontsize=7)
    fig.suptitle("[4] Cross-sample similarity per band — raw cos (DC-contaminated) vs DC-removed (spatial)")
    fig.tight_layout(); fig.savefig(charts / "S4_band_similarity.png", dpi=130)
    plt.close(fig)

    # ── (5) per-channel cross-sample correlation (free-middle) ────────────────
    print("[5] per-channel cross-sample correlation")
    # For each channel c, build per-sample free-middle spatial map (mean over free frames),
    # flatten spatial -> vector length H*W; cross-sample mean pairwise correlation.
    G = len(common_ids)
    chan_corr = np.zeros(CHANNELS)
    chan_mean_energy = np.zeros(CHANNELS)
    offmask = ~np.eye(G, dtype=bool)
    # Precompute per-channel per-sample flattened spatial maps
    maps = np.stack([vols[s][free_frames].mean(axis=0) for s in common_ids], axis=0)  # [G,H,W,C]
    for c in range(CHANNELS):
        X = maps[:, :, :, c].reshape(G, -1)       # [G, H*W]
        Xc_ = X - X.mean(axis=1, keepdims=True)
        Xn = Xc_ / (np.linalg.norm(Xc_, axis=1, keepdims=True) + 1e-12)
        corr = Xn @ Xn.T
        chan_corr[c] = corr[offmask].mean()
        chan_mean_energy[c] = float((X ** 2).mean())
    order = np.argsort(-chan_corr)
    topk = 20
    top_channels = [(int(c), float(chan_corr[c]), float(chan_mean_energy[c])) for c in order[:topk]]
    summary["S5_channel_corr"] = {
        "per_channel_corr": chan_corr.tolist(),
        "per_channel_mean_energy": chan_mean_energy.tolist(),
        "top20_channels": top_channels,
        "corr_mean_all": float(chan_corr.mean()),
        "corr_max": float(chan_corr.max()),
    }
    print(f"    channel cross-sample corr: mean={chan_corr.mean():+.3f} max={chan_corr.max():+.3f}")
    print(f"    top-10 channels (idx, corr, energy):")
    for c, cc, en in top_channels[:10]:
        print(f"      ch{c:3d}  corr={cc:+.3f}  energy={en:.3e}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(np.sort(chan_corr)[::-1], "-o", ms=3)
    ax[0].set_xlabel("channel rank"); ax[0].set_ylabel("cross-sample spatial corr (off-diag mean)")
    ax[0].set_title("[5] per-channel cross-sample correlation (sorted)"); ax[0].grid(alpha=0.3)
    ax[0].axhline(0, color="k", lw=0.5)
    sc = ax[1].scatter(chan_mean_energy, chan_corr, s=12)
    ax[1].set_xscale("log"); ax[1].set_xlabel("channel mean energy (free-middle)")
    ax[1].set_ylabel("cross-sample corr"); ax[1].set_title("corr vs energy per channel"); ax[1].grid(alpha=0.3)
    for c, cc, en in top_channels[:8]:
        ax[1].annotate(str(c), (en, cc), fontsize=7)
    fig.tight_layout(); fig.savefig(charts / "S5_channel_correlation.png", dpi=130)
    plt.close(fig)

    # ── (6) KMeans on per-(sample,frame) latent vectors ──────────────────────
    print("[6] KMeans on per-(sample,frame) vectors")
    from sklearn.cluster import KMeans
    rows = []
    labels_meta = []   # (sample_id, frame_idx, is_free)
    for s in common_ids:
        vol = vols[s]
        for fidx in range(Fc):
            rows.append(vol[fidx].ravel())
            labels_meta.append((s, fidx, fidx in free_frames))
    Xkm = np.stack(rows, 0)
    Xkm = (Xkm - Xkm.mean(axis=0)) / (Xkm.std(axis=0) + 1e-8)
    km_summary = {}
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xkm)
        lab = km.labels_
        comp = {}
        for cl in range(k):
            members = [labels_meta[i] for i in range(len(lab)) if lab[i] == cl]
            n_free = sum(1 for m in members if m[2])
            n_anchor = len(members) - n_free
            frame_hist = {}
            for m in members:
                frame_hist[m[1]] = frame_hist.get(m[1], 0) + 1
            comp[f"cluster_{cl}"] = {
                "size": len(members),
                "n_free": n_free, "n_anchor": n_anchor,
                "free_purity": float(n_free / max(len(members), 1)),
                "frame_hist": frame_hist,
            }
        km_summary[f"k={k}"] = comp
        print(f"    k={k}:")
        for cl in range(k):
            c = comp[f"cluster_{cl}"]
            print(f"      cluster {cl}: size={c['size']} free={c['n_free']} anchor={c['n_anchor']} (free purity {c['free_purity']:.2f})")
    summary["S6_kmeans"] = km_summary

    # Visualize: frame index vs cluster (k=2) as a sample x frame grid
    km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xkm)
    lab2 = km2.labels_.reshape(len(common_ids), Fc)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(lab2, aspect="auto", cmap="coolwarm", origin="lower")
    ax.set_xlabel("latent frame"); ax.set_ylabel("sample")
    ax.set_yticks(range(len(common_ids))); ax.set_yticklabels([s.replace("shadow_smoke_","s") for s in common_ids], fontsize=7)
    for f in [flo - 0.5, fhi + 0.5]:
        ax.axvline(f, color="k", lw=1.5, ls="--")
    ax.set_title(f"[6] KMeans(k=2) cluster per (sample, frame); dashed = free-middle [{flo},{fhi}]")
    plt.colorbar(im, ax=ax, ticks=[0, 1])
    fig.tight_layout(); fig.savefig(charts / "S6_kmeans_frame_grid.png", dpi=130)
    plt.close(fig)

    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] run_{run_id:04d} -> {out}")
    print(f"       charts: {charts}")
    print(f"       summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
