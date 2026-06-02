"""Quantify transition-hardness for shadow_smoke clips and correlate with
exp_030/032/033 recon results.

For each clip we extract four anchor frames:

    f0   = frame 0                    (first of start sub-clip)
    f24  = frame K-1 (=24)            (LAST of start sub-clip)
    f96  = frame N-K (=96)            (FIRST of end sub-clip)
    f120 = frame N-1 (=120)           (last of end sub-clip)

We compute two distance pairs:

    GAP  = (f24, f96)   — the discontinuity the model has to bridge
                          across the 71 inbetween frames.
    SPAN = (f0,  f120)  — the total transformation from clip start to
                          clip end.

Plus within-clip motion stats (mean optical-flow magnitude) for the start
sub-clip (0..24), end sub-clip (96..120), and the middle (25..95).

Plus a semantic-distance metric using CLIP ViT-B/32 image embeddings on
the GAP pair.

Saved to outputs/analysis/transition_hardness/run_NNNN/{metrics.csv,
correlations.csv, report.md}. Pearson + Spearman correlations against
exp_033 recon PSNR (and exp_030, exp_032 for reference).

CPU-only: no torch.cuda. Plenty of RAM. Runs in a couple of minutes.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from dataclasses import dataclass, asdict

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.exp_utils import next_run_dir  # noqa: E402

# ── Inputs ────────────────────────────────────────────────────────────────────
CLIPS = [
    ("shadow_smoke_1", "data/processed/transitions/shadow_smoke/shadow_smoke_1.mp4"),
    ("shadow_smoke_2", "data/processed/transitions/shadow_smoke/shadow_smoke_2.mp4"),
    ("shadow_smoke_3", "data/processed/transitions/shadow_smoke/shadow_smoke_3.mp4"),
    ("shadow_smoke_4", "data/processed/transitions/shadow_smoke/shadow_smoke_4.mp4"),
    ("shadow_smoke_5", "data/processed/transitions/shadow_smoke/shadow_smoke_5.mp4"),
    ("shadow_smoke_6", "data/processed/transitions/shadow_smoke/shadow_smoke_6.mp4"),
    ("shadow_smoke_7", "data/processed/transitions/shadow_smoke/shadow_smoke_7.mp4"),
    ("shadow_smoke_8", "data/processed/transitions/shadow_smoke/shadow_smoke_8.mp4"),
    ("shadow_smoke_9", "data/processed/transitions/shadow_smoke/shadow_smoke_9.mp4"),
    ("shadow_smoke_0", "data/processed/transitions/shadow_smoke/shadow_smoke.mp4"),
]
N_FRAMES = 121
K = 25  # frames per sub-clip — matches exp_030/032/033 config
RESIZE_MAX_DIM = 512  # resize each frame so max(H,W)=512 before any metric
                     # — matches exp_030/032/033's max_area=393216 (≈512×768).
                     # Without this, Farnebäck on 1660×1244 frames is too slow
                     # on the 2-core CPU budget and pixel-distance numbers
                     # would be incomparable across landscape/portrait clips.

# Exp recon PSNR (from each run's run.log) — kept in code for direct correlation.
EXP_PSNR = {
    "shadow_smoke_1": {"exp030": 17.45, "exp032": 38.45, "exp033": 16.38},
    "shadow_smoke_2": {"exp030": 18.25, "exp032": 34.73, "exp033": 26.36},
    "shadow_smoke_3": {"exp030": 19.23, "exp032": 44.75, "exp033": 28.32},
    "shadow_smoke_4": {"exp030": 30.73, "exp032": 39.76, "exp033": 33.12},
    "shadow_smoke_5": {"exp030": 16.51, "exp032": 40.92, "exp033": 16.35},
    "shadow_smoke_6": {"exp030": 17.09, "exp032": 36.11, "exp033": 16.48},
    "shadow_smoke_7": {"exp030": 14.64, "exp032": 40.83, "exp033": 24.97},
    "shadow_smoke_8": {"exp030": 20.52, "exp032": 45.25, "exp033": 29.58},
    "shadow_smoke_9": {"exp030": 15.18, "exp032": 42.62, "exp033": 18.65},
    "shadow_smoke_0": {"exp030": 26.52, "exp032": 44.76, "exp033": 29.32},
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_clip_frames(path: pathlib.Path, n: int) -> list[np.ndarray]:
    """Returns `n` BGR uint8 frames from `path`, resized so max(H,W)=RESIZE_MAX_DIM,
    edge-padded if the clip is short."""
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        s = RESIZE_MAX_DIM / max(h, w)
        if s != 1.0:
            frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))),
                               interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"empty clip {path}")
    if len(frames) < n:
        frames = frames + [frames[-1]] * (n - len(frames))
    return frames[:n]


def to_rgb_float(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_rgb_uint8(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def psnr_pair(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a, b = to_rgb_float(a_bgr), to_rgb_float(b_bgr)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(psnr_fn(a, b, data_range=1.0))


def ssim_pair(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a, b = to_rgb_float(a_bgr), to_rgb_float(b_bgr)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(ssim_fn(a, b, channel_axis=-1, data_range=1.0))


def lpips_pair(model, a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a, b = to_rgb_float(a_bgr), to_rgb_float(b_bgr)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    # LPIPS expects [-1,1], NCHW
    ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        d = model(ta, tb).item()
    return float(d)


def histogram_chi2(a_bgr: np.ndarray, b_bgr: np.ndarray, bins: int = 32) -> float:
    """Per-channel chi-squared distance between RGB histograms."""
    a = to_rgb_uint8(a_bgr)
    b = to_rgb_uint8(b_bgr)
    dists = []
    for c in range(3):
        ha, _ = np.histogram(a[..., c], bins=bins, range=(0, 256), density=True)
        hb, _ = np.histogram(b[..., c], bins=bins, range=(0, 256), density=True)
        denom = ha + hb + 1e-10
        dists.append(0.5 * np.sum((ha - hb) ** 2 / denom))
    return float(np.mean(dists))


def mean_rgb_l1(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a = to_rgb_float(a_bgr).mean(axis=(0, 1))
    b = to_rgb_float(b_bgr).mean(axis=(0, 1))
    return float(np.abs(a - b).mean())


def mean_flow_magnitude(frames_bgr: list[np.ndarray]) -> float:
    """Mean Farnebäck optical-flow magnitude across consecutive frame pairs."""
    if len(frames_bgr) < 2:
        return 0.0
    mags = []
    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for f in frames_bgr[1:]:
        cur_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(mag.mean()))
        prev_gray = cur_gray
    return float(np.mean(mags))


def clip_cos_distance(clip_model, preprocess, device, a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """1 − cos(CLIP image embedding) between two frames."""
    a = Image.fromarray(to_rgb_uint8(a_bgr))
    b = Image.fromarray(to_rgb_uint8(b_bgr))
    ta = preprocess(a).unsqueeze(0).to(device)
    tb = preprocess(b).unsqueeze(0).to(device)
    with torch.no_grad():
        ea = clip_model.encode_image(ta)
        eb = clip_model.encode_image(tb)
        ea = ea / ea.norm(dim=-1, keepdim=True)
        eb = eb / eb.norm(dim=-1, keepdim=True)
        cos = (ea * eb).sum(-1).item()
    return float(1 - cos)


# ── Per-clip metrics ──────────────────────────────────────────────────────────
@dataclass
class ClipMetrics:
    clip_id: str
    # boundary gap (frame 24 vs 96)
    gap_psnr: float
    gap_ssim: float
    gap_lpips: float
    gap_hist_chi2: float
    gap_rgb_l1: float
    gap_clip_cos_dist: float
    # total span (frame 0 vs 120)
    span_psnr: float
    span_ssim: float
    span_lpips: float
    span_clip_cos_dist: float
    # motion
    flow_start: float
    flow_middle: float
    flow_end: float


def measure_clip(clip_id: str, mp4_path: pathlib.Path, lpips_model, clip_model, clip_preprocess, device) -> ClipMetrics:
    frames = load_clip_frames(mp4_path, N_FRAMES)
    f0    = frames[0]
    f24   = frames[K - 1]
    f96   = frames[N_FRAMES - K]
    f120  = frames[N_FRAMES - 1]

    return ClipMetrics(
        clip_id=clip_id,
        gap_psnr=psnr_pair(f24, f96),
        gap_ssim=ssim_pair(f24, f96),
        gap_lpips=lpips_pair(lpips_model, f24, f96),
        gap_hist_chi2=histogram_chi2(f24, f96),
        gap_rgb_l1=mean_rgb_l1(f24, f96),
        gap_clip_cos_dist=clip_cos_distance(clip_model, clip_preprocess, device, f24, f96),
        span_psnr=psnr_pair(f0, f120),
        span_ssim=ssim_pair(f0, f120),
        span_lpips=lpips_pair(lpips_model, f0, f120),
        span_clip_cos_dist=clip_cos_distance(clip_model, clip_preprocess, device, f0, f120),
        flow_start=mean_flow_magnitude(frames[:K]),
        flow_middle=mean_flow_magnitude(frames[K:N_FRAMES - K + 1]),
        flow_end=mean_flow_magnitude(frames[N_FRAMES - K:]),
    )


# ── Correlation analysis ──────────────────────────────────────────────────────
def correlate(metrics: list[ClipMetrics]) -> list[dict]:
    rows = []
    ids = [m.clip_id for m in metrics]
    metric_keys = [
        k for k in asdict(metrics[0]).keys() if k != "clip_id"
    ]
    for exp in ("exp030", "exp032", "exp033"):
        psnr_vec = np.array([EXP_PSNR[i][exp] for i in ids], dtype=float)
        for mk in metric_keys:
            x = np.array([getattr(m, mk) for m in metrics], dtype=float)
            try:
                pr, pp = pearsonr(x, psnr_vec)
            except Exception:
                pr, pp = float("nan"), float("nan")
            try:
                sr, sp = spearmanr(x, psnr_vec)
            except Exception:
                sr, sp = float("nan"), float("nan")
            rows.append({
                "exp": exp,
                "vs_metric": mk,
                "pearson_r": round(pr, 3),
                "pearson_p": round(pp, 4),
                "spearman_r": round(sr, 3),
                "spearman_p": round(sp, 4),
            })
    return rows


def main():
    out_root = REPO_ROOT / "outputs" / "analysis" / "transition_hardness"
    run_id, run_dir = next_run_dir(out_root)
    print(f"[info] run_id = {run_id}  run_dir = {run_dir}")

    device = "cpu"

    print("[load] LPIPS (alex) …")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    print("[load] CLIP ViT-B/32 …")
    import clip as openai_clip
    clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    print("[measure] per-clip metrics …")
    metrics: list[ClipMetrics] = []
    t0 = time.time()
    for clip_id, rel_path in CLIPS:
        t = time.time()
        m = measure_clip(clip_id, REPO_ROOT / rel_path, lpips_model, clip_model, clip_preprocess, device)
        metrics.append(m)
        print(f"  [{clip_id}] gap_psnr={m.gap_psnr:.2f} gap_lpips={m.gap_lpips:.3f} "
              f"gap_clip={m.gap_clip_cos_dist:.3f} flow_mid={m.flow_middle:.2f}  ({time.time()-t:.1f}s)")
    print(f"[measure] done in {time.time()-t0:.1f}s")

    # Save metrics CSV
    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(metrics[0]).keys())
        w.writeheader()
        for m in metrics:
            w.writerow(asdict(m))
    print(f"[save] {metrics_csv}")

    # Save exp metrics for reference
    with (run_dir / "exp_psnr_reference.json").open("w") as f:
        json.dump(EXP_PSNR, f, indent=2)

    # Correlation analysis
    print("[correlate] …")
    corr = correlate(metrics)
    corr_csv = run_dir / "correlations.csv"
    with corr_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr[0].keys()))
        w.writeheader()
        w.writerows(corr)
    print(f"[save] {corr_csv}")

    # Print top correlations to console
    print()
    print("─── Top correlations with exp_033 recon PSNR (sorted by |spearman_r|) ───")
    e33 = [r for r in corr if r["exp"] == "exp033"]
    e33.sort(key=lambda r: -abs(r["spearman_r"]))
    print(f"{'metric':24s}  pearson_r  pearson_p  spearman_r  spearman_p")
    for r in e33:
        print(f"  {r['vs_metric']:22s}  {r['pearson_r']:>8.3f}  {r['pearson_p']:>8.4f}  "
              f"{r['spearman_r']:>9.3f}  {r['spearman_p']:>9.4f}")

    print()
    print("─── Top correlations with exp_030 recon PSNR ───")
    e30 = [r for r in corr if r["exp"] == "exp030"]
    e30.sort(key=lambda r: -abs(r["spearman_r"]))
    print(f"{'metric':24s}  pearson_r  pearson_p  spearman_r  spearman_p")
    for r in e30:
        print(f"  {r['vs_metric']:22s}  {r['pearson_r']:>8.3f}  {r['pearson_p']:>8.4f}  "
              f"{r['spearman_r']:>9.3f}  {r['spearman_p']:>9.4f}")

    print()
    print("─── Top correlations with exp_032 recon PSNR ───")
    e32 = [r for r in corr if r["exp"] == "exp032"]
    e32.sort(key=lambda r: -abs(r["spearman_r"]))
    print(f"{'metric':24s}  pearson_r  pearson_p  spearman_r  spearman_p")
    for r in e32:
        print(f"  {r['vs_metric']:22s}  {r['pearson_r']:>8.3f}  {r['pearson_p']:>8.4f}  "
              f"{r['spearman_r']:>9.3f}  {r['spearman_p']:>9.4f}")

    print(f"\n[done] run_id={run_id}")


if __name__ == "__main__":
    main()
