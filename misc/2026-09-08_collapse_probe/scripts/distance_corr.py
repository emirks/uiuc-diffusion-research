"""Endpoint distance (DINO / CLIP / pixel) vs confinement residual on the probe's 120 anchor pairs.
Distances are between the two anchor frames the clips are scored against: frame 8 (start anchor) and frame 120 (R1's end).
CLIP = 1 - cosine(ViT-B/32 image embeddings); DINO = realized_dino from the eval (1 - cos of DINOv2 CLS); pixel = gap_rel
of the R3 clip (anchor-to-anchor L2 in instrument units). Spearman rho per outcome; terciles for R3.
    source $LAB/envs-aarch64/activate && HF_HOME=$LAB/cache/huggingface HF_HUB_OFFLINE=1 python misc/2026-09-08_collapse_probe/scripts/distance_corr.py
"""
import io, subprocess, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy.stats import spearmanr
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe(); torch.set_num_threads(8)
C = "misc/2026-09-08_collapse_probe"
pc = pd.read_csv(f"{C}/results/per_clip.csv"); pr = pd.read_csv(f"{C}/results/paired.csv")

def frame(path, idx):
    out = subprocess.run([ff, "-v", "error", "-threads", "1", "-i", path, "-vf", f"select='eq(n\\,{idx})'", "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return Image.open(io.BytesIO(out)).convert("RGB")

from transformers import CLIPModel, CLIPProcessor
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"); clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
@torch.no_grad()
def emb(ims):
    v = clip.vision_model(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"])
    return torch.nn.functional.normalize(clip.visual_projection(v.pooler_output), dim=-1).numpy()

import os
cache = f"{C}/results/clip_dist.csv"
if os.path.exists(cache):
    cd = pd.read_csv(cache); clipd = {(a, int(b)): float(c) for a, b, c in zip(cd.prompt_id, cd.seed, cd.clip_dist)}
else:
    clipd = {}
    for i, (_, r) in enumerate(pc[pc.run == "R1"].iterrows()):
        z = emb([frame(r.path, 8), frame(r.path, 120)]); clipd[(r.prompt_id, int(r.seed))] = float(1 - (z[0] * z[1]).sum())
        if i % 20 == 0: print(f"  clip {i}/120", file=sys.stderr, flush=True)
    pd.DataFrame([dict(prompt_id=k[0], seed=k[1], clip_dist=v) for k, v in clipd.items()]).to_csv(cache, index=False)
pr["clip_dist"] = [clipd.get((p, int(s)), np.nan) for p, s in zip(pr.prompt_id, pr.seed)]
g = pc[pc.run == "R3"][["prompt_id", "seed", "gap_rel"]].rename(columns={"gap_rel": "pixel_gap"}); pr = pr.merge(g, on=["prompt_id", "seed"], how="left")
for run in ["R1", "R2", "R3"]:
    if f"DR_{run}" in pr.columns: continue
    d = pc[pc.run == run][["prompt_id", "seed", "DR_med"]].rename(columns={"DR_med": f"DR_{run}"}); pr = pr.merge(d, on=["prompt_id", "seed"], how="left")

def rho(x, y):
    m = np.isfinite(x) & np.isfinite(y); r = spearmanr(x[m], y[m]); return f"{r.statistic:+.2f} (p={r.pvalue:.3f}, n={m.sum()})"
print("distance descriptives by tier, median [IQR]  (anchor frame 8 vs anchor frame 120):")
for t, gg in pr.groupby("tier"):
    print(f"  {t:4s} DINO {gg.realized_dino.median():.2f} [{gg.realized_dino.quantile(.25):.2f},{gg.realized_dino.quantile(.75):.2f}]  "
          f"CLIP {gg.clip_dist.median():.2f} [{gg.clip_dist.quantile(.25):.2f},{gg.clip_dist.quantile(.75):.2f}]  "
          f"pixel {gg.pixel_gap.median():.2f} [{gg.pixel_gap.quantile(.25):.2f},{gg.pixel_gap.quantile(.75):.2f}]")
print("\nSpearman rho(distance, outcome)  — negative = more distant endpoints -> lower DR (closer to the line):")
for lab, sub in [("all 120", pr), ("scene-change only (90)", pr[pr.tier == "high"]), ("in-place only (30)", pr[pr.tier == "low"])]:
    print(f"\n [{lab}]")
    for y, yl in [("DR_R3", "R3 DR  both anchors, captions only"), ("DR_R2", "R2 DR  both anchors, full text"), ("DR_R1", "R1 DR  start only, full text"),
                  ("dDR_R3_minus_R2", "dDR R3-R2  (text removed)"), ("dDR_R2_minus_R1", "dDR R2-R1  (anchor added)")]:
        print(f"   {yl:36s} DINO {rho(sub.realized_dino.values, sub[y].values):26s} CLIP {rho(sub.clip_dist.values, sub[y].values):26s} pixel {rho(sub.pixel_gap.values, sub[y].values)}")
print("\nR3 by distance tercile (pooled 120):")
for col in ["realized_dino", "clip_dist", "pixel_gap"]:
    q = pd.qcut(pr[col], 3, labels=["near", "mid", "far"])
    s = pr.groupby(q, observed=True).apply(lambda g: (g.DR_R3 <= 0.12).mean()); m = pr.groupby(q, observed=True).DR_R3.median(); d = pr.groupby(q, observed=True).dDR_R3_minus_R2.median()
    print(f"  {col:14s} on-line share {', '.join(f'{k}={v*100:.0f}%' for k, v in s.items()):32s} DR_R3 median {', '.join(f'{k}={v:.2f}' for k, v in m.items()):28s} dDR(R3-R2) median {', '.join(f'{k}={v:+.2f}' for k, v in d.items())}")
print("\nwithin scene-change: R3 by CLIP tercile (DINO is saturated there):")
hi = pr[pr.tier == "high"]; q = pd.qcut(hi.clip_dist, 3, labels=["near", "mid", "far"])
print("  on-line share", {k: f"{v*100:.0f}%" for k, v in hi.groupby(q, observed=True).apply(lambda g: (g.DR_R3 <= 0.12).mean()).items()},
      " DR_R3 median", {k: f"{v:.2f}" for k, v in hi.groupby(q, observed=True).DR_R3.median().items()},
      " CLIP range", {k: f"{g.clip_dist.min():.2f}-{g.clip_dist.max():.2f}" for k, g in hi.groupby(q, observed=True)})
pr.to_csv(f"{C}/results/paired_with_distances.csv", index=False); print("\nwrote results/paired_with_distances.csv")
