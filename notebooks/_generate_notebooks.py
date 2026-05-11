"""
_generate_notebooks.py
======================
Creates the four analysis notebooks from scratch (no outputs — run in Jupyter
to regenerate them).

Run with:
    python3.10 _generate_notebooks.py
"""

import json, pathlib, uuid

NB_DIR = pathlib.Path(__file__).parent


def uid():
    return str(uuid.uuid4())[:8]


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uid(),
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uid(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save(notebook: dict, name: str):
    path = NB_DIR / name
    with open(path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"  ✓  {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# SHARED SETUP CELL  (used in every notebook)
# ──────────────────────────────────────────────────────────────────────────────

SETUP = """\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve()))  # find trajectory_utils.py

from trajectory_utils import *

# Load all 10 samples (extracts features, frees raw tensors)
records = load_all_records()

# Load manual GT transition annotations
gt_dict = load_gt_annotations()

n_gt = sum(1 for v in gt_dict.values() if v is not None)
print(f"\\n✓ GT annotations available for {n_gt}/{len(gt_dict)} samples")
for sid, gs in sorted(gt_dict.items()):
    gp_str = f"  p≈{gt_latent(sid, gt_dict):.1f}" if gs is not None else "  —"
    gs_str = f"{gs:.2f}s" if gs is not None else " —"
    print(f"  {sid:<50s}  {gs_str:>7s}{gp_str}")
"""

# ──────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 1 — VAE Latent Space
# ──────────────────────────────────────────────────────────────────────────────

NB1_TITLE = """\
# exp_021 · Level 1 — VAE Latent Space (`z_t`)

Analyses the **noisy video latent trajectory** recorded during Stage-1 denoising.

| Tensor | Shape | What it encodes |
|--------|-------|-----------------|
| `z_t[τ, :, p, :, :]` | `[S=40, C=128, F'=16, H'=16, W'=24]` | Noisy video latent at denoising step τ, latent frame p |
| `z_final` | `[C=128, F'=16, H'=16, W'=24]` | Fully denoised clean latent z₀ |

---

### Conditioning geometry

```
Latent frames:  p=0  p=1  p=2  p=3  |  p=4  p=5 ... p=11  |  p=12  p=13  p=14  p=15
                ←  start-clip (cyan) →  ← free middle (white) →  ← end-clip (magenta) →
                  K_LAT=4 frames          8 model-generated frames    K_LAT=4 frames
```

### Features extracted (each aggregates over C=128 channels and H'×W'=384 spatial positions)

| Feature | Computation | What it measures |
|---------|-------------|-----------------|
| **`norm_z[τ, p]`** | `‖z_t[τ, :, p, :, :]‖₂` | Noise level proxy. Decreases from τ=0 → τ=39. Conditioned frames should diverge from free-middle early. |
| **`speed_z[τ, p]`** | `‖z_t[τ,:,p+1,:,:] − z_t[τ,:,p,:,:]‖₂` | Spatial displacement between adjacent latent frames at step τ. Large = large content difference. |
| **`curvature_z[τ, p]`** | `‖z_t(p+2) − 2z_t(p+1) + z_t(p)‖₂` | **Primary dissolve signal.** Measures how sharply the frame trajectory bends at p. A spike = hard transition. |
| **`angular_z[τ, p]`** | `cos(Δz_t(p), Δz_t(p+1))` | Direction consistency. cos < 0 = trajectory reverses = hard content change. |
| **`step_size_z[τ, p]`** | `‖z_t[τ+1,p] − z_t[τ,p]‖₂` | How much frame p changes per denoising step. Frames with high late-step changes are uncertain. |

> **Shading:** 🔵 Cyan = start-clip `p=0..3` · 🟣 Magenta = end-clip `p=12..15` · ⬜ White = free-middle `p=4..11`  
> **Green solid line** = ground-truth transition annotation (from manual labelling).
"""

NB1_GALLERY_MD = """\
## Video Gallery — All 10 Generated Clips

Generated clips sorted by semantic class (border colour = class colour).

**Watch each video first** — note where you see the visual cut or dissolve,
then compare with the latent signals in the plots below.

**Semantic class key:**
- 🔵 **C1**: similar context / similar category / similar motion (easiest — expect smooth transition)
- 🟢 **C2**: similar context / similar category / different motion
- 🟠 **C5**: different context / similar category / similar motion
- 🔴 **C6**: different context / similar category / different motion
- 🟣 **C8**: different context / different category / different motion (hardest — expect sharp cut)
"""

NB1_GALLERY = """\
display(video_grid(records, cols=5, width=300))
"""

NB1_HEATMAP_MD = """\
---
## 1.1 — VAE Latent Feature Heatmaps (`τ × p`)

Each row in the grid below is one sample; each column is one feature.
The heatmap axes are denoising step τ (y, 0=noisy) vs. latent frame p (x).

**What to look for:**

| Feature | Dissolve signature |
|---------|--------------------|
| `curvature_z` | Vertical bright stripe at the transition frame — present across all 40 steps for hard cuts |
| `angular_z` | Blue vertical band (cos < 0) at the cut frame — the trajectory reverses direction |
| `pred_mag` | Persistent bright stripe from earliest steps — the velocity is always large at the transition frame |
| `norm_z` | Conditioned frames (cyan/magenta) should stand out from free-middle early in denoising |

**Compare across classes:** Do class 8 samples show a sharper curvature stripe than class 1?
"""

NB1_HEATMAP = """\
import matplotlib.gridspec as gridspec

n = len(records)
fig = plt.figure(figsize=(18, n * 2.4 + 1.0))
fig.suptitle(
    "Level 1 — VAE Latent Features  (τ × p)  |  cyan=start  magenta=end  green line=GT",
    fontsize=12, fontweight="bold", y=0.995
)

panels = [
    ("norm_z",      "‖z_t(p)‖  noise level",          "PuBu_r",   False),
    ("speed_z",     "speed_z  ‖Δ_p z_t‖",              "YlOrRd",   False),
    ("curvature_z", "curvature_z  ‖Δ²_p z_t‖ ◄DISSOLVE", "hot",  False),
    ("angular_z",   "angular_z  cos(Δ,Δ) ◄FLIP",       "RdYlGn",   True),
    ("pred_mag",    "pred_mag  ‖v_θ(p)‖",               "plasma",   False),
    ("step_size_z", "step_size  ‖Δ_τ z_t(p)‖",         "magma",    False),
]

n_cols = len(panels)
gs_grid = gridspec.GridSpec(n, n_cols, figure=fig, hspace=0.55, wspace=0.38)

for row, r in enumerate(records):
    f = r["feats"]
    F = f["F"]
    for col, (key, title, cmap, div) in enumerate(panels):
        ax = fig.add_subplot(gs_grid[row, col])
        heatmap(ax, f[key], title if row == 0 else "", cmap, div, F=F)
        if col == 0:
            ax.set_ylabel(f"{r['short_id']}\\nstep τ", fontsize=6.5, labelpad=3)

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.show()
print("Row = one sample.  Column = one feature.  Compare curvature_z stripe across classes.")
"""

NB1_CLEAN_MD = """\
---
## 1.2 — Clean Latent `z₀` Signals Per Sample

`z₀ = z_t[τ=39]` is the fully denoised latent — the direct precursor to the decoded video.
Its geometry is the most diagnostic for locating the dissolve frame.

**Four panels per sample:**
1. **`‖z₀(p)‖` latent energy** — frame energy profile
2. **`‖Δ²z₀‖` curvature** — where the latent trajectory bends most sharply
3. **`cos(Δ,Δ)` angular consistency** — where the trajectory reverses direction (red fill = reversal zone)
4. **`step_size_z` heatmap** — how much each frame changes at each denoising step (late = uncertain)

**Green solid line** = ground-truth transition (seconds converted to latent frame).  
If the curvature peak and angular minimum align with the GT line → the geometric signal correctly identifies the transition.
"""

NB1_CLEAN = """\
n = len(records)
fig, axes = plt.subplots(n, 4, figsize=(18, n * 2.0))
fig.suptitle(
    "Level 1 — Clean Latent z₀ Signals  |  cyan=start  magenta=end  🟢=GT annotation",
    fontsize=11, fontweight="bold"
)

for row, r in enumerate(records):
    f      = r["feats"]
    F      = f["F"]
    frames = np.arange(F)
    f_c2   = np.arange(F - 2) + 1.0
    col    = r["color"]
    sid    = r["sample_id"]

    for colidx in range(4):
        ax = axes[row, colidx]
        if colidx == 0:
            ax.bar(frames, f["norm_z0"], width=0.75, color=col, alpha=0.8)
            add_gt_vline(ax, sid, gt_dict, label=(row == 0))
            shade_cond(ax, F, F=F)
            ax.set_xlim(-0.5, F-0.5); ax.set_xticks(range(F))
            ax.tick_params(labelsize=6)
            if row == 0: ax.set_title("‖z₀(p)‖  latent energy", fontsize=8)
            ax.set_ylabel(r["short_id"], fontsize=6, labelpad=2)

        elif colidx == 1:
            ax.bar(f_c2, f["curvature_z0"], width=0.75, color="#FF5722", alpha=0.85)
            add_gt_vline(ax, sid, gt_dict, label=(row == 0))
            shade_cond(ax, F-2, F=F)
            ax.set_xlim(-0.5, F-0.5); ax.set_xticks(range(F))
            ax.tick_params(labelsize=6)
            if row == 0: ax.set_title("‖Δ²z₀‖  curvature  ◄ DISSOLVE", fontsize=8)

        elif colidx == 2:
            ang = f["angular_z0"]
            ax.plot(f_c2, ang, "o-", color="#2196F3", lw=1.5, ms=3.5)
            ax.axhline(0, color="gray", lw=0.8, ls=":")
            ax.fill_between(f_c2, ang, 0, where=(ang < 0), color="red", alpha=0.3)
            add_gt_vline(ax, sid, gt_dict, label=(row == 0))
            shade_cond(ax, F-2, F=F)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xlim(-0.5, F-0.5); ax.set_xticks(range(F))
            ax.tick_params(labelsize=6)
            min_lbl = f"min={f['angular_z0'].min():.3f}"
            if row == 0: ax.set_title(f"cos(Δ,Δ)  direction consistency\\n{min_lbl}", fontsize=8)
            else: ax.set_title(min_lbl, fontsize=7)

        else:
            ax.imshow(f["step_size_z"].T, aspect="auto", cmap="magma",
                      origin="lower", extent=[-0.5, f["S"]-0.5, -0.5, F-0.5])
            gp = gt_latent(sid, gt_dict)
            if gp is not None:
                ax.axhline(gp, color="#4CAF50", lw=1.5, ls="-", alpha=0.9)
            ax.set_xlabel("step τ", fontsize=6); ax.set_ylabel("frame p", fontsize=6)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title("step_size_z  ‖Δ_τ z_t(p)‖\\n(when does frame change?)", fontsize=7)

plt.tight_layout()
plt.show()
"""

NB1_INLINE_MD = """\
---
## 1.3 — Video + z₀ Signal Comparison (Inline)

Side-by-side: the generated video alongside three key 1-D signals from `z₀`.

**How to use this view:**
1. Play the video and note the perceptual transition time (seconds).
2. Convert: `p ≈ time_s × 24 / 8`.
3. Check if the curvature peak and angular minimum align with the green GT line.

All signals are from the **final clean latent** (`τ=39`).  
🟢 Green vertical line = ground-truth transition annotation.
"""

NB1_INLINE = """\
import io

all_html = []
all_html.append(
    '<h3 style="font-family:sans-serif">Level 1 — Video + z₀ Signals Per Sample</h3>'
    '<p style="font-family:sans-serif;font-size:12px">'
    'Each row: 🎬 video  |  curvature  |  angular consistency  |  velocity magnitude<br>'
    '🟢 solid = GT annotation</p>'
)

for r in records:
    f      = r["feats"]
    F      = f["F"]
    f_c2   = np.arange(F - 2) + 1.0
    frames = np.arange(F)
    ang    = f["angular_z0"]
    col    = r["color"]
    sid    = r["sample_id"]
    gs     = gt_dict.get(sid)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 2.4))
    gt_str = f"  |  GT={gs:.2f}s" if gs is not None else ""
    fig.suptitle(
        r["short_id"] + f"  |  Class {r['cls']}: {r['cls_label']}" + gt_str,
        fontsize=9, fontweight="bold", color=col
    )

    # curvature
    ax1.bar(f_c2, f["curvature_z0"], width=0.75, color="#FF5722", alpha=0.85)
    add_gt_vline(ax1, sid, gt_dict, label=True)
    shade_cond(ax1, F-2, F=F)
    ax1.set_xlim(-0.5, F-0.5); ax1.set_xticks(range(F))
    ax1.set_title(f"curvature_z0", fontsize=8)
    ax1.set_xlabel("frame p"); ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)

    # angular
    ax2.plot(f_c2, ang, "o-", color="#2196F3", lw=1.5, ms=3.5)
    ax2.axhline(0, color="gray", lw=0.8, ls=":")
    ax2.fill_between(f_c2, ang, 0, where=(ang < 0), color="red", alpha=0.3)
    add_gt_vline(ax2, sid, gt_dict, label=False)
    shade_cond(ax2, F-2, F=F)
    ax2.set_ylim(-1.15, 1.15); ax2.set_xlim(-0.5, F-0.5); ax2.set_xticks(range(F))
    ax2.set_title(f"angular_z0  [min={ang.min():.3f}]", fontsize=8)
    ax2.set_xlabel("frame p")
    ax2.tick_params(labelsize=7)

    # pred_mag0 (velocity magnitude at final denoising step)
    ax3.bar(frames, f["pred_mag0"], width=0.75, color="#9C27B0", alpha=0.85)
    add_gt_vline(ax3, sid, gt_dict, label=False)
    shade_cond(ax3, F, F=F)
    ax3.set_xlim(-0.5, F-0.5); ax3.set_xticks(range(F))
    ax3.set_title("pred_mag0  ‖v_θ(p)‖  (τ=39)", fontsize=8)
    ax3.set_xlabel("frame p")
    ax3.tick_params(labelsize=7)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = __import__("base64").b64encode(buf.read()).decode()

    vid_h  = video_html(r["video_path"], width=340, label=r["short_id"], cls=r["cls"])
    plot_h = f'<img src="data:image/png;base64,{img_b64}" style="height:195px;vertical-align:top">'
    all_html.append(
        f'<div style="display:flex;align-items:flex-start;gap:10px;'
        f'margin-bottom:10px;border-bottom:1px solid #ddd;padding-bottom:8px">'
        f'{vid_h}{plot_h}</div>'
    )

display(HTML("\\n".join(all_html)))
"""

NB1_PCA_MD = """\
---
## 1.4 — PCA of VAE Latent Frame Embeddings (`z₀`)

Complement to the bar-chart signals: project the 16 frame mean-embeddings of `z₀`
into 2D to see their geometric layout in one scatter.

**Computation:**  
For each latent frame `p`, average `z₀[C=128, :, :]` over the 16×24 spatial positions
→ one 128-dim vector. Stack all 16 → `[16, 128]` → centre → SVD → 2D.

| Symbol | Meaning |
|--------|---------|
| **Colour** | 🔵 blue=frame 0 → 🔴 red=frame 15 (temporal order) |
| **★** | Conditioned frame (`p=0..3`, `p=12..15`) — model must reconstruct exactly |
| **●** | Free-middle frame (`p=4..11`) — freely generated |
| **🟢 diamond** | GT annotation (if available) |
| **EV** | Explained variance of PC1 / PC2 |

**What to look for:**
- Conditioned frames (★) anchor opposite ends of the scatter; free-middle (●) interpolates.
- A **kink** in the frame trajectory = latent discontinuity at the transition.
- Compare shape across classes: does class 8 show a sharper kink than class 1?
"""

NB1_PCA = """\
import gc

cmap_frames = plt.get_cmap("RdYlBu_r")
n_cols = 5
n_rows = (len(records) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 3.6))
axes_flat = axes.flatten()

fig.suptitle(
    "Level 1 — PCA of VAE Latent Frame Embeddings  (z₀, τ=39)\\n"
    "Each dot = latent frame p.  Blue=early → Red=late.  "
    "★=conditioned  ●=free-middle  ◆=GT",
    fontsize=10, fontweight="bold"
)

for idx, r in enumerate(records):
    ax  = axes_flat[idx]
    sid = r["sample_id"]
    gp  = gt_latent(sid, gt_dict)

    print(f"  VAE PCA {r['short_id']} ...", end=" ", flush=True)
    _data = torch.load(r["traj_path"], weights_only=False, map_location="cpu")
    z0    = _data["z_final"].float()
    del _data; gc.collect()
    print("ok")

    frame_vecs  = z0.mean(dim=(2, 3)).T.numpy()   # [16, 128]
    frame_vecs -= frame_vecs.mean(axis=0)
    U, S_sv, _  = np.linalg.svd(frame_vecs, full_matrices=False)
    coords  = U[:, :2] * S_sv[:2]                # [16, 2]
    expvar  = (S_sv[:2]**2) / ((S_sv**2).sum() + 1e-12)

    # Draw temporal arrows
    for p in range(F_PRIME - 1):
        ax.annotate("", xy=(coords[p+1, 0], coords[p+1, 1]),
                    xytext=(coords[p, 0], coords[p, 1]),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=0.5, alpha=0.3))

    for p in range(F_PRIME):
        c_val  = cmap_frames(p / (F_PRIME - 1))
        marker = "*" if (p < K_LAT or p >= END_IDX) else "o"
        ms     = 130 if marker == "*" else 65
        ax.scatter(coords[p, 0], coords[p, 1], color=c_val, s=ms,
                   marker=marker, zorder=3, alpha=0.9)
        ax.annotate(str(p), (coords[p, 0], coords[p, 1]),
                    fontsize=5, ha="center", va="center",
                    color="white" if marker == "o" else "k", fontweight="bold")

    # GT diamond
    if gp is not None:
        p_lo = int(gp); p_hi = min(p_lo + 1, F_PRIME - 1)
        frac = gp - p_lo
        gx   = coords[p_lo, 0] * (1 - frac) + coords[p_hi, 0] * frac
        gy   = coords[p_lo, 1] * (1 - frac) + coords[p_hi, 1] * frac
        ax.scatter(gx, gy, marker="D", s=80, color="#4CAF50", zorder=5,
                   edgecolors="black", linewidths=0.6, label=f"GT p≈{gp:.1f}")
        ax.legend(fontsize=5.5, loc="lower right", framealpha=0.7)

    ax.set_title(f"{r['short_id']}\\nEV: {expvar[0]:.1%} / {expvar[1]:.1%}",
                 fontsize=7, color=r["color"])
    ax.set_xlabel("PC1", fontsize=6); ax.set_ylabel("PC2", fontsize=6)
    ax.tick_params(labelsize=5.5)

for ax in axes_flat[len(records):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
"""

NB1_CELLS = [
    md(NB1_TITLE),
    code(SETUP),
    md(NB1_GALLERY_MD),
    code(NB1_GALLERY),
    md(NB1_HEATMAP_MD),
    code(NB1_HEATMAP),
    md(NB1_CLEAN_MD),
    code(NB1_CLEAN),
    md(NB1_INLINE_MD),
    code(NB1_INLINE),
    md(NB1_PCA_MD),
    code(NB1_PCA),
]

save(nb(NB1_CELLS), "exp021_01_vae_latent.ipynb")


# ──────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 2 — Velocity Field
# ──────────────────────────────────────────────────────────────────────────────

NB2_TITLE = """\
# exp_021 · Level 2 — Velocity Field (`v_pred`)

`v_pred[τ, :, p, :, :]` is the **flow-matching velocity prediction** at denoising step τ,
latent frame p.  Same shape as `z_t`: `[S=40, C=128, F'=16, H'=16, W'=24]`.

---

### What the velocity encodes

In flow matching (LTX-2's training objective), the velocity `v_θ` estimates the direction
from the current noisy state toward the clean target:

$$v_\\theta(z_\\tau, \\tau, c) \\approx z_0 - z_\\tau \\cdot (1 - \\tau/T)^{-1}$$

The magnitude `‖v_pred[τ,:,p,:,:]‖` directly measures **how hard the model is working
to move frame p at step τ**.

---

### Why velocity is the primary transition detector

A frame undergoing a hard dissolve must travel a large distance in latent space
(its noise-free target is semantically very different from its neighbours).
Its velocity norm will be **persistently high across all 40 denoising steps**.

Visualised as the `pred_mag` heatmap (τ × p), this appears as a
**vertical bright stripe** at the transition frame `p*` — visible even from the
earliest denoising steps.

> **Green solid line** = ground-truth transition annotation throughout this notebook.
"""

NB2_HEATMAP_MD = """\
---
## 2.1 — Velocity Magnitude Heatmap (`pred_mag`)

One heatmap per sample (5 per row).  
x-axis = latent frame p (0–15) · y-axis = denoising step τ (row 0 = noisiest).  
Colour = `‖v_θ(z_τ, τ, c)(p)‖₂`, one global colour scale across all 10 heatmaps.

**Pattern guide:**

| Visual pattern | Meaning |
|----------------|---------|
| **Vertical bright stripe across all τ at frame p** | Transition is at frame p, committed from the first step. |
| **Bright stripe only in bottom rows (τ=25..39)** | Transition decided late — model was uncertain early. |
| **Uniformly bright across all frames** | No single transition frame — similar clips (expected for class 1). |
| **Bright at magenta region (p=12..15)** | Expected artefact from end-clip conditioning, not a dissolve signal. |
"""

NB2_HEATMAP = """\
n     = len(records)
ncols = 5
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.5))
axes_flat  = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else list(axes)
if nrows == 1 and ncols > 1:
    axes_flat = list(axes)
elif nrows > 1:
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

fig.suptitle(
    "Level 2 — pred_mag = ‖v_θ(z_τ, τ, c)(p)‖₂  (τ × p)\\n"
    "Bright vertical stripe at p = TRANSITION FRAME  |  🟢 solid=GT  cyan=start  magenta=end",
    fontsize=11, fontweight="bold"
)

global_vmax = max(float(np.percentile(r["feats"]["pred_mag"], 99)) for r in records)

for i, (ax, r) in enumerate(zip(axes_flat, records)):
    f   = r["feats"]
    sid = r["sample_id"]
    im  = ax.imshow(f["pred_mag"], aspect="auto", cmap="plasma",
                    vmin=0, vmax=global_vmax, origin="upper", interpolation="nearest",
                    extent=[-0.5, f["F"]-0.5, f["S"]-0.5, -0.5])
    ax.axvline(K_LAT   - 0.5, color="#00BCD4", lw=1.2, ls="--", alpha=0.8)
    ax.axvline(END_IDX - 0.5, color="#E91E63", lw=1.2, ls="--", alpha=0.8)
    gp = gt_latent(sid, gt_dict)
    if gp is not None:
        ax.axvline(gp, color="#4CAF50", lw=2.0, ls="-",
                   label=f"GT {gt_dict[sid]:.2f}s", alpha=0.9)
        ax.legend(fontsize=5.5, loc="upper right", framealpha=0.7)
    ax.set_title(r["short_id"], fontsize=7.5, color=r["color"])
    ax.set_xlabel("frame p", fontsize=7)
    ax.set_ylabel("step τ", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xticks(range(f["F"]))
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

for ax in axes_flat[n:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
"""

NB2_CROSS_MD = """\
---
## 2.2 — Cross-Sample Curvature + Angular Comparison

All 10 samples overlaid on the same axes. Colour = semantic class.

**Top panel — normalised curvature `‖z₀(p+2)−2z₀(p+1)+z₀(p)‖₂ / max`:**  
Each curve normalised to [0,1] for shape comparison.
- **Narrow peak** = hard cut at one specific frame
- **Broad hump** = gradual transition
- **Flat** = no transition (similar clips)

**Bottom panel — angular consistency `cos(Δz₀(p), Δz₀(p+1))`:**  
Red-filled region = cos < 0 = the latent trajectory reverses direction.

**Solid vertical lines** = GT annotations. 
Do all GT lines land at or near the curvature peak for that sample?
"""

NB2_CROSS = """\
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
fig.suptitle(
    "Level 2 — Cross-Sample Dissolve Signals (clean latent z₀)\\n"
    "Solid vertical = GT annotation  |  colour = class",
    fontsize=11, fontweight="bold"
)

F    = records[0]["feats"]["F"]
f_c2 = np.arange(F - 2) + 1.0

for r in records:
    f    = r["feats"]
    col  = r["color"]
    lbl  = r["short_id"]
    sid  = r["sample_id"]

    c0   = f["curvature_z0"]
    c0_n = c0 / (c0.max() + 1e-8)
    ax1.plot(f_c2, c0_n, "-", color=col, lw=2.0, alpha=0.85, label=lbl)
    add_gt_vline(ax1, sid, gt_dict, color=col, lw=1.8, ls="-", label=False)

    ax2.plot(f_c2, f["angular_z0"], "-", color=col, lw=2.0, alpha=0.85, label=lbl)
    add_gt_vline(ax2, sid, gt_dict, color=col, lw=1.8, ls="-", label=False)

ax1.set_title("Normalised curvature_z0 = ‖z₀(p+2)−2z₀(p+1)+z₀(p)‖₂ / max", fontsize=9)
ax1.set_ylabel("curvature / max", fontsize=9)
ax1.set_ylim(-0.05, 1.15)
ax1.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.8)
ax1.axhline(0, color="gray", lw=0.5, ls=":")

ax2.set_title("angular_z0 = cos(z₀(p+1)−z₀(p),  z₀(p+2)−z₀(p+1))", fontsize=9)
ax2.set_ylabel("cosine", fontsize=9)
ax2.set_ylim(-1.15, 1.15)
ax2.axhline(0, color="gray", lw=0.8, ls=":")
ax2.fill_between(f_c2, -1.15, 0, alpha=0.04, color="red", label="reversal zone")
ax2.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.8)

for ax in (ax1, ax2):
    shade_cond(ax, F-2, F=F)
    ax.set_xticks(range(F))
    ax.set_xlim(-0.5, F-0.5)
    ax.set_xlabel("frame  p", fontsize=9)
    ax.tick_params(labelsize=8)
    ax2b = ax.twiny()
    ax2b.set_xlim(-0.5, F-0.5)
    ax2b.set_xticks(range(F))
    ax2b.set_xticklabels(
        [f"{p*LTX_TEMPORAL_SCALE/VIDEO_FPS:.1f}s" for p in range(F)],
        fontsize=5.5, rotation=45
    )

plt.tight_layout()
plt.show()
"""

NB2_COMMIT_MD = """\
---
## 2.3 — Latent Curvature Frame Evolution over Denoising Steps

This plot tracks **where the curvature peak in `z_t[τ]` falls at each denoising step τ**.

Each line = `argmax_p curvature_z[τ, :]` — the frame with the highest curvature at step τ.

**What this shows:**  
As the model denoises (τ: 0 → 39), the frame-sequence geometry becomes more structured.
The curvature peak tracks which frame has the sharpest "bend" in the latent trajectory
at each denoising moment.

- **Flat horizontal line in the gold band from τ=0** → the transition location is already
  encoded in the noise geometry from the very first step.
- **Starts outside gold band, then settles in** → boundary frames dominate early, model
  progressively concentrates the transition into the free-middle region.
- **Jumpy across all 40 steps** → no stable transition signal.

**Dotted horizontal lines** = GT latent frame per sample.  
A line converging toward the dotted GT line → the curvature-peak signal tracks ground truth.

> **Shaded bands:** 🟡 gold = free-middle `p=4..11` · 🔵 cyan = start-clip · 🟣 magenta = end-clip.
"""

NB2_COMMIT = """\
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle(
    "Latent Curvature Frame Evolution — argmax(curvature_z[τ]) vs. denoising step τ\\n"
    "Tracks where the sharpest latent bend is at each denoising step\\n"
    "Dotted horizontal ▶ = GT latent frame per sample",
    fontsize=10, fontweight="bold"
)

S     = records[0]["feats"]["S"]
steps = np.arange(S)

for r in records:
    f   = r["feats"]
    col = r["color"]
    lbl = r["short_id"]
    ax.plot(steps, f["dissolve_by_step"], "-", color=col, lw=2.0, alpha=0.85, label=lbl)
    gp = gt_latent(r["sample_id"], gt_dict)
    if gp is not None:
        ax.axhline(gp, color=col, lw=1.2, ls=":", alpha=0.7)

ax.axhspan(K_LAT,      END_IDX - 1,   alpha=0.07, color="gold",
           label=f"free-middle p=[{K_LAT}..{END_IDX-1}]")
ax.axhspan(0,          K_LAT - 0.5,   alpha=0.07, color="#00BCD4")
ax.axhspan(END_IDX - 0.5, F_PRIME,    alpha=0.07, color="#E91E63")
ax.set_xlabel("denoising step τ  (0=noisy, 39=clean)", fontsize=9)
ax.set_ylabel("frame index with max curvature", fontsize=9)
ax.set_xticks(steps[::5])
ax.set_yticks(range(F_PRIME))

ax2y = ax.twinx()
ax2y.set_ylim(ax.get_ylim())
ax2y.set_yticks(list(range(F_PRIME)))
ax2y.set_yticklabels([f"{p*LTX_TEMPORAL_SCALE/VIDEO_FPS:.1f}s" for p in range(F_PRIME)], fontsize=7)
ax2y.set_ylabel("video time", fontsize=8)

ax.set_ylim(-0.5, F_PRIME - 0.5)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.85)
ax.grid(axis="both", linestyle=":", alpha=0.35)
ax.text(-1.5, K_LAT/2,           "start\\nclip", fontsize=7, ha="center", color="#00BCD4", fontweight="bold")
ax.text(-1.5, (K_LAT+END_IDX)/2, "free\\nmiddle", fontsize=7, ha="center", color="#888")
ax.text(-1.5, (END_IDX+F_PRIME)/2,"end\\nclip",  fontsize=7, ha="center", color="#E91E63", fontweight="bold")
plt.tight_layout()
plt.show()
print("Dotted horizontals = GT latent frame per sample.")
"""

NB2_THIRDS_MD = """\
---
## 2.4 — Velocity Profile by Denoising Third

Each sample's `pred_mag` profile is averaged over three equal thirds of the 40-step
denoising process:

| Column | Steps | What it captures |
|--------|-------|-----------------|
| **Early** (τ=0–13) | First 35% | Velocity in heavily noised space — prior about where large changes are needed |
| **Middle** (τ=14–26) | Middle 32% | Structure forming — velocity stabilising |
| **Final** (τ=27–39) | Last 33% | Detail refinement — lower overall velocity |

A transition frame with **consistently tall bars across all three thirds** = committed
from the very first step.

🟢 Green line = GT annotation.
"""

NB2_THIRDS = """\
fig, axes = plt.subplots(len(records), 3, figsize=(16, len(records) * 1.9))
fig.suptitle(
    "pred_mag across denoising thirds  (early / middle / final)\\n"
    "Per-frame velocity — persistent peak across thirds = early-committed transition\\n"
    "🟢 solid=GT",
    fontsize=10, fontweight="bold"
)

for row, r in enumerate(records):
    f      = r["feats"]
    pm     = f["pred_mag"]
    S      = f["S"]; F = f["F"]
    col    = r["color"]
    frames = np.arange(F)
    sid    = r["sample_id"]

    thirds = [
        (0,      S//3,   f"early  τ=0–{S//3-1}"),
        (S//3,   2*S//3, f"middle τ={S//3}–{2*S//3-1}"),
        (2*S//3, S,      f"final  τ={2*S//3}–{S-1}"),
    ]
    for tcol, (t0, t1, tlbl) in enumerate(thirds):
        ax = axes[row, tcol] if len(records) > 1 else axes[tcol]
        mean_pm = pm[t0:t1].mean(axis=0)
        ax.bar(frames, mean_pm, width=0.75, color=col, alpha=0.8)
        add_gt_vline(ax, sid, gt_dict, label=(tcol == 2 and row == 0))
        shade_cond(ax, F, F=F)
        ax.set_xlim(-0.5, F-0.5); ax.set_xticks(range(F))
        ax.tick_params(labelsize=6)
        if row == 0: ax.set_title(tlbl, fontsize=8)
        if tcol == 0: ax.set_ylabel(r["short_id"], fontsize=6, labelpad=2)

plt.tight_layout()
plt.show()
"""

NB2_CELLS = [
    md(NB2_TITLE),
    code(SETUP),
    md(NB2_HEATMAP_MD),
    code(NB2_HEATMAP),
    md(NB2_CROSS_MD),
    code(NB2_CROSS),
    md(NB2_COMMIT_MD),
    code(NB2_COMMIT),
    md(NB2_THIRDS_MD),
    code(NB2_THIRDS),
]

save(nb(NB2_CELLS), "exp021_02_velocity_field.ipynb")


# ──────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 3 — Transformer Hidden States
# ──────────────────────────────────────────────────────────────────────────────

NB3_TITLE = """\
# exp_021 · Level 3 — Transformer Hidden States (`hidden_states`)

The deepest analysis level: internal representations inside LTX-2's DiT transformer blocks.

---

### What is a transformer block in LTX-2?

LTX-2 is a **Diffusion Transformer (DiT)** with **48 transformer blocks** (attention + FFN).
We probe 4 depths:

| Label | Block index (of 48) | Depth | Expected role |
|-------|---------------------|-------|---------------|
| **L12** | Block 12 | 25% | Shallow — low-level spatial/positional patterns |
| **L24** | Block 24 | 50% | Mid — spatial + semantic mixing |
| **L35** | Block 35 | 73% | Deep — high-level semantic content |
| **L47** | Block 47 | 98% | Near-output — direct precursor to velocity prediction |

---

### What is a token in this context?

LTX-2 packs all spatial positions of all latent frames into one flat 1D token sequence:

```
Sequence length = F' × H' × W' = 16 × 16 × 24 = 6,144 tokens
```

Token `(p × 16 × 24) + (h × 24) + w` corresponds to latent frame `p`, height `h`, width `w`.

---

### Saved hidden states

```
hidden_states[τ][L].shape = [2, 6144, 4096]
                              ↑   ↑      ↑
                         CFG  tokens  hidden dim
```

- `[0]` = unconditional forward pass (CFG) · `[1]` = conditional ← always used here
- 6,144 = 16×16×24 packed tokens

---

### Frame mean-embedding (used in PCA and similarity)

For latent frame `p`, average the 384 token embeddings → one 4096-dim vector.
Stack 16 → `[16, 4096]` → SVD → 2D scatter.

---

> ⚠️ **Memory:** Each `.pt` file is ~2.4 GB when hidden states are included.
> Samples are loaded and freed one at a time.  
> **Green solid line / diamond** = GT annotation throughout.
"""

NB3_SETUP = """\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve()))

from trajectory_utils import *

records = load_all_records()
gt_dict = load_gt_annotations()

print(f"\\nLayer indices probed: {LAYER_INDICES}")
print(f"Denoising steps saved: {STEP_INDICES}")
hs_records = [r for r in records if r["has_hs"]]
print(f"\\n{len(hs_records)}/{len(records)} samples have hidden states")
"""

NB3_NORM_MD = """\
---
## 3.1 — Per-Frame Activation Norm Heatmap

For each sample that has hidden states, plot a grid:
- **Rows** = transformer block L ∈ {L12, L24, L35, L47}
- **Columns** = denoising step τ ∈ {0, 8, 16, 23, 31, 39}

Each cell = mean `‖h^L(p)‖` per frame `p` (bar chart, x=frame, y=norm).

**What to look for:**
- Frames at or near the transition should have elevated norm in deep blocks (L35/L47).
- Does the norm profile sharpen around the GT frame as denoising progresses (left → right)?
- Does it sharpen with depth (top → bottom)?
"""

NB3_NORM = """\
hs_records = [r for r in records if r["has_hs"]]
print(f"{len(hs_records)}/{len(records)} samples have hidden states.")
print("Loading one sample at a time (~2.4 GB) — may take 1–2 min total.\\n")

if not hs_records:
    print("No hidden states found — skipping.")
else:
    frames = np.arange(F_PRIME)

    for r in hs_records:
        print(f"  Loading {r['sample_id']} ...", end=" ", flush=True)
        full_data = torch.load(r["traj_path"], weights_only=False, map_location="cpu")
        hs  = full_data["hidden_states"]
        gp  = gt_latent(r["sample_id"], gt_dict)
        col = r["color"]
        sid = r["sample_id"]
        print("done")

        n_layers = len(LAYER_INDICES)
        n_steps  = len(STEP_INDICES)
        fig, axes = plt.subplots(n_layers, n_steps, figsize=(n_steps*2.2, n_layers*2.0))
        gs_val = gt_dict.get(sid)
        gt_str = f"  |  GT {gs_val:.2f}s" if gs_val is not None else ""
        fig.suptitle(
            f"Level 3 — Per-frame hidden-state norm  ‖h^L(p)‖\\n"
            f"{r['short_id']}{gt_str}  |  🟢 solid=GT",
            fontsize=9, fontweight="bold", color=col
        )

        all_norms = []
        for step_idx in STEP_INDICES:
            for layer_idx in LAYER_INDICES:
                h_t = hs.get(step_idx, {}).get(layer_idx)
                if h_t is not None:
                    nv = per_frame_norm(unpack_hidden(h_t.float()))
                    all_norms.append(nv)
        global_vmax = max(n.max() for n in all_norms) if all_norms else 1.0

        for si, step_idx in enumerate(STEP_INDICES):
            for li, layer_idx in enumerate(LAYER_INDICES):
                ax  = axes[li, si]
                h_t = hs.get(step_idx, {}).get(layer_idx)
                if h_t is None:
                    ax.set_visible(False); continue
                nv = per_frame_norm(unpack_hidden(h_t.float()))
                ax.bar(frames, nv, width=0.75, color=col, alpha=0.8)
                if gp is not None:
                    ax.axvline(gp, color="#4CAF50", lw=1.5, ls="-", alpha=0.9)
                shade_cond(ax, F_PRIME)
                ax.set_xlim(-0.5, F_PRIME-0.5); ax.set_xticks(range(F_PRIME))
                ax.set_ylim(0, global_vmax * 1.05)
                ax.tick_params(labelsize=5)
                if li == 0: ax.set_title(f"τ={step_idx}", fontsize=7)
                if si == 0: ax.set_ylabel(f"L{layer_idx}", fontsize=7)

        plt.tight_layout()
        plt.show()

        del full_data, hs
        gc.collect()
"""

NB3_PCA_MD = """\
---
## 3.2 — PCA of Transformer Frame Embeddings

**One figure per sample.** Grid: 4 rows (denoising step τ) × 4 cols (transformer block L).

Each scatter plot = 16 frame mean-embeddings projected to 2D via SVD.  
Blue=frame 0 → Red=frame 15. **★**=conditioned · **●**=free-middle · **🟢 diamond**=GT.

**Read across rows (τ: noise → clean):**
- τ=0 (noise): usually random scatter.
- τ=8 (early): coarse structure may emerge.
- τ=16 (mid): semantic content partially formed.
- τ=39 (clean): clearest — conditioned ★ frames anchor opposite ends.

**Read across columns (L12 → L47):**
- L12: mainly positional/spatial information.
- L47: semantic content dominates; expect tightest clustering.

**What to look for:**
- **Kink in the frame trajectory** near the GT frame (🟢 diamond)
- **Two clusters** (start-group vs end-group) = hard semantic separation
- **Single cluster** = block/step hasn't separated the two clip semantics yet
"""

NB3_PCA = """\
cmap_frames = plt.get_cmap("RdYlBu_r")
hs_records  = [r for r in records if r["has_hs"]]

if not hs_records:
    print("No hidden states — skipping PCA.")
else:
    print(f"PCA: {len(hs_records)} samples × {len(PCA_TIMESTEPS)} timesteps "
          f"× {len(LAYER_INDICES)} blocks\\n")

    for r in hs_records:
        sid = r["sample_id"]
        print(f"  Loading {sid} ...", end=" ", flush=True)
        full_data = torch.load(r["traj_path"], weights_only=False, map_location="cpu")
        hs  = full_data["hidden_states"]
        gp  = gt_latent(sid, gt_dict)
        col = r["color"]
        print("done")

        n_rows = len(PCA_TIMESTEPS)
        n_cols = len(LAYER_INDICES)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.3, n_rows * 3.3))

        gs_val = gt_dict.get(sid)
        gt_str = f"  |  GT={gs_val:.2f}s (p≈{gp:.1f})" if gs_val is not None else ""
        fig.suptitle(
            f"Level 3 — PCA frame embeddings  ·  {r['short_id']}\\n"
            f"Rows=denoising step τ  |  Cols=transformer block L (depth)\\n"
            f"{gt_str}  |  ◉=conditioned  ○=free-middle  ◆=GT",
            fontsize=9, fontweight="bold", color=col
        )

        for ri, step_idx in enumerate(PCA_TIMESTEPS):
            for ci, layer_idx in enumerate(LAYER_INDICES):
                ax  = axes[ri, ci]
                h_t = hs.get(step_idx, {}).get(layer_idx)
                if h_t is None:
                    ax.set_visible(False); continue

                coords, expvar = pca_frame_embeddings(unpack_hidden(h_t.float()))

                for p in range(F_PRIME - 1):
                    ax.annotate("", xy=(coords[p+1, 0], coords[p+1, 1]),
                                xytext=(coords[p, 0], coords[p, 1]),
                                arrowprops=dict(arrowstyle="-|>", color="gray",
                                               lw=0.5, alpha=0.35))
                for p in range(F_PRIME):
                    c_val  = cmap_frames(p / (F_PRIME - 1))
                    marker = "*" if (p < K_LAT or p >= END_IDX) else "o"
                    ms     = 100 if marker == "*" else 50
                    ax.scatter(coords[p, 0], coords[p, 1], color=c_val, s=ms,
                               marker=marker, zorder=3, alpha=0.9)
                    ax.annotate(str(p), (coords[p, 0], coords[p, 1]),
                                fontsize=4.5, ha="center", va="center",
                                color="white", fontweight="bold")

                if gp is not None:
                    p_lo = int(gp); p_hi = min(p_lo + 1, F_PRIME - 1)
                    frac = gp - p_lo
                    gx   = coords[p_lo, 0] * (1 - frac) + coords[p_hi, 0] * frac
                    gy   = coords[p_lo, 1] * (1 - frac) + coords[p_hi, 1] * frac
                    ax.scatter(gx, gy, marker="D", s=80, color="#4CAF50", zorder=5,
                               edgecolors="black", linewidths=0.6)

                step_lbl = {0: "τ=0 (noise)", 8: "τ=8 (early)",
                            16: "τ=16 (mid)",  39: "τ=39 (clean)"}.get(step_idx, f"τ={step_idx}")
                ax.set_title(
                    f"L{layer_idx}  {step_lbl}\\nEV:{expvar[0]:.0%}/{expvar[1]:.0%}",
                    fontsize=6.5
                )
                ax.tick_params(labelsize=5)

        plt.tight_layout()
        plt.show()

        del full_data, hs
        gc.collect()
"""

NB3_SIM_MD = """\
---
## 3.3 — Frame Cosine-Similarity Matrix

For each sample, at denoising steps τ ∈ {8 (early), 16 (mid), 39 (clean)},
plot the **[16×16] frame cosine-similarity matrix** at each of the 4 transformer blocks.

Entry `(p, q)` = cos(h̄^L(p), h̄^L(q)) — similarity between the mean embeddings
of frames p and q at block L.

**Pattern guide:**

| Pattern | Meaning |
|---------|---------|
| **Block diagonal** (two green blobs + red off-diagonal) | Hard semantic boundary at the cluster edge → the dissolve frame |
| Smooth gradient (similarity decreasing with temporal distance) | Gradual transition — no hard cut |
| Uniform green | This block doesn't differentiate frames — encodes only spatial content |

**Green dashed lines** = GT annotation.
**Cyan/magenta dotted lines** = conditioning boundaries (p=3.5 and p=11.5).

**Comparing L12 → L47:** Does the block-diagonal become sharper in deeper layers
as representations become more semantically discriminative?
"""

NB3_SIM = """\
SIM_TIMESTEPS = [8, 16, 39]
hs_records    = [r for r in records if r["has_hs"]]

if not hs_records:
    print("No hidden states — skipping similarity matrix.")
else:
    for r in hs_records:
        sid = r["sample_id"]
        print(f"Sim matrix: loading {sid} ...", end=" ", flush=True)
        full_data = torch.load(r["traj_path"], weights_only=False, map_location="cpu")
        hs  = full_data["hidden_states"]
        gp  = gt_latent(sid, gt_dict)
        col = r["color"]
        print("done")

        for target_step in SIM_TIMESTEPS:
            step_lbl = {8: "early (τ=8)", 16: "mid (τ=16)", 39: "clean (τ=39)"}[target_step]
            fig, axes = plt.subplots(1, len(LAYER_INDICES),
                                     figsize=(len(LAYER_INDICES) * 2.9, 3.0))
            gs_val = gt_dict.get(sid)
            gt_str = f"  GT {gs_val:.2f}s" if gs_val is not None else ""
            fig.suptitle(
                f"Level 3 — Frame cosine-sim  at {step_lbl}  ·  {r['short_id']}\\n"
                f"🟢 dashed=GT{gt_str}  |  Block-diagonal = hard semantic cut",
                fontsize=8.5, fontweight="bold", color=col
            )

            for ci, layer_idx in enumerate(LAYER_INDICES):
                ax  = axes[ci]
                h_t = hs.get(target_step, {}).get(layer_idx)
                if h_t is None:
                    ax.set_visible(False); continue

                sim = cosine_sim_matrix(unpack_hidden(h_t.float()))
                im  = ax.imshow(sim, vmin=-0.3, vmax=1.0, cmap="RdYlGn",
                                origin="upper", interpolation="nearest")
                for bnd, c in [(K_LAT - 0.5, "#00BCD4"), (END_IDX - 0.5, "#E91E63")]:
                    ax.axhline(bnd, color=c, lw=1.0, ls=":", alpha=0.8)
                    ax.axvline(bnd, color=c, lw=1.0, ls=":", alpha=0.8)
                if gp is not None:
                    ax.axhline(gp - 0.5, color="#4CAF50", lw=1.8, ls="--", alpha=0.9)
                    ax.axvline(gp - 0.5, color="#4CAF50", lw=1.8, ls="--", alpha=0.9)
                ax.set_xticks(range(0, F_PRIME, 4))
                ax.set_yticks(range(0, F_PRIME, 4))
                ax.tick_params(labelsize=6.5)
                ax.set_title(f"L{layer_idx}", fontsize=9)
                ax.set_xlabel("frame p", fontsize=7)
                if ci == 0:
                    ax.set_ylabel("frame q", fontsize=7)
                plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)

            plt.tight_layout()
            plt.show()

        del full_data, hs
        gc.collect()
"""

NB3_CELLS = [
    md(NB3_TITLE),
    code(NB3_SETUP),
    md(NB3_NORM_MD),
    code(NB3_NORM),
    md(NB3_PCA_MD),
    code(NB3_PCA),
    md(NB3_SIM_MD),
    code(NB3_SIM),
]

save(nb(NB3_CELLS), "exp021_03_transformer.ipynb")


# ──────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 4 — Summary
# ──────────────────────────────────────────────────────────────────────────────

NB4_TITLE = """\
# exp_021 · Summary — Cross-Level Metrics

Aggregates geometric signals from Levels 1–3 and compares them against
ground-truth annotations for all 10 samples.

---

### Column definitions

| Column | Source | What it measures |
|--------|--------|-----------------|
| `strength` | Level 1 `curvature_z0` | max/mean curvature ratio. > 1 = peak exists; higher = sharper cut. |
| `angular_min` | Level 1 `angular_z0` | Most negative cosine. < 0 = direction reversal. More negative = harder cut. |
| `gt_s` | Manual annotation | Ground-truth transition time in seconds. |
| `gt_p` | Converted from gt_s | Ground-truth latent frame (float, sub-frame precision). |
| `commit_step` | Level 2 `dissolve_by_step` | Earliest denoising step τ at which the curvature-peak location stabilises. Lower = more decisive. |
| `pred_peak_p` | Level 2 `pred_mag` | Frame with highest mean velocity across all 40 steps. |

---

> **Class guide:** 🔵 C1 (easiest) · 🟢 C2 · 🟠 C5 · 🔴 C6 · 🟣 C8 (hardest)
"""

NB4_TABLE = """\
def _commit_step_fn(dissolve_by_step: np.ndarray, tolerance: int = 1) -> int:
    final = dissolve_by_step[-1]
    for i in range(len(dissolve_by_step)):
        if np.all(np.abs(dissolve_by_step[i:] - final) <= tolerance):
            return int(i)
    return len(dissolve_by_step) - 1

rows = []
for r in records:
    f         = r["feats"]
    pm        = f["pred_mag"]
    mean_pm   = pm.mean(axis=0)
    pred_peak = int(np.argmax(mean_pm))
    commit    = _commit_step_fn(f["dissolve_by_step"])
    gs        = gt_dict.get(r["sample_id"])
    gp_approx = round(gt_latent(r["sample_id"], gt_dict), 1) if gs is not None else None

    rows.append({
        "sample":       r["short_id"],
        "class":        r["cls"],
        "cls_label":    r["cls_label"],
        "gt_s":         round(gs, 2) if gs is not None else None,
        "gt_p":         gp_approx,
        "strength":     round(f["dissolve_z0_strength"] if "dissolve_z0_strength" in f
                              else float(f["curvature_z0"].max() /
                                         (f["curvature_z0"].mean() + 1e-8)), 2),
        "angular_min":  round(float(f["angular_z0"].min()), 3),
        "pred_peak_p":  pred_peak,
        "commit_step":  commit,
    })

df_table = pd.DataFrame(rows).sort_values(["class", "sample"]).reset_index(drop=True)

def _cls_color(cls_str):
    return CLASS_COLORS.get(str(cls_str), "#888")

def _row_style(row):
    col = _cls_color(str(row["class"]))
    return [f"background-color:{col}18; border-left:4px solid {col}"] * len(row)

styled = (
    df_table.style
    .apply(_row_style, axis=1)
    .background_gradient(subset=["strength"],    cmap="YlOrRd", vmin=1, vmax=5)
    .background_gradient(subset=["angular_min"], cmap="RdYlGn", vmin=-1, vmax=1)
    .background_gradient(subset=["commit_step"], cmap="PuBu",   vmin=0, vmax=39)
    .format({
        "strength":    "{:.2f}×",
        "angular_min": "{:.3f}",
        "gt_s":        lambda x: f"{x:.2f}s" if x is not None else "—",
        "gt_p":        lambda x: f"p≈{x:.1f}" if x is not None else "—",
    })
    .set_caption("Geometric Dissolve Metrics — all 10 samples  |  gt_s = manual GT annotation")
    .set_table_styles([
        {"selector": "caption", "props": [("font-size", "13px"), ("font-weight", "bold"),
                                          ("text-align", "left"), ("margin-bottom", "8px")]},
        {"selector": "th", "props": [("background-color", "#2c2c2c"), ("color", "white"),
                                     ("font-size", "11px"), ("padding", "5px 10px")]},
        {"selector": "td", "props": [("font-size", "11px"), ("padding", "4px 10px")]},
    ])
)
display(styled)

# Class-level aggregation
agg = df_table.groupby("class").agg(
    n=("sample", "count"),
    mean_strength=("strength", "mean"),
    mean_angular_min=("angular_min", "mean"),
    mean_commit_step=("commit_step", "mean"),
).round(3)
print("\\nClass-level aggregation:")
display(agg)
"""

NB4_SCATTER_MD = """\
---
## Summary Charts

**Left — Dissolve Strength vs. Direction Reversal (scatter)**

Each dot = one sample, coloured by class.

- **Upper-left** (high strength + very negative angular_min) = hard, abrupt dissolve → expected for class 6/8
- **Lower-right** (low strength + near-zero angular_min) = smooth or absent transition → expected for class 1
- Diagonal trend = the two signals agree — the cut is geometrically clear

**Right — Commitment Step by Sample (bar chart)**

y-axis = denoising step τ at which the curvature-peak location stabilises.

- **Low bar** (τ < 10) = model decided the transition location early — confident, clear conditioning
- **High bar** (τ > 30) = late commitment — ambiguous input pair
"""

NB4_SCATTER = """\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Summary — Geometric Dissolve Signals by Semantic Class  |  🟢=GT available",
    fontsize=11, fontweight="bold"
)

# scatter: strength vs angular_min
ax = axes[0]
for r in records:
    f   = r["feats"]
    col = r["color"]
    sid = r["sample_id"]
    strength    = float(f["curvature_z0"].max() / (f["curvature_z0"].mean() + 1e-8))
    angular_min = float(f["angular_z0"].min())
    ax.scatter(strength, angular_min, color=col, s=110, zorder=3, alpha=0.9,
               label=r["short_id"])
    ax.annotate(f"C{r['cls']}", (strength, angular_min),
                fontsize=8, ha="center", va="center", color="white", fontweight="bold")
    gs = gt_dict.get(sid)
    if gs is not None:
        ax.scatter(strength, angular_min, s=260, facecolors="none",
                   edgecolors="#4CAF50", linewidths=2.0, zorder=4)
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.axvline(1, color="gray", lw=0.8, ls=":")
ax.set_xlabel("dissolve strength  (max/mean curvature z₀)", fontsize=9)
ax.set_ylabel("angular_min  (most negative cosine)", fontsize=9)
ax.set_title("Curvature Strength vs. Direction Reversal\\n(green ring = GT annotation available)", fontsize=9)
ax.legend(fontsize=6, ncol=1, loc="upper left", framealpha=0.8)

# bar: commit_step by sample
ax = axes[1]
commit_vals = [_commit_step_fn(r["feats"]["dissolve_by_step"]) for r in records]
ax.bar(range(len(records)), commit_vals,
       color=[r["color"] for r in records], alpha=0.85)
ax.set_xticks(range(len(records)))
ax.set_xticklabels(
    [r["short_id"].replace("[C", "C").split("]")[0] + "]" for r in records],
    rotation=45, ha="right", fontsize=7
)
ax.set_ylabel("commit step τ", fontsize=9)
ax.set_title("Denoising Step of Curvature-Peak Stabilisation\\n(lower = model commits earlier)", fontsize=9)
ax.set_ylim(0, S_STEPS)
ax.axhline(S_STEPS/2, color="gray", lw=0.8, ls=":")

plt.tight_layout()
plt.show()
"""

NB4_FINAL_MD = """\
---
## Final View — Video + Velocity Heatmap Per Sample

Each row: the generated video alongside its `pred_mag` heatmap (velocity magnitude τ × p).

**How to cross-validate:**
1. Play the video and note the second where the visual transition occurs.
2. Convert: `p ≈ floor(time_s × 24 / 8)`.
3. Check if the **green GT line** aligns with the bright vertical stripe in the heatmap.

**If GT line ≈ bright stripe** → `pred_mag` and ground-truth agree.  
**If they disagree** → possible causes: gradual blend (not a hard cut), GT annotation
is at a perceptually ambiguous point, or the model placed the transition differently
from what was perceptually expected.
"""

NB4_FINAL = """\
import io

all_html = []
all_html.append(
    '<h3 style="font-family:sans-serif">Final View — Video + pred_mag heatmap (per sample)</h3>'
    '<p style="font-family:sans-serif;font-size:12px">'
    'Each row: 🎬 generated video  |  pred_mag (τ × p) heatmap.<br>'
    '🟢 solid=GT annotation  |  cyan=start-clip  magenta=end-clip</p>'
)

for r in records:
    f    = r["feats"]
    pm   = f["pred_mag"]
    col  = r["color"]
    sid  = r["sample_id"]
    vmax = float(np.percentile(pm, 99))
    gs   = gt_dict.get(sid)
    gp   = gt_latent(sid, gt_dict)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    im = ax.imshow(pm, aspect="auto", cmap="plasma", vmin=0, vmax=vmax,
                   origin="upper", interpolation="nearest",
                   extent=[-0.5, f["F"]-0.5, f["S"]-0.5, -0.5])
    ax.axvline(K_LAT   - 0.5, color="#00BCD4", lw=1.2, ls="--", alpha=0.8)
    ax.axvline(END_IDX - 0.5, color="#E91E63", lw=1.2, ls="--", alpha=0.8)
    if gp is not None:
        ax.axvline(gp, color="#4CAF50", lw=2.2, ls="-",
                   label=f"GT {gs:.2f}s (p≈{gp:.1f})", alpha=0.9)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.set_xlabel("frame p", fontsize=8)
    ax.set_ylabel("step τ", fontsize=8)
    ax.set_xticks(range(f["F"]))
    ax.tick_params(labelsize=6.5)
    strength    = float(f["curvature_z0"].max() / (f["curvature_z0"].mean() + 1e-8))
    angular_min = float(f["angular_z0"].min())
    ax.set_title(
        f"pred_mag  [strength={strength:.1f}×  angular_min={angular_min:.3f}]",
        fontsize=8.5
    )
    plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = __import__("base64").b64encode(buf.read()).decode()

    vid_h  = video_html(r["video_path"], width=340, label=r["short_id"], cls=r["cls"])
    plot_h = f'<img src="data:image/png;base64,{img_b64}" style="height:215px;vertical-align:top">'
    all_html.append(
        f'<div style="display:flex;align-items:flex-start;gap:12px;'
        f'margin-bottom:12px;border-bottom:1px solid #ddd;padding-bottom:10px">'
        f'{vid_h}{plot_h}</div>'
    )

display(HTML("\\n".join(all_html)))
"""

NB4_NEXTSTEPS = """\
---
## Next Steps & Research Directions

### Transition time control (highest ROI)
1. **`end_idx` ablation** — Change `end_clip_index` in the config to `{6, 8, 10, 12}`.  
   Hypothesis: the velocity stripe in `pred_mag` shifts with it.
2. **`z_t` mid-trajectory intervention** — Run Stage-1 to the commit step τ*.  
   Then blend `z_t[:,:,p_target,:,:]` toward a shifted position. Continue denoising.  
   Hypothesis: transition shifts by the same amount.

### Style control (medium ROI)
3. **Guidance scale schedule** — Instead of constant `guidance_scale=4.0`, ramp it up
   at the (τ, p) region where `pred_mag` peaks.  
   Hypothesis: amplifies text conditioning exactly where the model generates the transition.
4. **Hidden-state probing** — Train a linear probe to predict "frame is in transition region"
   from Level 3 hidden states.  
   This quantifies which transformer block encodes transition structure.

### Immediate validation with this notebook
- Does `strength` correlate with semantic class? (summary table + scatter)
- Does `commit_step` differ between easy (class 1) and hard (class 8) transitions?
- In the PCA plots: does the GT frame appear as a trajectory kink or cluster boundary?
- In the similarity matrix: do class 6/8 show sharper block structure than class 1?
- Cross-validate: play each video and note the perceptual cut time. Does it match `gt_s`?
"""

NB4_CELLS = [
    md(NB4_TITLE),
    code(SETUP),
    code(NB4_TABLE),
    md(NB4_SCATTER_MD),
    code(NB4_SCATTER),
    md(NB4_FINAL_MD),
    code(NB4_FINAL),
    md(NB4_NEXTSTEPS),
]

save(nb(NB4_CELLS), "exp021_04_summary.ipynb")

print("\nAll notebooks generated successfully.")
