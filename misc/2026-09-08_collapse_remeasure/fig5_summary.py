"""Compact summary figure: confinement residual DR per arm x prompt x conditioning (grid v3) + guidance sweep (v2)."""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{HERE}/per_clip.csv"); df = df[df["error"].isna()] if "error" in df else df
df["is_dup"] = df.groupby(["arm", "variant", "md5"]).cumcount() > 0
u = df[(~df.is_dup) & (~df.foreign.astype(bool)) & (~df.static.astype(bool)) & df.DR_med.notna()]
t2 = u[u.variant == "tier2_start__dai"]; u = u[u.variant != "tier2_start__dai"]
cov = pd.read_csv(f"{HERE}/endpoint_covariates.csv"); gt = cov.gt_DR_med.dropna()
REAL = (gt.quantile(.25), gt.median(), gt.quantile(.75))

fig = plt.figure(figsize=(12.5, 5.6)); gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1, 1.05], wspace=0.08)
ax = fig.add_subplot(gs[0, :2]); axw = fig.add_subplot(gs[0, 2])

rows = [("base_cond", "neutral", "base LTX-2\nneutral prompt"), ("base_cond", "effect", "base LTX-2\ntransition described in prompt"),
        ("dualforce_control", "neutral", "trained adapter\nneutral prompt"), ("dualforce_control", "effect", "trained adapter\ntransition in prompt"),
        ("dualforce_dcg_w6", "neutral", "trained + guidance w=6\nneutral prompt"), ("dualforce_dcg_w6", "effect", "trained + guidance w=6\ntransition in prompt")]
COL = {"both": "#c0392b", "start": "#7f8c8d"}; OFF = {"both": +0.17, "start": -0.17}; MK = {"both": "o", "start": "^"}
rng = np.random.default_rng(0)
g3 = u[u.grid == "v3"]
for i, (arm, prompt, label) in enumerate(rows):
    y0 = len(rows) - 1 - i
    for cond in ["both", "start"]:
        s = g3[(g3.arm == arm) & (g3.prompt == prompt) & (g3.condition == cond)].DR_med.values
        if len(s) == 0: continue
        y = y0 + OFF[cond]
        ax.scatter(s, y + rng.uniform(-0.07, 0.07, len(s)), s=6, color=COL[cond], alpha=0.18, edgecolors="none", zorder=1)
        q25, q50, q75 = np.percentile(s, [25, 50, 75])
        ax.plot([q25, q75], [y, y], color=COL[cond], lw=3, solid_capstyle="butt", zorder=3)
        ax.scatter([q50], [y], s=70, marker=MK[cond], color=COL[cond], edgecolors="white", linewidths=1.2, zorder=4)
        ax.text(min(q25, 0.985) - 0.012 if False else q75 + 0.02, y, f"{q50:.2f}  (n={len(s)})", va="center", fontsize=7.5, color=COL[cond], alpha=0.9)
# paired end-anchor probe: base neutral both -> start-only regen (v2 rows)
b = u[(u.grid == "v2") & (u.arm == "base_cond") & (u.prompt == "neutral") & (u.condition == "both")].DR_med.median(); r = t2.DR_med.median()
yb = len(rows) - 1 + OFF["both"]
ax.annotate("", xy=(r, yb + 0.36), xytext=(b, yb + 0.36), arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
ax.text(r + 0.02, yb + 0.36, "same 30 clips regenerated\nwith the end anchor dropped", ha="left", va="center", fontsize=7.5)

ax.axvline(0, color=COL["both"], lw=1, ls="--"); ax.text(0.012, -0.55, "null family: frame-wise\nblend of the two anchors", fontsize=7.5, color=COL["both"], va="center")
ax.axvspan(REAL[0], REAL[2], color="#27ae60", alpha=0.10, lw=0); ax.axvline(REAL[1], color="#27ae60", lw=1)
ax.text(REAL[2] + 0.012, -0.55, f"real transitions between the same\nendpoints: median {REAL[1]:.2f}, band = IQR", fontsize=7.5, color="#1e8449", va="center")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows][::-1], fontsize=9)
ax.set_xlim(-0.02, 1.25); ax.set_ylim(-0.9, len(rows) - 0.25)
ax.set_xlabel("confinement residual DR  (0 = on the endpoint line; dot = median, bar = IQR, dots = clips)")
ax.grid(axis="x", alpha=.3); ax.set_title("Grid v3, unique clean clips, foreign endpoints excluded", fontsize=10, loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color=COL["both"], lw=3, label="both anchors given"),
                   Line2D([], [], marker="^", color=COL["start"], lw=3, label="start anchor only")], loc="upper right", fontsize=8.5, frameon=True)

# guidance sweep (v2 neutral, seed 42)
sw = u[(u.grid == "v2") & (u.prompt == "neutral") & (u.seed.astype(str) == "42")]
wmap = {"dualforce_control": 0, "dualforce_dcg_w1": 1, "dualforce_dcg_w1p5": 1.5, "dualforce_dcg_w3": 3, "dualforce_dcg_w6": 6}
for cond in ["both", "start"]:
    xs, med, lo, hi = [], [], [], []
    for arm, w in sorted(wmap.items(), key=lambda kv: kv[1]):
        s = sw[(sw.arm == arm) & (sw.condition == cond)].DR_med.values
        if len(s) == 0: continue
        q = np.percentile(s, [25, 50, 75]); xs.append(w); med.append(q[1]); lo.append(q[1] - q[0]); hi.append(q[2] - q[1])
    axw.errorbar(xs, med, yerr=[lo, hi], fmt=MK[cond] + "-", color=COL[cond], capsize=3, lw=1.5, ms=6, label="both anchors" if cond == "both" else "start only")
axw.axhspan(REAL[0], REAL[2], color="#27ae60", alpha=0.10, lw=0); axw.axhline(REAL[1], color="#27ae60", lw=1)
axw.axhline(0, color=COL["both"], lw=1, ls="--")
axw.set_xticks([0, 1, 1.5, 3, 6]); axw.set_xticklabels(["none", "1", "1.5", "3", "6"]); axw.set_xlabel("null-operator guidance weight w")
axw.set_ylim(-0.02, 1.25); axw.yaxis.tick_right(); axw.set_ylabel("DR (median, IQR)", rotation=270, labelpad=14); axw.yaxis.set_label_position("right")
axw.grid(alpha=.3); axw.legend(fontsize=8, loc="upper left"); axw.set_title("Trained model, grid v2 neutral, seed 42", fontsize=10, loc="left")
fig.suptitle("Distance of the generated middle from the endpoint line (confinement residual DR)", fontsize=11, y=0.995)
fig.subplots_adjust(left=0.19, right=0.93, top=0.88, bottom=0.12); plt.savefig(f"{HERE}/figs/fig5_summary.png", dpi=160); print("ok")
