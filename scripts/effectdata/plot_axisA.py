#!/usr/bin/env python3
"""Axis-A counterfactual-degree distribution for EffectData (one chart).

Self-contained: parses data/raw/effectdata/annotations.json, builds the per-endpoint
operator-degree histogram, and renders it. Deg-1 (not counterfactual) muted gray,
deg>=2 (counterfactual core) series-blue; log count axis. Palette = dataviz reference.
Writes axisA_degree.png into the dataset dir (referenced by counterfactuality.md)."""
import json
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "effectdata"

# palette (light surface)
SURF="#fcfcfb"; INK="#0b0b0b"; SEC="#52514e"; MUTE="#898781"
GRID="#e1e0d9"; BASE="#c3c2b7"; BLUE="#2a78d6"; GRAY="#b8b7b0"

def endpoint_degrees():
    ann = json.load(open(DATA / "annotations.json")); TAGS = {"F", "M", "Z"}
    S2E = defaultdict(set)
    for key, rec in ann.items():
        eff = rec.get("video_path", key).split("/")[0]
        base = key[:-4] if key.lower().endswith(".mp4") else key
        rest = base[len(eff)+1:] if base.startswith(eff+",") else (base.split(",",1)[1] if "," in base else "")
        if not rest:
            continue
        p = rest.rsplit(",", 1)
        sid = p[0] if (len(p) == 2 and p[1] in TAGS) else rest
        S2E[sid].add(eff)
    return [len(v) for v in S2E.values()]

deg = endpoint_degrees()
h = Counter(deg); N = len(deg); maxd = max(deg)
xs = list(range(1, maxd+1)); ys = [h.get(k, 0) for k in xs]
singles = h[1]; cf = N - singles
mean_cf = sum(k*h[k] for k in xs if k >= 2) / cf

fig, ax = plt.subplots(figsize=(10, 5.4), dpi=160)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.bar(xs, ys, width=0.78, color=[GRAY if k == 1 else BLUE for k in xs], zorder=3)
ax.set_yscale("log"); ax.set_ylim(0.7, 50000); ax.set_xlim(0.3, maxd+0.7)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(BASE)
ax.tick_params(colors=MUTE, labelsize=9, length=0)
ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, maxd]))

ax.text(1, singles*1.35, f"{singles:,}\n({100*singles/N:.1f}%)", ha="center", va="bottom",
        fontsize=9, color=SEC, linespacing=1.2, fontweight="bold")
peak = max(range(7, 13), key=lambda k: h.get(k, 0))
ax.annotate('"hero" subjects\nreused across\nmany effects', xy=(peak, h[peak]),
            xytext=(15, 2600), fontsize=8.5, color=SEC, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=MUTE, lw=1))
ax.axvline(mean_cf, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=4)
ax.text(mean_cf+0.25, 20000, f"mean = {mean_cf:.2f}\n(among deg≥2)", fontsize=8.5, color=INK, va="top")

fig.subplots_adjust(top=0.80, left=0.085, right=0.975, bottom=0.135)
fig.text(0.085, 0.95, "EffectData — counterfactual degree per endpoint (Axis A)",
         fontsize=14, color=INK, fontweight="bold", ha="left", va="top")
fig.text(0.085, 0.885, "how many distinct effects share the same start frame   ·   "
         f"{N:,} endpoints   ·   log count axis", fontsize=9.5, color=SEC, ha="left", va="top")
ax.set_xlabel("operators (effects) sharing one start frame  →  counterfactual degree", fontsize=10, color=SEC)
ax.set_ylabel("number of endpoints (log)", fontsize=10, color=SEC)
ax.text(0.985, 0.95, "■ degree 1 — not counterfactual (50%)", transform=ax.transAxes,
        ha="right", va="top", fontsize=8.8, color=GRAY, fontweight="bold")
ax.text(0.985, 0.885, f"■ degree ≥2 — counterfactual core ({cf:,} endpoints)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color=BLUE, fontweight="bold")

out = DATA / "axisA_degree.png"
fig.savefig(out, facecolor=SURF, bbox_inches="tight")
print(f"saved {out}  ({N:,} endpoints, mean_cf {mean_cf:.3f}, singles {singles:,})")
