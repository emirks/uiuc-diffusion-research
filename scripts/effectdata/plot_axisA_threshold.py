#!/usr/bin/env python3
"""Axis-A degree-threshold efficiency curve for EffectData.

Keep only endpoints with counterfactual degree >= M; plot, vs M, the % of DATA
(clips) kept and the % of AXIS A (counterfactual pairs) kept. Same % axis, so the
gap between the curves = counterfactuality retained per unit data. Self-contained
(reads data/raw/effectdata/annotations.json). Writes axisA_threshold.png into the
dataset dir."""
import json
from pathlib import Path
from collections import defaultdict
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "effectdata"
SURF="#fcfcfb"; INK="#0b0b0b"; SEC="#52514e"; MUTE="#898781"
GRID="#e1e0d9"; BASE="#c3c2b7"; BLUE="#2a78d6"; ORANGE="#eb6834"; FILL="#cde2fb"

TAGS={"F","M","Z"}
ann=json.load(open(DATA/"annotations.json"))
subj_eff=defaultdict(set); subj_clips=defaultdict(int)
for fn,rec in ann.items():
    eff=rec["video_path"].split("/")[0]; rest=fn[:-4][len(eff)+1:]
    p=rest.rsplit(",",1); sid=p[0] if (len(p)==2 and p[1] in TAGS) else rest
    subj_eff[sid].add(eff); subj_clips[sid]+=1
degs=[(len(e),subj_clips[s]) for s,e in subj_eff.items()]
TOT_A=sum(comb(d,2) for d,_ in degs); TOT_CL=sum(c for _,c in degs)

XMAX=20
Ms=list(range(1,XMAX+1))
data_pct=[100*sum(c for d,c in degs if d>=M)/TOT_CL for M in Ms]
axisA_pct=[100*sum(comb(d,2) for d,_ in degs if d>=M)/TOT_A for M in Ms]

fig,ax=plt.subplots(figsize=(10,5.4),dpi=160)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.fill_between(Ms,data_pct,axisA_pct,color=FILL,alpha=0.55,zorder=1,
                label="counterfactuality kept beyond data cost")
ax.plot(Ms,axisA_pct,color=BLUE,lw=2,marker="o",ms=5,zorder=4,label="Axis A (counterfactual) retained")
ax.plot(Ms,data_pct,color=ORANGE,lw=2,marker="o",ms=5,zorder=3,label="data (clips) retained")

ax.set_ylim(0,103); ax.set_xlim(0.6,XMAX+0.4)
ax.set_xticks(Ms)
ax.set_yticks(range(0,101,20)); ax.set_yticklabels([f"{v}%" for v in range(0,101,20)])
ax.grid(axis="y",color=GRID,linewidth=0.8,zorder=0); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(BASE)
ax.tick_params(colors=MUTE,labelsize=9,length=0)

# annotate the free win at M>=2 and a sweet spot
ax.annotate("M≥2: drop singletons →\n100% Axis A at 79% data (free)",
            xy=(2,100),xytext=(3.2,58),fontsize=8.5,color=SEC,va="center",
            arrowprops=dict(arrowstyle="->",color=MUTE,lw=1))
ax.annotate("M≥4: 89% Axis A\nfor 40% of the data",
            xy=(4,axisA_pct[3]),xytext=(8.5,90),fontsize=8.5,color=SEC,va="center",
            arrowprops=dict(arrowstyle="->",color=MUTE,lw=1))

fig.subplots_adjust(top=0.80,left=0.075,right=0.975,bottom=0.135)
fig.text(0.075,0.95,"EffectData — Axis A efficiency vs degree threshold",
         fontsize=14,color=INK,fontweight="bold",ha="left",va="top")
fig.text(0.075,0.885,"keep only endpoints whose counterfactual degree ≥ M   ·   "
         "the blue–orange gap = counterfactuality kept per unit data",
         fontsize=9.5,color=SEC,ha="left",va="top")
ax.set_xlabel("M  —  counterfactual-degree threshold (operators sharing one start frame)",fontsize=10,color=SEC)
ax.set_ylabel("% retained",fontsize=10,color=SEC)
leg=ax.legend(loc="upper right",frameon=False,fontsize=9,labelcolor=SEC)

out=DATA/"axisA_threshold.png"
fig.savefig(out,facecolor=SURF,bbox_inches="tight")
print(f"saved {out}")
