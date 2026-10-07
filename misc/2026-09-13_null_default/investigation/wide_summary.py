"""Three populations on two fit-free pixel numbers (off-line distance = distance from the lerp family; progress Gini):
REAL (two-sided 73 + one-sided 620), GENERATED (all clean base/dualforce/dcg gens, v2+v3, any text, both/start), R3 (probe null).
Also per-arm rows and pairwise MWU AUCs."""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
I="misc/2026-09-13_null_default/investigation/"
w=pd.read_csv(I+"wide_pix.csv"); g=pd.read_csv(I+"gt_all_pix.csv")
real2=g[g.group=="GT:all-two-sided"].assign(pop="REAL two-sided"); real1=w[w.group=="REAL:one-sided"].assign(pop="REAL one-sided")
gen=w[w.group.str.startswith("GEN:")].assign(pop="GENERATED")
r3=g[g.group=="S-PROBE:R3:high"].assign(pop="R3 (null, scene change)")
r3i=g[g.group=="S-PROBE:R3:inplace"].assign(pop="R3 (null, in-place)")
allx=pd.concat([real2,real1,gen,r3,r3i],ignore_index=True)
def q(x): return f"{np.median(x):.2f} [{np.percentile(x,25):.2f}, {np.percentile(x,75):.2f}]"
L=["## Three populations (pixel, per frame)\n","| population | n | off-line distance | progress Gini |","|---|---|---|---|"]
for p in ["REAL two-sided","REAL one-sided","GENERATED","R3 (null, scene change)","R3 (null, in-place)"]:
    x=allx[allx["pop"]==p]; L.append(f"| {p} | {len(x)} | {q(x.offline)} | {q(x.gini)} |")
L+=["\n## Generated, by arm / grid / text / anchors (pixel)\n","| arm | grid | text | anchors | n | off-line distance | progress Gini |","|---|---|---|---|---|---|---|"]
for grp,x in gen.groupby("group",sort=True):
    _,arm,grid,prompt,cond=grp.split(":"); L.append(f"| {arm} | {grid} | {prompt} | {cond} | {len(x)} | {q(x.offline)} | {q(x.gini)} |")
def auc(a,b,alt): u=mannwhitneyu(a,b,alternative=alt); return f"{u.statistic/(len(a)*len(b)):.3f} (p {u.pvalue:.0e})"
real=pd.concat([real2,real1])
L+=["\n## Separation (probability that a random clip of the first population beats a random clip of the second)\n","| comparison | off-line: first lower | Gini: first higher |","|---|---|---|"]
for a,b,na,nb in [(gen,real,"GENERATED","REAL all"),(r3,real,"R3 scene-change","REAL all"),(r3,gen,"R3 scene-change","GENERATED"),(gen,real2,"GENERATED","REAL two-sided"),(r3,real2,"R3 scene-change","REAL two-sided")]:
    L.append(f"| {na} vs {nb} | {auc(a.offline,b.offline,'less')} | {auc(a.gini,b.gini,'greater')} |")
# both-anchor generated only (the CTT setting) vs real two-sided
gb=gen[gen.group.str.endswith(":both")]
L.append(f"| GENERATED both-anchor only ({len(gb)}) vs REAL two-sided | {auc(gb.offline,real2.offline,'less')} | {auc(gb.gini,real2.gini,'greater')} |")
txt="\n".join(L); print(txt); open(I+"WIDE_TABLE.md","w").write("# Reals vs generateds vs R3 — pixel, per frame, no thresholds\n\nOff-line distance = mean distance of interior frames from the line through the two anchor frames (gap units; 0 = a member of the lerp family). Progress Gini = concentration of per-frame progress along that line (0 = cross-fade, 1 = cut). Windows: both-anchor rows a=8,b=113; start-only rows and one-sided reals a=8,b=120 (own last frame). GENERATED = every unique, clean, 121-f clip of base_cond / dualforce_control / dualforce_dcg (v2+v3, neutral+effect, both+start) from the Sep-08 re-measure table. Medians [IQR].\n\n"+txt+"\n")
