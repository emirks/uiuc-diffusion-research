import pandas as pd, numpy as np
from scipy.stats import spearmanr, mannwhitneyu
pd.set_option("display.width",200); pd.set_option("display.max_columns",30)
cs=pd.read_csv("/u/emirkisa/.claude-lab/jobs/9505994b/tmp/cutstats.csv")
cov=pd.read_csv("misc/2026-09-08_collapse_remeasure/endpoint_covariates.csv")[["endpoint","endpoint_source","endpoint_class","dino_dist","clip_dist","pixel_gap_rel"]]
cs["abrupt"]=(cs.step_share>0.06)&(cs.step_over_gap>0.56)
cs["cut"]=cs.cls=="CUT"; cs["online"]=cs.DR_med<=0.12
cs["cond"]=np.where(cs.two_sided==True,"both",np.where(cs.variant=="tier2_start__dai","start(regen)","start"))
def rate(df,c): return f"{100*df[c].mean():.0f}% ({int(df[c].sum())}/{len(df)})"
print("=== A. base LTX-2, all unique clips: on-line CUT class vs abrupt-cut detector, by conditioning x prompt (clean = non-foreign)")
for clean in (True,False):
    d=cs[~cs.foreign.astype(bool)] if clean else cs
    print("--- clean" if clean else "--- all cells")
    g=d.groupby(["cond","prompt","grid"])
    print(pd.DataFrame({"n":g.size(),"CUT_class":g.apply(lambda x:rate(x,"cut")),"abrupt":g.apply(lambda x:rate(x,"abrupt")),"abrupt&online":g.apply(lambda x:rate(x[x.online],"abrupt") if x.online.any() else "-"),"abrupt&offline":g.apply(lambda x:rate(x[~x.online],"abrupt")),"DR_med":g.DR_med.median().round(2)}))
print("\n=== B. both-anchor base clips joined to endpoint distance (unique clips; distance = between the two GT anchor frames)")
b=cs[cs.cond=="both"].merge(cov,on="endpoint",how="left")
b["clean"]=~b.foreign.astype(bool)
for metric in ("dino_dist","clip_dist","pixel_gap_rel"):
    print(f"\n--- {metric}: terciles over endpoints")
    ep=b.drop_duplicates("endpoint"); q=ep[metric].quantile([1/3,2/3]).values
    b["terc"]=pd.cut(b[metric],[-1,q[0],q[1],9],labels=["close","mid","far"])
    for clean in (True,False):
        d=b[b.clean] if clean else b
        print("  clean" if clean else "  all", "| edges", np.round(q,2))
        g=d.groupby(["prompt","terc"],observed=True)
        t=pd.DataFrame({"n":g.size(),"endpoints":g.endpoint.nunique(),"CUT_class":g.apply(lambda x:rate(x,"cut")),"abrupt":g.apply(lambda x:rate(x,"abrupt")),"DR_med":g.DR_med.median().round(2),"M_med":g.M.median().round(2)})
        print(t.to_string())
    for pr in ("neutral","effect"):
        d=b[(b.prompt==pr)&b.clean]
        rho,p=spearmanr(d[metric],d.DR_med); 
        mw=mannwhitneyu(d[d.cut][metric],d[~d.cut][metric],alternative="two-sided") if d.cut.sum()>2 else None
        ma=mannwhitneyu(d[d.abrupt][metric],d[~d.abrupt][metric],alternative="two-sided") if d.abrupt.sum()>2 else None
        print(f"  {pr} clean: Spearman(DR,{metric}) rho={rho:.2f} p={p:.3f}; median {metric} CUT vs not: {d[d.cut][metric].median():.2f} vs {d[~d.cut][metric].median():.2f} (MW p={mw.pvalue if mw else float('nan'):.3f}); abrupt vs not: {d[d.abrupt][metric].median():.2f} vs {d[~d.abrupt][metric].median():.2f} (MW p={ma.pvalue if ma else float('nan'):.3f})")
print("\n=== C. where the largest step sits (step_pos, 0=start anchor,1=end anchor) for abrupt both-anchor clean clips, by prompt")
d=b[b.clean&b.abrupt]
print(d.groupby("prompt").step_pos.describe()[["count","25%","50%","75%"]].round(2))
print("share of abrupt clips whose step is inside (0.1,0.9):", round(((d.step_pos>0.1)&(d.step_pos<0.9)).mean(),2))
print("\n=== D. abrupt clips: on-line (CUT class) vs off-line split, both-anchor clean, by prompt")
d=b[b.clean&b.abrupt]; print(d.groupby(["prompt","online"]).size())
print("\n=== E. endpoint list, clean both-anchor neutral: per endpoint CUT and abrupt counts with distances")
d=b[b.clean&(b.prompt=="neutral")]
print(d.groupby(["endpoint","endpoint_class"]).agg(n=("md5","size"),cut=("cut","sum"),abrupt=("abrupt","sum"),DR=("DR_med","median"),dino=("dino_dist","first"),clip=("clip_dist","first"),pix=("pixel_gap_rel","first")).round(2).sort_values("dino").to_string())
b.to_csv("/u/emirkisa/.claude-lab/jobs/9505994b/tmp/both_anchor_with_dist.csv",index=False)
