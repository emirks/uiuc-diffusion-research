import sys, glob, numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
sys.path.insert(0,"misc/2026-09-08_collapse_remeasure")
from instrument import load_matrix, window_indices
cs=pd.read_csv("misc/2026-08-24_lerp_collapse/distance_vs_cut/cutstats.csv")
cs["abrupt"]=(cs.step_share>0.06)&(cs.step_over_gap>0.56)
pc=pd.read_csv("misc/2026-09-08_collapse_remeasure/per_clip.csv").drop_duplicates("md5")[["md5","path"]]
def prof(path,two_sided):
    M=load_matrix(path); T=M.shape[0]; a,b,_=window_indices(T,9,8,two_sided)
    gap=np.linalg.norm(M[b]-M[a])+1e-8; dA=np.linalg.norm(M-M[a],axis=1)/gap; dB=np.linalg.norm(M-M[b],axis=1)/gap
    i=dA[a:b+1]; transit=int(((i>0.15)&(i<0.85)).sum())
    jump3=float(max(i[t+3]-i[t] for t in range(len(i)-3))); jump1=float(np.max(np.diff(i)))
    # frames near A, near B, elsewhere (in gap units)
    nearA=float((dA[a+1:b]<0.15).mean()); nearB=float((dB[a+1:b]<0.15).mean())
    return dict(gap=float(gap),transit_frames=transit,jump3=jump3,jump1=jump1,fracA=nearA,fracB=nearB,fracElse=1-nearA-nearB)
rows=[]
sel=cs[(cs.variant.isin(["02_neutral__dai","04_neutral_v3__dai","01_effect__dai","06_effect_v3__dai"]))&(cs.two_sided==True)&(~cs.foreign.astype(bool))]
for _,r in sel.iterrows():
    p=pc[pc.md5==r.md5].path.iloc[0]; rows.append(dict(group=f"both/{r.prompt}",endpoint=r.endpoint,seed=r.seed,abrupt=r.abrupt,cls=r.cls,**prof(p,True)))
for _,r in cs[(cs.variant=="tier2_start__dai")&(~cs.foreign.astype(bool))].iterrows():
    p=pc[pc.md5==r.md5].path.iloc[0]; rows.append(dict(group="start(regen)/neutral",endpoint=r.endpoint,seed=r.seed,abrupt=r.abrupt,cls=r.cls,**prof(p,False)))
eps=sorted(set(sel.endpoint))
for ep in eps:
    g=glob.glob(f"data/processed/transitions_std121/**/{ep}.mp4",recursive=True)
    if g: rows.append(dict(group="GT clip (same endpoints)",endpoint=ep,seed=-1,abrupt=np.nan,cls="GT",**prof(g[0],True)))
df=pd.DataFrame(rows); pd.set_option("display.width",250)
g=df.groupby("group")
print(pd.DataFrame({"n":g.size(),"endpoints":g.endpoint.nunique(),"transit_frames_med":g.transit_frames.median(),"transit<=3":g.transit_frames.apply(lambda x:f"{100*(x<=3).mean():.0f}%"),"jump3_med":g.jump3.median().round(2),"jump3>=0.7":g.jump3.apply(lambda x:f"{100*(x>=0.7).mean():.0f}%"),"jump1_med":g.jump1.median().round(2),"fracA":g.fracA.median().round(2),"fracB":g.fracB.median().round(2),"fracElse":g.fracElse.median().round(2)}).to_string())
print("\nGT clips found:",(df.group.str.startswith("GT")).sum(),"of",len(eps))
print("\nper-clip both/neutral: transit_frames, jump3, fracA/fracB/fracElse")
print(df[df.group=="both/neutral"].sort_values("jump3")[["endpoint","seed","cls","abrupt","transit_frames","jump3","jump1","fracA","fracB","fracElse"]].round(2).to_string(index=False))
df.to_csv("misc/2026-08-24_lerp_collapse/distance_vs_cut/transit_profiles.csv",index=False)
