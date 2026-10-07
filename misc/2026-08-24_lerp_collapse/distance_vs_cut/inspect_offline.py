import sys, os, glob, numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
sys.path.insert(0,"misc/2026-09-08_collapse_remeasure")
from instrument import load_matrix, window_indices
cs=pd.read_csv("misc/2026-08-24_lerp_collapse/distance_vs_cut/cutstats.csv")
cs["abrupt"]=(cs.step_share>0.06)&(cs.step_over_gap>0.56); cs["online"]=cs.DR_med<=0.12
pc=pd.read_csv("misc/2026-09-08_collapse_remeasure/per_clip.csv").drop_duplicates("md5")[["md5","path"]]
def profile(path, two_sided):
    M=load_matrix(path); T=M.shape[0]; a,b,_=window_indices(T,9,8,two_sided)
    A,B=M[a],M[b]; gap=np.linalg.norm(B-A)+1e-8
    dA=np.linalg.norm(M-A,axis=1)/gap; dB=np.linalg.norm(M-B,axis=1)/gap
    d=np.linalg.norm(np.diff(M[a:b+1],axis=0),axis=1); k=int(d.argmax()); t_cut=a+k
    pre=slice(a,t_cut+1); post=slice(t_cut+1,b+1)
    return dict(T=T,a=a,b=b,gap=float(gap),path_over_gap=float(d.sum()/gap),max_step_over_gap=float(d.max()/gap),t_cut=t_cut,
        pre_dA=float(np.median(dA[pre])),pre_dB=float(np.median(dB[pre])),post_dA=float(np.median(dA[post])) if t_cut+1<=b else np.nan,post_dB=float(np.median(dB[post])) if t_cut+1<=b else np.nan,
        suffix_dB=float(np.median(dB[b:])) , dA=dA, dB=dB, d=d)
# 1. the display_transition_1 example
q=cs[(cs.variant=="02_neutral__dai")&(cs.endpoint=="display_transition_1")&(cs.seed==42)&(cs.two_sided==True)].iloc[0]
p=pc[pc.md5==q.md5].path.iloc[0]; pr=profile(p,True)
print("display_transition_1 s42 both:",{k:(round(v,3) if isinstance(v,float) else v) for k,v in pr.items() if k not in("dA","dB","d")})
print("frames 36..64: t, dA, dB, step")
for t in range(36,65,2): print(t, round(pr["dA"][t],2), round(pr["dB"][t],2), round(pr["d"][t-pr["a"]]/pr["gap"],2))
# zoom strip around the cut + GT endpoints
def frames(path):
    cap=cv2.VideoCapture(path); fr=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        fr.append(f)
    return fr
fr=frames(p); tc=pr["t_cut"]
idx=list(range(max(0,tc-6),min(len(fr),tc+8)))
tiles=[cv2.resize(fr[i],(200,150)) for i in idx]
for t,i in zip(tiles,idx): cv2.putText(t,str(i),(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
cv2.imwrite("misc/2026-08-24_lerp_collapse/distance_vs_cut/strips/zoom_display_transition_1_s42_both.png",np.hstack(tiles))
gt=glob.glob("data/processed/transitions_std121/**/display_transition_1.mp4",recursive=True)
if gt:
    g=frames(gt[0]); tiles=[cv2.resize(g[i],(200,150)) for i in [0,8,40,80,112,120]]+[cv2.resize(fr[i],(200,150)) for i in [0,8,112,120]]
    for t,l in zip(tiles,["GT0","GT8","GT40","GT80","GT112","GT120","gen0","gen8","gen112","gen120"]): cv2.putText(t,l,(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    cv2.imwrite("misc/2026-08-24_lerp_collapse/distance_vs_cut/strips/gt_vs_gen_display_transition_1.png",np.hstack(tiles))
# 2. all abrupt both-anchor neutral clean clips: where does the post-cut segment sit?
rows=[]
for _,r in cs[(cs.variant.isin(["02_neutral__dai","04_neutral_v3__dai"]))&(cs.two_sided==True)&(~cs.foreign.astype(bool))].iterrows():
    p=pc[pc.md5==r.md5].path.iloc[0]; pr=profile(p,True)
    rows.append(dict(endpoint=r.endpoint,seed=r.seed,grid=r.grid,abrupt=r.abrupt,online=r.online,cls=r.cls,DR=r.DR_med,**{k:round(v,2) for k,v in pr.items() if k not in("dA","dB","d","T","a","b")}))
df=pd.DataFrame(rows); pd.set_option("display.width",250)
print("\n=== both-anchor neutral clean (v2+v3): pre/post-cut position relative to anchors (dA,dB in gap units; on-line means near the A-B segment)")
print(df.sort_values(["abrupt","online","DR"]).to_string(index=False))
df.to_csv("misc/2026-08-24_lerp_collapse/distance_vs_cut/both_neutral_cut_profiles.csv",index=False)
