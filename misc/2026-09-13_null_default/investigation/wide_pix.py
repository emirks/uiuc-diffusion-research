"""Per-frame PIX off-line distance + progress Gini for: ALL unique clean 121-f generations in the Sep-08 re-measure table
(base_cond, dualforce_control, dcg w1..w6; v2+v3; neutral/effect; both/start) + ALL 620 one-sided real clips (own last frame
as end anchor, like start-only gens). Two-sided reals (73), probe R1/R2/R3 already in gt_all_pix.csv. Serial decode, incremental save."""
import sys, json, os, numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
sys.path.insert(0,"misc/2026-09-08_collapse_remeasure"); from instrument import load_matrix
OUT="misc/2026-09-13_null_default/investigation/wide_pix.csv"
done=set(pd.read_csv(OUT).clip_id) if os.path.exists(OUT) else set()
def gini(x):
    x=np.sort(np.abs(x)); n=len(x); s=x.sum(); return 0.0 if s==0 else (2*np.sum(np.arange(1,n+1)*x)/(n*s)-(n+1)/n)
def meas(path,a_i,b_i):
    M=load_matrix(path)
    if M is None or M.shape[0]<=b_i: return None
    a,b=M[a_i],M[b_i]; u=b-a; g2=float(u@u)+1e-12; X=M[a_i:b_i+1]; tau=((X-a)@u)/g2; rho=np.linalg.norm((X-a)-tau[:,None]*u,axis=1)/np.sqrt(g2)
    return dict(gini=gini(np.diff(tau)),offline=float(rho[1:-1].mean()),DR=float(np.median(rho[1:-1])),gap_rel=float(np.sqrt(g2)/(0.5*(np.linalg.norm(a)+np.linalg.norm(b))+1e-8)))
jobs=[]
df=pd.read_csv("misc/2026-09-08_collapse_remeasure/per_clip.csv",keep_default_na=False,low_memory=False)
df=df[(df.error=="")].drop_duplicates("md5"); df=df[(df.foreign.astype(str).str.lower()!="true")&(df.static.astype(str).str.lower()!="true")&(df["T"].astype(float)==121)]
for _,r in df.iterrows():
    jobs.append(dict(clip_id=f"{r.arm}|{r.variant}|{r.md5[:10]}",group=f"GEN:{r.arm}:{r.grid}:{r.prompt}:{r.condition}",cls=r.endpoint,path=r.path,a_idx=int(float(r.a_idx)),b_idx=int(float(r.b_idx))))
m=json.load(open("data/processed/transitions_std121/corpus_manifest.json"))["clips"]
for k,v in m.items():
    if "/onesided_" in v["source"]: jobs.append(dict(clip_id="GT1__"+k.split("/")[-1][:-4],group="REAL:one-sided",cls=v["class"],path=f"data/processed/transitions_std121/{k}",a_idx=8,b_idx=120))
print("jobs",len(jobs),"already",len(done),flush=True)
rows=[]; n=0
for j in jobs:
    if j["clip_id"] in done: continue
    r=meas(j["path"],j["a_idx"],j["b_idx"])
    if r: rows.append({**j,**r})
    n+=1
    if n%100==0:
        pd.concat([pd.read_csv(OUT)] if os.path.exists(OUT) else []+[pd.DataFrame(rows)]).to_csv(OUT,index=False) if False else None
        prev=pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(); pd.concat([prev,pd.DataFrame(rows)],ignore_index=True).to_csv(OUT,index=False); rows=[]; print("..",n,flush=True)
prev=pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(); pd.concat([prev,pd.DataFrame(rows)],ignore_index=True).to_csv(OUT,index=False)
print("done",len(pd.read_csv(OUT)),flush=True)
