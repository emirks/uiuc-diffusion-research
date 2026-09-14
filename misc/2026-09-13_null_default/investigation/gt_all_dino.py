"""DINO CLS per frame (CPU) for the two-sided real clips that have no cached features yet, then per-frame DINO
Gini / off-line distance for GT-all vs the base groups (cached). Output gt_all_dino.csv/.md"""
import json, glob, sys, os, numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
sys.path.insert(0,"misc/2026-09-13_null_default/scripts"); import common as C
from extract_features import DinoCLS
import torch; torch.set_num_threads(4)
man=C.load_manifest(); man["g"]=man.stratum+":"+man.group+np.where(man.stratum=="S-PROBE",":"+man.tier,"")
m=json.load(open("data/processed/transitions_std121/corpus_manifest.json"))["clips"]
grid_two=set()
for f in glob.glob("store/gens/005_base_cond/0[24]_neutral*/*.jsonl"):
    for line in open(f):
        try: r=json.loads(line)
        except: continue
        if r.get("sided")=="two" and (r.get("endpoint_class") or r.get("class")): grid_two.add(r.get("endpoint_class") or r.get("class"))
folder_two={v["class"] for v in m.values() if "/twosided_" in v["source"]}
two_cls=(grid_two|folder_two)-{"davis"}
gt=[(k,v["class"]) for k,v in m.items() if v["class"] in two_cls]
outdir=C.LAB/"cache/null_default/features_gt_all"; outdir.mkdir(exist_ok=True)
dino=None
def gini(x):
    x=np.sort(np.abs(x)); n=len(x); s=x.sum(); return 0.0 if s==0 else (2*np.sum(np.arange(1,n+1)*x)/(n*s)-(n+1)/n)
def meas(d,a_i,b_i):
    a,b=d[a_i],d[b_i]; u=b-a; g2=float(u@u)+1e-12; X=d[a_i:b_i+1]; tau=((X-a)@u)/g2; rho=np.linalg.norm((X-a)-tau[:,None]*u,axis=1)/np.sqrt(g2)
    return dict(gini=gini(np.diff(tau)),offline=float(rho[1:-1].mean()),DR=float(np.median(rho[1:-1])))
rows=[]
for i,(k,cls) in enumerate(gt):
    cid="GT__"+k.split("/")[-1][:-4]; p=C.FEATURES/f"{cid}.npz"; p2=outdir/f"{cid}.npz"
    if p.exists(): d=np.load(p)["dino"].astype(np.float32)
    elif p2.exists(): d=np.load(p2)["dino"].astype(np.float32)
    else:
        if dino is None: dino=DinoCLS("cpu",torch.float32)
        fr=C.decode_rgb(C.REPO/f"data/processed/transitions_std121/{k}"); d=dino.embed(fr); np.savez_compressed(p2,dino=d.astype(np.float16),a_idx=8,b_idx=113,T=len(fr))
    rows.append(dict(group="GT:all-two-sided",cls=cls,clip_id=cid,**meas(d,8,113)))
    if (i+1)%10==0: print("gt",i+1,"/",len(gt),flush=True)
for g in ["S-GRID:NULLGEN","S-GRID-F:NULLGEN","S-PROBE:R3:high","S-PROBE:R1:high","S-PROBE:R2:high","S-PROBE:R3:inplace"]:
    for _,x in man[man.g==g].iterrows():
        d=np.load(C.FEATURES/f"{x.clip_id}.npz")["dino"].astype(np.float32); rows.append(dict(group=g,cls=x.endpoint,clip_id=x.clip_id,**meas(d,int(x.a_idx),int(x.b_idx))))
df=pd.DataFrame(rows); df.to_csv("misc/2026-09-13_null_default/investigation/gt_all_dino.csv",index=False)
def q(x): return f"{np.median(x):.2f} [{np.percentile(x,25):.2f}, {np.percentile(x,75):.2f}]"
out=["| group | n | progress Gini (DINO) | off-line distance (DINO) | DR (DINO) |","|---|---|---|---|---|"]
for g,x in df.groupby("group",sort=False): out.append(f"| {g} | {len(x)} | {q(x.gini)} | {q(x.offline)} | {q(x.DR)} |")
open("misc/2026-09-13_null_default/investigation/gt_all_dino.md","w").write("# Per-frame DINO measures, ALL two-sided real clips vs base groups\n\n"+"\n".join(out)+"\n"); print("\n".join(out))
