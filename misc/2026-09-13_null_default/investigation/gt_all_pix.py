"""Per-frame PIX measures (progress Gini, off-line distance, DR, fitted change duration) for ALL real two-sided
transition clips (union: grid two-sided classes + twosided_transitions folder classes) and for the base groups,
all decoded identically (instrument.load_matrix, 128 px, blur 1, serial). Output: gt_all_pix.csv / .md"""
import json, glob, sys, collections, numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
sys.path.insert(0,"misc/2026-09-08_collapse_remeasure"); from instrument import load_matrix
sys.path.insert(0,"misc/2026-09-13_null_default/scripts"); import common as C
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
print("GT two-sided union:",len(gt),"clips in",len(two_cls),"classes",flush=True)
def gini(x):
    x=np.sort(np.abs(x)); n=len(x); s=x.sum(); return 0.0 if s==0 else (2*np.sum(np.arange(1,n+1)*x)/(n*s)-(n+1)/n)
S0=np.linspace(0,1,201); D=np.array([0.5,1,2,3,5,8,12,16,24,32,48,64,80,104])/104
def fitd(s,tau):
    fam=np.clip((s[None,None,:]-S0[:,None,None])/D[None,:,None],0,1); err=np.sqrt(((fam-tau[None,None,:])**2).mean(-1)); i,j=np.unravel_index(err.argmin(),err.shape); return D[j]*104, err[i,j]
def meas(path,a_i,b_i):
    M=load_matrix(path)
    if M is None or M.shape[0]<=b_i: return None
    a,b=M[a_i],M[b_i]; u=b-a; g2=float(u@u)+1e-12; X=M[a_i:b_i+1]; tau=((X-a)@u)/g2
    rho=np.linalg.norm((X-a)-tau[:,None]*u,axis=1)/np.sqrt(g2); s=np.linspace(0,1,len(tau)); d,e=fitd(s,tau)
    return dict(gini=gini(np.diff(tau)),offline=float(rho[1:-1].mean()),DR=float(np.median(rho[1:-1])),d_frames=d,fit_rmse=e,lerp_rmse=float(np.sqrt(((tau-s)**2).mean())),gap_rel=float(np.sqrt(g2)/(0.5*(np.linalg.norm(a)+np.linalg.norm(b))+1e-8)))
rows=[]
for k,cls in gt:
    r=meas(f"data/processed/transitions_std121/{k}",8,113)
    if r: rows.append(dict(group="GT:all-two-sided",cls=cls,clip_id="GT__"+k.split("/")[-1][:-4],**r))
for g in ["S-GRID:NULLGEN","S-GRID-F:NULLGEN","S-PROBE:R3:high","S-PROBE:R1:high","S-PROBE:R2:high","S-PROBE:R3:inplace"]:
    for _,x in man[man.g==g].iterrows():
        r=meas(str(C.REPO/x.path),int(x.a_idx),int(x.b_idx))
        if r: rows.append(dict(group=g,cls=x.endpoint,clip_id=x.clip_id,**r))
    print("done",g,flush=True)
df=pd.DataFrame(rows); df.to_csv("misc/2026-09-13_null_default/investigation/gt_all_pix.csv",index=False)
def q(x,f=2): return f"{np.median(x):.{f}f} [{np.percentile(x,25):.{f}f}, {np.percentile(x,75):.{f}f}]"
out=["| group | n | progress Gini | off-line distance | DR | change duration (frames) |","|---|---|---|---|---|---|"]
for g,x in df.groupby("group",sort=False):
    out.append(f"| {g} | {len(x)} | {q(x.gini)} | {q(x.offline)} | {q(x.DR)} | {q(x.d_frames,0)} |")
open("misc/2026-09-13_null_default/investigation/gt_all_pix.md","w").write("# Per-frame PIX measures, ALL two-sided real clips vs base groups (same decoder)\n\n"+"\n".join(out)+"\n"); print("\n".join(out))
