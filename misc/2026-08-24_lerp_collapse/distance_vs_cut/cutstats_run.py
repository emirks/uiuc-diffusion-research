import sys, os, time, pandas as pd, numpy as np
sys.path.insert(0,"misc/2026-09-08_collapse_remeasure")
from instrument import load_matrix, window_indices
pc=pd.read_csv("misc/2026-09-08_collapse_remeasure/per_clip.csv")
sel=pc[(pc.arm=="base_cond")].drop_duplicates("md5")
sel=pd.concat([sel[(sel.two_sided==True)|(sel.variant=="tier2_start__dai")], sel[~((sel.two_sided==True)|(sel.variant=="tier2_start__dai"))]])
out="/u/emirkisa/.claude-lab/jobs/9505994b/tmp/cutstats.csv"
prev=pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
done=set(prev.md5) if len(prev) else set()
def cutstats(M, two_sided):
    T=M.shape[0]; a,b,_=window_indices(T,9,8,two_sided)
    seg=M[a:b+1]; d=np.linalg.norm(np.diff(seg,axis=0),axis=1); path=d.sum()+1e-8
    gap=np.linalg.norm(M[b]-M[a])+1e-8
    return dict(step_share=float(d.max()/path), step_over_gap=float(d.max()/gap), step_pos=float(d.argmax()/len(d)), n_steps_big=int((d>0.5*d.max()).sum()))
rows=[]; t0=time.time()
def save():
    global prev
    if rows:
        prev=pd.concat([prev,pd.DataFrame(rows)],ignore_index=True); prev.to_csv(out,index=False); rows.clear()
for _,r in sel.iterrows():
    if r.md5 in done: continue
    p=r.path if os.path.isabs(r.path) else os.path.join(os.getcwd(),r.path)
    try:
        M=load_matrix(p); s=cutstats(M, bool(r.two_sided))
    except Exception as e:
        s=dict(step_share=np.nan,step_over_gap=np.nan,step_pos=np.nan,n_steps_big=-1,err=str(e)[:80])
    s.update(md5=r.md5, variant=r.variant, endpoint=r.endpoint, seed=r.seed, two_sided=r.two_sided, sided=r.sided, foreign=r.foreign, cls=r.cls, DR_med=r.DR_med, M=r.M, prompt=r.prompt, grid=r.grid)
    rows.append(s)
    if len(rows)>=100: save()
    if time.time()-t0>570: break
save()
print("scored",len(prev),"of",len(sel),"elapsed",round(time.time()-t0))
