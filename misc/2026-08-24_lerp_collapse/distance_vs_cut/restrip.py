import numpy as np, pandas as pd, cv2, os
cv2.setNumThreads(1)
cs=pd.read_csv("misc/2026-08-24_lerp_collapse/distance_vs_cut/cutstats.csv")
pc=pd.read_csv("misc/2026-09-08_collapse_remeasure/per_clip.csv").drop_duplicates("md5")[["md5","path"]]
def frames(path):
    cap=cv2.VideoCapture(path); fr=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        fr.append(f)
    return fr
for ep,seed in [("hero_flight_5",42),("display_transition_1",42),("raven_transition_2",42),("shadow_smoke_2",42),("flame_transition_2",42),("earth_wave_5",42)]:
    for v in ("02_neutral__dai","04_neutral_v3__dai","tier2_start__dai"):
        q=cs[(cs.variant==v)&(cs.endpoint==ep)&(cs.seed==seed)&((cs.two_sided==True)|(v=="tier2_start__dai"))]
        if len(q)==0: continue
        q=q.iloc[0]; path=pc[pc.md5==q.md5].path.iloc[0]; fr=frames(path)
        idx=np.linspace(0,len(fr)-1,14).astype(int); tiles=[cv2.resize(fr[i],(200,150)) for i in idx]
        for t,i in zip(tiles,idx): cv2.putText(t,str(i),(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        img=np.hstack(tiles); tag='both' if q.two_sided else 'start'
        cv2.putText(img,f"{ep} s{seed} {tag} {q.grid} cls={q.cls} DR={q.DR_med:.2f} step/gap={q.step_over_gap:.2f} pos={q.step_pos:.2f}",(4,146),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        out=f"misc/2026-08-24_lerp_collapse/distance_vs_cut/strips/{ep}_s{seed}_{tag}_{q.grid}.png"; cv2.imwrite(out,img); print(out, len(fr), os.path.basename(path))
