"""Reference-swap stability of the appearance-transfer kernel (v3 + v4).

The idea under test (user): if the reference is only an OPERATOR (class-defining),
then a GT middle judged against ANY other same-class reference should score high and
stably -> single/random/mean reference scoring is a valid universal yardstick.
We measure, per class, on the corpus itself (GT as 'generation'):
  same-mean  mean kernel similarity of a clip vs other same-class clips
  sigma_ref  mean per-clip std across choice of same-class reference (the swap noise)
  cross      mean similarity vs all other-class clips
  gap        same-mean - cross (headroom)
  top1       leave-own-out nearest-neighbour class retrieval (sanity vs exam recall)
  rank_rho   mean Spearman agreement of item orderings induced by two different
             single references (would two judges disagree?)
"""
import collections
import itertools
import json
import sys

import numpy as np

R = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/"


def analyze(npz_path, matrix_key, label, trust_key="m1a"):
    z = np.load(npz_path, allow_pickle=True)
    if matrix_key not in z:
        print(f"[{label}] matrix '{matrix_key}' not found; available: {list(z.keys())}")
        return None
    keys = [str(k) for k in z["keys"]]
    D = z[matrix_key].astype(float)
    S = 1.0 - D
    cls = [k.split("/")[0] for k in keys]
    tm = json.load(open(R + "outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json"))
    byc = collections.defaultdict(list)
    for i, c in enumerate(cls):
        byc[c].append(i)
    rows = []
    for c, idx in sorted(byc.items()):
        n = len(idx)
        if n < 4:
            continue
        idx = np.array(idx)
        sub = S[np.ix_(idx, idx)]
        off = sub[~np.eye(n, dtype=bool)]
        sig_ref = np.mean([np.std(np.delete(sub[i], i)) for i in range(n)])
        other = np.setdiff1d(np.arange(len(keys)), idx)
        cross = S[np.ix_(idx, other)].mean()
        r1 = 0
        for i_glob in idx:
            row = S[i_glob].copy()
            row[i_glob] = -9
            r1 += cls[int(np.argmax(row))] == c
        taus = []
        for j, k in itertools.combinations(range(n), 2):
            rest = [i for i in range(n) if i not in (j, k)]
            if len(rest) < 3:
                continue
            a, b = sub[rest, j], sub[rest, k]
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(b))
            taus.append(np.corrcoef(ra, rb)[0, 1])
        rows.append(dict(cls=c, n=n, same=off.mean(), sig=sig_ref, cross=cross,
                         gap=off.mean() - cross, top1=r1 / n,
                         rho=float(np.nanmean(taus)) if taus else float("nan"),
                         trust=tm.get(c, {}).get(trust_key)))
    print(f"\n===== {label} =====")
    print(f"{'class':22s} {'n':>3s} {'same':>6s} {'σ_ref':>6s} {'cross':>6s} {'gap':>6s} {'top1':>5s} {'rankρ':>6s} {'trust':>5s}")
    for r in sorted(rows, key=lambda x: -x["gap"]):
        print(f"{r['cls']:22s} {r['n']:>3d} {r['same']:>6.3f} {r['sig']:>6.3f} {r['cross']:>6.3f} "
              f"{r['gap']:>6.3f} {r['top1']:>5.0%} {r['rho']:>6.2f} {str(r['trust']):>5s}")
    a = np.array([[r["same"], r["sig"], r["gap"], r["rho"], r["top1"]] for r in rows])
    t = np.array([[r["same"], r["sig"], r["gap"], r["rho"], r["top1"]] for r in rows if r["trust"]])
    print(f"ALL   ({len(rows)} classes): same {a[:,0].mean():.3f}  σ_ref {a[:,1].mean():.3f}  gap {a[:,2].mean():.3f}  rankρ {np.nanmean(a[:,3]):.2f}  top1 {a[:,4].mean():.0%}")
    if len(t):
        print(f"TRUSTED({len(t)} classes): same {t[:,0].mean():.3f}  σ_ref {t[:,1].mean():.3f}  gap {t[:,2].mean():.3f}  rankρ {np.nanmean(t[:,3]):.2f}  top1 {t[:,4].mean():.0%}")
    return rows


v3 = R + "outputs/eval/certification/3.0.0-draft.8/analysis/distance_matrices.npz"
v4 = R + ".claude/worktrees/eval-v4-cert/outputs/eval/certification/4.0.0-draft.1/analysis/distance_matrices.npz"
z4 = np.load(v4, allow_pickle=True)
print("v4 matrices available:", list(z4.keys()))
analyze(v3, "m1a__v3_sided", "v3.0.0 certified kernel (app_ref = m1a__v3_sided)")
k4 = [k for k in z4.keys() if k != "keys"]
pick = next((k for k in k4 if "s3" in k.lower() or "m1a" in k.lower()), k4[0])
analyze(v4, pick, f"v4.0.0 kernel ({pick})")
