#!/usr/bin/env python3
"""Both-axis counterfactuality frontier for EffectData (metadata-only, no video download).

Model (one-sided / prefix-anchored, endpoint = shared START frame):
  operator = effect class          (node on one side)
  endpoint = subject/source id      (node on the other side; the shared start frame)
  edge     = a clip realizing effect e on subject s
  Axis A (counterfactual)  = same start, different effect  = sum_s C(deg(s), 2)
  Axis B (demonstration)   = same effect, different start  = sum_e C(deg(e), 2)
  R (both at once)         = # rectangles (2 effects x 2 subjects, all 4 edges present)
                           = sum over effect-pairs  C(#shared subjects, 2)

Size axis = # clips kept (download cost). ~6.19 MB/clip (822 GB / 132,850).
Reads data/raw/effectdata/annotations.json; writes frontier_results.json beside this script.
"""
import json, itertools
from pathlib import Path
from collections import defaultdict, Counter
from math import comb

HERE = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "effectdata"
MB_PER_CLIP = 822_000 / 132_850  # ~6.19 MB

def load_graph(path=None):
    ann = json.load(open(path or (DATA / "annotations.json")))
    E2S = defaultdict(set)      # effect  -> set(subjects)
    S2E = defaultdict(set)      # subject -> set(effects)
    cell_clips = Counter()      # (effect, subject) -> #clips
    tags_per_subj = defaultdict(set)
    n_clips = 0; bad = 0
    TAGS = {"F", "M", "Z"}
    for key, rec in ann.items():
        effect = rec.get("video_path", key).split("/")[0]
        base = key[:-4] if key.lower().endswith(".mp4") else key
        # effect (comma-safe via video_path prefix) then "sid[,tag]"
        if base.startswith(effect + ","):
            rest = base[len(effect) + 1:]
        else:
            rest = base.split(",", 1)[1] if "," in base else ""
        if not rest:
            bad += 1; continue
        p = rest.rsplit(",", 1)
        if len(p) == 2 and p[1] in TAGS:
            sid, tag = p                      # tagged scheme <effect>,<id>,<F|M|Z>
        else:
            sid, tag = rest, "-"              # untagged scheme <effect>,<uuid>
        E2S[effect].add(sid)
        S2E[sid].add(effect)
        cell_clips[(effect, sid)] += 1
        tags_per_subj[sid].add(tag)
        n_clips += 1
    return dict(E2S), dict(S2E), cell_clips, tags_per_subj, n_clips, bad

def metrics(e2s, s2e, cell_clips):
    n_eff = len(e2s); n_sub = len(s2e)
    n_edges = sum(len(v) for v in e2s.values())
    surv = {(e, s) for e, ss in e2s.items() for s in ss}
    n_clips = sum(c for cell, c in cell_clips.items() if cell in surv)
    axisB = sum(comb(len(v), 2) for v in e2s.values())
    axisA = sum(comb(len(v), 2) for v in s2e.values())
    co = Counter()
    for effs in s2e.values():
        if len(effs) >= 2:
            for a, b in itertools.combinations(sorted(effs), 2):
                co[(a, b)] += 1
    R = sum(comb(c, 2) for c in co.values())
    subj_deg2 = sum(1 for v in s2e.values() if len(v) >= 2)
    return dict(n_eff=n_eff, n_sub=n_sub, n_edges=n_edges, n_clips=n_clips,
                axisA=axisA, axisB=axisB, R=R, subj_deg2=subj_deg2)

def kcore(E2S, S2E, k):
    e2s = {e: set(v) for e, v in E2S.items()}
    s2e = {s: set(v) for s, v in S2E.items()}
    changed = True
    while changed:
        changed = False
        for e in list(e2s):
            if len(e2s[e]) < k:
                for s in e2s[e]:
                    s2e[s].discard(e)
                del e2s[e]; changed = True
        for s in list(s2e):
            if len(s2e[s]) < k:
                for e in s2e[s]:
                    e2s[e].discard(s)
                del s2e[s]; changed = True
    e2s = {e: v for e, v in e2s.items() if v}
    s2e = {s: v for s, v in s2e.items() if v}
    return e2s, s2e

def pct(x, tot):
    return f"{100*x/tot:5.1f}%" if tot else "  n/a"

def main():
    import statistics as st
    E2S, S2E, cell_clips, tags, n_clips_all, bad = load_graph()
    full = metrics(E2S, S2E, cell_clips)

    print(f"# EffectData both-axis counterfactuality  (metadata-only)\n")
    print(f"parse: {n_clips_all} clips, {bad} unparsable")
    print(f"\n## Full-dataset baseline")
    print(f"  clips (size)            : {full['n_clips']:>10,}   (~{full['n_clips']*MB_PER_CLIP/1000:.0f} GB)")
    print(f"  operators (effects)     : {full['n_eff']:>10,}")
    print(f"  endpoints (subjects)    : {full['n_sub']:>10,}")
    print(f"  subjects with >=2 effs  : {full['subj_deg2']:>10,}   ({100*full['subj_deg2']/full['n_sub']:.1f}%)")
    print(f"  Axis A (counterfactual) : {full['axisA']:>10,}  pairs")
    print(f"  Axis B (demonstration)  : {full['axisB']:>10,}  pairs")
    print(f"  R (both-axis rectangles): {full['R']:>10,}")

    sdeg = [len(v) for v in S2E.values()]; edeg = [len(v) for v in E2S.values()]
    print(f"\n## Degree stats")
    print(f"  effects per subject : mean {st.mean(sdeg):.2f}  var {st.pvariance(sdeg):.2f}  max {max(sdeg)}")
    print(f"  subjects per effect : mean {st.mean(edeg):.2f}  var {st.pvariance(edeg):.2f}  "
          f"min {min(edeg)}  max {max(edeg)}")

    print(f"\n## Frontier -- k-core sweep (densest both-axis core)")
    hdr = f"{'k':>2} | {'effects':>7} | {'subjects':>8} | {'clips':>8} | {'GB':>5} | " \
          f"{'AxisA':>10} | {'AxisB':>10} | {'R':>11} | {'R%':>6}"
    print(hdr); print("-"*len(hdr))
    rows = [dict(k=1, **full)]
    print(f"{'1':>2} | {full['n_eff']:>7,} | {full['n_sub']:>8,} | {full['n_clips']:>8,} | "
          f"{full['n_clips']*MB_PER_CLIP/1000:>5.0f} | {full['axisA']:>10,} | "
          f"{full['axisB']:>10,} | {full['R']:>11,} | {pct(full['R'], full['R']):>6}")
    for k in [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]:
        e2s, s2e = kcore(E2S, S2E, k)
        if not e2s:
            break
        m = metrics(e2s, s2e, cell_clips)
        rows.append(dict(k=k, **m))
        print(f"{k:>2} | {m['n_eff']:>7,} | {m['n_sub']:>8,} | {m['n_clips']:>8,} | "
              f"{m['n_clips']*MB_PER_CLIP/1000:>5.0f} | {m['axisA']:>10,} | "
              f"{m['axisB']:>10,} | {m['R']:>11,} | {pct(m['R'], full['R']):>6}")

    out = HERE / "frontier_results.json"
    json.dump({"full": full, "kcore": rows, "mb_per_clip": MB_PER_CLIP}, open(out, "w"), indent=1)
    print(f"\n[saved] {out}")

if __name__ == "__main__":
    main()
