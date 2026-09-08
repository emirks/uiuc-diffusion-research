#!/usr/bin/env python
"""EffectData gapper screen — pick 300 effects WISELY (owner 2026-09-08: "300 effects, one seed, pick wisely not random").

Roles:
  exploit  150  highest kNN-predicted DCG-vs-prompt gap from BGE-large text embeddings of the effect descriptions,
                fitted on the 34 EffectData effects measured in grid v3 (eval/gap_predictors.csv)
  explore  140  one medoid per k-means cluster over the remaining eligible effects (coverage of effect space)
  anchor    10  measured effects (5 largest + 5 smallest gaps) re-drawn with NEW subjects: calibrate the one-seed screen
Eligibility: single-category effect, >= 20 clips (grid_v3.yaml), >= 4 portrait-ok S6-ROSTER clips from >= 4 distinct
subjects (reference + 3 endpoints: known shape, captioned subjects — no new captioning) and >= 12 portrait-ok clips in
total (pool topped up from non-roster clips, shapes probed lazily from the zips and cached). Per effect: 1 reference clip, 3 same-content
endpoints (3 other subjects, frame-0 anchor), a GT pool of up to 8 further clips. Deterministic (sorted, fixed seeds).
Output: misc/2026-09-08_ed_gapper_screen/selection.json (+ embeddings.npz).
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder"))
import build_registry_v3 as B  # noqa: E402

CAMP = REPO / "misc/2026-09-08_ed_gapper_screen"
MEASURED = REPO / "misc/2026-09-07_eval_grid_v2/eval/gap_predictors.csv"
N_EXPLOIT, N_EXPLORE, N_ANCHOR = 150, 140, 10
MIN_PORTRAIT_TOTAL, MIN_SUBJECTS, POOL = 12, 4, 8


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "BAAI/bge-large-en-v1.5"
    tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = model(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).numpy())
    return np.concatenate(out)


def main():
    cfg = yaml.safe_load((REPO / "eval_ladder/grid_v3.yaml").read_text())["effectdata"]
    ed = B.EffectData(cfg, probe=True)
    pref = cfg["clip_prefix"]
    ann = json.loads((REPO / cfg["annotations"]).read_text()); recs = list(ann.values()) if isinstance(ann, dict) else ann
    instr = collections.defaultdict(collections.Counter)
    for r in recs:
        instr[r["video_path"].split("/")[0]][r["instruction_en"]] += 1
    caps = json.loads((REPO / cfg["captions"]).read_text())["descriptions"]

    def roster_portrait(e):
        return sorted((c for c in ed.clips[e] if c[0] in ed.roster_shape and ed.portrait_ok(c[0])), key=lambda c: c[0])

    def portrait_lazy(e, need=MIN_PORTRAIT_TOTAL):
        """roster portrait clips first (known shapes), then non-roster clips probed one by one until `need` are found."""
        out = roster_portrait(e)
        if len(out) >= need:
            return out
        for c in sorted(ed.clips[e], key=lambda c: c[0]):
            if c[0] in ed.roster_shape:
                continue
            if ed.portrait_ok(c[0]):
                out.append(c)
                if len(out) >= need:
                    break
        return out

    eligible = []
    for i, e in enumerate(sorted(ed.clips)):
        if not ed.eligible_effect(e):
            continue
        rp = roster_portrait(e)
        if len({s for _, s, _ in rp}) < MIN_SUBJECTS or not all(f"{s}|A" in caps for _, s, _ in rp):
            continue
        if len(portrait_lazy(e)) >= MIN_PORTRAIT_TOTAL:
            eligible.append(e)
        if i % 500 == 0:
            ed.save(); print(f"  ..{i} effects scanned, eligible so far {len(eligible)}", flush=True)
    ed.save()
    print(f"[select] effects {len(ed.clips)}; eligible {len(eligible)}")

    measured = {r["class"]: r for r in csv.DictReader(open(MEASURED)) if r["family"] == "ED"}
    norm2eff = {B.EffectData.class_name(pref, e): e for e in ed.clips}
    meas_eff = {norm2eff[c]: {"gap_eff": float(r["gap_eff"]), "base_eff": float(r["base_eff"])} for c, r in measured.items() if c in norm2eff}
    print(f"[select] measured effects mapped: {len(meas_eff)}/{len(measured)}")

    texts = {e: f"{ed.category(e)}. {instr[e].most_common(1)[0][0]}" for e in set(eligible) | set(meas_eff)}
    for e in list(meas_eff)[:2]:
        print(f"  text sample [{e}]: {texts[e][:220]}")
    names = sorted(texts); X = embed([texts[e] for e in names]); idx = {e: i for i, e in enumerate(names)}
    np.savez(CAMP / "embeddings.npz", names=np.array(names), X=X)

    # exploit: kNN (k=5, cosine, similarity-weighted) regression of gap_eff / base_eff from the measured 34
    M = sorted(meas_eff); XM = X[[idx[e] for e in M]]; gM = np.array([meas_eff[e]["gap_eff"] for e in M]); bM = np.array([meas_eff[e]["base_eff"] for e in M])
    pred = {}
    for e in eligible:
        if e in meas_eff:
            continue
        s = XM @ X[idx[e]]; k = np.argsort(-s)[:5]; w = np.maximum(s[k], 0) + 1e-6
        pred[e] = {"pred_gap": float((w * gM[k]).sum() / w.sum()), "pred_base": float((w * bM[k]).sum() / w.sum()), "nn": [M[i] for i in k[:3]], "nn_sim": float(s[k].mean())}
    exploit = sorted(pred, key=lambda e: -pred[e]["pred_gap"])[:N_EXPLOIT]
    # explore: k-means medoids over the rest
    rest = [e for e in eligible if e not in meas_eff and e not in set(exploit)]
    from sklearn.cluster import KMeans
    XR = X[[idx[e] for e in rest]]
    km = KMeans(n_clusters=N_EXPLORE, n_init=4, random_state=0).fit(XR)
    explore = []
    for c in range(N_EXPLORE):
        mem = np.where(km.labels_ == c)[0]
        if not len(mem):
            continue
        j = mem[np.argmin(np.linalg.norm(XR[mem] - km.cluster_centers_[c], axis=1))]; explore.append(rest[j])
    # anchors: 5 largest + 5 smallest measured gaps that still have >= 4 fresh subjects
    tier_used = collections.defaultdict(set)
    for r in map(json.loads, filter(str.strip, open(REPO / "eval_ladder/registry_v3.jsonl"))):
        for k in ("reference", "endpoint"):
            v = r.get(k)
            if v and v.startswith(pref + "."):
                tier_used[".".join(v.split(".")[:2])].add(v)
    def fresh(e):
        used = tier_used.get(B.EffectData.class_name(pref, e), set())
        return [c for c in portrait_lazy(e, MIN_PORTRAIT_TOTAL + 4) if B.EffectData.std_stem(pref, c[0]) not in used]
    by_gap = sorted(meas_eff, key=lambda e: meas_eff[e]["gap_eff"])
    anchors = [e for e in reversed(by_gap) if len({s for _, s, _ in fresh(e)}) >= MIN_SUBJECTS][:5] + [e for e in by_gap if len({s for _, s, _ in fresh(e)}) >= MIN_SUBJECTS][:5]

    def assign(e, role):
        clips = fresh(e) if role == "anchor" else portrait_lazy(e)
        roster = [c for c in clips if c[0] in ed.roster_shape]          # reference + endpoints: captioned roster subjects
        ref = roster[0]; eps = []
        for c in roster[1:]:
            if c[1] != ref[1] and c[1] not in {x[1] for x in eps}:
                eps.append(c)
            if len(eps) == 3:
                break
        assert len(eps) == 3, e
        taken = {ref[0]} | {c[0] for c in eps}
        pool = [c for c in clips if c[0] not in taken][:POOL]
        d = {"effect": e, "cls": B.EffectData.class_name(pref, e), "role": role, "category": ed.category(e), "text": texts[e],
             "reference": B.EffectData.std_stem(pref, ref[0]), "reference_subject": ref[1],
             "endpoints": [{"std": B.EffectData.std_stem(pref, c[0]), "subject": c[1]} for c in eps],
             "pool": [B.EffectData.std_stem(pref, c[0]) for c in pool], "n_portrait_found": len(clips), "n_roster_portrait": len(roster)}
        d.update(pred.get(e, {})); d.update({f"measured_{k}": v for k, v in meas_eff.get(e, {}).items()})
        return d

    sel = [assign(e, "exploit") for e in exploit] + [assign(e, "explore") for e in explore] + [assign(e, "anchor") for e in anchors]
    out = {"created": "2026-09-08", "rule": __doc__.strip().splitlines()[0], "counts": collections.Counter(d["role"] for d in sel),
           "eligible": len(eligible), "measured": len(meas_eff), "effects": sel}
    (CAMP / "selection.json").write_text(json.dumps(out, indent=1))
    n_clips = sum(1 + 3 + len(d["pool"]) for d in sel)
    print(f"[select] roles {dict(out['counts'])}; clips to standardise {n_clips}; rows {3 * len(sel)}")
    print("[select] exploit pred_gap range:", f"{min(pred[e]['pred_gap'] for e in exploit):+.1f}..{max(pred[e]['pred_gap'] for e in exploit):+.1f}",
          "| explore pred_gap median:", f"{np.median([pred[e]['pred_gap'] for e in explore]):+.1f}")
    print("[select] anchors:", [(e, round(meas_eff[e]["gap_eff"], 1)) for e in anchors])
    print("[select] exploit top-8:", [(e, round(pred[e]["pred_gap"], 1)) for e in exploit[:8]])
    print(f"[select] -> {CAMP.relative_to(REPO)}/selection.json")


if __name__ == "__main__":
    main()
