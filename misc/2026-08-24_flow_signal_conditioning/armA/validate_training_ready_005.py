#!/usr/bin/env python3
"""Training-readiness audit of the 44-ch DINO signal cache for 005_ctt_v2plus_s6reshape.
CPU-only, read-only. Successor of validate_training_ready.py (004).

Per the Round-4 directive, only V1/V6/V8 are re-run here against the 005 signal root
($LAB/cache/armA_signals_005/feat) + the 005 code-side root + the r832 S6 roster:

  V1 COVERAGE   every (stratum,stem) target∪ref in 005 samples.jsonl -> feat exists AND F shape
                == the row's latent shape (S6 keyed to the r832 ROSTER latent_fhw). Bar: 100% per
                training stratum {S0,S1,S2a,S2b,S4,S6}.
  V6 EVAL       eval__ corpus present (223).
  V8 NORM-SMOKE (NORM_dino_v4.json) >=600 rows across all strata + >=100 per S6 grid: resolve feat
                id, x_norm=clip((t(x)-loc)/scale,-5,5), assert finite AND within [-5,5], and that
                S6__<stem> resolves.

V2 (S6 full-open), V3 (raw integrity), V4 (PCA health), V5 (channel health) are NOT re-run — they
were run in Round 3; this report cites those artifacts by path.

Writes VALIDATION_TRAINING_READY_005.md; exit 0 iff V1 & V8 PASS.
"""
from __future__ import annotations
import sys, os, json, glob, random, argparse
from collections import defaultdict
import numpy as np

LAB = "/taiga/illinois/eng/cs/jrehg/users/emirkisa"
DR = f"{LAB}/diffusion-research"
CACHE = f"{LAB}/cache/armA_signals_005"
FEAT = f"{CACHE}/feat"
ROOT = f"{DR}/outputs/ctt_v2/roots/ctt_v2plus_s6reshape_mix"
ENC_R = f"{DR}/outputs/ctt_v2/encodes/EFFECTDATA_r832"
ARMA = f"{DR}/misc/2026-08-24_flow_signal_conditioning/armA"
R3 = f"{DR}/misc/2026-08-30_s6_reshape/r3"
sys.path.insert(0, ARMA)
from armA_extract import CH_NAMES              # noqa: E402
from fit_norm_dino import npz_F_shape          # noqa: E402

TRAIN_STRATA = ["S0", "S1", "S2a", "S2b", "S4", "S6"]
S6_GRIDS = [(11, 16, 26), (11, 26, 16)]


def feat_id(stratum, stem):
    return f"train__{stratum}__{stem}" if stratum in ("S0", "S1") else f"{stratum}__{stem}"


def roster():
    return json.load(open(f"{ENC_R}/ROSTER.json"))["clips"]


def load_consumed(path, strata):
    s6shape = {c["stem"]: tuple(int(x) for x in c["latent_fhw"]) for c in roster()} if "S6" in strata else {}
    need = {}
    for l in open(path):
        r = json.loads(l)
        s = r["stratum"]
        if s not in strata:
            continue
        rowshp = tuple(int(x) for x in r["shape"])
        for stem in (r["target"], r["reference"]):
            need[(s, stem)] = s6shape[stem] if s == "S6" else rowshp
    return need


def v1_coverage():
    need = load_consumed(f"{ROOT}/samples.jsonl", set(TRAIN_STRATA))
    per = defaultdict(lambda: dict(total=0, hit=0, shape_ok=0, miss=[], shape_bad=[]))
    for (s, stem), shp in need.items():
        p = f"{FEAT}/{feat_id(s, stem)}.npz"
        d = per[s]; d["total"] += 1
        if os.path.exists(p):
            d["hit"] += 1
            if npz_F_shape(p)[:3] == shp:
                d["shape_ok"] += 1
            else:
                d["shape_bad"].append(stem)
        else:
            d["miss"].append(stem)
    return per, len(need)


def v8_norm_smoke(norm_json, n=600, per_grid=100, seed=11):
    doc = json.load(open(norm_json))
    loc = np.array([c["loc"] for c in doc["channels"]], np.float64)
    scale = np.array([c["scale"] for c in doc["channels"]], np.float64)
    tr = [c["transform"] for c in doc["channels"]]
    need = load_consumed(f"{ROOT}/samples.jsonl", set(TRAIN_STRATA))
    keys = list(need.keys())
    rng = random.Random(seed); rng.shuffle(keys)
    # force-include >=per_grid rows for each S6 grid
    shp = {c["stem"]: tuple(int(x) for x in c["latent_fhw"]) for c in roster()}
    s6_by_grid = defaultdict(list)
    for (s, stem) in keys:
        if s == "S6":
            s6_by_grid[shp[stem]].append((s, stem))
    forced = []
    for g in S6_GRIDS:
        forced += s6_by_grid[g][:per_grid]
    picks = forced + keys[:n]
    finite_bad = []; clip_bad = []; unresolved = []; ok = 0
    grid_seen = defaultdict(int); shapes_seen = set()
    for (s, stem) in picks:
        p = f"{FEAT}/{feat_id(s, stem)}.npz"
        if not os.path.exists(p):
            unresolved.append(feat_id(s, stem)); continue
        F = np.asarray(np.load(p, allow_pickle=True)["F"], np.float64)
        shapes_seen.add(F.shape[:3])
        if s == "S6":
            grid_seen[F.shape[:3]] += 1
        t = np.where(np.array(tr) == "asinh", np.arcsinh(F), F)
        z = np.clip((t - loc) / scale, -5.0, 5.0)
        if not np.isfinite(z).all():
            finite_bad.append(feat_id(s, stem))
        if z.min() < -5.0 - 1e-6 or z.max() > 5.0 + 1e-6:
            clip_bad.append(feat_id(s, stem))
        ok += 1
    return dict(applied=ok, finite_bad=finite_bad, clip_bad=clip_bad, unresolved=unresolved,
                s6_shapes=sorted(str(x) for x in shapes_seen if x[0] == 11),
                grid_seen={str(list(k)): v for k, v in grid_seen.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ARMA}/VALIDATION_TRAINING_READY_005.md")
    ap.add_argument("--norm", default=f"{DR}/store/datasets/003_dino_signals/NORM_dino_v4.json")
    a = ap.parse_args()
    L = []; P = L.append

    print("[V1] coverage over 005 ...", flush=True); per, ntot = v1_coverage()
    eval_ct = len(glob.glob(f"{FEAT}/eval__*.npz"))
    print("[V8] norm-apply smoke (NORM_dino_v4) ...", flush=True)
    v8 = v8_norm_smoke(a.norm)

    cov_pass = all(per[s]["hit"] == per[s]["total"] and per[s]["shape_ok"] == per[s]["total"] for s in TRAIN_STRATA)
    v6_pass = eval_ct == 223
    grid_ok = all(v8["grid_seen"].get(str(list(g)), 0) >= 100 for g in S6_GRIDS)
    v8_pass = (not v8["finite_bad"] and not v8["clip_bad"] and not v8["unresolved"]
               and v8["applied"] >= 600 and grid_ok)

    P("# VALIDATION — 44-ch DINO signal, training-readiness for `005_ctt_v2plus_s6reshape`\n")
    P(f"Read-only audit on `{os.uname().nodename}`; {ntot:,} consumed (stratum,stem) keys from "
      f"`005_ctt_v2plus_s6reshape/samples.jsonl`; signal root `{FEAT}`; S6 keyed to the r832 ROSTER.\n")

    P("## V1 · Coverage — every training row's signal present + shape-matched (bar 100%)\n")
    P("| stratum | consumed keys | feat hit | shape match | gaps |")
    P("|---|--:|--:|--:|--:|")
    for s in TRAIN_STRATA:
        d = per[s]
        P(f"| {s} | {d['total']:,} | {d['hit']:,} | {d['shape_ok']:,} | {len(d['miss'])+len(d['shape_bad'])} |")
    P(f"\n**V1: {'PASS' if cov_pass else 'FAIL'}**")
    for s in TRAIN_STRATA:
        if per[s]["miss"]: P(f"  - {s} MISSING (5): {per[s]['miss'][:5]}")
        if per[s]["shape_bad"]: P(f"  - {s} SHAPE-BAD (5): {per[s]['shape_bad'][:5]}")
    P("")

    P("## V6 · Eval corpus\n")
    P(f"eval__ feat present: **{eval_ct}** (bar 223) {'✓' if v6_pass else '✗'}\n")

    P("## V8 · Norm-apply smoke (join + NORM_dino_v4, the trainer's signal-loader contract)\n")
    P(f"norm = `{os.path.basename(a.norm)}`. Applied to **{v8['applied']}** rows (bar ≥600); "
      f"S6 grids exercised {v8['grid_seen']} (bar ≥100 each). "
      f"non-finite {len(v8['finite_bad'])}, out-of-[-5,5] {len(v8['clip_bad'])}, unresolved {len(v8['unresolved'])}. "
      f"**V8: {'PASS' if v8_pass else 'FAIL'}**")
    for k in ("finite_bad", "clip_bad", "unresolved"):
        if v8[k]: P(f"  - {k} (5): {v8[k][:5]}")
    P("")

    P("## V2–V5 · Round-3 artifacts (NOT re-run here)\n")
    P("These passed in Round 3 (advisor-verified 2026-08-30); cited by path:\n")
    P(f"- **V2 S6 full-open + V-verify** (28,644 feat set-equal, census 14,523/14,121, shape/chan/"
      f"finite 0, 0 .tmp): `{R3}/../STATUS.md` + verify job 3049773 log.")
    P(f"- **V4 PCA health** (pooled 0.3744 ≥0.339; 4 native 0.3703–0.3781 ≥0.316; SECONDARY paired "
      f"ratio 0.9694): `{R3}/health_full.json`.")
    P(f"- **Determinism** (24/24 bitwise): `{R3}/determinism.json`.")
    P(f"- **NORM_dino_v4 gates G-N1..G-N5 PASS**, S6 131,074,944 cells, non-S6 moments==v3: "
      f"`{DR}/store/datasets/003_dino_signals/NORM_REPORT_v4.md`.\n")

    overall = cov_pass and v6_pass and v8_pass
    P(f"## Overall: {'READY' if overall else 'NOT READY'} — "
      f"V1 {'✓' if cov_pass else '✗'} · V6 {'✓' if v6_pass else '✗'} · V8 {'✓' if v8_pass else '✗'} "
      f"(V2–V5 cited from Round 3).\n")

    open(a.out, "w").write("\n".join(L))
    print("wrote", a.out, flush=True)
    print("SUMMARY",
          f"V1={'PASS' if cov_pass else 'FAIL'}",
          f"V6={'PASS' if v6_pass else 'FAIL'}(eval={eval_ct})",
          f"V8={'PASS' if v8_pass else 'FAIL'}(applied={v8['applied']},grids={v8['grid_seen']})",
          f"OVERALL={'READY' if overall else 'NOT_READY'}", flush=True)
    sys.exit(0 if overall else 3)


if __name__ == "__main__":
    main()
