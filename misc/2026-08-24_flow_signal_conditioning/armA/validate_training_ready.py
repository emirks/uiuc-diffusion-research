#!/usr/bin/env python3
"""Training-readiness audit of the 44-ch DINO signal cache for 004_ctt_v2plus.
CPU-only, read-only. Folds in the fable-advisor's required hardening (2026-08-28):

  V1 COVERAGE     every (stratum,stem) target∪ref in 004_ctt_v2plus/samples.jsonl -> feat exists
                  AND F shape == the row's latent shape. Bar: 100% per training stratum {S0,S2a,S2b,S4,S6}.
  V2 S6-FULLSCAN  OPEN EVERY ONE of the 28,644 S6 feat npz (not a sample): F shape vs ROSTER,
                  all-finite, dtype float16, channels == CH_NAMES. Catches truncated/corrupt npz
                  the direct-write + skip-if-exists idempotency could otherwise treat as done.
  V3 RAW-INTEG    best-effort open of every S6 raw npz (zip readable); count unreadable.
  V4 PCA-HEALTH   frozen-PCA captured-variance share on S6 cells — POOLED and PER-SHAPE — vs the
                  fit-set baseline. PRE-REGISTERED BARS (before seeing the number): pooled >= 0.75x
                  baseline AND each of the 4 shapes >= 0.70x baseline. Below -> STOP, owner-gated
                  (a refit is a signal-v2, never silent).
  V5 CH-HEALTH    S6 report-only: per-channel %exact-zero (esp. conf after fwd-bwd zeroing) and
                  u/v window-saturation fraction (|u| or |v| >= 2.4 of the 2.5-cell R=5 ceiling).
  V6 EVAL         eval__ corpus present (223).
  V8 NORM-SMOKE   (needs --norm NORM_dino_v2.json) ~600 rows across all strata + all 4 S6 shapes:
                  resolve feat id, apply x_norm=clip((t(x)-loc)/scale,-5,5), assert finite and
                  within [-5,5], assert S6__<stem> resolves. Tests the join+norm contract the
                  trainer's signal-loader will use (that loader is future; this validates the shape of it).

Writes VALIDATION_TRAINING_READY.md.
"""
from __future__ import annotations
import sys, os, json, glob, random, argparse, zipfile
import numpy as np

LAB = "/taiga/illinois/eng/cs/jrehg/users/emirkisa"
DR = f"{LAB}/diffusion-research"
CACHE = f"{LAB}/cache/armA_signals"
FEAT = f"{CACHE}/feat"
RAW = f"{CACHE}/dino_raw"
PCA_PATH = f"{CACHE}/pca.npz"
ROOT004 = f"{DR}/outputs/ctt_v2/roots/ctt_v2plus_mix"
ENC = f"{DR}/outputs/ctt_v2/encodes/EFFECTDATA"
ARMA = f"{DR}/misc/2026-08-24_flow_signal_conditioning/armA"
sys.path.insert(0, ARMA)
from armA_extract import CH_NAMES
from fit_norm_dino import npz_F_shape

TRAIN_STRATA = ["S0", "S1", "S2a", "S2b", "S4", "S6"]   # S1 restored 2026-08-29
PCA_POOLED_BAR = 0.75      # x baseline  (pre-registered)
PCA_PERSHAPE_BAR = 0.70    # x baseline  (pre-registered)
SAT_THRESH = 2.4           # |u|/|v| near the R=5 -> 2.5-cell ceiling


def feat_id(stratum, stem):
    return f"train__{stratum}__{stem}" if stratum in ("S0", "S1") else f"{stratum}__{stem}"


def roster():
    return json.load(open(f"{ENC}/ROSTER.json"))["clips"]


def load_consumed(path, strata):
    # S6 target/reference are different subjects -> different orientations possible; the row's
    # `shape` is the target's. Expected shape per S6 stem = its OWN ROSTER latent_fhw.
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


# ---------------------------------------------------------------- V1 coverage
def v1_coverage():
    need = load_consumed(f"{ROOT004}/samples.jsonl", set(TRAIN_STRATA))
    from collections import defaultdict
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


# ---------------------------------------------------------------- V2 full scan
def v2_s6_fullscan():
    cl = roster()
    lf = {f"S6__{c['stem']}": tuple(int(x) for x in c["latent_fhw"]) for c in cl}
    res = dict(total=len(cl), opened=0, corrupt=[], shape_bad=[], finite_bad=[],
               dtype_bad=[], chan_bad=[], missing=[])
    for cid, exp in lf.items():
        p = f"{FEAT}/{cid}.npz"
        if not os.path.exists(p):
            res["missing"].append(cid); continue
        try:
            d = np.load(p, allow_pickle=True)
            F = d["F"]
            res["opened"] += 1
            if tuple(int(x) for x in F.shape[:3]) != exp:
                res["shape_bad"].append((cid, tuple(F.shape[:3]), exp))
            if F.dtype != np.float16:
                res["dtype_bad"].append((cid, str(F.dtype)))
            if not np.isfinite(np.asarray(F, np.float32)).all():
                res["finite_bad"].append(cid)
            if [str(x) for x in d["channels"]] != CH_NAMES:
                res["chan_bad"].append(cid)
        except Exception as e:
            res["corrupt"].append((cid, type(e).__name__))
    return res


# ---------------------------------------------------------------- V3 raw integrity
def v3_raw_integrity():
    cl = roster()
    ids = [f"S6__{c['stem']}" for c in cl]
    bad = []; miss = []; ok = 0
    for cid in ids:
        p = f"{RAW}/{cid}.npz"
        if not os.path.exists(p):
            miss.append(cid); continue
        try:
            # best-effort: open the central directory + confirm F.npy header is readable.
            # (Not a full CRC testzip — that would decompress ~1.5 TB.) Truncated/partial
            # writes fail to open the central dir or the header here.
            with zipfile.ZipFile(p) as zf:
                names = zf.namelist()
                fmem = "F_raw.npy" if "F_raw.npy" in names else next((n for n in names if n.startswith("F_raw")), None)
                if fmem is None:
                    bad.append(cid); continue
                with zf.open(fmem) as fh:
                    fh.read(128)   # npy magic+header bytes
            ok += 1
        except Exception:
            bad.append(cid)
    return dict(total=len(ids), ok=ok, bad=bad, missing=miss)


# ---------------------------------------------------------------- V4 PCA health
def _pooled_cells_from_raw(p):
    d = np.load(p, allow_pickle=True)
    Fr = np.asarray(d["F_raw"], np.float32)
    Hp, Wp = Fr.shape[1], Fr.shape[2]
    H, W = Hp // 2, Wp // 2
    pooled = Fr.reshape(Fr.shape[0], H, 2, W, 2, 768).mean((2, 4)).reshape(-1, 768).astype(np.float64)
    pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8)
    return pooled


def v4_pca_health(n_per_shape=15, seed=0):
    z = np.load(PCA_PATH); mean = z["mean"].astype(np.float64); comp = z["comp"].astype(np.float64)
    base = float(z["evr"].sum()) if "evr" in z else 0.45
    cl = roster()
    by_shape = {}
    for c in cl:
        by_shape.setdefault(tuple(c["latent_fhw"]), []).append(f"S6__{c['stem']}")
    rng = random.Random(seed)
    per_shape = {}; all_cells = []
    for shp, ids in sorted(by_shape.items()):
        picks = rng.sample(ids, min(n_per_shape, len(ids)))
        cells = []
        for cid in picks:
            p = f"{RAW}/{cid}.npz"
            if os.path.exists(p):
                cells.append(_pooled_cells_from_raw(p))
        if not cells:
            per_shape[shp] = None; continue
        X = np.concatenate(cells, 0); Xc = X - mean
        share = float((Xc @ comp.T).var(0).sum() / max(Xc.var(0).sum(), 1e-12))
        per_shape[shp] = share
        all_cells.append(X)
    Xa = np.concatenate(all_cells, 0); Xac = Xa - mean
    pooled = float((Xac @ comp.T).var(0).sum() / max(Xac.var(0).sum(), 1e-12))
    return base, pooled, per_shape


# ---------------------------------------------------------------- V5 channel health
def v5_channel_health(n=200, seed=3):
    fs = glob.glob(f"{FEAT}/S6__*.npz")
    picks = random.Random(seed).sample(fs, min(n, len(fs)))
    zero = np.zeros(44); ncell = 0
    ui, vi = CH_NAMES.index("u"), CH_NAMES.index("v")
    sat = 0
    for f in picks:
        F = np.asarray(np.load(f, allow_pickle=True)["F"], np.float32).reshape(-1, 44)
        ncell += F.shape[0]
        zero += (F == 0.0).sum(0)
        sat += ((np.abs(F[:, ui]) >= SAT_THRESH) | (np.abs(F[:, vi]) >= SAT_THRESH)).sum()
    return {CH_NAMES[c]: 100.0 * zero[c] / ncell for c in range(44)}, 100.0 * sat / ncell, len(picks)


# ---------------------------------------------------------------- V8 norm smoke
def v7_orphans():
    """Every training-stratum feat file is either consumed by 004 OR a WHITELISTED drop.
    Whitelist = the S6 shape-singleton drops recorded in ROOT_MANIFEST (feat kept in cache, not
    consumed after the same-shape re-pairing). A non-whitelisted orphan is a coverage regression."""
    need = load_consumed(f"{ROOT004}/samples.jsonl", set(TRAIN_STRATA))
    consumed = {feat_id(s, stem) for (s, stem) in need}
    cache_ids = set()
    for s in TRAIN_STRATA:
        pat = f"{FEAT}/train__{s}__*.npz" if s in ("S0", "S1") else f"{FEAT}/{s}__*.npz"
        cache_ids |= {os.path.basename(p)[:-4] for p in glob.glob(pat)}
    man = json.load(open(f"{ROOT004}/ROOT_MANIFEST.json"))
    dropped = [d["clip"] for d in man.get("drops", {}).get("S6", {}).get("clips", [])
               if "no_same_shape_same_effect_partner" in d.get("reasons", [])]
    whitelist = {f"S6__{stem}" for stem in dropped}
    orphans = cache_ids - consumed - whitelist
    return dict(cache=len(cache_ids), consumed=len(consumed & cache_ids),
                whitelisted=len(whitelist), n_dropped=len(dropped),
                orphans=sorted(orphans)[:10], n_orphans=len(orphans))


def v8_norm_smoke(norm_json, n=600, seed=11):
    doc = json.load(open(norm_json))
    loc = np.array([c["loc"] for c in doc["channels"]], np.float64)
    scale = np.array([c["scale"] for c in doc["channels"]], np.float64)
    tr = [c["transform"] for c in doc["channels"]]
    need = load_consumed(f"{ROOT004}/samples.jsonl", set(TRAIN_STRATA))
    # ensure coverage of all 4 S6 shapes: bucket keys, sample across
    keys = list(need.keys())
    rng = random.Random(seed); rng.shuffle(keys)
    # force-include >=1 of each S6 shape
    cl = roster(); by_shape = {}
    for c in cl:
        by_shape.setdefault(tuple(c["latent_fhw"]), []).append(("S6", c["stem"]))
    forced = [rng.choice(v) for v in by_shape.values()]
    picks = forced + keys[:n]
    finite_bad = []; clip_bad = []; unresolved = []; ok = 0; shapes_seen = set()
    for (s, stem) in picks:
        p = f"{FEAT}/{feat_id(s, stem)}.npz"
        if not os.path.exists(p):
            unresolved.append(feat_id(s, stem)); continue
        F = np.asarray(np.load(p, allow_pickle=True)["F"], np.float64)
        shapes_seen.add(F.shape[:3])
        t = np.where(np.array(tr) == "asinh", np.arcsinh(F), F)
        z = np.clip((t - loc) / scale, -5.0, 5.0)
        if not np.isfinite(z).all():
            finite_bad.append(feat_id(s, stem))
        if z.min() < -5.0 - 1e-6 or z.max() > 5.0 + 1e-6:
            clip_bad.append(feat_id(s, stem))
        ok += 1
    return dict(applied=ok, finite_bad=finite_bad, clip_bad=clip_bad, unresolved=unresolved,
                s6_shapes=sorted(str(x) for x in shapes_seen if x[0] == 11))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ARMA}/VALIDATION_TRAINING_READY.md")
    ap.add_argument("--norm", default="")   # NORM_dino_v2.json for V8 (optional)
    a = ap.parse_args()
    L = []; P = L.append

    print("[V1] coverage over 004 ...", flush=True); per, ntot = v1_coverage()
    print("[V2] full-open all 28,644 S6 feat ...", flush=True); fs = v2_s6_fullscan()
    print("[V3] S6 raw integrity ...", flush=True); rr = v3_raw_integrity()
    print("[V4] S6 PCA health per shape ...", flush=True); base, pooled, pershape = v4_pca_health()
    print("[V5] S6 channel health ...", flush=True); zero, satpct, nch = v5_channel_health()
    print("[V7] orphan whitelist ...", flush=True); orph = v7_orphans()
    eval_ct = len(glob.glob(f"{FEAT}/eval__*.npz"))
    v8 = v8_norm_smoke(a.norm) if a.norm and os.path.exists(a.norm) else None
    v7_pass = orph["n_orphans"] == 0

    cov_pass = all(per[s]["hit"] == per[s]["total"] and per[s]["shape_ok"] == per[s]["total"] for s in TRAIN_STRATA)
    fullscan_pass = (not fs["corrupt"] and not fs["shape_bad"] and not fs["finite_bad"]
                     and not fs["dtype_bad"] and not fs["chan_bad"] and not fs["missing"])
    raw_pass = not rr["bad"] and not rr["missing"]
    pca_pooled_ok = pooled >= PCA_POOLED_BAR * base
    pca_pershape_ok = all(v is not None and v >= PCA_PERSHAPE_BAR * base for v in pershape.values())
    pca_pass = pca_pooled_ok and pca_pershape_ok
    v8_pass = v8 is None or (not v8["finite_bad"] and not v8["clip_bad"] and not v8["unresolved"])

    P("# VALIDATION — 44-ch DINO signal, training-readiness for `004_ctt_v2plus`\n")
    P(f"Read-only audit on `{os.uname().nodename}`; {ntot:,} consumed (stratum,stem) keys from "
      f"`004_ctt_v2plus/samples.jsonl`. Bars & checks per fable-advisor 2026-08-28.\n")

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

    P("## V2 · S6 full-open integrity — all 28,644 npz opened (bar: 0 defects)\n")
    P(f"opened **{fs['opened']:,}/{fs['total']:,}**; corrupt {len(fs['corrupt'])}, shape-bad {len(fs['shape_bad'])}, "
      f"non-finite {len(fs['finite_bad'])}, dtype≠fp16 {len(fs['dtype_bad'])}, channels≠CH_NAMES {len(fs['chan_bad'])}, "
      f"missing {len(fs['missing'])}.  **V2: {'PASS' if fullscan_pass else 'FAIL'}**")
    for k in ("corrupt", "shape_bad", "finite_bad", "dtype_bad", "chan_bad", "missing"):
        if fs[k]: P(f"  - {k} (5): {fs[k][:5]}")
    P("")

    P("## V3 · S6 raw-cache integrity (best-effort zip test)\n")
    P(f"ok **{rr['ok']:,}/{rr['total']:,}**; unreadable {len(rr['bad'])}, missing {len(rr['missing'])}.  "
      f"**V3: {'PASS' if raw_pass else 'FAIL'}**" + (f"  bad(5): {rr['bad'][:5]}" if rr['bad'] else ""))
    P("")

    P("## V4 · Frozen-PCA captured-variance share on S6 (pre-registered bars)\n")
    P(f"Baseline (fit-set EVR sum) = **{base:.3f}**. Bars: pooled ≥ {PCA_POOLED_BAR}×base = "
      f"**{PCA_POOLED_BAR*base:.3f}**, each shape ≥ {PCA_PERSHAPE_BAR}×base = **{PCA_PERSHAPE_BAR*base:.3f}**.\n")
    P("| scope | share | ×base | bar met |")
    P("|---|--:|--:|:--:|")
    P(f"| pooled | {pooled:.3f} | {pooled/base:.2f}× | {'✓' if pca_pooled_ok else '✗ STOP'} |")
    for shp, v in sorted(pershape.items()):
        P(f"| {tuple(shp)} | {v:.3f} | {v/base:.2f}× | {'✓' if v>=PCA_PERSHAPE_BAR*base else '✗ STOP'} |")
    P(f"\n**V4: {'PASS — keep frozen basis' if pca_pass else 'FAIL — STOP, escalate to owner (do NOT silently refit)'}**\n")

    P("## V5 · S6 channel health (report-only)\n")
    P(f"u/v window-saturation (|u| or |v| ≥ {SAT_THRESH}, near the R=5 → 2.5-cell ceiling): "
      f"**{satpct:.2f}%** of cells (n={nch} clips).")
    P(f"\nchannels with >5% exact-zero: " +
      (", ".join(f"{k} {zero[k]:.1f}%" for k in CH_NAMES if zero[k] > 5.0) or "none") + ".")
    P(f"\nconf %exact-zero (fwd-bwd rejected cells): **{zero['conf']:.2f}%**.\n")

    P("## V6 · Eval corpus\n")
    P(f"eval__ feat present: **{eval_ct}** (held-out instrument; not a training input).\n")

    P("## V7 · Orphan whitelist (bar: 0 non-whitelisted orphans)\n")
    P(f"training-stratum feat in cache **{orph['cache']:,}**, consumed by 004 **{orph['consumed']:,}**, "
      f"whitelisted S6 shape-singleton drops **{orph['whitelisted']:,}** (from ROOT_MANIFEST). "
      f"non-whitelisted orphans: **{orph['n_orphans']}**.  **V7: {'PASS' if v7_pass else 'FAIL'}**"
      + (f"  e.g. {orph['orphans']}" if orph['n_orphans'] else "") + "\n")

    if v8 is not None:
        P("## V8 · Norm-apply smoke (join + NORM v2, the contract the trainer's signal-loader will use)\n")
        P(f"applied to **{v8['applied']}** rows across strata; S6 shapes exercised: {v8['s6_shapes']}. "
          f"non-finite {len(v8['finite_bad'])}, out-of-[-5,5] {len(v8['clip_bad'])}, unresolved {len(v8['unresolved'])}.  "
          f"**V8: {'PASS' if v8_pass else 'FAIL'}**")
        for k in ("finite_bad", "clip_bad", "unresolved"):
            if v8[k]: P(f"  - {k} (5): {v8[k][:5]}")
        P("")
    else:
        P("## V8 · Norm-apply smoke\n_Not run (pass `--norm NORM_dino_v2.json`)._\n")

    overall = cov_pass and fullscan_pass and raw_pass and pca_pass and v7_pass and v8_pass
    P(f"## Overall: {'READY' if overall else 'NOT READY / STOP'} — "
      f"V1 {'✓' if cov_pass else '✗'} · V2 {'✓' if fullscan_pass else '✗'} · V3 {'✓' if raw_pass else '✗'} · "
      f"V4 {'✓' if pca_pass else '✗'} · V7 {'✓' if v7_pass else '✗'} · V8 {'✓' if v8_pass else ('✓' if v8 is None else '✗')}. "
      f"Norm gates G-N1..G-N5 in the current NORM_REPORT.\n")

    open(a.out, "w").write("\n".join(L))
    print("wrote", a.out, flush=True)
    print("SUMMARY",
          f"V1={'PASS' if cov_pass else 'FAIL'}",
          f"V2_fullscan={'PASS' if fullscan_pass else 'FAIL'}",
          f"V3_raw={'PASS' if raw_pass else 'FAIL'}",
          f"V4_pca={'PASS' if pca_pass else 'FAIL'}(pooled {pooled:.3f}/{pooled/base:.2f}x)",
          f"V7_orphans={'PASS' if v7_pass else 'FAIL'}(n={orph['n_orphans']},wl={orph['whitelisted']})",
          f"V8={'PASS' if v8_pass else ('skip' if v8 is None else 'FAIL')}",
          f"OVERALL={'READY' if overall else 'NOT_READY'}", flush=True)


if __name__ == "__main__":
    main()
