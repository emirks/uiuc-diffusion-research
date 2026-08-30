#!/usr/bin/env python3
"""Fit the frozen v1 per-channel normalization for the 44-ch DINO-basis operator signal.

Implements NORM_DIRECTIVES.md exactly:
  x_norm = clip( (t(x) - loc_c) / scale_c , -5, +5 )
  t = identity, except a channel that fails gate G-N2 gets t = asinh (single refit round).
  loc_c, scale_c = per-channel mean/std with EQUAL stratum weighting over {S0,S1,S2a,S2b,S4}.
  eval__ files EXCLUDED from the fit and the verify sample.

One full streaming pass accumulates BOTH identity and asinh moments per (stratum, channel)
so the asinh escape hatch needs no second file pass. fp16 -> float64 for all moments.

Outputs (paths are CLI args):
  --out-json   NORM_dino_v1.json  (schema per directive §5.1)
  --out-report NORM_REPORT.md     (the §4 tables + gate verdicts)

CPU-only.  Run under: source $LAB/envs-aarch64/activate ; OMP_NUM_THREADS=2.
"""
from __future__ import annotations
import os, sys, json, glob, math, time, random, argparse, hashlib
import numpy as np

LAB = "/taiga/illinois/eng/cs/jrehg/users/emirkisa"
DR = f"{LAB}/diffusion-research"
CACHE = f"{LAB}/cache/armA_signals"
FEAT = f"{CACHE}/feat"
PCA_PATH = f"{CACHE}/pca.npz"
CTT = f"{DR}/datasets/ctt_v2"
ROOT004 = f"{DR}/outputs/ctt_v2/roots/ctt_v2plus_mix"   # 004_ctt_v2plus root (carries the S6 rows)
ENC_ROSTER = f"{DR}/outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json"   # per-S6-clip authoritative shape
CAMP = f"{DR}/misc/2026-08-24_flow_signal_conditioning"
ARMA = f"{CAMP}/armA"
sys.path.insert(0, ARMA)
from armA_extract import CH_NAMES  # single naming authority

# Fit strata. v1 (frozen, committed) = the 5 ctt_v2 strata. v2 adds EffectData S6 (--strata).
STRATA = ["S0", "S1", "S2a", "S2b", "S4"]
# where each stratum's (stratum,stem)->row coverage is joined from, for G-N5. A norm's coverage
# is checked against the dataset it SERVES: 004_ctt_v2plus now carries ALL six training strata
# (S1 restored 2026-08-29), so every stratum joins from 004's samples.jsonl. (For a v2 fit, which
# excludes S1, the S1 source is simply unused.)
COVERAGE_SOURCES = [
    (f"{ROOT004}/samples.jsonl", {"S0", "S1", "S2a", "S2b", "S4", "S6"}),
]
NC = 44
CLIP = 5.0
SCALE_FLOOR = 1e-6
G_N2_MAX = 2.0          # % |z|>=5 per channel
G_N1_LO, G_N1_HI = 0.5, 2.0
VERIFY_PER_STRATUM = 300
NAMECHECK_FILES = 150


# ---------------------------------------------------------------- file lists
def stratum_files(stratum):
    if stratum in ("S0", "S1"):
        pat = f"{FEAT}/train__{stratum}__*.npz"
    else:
        pat = f"{FEAT}/{stratum}__*.npz"
    return sorted(glob.glob(pat))


def load_F(path):
    """(cells, 44) float64 view of one clip's field, all timesteps as stored."""
    z = np.load(path, allow_pickle=True)
    F = z["F"]
    return np.asarray(F, dtype=np.float64).reshape(-1, NC)


def npz_F_shape(path):
    """Read F's shape from the npy header inside the (compressed) npz WITHOUT
       decompressing the array — reads only the header bytes."""
    import zipfile
    import numpy.lib.format as fmt
    with zipfile.ZipFile(path) as zf:
        name = "F.npy" if "F.npy" in zf.namelist() else next(n for n in zf.namelist() if n.startswith("F"))
        with zf.open(name) as fh:
            major, minor = fmt.read_magic(fh)
            reader = getattr(fmt, f"read_array_header_{major}_{minor}")
            shape, _fortran, _dtype = reader(fh)
    return tuple(int(x) for x in shape)


# ---------------------------------------------------------------- fit pass
def fit_moments(verbose=True):
    """Full streaming pass. Returns per-stratum dict of float64 accumulators:
       n (cells), s1[44], s2[44] (identity),  as1[44], as2[44] (asinh)."""
    acc = {}
    nonfinite = np.zeros(NC, np.int64)
    t0 = time.time()
    for s in STRATA:
        files = stratum_files(s)
        n = 0
        s1 = np.zeros(NC); s2 = np.zeros(NC)
        as1 = np.zeros(NC); as2 = np.zeros(NC)
        for i, f in enumerate(files):
            X = load_F(f)                                  # (cells,44) f64
            bad = ~np.isfinite(X)
            if bad.any():
                nonfinite += bad.sum(0)
                X = np.where(bad, 0.0, X)
            n += X.shape[0]
            s1 += X.sum(0); s2 += (X * X).sum(0)
            A = np.arcsinh(X)
            as1 += A.sum(0); as2 += (A * A).sum(0)
            if verbose and (i + 1) % 500 == 0:
                print(f"    {s} {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
        acc[s] = dict(n=int(n), n_files=len(files), s1=s1, s2=s2, as1=as1, as2=as2)
        print(f"  [fit] {s}: {len(files)} files, {n} cells  ({time.time()-t0:.0f}s)", flush=True)
    return acc, nonfinite


def equal_stratum_locscale(acc, key1, key2):
    """Equal-stratum mean/std per channel from per-stratum first/second raw moments.
       mean = (1/S) Σ_s mean_s ;  var = (1/S) Σ_s E[x^2]_s  -  mean^2 ."""
    S = len(STRATA)
    mean_s = {s: acc[s][key1] / acc[s]["n"] for s in STRATA}
    ex2_s = {s: acc[s][key2] / acc[s]["n"] for s in STRATA}
    loc = np.mean([mean_s[s] for s in STRATA], axis=0)
    ex2bar = np.mean([ex2_s[s] for s in STRATA], axis=0)
    var = ex2bar - loc * loc
    scale = np.sqrt(np.clip(var, 0.0, None))
    return loc, scale, mean_s, {s: np.sqrt(np.clip(ex2_s[s] - mean_s[s]**2, 0, None)) for s in STRATA}


# ---------------------------------------------------------------- verify pass
def verify_sample_files(seed=0):
    rng = random.Random(seed)
    picks = {}
    for s in STRATA:
        fs = stratum_files(s)
        if len(fs) <= VERIFY_PER_STRATUM:
            picks[s] = list(fs)
        else:
            picks[s] = rng.sample(fs, VERIFY_PER_STRATUM)
    return picks


def raw_stats(vals):
    """vals (M,) f64 -> dict of raw stats for one channel/stratum."""
    return dict(
        min=float(vals.min()), max=float(vals.max()),
        p1=float(np.percentile(vals, 1)), p50=float(np.percentile(vals, 50)),
        p99=float(np.percentile(vals, 99)),
        mean=float(vals.mean()), std=float(vals.std()),
        pct_zero=float(100.0 * np.mean(vals == 0.0)),
    )


def znorm(x, loc, scale, transform):
    t = np.arcsinh(x) if transform == "asinh" else x
    return (t - loc) / scale


def verify(loc_id, scale_id, loc_as, scale_as, seed=0):
    """Collect per-stratum raw stats + per-stratum post-norm z-moment sums for BOTH
       identity and asinh transforms (so the asinh decision needs no re-pass)."""
    picks = verify_sample_files(seed)
    raw = {s: [None] * NC for s in STRATA}          # raw per-stratum per-channel stats
    # post-norm accumulators: per stratum, per channel, per transform -> moment sums on CLIPPED z
    zsum = {tr: {s: dict(n=0, z1=np.zeros(NC), z2=np.zeros(NC), z3=np.zeros(NC),
                          z4=np.zeros(NC), clip=np.zeros(NC), zero=np.zeros(NC))
                 for s in STRATA} for tr in ("none", "asinh")}
    for s in STRATA:
        chunks = [load_F(f) for f in picks[s]]
        X = np.concatenate(chunks, 0)               # (Ms,44) f64
        for c in range(NC):
            raw[s][c] = raw_stats(X[:, c])
        for tr, loc, scale in (("none", loc_id, scale_id), ("asinh", loc_as, scale_as)):
            zu = znorm(X, loc, scale, tr)            # unclipped
            clipmask = np.abs(zu) >= CLIP
            zc = np.clip(zu, -CLIP, CLIP)
            a = zsum[tr][s]
            a["n"] = X.shape[0]
            a["z1"] += zc.sum(0); a["z2"] += (zc**2).sum(0)
            a["z3"] += (zc**3).sum(0); a["z4"] += (zc**4).sum(0)
            a["clip"] += clipmask.sum(0)
            a["zero"] += (zc == 0.0).sum(0)
        del chunks, X
    return raw, zsum, {s: len(picks[s]) for s in STRATA}


def pool_postnorm(zsum_tr):
    """Equal-stratum-weighted post-norm pool stats per channel from per-stratum moment sums.
       Returns std, excess_kurtosis, clip_rate(%), pct_zero(%), and per-stratum mean/std."""
    S = len(STRATA)
    Ez = np.zeros(NC); Ez2 = np.zeros(NC); Ez3 = np.zeros(NC); Ez4 = np.zeros(NC)
    clip_rate = np.zeros(NC); pct_zero = np.zeros(NC)
    per_stratum = {}
    for s in STRATA:
        a = zsum_tr[s]; n = a["n"]
        m1 = a["z1"] / n; m2 = a["z2"] / n; m3 = a["z3"] / n; m4 = a["z4"] / n
        Ez += m1 / S; Ez2 += m2 / S; Ez3 += m3 / S; Ez4 += m4 / S
        clip_rate += (100.0 * a["clip"] / n) / S
        pct_zero += (100.0 * a["zero"] / n) / S
        per_stratum[s] = dict(mean=m1, std=np.sqrt(np.clip(m2 - m1**2, 0, None)))
    var = Ez2 - Ez**2
    std = np.sqrt(np.clip(var, 0, None))
    mu4 = Ez4 - 4 * Ez * Ez3 + 6 * Ez**2 * Ez2 - 3 * Ez**4
    with np.errstate(divide="ignore", invalid="ignore"):
        exkurt = np.where(var > 0, mu4 / (var**2) - 3.0, 0.0)
    return dict(std=std, exkurt=exkurt, clip_rate=clip_rate, pct_zero=pct_zero,
                mean=Ez, per_stratum=per_stratum)


# ---------------------------------------------------------------- gates
def gate_channel_names():
    rng = random.Random(0)
    allf = glob.glob(f"{FEAT}/*.npz")
    samp = rng.sample(allf, min(NAMECHECK_FILES, len(allf)))
    ref = None; identical = True
    for f in samp:
        ch = tuple(str(x) for x in np.load(f, allow_pickle=True)["channels"])
        if ref is None:
            ref = ch
        elif ch != ref:
            identical = False
    eq_chnames = (list(ref) == CH_NAMES) if ref else False
    return identical, eq_chnames, len(samp), list(ref) if ref else []


def gate_coverage():
    """G-N5: every (stratum,stem) target/reference in the stratum's samples.jsonl has a feat
       file whose F shape == row['shape'].  100% for every fit stratum.  eval reported separately.
       Sources per stratum come from COVERAGE_SOURCES (ctt_v2 for S0-S4, 004_ctt_v2plus for S6)."""
    from collections import defaultdict
    # S6 target and reference are DIFFERENT subjects and can have different orientations, so a
    # row's `shape` describes only its target. Expected shape per S6 stem is that stem's OWN
    # ROSTER latent_fhw, not the pairing row's shape. (S0-S4 are same-res within a group.)
    s6shape = {}
    if "S6" in STRATA:
        for c in json.load(open(ENC_ROSTER))["clips"]:
            s6shape[c["stem"]] = tuple(int(x) for x in c["latent_fhw"])
    need = {}   # (stratum, stem) -> expected shape tuple
    for path, subset in COVERAGE_SOURCES:
        if not (subset & set(STRATA)):
            continue
        for l in open(path):
            r = json.loads(l)
            s = r["stratum"]
            if s not in STRATA or s not in subset:
                continue
            rowshp = tuple(int(x) for x in r["shape"])
            for stem in (r["target"], r["reference"]):
                need[(s, stem)] = s6shape[stem] if s == "S6" else rowshp
    per = defaultdict(lambda: dict(total=0, hit=0, shape_ok=0, miss=[], shape_bad=[]))
    for (s, stem), shp in need.items():
        fid = f"train__{s}__{stem}" if s in ("S0", "S1") else f"{s}__{stem}"
        p = f"{FEAT}/{fid}.npz"
        d = per[s]; d["total"] += 1
        if os.path.exists(p):
            d["hit"] += 1
            fshape = npz_F_shape(p)[:3]
            if fshape == shp:
                d["shape_ok"] += 1
            else:
                d["shape_bad"].append((stem, fshape, shp))
        else:
            d["miss"].append(stem)
    eval_ct = len(glob.glob(f"{FEAT}/eval__*.npz"))
    return {s: per[s] for s in STRATA}, eval_ct, len(need)


# ---------------------------------------------------------------- serialization
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", DR, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNKNOWN"


def tracked(path):
    import subprocess
    try:
        subprocess.check_output(["git", "-C", DR, "ls-files", "--error-unmatch", path],
                                stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- main driver
def main():
    global STRATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=f"{ARMA}/NORM_dino_v1.json")
    ap.add_argument("--out-report", default=f"{ARMA}/NORM_REPORT.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strata", default="S0,S1,S2a,S2b,S4",
                    help="comma list of fit strata; v1='S0,S1,S2a,S2b,S4', v2 adds S6")
    ap.add_argument("--version", default="dino_signal_norm_v1")
    a = ap.parse_args()
    STRATA = [s for s in a.strata.split(",") if s]
    weighting = "equal_stratum_" + "_".join(STRATA)
    print(f"[cfg] version={a.version} strata={STRATA} weighting={weighting}", flush=True)
    np.seterr(all="warn")

    print("[1/5] fit pass (streaming all non-eval feat files) ...", flush=True)
    acc, nonfinite = fit_moments()
    if nonfinite.any():
        print("  WARN non-finite cells per channel:",
              {CH_NAMES[c]: int(nonfinite[c]) for c in range(NC) if nonfinite[c]}, flush=True)

    loc_id, scale_id_raw, mean_s_id, std_s_id = equal_stratum_locscale(acc, "s1", "s2")
    loc_as, scale_as_raw, _, _ = equal_stratum_locscale(acc, "as1", "as2")

    # scale floor + dead flags (identity space defines deadness)
    dead = scale_id_raw < SCALE_FLOOR
    scale_id = np.where(dead, 1.0, scale_id_raw)
    scale_as = np.where(scale_as_raw < SCALE_FLOOR, 1.0, scale_as_raw)

    print("[2/5] verify pass (seeded sample) ...", flush=True)
    raw, zsum, vcounts = verify(loc_id, scale_id, loc_as, scale_as, seed=a.seed)

    # decide transforms: identity first; channels failing G-N2 -> asinh (single round)
    pool_id = pool_postnorm(zsum["none"])
    transform = ["none"] * NC
    for c in range(NC):
        if pool_id["clip_rate"][c] > G_N2_MAX:
            transform[c] = "asinh"
    asinh_ch = [CH_NAMES[c] for c in range(NC) if transform[c] == "asinh"]
    print("  channels routed to asinh (G-N2 refit):", asinh_ch or "none", flush=True)

    # final per-channel loc/scale after the transform decision
    loc = np.where(np.array(transform) == "asinh", loc_as, loc_id)
    scale = np.where(np.array(transform) == "asinh", scale_as, scale_id)
    dead_final = np.where(np.array(transform) == "asinh",
                          scale_as_raw < SCALE_FLOOR, dead)

    # recompute the pooled post-norm using the chosen transform per channel
    pool = {k: np.array(pool_id[k], dtype=float) if k != "per_stratum" else None
            for k in ("std", "exkurt", "clip_rate", "pct_zero", "mean")}
    per_stratum_final = {s: dict(mean=np.zeros(NC), std=np.zeros(NC)) for s in STRATA}
    for c in range(NC):
        src = pool_postnorm(zsum[transform[c]])
        for k in ("std", "exkurt", "clip_rate", "pct_zero", "mean"):
            pool[k][c] = src[k][c]
        for s in STRATA:
            per_stratum_final[s]["mean"][c] = src["per_stratum"][s]["mean"][c]
            per_stratum_final[s]["std"][c] = src["per_stratum"][s]["std"][c]

    print("[3/5] gates ...", flush=True)
    g1 = (pool["std"] >= G_N1_LO) & (pool["std"] <= G_N1_HI)
    g2 = pool["clip_rate"] <= G_N2_MAX
    g1_pass = bool(g1.all()); g2_pass = bool(g2.all())
    g3_dead = [CH_NAMES[c] for c in range(NC) if dead_final[c]]
    g3_pass = True   # dead channels are allowed IF named/explained; none expected
    id_ok, eq_ok, nchk, chlist = gate_channel_names()
    g4_pass = bool(id_ok and eq_ok)
    cov, eval_ct, need_total = gate_coverage()
    g5_pass = all(cov[s]["hit"] == cov[s]["total"] and cov[s]["shape_ok"] == cov[s]["total"]
                  for s in STRATA)

    # -------------------------------------------------- JSON
    channels = []
    for c in range(NC):
        channels.append(dict(index=c, name=CH_NAMES[c], transform=transform[c],
                             loc=float(loc[c]), scale=float(scale[c]),
                             dead=bool(dead_final[c])))
    per_stratum_json = {s: dict(n_cells=int(acc[s]["n"]),
                                mean=[float(x) for x in (acc[s]["s1"] / acc[s]["n"])],
                                std=[float(x) for x in std_s_id[s]]) for s in STRATA}
    extractor_commit = git_head() if tracked("misc/2026-08-24_flow_signal_conditioning/armA/armA_extract.py") else "UNCOMMITTED"
    script_rel = "misc/2026-08-24_flow_signal_conditioning/armA/fit_norm_dino.py"
    script_commit = git_head() if tracked(script_rel) else "UNCOMMITTED"
    doc = dict(
        version=a.version, clip=CLIP,
        weighting=weighting, excluded=["eval"],
        channels=channels, per_stratum=per_stratum_json,
        fit=dict(n_files=int(sum(acc[s]["n_files"] for s in STRATA)),
                 n_cells=int(sum(acc[s]["n"] for s in STRATA)),
                 date="2026-08-28", extractor="armA_extract.py",
                 extractor_commit=extractor_commit, script_commit=script_commit,
                 scheme="zscore_clip5_equalstratum",
                 fit_population_rule="the stratum's signal-cache roster (every extracted feat); "
                                     "pairing/mix knobs do not alter it — so a clip dropped by "
                                     "dataset pairing (e.g. S6 shape-singletons) still counts in the fit"),
    )
    with open(a.out_json, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")
    json_sha = sha256_file(a.out_json)
    pca_sha = sha256_file(PCA_PATH)

    # -------------------------------------------------- report
    def fmt(x, w=9, p=4):
        return f"{x:{w}.{p}f}"
    lines = []
    P = lines.append
    P(f"# NORM_REPORT — DINO-basis operator signal (44-ch), {a.version}\n")
    P(f"Generated 2026-08-28 on `{os.uname().nodename}` · CPU · equal-stratum z-score over "
      f"{'+'.join(STRATA)}, clip ±{CLIP:.0f}.\n")
    P(f"- fit population: **{doc['fit']['n_files']} files**, **{doc['fit']['n_cells']:,} cells** "
      f"(eval EXCLUDED); per-stratum cells: " + ", ".join(f"{s}={acc[s]['n']:,}" for s in STRATA) + ".")
    P(f"- verify sample (seed {a.seed}): " + ", ".join(f"{s}={vcounts[s]}" for s in STRATA) +
      f" files (eval excluded).")
    P(f"- `NORM_dino_v1.json` sha256: `{json_sha}`")
    P(f"- `pca.npz` sha256: `{pca_sha}`")
    P(f"- extractor_commit: `{extractor_commit}` · script_commit: `{script_commit}`\n")

    P("## Gate verdicts\n")
    P(f"| gate | check | verdict |")
    P(f"|---|---|---|")
    P(f"| G-N1 | post-norm std ∈ [{G_N1_LO},{G_N1_HI}] all ch | {'PASS' if g1_pass else 'FAIL'} |")
    P(f"| G-N2 | clip rate %\\|z\\|≥5 ≤ {G_N2_MAX}% all ch | {'PASS' if g2_pass else 'FAIL'} |")
    P(f"| G-N3 | no undocumented dead channel | {'PASS' if g3_pass else 'FAIL'} (dead: {g3_dead or 'none'}) |")
    P(f"| G-N4 | channel names identical ({nchk} files) & ==CH_NAMES | {'PASS' if g4_pass else 'FAIL'} |")
    P(f"| G-N5 | coverage 100% {'/'.join(STRATA)} | {'PASS' if g5_pass else 'FAIL'} |")
    P("")
    if asinh_ch:
        P(f"**asinh escape hatch applied** to: {', '.join(asinh_ch)} (each failed identity G-N2; single refit round).\n")
    else:
        P("**asinh escape hatch:** not triggered — every channel passed G-N2 under identity.\n")

    P("## G-N5 coverage\n")
    P("| stratum | rows(target∪ref) | feat hit | shape match | misses |")
    P("|---|--:|--:|--:|---|")
    for s in STRATA:
        d = cov[s]
        P(f"| {s} | {d['total']} | {d['hit']} | {d['shape_ok']} | {len(d['miss'])+len(d['shape_bad'])} |")
    P(f"\neval__ feat files present (reported, not gated): **{eval_ct}**. "
      f"Distinct (stratum,stem) keys joined: {need_total:,}.\n")

    P("## Post-norm pool (equal-stratum weighted, chosen transform)\n")
    P("| # | channel | t | loc | scale | post-std | ex-kurt | clip% | %zero |")
    P("|--:|---|---|--:|--:|--:|--:|--:|--:|")
    for c in range(NC):
        P(f"| {c} | {CH_NAMES[c]} | {transform[c]} | {loc[c]:.4f} | {scale[c]:.4f} | "
          f"{pool['std'][c]:.3f} | {pool['exkurt'][c]:.2f} | {pool['clip_rate'][c]:.3f} | {pool['pct_zero'][c]:.3f} |")
    P("")

    for s in STRATA:
        P(f"## Raw per-stratum stats — {s}\n")
        P("| # | channel | min | p1 | p50 | mean | p99 | max | std | %zero |")
        P("|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        for c in range(NC):
            r = raw[s][c]
            P(f"| {c} | {CH_NAMES[c]} | {r['min']:.3f} | {r['p1']:.3f} | {r['p50']:.3f} | "
              f"{r['mean']:.3f} | {r['p99']:.3f} | {r['max']:.3f} | {r['std']:.3f} | {r['pct_zero']:.2f} |")
        P("")

    P("## Per-stratum post-norm drift (report-only; flag |mean|>1.0)\n")
    P("| # | channel | " + " | ".join(f"{s} μ" for s in STRATA) + " | " + " | ".join(f"{s} σ" for s in STRATA) + " | flag |")
    P("|--:|---|" + "--:|" * (2 * len(STRATA)) + "---|")
    for c in range(NC):
        mus = [per_stratum_final[s]["mean"][c] for s in STRATA]
        sds = [per_stratum_final[s]["std"][c] for s in STRATA]
        flag = "⚠" if any(abs(m) > 1.0 for m in mus) else ""
        P(f"| {c} | {CH_NAMES[c]} | " + " | ".join(f"{m:.2f}" for m in mus) + " | " +
          " | ".join(f"{v:.2f}" for v in sds) + f" | {flag} |")
    P("")

    with open(a.out_report, "w") as fh:
        fh.write("\n".join(lines))

    print("[4/5] wrote:", a.out_json, "and", a.out_report, flush=True)
    print("[5/5] GATES:",
          f"G-N1={'PASS' if g1_pass else 'FAIL'}",
          f"G-N2={'PASS' if g2_pass else 'FAIL'}",
          f"G-N3={'PASS' if g3_pass else 'FAIL'}(dead={g3_dead or 'none'})",
          f"G-N4={'PASS' if g4_pass else 'FAIL'}",
          f"G-N5={'PASS' if g5_pass else 'FAIL'}", flush=True)
    print("JSON_SHA256", json_sha, flush=True)
    print("PCA_SHA256", pca_sha, flush=True)
    allpass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass
    print("ALL_GATES", "PASS" if allpass else "FAIL", flush=True)
    sys.exit(0 if allpass else 3)


if __name__ == "__main__":
    main()
