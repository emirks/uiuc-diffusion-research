"""A2 - PIX descriptors (SPEC 3) + PIX landmarks (SPEC 4) -> results/pix_per_clip.csv.

Byte-for-byte the Sep-08 instrument for the 128px/blur decode: S-GRID-family main clips use
instrument.load_matrix directly (so the regression check matches to ~1e-3). S-SWEEP main and
all landmark clips are decoded with cv2 (single-threaded) and pushed through the same
resize-128 + Gaussian-blur pipeline. Landmarks (LERP/CUT50/FREEZE) are built at native
resolution from each landmark-source clip's own frames a_idx/b_idx (SPEC 4).

Also: fills the manifest's static_pix column, runs the SPEC 3 regression check against the
Sep-08 per_clip.csv (DR/M) and the distance_vs_cut cutstats.csv (step_share), and prints the
SPEC 4 self-checks (LERP DR, CUT50 cross/step_share).

    python misc/2026-09-13_null_default/scripts/pix_features.py [--nproc 4]
"""
from __future__ import annotations

import argparse
import os
import sys

# single-threaded BLAS/OMP: this login node's per-process thread limit is tiny; without this
# OpenBLAS tries to spawn 128 threads per Pool worker and thread creation fails.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

sys.path.insert(0, str(C.REMEASURE))
from instrument import load_matrix  # noqa: E402

import cv2  # noqa: E402
cv2.setNumThreads(1)

SIZE, BLUR = 128, 1.0
RESULTS = C.CAMP / "results"


# --------------------------------------------------------------------------- 128px pipeline
def frames_to_matrix(frames):
    """(T,H,W,3) uint8/float -> (T, 128*128*3) float32 in [0,1], INTER_AREA + blur sigma=1.

    Same pipeline as instrument.load_matrix, applied frame-by-frame so a native landmark clip
    never has to be fully materialised at high resolution."""
    out = np.empty((len(frames), SIZE * SIZE * 3), np.float32)
    for i, f in enumerate(frames):
        g = cv2.resize(np.asarray(f, np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), BLUR)
        out[i] = (g / 255.0).reshape(-1)
    return out


# --------------------------------------------------------------------------- descriptors (SPEC 3)
def measure_pix(M, a_idx, b_idx, eps=1e-8):
    T = M.shape[0]
    a, b = M[a_idx], M[b_idx]
    u = b - a
    gap = float(np.linalg.norm(u))
    gap_rel = float(gap / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + eps))
    out = dict(T=T, a_idx=a_idx, b_idx=b_idx, gap=gap, gap_rel=gap_rel, static=bool(gap_rel < 0.12))
    X = M[a_idx + 1:b_idx]                                   # interior, open
    n = X.shape[0]
    out["n_interior"] = int(n)
    seg = M[a_idx:b_idx + 1]                                 # closed [a,b] for step stats
    d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    path = float(d.sum() + 1e-8)
    out["step_share"] = float(d.max() / path)
    out["step_pos"] = float(d.argmax() / len(d))
    out["path_over_gap"] = float(path / (gap + eps))
    for i, s in enumerate(np.array_split(d, 10)):
        out[f"speed_{i}"] = float(s.sum() / path)
    nan_keys = (["DR", "DR_mean", "M", "cross", "nu_max", "nu_mean", "explained", "tau_med"]
                + [f"tau_{i}" for i in range(10)])
    if gap < eps or n < 3:
        for k in nan_keys:
            out[k] = np.nan
        return out
    uhat = u / gap
    V = X - a
    t = V @ uhat
    resid = np.linalg.norm(V - np.outer(t, uhat), axis=1) / gap
    tau = t / gap
    out["DR"] = float(np.median(resid))
    out["DR_mean"] = float(resid.mean())
    out["M"] = float(((tau >= 0.25) & (tau <= 0.75)).mean())
    ge = np.where(tau >= 0.5)[0]
    out["cross"] = float(ge[0] / n) if len(ge) else 1.0
    out["tau_med"] = float(np.median(tau))
    for i, s in enumerate(np.array_split(tau, 10)):
        out[f"tau_{i}"] = float(s.mean()) if len(s) else np.nan
    da = np.linalg.norm(X - a, axis=1)
    db = np.linalg.norm(X - b, axis=1)
    nu = np.minimum(da, db) / gap
    out["nu_max"] = float(nu.max())
    out["nu_mean"] = float(nu.mean())
    out["explained"] = float((nu < 0.15).mean())
    return out


# --------------------------------------------------------------------------- per-clip worker
def work(row):
    cv2.setNumThreads(1)
    stratum, group = row["stratum"], row["group"]
    a_idx, b_idx = int(row["a_idx"]), int(row["b_idx"])
    ds = C.sweep_downscale_wh() if stratum == "S-SWEEP" else None
    recs = []
    try:
        # main matrix: instrument-exact for the S-GRID family, cv2+downscale for S-SWEEP
        if stratum == "S-SWEEP":
            M = frames_to_matrix(C.decode_rgb(C.REPO / row["path"], downscale_wh=ds))
        else:
            M = load_matrix(str(C.REPO / row["path"]))
        m = measure_pix(M, a_idx, b_idx)
        recs.append(dict(clip_id=row["clip_id"], kind="main", **_meta(row), **m))
        # landmarks from this clip's own native frames
        if C.is_landmark_source(stratum, group):
            fr = C.decode_rgb(C.REPO / row["path"], downscale_wh=ds)
            lms = C.build_pixel_landmarks(fr, a_idx, b_idx)
            for kind, clip in lms.items():
                Ml = frames_to_matrix(clip)
                ml = measure_pix(Ml, a_idx, b_idx)
                recs.append(dict(clip_id=row["clip_id"], kind=kind, **_meta(row), **ml))
    except Exception as e:
        recs.append(dict(clip_id=row["clip_id"], kind="ERROR", **_meta(row),
                         error=f"{type(e).__name__}: {e}"))
    return recs


def _meta(row):
    return dict(stratum=row["stratum"], group=row["group"], endpoint=row["endpoint"],
                seed=row["seed"], md5=row["md5"], grid=row["grid"], tier=row["tier"])


# --------------------------------------------------------------------------- checks
def regression_check(df_main):
    """SPEC 3: PIX DR/M vs Sep-08 per_clip.csv and step_share vs cutstats.csv, by md5."""
    pc = pd.read_csv(C.REMEASURE / "per_clip.csv")[["md5", "DR_med", "M"]].drop_duplicates("md5")
    pc = pc.rename(columns={"DR_med": "DR_ref", "M": "M_ref"})
    cut = pd.read_csv(C.REPO / "misc/2026-08-24_lerp_collapse/distance_vs_cut/cutstats.csv")
    cut = cut[["md5", "step_share"]].drop_duplicates("md5").rename(columns={"step_share": "step_share_ref"})
    grid = df_main[df_main.stratum.isin(["S-GRID", "S-GRID-F", "S-GRID-START"])]
    sub = grid.merge(pc, on="md5", how="inner")
    dev = {}
    if len(sub):
        dev["DR_n"] = len(sub)
        dev["DR_maxabs"] = float((sub.DR - sub.DR_ref).abs().max())
        dev["M_maxabs"] = float((sub.M - sub.M_ref).abs().max())
    subc = grid.merge(cut, on="md5", how="inner")
    if len(subc):
        dev["step_n"] = len(subc)
        dev["step_maxabs"] = float((subc.step_share - subc.step_share_ref).abs().max())
    return dev, sub, subc


def self_checks(df):
    out = {}
    for kind in C.PIX_LANDMARKS:
        s = df[df.kind == kind]
        if not len(s):
            continue
        out[kind] = dict(n=len(s), DR_med=float(np.nanmedian(s.DR)), DR_max=float(np.nanmax(s.DR)),
                         cross_med=float(np.nanmedian(s.cross)), step_share_med=float(np.nanmedian(s.step_share)))
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    man = C.load_manifest()
    rows = man.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    print(f"[pix] {len(rows)} manifest clips, nproc={args.nproc}", flush=True)

    recs = []
    with Pool(args.nproc) as pool:
        for i, rr in enumerate(pool.imap_unordered(work, rows, chunksize=4)):
            recs.extend(rr)
            if (i + 1) % 100 == 0:
                print(f"  ..{i+1}/{len(rows)}", flush=True)
    df = pd.DataFrame(recs)
    errs = df[df.kind == "ERROR"]
    if len(errs):
        print(f"[pix] {len(errs)} ERRORS:")
        print(errs[["clip_id", "error"]].to_string())
    df = df[df.kind != "ERROR"].copy()
    df.to_csv(RESULTS / "pix_per_clip.csv", index=False)
    print(f"[pix] wrote {len(df)} rows -> {RESULTS/'pix_per_clip.csv'}")

    # fill static_pix in the manifest
    main_df = df[df.kind == "main"].set_index("clip_id")
    man["static_pix"] = man.clip_id.map(main_df["static"]).astype("boolean")
    man.to_csv(C.MANIFEST_CSV, index=False)
    print(f"[pix] static_pix filled in manifest; static count = {int(man['static_pix'].sum())}/{len(man)}")

    # regression check
    dev, _, _ = regression_check(df[df.kind == "main"])
    print("\n[pix] SPEC-3 regression check (vs Sep-08 per_clip.csv + cutstats.csv, matched md5):")
    print("   " + str(dev))

    # self checks
    sc = self_checks(df)
    print("\n[pix] SPEC-4 landmark self-checks (bars: LERP DR<0.02; CUT50 cross~0.5, step_share~1):")
    for k, v in sc.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
