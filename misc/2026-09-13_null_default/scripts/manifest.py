"""A1 - build misc/2026-09-13_null_default/manifest.csv.

Enumerates the five strata (SPEC 1), md5-dedups the byte-identical base_cond rows, marks the
davis 'foreign' rows the way score_store.is_foreign did, and records the window (a_idx/b_idx)
each stratum's scoring contract implies. static_pix is left blank here and filled by
pix_features.py (A2).

Run from the repo root with the aarch64 env active:
    python misc/2026-09-13_null_default/scripts/manifest.py [--verify-probe]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

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
from instrument import md5_file  # noqa: E402

os.chdir(C.REPO)

COLS = ["clip_id", "stratum", "group", "tier", "endpoint", "seed", "path", "T", "a_idx",
        "b_idx", "two_sided", "md5", "static_pix", "twin_clip_id", "prompt_cat", "grid", "notes"]

NEUTRAL_VARIANTS = {"02_neutral__dai": "v2", "04_neutral_v3__dai": "v3"}
PROBE = C.REPO / "misc" / "2026-09-08_collapse_probe"
SWEEP = C.REPO / "outputs/videos/exp_024_ltx2_prompt_sweep/run_0003"


# --------------------------------------------------------------------------- md5 helper
def md5_of(path):
    try:
        return md5_file(str(C.REPO / path)) if not os.path.isabs(str(path)) else md5_file(str(path))
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def md5_many(paths, nproc=4):
    with Pool(nproc) as pool:
        return dict(zip(paths, pool.map(md5_of, paths)))


# --------------------------------------------------------------------------- S-GRID family
def grid_rows():
    """S-GRID (both, clean) + S-GRID-F (both, foreign) + S-GRID-START (start, clean).

    Reuses the Sep-08 per_clip.csv (already joined item->grid with sided/endpoint/foreign/md5/
    window), so the md5s match the regression check by construction.
    """
    pc = pd.read_csv(C.REMEASURE / "per_clip.csv")
    pc = pc[pc.variant.isin(NEUTRAL_VARIANTS)].copy()
    pc["grid"] = pc.variant.map(NEUTRAL_VARIANTS)
    rows = []
    # ---- S-GRID (both, clean) and S-GRID-F (both, foreign) ------------------
    for (variant, foreign), g in pc[pc.condition == "both"].groupby(["variant", "foreign"]):
        grid = NEUTRAL_VARIANTS[variant]
        stratum = "S-GRID-F" if foreign else "S-GRID"
        rep = g.sort_values(["item", "seed"]).drop_duplicates("md5")   # one row per byte-identical group
        for _, r in rep.iterrows():
            dupes = int((g.md5 == r.md5).sum())
            cid = ("gridF" if foreign else "null") + f"_{grid}__{r.endpoint}__s{int(r.seed)}"
            rows.append(dict(
                clip_id=cid, stratum=stratum, group="NULL", tier="", endpoint=r.endpoint,
                seed=int(r.seed), path=r.path, T=int(r["T"]), a_idx=int(r.a_idx), b_idx=int(r.b_idx),
                two_sided=bool(r.two_sided), md5=r.md5, static_pix="",
                twin_clip_id=("" if foreign else f"GT__{r.endpoint}"),
                prompt_cat="neutral", grid=grid,
                notes=f"dedup_group={dupes};endpoint_class={r.endpoint_class}"))
    # ---- S-GRID-START (start-only, clean) ----------------------------------
    for variant, g in pc[(pc.condition == "start") & (~pc.foreign)].groupby("variant"):
        grid = NEUTRAL_VARIANTS[variant]
        rep = g.sort_values(["item", "seed"]).drop_duplicates("md5")
        for _, r in rep.iterrows():
            dupes = int((g.md5 == r.md5).sum())
            cid = f"start_{grid}__{r.endpoint}__s{int(r.seed)}"
            rows.append(dict(
                clip_id=cid, stratum="S-GRID-START", group="NULL", tier="", endpoint=r.endpoint,
                seed=int(r.seed), path=r.path, T=int(r["T"]), a_idx=int(r.a_idx), b_idx=int(r.b_idx),
                two_sided=False, md5=r.md5, static_pix="", twin_clip_id="",
                prompt_cat="neutral", grid=grid,
                notes=f"dedup_group={dupes};start_anchor={r.endpoint}_start9;endpoint_class={r.endpoint_class}"))
    return rows, pc


# --------------------------------------------------------------------------- GT twins
def gt_rows(pc):
    """One row per unique endpoint used by S-GRID (both, clean). GT clip via recursive glob."""
    both = pc[(pc.condition == "both") & (~pc.foreign)]
    eps = sorted(both.endpoint.unique())
    rows, missing = [], []
    for ep in eps:
        hits = glob.glob(f"data/processed/transitions_std121/**/{ep}.mp4", recursive=True)
        if not hits:
            missing.append(ep)
            continue
        path = sorted(hits)[0]
        rows.append(dict(
            clip_id=f"GT__{ep}", stratum="S-GRID", group="GT", tier="", endpoint=ep, seed="",
            path=path, T=121, a_idx=8, b_idx=113, two_sided=True, md5=None, static_pix="",
            twin_clip_id="", prompt_cat="real", grid="",
            notes=f"gt_twin;class={Path(path).parent.name}" + (f";multi={len(hits)}" if len(hits) > 1 else "")))
    return rows, missing


# --------------------------------------------------------------------------- S-PROBE
def probe_rows():
    """R1 (start-only, full prompt) + R2 (both, full) + R3 (both, captions). 40 prompts x 3 seeds."""
    import json
    prompts = [json.loads(l) for l in open(PROBE / "prompts" / "prompts.jsonl")]
    TIER = {"high": "high", "low": "inplace"}
    rows = []
    for p in prompts:
        pid, ep = p["prompt_id"], p["endpoint"]
        tier = TIER[p["tier"]]
        for seed in (42, 43, 44):
            r1 = PROBE / "out/r1/base_cond_neutral" / f"probe__{pid}__{ep}__s{seed}.mp4"
            r2 = PROBE / "out/r2/base_cond_neutral" / f"probe_r2__{pid}__s{seed}.mp4"
            r3 = PROBE / "out/r3/base_cond_neutral" / f"probe_r3__{pid}__s{seed}.mp4"
            r1id = f"probe__{pid}__{ep}__s{seed}"
            if r1.exists():
                rows.append(dict(clip_id=r1id, stratum="S-PROBE", group="R1", tier=tier,
                                 endpoint=ep, seed=seed, path=os.path.relpath(r1, C.REPO), T=121,
                                 a_idx=8, b_idx=120, two_sided=False, md5=None, static_pix="",
                                 twin_clip_id="", prompt_cat="full", grid="",
                                 notes=f"prompt_id={pid};witness"))
            if r2.exists():
                rows.append(dict(clip_id=f"probe_r2__{pid}__s{seed}", stratum="S-PROBE", group="R2",
                                 tier=tier, endpoint=ep, seed=seed, path=os.path.relpath(r2, C.REPO),
                                 T=121, a_idx=8, b_idx=113, two_sided=True, md5=None, static_pix="",
                                 twin_clip_id=r1id, prompt_cat="full", grid="", notes=f"prompt_id={pid}"))
            if r3.exists():
                rows.append(dict(clip_id=f"probe_r3__{pid}__s{seed}", stratum="S-PROBE", group="R3",
                                 tier=tier, endpoint=ep, seed=seed, path=os.path.relpath(r3, C.REPO),
                                 T=121, a_idx=8, b_idx=113, two_sided=True, md5=None, static_pix="",
                                 twin_clip_id=r1id, prompt_cat="captions", grid="", notes=f"prompt_id={pid}"))
    return rows


# --------------------------------------------------------------------------- S-SWEEP
def sweep_rows():
    """10 DAVIS pairs x 6 categories x 1 seed. T verified per file; a_idx=24, b_idx=T-25."""
    import cv2
    cv2.setNumThreads(1)
    CATS = ["A_empty", "A_word", "B", "C", "D", "E"]
    PROMPT_CAT = {"A_empty": "empty", "A_word": "word", "B": "generic", "C": "abstract",
                  "D": "typed", "E": "timed"}
    rows = []
    for pair in sorted(os.listdir(SWEEP)):
        pdir = SWEEP / pair
        if not pdir.is_dir():
            continue
        for cat in CATS:
            mp4s = sorted(glob.glob(str(pdir / cat / "s*_cat*_steps40.mp4")))
            mp4s = [m for m in mp4s if "stage1" not in m]
            for m in mp4s:
                seed = int(Path(m).stem.split("_")[0][1:])   # s42_... -> 42
                cap = cv2.VideoCapture(m)
                T = 0
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    T += 1
                cap.release()
                a_idx, b_idx = 24, T - 25
                rows.append(dict(
                    clip_id=f"sweep__{pair}__{cat}__s{seed}", stratum="S-SWEEP", group=cat, tier="",
                    endpoint=pair, seed=seed, path=os.path.relpath(m, C.REPO), T=T, a_idx=a_idx,
                    b_idx=b_idx, two_sided=True, md5=None, static_pix="", twin_clip_id="",
                    prompt_cat=PROMPT_CAT[cat], grid="", notes=f"davis_pair;cats=25condframes/end"))
    return rows


# --------------------------------------------------------------------------- probe window verify
def verify_probe(n_pairs=3):
    """SPEC A1: R1[112:121] vs R3 suffix pixel equality + the implied a_idx/b_idx.

    R2/R3 are scored two-sided with prefix=9 suffix=8 -> a_idx=8, b_idx=T-8=113. The end anchor
    is the cut of R1 frames 112..120 (splice_r1.py). We check where R3's tail best matches
    R1[112:121] and report the frame-aligned MAE (0-255).
    """
    import cv2
    cv2.setNumThreads(1)

    def dec(p):
        cap = cv2.VideoCapture(str(p)); fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(f)
        cap.release()
        return np.stack(fr).astype(np.float32)

    r1files = sorted(glob.glob(str(PROBE / "out/r1/base_cond_neutral/*.mp4")))
    lines = []
    done = 0
    for f in r1files:
        stem = Path(f).stem                    # probe__P01H__wireframe_7__s42
        parts = stem.split("__")
        pid, seed = parts[1], parts[-1]
        r3 = PROBE / "out/r3/base_cond_neutral" / f"probe_r3__{pid}__{seed}.mp4"
        if not r3.exists():
            continue
        A, B = dec(f), dec(str(r3))
        a = A[112:121]
        best = min(range(108, 113), key=lambda o: float(np.abs(a - B[o:o + 9]).mean()))
        best_mae = float(np.abs(a - B[best:best + 9]).mean())
        same_mae = float(np.abs(a - B[112:121]).mean())
        lines.append(f"  {pid} {seed}: R1frames={len(A)} R3frames={len(B)} "
                     f"best-align R3[{best}:{best+9}] MAE={best_mae:.2f}  "
                     f"same-index R3[112:121] MAE={same_mae:.2f} (0-255)")
        done += 1
        if done >= n_pairs:
            break
    return lines


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-probe", action="store_true")
    args = ap.parse_args()

    grid, pc = grid_rows()
    gt, missing = gt_rows(pc)
    probe = probe_rows()
    sweep = sweep_rows()
    rows = grid + gt + probe + sweep

    # md5 for GT / probe / sweep (grid md5s came from per_clip.csv)
    need = [r["path"] for r in rows if r["md5"] is None]
    m = md5_many(need, nproc=4)
    for r in rows:
        if r["md5"] is None:
            r["md5"] = m[r["path"]]

    df = pd.DataFrame(rows, columns=COLS)
    assert df.clip_id.is_unique, "duplicate clip_id:\n" + str(df.clip_id[df.clip_id.duplicated()].tolist())
    df.to_csv(C.MANIFEST_CSV, index=False)

    print(f"[manifest] {len(df)} clips -> {C.MANIFEST_CSV}")
    print("\nper stratum x group:")
    print(df.groupby(["stratum", "group"]).size().to_string())
    print("\nper stratum x grid (S-GRID family):")
    print(df[df.grid != ""].groupby(["stratum", "grid"]).size().to_string())
    print("\nS-PROBE tier x group:")
    print(df[df.stratum == "S-PROBE"].groupby(["tier", "group"]).size().to_string())
    print("\nlandmark-source rows:", sum(C.is_landmark_source(r.stratum, r.group) for r in df.itertuples()))
    err = df[df.md5.astype(str).str.startswith("ERR")]
    print("md5 errors:", len(err))
    if missing:
        print("GT twins MISSING for endpoints:", missing)
    print("T values present:", sorted(df["T"].unique().tolist()))

    if args.verify_probe:
        print("\n[verify-probe] R1[112:121] vs R3 tail (a_idx=8, b_idx=113 for R2/R3):")
        for l in verify_probe():
            print(l)


if __name__ == "__main__":
    main()
