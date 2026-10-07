"""C4 step 0 - integrity pass over every Phase B npz.

For every manifest clip (main) and its landmark set (LERP/CUT50/FREEZE/LATLERP for landmark
sources), check:
  - the npz exists;
  - main + LERP/CUT50/FREEZE carry dino/vae/trans; LATLERP carries vae only;
  - dino.shape == (T,768)   (a truncated read-stall decode shows as dino.shape[0] < T);
  - trans.shape[0] == n_lat = (T-1)//8+1 ; vae.shape[0] == n_lat ;
  - spatial dims (H,W) == (20,15) for 121-f clips, (16,24) for S-SWEEP;
  - every stored array finite.

Writes results/integrity.txt (counts + bad-file list). Read-only; no features touched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

RESULTS = C.CAMP / "results"
FULL_KINDS = ("main", "LERP", "CUT50", "FREEZE")   # carry dino+vae+trans
LAT_KIND = "LATLERP"                                # vae only


def npz_path(clip_id, kind):
    return C.FEATURES / (f"{clip_id}.npz" if kind == "main" else f"{clip_id}__{kind}.npz")


def expected_hw(stratum):
    return (16, 24) if stratum == "S-SWEEP" else (20, 15)


def check_one(path, kind, stratum):
    """Return list of problem strings (empty = clean)."""
    probs = []
    if not path.exists():
        return ["MISSING"]
    try:
        z = np.load(path)
    except Exception as e:
        return [f"LOAD_FAIL {type(e).__name__}: {e}"]
    files = set(z.files)
    for req in ("a_idx", "b_idx", "T"):
        if req not in files:
            probs.append(f"no {req}")
    if probs:
        return probs
    T = int(z["T"])
    n_lat = (T - 1) // 8 + 1
    H, W = expected_hw(stratum)

    if kind == LAT_KIND:
        want = {"vae", "a_idx", "b_idx", "T"}
        if files != want:
            probs.append(f"keys {sorted(files)} != {sorted(want)} (LATLERP=vae only)")
        arrays = ["vae"]
    else:
        want = {"dino", "vae", "trans", "a_idx", "b_idx", "T"}
        missing = want - files
        extra = files - want
        if missing:
            probs.append(f"missing keys {sorted(missing)}")
        if extra:
            probs.append(f"unexpected keys {sorted(extra)}")
        arrays = [k for k in ("dino", "vae", "trans") if k in files]

    # shapes
    if "dino" in files:
        d = z["dino"]
        if d.shape != (T, 768):
            probs.append(f"dino.shape {d.shape} != {(T, 768)}")
    if "vae" in files:
        v = z["vae"]
        if v.ndim != 4 or v.shape[0] != n_lat or v.shape[1] != 128 or tuple(v.shape[2:]) != (H, W):
            probs.append(f"vae.shape {v.shape} != {(n_lat, 128, H, W)}")
    if "trans" in files:
        t = z["trans"]
        if t.ndim != 4 or t.shape[0] != n_lat or tuple(t.shape[1:3]) != (H, W) or t.shape[3] != 44:
            probs.append(f"trans.shape {t.shape} != {(n_lat, H, W, 44)}")

    # finiteness
    for k in arrays:
        arr = z[k]
        if not np.isfinite(np.asarray(arr, np.float64)).all():
            nbad = int((~np.isfinite(np.asarray(arr, np.float64))).sum())
            probs.append(f"{k} has {nbad} non-finite")
    return probs


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    man = C.load_manifest()
    checked = 0
    bad = []          # (clip_id, kind, stratum, [problems])
    per_kind = {}     # kind -> count checked
    per_stratum = {}  # stratum -> count checked
    for _, r in man.iterrows():
        stratum = r["stratum"]
        kinds = list(FULL_KINDS[:1])  # 'main'
        if C.is_landmark_source(stratum, r["group"]):
            kinds = list(FULL_KINDS) + [LAT_KIND]
        for kind in kinds:
            p = npz_path(r["clip_id"], kind)
            probs = check_one(p, kind, stratum)
            checked += 1
            per_kind[kind] = per_kind.get(kind, 0) + 1
            per_stratum[stratum] = per_stratum.get(stratum, 0) + 1
            if probs:
                bad.append((r["clip_id"], kind, stratum, probs))

    # also flag any npz on disk NOT referenced by the manifest enumeration
    on_disk = {p.name for p in C.FEATURES.glob("*.npz")}
    expected_names = set()
    for _, r in man.iterrows():
        expected_names.add(f"{r['clip_id']}.npz")
        if C.is_landmark_source(r["stratum"], r["group"]):
            for k in ("LERP", "CUT50", "FREEZE", "LATLERP"):
                expected_names.add(f"{r['clip_id']}__{k}.npz")
    orphan = sorted(on_disk - expected_names)
    missing_files = sorted(expected_names - on_disk)

    lines = []
    lines.append("# INTEGRITY PASS - null-default Phase B npz (C4 step 0)")
    lines.append(f"features dir: {C.FEATURES}")
    lines.append("")
    lines.append(f"expected npz (manifest enumeration): {len(expected_names)}")
    lines.append(f"npz on disk:                         {len(on_disk)}")
    lines.append(f"checks run (main+landmark kinds):    {checked}")
    lines.append("")
    lines.append("checks per kind:    " + ", ".join(f"{k}={v}" for k, v in sorted(per_kind.items())))
    lines.append("checks per stratum: " + ", ".join(f"{k}={v}" for k, v in sorted(per_stratum.items())))
    lines.append("")
    lines.append(f"orphan npz on disk (not in manifest enumeration): {len(orphan)}")
    for o in orphan[:50]:
        lines.append(f"    ORPHAN {o}")
    lines.append("")
    lines.append(f"expected-but-absent files: {len(missing_files)}")
    for m in missing_files[:100]:
        lines.append(f"    ABSENT {m}")
    lines.append("")
    lines.append(f"BAD FILES: {len(bad)}")
    for cid, kind, stratum, probs in bad:
        lines.append(f"    BAD [{stratum}] {cid} ({kind}): " + "; ".join(probs))
    lines.append("")
    verdict = "CLEAN - all npz pass" if (not bad and not missing_files) else "PROBLEMS FOUND - see list above"
    lines.append(f"VERDICT: {verdict}")
    txt = "\n".join(lines) + "\n"
    (RESULTS / "integrity.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
