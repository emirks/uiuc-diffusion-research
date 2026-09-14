"""C1 - assemble results/per_clip.csv (one row per clip_id x kind x space) + trans_profiles.csv.

Spaces: PIX (reused verbatim from results/pix_per_clip.csv), DINO (npz `dino`), VAE (npz `vae`,
flattened per latent timestep, window lat_index(a)..lat_index(b)), TRANS (npz `trans` reduced to
the SPEC 2.4 global summaries over the same latent window, 0-based channels).

Resumable / partial-safe: any clip whose npz is missing is skipped (counted). Run against a
partial feature tree during Phase B; finalise when all 2513 npz exist.

    python misc/2026-09-13_null_default/scripts/descriptors.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

RESULTS = C.CAMP / "results"
PIX_CSV = RESULTS / "pix_per_clip.csv"

POINT_SPACES = ("PIX", "DINO", "VAE")
# the 18-d embedding fields (SPEC 3) live inside the point descriptor set
DESC_POINT = (["DR", "DR_mean", "M", "cross", "nu_max", "nu_mean", "explained", "step_share",
               "step_pos", "path_over_gap", "gap", "gap_rel", "static", "tau_med", "n_interior"]
              + [f"tau_{i}" for i in range(10)] + [f"speed_{i}" for i in range(10)])
DESC_TRANS = ["nu_max", "nu_mean", "swap_sharp", "swap_pos", "trans_mean", "trans_peak",
              "inplace_peak_share", "local_at_peak", "ent_mean"]
META = ["clip_id", "kind", "space", "stratum", "group", "tier", "endpoint", "seed", "grid",
        "prompt_id", "twin_clip_id", "landmark_owner", "a_idx", "b_idx", "T"]


def prompt_id_of(row):
    m = re.search(r"prompt_id=([A-Za-z0-9]+)", str(row.get("notes", "")))
    return m.group(1) if m else ""


def resolve_owner(row, r3_map):
    """The clip whose landmark set this clip is compared against (SPEC 1 / coordinator)."""
    if C.is_landmark_source(row["stratum"], row["group"]):
        return row["clip_id"]
    if row["group"] == "NULLGEN":                      # S-GRID null -> its GT twin
        return row["twin_clip_id"]
    if row["group"] in ("R1", "R2"):                   # -> the R3 of the same prompt+seed
        return r3_map.get((prompt_id_of(row), str(row["seed"])), "")
    return ""


def kinds_for(row):
    yield "main"
    if C.is_landmark_source(row["stratum"], row["group"]):
        yield from ("LERP", "CUT50", "FREEZE", "LATLERP")


def npz_path(clip_id, kind):
    return C.FEATURES / (f"{clip_id}.npz" if kind == "main" else f"{clip_id}__{kind}.npz")


# --------------------------------------------------------------------------- q90 for local_t
def compute_q90(man):
    """90th percentile of channel 37 (dir/one_minus_cos) over all GT-twin cells (S-GRID), window."""
    vals = []
    n = 0
    for _, r in man[man.group == "GT"].iterrows():
        p = npz_path(r["clip_id"], "main")
        if not p.exists():
            continue
        z = np.load(p)
        if "trans" not in z.files:
            continue
        a_lat, b_lat = C.lat_index(int(z["a_idx"])), C.lat_index(int(z["b_idx"]))
        dirf = z["trans"][a_lat:b_lat + 1, :, :, C.CH["dir"]].astype(np.float32).ravel()
        vals.append(dirf)
        n += 1
    if not vals:
        return float("nan"), 0
    return float(np.percentile(np.concatenate(vals), 90)), n


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-existing", action="store_true", default=True)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    man = C.load_manifest()
    pix = pd.read_csv(PIX_CSV, keep_default_na=False)
    pix_by = {(r["clip_id"], r["kind"]): r for _, r in pix.iterrows()}
    r3_map = {(prompt_id_of(r), str(r["seed"])): r["clip_id"]
              for _, r in man[man.group == "R3"].iterrows()}

    q90, n_gt = compute_q90(man)
    (RESULTS / "trans_q90.json").write_text(json.dumps({"q90_dir_ch37": q90, "n_gt_clips": n_gt}))
    print(f"[desc] local_t q90 (ch37 over {n_gt} GT-twin clips) = {q90}", flush=True)

    rows, profiles, skipped, errors = [], [], 0, []
    for _, r in man.iterrows():
        owner = resolve_owner(r, r3_map)
        meta = dict(clip_id=r["clip_id"], stratum=r["stratum"], group=r["group"], tier=r["tier"],
                    endpoint=r["endpoint"], seed=r["seed"], grid=r["grid"],
                    prompt_id=prompt_id_of(r), twin_clip_id=r["twin_clip_id"], landmark_owner=owner)
        for kind in kinds_for(r):
            p = npz_path(r["clip_id"], kind)
            if not p.exists():
                skipped += 1
                continue
            try:
                z = np.load(p)
                a_idx, b_idx, T = int(z["a_idx"]), int(z["b_idx"]), int(z["T"])
                a_lat, b_lat = C.lat_index(a_idx), C.lat_index(b_idx)
                base = dict(meta, kind=kind, a_idx=a_idx, b_idx=b_idx, T=T)

                # PIX (verbatim from pix_per_clip) - not defined for LATLERP
                if kind != "LATLERP" and (r["clip_id"], kind) in pix_by:
                    pr = pix_by[(r["clip_id"], kind)]
                    row = dict(base, space="PIX")
                    for k in DESC_POINT:
                        row[k] = pr.get(k, "")
                    rows.append(row)

                # DINO
                if kind != "LATLERP" and "dino" in z.files:
                    d = z["dino"].astype(np.float32)
                    rows.append(dict(base, space="DINO", **C.measure_traj(d, a_idx, b_idx)))

                # VAE (flatten per latent timestep, latent window)
                if "vae" in z.files:
                    v = z["vae"].astype(np.float32).reshape(z["vae"].shape[0], -1)
                    rows.append(dict(base, space="VAE", **C.measure_traj(v, a_lat, b_lat)))

                # TRANS (SPEC 2.4 summaries) + profiles
                if kind != "LATLERP" and "trans" in z.files:
                    tc = C.trans_timecourses(z["trans"])
                    local_t = C.trans_local_t(z["trans"], q90)
                    rows.append(dict(base, space="TRANS",
                                     **C.trans_descriptors(tc, local_t, a_lat, b_lat)))
                    T_lat = len(tc["nu_t"])
                    for l in range(T_lat):
                        profiles.append(dict(clip_id=r["clip_id"], kind=kind, stratum=r["stratum"],
                                             group=r["group"], latent_t=l,
                                             in_window=int(a_lat <= l <= b_lat),
                                             nu_t=float(tc["nu_t"][l]), sA_t=float(tc["sA_t"][l]),
                                             sB_t=float(tc["sB_t"][l]), swap_t=float(tc["swap_t"][l]),
                                             trans_t=float(tc["trans_t"][l]), ent_t=float(tc["ent_t"][l]),
                                             inplace_t=float(tc["inplace_t"][l]), dir_t=float(tc["dir_t"][l]),
                                             dlab_t=float(tc["dlab_t"][l]), local_t=float(local_t[l])))
            except Exception as e:
                errors.append((r["clip_id"], kind, f"{type(e).__name__}: {e}"))

    seen = set(META)
    allcols = list(META)
    for c in DESC_POINT + DESC_TRANS:
        if c not in seen:
            allcols.append(c)
            seen.add(c)
    df = pd.DataFrame(rows)
    for c in allcols:
        if c not in df:
            df[c] = np.nan
    df = df[allcols]
    df.to_csv(RESULTS / "per_clip.csv", index=False)
    pd.DataFrame(profiles).to_csv(RESULTS / "trans_profiles.csv", index=False)

    print(f"[desc] per_clip.csv rows={len(df)} (skipped {skipped} missing-npz kinds); "
          f"trans_profiles rows={len(profiles)}; errors={len(errors)}", flush=True)
    if errors:
        for e in errors[:20]:
            print("   ERR", e, flush=True)
    print("[desc] rows per space:", df.space.value_counts().to_dict(), flush=True)
    print("[desc] rows per (space,kind):")
    print(df.groupby(["space", "kind"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
