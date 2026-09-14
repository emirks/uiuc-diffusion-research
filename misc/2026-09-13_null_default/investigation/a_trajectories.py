"""Step A — the default as a TRAJECTORY. Anchor-relative, time-normalised curves per clip per space.

Outputs (investigation/):
  cache/curves_<SPACE>.npz        per-clip curves (tau, rho, dA, dB, nu on a K-grid; crossing; jumps)
  curves_mean_std.csv             mean/std per group x space x s (raw and crossing-aligned)
  curves_per_clip.csv             per-clip scalars (s_cross, jump1, jump3, never_crossed, group, space)
  clusters.csv                    k-means prototypes shares per group (PIX, DINO, VAE)
  fig_traj_<SPACE>.png, fig_clusters_<SPACE>.png
Serial decode, cv2 single-threaded, <=4 BLAS threads, no GPU.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np, pandas as pd
from pathlib import Path

LAB = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa"); REPO = LAB / "diffusion-research"
CAMP = REPO / "misc/2026-09-13_null_default"; INV = CAMP / "investigation"; CACHE = INV / "cache"
sys.path.insert(0, str(CAMP / "scripts")); import common
sys.path.insert(0, str(REPO / "misc/2026-09-08_collapse_remeasure")); from instrument import load_matrix
import cv2; cv2.setNumThreads(1)

K = 48
SG = np.linspace(0.0, 1.0, K)
FOCUS = {("S-GRID", "NULLGEN"), ("S-GRID", "GT"), ("S-GRID-F", "NULLGEN"),
         ("S-PROBE", "R1"), ("S-PROBE", "R2"), ("S-PROBE", "R3"),
         ("S-SWEEP", "A_empty"), ("S-SWEEP", "A_word")}


def glabel(row):
    g = f"{row.stratum}:{row.group}"
    if row.stratum == "S-PROBE":
        g += f":{row.tier}"
    return g


def curves_from_M(M, a_idx, b_idx):
    """M (T,d). Returns dict of K-grid curves + scalars. Anchors appended as s=0 / s=1 points."""
    a, b = M[a_idx].astype(np.float64), M[b_idx].astype(np.float64)
    u = b - a; gap = float(np.linalg.norm(u)) + 1e-8
    X = M[a_idx + 1:b_idx].astype(np.float64); n = X.shape[0]
    V = X - a; uhat = u / gap; t = V @ uhat; tau = t / gap
    rho = np.linalg.norm(V - np.outer(t, uhat), axis=1) / gap
    dA = np.linalg.norm(V, axis=1) / gap; dB = np.linalg.norm(X - b, axis=1) / gap
    nu = np.minimum(dA, dB)
    s = (np.arange(n) + 1) / (n + 1)
    s_full = np.concatenate([[0.0], s, [1.0]])
    def full(v, v0, v1):
        return np.concatenate([[v0], v, [v1]])
    tau_f, rho_f = full(tau, 0, 1), full(rho, 0, 0)
    dA_f, dB_f, nu_f = full(dA, 0, 1), full(dB, 1, 0), full(nu, 0, 0)
    # crossing (first interior index with tau >= 0.5) in s units
    ge = np.where(tau >= 0.5)[0]
    s_c = float(s[ge[0]]) if len(ge) else float("nan")
    # jumps in tau over 1 and 3 steps on the segment [a_idx..b_idx]
    tau_seg = tau_f
    d1 = np.abs(np.diff(tau_seg)); d3 = np.abs(tau_seg[3:] - tau_seg[:-3]) if len(tau_seg) > 3 else d1
    # also step-based: largest single step over the path (step_share) on the segment
    seg = M[a_idx:b_idx + 1].astype(np.float64); dd = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    path = dd.sum() + 1e-8
    out = dict(gap=gap, n=n, s_cross=s_c, jump1=float(d1.max()), jump3=float(d3.max()),
               step_share=float(dd.max() / path), path_over_gap=float(path / gap),
               DR=float(np.median(rho)), M=float(((tau >= 0.25) & (tau <= 0.75)).mean()),
               nu_max=float(nu.max()))
    grid = {}
    for name, v in dict(tau=tau_f, rho=rho_f, dA=dA_f, dB=dB_f, nu=nu_f).items():
        grid[name] = np.interp(SG, s_full, v)
    # crossing-aligned: warp so that s_c -> 0.5
    if np.isfinite(s_c) and 0 < s_c < 1:
        src = np.where(SG <= 0.5, s_c * 2 * SG, s_c + (SG - 0.5) * 2 * (1 - s_c))
        for name, v in dict(tau=tau_f, rho=rho_f, dA=dA_f, dB=dB_f, nu=nu_f).items():
            grid[name + "_al"] = np.interp(src, s_full, v)
    else:
        for name in ("tau", "rho", "dA", "dB", "nu"):
            grid[name + "_al"] = np.full(K, np.nan)
    return out, grid


def analytic_landmark(kind, n):
    """1-D landmark trajectories (a=0, b=1) with n interior points -> (T=n+2, 1) matrix."""
    alpha = (np.arange(n) + 1) / (n + 1)
    if kind == "LERP":
        v = alpha
    elif kind == "CUT50":
        v = (np.arange(n) >= n / 2.0).astype(float)
    elif kind == "FREEZE":
        v = np.zeros(n)
    else:
        raise ValueError(kind)
    return np.concatenate([[0.0], v, [1.0]])[:, None]


def main():
    t0 = time.time()
    df = common.load_manifest()
    sel = df[[(r.stratum, r.group) in FOCUS for r in df.itertuples()]].copy()
    sel["glabel"] = [glabel(r) for r in sel.itertuples()]
    print(f"[A] {len(sel)} main clips selected", flush=True)
    # landmark owners: the clip whose anchors define landmarks for this row
    pcm = pd.read_csv(CAMP / "results/per_clip.csv", keep_default_na=False, usecols=["clip_id", "kind", "space", "landmark_owner"])
    pcm = pcm[(pcm.kind == "main") & (pcm.space == "VAE")].drop_duplicates("clip_id").set_index("clip_id")["landmark_owner"]
    owners = {}
    for r in sel.itertuples():
        o = pcm.get(r.clip_id, "")
        owners[r.clip_id] = o if o else r.clip_id
    miss = [c for c, o in owners.items() if not (F_ := common.FEATURES / f"{o}__LERP.npz").exists()]
    print(f"[A] owners resolved; {len(miss)} missing landmark files", flush=True)
    F = common.FEATURES
    recs = []      # per-clip scalar rows
    grids = {sp: {} for sp in ("PIX", "DINO", "VAE")}   # space -> key -> dict of curves

    def add(space, key, glab, kind, out, grid, owner=None):
        grids[space][key] = grid
        recs.append(dict(key=key, space=space, glabel=glab, kind=kind, owner=owner or "", **out))

    done_landmarks = set()
    for i, r in enumerate(sel.itertuples()):
        own = owners[r.clip_id]
        # ---- PIX (decode serially)
        M = load_matrix(REPO / r.path)
        if M is None or M.shape[0] < r.T:
            print(f"[A] decode problem {r.clip_id} T={None if M is None else M.shape[0]}", flush=True)
        else:
            out, grid = curves_from_M(M, r.a_idx, r.b_idx)
            add("PIX", r.clip_id, r.glabel, "main", out, grid)
        # ---- DINO
        z = np.load(F / f"{r.clip_id}.npz")
        out, grid = curves_from_M(z["dino"].astype(np.float32), r.a_idx, r.b_idx)
        add("DINO", r.clip_id, r.glabel, "main", out, grid)
        # ---- VAE
        vae = z["vae"].astype(np.float32); vae = vae.reshape(vae.shape[0], -1)
        la, lb = common.lat_index(r.a_idx), common.lat_index(r.b_idx)
        out, grid = curves_from_M(vae, la, lb)
        add("VAE", r.clip_id, r.glabel, "main", out, grid)
        # ---- landmarks (once per owner), labelled by the stratum they serve
        lm_label = f"{r.stratum}:LM" + (f":{r.tier}" if r.stratum == "S-PROBE" else "")
        if (own, lm_label) not in done_landmarks:
            done_landmarks.add((own, lm_label))
            n_int = r.b_idx - r.a_idx - 1
            for kind in ("LERP", "CUT50", "FREEZE"):
                Ml = analytic_landmark(kind, n_int)
                out, grid = curves_from_M(Ml, 0, Ml.shape[0] - 1)
                add("PIX", f"{own}__{kind}", lm_label, kind, out, grid, owner=own)
                zl = np.load(F / f"{own}__{kind}.npz")
                out, grid = curves_from_M(zl["dino"].astype(np.float32), r.a_idx, r.b_idx)
                add("DINO", f"{own}__{kind}", lm_label, kind, out, grid, owner=own)
                vl = zl["vae"].astype(np.float32); vl = vl.reshape(vl.shape[0], -1)
                out, grid = curves_from_M(vl, la, lb)
                add("VAE", f"{own}__{kind}", lm_label, kind, out, grid, owner=own)
            zl = np.load(F / f"{own}__LATLERP.npz")
            vl = zl["vae"].astype(np.float32); vl = vl.reshape(vl.shape[0], -1)
            out, grid = curves_from_M(vl, la, lb)
            add("VAE", f"{own}__LATLERP", lm_label, "LATLERP", out, grid, owner=own)
        if (i + 1) % 25 == 0:
            print(f"[A] {i+1}/{len(sel)} clips  {time.time()-t0:.0f}s", flush=True)
    rec = pd.DataFrame(recs)
    rec.to_csv(INV / "curves_per_clip.csv", index=False)
    for sp in grids:
        keys = list(grids[sp].keys())
        arr = {name: np.stack([grids[sp][k][name] for k in keys]) for name in grids[sp][keys[0]].keys()}
        np.savez_compressed(CACHE / f"curves_{sp}.npz", keys=np.array(keys), **arr)
    print(f"[A] done in {time.time()-t0:.0f}s; {len(rec)} curve rows", flush=True)


if __name__ == "__main__":
    main()
