"""Shared helpers for the null-default campaign (manifest + PIX + feature extraction).

One place for: repo/cache paths, manifest loading, the latent-frame index rule (SPEC 2.3),
which manifest rows own their own landmarks (SPEC 4), and construction of the pixel-space
landmark clips (LERP / CUT50 / FREEZE) from a clip's own native frames.

Decoding convention for the campaign: cv2 (BGR->RGB), single-threaded, exactly like the
Sep-08 instrument (load_matrix), armA (pipeline.load_frames) and the lerp campaign. All four
feature spaces read the same decoded frames so within-space distances are self-consistent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

LAB = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa")
REPO = LAB / "diffusion-research"
CAMP = REPO / "misc" / "2026-09-13_null_default"
REMEASURE = REPO / "misc" / "2026-09-08_collapse_remeasure"
FEATURES = Path(os.environ.get("NULL_FEATURES_DIR", str(LAB / "cache" / "null_default" / "features")))

MANIFEST_CSV = CAMP / "manifest.csv"

# landmark kinds
PIX_LANDMARKS = ("LERP", "CUT50", "FREEZE")   # built in pixel space
VAE_ONLY_LANDMARK = "LATLERP"                  # built in latent space


# --------------------------------------------------------------------------- latent index rule
def lat_index(f: int) -> int:
    """SPEC 2.3 pixel frame -> latent timestep. l(0)=0 else (f-1)//8 + 1."""
    return 0 if f == 0 else (f - 1) // 8 + 1


def n_latent(T: int) -> int:
    assert (T - 1) % 8 == 0, f"T={T} is not 8k+1"
    return (T - 1) // 8 + 1


# --------------------------------------------------------------------------- landmark ownership
def is_landmark_source(stratum: str, group: str) -> bool:
    """Rows whose OWN anchors define a landmark set (SPEC 1/4).

    S-GRID NULL clips use the GT twin's landmarks; R1/R2 use R3's anchors; so those groups
    do NOT own landmarks here. Everything else builds landmarks from its own a_idx/b_idx.
    """
    if group == "GT":
        return True
    if group == "R3":
        return True
    if stratum in ("S-GRID-F", "S-GRID-START", "S-SWEEP"):
        return True
    return False


# --------------------------------------------------------------------------- decode
def decode_rgb(path, downscale_wh=None):
    """Decode an mp4 to (T,H,W,3) uint8 RGB with cv2, single-threaded.

    downscale_wh = (W,H) resizes every frame with INTER_AREA before returning (S-SWEEP).
    """
    import cv2, time
    cv2.setNumThreads(1)
    # retry: /taiga (Lustre) reads occasionally stall past cv2's 30 s stream timeout, which
    # surfaces as "moov atom not found" / zero frames on a file that is intact (seen on job
    # 3146770 shard 4, 2026-09-14). Three attempts with a pause before giving up.
    last_err = None
    for attempt in range(3):
        cap = cv2.VideoCapture(str(path))
        out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if downscale_wh is not None:
                f = cv2.resize(f, downscale_wh, interpolation=cv2.INTER_AREA)
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if out:
            return np.stack(out)
        last_err = f"no frames decoded (attempt {attempt + 1}/3): {path}"
        print(f"[decode_rgb] {last_err}", flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(last_err)


def sweep_downscale_wh():
    """S-SWEEP native 1536x1024 (WxH) -> 768x512 (WxH); SPEC 512x768 (HxW)."""
    return (768, 512)


# --------------------------------------------------------------------------- pixel landmarks
def build_pixel_landmarks(frames_rgb, a_idx: int, b_idx: int):
    """Given a clip's native RGB frames (T,H,W,3) uint8, return {LERP,CUT50,FREEZE} clips.

    Each landmark clip is full length T: prefix [0..a_idx]=a, suffix [b_idx+1..T-1]=b, the
    interior I=(a_idx,b_idx) carries the landmark trajectory, frame a_idx=a and frame b_idx=b.
    Returned as float32 in [0,255] (the blend produces fractional values); the downstream
    128px/blur pipeline and the VAE/DINO/TRANS extractors accept float frames.
    """
    T = frames_rgb.shape[0]
    a = frames_rgb[a_idx].astype(np.float32)
    b = frames_rgb[b_idx].astype(np.float32)
    interior = list(range(a_idx + 1, b_idx))       # open interval
    n = len(interior)

    def blank():
        out = np.empty_like(frames_rgb, dtype=np.float32)
        for t in range(T):
            out[t] = a if t <= a_idx else b          # prefix=a incl a_idx, suffix=b incl b_idx
        return out

    lerp = blank()
    cut = blank()
    frz = blank()
    for k, t in enumerate(interior):
        alpha = (k + 1) / (n + 1)                    # 0<alpha<1 strictly inside I
        lerp[t] = (1.0 - alpha) * a + alpha * b
        cut[t] = a if (k < n / 2.0) else b           # first half a, second half b
        frz[t] = a                                   # freeze on a, jump to b at b_idx
    return {"LERP": lerp, "CUT50": cut, "FREEZE": frz}


def build_latlerp_latents(vae_main, a_idx: int, b_idx: int):
    """VAE landmark: linear interpolation of the two anchor LATENT frames (SPEC 4).

    vae_main: (T_lat, C, H, W) latents of the clip. Returns (T_lat, C, H, W) with prefix=lat_a,
    suffix=lat_b, and the interior latent indices linearly blended between them.
    """
    la, lb = lat_index(a_idx), lat_index(b_idx)
    out = np.array(vae_main, copy=True)
    A = vae_main[la]
    B = vae_main[lb]
    span = max(lb - la, 1)
    for l in range(vae_main.shape[0]):
        if l <= la:
            out[l] = A
        elif l >= lb:
            out[l] = B
        else:
            w = (l - la) / span
            out[l] = (1.0 - w) * A + w * B
    return out


# --------------------------------------------------------------------------- manifest
def load_manifest():
    """Canonical manifest reader.

    IMPORTANT: the group value "NULL" is in pandas' default na_values, so a PLAIN
    pd.read_csv(manifest.csv) silently turns every NULL group into NaN. We read every column as
    a string with keep_default_na=False so "NULL" survives, then coerce the numeric/bool columns.
    Any downstream reader (Phase C) must do the same, or read via this function.
    """
    import pandas as pd
    df = pd.read_csv(MANIFEST_CSV, keep_default_na=False, dtype=str)
    for c in ("a_idx", "b_idx", "T"):
        df[c] = df[c].astype(int)
    df["two_sided"] = df["two_sided"].str.strip().str.lower().isin(("true", "1"))
    return df


# --------------------------------------------------------------------------- trajectory measure
def measure_traj(M, a_idx, b_idx, eps=1e-8):
    """SPEC 3 common descriptor for a per-timestep feature sequence M (T, d), anchors a_idx/b_idx.

    Byte-identical to pix_features.measure_pix so PIX/DINO/VAE use one definition. For DINO the
    timesteps are video frames; for VAE they are latent timesteps and a_idx/b_idx must be the
    LATENT indices lat_index(a)/lat_index(b).
    """
    T = M.shape[0]
    a, b = M[a_idx], M[b_idx]
    u = b - a
    gap = float(np.linalg.norm(u))
    gap_rel = float(gap / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + eps))
    out = dict(T=T, a_idx=int(a_idx), b_idx=int(b_idx), gap=gap, gap_rel=gap_rel,
               static=bool(gap_rel < 0.12))
    X = M[a_idx + 1:b_idx]
    n = X.shape[0]
    out["n_interior"] = int(n)
    seg = M[a_idx:b_idx + 1]
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
            out[k] = float("nan")
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
        out[f"tau_{i}"] = float(s.mean()) if len(s) else float("nan")
    da = np.linalg.norm(X - a, axis=1)
    db = np.linalg.norm(X - b, axis=1)
    nu = np.minimum(da, db) / gap
    out["nu_max"] = float(nu.max())
    out["nu_mean"] = float(nu.mean())
    out["explained"] = float((nu < 0.15).mean())
    return out


# --------------------------------------------------------------------------- TRANS summaries (SPEC 2.4)
# 0-based channel indices (SPEC 2.4 uses 1-based; documented in PHASE_A_REPORT 8.4)
CH = dict(u=32, v=33, conf=34, ent=35, inplace=36, dir=37, sA=38, sB=39, nu=40, dlab=41)


def trans_timecourses(field):
    """field (T_lat,H,W,44) -> dict of per-latent-timestep global summaries (SPEC 2.4).

    local_t needs a q90 threshold on channel 37 from the GT-twin pool; it is filled in by the
    caller (kept NaN here). Returns arrays of length T_lat.
    """
    f = field.astype(np.float32)
    cellmean = lambda ch: f[..., ch].reshape(f.shape[0], -1).mean(1)
    nu_t = cellmean(CH["nu"])
    sA_t = cellmean(CH["sA"])
    sB_t = cellmean(CH["sB"])
    swap_t = sB_t - sA_t
    speed = np.sqrt(f[..., CH["u"]] ** 2 + f[..., CH["v"]] ** 2)
    trans_t = (f[..., CH["conf"]] * speed).reshape(f.shape[0], -1).mean(1)
    ent_t = cellmean(CH["ent"])
    inplace_t = cellmean(CH["inplace"])
    dir_t = cellmean(CH["dir"])
    dlab_t = cellmean(CH["dlab"])
    return dict(nu_t=nu_t, sA_t=sA_t, sB_t=sB_t, swap_t=swap_t, trans_t=trans_t, ent_t=ent_t,
                inplace_t=inplace_t, dir_t=dir_t, dlab_t=dlab_t)


def trans_local_t(field, q90):
    """local_t = fraction of cells with channel 37 (dir / one_minus_cos) > q90, per timestep."""
    dirf = field[..., CH["dir"]].astype(np.float32).reshape(field.shape[0], -1)
    return (dirf > q90).mean(1)


def trans_descriptors(tc, local_t, a_lat, b_lat):
    """SPEC 2.4 descriptors from the timecourses, over the latent window a_lat..b_lat inclusive."""
    w = slice(a_lat, b_lat + 1)
    nu = tc["nu_t"][w]
    swap = tc["swap_t"]
    dswap = np.abs(np.diff(swap[w]))                     # adjacent |Δswap| inside the window
    denom = abs(float(swap[b_lat] - swap[a_lat])) + 1e-8
    inplace = tc["inplace_t"][w]
    ip_sum = float(inplace.sum() + 1e-8)
    ip_argmax_global = a_lat + int(np.argmax(inplace))
    T_lat = len(tc["nu_t"])
    out = dict(
        nu_max=float(nu.max()), nu_mean=float(nu.mean()),
        swap_sharp=(float(dswap.max()) / denom) if len(dswap) else float("nan"),
        swap_pos=(float((a_lat + 1 + int(np.argmax(dswap))) / T_lat)) if len(dswap) else float("nan"),
        trans_mean=float(tc["trans_t"][w].mean()), trans_peak=float(tc["trans_t"][w].max()),
        inplace_peak_share=float(inplace.max() / ip_sum),
        local_at_peak=float(local_t[ip_argmax_global]),
        ent_mean=float(tc["ent_t"][w].mean()),
    )
    return out
