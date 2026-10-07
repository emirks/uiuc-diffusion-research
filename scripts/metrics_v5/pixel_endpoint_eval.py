#!/usr/bin/env python
"""Pixel fidelity of the GIVEN endpoint frames (metrics v5, eval 050).

Per generation, PSNR / SSIM / LPIPS between the output's endpoint frames and the given endpoint frames.
Two families of pairing (frozen protocol, brief R6 2026-09-24):

  MAIN (frame-level, comparable across EVERY system): the single given START frame and, on two-sided rows,
    the single given END frame. out frame 0 vs start9 frame 0; out frame T-1 vs end9 frame 8 (target frame 120).
    start9/end9 are eval_ladder/conds/<endpoint>_{start9,end9}.mp4 (480x640, 24 fps, 9 frames) for EVERY grid
    type, VACE16 included -- this is the cross-system comparison.
  SECONDARY (window mean, the clip-conditioned systems; a stricter self-check, NOT the comparison): the mean
    over the whole given window with the eval-043 pairing. (n_pre, n_suf) = store_eval_common.windows(gtype, sided):
    HF 9/8|0, VACE16 6/4|0, ED + externals 1/1|0 (window IS the frame there). out[t] vs start clip[t] for t<n_pre;
    out[T-n_suf+j] vs end clip[len-n_suf+j] for j<n_suf. Start/end clips = start9/end9 mp4 everywhere except
    VACE16, whose given frames are the lossless PNGs misc/2026-09-20_teg_baselines/conds_16fps/<endpoint>_start6_NN.png
    / _end4_NN.png (what the pipeline consumed).

Metrics (uint8 [0,255], full native resolution, PyAV rgb24, no resize; if a pair's shapes differ the GIVEN frame
is bilinear-resized to the output's size and resized:true is recorded -- expected never, all frames 640x480):
  PSNR (dB): 10*log10(255^2/MSE) over all pixels+3 channels; MSE==0 -> 100.0.
  SSIM: Wang et al. 2004, 11x11 Gaussian window sigma 1.5, K1 0.01, K2 0.03, L 255, 'valid' padding, per RGB channel
    averaged over 3 channels (torch conv2d groups=3, float64; verified against a numpy sliding-window loop < 1e-6).
  LPIPS: lpips.LPIPS(net='alex', version='0.1'), input (x/127.5 - 1) float32, full resolution, no_grad.

Caveat: PSNR is capped by the H.264 encoding of both the output and the given clips (~40-45 dB ceiling; a perfect
copy through a VAE lands lower). The frame-level numbers are the cross-system comparison; the window means are defined
only for clip-conditioned systems and are NOT comparable across grid types.

Parallel/resumable: roster-driven; one rows.jsonl per harness arm, written atomically only when the arm is COMPLETE
(rows == videos). --shard i/n slices the sorted harness arms; a store/evals/<id>/_locks/<arm>.lock (host+pid, O_EXCL)
claims an arm so shards + the login CPU race never double-score one (a lock older than STALE with no complete rows is
stolen). --device cuda|cpu selects the LPIPS device (decode/PSNR/SSIM stay on the CPU Pool of --workers); every row
records `device`. --wait loops (sleep between passes) until every arm is complete (the login guarantor). --finalize
writes meta.yaml (coverage + producing device per arm). --self-check runs the three asserts on CPU.
"""
from __future__ import annotations
import argparse, json, math, os, socket, sys, time
import multiprocessing as mp
from datetime import date
from functools import lru_cache
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from store_eval_common import (parse_stem, harness_arm_of, grid_of, grid_type, windows,  # noqa: E402
                               cond_clips, git_sha, atomic_write)

ROSTER = REPO_ROOT / "scripts/metrics_v5/roster.json"
CONDS_DIR = REPO_ROOT / "eval_ladder" / "conds"
CONDS16_DIR = REPO_ROOT / "misc/2026-09-20_teg_baselines/conds_16fps"
STALE_SECS = 1200            # a lock older than this (> the 15-min job limit) with no complete rows is stolen

# ---- SSIM window (Wang et al. 2004): 11x11 Gaussian, sigma 1.5, normalized to sum 1 -----------------
def _gauss2d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    c = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(c ** 2) / (2.0 * sigma ** 2)); g /= g.sum()
    w = np.outer(g, g); return w / w.sum()


_GAUSS = _gauss2d(11, 1.5)
_C1 = (0.01 * 255.0) ** 2
_C2 = (0.03 * 255.0) ** 2


# ---- metrics ---------------------------------------------------------------------------------------
def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    return 100.0 if mse == 0.0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim_torch(x: np.ndarray, y: np.ndarray) -> float:
    """SSIM via torch conv2d (groups=3), float64; per-channel valid conv, mean over channels+positions."""
    import torch
    import torch.nn.functional as F
    w = torch.from_numpy(_GAUSS).reshape(1, 1, 11, 11).repeat(3, 1, 1, 1)   # (3,1,11,11) float64
    X = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64)).permute(2, 0, 1).unsqueeze(0)
    Y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float64)).permute(2, 0, 1).unsqueeze(0)
    conv = lambda t: F.conv2d(t, w, groups=3)                               # 'valid' (no padding)
    mux, muy = conv(X), conv(Y)
    mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
    sx = conv(X * X) - mux2
    sy = conv(Y * Y) - muy2
    sxy = conv(X * Y) - muxy
    smap = ((2 * muxy + _C1) * (2 * sxy + _C2)) / ((mux2 + muy2 + _C1) * (sx + sy + _C2))
    return float(smap.mean().item())


def ssim_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Independent numpy sliding-window reference for the self-check (same formula, no torch)."""
    from numpy.lib.stride_tricks import sliding_window_view
    x = x.astype(np.float64); y = y.astype(np.float64)
    vals = []
    for c in range(3):
        xc, yc = x[..., c], y[..., c]
        wconv = lambda a: np.einsum("ijkl,kl->ij", sliding_window_view(a, (11, 11)), _GAUSS)
        mux, muy = wconv(xc), wconv(yc)
        mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
        sx = wconv(xc * xc) - mux2
        sy = wconv(yc * yc) - muy2
        sxy = wconv(xc * yc) - muxy
        smap = ((2 * muxy + _C1) * (2 * sxy + _C2)) / ((mux2 + muy2 + _C1) * (sx + sy + _C2))
        vals.append(float(smap.mean()))
    return float(np.mean(vals))


# ---- decode (native resolution, PyAV rgb24) --------------------------------------------------------
def decode_video(path) -> np.ndarray:
    import av
    frames = []
    with av.open(str(path)) as container:
        vs = container.streams.video[0]
        vs.thread_type = "NONE"; vs.thread_count = 1      # login pids cap (reviewer 2026-09-24): no per-decoder thread pool
        for f in container.decode(vs):
            frames.append(f.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"no frames from {path}")
    return np.stack(frames)


@lru_cache(maxsize=96)
def _decode_mp4_cached(path: str) -> np.ndarray:
    return decode_video(path)


@lru_cache(maxsize=96)
def _decode_png_stack(dir_path: str, endpoint: str, kind: str, n: int) -> np.ndarray:
    """kind in {start6,end4}: stack of <endpoint>_<kind>_NN.png, NN = 00..n-1 (lossless, VACE window)."""
    from PIL import Image
    frs = []
    for i in range(n):
        p = Path(dir_path) / f"{endpoint}_{kind}_{i:02d}.png"
        if not p.exists():
            raise FileNotFoundError(str(p))
        frs.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
    return np.stack(frs)


def _match(out_f: np.ndarray, given_f: np.ndarray):
    """Return (given_f_matched, resized_bool): resize the GIVEN frame to the output's HxW if they differ."""
    if out_f.shape == given_f.shape:
        return given_f, False
    from PIL import Image
    h, w = out_f.shape[:2]
    r = np.asarray(Image.fromarray(given_f).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    return r, True


# ---- worker ----------------------------------------------------------------------------------------
_NET = None
_DEVICE = "cpu"


def _init(device: str):
    global _NET, _DEVICE
    import torch
    torch.set_num_threads(1)
    try:
        import cv2; cv2.setNumThreads(1)
    except Exception:
        pass
    import lpips
    _DEVICE = device
    _NET = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()


def _lpips(x: np.ndarray, y: np.ndarray) -> float:
    import torch
    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1.0).to(_DEVICE)
    with torch.no_grad():
        return float(_NET(t(x), t(y)).item())


def _trio(out_f: np.ndarray, given_f: np.ndarray):
    g, rz = _match(out_f, given_f)
    return psnr(out_f, g), ssim_torch(out_f, g), _lpips(out_f, g), rz


def _mean(vals):
    return float(np.mean(vals)) if vals else None


def work(task):
    vpath, endpoint, sided, gtype = task
    v = Path(vpath)
    item_id, seed = parse_stem(v.stem)
    n_pre, n_suf = windows(gtype, sided)
    two = sided == "two"
    r = dict(item_id=item_id, seed=seed, arm=harness_arm_of(item_id), endpoint=endpoint, sided=sided,
             gtype=gtype, n_pre=n_pre, n_suf=n_suf, T=None, device=_DEVICE, resized=False, missing=[],
             psnr_A=None, ssim_A=None, lpips_A=None, psnr_B=None, ssim_B=None, lpips_B=None,
             psnr_Aw=None, ssim_Aw=None, lpips_Aw=None, n_Aw=None,
             psnr_Bw=None, ssim_Bw=None, lpips_Bw=None, n_Bw=None)
    try:
        out = decode_video(v)
    except Exception as e:  # noqa: BLE001
        r["missing"].append(f"gen:{type(e).__name__}"); return r
    T = out.shape[0]; r["T"] = T
    # MAIN frame-level uses start9/end9 mp4 for EVERY grid type (the cross-system comparison)
    p_start9 = CONDS_DIR / f"{endpoint}_start9.mp4"
    p_end9 = CONDS_DIR / f"{endpoint}_end9.mp4"
    resized = False
    # SECONDARY window given clips
    if gtype == "VACE16":
        start_win = end_win = None
        try:
            start_win = _decode_png_stack(str(CONDS16_DIR), endpoint, "start6", n_pre)
        except Exception:  # noqa: BLE001
            r["missing"].append(f"condA_png:{endpoint}_start6")
        if two:
            try:
                end_win = _decode_png_stack(str(CONDS16_DIR), endpoint, "end4", n_suf)
            except Exception:  # noqa: BLE001
                r["missing"].append(f"condB_png:{endpoint}_end4")
    else:
        start_win = end_win = None
        if p_start9.exists():
            start_win = _decode_mp4_cached(str(p_start9))
        else:
            r["missing"].append(f"condA:{p_start9.name}")
        if two:
            if p_end9.exists():
                end_win = _decode_mp4_cached(str(p_end9))
            else:
                r["missing"].append(f"condB:{p_end9.name}")

    # ---- window A (SECONDARY); frame-level A reuses window pair 0 when the source is start9 (non-VACE)
    if start_win is not None:
        pa, sa, la = [], [], []
        for t in range(n_pre):
            ps, ss, ls, rz = _trio(out[t], start_win[t]); resized = resized or rz
            pa.append(ps); sa.append(ss); la.append(ls)
        r["psnr_Aw"], r["ssim_Aw"], r["lpips_Aw"], r["n_Aw"] = _mean(pa), _mean(sa), _mean(la), n_pre
        if gtype != "VACE16":
            r["psnr_A"], r["ssim_A"], r["lpips_A"] = pa[0], sa[0], la[0]
    # ---- window B (SECONDARY, two-sided only)
    if two and end_win is not None:
        L = end_win.shape[0]; pb, sb, lb = [], [], []
        for j in range(n_suf):
            og = out[T - n_suf + j]; gg = end_win[L - n_suf + j]
            ps, ss, ls, rz = _trio(og, gg); resized = resized or rz
            pb.append(ps); sb.append(ss); lb.append(ls)
        r["psnr_Bw"], r["ssim_Bw"], r["lpips_Bw"], r["n_Bw"] = _mean(pb), _mean(sb), _mean(lb), n_suf
        if gtype != "VACE16":
            r["psnr_B"], r["ssim_B"], r["lpips_B"] = pb[-1], sb[-1], lb[-1]
    # ---- MAIN frame-level (start9[0] / end9[8]) -- always start9/end9 mp4; for non-VACE reused above
    if r["psnr_A"] is None and p_start9.exists():
        s9 = _decode_mp4_cached(str(p_start9))
        ps, ss, ls, rz = _trio(out[0], s9[0]); resized = resized or rz
        r["psnr_A"], r["ssim_A"], r["lpips_A"] = ps, ss, ls
    if two and r["psnr_B"] is None and p_end9.exists():
        e9 = _decode_mp4_cached(str(p_end9))
        ps, ss, ls, rz = _trio(out[T - 1], e9[8]); resized = resized or rz
        r["psnr_B"], r["ssim_B"], r["lpips_B"] = ps, ss, ls
    r["resized"] = bool(resized)
    return r


# ---- roster -> {harness_arm: [tasks]} --------------------------------------------------------------
def arm_tasks(roster_path: Path) -> dict:
    roster = json.loads(roster_path.read_text())
    variants = []
    for a in roster["arms"]:
        for g in a["gens"]:
            if g not in variants:
                variants.append(g)
    out = {}
    for g in variants:
        vdir = REPO_ROOT / g
        vids = sorted((vdir / "videos").glob("*.mp4"))
        if not vids:
            continue
        grid = grid_of(vdir)
        gt = grid_type(vdir, harness_arm_of(parse_stem(vids[0].stem)[0]).split("_")[0])
        for v in vids:
            item_id, _ = parse_stem(v.stem); gg = grid.get(item_id)
            if not gg:
                continue
            ha = harness_arm_of(item_id)
            out.setdefault(ha, []).append((str(v), gg["endpoint"], gg.get("sided", "one"), gt))
    return out


# ---- locks -----------------------------------------------------------------------------------------
def _rows_complete(rows_path: Path, n_expected: int) -> bool:
    if not rows_path.exists():
        return False
    try:
        return sum(1 for ln in rows_path.read_text().splitlines() if ln.strip()) == n_expected
    except Exception:  # noqa: BLE001
        return False


def _acquire(lockdir: Path, arm: str) -> bool:
    """O_EXCL lock; steal a stale lock (age > STALE_SECS). Returns True if this process owns the arm now."""
    lockdir.mkdir(parents=True, exist_ok=True)
    lp = lockdir / f"{arm}.lock"
    payload = f"{socket.gethostname()} pid={os.getpid()} t={time.time():.0f}\n"
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, payload.encode()); os.close(fd); return True
    except FileExistsError:
        try:
            age = time.time() - lp.stat().st_mtime
        except FileNotFoundError:
            return _acquire(lockdir, arm)
        if age > STALE_SECS:
            try:
                os.remove(lp)
            except FileNotFoundError:
                pass
            return _acquire(lockdir, arm)
        return False


def _release(lockdir: Path, arm: str) -> None:
    try:
        os.remove(lockdir / f"{arm}.lock")
    except FileNotFoundError:
        pass


# ---- run -------------------------------------------------------------------------------------------
def run(args) -> int:
    ed = REPO_ROOT / "store" / "evals" / args.eval_id
    ed.mkdir(parents=True, exist_ok=True)
    lockdir = ed / "_locks"
    tasks_by_arm = arm_tasks(Path(args.roster))
    arms_sorted = sorted(tasks_by_arm)
    i, n = (int(x) for x in args.shard.split("/"))
    my_arms = arms_sorted[i::n] if n > 1 else arms_sorted
    device = args.device
    ctx = mp.get_context("spawn")      # reviewer 2026-09-24 17:45: the fork context deadlocked both CPU pools (parent had torch/av threads) -> spawn everywhere
    print(f"[run] host={socket.gethostname()} shard={i}/{n} device={device} workers={args.workers} "
          f"arms(all)={len(arms_sorted)} arms(mine)={len(my_arms)} eval={args.eval_id}", flush=True)
    pool = ctx.Pool(args.workers, initializer=_init, initargs=(device,))
    try:
        while True:
            did_any = False
            for arm in my_arms:
                tasks = tasks_by_arm[arm]
                rows_path = ed / arm / "rows.jsonl"
                if _rows_complete(rows_path, len(tasks)):
                    continue
                if not _acquire(lockdir, arm):
                    continue
                t0 = time.time()
                try:
                    rows = list(pool.imap(work, tasks, chunksize=8))
                    (ed / arm).mkdir(parents=True, exist_ok=True)
                    atomic_write(rows_path, "".join(json.dumps(r) + "\n" for r in rows))
                    did_any = True
                    nA = sum(1 for r in rows if isinstance(r.get("psnr_A"), float))
                    nB = sum(1 for r in rows if isinstance(r.get("psnr_B"), float))
                    miss = sum(1 for r in rows if r["missing"])
                    mA = _mean([r["psnr_A"] for r in rows if isinstance(r.get("psnr_A"), float)])
                    print(f"[arm] {arm:<40} n={len(rows)} A={nA} B={nB} missing={miss} "
                          f"psnrA={mA:.3f}" if mA is not None else f"[arm] {arm:<40} n={len(rows)} A={nA} B={nB} missing={miss} psnrA=NA",
                          f"{time.time()-t0:.0f}s dev={device}", flush=True)
                finally:
                    _release(lockdir, arm)
            if not args.wait:
                break
            remaining = [a for a in arms_sorted if not _rows_complete(ed / a / "rows.jsonl", len(tasks_by_arm[a]))]
            if not remaining:
                print("[run] all arms complete", flush=True); break
            if not did_any:
                print(f"[run] waiting on {len(remaining)} arms (locked/other shards): {remaining[:6]}...", flush=True)
                time.sleep(60)
    finally:
        pool.close(); pool.join()
    return 0


# ---- finalize (meta.yaml) --------------------------------------------------------------------------
DEFINITION = [ln.strip() for ln in __doc__.splitlines() if ln.strip()][3:22]
WHY = ("Prior works take an image input; this reports the pixel fidelity (PSNR/SSIM/LPIPS) of the given endpoint "
       "frames, comparably across every system (frame-level, start9[0] / end9[8]) plus a within-system window mean "
       "for the clip-conditioned arms. Complements the DINO identity of eval 043 (a perceptual/pixel counterpart).")
CAVEAT = ("PSNR is capped by the H.264 encoding of BOTH the output and the given clips (~40-45 dB ceiling; a perfect "
          "copy through a VAE lands lower -- see the re-encoding reference in the R6 RECORD). The frame-level numbers "
          "(psnr_A/ssim_A/lpips_A, and _B on two-sided rows) are the cross-system comparison; the window means "
          "(psnr_Aw.. / psnr_Bw..) are defined only for clip-conditioned systems and are NOT comparable across grid "
          "types (HF 9/8, VACE16 6/4 from the lossless PNGs, ED/externals 1/1 = the frame). GPU-vs-CPU LPIPS differ "
          "at ~1e-6 (device recorded per row).")


def finalize(args) -> int:
    ed = REPO_ROOT / "store" / "evals" / args.eval_id
    tasks_by_arm = arm_tasks(Path(args.roster))
    results = []
    incomplete = []
    for arm in sorted(tasks_by_arm):
        rp = ed / arm / "rows.jsonl"
        n_exp = len(tasks_by_arm[arm])
        if not _rows_complete(rp, n_exp):
            incomplete.append((arm, (sum(1 for l in rp.read_text().splitlines() if l.strip()) if rp.exists() else 0), n_exp))
            continue
        rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
        fin = lambda k: [r[k] for r in rows if isinstance(r.get(k), float) and math.isfinite(r[k])]
        devs = sorted({r.get("device") for r in rows})
        gen = None
        cov = dict(n=len(rows), A_defined=len(fin("psnr_A")), B_defined=len(fin("psnr_B")),
                   missing=sum(1 for r in rows if r["missing"]),
                   resized=sum(1 for r in rows if r.get("resized")),
                   device=(devs[0] if len(devs) == 1 else "+".join(devs)),
                   psnr_A_mean=(round(float(np.mean(fin("psnr_A"))), 4) if fin("psnr_A") else None),
                   ssim_A_mean=(round(float(np.mean(fin("ssim_A"))), 4) if fin("ssim_A") else None),
                   lpips_A_mean=(round(float(np.mean(fin("lpips_A"))), 4) if fin("lpips_A") else None),
                   psnr_B_mean=(round(float(np.mean(fin("psnr_B"))), 4) if fin("psnr_B") else None),
                   ssim_B_mean=(round(float(np.mean(fin("ssim_B"))), 4) if fin("ssim_B") else None),
                   lpips_B_mean=(round(float(np.mean(fin("lpips_B"))), 4) if fin("lpips_B") else None))
        results.append((arm, cov))
    if incomplete:
        print(f"[finalize] {len(incomplete)} arms INCOMPLETE, meta NOT written: {incomplete[:8]}", flush=True)
        return 2
    L = [f"id: {args.eval_id}", f"seq: {int(args.eval_id.split('_', 1)[0])}", "shelf: evals",
         f"created: '{args.date}'", "machine: dai (ghx4 array + login CPU race; LPIPS cuda/cpu, decode/PSNR/SSIM CPU)",
         f"instrument: scripts/metrics_v5/pixel_endpoint_eval.py @ {git_sha()}",
         "namespaces: []  # pixels only (PyAV rgb24), no feature-store namespace", "definition:"]
    L += [f"  - {json.dumps(d)}" for d in DEFINITION]
    L += [f"why: {json.dumps(WHY)}", f"caveat: {json.dumps(CAVEAT)}", "arms_scored:"]
    for arm, cov in results:
        L += [f"  {arm}:", f"    rows: {cov['n']}",
              "    coverage: {" + ", ".join(f"{k}: {json.dumps(v) if isinstance(v, str) else v}" for k, v in cov.items() if k != "n") + "}"]
    atomic_write(ed / "meta.yaml", "\n".join(L) + "\n")
    print(f"[finalize] wrote {ed/'meta.yaml'} ({len(results)} arms)", flush=True)
    return 0


# ---- self-checks -----------------------------------------------------------------------------------
def self_check(args) -> int:
    _init("cpu")
    print("[self-check] device=cpu", flush=True)
    tasks_by_arm = arm_tasks(Path(args.roster))
    # (1) SSIM torch vs numpy on 5 real pairs (max|diff| < 1e-6)
    hf = "store/gens/032_dualforce_dcg_w6/03_neutral_v3__dai"
    vids = sorted((REPO_ROOT / hf / "videos").glob("*.mp4"))[:5]
    grid = grid_of(REPO_ROOT / hf)
    dmax = 0.0
    for v in vids:
        out = decode_video(v)
        ep = grid[parse_stem(v.stem)[0]]["endpoint"]
        s9 = decode_video(CONDS_DIR / f"{ep}_start9.mp4")
        st, sn = ssim_torch(out[0], s9[0]), ssim_numpy(out[0], s9[0])
        dmax = max(dmax, abs(st - sn))
        print(f"  ssim pair {v.stem[:30]:30} torch={st:.10f} numpy={sn:.10f} |d|={abs(st-sn):.2e}", flush=True)
    print(f"[self-check 1] SSIM torch-vs-numpy max|diff| = {dmax:.3e} (bar < 1e-6): {'PASS' if dmax < 1e-6 else 'FAIL'}", flush=True)
    assert dmax < 1e-6, dmax
    # (2) identity: start9 frame 0 vs itself -> PSNR 100, SSIM 1.0, LPIPS 0.0 (3 clips)
    eps = [grid[parse_stem(v.stem)[0]]["endpoint"] for v in vids[:3]]
    for ep in eps:
        f0 = decode_video(CONDS_DIR / f"{ep}_start9.mp4")[0]
        p, s, l = psnr(f0, f0), ssim_torch(f0, f0), _lpips(f0, f0)
        print(f"  identity {ep:24} PSNR={p} SSIM={s:.10f} LPIPS={l}", flush=True)
        assert p == 100.0 and abs(s - 1.0) < 1e-12 and l == 0.0, (ep, p, s, l)
    print("[self-check 2] identity (PSNR 100 / SSIM 1.0 / LPIPS 0.0): PASS", flush=True)
    # (3) one own HF gen: the window mean restricted to n_pre=1 equals the frame-level value
    v = vids[0]; ep = grid[parse_stem(v.stem)[0]]["endpoint"]
    out = decode_video(v); s9 = decode_video(CONDS_DIR / f"{ep}_start9.mp4")
    pA, sA, lA, _ = _trio(out[0], s9[0])          # frame-level A
    pAw1, sAw1, lAw1, _ = _trio(out[0], s9[0])    # window mean over n_pre=1 (the first pair)
    print(f"  frameA=({pA:.6f},{sA:.6f},{lA:.6f}) window(n=1)=({pAw1:.6f},{sAw1:.6f},{lAw1:.6f})", flush=True)
    assert (pA, sA, lA) == (pAw1, sAw1, lAw1)
    print("[self-check 3] HF window(n_pre=1) == frame-level A: PASS", flush=True)
    print("[self-check] ALL THREE PASS", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-id", default=f"050_endpoint_pixel_gridv3__dai__{date.today().isoformat()}")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--roster", default=str(ROSTER))
    ap.add_argument("--shard", default="0/1", help="i/n: slice the sorted harness arms (round-robin)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--wait", action="store_true", help="loop until every arm is complete (the login guarantor)")
    ap.add_argument("--no-index", action="store_true", help="accepted for parity; this eval never writes INDEX")
    ap.add_argument("--finalize", action="store_true", help="write meta.yaml (all arms must be complete)")
    ap.add_argument("--self-check", action="store_true", help="run the three CPU self-checks and exit")
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check(args)
    if args.finalize:
        return finalize(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
