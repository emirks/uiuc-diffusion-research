#!/usr/bin/env python3
"""S6 (EffectData) 44-ch signal extraction — reuses armA_extract's FROZEN compute + PCA.

Adds the EffectData breadth stratum (S6) to the SAME `$LAB/cache/armA_signals` signal cache
used by S0/S1/S2a/S2b/S4/eval, produced by BYTE-IDENTICAL channel math and the SAME frozen
`pca.npz`. Nothing in `armA_extract.py` is modified; this driver only imports it.

Why S6 needs no geometry changes (unlike S4's 16-row crop):
  - EffectData clips are NATIVE VAE-legal (704/1056/1248 px, all /32) — encode used
    process_videos with native==bucket ⇒ identity resize + 0-px crop. So the DINO patch grid
    covers the exact same FOV as the latent grid; `armA_extract._fit_to_grid` is a no-op.
  - Clips are EXACTLY 81 frames ⇒ select_frames(81) -> T_lat=11, matching the (11,H,W) latent.
  - One-sided (frame-0 anchor): A=first frame, B=last frame, sided="one" (same as S4).

Clips are staged just-in-time from per-effect zips (data/raw/effectdata/Videos/<Effect>.zip,
member = ROSTER `video_path`) to a node-local dir (S6_CLIPS_DIR, set to /tmp by the sbatch).
Contiguous sharding over the frozen ROSTER order keeps a shard to few effects (few zip opens);
idempotent (skip if feat+raw already exist).

    python armA_extract_s6.py shapes
    python armA_extract_s6.py extract --shard i/n [--limit N]     # GPU
    python armA_extract_s6.py verify                              # CPU count/shape assert
"""
from __future__ import annotations
import sys, os, json, argparse, time, math, zipfile, glob
import numpy as np

ROOT = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
ARMA = f"{ROOT}/misc/2026-08-24_flow_signal_conditioning/armA"
sys.path.insert(0, ARMA)
import armA_extract as A                                    # frozen compute + PCA path + CH_NAMES

ENC = f"{ROOT}/outputs/ctt_v2/encodes/EFFECTDATA"
ROSTER_P = f"{ENC}/ROSTER.json"
ZIPS = f"{ROOT}/data/raw/effectdata/Videos"
CLIPS = os.environ.get("S6_CLIPS_DIR", f"{ENC}/clips")      # node-local /tmp in sbatch
FRAMES = 81


def _atomic_savez(path, **arrays):
    """Atomic compressed npz write. NOTE: np.savez_compressed APPENDS '.npz' if the name does
    not already end in it — so the tmp name must itself end in '.npz'. We use a dot-prefixed
    sibling ('.tmp.<name>.npz') so it (a) is not re-suffixed by numpy and (b) never matches a
    'S6__*.npz' glob if a write is interrupted. os.replace is atomic within a filesystem."""
    d, b = os.path.dirname(path), os.path.basename(path)
    tmp = os.path.join(d, ".tmp." + b)                      # b ends in .npz -> numpy writes exactly tmp
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def roster_clips():
    r = json.load(open(ROSTER_P))
    return r["clips"]                                       # frozen order


def shard_slice(clips, si, sn):
    """CONTIGUOUS block (roster is grouped by effect ⇒ few zips per shard)."""
    per = math.ceil(len(clips) / sn)
    return clips[si * per:(si + 1) * per]


def stage_effect(zf_effect, items):
    """Extract this effect's clips (flat <stem>.mp4) from one already-open zip."""
    for c in items:
        dst = f"{CLIPS}/{c['stem']}.mp4"
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        tmp = dst + ".tmp"
        with zf_effect.open(c["video_path"]) as s, open(tmp, "wb") as o:
            o.write(s.read())
        os.replace(tmp, dst)


def run_extract(shard, device, limit=0):
    z = np.load(A.PCA_PATH)
    pca_mean, pca_comp = z["mean"], z["comp"]
    clips = roster_clips()
    si, sn = (int(x) for x in shard.split("/"))
    mine = shard_slice(clips, si, sn)
    if limit:
        mine = mine[:limit]
    if not mine:
        print(f"[s6 {si}/{sn}] empty shard", flush=True); return
    # early skip: if this shard is already complete, don't even load DINO (cheap idempotent re-runs)
    todo0 = [c for c in mine if not (os.path.exists(f"{A.FEAT}/S6__{c['stem']}.npz")
                                     and os.path.exists(f"{A.RAW}/S6__{c['stem']}.npz"))]
    if not todo0:
        print(f"[s6 {si}/{sn}] all {len(mine)} present — skip (no DINO load)", flush=True); return
    os.makedirs(A.FEAT, exist_ok=True); os.makedirs(A.RAW, exist_ok=True)
    os.makedirs(CLIPS, exist_ok=True)
    dino = A.Dino(device)

    # group this shard's clips by effect (one zip open per effect)
    by_effect = {}
    for c in mine:
        by_effect.setdefault(c["effect"], []).append(c)

    t0 = time.time(); n = 0; new = 0; fails = []
    for effect, items in by_effect.items():
        # skip staging entirely if every clip of this effect is already done
        todo = [c for c in items
                if not (os.path.exists(f"{A.FEAT}/S6__{c['stem']}.npz")
                        and os.path.exists(f"{A.RAW}/S6__{c['stem']}.npz"))]
        if todo:
            zp = f"{ZIPS}/{effect}.zip"
            if not os.path.exists(zp):
                for c in todo:
                    fails.append((f"S6__{c['stem']}", "missing_zip")); print("FAIL missing_zip", zp, flush=True)
                n += len(items); continue
            with zipfile.ZipFile(zp) as zf:
                stage_effect(zf, todo)
        for c in items:
            cid = f"S6__{c['stem']}"
            fn = f"{A.FEAT}/{cid}.npz"; rawfn = f"{A.RAW}/{cid}.npz"
            if os.path.exists(fn) and os.path.exists(rawfn):
                n += 1; continue
            path = f"{CLIPS}/{c['stem']}.mp4"
            exp = [int(x) for x in c["latent_fhw"]]          # (T_lat, H_lat, W_lat)
            try:
                P_raw, centers, B, uniq, H, W, T_lat = A._load_clip(dino, path, shape=exp)
                if not os.path.exists(rawfn):
                    _atomic_savez(
                        rawfn, F_raw=P_raw.cpu().numpy().astype(np.float16),
                        frames=np.array(uniq), centers=np.array(centers), B=B,
                        grid=np.array([2 * H, 2 * W]), dim=768, id=cid, clip=c["stem"],
                        pop="S6", stratum="S6", resize="28/32 antialiased",
                        crop_top=A._last_crop[0], crop_left=A._last_crop[1],
                        note="EffectData S6: native VAE-legal (no crop), 81f one-sided; "
                             "DINOv2-base last_hidden_state patch tokens (CLS+reg dropped), raw un-normalized")
                field = A.compute_clip(P_raw, centers, B, uniq, pca_mean, pca_comp, H, W, device)
                assert tuple(int(x) for x in field.shape[:3]) == tuple(exp), \
                    f"{cid} field {field.shape[:3]} != latent {exp}"
                _atomic_savez(fn, F=field, id=cid, clip=c["stem"], pop="S6",
                              stratum="S6", cls=c["effect"], sided="one",
                              channels=np.array(A.CH_NAMES))
            except Exception as e:
                import traceback
                fails.append((cid, type(e).__name__)); print("FAIL", cid, type(e).__name__, e, flush=True)
                traceback.print_exc(); n += 1; continue
            n += 1; new += 1
            if new % 50 == 0:
                print(f"  {new} new / {n}/{len(mine)}  {time.time()-t0:.0f}s "
                      f"({(time.time()-t0)/max(new,1):.2f}s/clip)", flush=True)
    print(f"[s6 extract {si}/{sn}] {n}/{len(mine)} ({new} new) in {time.time()-t0:.0f}s; "
          f"fails={len(fails)}" + (f" {fails[:5]}" if fails else ""), flush=True)


def verify():
    clips = roster_clips()
    exp = {f"S6__{c['stem']}" for c in clips}
    lf = {f"S6__{c['stem']}": tuple(int(x) for x in c["latent_fhw"]) for c in clips}
    got_f = {os.path.basename(p)[:-4] for p in glob.glob(f"{A.FEAT}/S6__*.npz")}
    got_r = {os.path.basename(p)[:-4] for p in glob.glob(f"{A.RAW}/S6__*.npz")}
    print(f"roster {len(exp)} | feat {len(got_f)} | raw {len(got_r)}")
    for name, got in (("feat", got_f), ("raw", got_r)):
        miss = exp - got
        print(f"  {name}: {'OK set-equal' if got == exp else f'MISSING {len(miss)} {sorted(miss)[:3]}'} "
              f"| extra {len(got - exp)}")
    # spot-check 8 feat shapes vs roster
    import random
    for cid in random.Random(0).sample(sorted(exp & got_f), min(8, len(exp & got_f))):
        d = np.load(f"{A.FEAT}/{cid}.npz", allow_pickle=True)
        sh = tuple(int(x) for x in d["F"].shape[:3])
        ok = sh == lf[cid] and list(str(x) for x in d["channels"]) == A.CH_NAMES
        print(f"  {'OK ' if ok else 'BAD'} {cid[:48]:48s} F={d['F'].shape} sided={str(d['sided'])} chan_ok={list(str(x) for x in d['channels'])==A.CH_NAMES}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract"); pe.add_argument("--shard", default="0/1")
    pe.add_argument("--device", default="cuda"); pe.add_argument("--limit", type=int, default=0)
    sub.add_parser("verify")
    sp = sub.add_parser("shapes")
    a = ap.parse_args()
    if a.cmd == "extract": run_extract(a.shard, a.device, a.limit)
    elif a.cmd == "verify": verify()
    elif a.cmd == "shapes":
        from collections import Counter
        c = Counter(tuple(x["latent_fhw"]) for x in roster_clips())
        for k, v in sorted(c.items()): print(f"  latent_fhw={k}  n={v}")
        print(f"  total {sum(c.values())}")


if __name__ == "__main__":
    main()
