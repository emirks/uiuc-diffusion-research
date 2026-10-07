"""CTT v2 — S6 (EffectData) VAE-encode `latents/` + `cond_clean/`, DeltaAI-native.

Sibling of `scripts/ctt_v2/encode/encode_strata.py` (Campus-Cluster S0–S4) but rewritten for
DeltaAI (aarch64, /taiga) and for the two things EffectData does differently:

  1. FOUR native shapes, not one. Every S0–S4 stratum has a single `--resolution-buckets`;
     EffectData subjects come at 4 VAE-legal resolutions (all 81f / 24fps, no crop). A single
     `process_videos.py` invocation takes ONE bucket, so a shard is always single-shape and we
     pass that shape's exact WxHx81. Native == bucket ⇒ `_resize_and_crop` scales by 1.0 with a
     0-pixel crop (identity), same trick S4 used for its width.
  2. ONE-SIDED. cond_clean is a bitwise copy (`write_cond_clean(correct_suffix=False)`), so the
     step never loads the VAE — the ONLY GPU work is the latent encode.

Clips live inside per-effect zips (`data/raw/effectdata/Videos/<Effect>.zip`, member
`<Effect>/<stem>.mp4`); `encode` extracts a shard's clips just-in-time (idempotent) so staging
parallelises across the array instead of serialising on a login node.

Sharding is per-shape: SLURM_ARRAY_TASK_ID -> (shape = id // NSHARDS_PER_SHAPE,
shard = id % NSHARDS_PER_SHAPE). The roster (`ROSTER.json`, frozen) defines clip order, so a
resubmit at the same NSHARDS_PER_SHAPE re-derives the identical partition. Everything is
idempotent (skip-if-exists, .tmp + os.replace), so a preempted task refills cleanly.

Authority
---------
* outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json  — frozen roster (28,644 clips, 4 shapes)
* eval_ladder/encode_conditioning.py             — write_cond_clean (the ONE cond definition)
* scripts/ctt_v2/encode/encode_strata.py         — the proven process_videos call this mirrors

Usage
-----
    # smoke gate (1 GPU, ~10 min): a few clips of EVERY shape + shape/fps assert. Exits !=0 on mismatch.
    python encode_s6.py pilot --per-shape 6

    # one shard of one shape (GPU, job array)
    python encode_s6.py encode --shape $((SLURM_ARRAY_TASK_ID / NSPS)) --shard $((SLURM_ARRAY_TASK_ID % NSPS))

    # CPU: count assert + shape/fps/cond_clean spot check
    python encode_s6.py verify
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                              # .../diffusion-research
LAB = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa")
TRAINER = REPO_ROOT / "src/LTX-2-official/packages/ltx-trainer"
VENV_PY = LAB / "envs-aarch64/ltx2/bin/python"           # aarch64 py3.12 (source $LAB/envs-aarch64/activate)
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"

sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))       # encode_conditioning
sys.path.insert(0, os.environ.get("LTX_TRAINER_SRC", str(TRAINER / "src")))

ENC = REPO_ROOT / "outputs/ctt_v2/encodes/EFFECTDATA"    # roster already lives here
ZIPS = REPO_ROOT / "data/raw/effectdata/Videos"
LAT, CC = ENC / "latents", ENC / "cond_clean"            # the deliverables: stay on /taiga
ROSTER_P = ENC / "ROSTER.json"
#: extracted clip mp4s are TRANSIENT (only needed for the latent encode + frame-count preflight;
#: one-sided cond_clean never opens them). Default is /taiga, but the sbatch overrides to
#: node-local /tmp (3.9 TB, purged at job end) so 143 GB of mp4s never touch the 92%-full taiga.
CLIPS = Path(os.environ.get("S6_CLIPS_DIR", str(ENC / "clips")))

# --------------------------------------------------------------------------------------
# HARDCODED partition width. NEVER derive from SLURM_ARRAY_TASK_COUNT (see encode_strata's
# 2026-07-28 silent-repartition incident). 4 shapes x 12 = 48 array tasks.
# --------------------------------------------------------------------------------------
NSHARDS_PER_SHAPE = 12
FPS = 24.0
FRAMES = 81


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------- roster
def _roster() -> dict:
    if not ROSTER_P.exists():
        raise SystemExit(f"[s6] no frozen roster at {ROSTER_P}")
    return json.loads(ROSTER_P.read_text())


def shapes() -> list[dict]:
    """The 4 native shapes, in a STABLE order (sorted by (w,h)). Each carries its clip list."""
    r = _roster()
    by_wh: dict[tuple[int, int], list[dict]] = {}
    for c in r["clips"]:
        by_wh.setdefault((int(c["w"]), int(c["h"])), []).append(c)
    out = []
    for (w, h) in sorted(by_wh):                          # deterministic shape index
        clips = by_wh[(w, h)]                             # roster order preserved within a shape
        fhw = clips[0]["latent_fhw"]                      # [F', H', W']
        out.append({"w": w, "h": h, "label": f"{w}x{h}x{FRAMES}",
                    "latent_fhw": fhw, "n": len(clips), "clips": clips})
    return out


def shard_stems(shape: dict, shard: int, nshards: int) -> list[dict]:
    return [c for i, c in enumerate(shape["clips"]) if i % nshards == shard] if 0 <= shard < nshards else []


# ----------------------------------------------------------------------------------- stage
def _zip_for(effect: str) -> Path:
    return ZIPS / f"{effect}.zip"


def ensure_clips(clips: list[dict]) -> None:
    """Extract any missing clip mp4s from their per-effect zips into CLIPS/ (flat, idempotent).

    Grouped by zip so each archive is opened once. Member path inside the zip is the roster's
    `video_path` (`<Effect>/<stem>.mp4`); we write it flat as `<stem>.mp4`.
    """
    CLIPS.mkdir(parents=True, exist_ok=True)
    need: dict[str, list[dict]] = {}
    for c in clips:
        dst = CLIPS / f"{c['stem']}.mp4"
        if not (dst.exists() and dst.stat().st_size > 0):
            need.setdefault(c["effect"], []).append(c)
    if not need:
        return
    total = sum(len(v) for v in need.values())
    log(f"[stage] extracting {total} clips from {len(need)} zips")
    done = 0
    for effect, items in need.items():
        zp = _zip_for(effect)
        if not zp.exists():
            raise SystemExit(f"[stage] missing zip {zp}")
        with zipfile.ZipFile(zp) as zf:
            for c in items:
                member = c["video_path"]                  # e.g. Effect/Effect,subj,Z.mp4
                dst = CLIPS / f"{c['stem']}.mp4"
                tmp = dst.with_suffix(".mp4.tmp")
                with zf.open(member) as src, open(tmp, "wb") as fh:
                    fh.write(src.read())
                os.replace(tmp, dst)
                done += 1
                if done % 200 == 0:
                    log(f"[stage] {done}/{total} extracted")
    log(f"[stage] {done} clips extracted -> {CLIPS}")


# ---------------------------------------------------------------------------------- encode
def run_process_videos(manifest: Path, out_dir: Path, w: int, h: int, f: int,
                       device: str, vae_tiling: bool) -> None:
    cmd = [str(VENV_PY), "scripts/process_videos.py", str(manifest),
           "--resolution-buckets", f"{w}x{h}x{f}",
           "--output-dir", str(out_dir), "--model-path", str(MODEL),
           "--video-column", "video", "--device", device]
    if vae_tiling:
        cmd.append("--vae-tiling")
    log("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(TRAINER), check=True,
                   env={**os.environ, "PYTHONPATH": str(TRAINER / "src")})


def encode_shape_shard(shape: dict, shard: int, nshards: int, device: str,
                       limit: int = 0, vae_tiling: bool = False, clips_override=None) -> dict:
    import torch

    import encode_conditioning as ec  # noqa: PLC0415

    w, h, f = shape["w"], shape["h"], FRAMES
    want = clips_override if clips_override is not None else shard_stems(shape, shard, nshards)
    if limit:
        want = want[:limit]
    if not want:
        log(f"[encode] {shape['label']} shard {shard}: empty — skip")
        return {"shape": shape["label"], "shard": shard, "clips": 0}
    stems = [c["stem"] for c in want]
    log(f"[encode] {shape['label']} shard {shard}/{nshards}: {len(want)} clips, bucket {w}x{h}x{f}")

    for d in (LAT, CC):
        d.mkdir(parents=True, exist_ok=True)
    ensure_clips(want)

    # -- pre-flight: process_videos SILENTLY DROPS clips shorter than the bucket -------------
    from ltx_trainer.video_utils import get_video_frame_count  # noqa: PLC0415

    short = [(s, int(get_video_frame_count(CLIPS / f"{s}.mp4")))
             for s in stems if int(get_video_frame_count(CLIPS / f"{s}.mp4")) < f]
    if short:
        raise SystemExit(f"[encode] {shape['label']}: {len(short)} clips < {f} frames "
                         f"(SILENT-SKIP risk): {short[:5]}")

    # -- step 1: full-video latents ---------------------------------------------------------
    todo = [s for s in stems if not (LAT / f"{s}.pt").exists()]
    log(f"[encode] {shape['label']}: {len(todo)} latents to encode ({len(stems) - len(todo)} present)")
    if todo:
        man = CLIPS / f"_manifest_{shape['label']}_shard{shard:02d}.json"
        man.write_text(json.dumps([{"video": f"{s}.mp4"} for s in todo], indent=1))
        run_process_videos(man, LAT, w, h, f, device, vae_tiling)
    miss = [s for s in stems if not (LAT / f"{s}.pt").exists()]
    if miss:
        raise SystemExit(f"[encode] {shape['label']}: {len(miss)} latents missing after encode "
                         f"(SILENT DROP): {miss[:5]}")

    # -- step 2: cond_clean (one-sided => bitwise copy, no VAE) ------------------------------
    todo_cc = [s for s in stems if not (CC / f"{s}.pt").exists()]
    log(f"[encode] {shape['label']}: {len(todo_cc)} cond_clean to write (bitwise copy)")
    lf = tuple(shape["latent_fhw"])                       # [F',H',W'] expected
    t0, shapes_seen = time.time(), set()
    for i, s in enumerate(todo_cc, 1):
        src = LAT / f"{s}.pt"
        d = torch.load(src, map_location="cpu", weights_only=True)
        assert set(d) >= {"latents", "num_frames", "height", "width", "fps"}, \
            f"{s}: payload keys {sorted(d)}"
        assert abs(float(d["fps"]) - FPS) < 1e-6, f"{s}: fps={d['fps']} want {FPS}"
        assert (int(d["num_frames"]), int(d["height"]), int(d["width"])) == lf, \
            f"{s}: latent FHW {(d['num_frames'], d['height'], d['width'])} want {lf}"
        shapes_seen.add(tuple(d["latents"].shape))
        del d
        tmp = CC / f"{s}.pt.tmp"
        ec.write_cond_clean(orig_pt=src, dst_pt=tmp, clip_mp4=CLIPS / f"{s}.mp4",
                            correct_suffix=False, vae=None, device=device)
        os.replace(tmp, CC / f"{s}.pt")
        if i % 200 == 0 or i == len(todo_cc):
            log(f"[encode] {shape['label']}: cond_clean {i}/{len(todo_cc)} "
                f"({(time.time() - t0) / i:.3f}s/clip)")

    miss_cc = [s for s in stems if not (CC / f"{s}.pt").exists()]
    if miss_cc:
        raise SystemExit(f"[encode] {shape['label']}: cond_clean missing: {miss_cc[:5]}")
    log(f"[encode] {shape['label']} shard {shard} DONE: {len(stems)} latents + {len(stems)} cond_clean; "
        f"latent shapes={sorted(str(x) for x in shapes_seen)}")
    return {"shape": shape["label"], "shard": shard, "clips": len(stems),
            "latents_new": len(todo), "cond_clean_new": len(todo_cc),
            "shapes": sorted(str(x) for x in shapes_seen)}


# ----------------------------------------------------------------------------------- pilot
def pilot(per_shape: int, device: str, vae_tiling: bool) -> int:
    """Encode the first `per_shape` clips of EVERY shape, then assert shape/fps/cond_clean.
    Exits non-zero on any mismatch — this is the afterok gate for the full array."""
    import torch

    shp = shapes()
    log(f"[pilot] {len(shp)} shapes x {per_shape} clips each, tiling={vae_tiling}")
    for sh in shp:
        encode_shape_shard(sh, shard=0, nshards=1, device=device, limit=per_shape,
                           vae_tiling=vae_tiling, clips_override=sh["clips"][:per_shape])
    # assertions
    bad = 0
    for sh in shp:
        lf = tuple(sh["latent_fhw"])
        for c in sh["clips"][:per_shape]:
            s = c["stem"]
            a = torch.load(LAT / f"{s}.pt", map_location="cpu", weights_only=True)
            b = torch.load(CC / f"{s}.pt", map_location="cpu", weights_only=True)
            ok = (abs(float(a["fps"]) - FPS) < 1e-6
                  and (int(a["num_frames"]), int(a["height"]), int(a["width"])) == lf
                  and torch.equal(a["latents"], b["latents"]))       # one-sided: byte-identical
            bad += not ok
            print(f"  {'OK ' if ok else 'BAD'} {sh['label']:14s} {s:44s} "
                  f"shape={tuple(a['latents'].shape)} fps={a['fps']} cc==lat={torch.equal(a['latents'], b['latents'])}")
    print(f"[pilot] {'ALL PASS' if bad == 0 else f'{bad} FAILURES'}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------------- verify
def verify(n_sample: int, seed: int, count: bool) -> int:
    import torch

    rng = random.Random(seed)
    shp = shapes()
    exp = {c["stem"] for sh in shp for c in sh["clips"]}
    got_l = {p.stem for p in LAT.glob("*.pt")} if LAT.exists() else set()
    got_c = {p.stem for p in CC.glob("*.pt")} if CC.exists() else set()
    print(f"roster {len(exp)} | latents {len(got_l)} | cond_clean {len(got_c)}")
    bad = 0
    for name, got in (("latents", got_l), ("cond_clean", got_c)):
        if got == exp:
            print(f"  OK {name}: exactly {len(exp)}, set-equal to roster")
        elif count:
            bad += 1
            print(f"  BAD {name}: MISSING {len(exp - got)} {sorted(exp - got)[:5]} | EXTRA {len(got - exp)}")
        else:
            print(f"  .. {name}: {len(got)}/{len(exp)} (count assert skipped)")
    lf_by_stem = {c["stem"]: tuple(sh["latent_fhw"]) for sh in shp for c in sh["clips"]}
    pool = sorted(exp & got_l & got_c)
    for s in (rng.sample(pool, min(n_sample, len(pool))) if pool else []):
        a = torch.load(LAT / f"{s}.pt", map_location="cpu", weights_only=True)
        b = torch.load(CC / f"{s}.pt", map_location="cpu", weights_only=True)
        lf = lf_by_stem[s]
        ok = (abs(float(a["fps"]) - FPS) < 1e-6
              and (int(a["num_frames"]), int(a["height"]), int(a["width"])) == lf
              and torch.equal(a["latents"], b["latents"]))
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {s:44s} shape={tuple(a['latents'].shape)} "
              f"fps={a['fps']} cc==lat={torch.equal(a['latents'], b['latents'])}")
    print(f"[verify] {'ALL CHECKS PASS' if bad == 0 else f'{bad} FAILURES'}")
    return 1 if bad else 0


# ------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="encode one shard of one shape (GPU)")
    pe.add_argument("--shape", type=int, required=True, help="shape index 0..3 (see `shapes`)")
    pe.add_argument("--shard", type=int, required=True)
    pe.add_argument("--nshards", type=int, default=NSHARDS_PER_SHAPE)
    pe.add_argument("--device", default="cuda")
    pe.add_argument("--limit", type=int, default=0)
    pe.add_argument("--vae-tiling", action="store_true")

    pp = sub.add_parser("pilot", help="encode a few clips of every shape + assert (GPU)")
    pp.add_argument("--per-shape", type=int, default=6)
    pp.add_argument("--device", default="cuda")
    pp.add_argument("--vae-tiling", action="store_true")

    pv = sub.add_parser("verify", help="count assert + shape/fps/cond_clean spot check (CPU)")
    pv.add_argument("--n-sample", type=int, default=6)
    pv.add_argument("--no-count", action="store_true")

    ps = sub.add_parser("shapes", help="print the 4 shapes + counts and exit (CPU)")

    args = ap.parse_args()
    if args.cmd == "shapes":
        for i, sh in enumerate(shapes()):
            print(f"  [{i}] {sh['label']:14s} latent_fhw={sh['latent_fhw']} n={sh['n']}")
        return 0
    if args.cmd == "verify":
        return verify(args.n_sample, seed=42, count=not args.no_count)
    if args.cmd == "pilot":
        return pilot(args.per_shape, args.device, args.vae_tiling)

    shp = shapes()
    if not (0 <= args.shape < len(shp)):
        raise SystemExit(f"[encode] shape index {args.shape} out of range 0..{len(shp) - 1}")
    log(f"[encode] host={os.uname().nodename} shape={args.shape} shard={args.shard} "
        f"nshards={args.nshards}")
    rep = encode_shape_shard(shp[args.shape], args.shard, args.nshards, args.device,
                             args.limit, args.vae_tiling)
    log("[encode] shard report: " + json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
