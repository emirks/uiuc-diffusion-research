#!/usr/bin/env python
"""Two caption anchors per endpoint-bank clip (A-role opening, B-role ending).

PRIMARY DELIVERABLE: 9-frame mp4 anchors (`<clip_id>__A.mp4`, `<clip_id>__B.mp4`), which are
what the Gemini captioner consumes - measured at 120 prompt tokens per anchor vs 1,197 for
the equivalent 3x3 JPEG sheet, better grounded, 17.9 captions/s at 150-way concurrency.
The 3x3 JPEG filmstrips are still produced as a human/agent-readable cross-check.

Every procedural transition clip in this dataset is 121 frames / 480x640 / 24 fps and is
assembled from TWO source clips: frames [0..i0] are a byte-identical copy of source A and
frames [j0..120] a byte-identical copy of source B (anchor window 9 frames each side, the
transition window is [8, 112]).  The training caption is

    "{A-side description}. sksz. {lowercase B-side description}."

so each *content* clip needs TWO descriptions - how it looks as the A-side (its opening)
and how it looks as the B-side (its ending).  This script renders the two filmstrips those
descriptions are read from:

    A-role strip : frames   0,  1,  2,  3,  4,  5,  6,  7,  8   (pure-prefix anchor window)
    B-role strip : frames 112,113,114,115,116,117,118,119,120   (pure-suffix anchor window)

The JPEG sheets lay those frames out 3x3, first frame top-left, left-to-right then
top-to-bottom; the mp4 anchors carry the same frames as video at the source's own 24 fps.

VIDEO ANCHOR ENCODING (owner-specified, verified ~63 KB/file)
    ffmpeg -y -v error -threads 1 -i <src> -vf "select='lt(n\,9)'"    -vsync 0 \
           -c:v libx264 -crf 20 -pix_fmt yuv420p <out>__A.mp4
    ffmpeg -y -v error -threads 1 -i <src> -vf "select='gte(n\,112)'" -vsync 0 \
           -c:v libx264 -crf 20 -pix_fmt yuv420p <out>__B.mp4
`-threads 1` is mandatory: ffmpeg's internal threading blows through the login node's
per-user thread ceiling at 12-way concurrency (`pthread_create() failed`), so the worker
pool must be the only concurrency knob.  This script is meant to run as an sbatch job on
the `secondary` CPU partition (see job_strips.sbatch), not on a login node.

FILMSTRIP CONVENTION (measured from data/processed/s4_refvfx/filmstrips/*.jpg, n=2000)
--------------------------------------------------------------------------------------
  grid          3x3, row-major
  gutters       4 px pure white (255,255,255) between tiles AND as an outer border
  tile          source scaled to a fixed height of 320 px, aspect preserved, width rounded
                to an even number (S4's 832x464 sources -> 574x320 tiles -> 1738x976 sheets;
                this bank's 480x640 sources -> 240x320 tiles -> 736x976 sheets, i.e. the
                same sheet HEIGHT and the same tile height, narrower because the content is
                portrait rather than landscape)
  encoding      baseline JPEG, 4:2:0 chroma subsampling, single quantization table - the
                table below is byte-identical to the one in all 2000 S4 filmstrips and to
                what `ffmpeg -q:v 3` (mjpeg) emits.  We re-use the table verbatim via PIL's
                `qtables=` so the quantization matches exactly rather than approximately.

Reads the banks read-only from the main tree; writes only under --out.

Usage
    python extract_caption_strips.py                      # all banks, 12 workers
    python extract_caption_strips.py --workers 8 --force  # re-render everything
    python extract_caption_strips.py --verify-only        # probe/verify, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- contract ---
N_FRAMES = 121
STD_W, STD_H = 480, 640
STD_FPS = 24

A_FRAMES = tuple(range(0, 9))        # pure-prefix anchor window
B_FRAMES = tuple(range(112, 121))    # pure-suffix anchor window

GRID = 3                             # 3x3
TILE_H = 320                         # S4 convention: fixed tile height, aspect preserved
GUTTER = 4                           # white padding between tiles and around the sheet
BG = (255, 255, 255)

# Luminance/chrominance quantization table shared by all 2000 S4 filmstrips; identical to
# ffmpeg's mjpeg table at -q:v 3.  Zigzag order, as PIL's `quantization`/`qtables` expect.
S4_QTABLE = [
      8,   6,   7,   8,   9,  10,  10,  12,
      6,   6,   8,   9,  10,  10,  12,  13,
      7,   8,   9,  10,  10,  12,  12,  14,
      8,   8,   9,  10,  10,  12,  13,  15,
      8,   9,  10,  10,  12,  13,  15,  18,
      9,  10,  10,  12,  13,  15,  18,  21,
      9,  10,  10,  12,  14,  17,  21,  25,
     10,  10,  13,  14,  17,  21,  25,  31,
]
SUBSAMPLING = 2                      # 4:2:0, as in the S4 sheets

# mp4 anchor encoding (owner-specified).  {sel} is the per-role select expression.
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
SELECT = {"A": r"select='lt(n\,9)'", "B": r"select='gte(n\,112)'"}
X264 = ["-vsync", "0", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"]

DEFAULT_DATA = Path(
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/processed"
)
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "data" / "processed" / "caption_strips"


# --------------------------------------------------------------- inventory ---
def build_inventory(data_root: Path):
    """-> (clips, pool_report).  clips: list of {clip_id, bank, mp4, roles}."""
    synth = data_root / "synth_endpoints"
    v2 = data_root / "ctt_v2_strata" / "endpoints_v2"
    hv = data_root / "humanvid_bank"

    clips: dict[str, dict] = {}

    def add(clip_id, bank, mp4, origin):
        e = clips.setdefault(
            clip_id, {"clip_id": clip_id, "bank": bank, "mp4": str(mp4), "origins": []}
        )
        e["origins"].append(origin)
        if os.path.realpath(e["mp4"]) != os.path.realpath(str(mp4)):
            raise RuntimeError(f"clip_id collision with different mp4: {clip_id}")

    for c in json.loads((synth / "bank_tightened.json").read_text())["clips"]:
        add(c["clip_id"], "synth_endpoints", synth / c["mp4"], "bank_tightened")

    for c in json.loads((v2 / "bank_tightened_v2.json").read_text())["clips"]:
        add(c["clip_id"], "synth_endpoints", v2 / c["mp4"], "bank_tightened_v2")

    with (hv / "manifest.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                add(r["clip_id"], "humanvid", hv / r["mp4"], "humanvid_manifest")

    # CONTENT_POOL is the authoritative list of what S2 actually drew from.
    pool_path = data_root / "ctt_v2_strata" / "CONTENT_POOL.json"
    pool = json.loads(pool_path.read_text())
    pool_ids, missing_from_banks = [], []
    for row in pool["training"] + pool["reserved"]:
        cid = row["clip_id"]
        pool_ids.append(cid)
        if cid not in clips:
            missing_from_banks.append({"clip_id": cid, "mp4": row["mp4"], "role": row["role"]})
            add(cid, "synth_endpoints", Path(row["mp4"]), "content_pool_only")
        else:
            clips[cid]["origins"].append(f"pool:{row['role']}")

    pool_report = {
        "pool_json": str(pool_path),
        "n_pool": len(pool_ids),
        "n_training": pool["n_training"],
        "n_reserved": pool["n_reserved"],
        "pool_clips_missing_from_banks": missing_from_banks,
        "bank_clips_not_in_pool": sorted(
            cid for cid, e in clips.items()
            if e["bank"] == "synth_endpoints"
            and not any(o.startswith("pool:") for o in e["origins"])
        ),
    }
    return sorted(clips.values(), key=lambda e: (e["bank"], e["clip_id"])), pool_report


# ----------------------------------------------------------------- render ----
def tile_size(w: int, h: int) -> tuple[int, int]:
    tw = int(round(w * TILE_H / h))
    tw -= tw % 2                                    # ffmpeg `scale=-2:320` rounding
    return max(tw, 2), TILE_H


def compose(frames: list[np.ndarray]) -> Image.Image:
    assert len(frames) == GRID * GRID
    h, w = frames[0].shape[:2]
    tw, th = tile_size(w, h)
    sheet_w = 2 * GUTTER + GRID * tw + (GRID - 1) * GUTTER
    sheet_h = 2 * GUTTER + GRID * th + (GRID - 1) * GUTTER
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    for k, arr in enumerate(frames):
        tile = Image.fromarray(arr)
        if (tile.width, tile.height) != (tw, th):
            tile = tile.resize((tw, th), Image.BICUBIC)   # swscale default is bicubic
        r, c = divmod(k, GRID)
        sheet.paste(tile, (GUTTER + c * (tw + GUTTER), GUTTER + r * (th + GUTTER)))
    return sheet


def save_jpeg(img: Image.Image, path: Path) -> None:
    tmp = path.with_suffix(".jpg.tmp")
    img.save(tmp, "JPEG", qtables=[S4_QTABLE], subsampling=SUBSAMPLING)
    os.replace(tmp, path)


def save_anchor(src: str, role: str, path: Path) -> None:
    """9-frame mp4 anchor, single-threaded ffmpeg (login-node thread ceiling)."""
    tmp = path.with_suffix(".mp4.tmp.mp4")
    cmd = [FFMPEG, "-y", "-v", "error", "-threads", "1", "-i", src,
           "-vf", SELECT[role], *X264, str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg {role} rc={r.returncode}: {r.stderr.strip()[:300]}")
    os.replace(tmp, path)


# ------------------------------------------------------------------ worker ---
def process(job: dict) -> dict:
    cid, mp4 = job["clip_id"], job["mp4"]
    out_a, out_b = Path(job["A"]), Path(job["B"])
    vid_a, vid_b = Path(job["Av"]), Path(job["Bv"])
    want = set(A_FRAMES) | set(B_FRAMES)
    grabbed: dict[int, np.ndarray] = {}
    n = 0
    try:
        with av.open(mp4) as container:
            st = container.streams.video[0]
            w, h = st.codec_context.width, st.codec_context.height
            fps = st.average_rate
            for frame in container.decode(video=0):
                if n in want:
                    grabbed[n] = frame.to_ndarray(format="rgb24")
                n += 1
    except Exception as exc:                                    # noqa: BLE001
        return {"clip_id": cid, "ok": False, "reason": f"decode_error: {type(exc).__name__}: {exc}"}

    std = {"w": int(w), "h": int(h), "frames": int(n),
           "fps": float(fps) if fps is not None else None}

    bad = []
    if n != N_FRAMES:
        bad.append(f"frames={n} (want {N_FRAMES})")
    if (w, h) != (STD_W, STD_H):
        bad.append(f"size={w}x{h} (want {STD_W}x{STD_H})")
    if fps is None or Fraction(fps) != Fraction(STD_FPS):
        bad.append(f"fps={fps} (want {STD_FPS})")
    if bad:
        return {"clip_id": cid, "ok": False, "reason": "; ".join(bad), "std": std}
    if not want.issubset(grabbed):
        return {"clip_id": cid, "ok": False, "reason": "missing decoded frames", "std": std}

    strips = vids = 0
    try:
        if job["write_A"]:
            save_jpeg(compose([grabbed[i] for i in A_FRAMES]), out_a)
            strips += 1
        if job["write_B"]:
            save_jpeg(compose([grabbed[i] for i in B_FRAMES]), out_b)
            strips += 1
        if job["write_Av"]:
            save_anchor(mp4, "A", vid_a)
            vids += 1
        if job["write_Bv"]:
            save_anchor(mp4, "B", vid_b)
            vids += 1
    except Exception as exc:                                    # noqa: BLE001
        return {"clip_id": cid, "ok": False, "std": std,
                "reason": f"encode_error: {type(exc).__name__}: {exc}"}

    # the anchors must contain exactly the 9 frames we asked for
    for role, path, want_n in (("A", vid_a, len(A_FRAMES)), ("B", vid_b, len(B_FRAMES))):
        try:
            with av.open(str(path)) as c:
                got = sum(1 for _ in c.decode(video=0))
        except Exception as exc:                                # noqa: BLE001
            return {"clip_id": cid, "ok": False, "std": std,
                    "reason": f"anchor_{role}_unreadable: {type(exc).__name__}: {exc}"}
        if got != want_n:
            return {"clip_id": cid, "ok": False, "std": std,
                    "reason": f"anchor_{role} has {got} frames (want {want_n})"}

    return {"clip_id": cid, "ok": True, "std": std, "strips": strips, "videos": vids}


# -------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="re-render strips that already exist")
    ap.add_argument("--verify-only", action="store_true", help="verify + inventory, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N clips")
    args = ap.parse_args()

    try:
        os.nice(args.nice)
    except OSError:
        pass

    t0 = time.time()
    clips, pool_report = build_inventory(args.data_root)
    if args.limit:
        clips = clips[: args.limit]
    print(f"[inventory] {len(clips)} unique clips "
          f"({sum(c['bank'] == 'synth_endpoints' for c in clips)} synth_endpoints / "
          f"{sum(c['bank'] == 'humanvid' for c in clips)} humanvid)", flush=True)
    print(f"[pool] {pool_report['n_pool']} pool clips "
          f"({pool_report['n_training']} training + {pool_report['n_reserved']} reserved); "
          f"missing from banks: {len(pool_report['pool_clips_missing_from_banks'])}; "
          f"bank clips outside the pool: {len(pool_report['bank_clips_not_in_pool'])}", flush=True)

    index_path = args.out / "strips_index.json"
    prev = {}
    if index_path.exists():
        try:
            prev = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            prev = {}

    def have(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 0

    jobs, index, reused, skipped = [], {}, 0, []
    for c in clips:
        d = args.out / c["bank"]
        a, b = d / f"{c['clip_id']}__A.jpg", d / f"{c['clip_id']}__B.jpg"
        av_, bv = d / f"{c['clip_id']}__A.mp4", d / f"{c['clip_id']}__B.mp4"
        entry = {"bank": c["bank"], "mp4": c["mp4"],
                 "A_strip": str(a), "B_strip": str(b),
                 "A_video": str(av_), "B_video": str(bv)}
        entry_prev = prev.get(c["clip_id"])
        present = {p: have(p) for p in (a, b, av_, bv)}
        if all(present.values()) and entry_prev and "std" in entry_prev and not args.force:
            index[c["clip_id"]] = {**entry, "std": entry_prev["std"]}
            reused += 1
            continue
        jobs.append({**entry, "clip_id": c["clip_id"], "A": str(a), "B": str(b),
                     "Av": str(av_), "Bv": str(bv),
                     "write_A": args.force or not present[a],
                     "write_B": args.force or not present[b],
                     "write_Av": args.force or not present[av_],
                     "write_Bv": args.force or not present[bv]})

    print(f"[plan] {len(jobs)} clips to decode, {reused} reused from a previous run", flush=True)
    if args.verify_only:
        for j in jobs:
            j["write_A"] = j["write_B"] = j["write_Av"] = j["write_Bv"] = False
    else:
        for bank in {c["bank"] for c in clips}:
            (args.out / bank).mkdir(parents=True, exist_ok=True)

    n_strips = n_videos = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:                        # noqa: BLE001
                    r = {"clip_id": j["clip_id"], "ok": False,
                         "reason": f"worker_error: {type(exc).__name__}: {exc}"}
                if r["ok"]:
                    n_strips += r.get("strips", 0)
                    n_videos += r.get("videos", 0)
                    index[r["clip_id"]] = {
                        "bank": j["bank"], "mp4": j["mp4"],
                        "A_strip": j["A"], "B_strip": j["B"],
                        "A_video": j["Av"], "B_video": j["Bv"], "std": r["std"]}
                else:
                    skipped.append({"clip_id": r["clip_id"], "bank": j["bank"], "mp4": j["mp4"],
                                    "reason": r["reason"], "std": r.get("std")})
                if i % 250 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] {time.time() - t0:6.1f}s "
                          f"strips={n_strips} anchors={n_videos} skipped={len(skipped)}",
                          flush=True)

    index = dict(sorted(index.items()))
    if not args.verify_only:
        args.out.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=1) + "\n")
        (args.out / "strips_skipped.json").write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "convention": {"grid": f"{GRID}x{GRID}", "tile_height": TILE_H, "gutter": GUTTER,
                            "jpeg": "ffmpeg -q:v 3 qtable, 4:2:0",
                            "matched_from": "data/processed/s4_refvfx/filmstrips",
                            "video": "libx264 crf 20 yuv420p, 9 frames, -threads 1"},
             "A_frames": list(A_FRAMES), "B_frames": list(B_FRAMES),
             "n_indexed": len(index), "n_skipped": len(skipped),
             "skipped": skipped, "pool_report": pool_report}, indent=1) + "\n")

    dt = time.time() - t0
    print(f"\n[done] indexed {len(index)} clips ({2 * len(index)} strips + "
          f"{2 * len(index)} anchors; wrote {n_strips} strips / {n_videos} anchors this run, "
          f"{reused} clips reused), skipped {len(skipped)}, wall {dt:.1f}s")
    for s in skipped[:20]:
        print(f"  SKIP {s['clip_id']}: {s['reason']}")
    if not args.verify_only:
        print(f"[index] {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
