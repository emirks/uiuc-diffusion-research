#!/usr/bin/env python
"""grid v3 — media preparation: sources -> standard clips -> conditioning windows -> caption strips.

Everything the corpus build did, replicated rule-for-rule so the new members are indistinguishable
from the old ones downstream (run_gen, run_eval.pool_refs, the manifest builder, the captioner):

  standard clip   480x640 (cover-scale + centre-crop), 24 fps, libx264 crf 14 yuv420p.
                  Higgsfield: WHOLE-CLIP linspace resample to 121 frames — measured on the existing
                  corpus 2026-09-07 (shadow_0 141->121 and flying_cam_transition_4 242->121 map to
                  round(linspace(0, N-1, 121)); the raw pull IS the corpus source, frame-identical).
                  EffectData: native 81 frames, no temporal resample (owner 2026-09-07).
  source copy     data/processed/transitions/<side>_transitions/<raw_dir>/<stem>.mp4 — the raw tree the
                  manifest builder (build_corpus_manifest.py) resolves classes and sidedness from.
  cond windows    encode_conditioning.cut_windows (start9 / end9, isolation-encoded downstream) for
                  121-frame clips; the identical prefix command for 81-frame EffectData clips.
                  Reserve two-sided pairs `hvpair.<a>.<b>`: prefix = a's first 9 frames, suffix = b's
                  FIRST 9 frames (the DAVIS rule: any well-framed window is a valid end scene).
  caption strips  9-frame A (frames 0-8) and B (frames 112-120) mp4 anchors, the
                  extract_caption_strips.py encoding, for every NEW Higgsfield endpoint clip; appended to
                  data/processed/caption_strips/strips_index.json (original backed up once).

CPU only; single-threaded ffmpeg/PyAV per worker (login-node thread ceiling).
Run:  PATH=<ffmpeg 6.1.1 bin>:$PATH OPENBLAS_NUM_THREADS=1 python scripts/grid_v3/prepare_media.py [--workers 3] [--only higgsfield|effectdata|reserve]
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

cv2.setNumThreads(1)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder"))
import encode_conditioning as ec  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
W, H, FPS = 480, 640, 24
STD = REPO / "data/processed/transitions_std121"
RAW_TREE = REPO / "data/processed/transitions"
RAW_PULL = REPO / "data/processed/higgsfield_transitions"
STRIPS = REPO / "data/processed/caption_strips"
GRID = yaml.safe_load((REPO / "eval_ladder/grid_v3.yaml").read_text())
PENDING = REPO / "eval_ladder/registry_v3_pending.jsonl"
HEALTH = REPO / "misc/2026-09-07_eval_grid_v2/health.json"
OUT_MANIFEST = REPO / "misc/2026-09-07_eval_grid_v2/media_manifest.json"


# --------------------------------------------------------------------------- video helpers
def decode(path: Path) -> tuple[list[np.ndarray], float]:
    import av
    c = av.open(str(path))
    s = c.streams.video[0]
    s.thread_type = "NONE"
    s.thread_count = 1
    frames = [f.to_ndarray(format="bgr24") for f in c.decode(s)]
    fps = float(s.average_rate or FPS)
    c.close()
    return frames, fps


def cover_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(W / w, H / h)
    nw, nh = max(W, int(round(w * scale))), max(H, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    r = cv2.resize(frame, (nw, nh), interpolation=interp)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return np.ascontiguousarray(r[y0:y0 + H, x0:x0 + W])


def encode_std(frames: list[np.ndarray], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    p = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-threads", "1", str(tmp)],
        stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg encode failed: {dst}"
    os.replace(tmp, dst)


def to_std(src: Path, dst: Path, n_out: int | None) -> dict:
    """n_out=121: whole-clip linspace resample (corpus rule). n_out=None: keep every frame (EffectData native)."""
    frames, fps = decode(src)
    n = len(frames)
    idx = np.round(np.linspace(0, n - 1, n_out)).astype(int) if n_out else np.arange(n)
    encode_std([cover_crop(frames[i]) for i in idx], dst)
    return {"src_frames": n, "src_fps": fps, "src_wh": [frames[0].shape[1], frames[0].shape[0]],
            "std_frames": int(len(idx)), "rule": "linspace_121" if n_out else "native"}


def ffmpeg_select(src: Path, dst: Path, select: str, crf: int, preset: str | None = None) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", "-i", str(src),
           "-vf", f"{select},setpts=N/24/TB", "-r", "24", "-c:v", "libx264"]
    if preset:
        cmd += ["-preset", preset]
    cmd += ["-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", "1", str(dst)]
    subprocess.run(cmd, check=True)


def cut_prefix_window(std_mp4: Path, clip: str) -> None:
    """The prefix half of encode_conditioning.cut_windows, for clips shorter than 121 frames."""
    ffmpeg_select(std_mp4, ec.CONDS / f"{clip}_start9.mp4", "select='lt(n,9)'", crf=12, preset="slow")


def cut_strips(std_mp4: Path, clip: str, n_frames: int, out_dir: Path) -> dict:
    a = out_dir / f"{clip}__A.mp4"
    b = out_dir / f"{clip}__B.mp4"
    ffmpeg_select(std_mp4, a, r"select='lt(n\,9)'", crf=20)
    ffmpeg_select(std_mp4, b, rf"select='gte(n\,{n_frames - 9})'", crf=20)
    return {"A_video": str(a), "B_video": str(b)}


# --------------------------------------------------------------------------- work lists
def higgsfield_jobs() -> list[dict]:
    man = json.loads((STD / "corpus_manifest.json").read_text())
    jobs = []
    for key in ("tier1_topups", "tier1_reference_only", "flame_additions", "zs_pool_topups"):
        for cls, stems in GRID[key].items():
            raw_dir = REPO / man["classes"][cls]["raw_dir"]
            sided = man["classes"][cls]["sidedness"]
            for stem in stems:
                jobs.append({"cls": cls, "stem": stem, "raw_dir": raw_dir, "sided": sided, "n_out": 121})
    for cls, sided in GRID["new_zero_shot_classes"].items():
        raw_dir = RAW_TREE / f"{'onesided' if sided == 'one' else 'twosided'}_transitions" / \
            f"{'onesided' if sided == 'one' else 'twosided'}_object_{cls.replace('_', '-')}"
        for p in sorted((RAW_PULL / cls).glob("*.mp4")):
            jobs.append({"cls": cls, "stem": p.stem, "raw_dir": raw_dir,
                         "sided": "onesided" if sided == "one" else "twosided", "n_out": 121})
    return jobs


def raw_pull_path(stem: str, cls: str) -> Path:
    p = RAW_PULL / cls / f"{stem}.mp4"
    if p.exists():
        return p
    hits = sorted(RAW_PULL.glob(f"*/{stem}.mp4"))
    assert hits, f"raw clip not found: {stem}"
    return hits[0]


def effectdata_jobs() -> list[dict]:
    cfg = GRID["effectdata"]
    ann = json.loads((REPO / cfg["annotations"]).read_text())
    recs = list(ann.values()) if isinstance(ann, dict) else ann
    pref = cfg["clip_prefix"]
    member: dict[str, tuple[str, str]] = {}          # std_stem -> (effect, zip member path)
    for r in recs:
        fn = r["video_path"].rsplit("/", 1)[-1][:-4]
        parts = fn.split(",")
        if len(parts) != 3:
            continue
        effect, subject, tag = parts
        std_stem = f"{pref}.{effect.replace('-', '_')}.{subject}.{tag}"
        member[std_stem] = (effect, r["video_path"])
    need: set[str] = set()
    for r in map(json.loads, PENDING.read_text().splitlines()):
        for k in ("endpoint", "reference"):
            v = r.get(k)
            if v and v.startswith(pref + "."):
                need.add(v)
    blocks = json.loads(HEALTH.read_text())["effectdata_blocks"]
    for b in blocks:
        for e, pool in b.get("pools", {}).items():
            need.update(pool)
    jobs = []
    for std_stem in sorted(need):
        effect, mem = member[std_stem]
        cls = ".".join(std_stem.split(".")[:2])
        jobs.append({"cls": cls, "stem": std_stem, "effect": effect, "member": mem,
                     "raw_dir": RAW_TREE / "onesided_transitions" / f"onesided_object_{cls}", "sided": "onesided", "n_out": None})
    return jobs


def reserve_jobs() -> tuple[list[dict], list[tuple[str, str, str]]]:
    pool = json.loads((REPO / GRID["reserve"]["pool"]).read_text())
    mp4 = {r["clip_id"]: Path(r["mp4"].replace("/projects/illinois", "/taiga/illinois"))
           for r in pool["reserved"] if r["bank"] == GRID["reserve"]["bank"]}
    singles, pairs = set(), []
    for r in map(json.loads, PENDING.read_text().splitlines()):
        ep = r["endpoint"]
        if r["endpoint_source"] != "humanvid":
            continue
        if ep.startswith("hvpair."):
            _, a, b = ep.split(".", 2)
            pairs.append((ep, a, b))
            singles.update((a, b))
        else:
            singles.add(ep)
    return [{"clip": c, "mp4": mp4[c]} for c in sorted(singles)], sorted(set(pairs))


# --------------------------------------------------------------------------- runners
def do_higgsfield(job: dict) -> dict:
    cls, stem = job["cls"], job["stem"]
    src = raw_pull_path(stem, cls)
    src_copy = job["raw_dir"] / f"{stem}.mp4"
    if not src_copy.exists():
        src_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, src_copy)
    dst = STD / cls / f"{stem}.mp4"
    info = {"cls": cls, "stem": stem, "source": str(src_copy.relative_to(REPO)), "std": str(dst.relative_to(REPO)), "sided": job["sided"]}
    if not dst.exists():
        info.update(to_std(src, dst, job["n_out"]))
    ec.cut_windows(dst, stem)
    info["conds"] = [str(ec.CONDS / f"{stem}_start9.mp4"), str(ec.CONDS / f"{stem}_end9.mp4")]
    info["strips"] = cut_strips(dst, stem, 121, STRIPS / "grid_v3")
    return info


def do_effectdata(job: dict, zips: dict) -> dict:
    cls, stem = job["cls"], job["stem"]
    src_copy = job["raw_dir"] / f"{stem}.mp4"
    if not src_copy.exists():
        src_copy.parent.mkdir(parents=True, exist_ok=True)
        z = zips.setdefault(job["effect"], zipfile.ZipFile(REPO / GRID["effectdata"]["zips"] / f"{job['effect']}.zip"))
        src_copy.write_bytes(z.read(job["member"]))
    dst = STD / cls / f"{stem}.mp4"
    info = {"cls": cls, "stem": stem, "source": str(src_copy.relative_to(REPO)), "std": str(dst.relative_to(REPO)), "sided": "onesided"}
    if not dst.exists():
        info.update(to_std(src_copy, dst, None))
        assert info["std_frames"] == GRID["effectdata"]["frames"] if "frames" in GRID["effectdata"] else info["std_frames"] == 81, info
    cut_prefix_window(dst, stem)
    info["conds"] = [str(ec.CONDS / f"{stem}_start9.mp4")]
    return info


def do_reserve(job: dict) -> dict:
    ec.cut_windows(job["mp4"], job["clip"])
    return {"clip": job["clip"], "mp4": str(job["mp4"]), "conds": [str(ec.CONDS / f"{job['clip']}_start9.mp4"), str(ec.CONDS / f"{job['clip']}_end9.mp4")]}


def do_pair(pair: tuple[str, str, str]) -> dict:
    ep, a, b = pair
    pre, suf = ec.CONDS / f"{ep}_start9.mp4", ec.CONDS / f"{ep}_end9.mp4"
    if not pre.exists():
        shutil.copy2(ec.CONDS / f"{a}_start9.mp4", pre)
    if not suf.exists():
        shutil.copy2(ec.CONDS / f"{b}_start9.mp4", suf)         # DAVIS rule: suffix = FIRST 9 frames of B
    return {"clip": ep, "prefix_from": a, "suffix_from": b, "conds": [str(pre), str(suf)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", choices=["higgsfield", "effectdata", "reserve"], default=None)
    args = ap.parse_args()
    ec.CONDS.mkdir(parents=True, exist_ok=True)
    out: dict = {"higgsfield": [], "effectdata": [], "reserve": [], "reserve_pairs": [], "errors": []}

    def run(name, jobs, fn):
        print(f"[{name}] {len(jobs)} jobs", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(lambda j: _safe(fn, j, name, out), jobs), 1):
                if r:
                    out[name].append(r)
                if i % 25 == 0:
                    print(f"  [{name}] {i}/{len(jobs)}", flush=True)

    if args.only in (None, "higgsfield"):
        run("higgsfield", higgsfield_jobs(), do_higgsfield)
    if args.only in (None, "effectdata"):
        zips: dict = {}
        run("effectdata", effectdata_jobs(), lambda j: do_effectdata(j, zips))
    if args.only in (None, "reserve"):
        singles, pairs = reserve_jobs()
        run("reserve", singles, do_reserve)
        out["reserve_pairs"] = [do_pair(p) for p in pairs]

    # strips index: append the new Higgsfield endpoint clips (additive; original backed up once)
    if out["higgsfield"]:
        idx_path = STRIPS / "strips_index.json"
        bak = STRIPS / "strips_index.pre_grid_v3.json"
        if not bak.exists():
            shutil.copy2(idx_path, bak)
        idx = json.loads(idx_path.read_text())
        for r in out["higgsfield"]:
            idx[r["stem"]] = {"bank": "grid_v3", "mp4": str(REPO / r["std"]), "A_strip": None, "B_strip": None,
                              "A_video": r["strips"]["A_video"], "B_video": r["strips"]["B_video"],
                              "std": {"w": W, "h": H, "frames": 121, "fps": float(FPS)}}
        idx_path.write_text(json.dumps(idx, indent=1, sort_keys=True))
        print(f"[strips] index now {len(idx)} clips (+{len(out['higgsfield'])})")
    OUT_MANIFEST.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"[done] higgsfield {len(out['higgsfield'])} · effectdata {len(out['effectdata'])} · reserve {len(out['reserve'])} "
          f"(+{len(out['reserve_pairs'])} pairs) · errors {len(out['errors'])} -> {OUT_MANIFEST.relative_to(REPO)}")


def _safe(fn, job, name, out):
    try:
        return fn(job)
    except Exception as e:  # keep going; the manifest lists failures
        out["errors"].append({"lane": name, "job": {k: str(v) for k, v in job.items()} if isinstance(job, dict) else str(job), "error": repr(e)[:300]})
        print(f"  [{name}] ERROR {job.get('stem') if isinstance(job, dict) else job}: {e!r}"[:200], flush=True)
        return None


if __name__ == "__main__":
    main()
