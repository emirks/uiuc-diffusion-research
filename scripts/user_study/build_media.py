#!/usr/bin/env python3
"""Build the opaque media pack for the SEGUE human study.

Reads misc/2026-09-22_user_study/pairs.json, re-encodes every distinct source
clip and extracts every distinct given-frame still into
misc/2026-09-22_user_study/media/ under content-addressed opaque names
(sha1 of the source path). Rerunnable: files already present with a matching
manifest entry are kept.

Run with the aarch64 media python (imageio_ffmpeg binary; no system ffmpeg):
    $LAB/envs-aarch64/ltx2/bin/python scripts/user_study/build_media.py

Constraints (login node): strictly sequential, -threads 2 (pids cap).
"""
import hashlib
import json
import os
import subprocess
import sys

import imageio_ffmpeg as iio

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUDY = os.path.join(REPO, "misc", "2026-09-22_user_study")
MEDIA = os.path.join(STUDY, "media")
PAIRS = os.path.join(STUDY, "pairs.json")
ATTENTION = os.path.join(STUDY, "attention.json")
EXAMPLES = os.path.join(STUDY, "examples.json")
MANIFEST = os.path.join(STUDY, "media_manifest.json")
FFMPEG = iio.get_ffmpeg_exe()


def mh(path):
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def collect_sources(pairs, attention=(), examples=()):
    """source_path -> {kind, ext, frame}. Videos are clips; stills are one frame.

    ``attention`` (attention.json entries, if any) adds the ground-truth REAL
    clips; the COPY, the shown reference and the stills are already covered by
    ``pairs`` (every picked row is also a pair), so setdefault dedupes them.
    """
    src = {}
    for p in pairs:
        for k in ("segue_clip", "opponent_clip", "reference_clip"):
            src.setdefault(p[k], {"kind": "clip", "ext": "mp4", "frame": None})
        src.setdefault(p["start_still"], {"kind": "still", "ext": "jpg", "frame": 0})
        if p["end_still"]:
            src.setdefault(p["end_still"], {"kind": "still", "ext": "jpg", "frame": 8})
    for e in attention:
        for k in ("real_clip", "copy_clip", "reference_clip"):
            src.setdefault(e[k], {"kind": "clip", "ext": "mp4", "frame": None})
        src.setdefault(e["start_still"], {"kind": "still", "ext": "jpg", "frame": 0})
        if e["end_still"]:
            src.setdefault(e["end_still"], {"kind": "still", "ext": "jpg", "frame": 8})
    for e in examples:  # worked examples (examples.json); usually study media already covered above
        for k in ("a_clip", "b_clip", "reference_clip"):
            src.setdefault(e[k], {"kind": "clip", "ext": "mp4", "frame": None})
        src.setdefault(e["start_still"], {"kind": "still", "ext": "jpg", "frame": 0})
        if e.get("end_still"):
            src.setdefault(e["end_still"], {"kind": "still", "ext": "jpg", "frame": 8})
    return src


# External baselines are stored duration-matched (49f @ 9.719 fps, 33f @ 6.5455 fps). The study plays them at the
# frame rate the authors' own inference code writes (owner, 2026-09-25 evening): every frame is kept, only the
# timestamps change. Keyed by the store arm directory; anything else keeps its stored rate (SEGUE / Base LTX-2 24 fps).
NATIVE_FPS = {
    "store/gens/011_vap/": 16.0,          # Video-As-Prompt infer/cog_vap.py: export_to_video(..., fps=16)
    "store/gens/012_vfxmaster/": 8.0,     # VFXMaster repo/code/inference.py: --fps default 8
    "store/gens/003_refvfx/": 15.0,       # refVFX_inference/infer_refvfx.py: --fps default 15
}


def native_fps(src):
    for prefix, fps in NATIVE_FPS.items():
        if prefix in src:
            return fps
    return None


def encode_clip(src, out, fps=None):
    vf = ["-vf", "setpts=N/(%g*TB)" % fps, "-r", "%g" % fps] if fps else []
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-map", "0:v:0"] + vf + ["-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "26", "-preset", "medium", "-an",
         "-movflags", "+faststart", "-threads", "2", out],
        check=True)


def extract_still(src, frame, out):
    # ffmpeg mjpeg -q:v 3 approximates JPEG quality ~90 (no 0-100 knob exists).
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-vf", "select=eq(n\\,%d)" % frame, "-frames:v", "1",
         "-q:v", "3", "-threads", "2", out],
        check=True)


def probe_video(path):
    # single ffmpeg spawn; libx264 preserves timing so fps = frames/secs.
    n, secs = iio.count_frames_and_secs(path)
    fps = round(n / secs, 3) if secs else None
    return n, fps, round(secs, 3)


def main():
    if not os.path.exists(PAIRS):
        sys.exit(f"missing {PAIRS}; run select_pairs.py first")
    os.makedirs(MEDIA, exist_ok=True)
    with open(PAIRS) as f:
        pairs = json.load(f)
    attention = []
    if os.path.exists(ATTENTION):
        with open(ATTENTION) as f:
            attention = json.load(f)
        print(f"[media] read {len(attention)} attention entries (REAL clips added)")
    examples = []
    if os.path.exists(EXAMPLES):
        with open(EXAMPLES) as f:
            examples = json.load(f)
        print(f"[media] read {len(examples)} example entries")
    sources = collect_sources(pairs, attention, examples)

    old = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            old = json.load(f)

    manifest = {}
    encoded = skipped = 0
    for i, (src, info) in enumerate(sorted(sources.items()), 1):
        h = mh(src)
        out = os.path.join(MEDIA, f"{h}.{info['ext']}")
        if h in manifest:
            sys.exit(f"hash collision {h}: {manifest[h]['source']} vs {src}")
        # rerun skip: file present and previous manifest entry matches this source (and its native-rate override)
        nfps = native_fps(src) if info["kind"] == "clip" else None
        if os.path.exists(out) and h in old and old[h].get("source") == src and old[h].get("native_fps") == nfps:
            manifest[h] = old[h]
            skipped += 1
            continue
        if info["kind"] == "clip":
            encode_clip(src, out, nfps)
            frames, fps, dur = probe_video(out)
        else:
            extract_still(src, info["frame"], out)
            frames, fps, dur = 1, None, None
        manifest[h] = {"source": src, "kind": info["kind"], "frames": frames,
                       "fps": fps, "duration_s": dur, "bytes": os.path.getsize(out), "native_fps": nfps}
        encoded += 1
        if i % 25 == 0 or i == len(sources):
            print(f"[media] {i}/{len(sources)} (encoded={encoded} skipped={skipped})")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)

    total = sum(e["bytes"] for e in manifest.values())
    clips = sum(1 for e in manifest.values() if e["kind"] == "clip")
    stills = sum(1 for e in manifest.values() if e["kind"] == "still")
    print(f"[media] files={len(manifest)} (clips={clips} stills={stills}) "
          f"encoded={encoded} skipped={skipped}")
    print(f"[media] total size = {total/1e6:.1f} MB")
    print(f"[media] wrote {MANIFEST}")


if __name__ == "__main__":
    main()
