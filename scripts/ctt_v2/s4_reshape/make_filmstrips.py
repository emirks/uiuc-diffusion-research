"""ctt_v2 S4 — select the refVFX I2V-LoRA subset and extract first-9-frame filmstrips.

WHY FILMSTRIPS AND NOT A TEXT REWRITE
-------------------------------------
The refVFX prompts DO parse into a clean, leak-free start-scene sentence for 96.3% of rows
(see extract_s1.py). But measured against the real corpus the rewritten captions are far too
terse, and the gap is nearly disjoint:

    S0 corpus S1 words:  p10 25 · p50 34 · p90 40
    S4 rewritten S1:     p10  5 · p50  8 · p90 13     (98% under 16 words)

Caption length alone would therefore almost perfectly identify the stratum -- a textbook
routing shortcut, and the exact failure the campaign's anti-routing rule exists to catch. So
S4 captions are WRITTEN FROM THE FRAMES, to the same spec as the corpus captions. The text
rewrite is retained as a verified fallback and as a cross-check on the vision captions.

THE FIRST-9 CAVEAT (owner, 2026-07-25)
--------------------------------------
The 9 frames are the conditioning prefix, and for a fast effect the transition has ALREADY
BEGUN inside them. The caption must describe the start state only. That cannot be enforced by
a regex, so it is an explicit instruction to the captioning agent plus a post-hoc verification
pass -- the filmstrip is laid out left-to-right with frame indices so the agent can SEE which
frames have drifted and describe frame 0.

Selection: stratified across triggers, with a held-out trigger split that is EVAL-ONLY and
never trained on.

Usage:
    python scripts/ctt_v2/s4_reshape/make_filmstrips.py --n 2000 --holdout 5
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")

RAW = LAB / "diffusion-research/data/raw/refvfx"
INDEX = RAW / "_viewer_index/lora.jsonl.gz"
SHARDS = RAW / "data/I2V_LoRA"
OUT = REPO_ROOT / "data/processed/s4_refvfx"
STRIPS = OUT / "filmstrips"

PREFIX_FRAMES = 9
#: 3x3 rather than 9x1. A 9-wide strip of 832x464 tiles is 3658x232 -- so short that a vision
#: model loses facial and garment detail, which is most of what the caption has to describe.
#: 3x3 at 320px tiles is ~1730x975, reading order left-to-right then top-to-bottom.
STRIP_H = 320
TILE = "3x3"


def shard_path(sh: int) -> Path:
    return SHARDS / f"shard-{sh:05d}.tar"


def read_range(path: Path, offset: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def load_rows() -> list[dict]:
    rows = [json.loads(x) for x in gzip.open(INDEX, "rt")]
    return [r for r in rows if r["pr"].strip() and r["et"].strip() and "out" in r["m"]]


def select(rows: list[dict], n: int, holdout: int, seed: int = 42):
    """Stratified pick across triggers, with `holdout` whole triggers reserved for eval."""
    by_trig: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_trig[r["et"]].append(r)
    trigs = sorted(by_trig)
    rng = random.Random(seed)
    held = set(rng.sample(trigs, holdout))
    train_trigs = [t for t in trigs if t not in held]

    per = n // len(train_trigs)
    picked: list[dict] = []
    for t in train_trigs:
        pool = sorted(by_trig[t], key=lambda r: r["k"])
        rng.shuffle(pool)
        picked += pool[:per]
    # top up deterministically to hit n exactly
    if len(picked) < n:
        rest = [r for t in train_trigs for r in sorted(by_trig[t], key=lambda r: r["k"])
                if r not in picked]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]
    return picked, sorted(held), train_trigs


def make_strip(mp4_bytes: bytes, dst: Path) -> bool:
    """Write a 3x3 montage of the first 9 frames in reading order (frame 0 top-left).

    No frame-index overlay: this ffmpeg build has no `drawtext` (built without libfreetype).
    Reading order carries the ordering instead, and the captioning instruction states it.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(mp4_bytes)
        tmp.flush()
        vf = (f"select='lt(n,{PREFIX_FRAMES})',scale=-1:{STRIP_H},"
              f"tile={TILE}:margin=4:padding=4:color=white")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp.name,
               "-vf", vf, "-frames:v", "1", "-q:v", "3", str(dst)]
        return subprocess.run(cmd, capture_output=True).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--holdout", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap extraction (pilot runs)")
    args = ap.parse_args()

    rows = load_rows()
    picked, held, train_trigs = select(rows, args.n, args.holdout)
    OUT.mkdir(parents=True, exist_ok=True)
    STRIPS.mkdir(parents=True, exist_ok=True)

    print(f"[s4] pool={len(rows)}  triggers={len(train_trigs) + len(held)}  "
          f"held_out={len(held)}  selected={len(picked)}")
    print(f"[s4] held-out triggers (EVAL ONLY, never trained): {held}")

    todo = picked[: args.limit] if args.limit else picked
    manifest, failed = [], 0
    for i, r in enumerate(todo, 1):
        dst = STRIPS / f"{r['k']}.jpg"
        if not dst.exists():
            off, size, _ = r["m"]["out"]
            data = read_range(shard_path(r["sh"]), off, size)
            if not make_strip(data, dst):
                failed += 1
                continue
        manifest.append({"k": r["k"], "shard": r["sh"], "effect": r["et"],
                         "out": r["m"]["out"], "strip": str(dst.relative_to(REPO_ROOT))})
        if i % 100 == 0 or i == len(todo):
            print(f"[s4] filmstrips {i}/{len(todo)} (failed {failed})", flush=True)

    (OUT / "selection.json").write_text(json.dumps({
        "n_requested": args.n, "held_out_triggers": held,
        "train_triggers": train_trigs, "samples": manifest,
    }, indent=1))
    print(f"[s4] wrote {len(manifest)} filmstrips, {failed} failed -> {OUT / 'selection.json'}")


if __name__ == "__main__":
    main()
