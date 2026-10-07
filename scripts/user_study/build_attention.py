#!/usr/bin/env python3
"""Build the attention-check pool for the SEGUE human study.

An attention pair is a per-session rater-quality probe (never counted in win
rates). For a picked row whose content is "same" and whose endpoint has a real
ground-truth transition clip in the std121 tree, we build:

  REAL = data/processed/transitions_std121/<endpoint_class>/<endpoint>.mp4
         the true transition on those exact endpoints -- it opens on the given
         start frame and (for two-sided rows) ends on the given end frame, and
         shows the endpoint's OWN scenes, not the reference's.
  COPY = the row's reference clip itself.

Only two questions are scored, both expecting REAL:
  endpoint        -> REAL  (the copy does not show the given frames)
  disentanglement -> REAL  (the copy IS the reference's own content)
transition and quality are NOT scored (both clips show the same kind of effect).

Reads misc/2026-09-22_user_study/rows.json, writes attention.json alongside.
Never served directly: serve.py draws from it and re-encodes the REAL clips via
build_media.py under opaque hashed names. The COPY, the shown reference and the
given-frame stills are already in the media pack because every picked row is
also a pair.

    $LAB/envs-aarch64/ltx2/bin/python scripts/user_study/build_attention.py
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUDY = os.path.join(REPO, "misc", "2026-09-22_user_study")
ROWS = os.path.join(STUDY, "rows.json")
OUT = os.path.join(STUDY, "attention.json")
STD121 = os.path.join(REPO, "data", "processed", "transitions_std121")


def real_clip_for(endpoint):
    """Locate the ground-truth transition clip for an endpoint.

    Basenames are globally unique across the std121 tree, so a glob over the
    class dirs returns exactly one match; ed.* endpoints resolve under their
    own ed.* class dir automatically. Returns the absolute path or None.
    """
    matches = glob.glob(os.path.join(STD121, "*", glob.escape(endpoint) + ".mp4"))
    if len(matches) == 1:
        return matches[0]
    return None  # 0 = missing; >1 would violate basename uniqueness (never seen)


def main():
    if not os.path.exists(ROWS):
        sys.exit(f"missing {ROWS}; run select_pairs.py first")
    with open(ROWS) as f:
        rows = json.load(f)

    pool = []
    per_task = {"one": 0, "two": 0}
    skipped = []
    for r in rows:
        if r.get("content") != "same":
            continue
        ep = r["endpoint"]
        real = real_clip_for(ep)
        if not real:
            skipped.append((r["task"], ep, "no std121 clip"))
            continue
        copy = r["reference_clip"]  # absolute path, already in the media pack
        if os.path.abspath(real) == os.path.abspath(copy):
            skipped.append((r["task"], ep, "real == copy (degenerate)"))
            continue
        pool.append({
            "task": r["task"], "endpoint": ep, "reference": r["reference"],
            "real_clip": real, "copy_clip": copy,
            "reference_clip": r["reference_clip"],
            "start_still": r["start_still"], "end_still": r["end_still"],
        })
        per_task[r["task"]] += 1

    # deterministic order, then opaque a-ids (the a... id space; ids are opaque)
    pool.sort(key=lambda e: (e["task"], e["endpoint"], e["reference"]))
    width = max(3, len(str(max(len(pool) - 1, 0))))
    for i, e in enumerate(pool):
        pool[i] = {"a": "a" + str(i).zfill(width), **e}

    with open(OUT, "w") as f:
        json.dump(pool, f, indent=1)

    print(f"[attention] pool size: task two = {per_task['two']}, "
          f"task one = {per_task['one']}, total = {len(pool)}")
    if skipped:
        print(f"[attention] skipped {len(skipped)} candidate row(s):")
        for t, ep, why in skipped:
            print(f"[attention]   task {t}  {ep}  ({why})")
    print(f"[attention] wrote {OUT}")


if __name__ == "__main__":
    main()
