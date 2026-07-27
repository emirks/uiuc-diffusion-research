#!/usr/bin/env python
"""COLLECT: HumanVid manifest -> _work/candidates.jsonl for the screening cascade.

One candidate per downloaded clip (one center window each — volume is ample, and
multiple windows of one scene would only feed the dedup stage). License is recorded
per clip: Pexels, ML-use flag noted, owner-cleared 2026-07-27.
"""
import gzip
import json
import os

from hv_common import F, HV_MANIFEST, RAW_HV, WORK

os.makedirs(WORK, exist_ok=True)
OUT = os.path.join(WORK, "candidates.jsonl")

LICENSE = ("Pexels (via HumanVid URL lists). Pexels ToS ML-use restriction documented in "
           "notes/dataset/humanvid_real.md; owner cleared use 2026-07-27.")

n_in = n_missing = n_short = 0
rows = []
for line in gzip.open(HV_MANIFEST, "rt"):
    c = json.loads(line)
    n_in += 1
    sub = "horizontal" if c["split"] == "horizontal" else "vertical"
    path = os.path.join(RAW_HV, "videos", sub, os.path.basename(c["url"]))
    if not os.path.exists(path):
        n_missing += 1
        continue
    if int(c.get("frames") or 0) < F:
        n_short += 1
        continue
    rows.append({
        "orig_ref": f"humanvid/{c['id']}",
        "orig_id": c["id"],
        "source": "humanvid",
        "path": path,
        "url": c["url"],
        "license": LICENSE,
        "split": c["split"],
        "manifest_fps": c.get("fps"),
        "manifest_frames": c.get("frames"),
        "manifest_wh": [c.get("width"), c.get("height")],
    })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"[collect] manifest={n_in} missing_file={n_missing} too_short={n_short} "
      f"candidates={len(rows)} -> {OUT}")
