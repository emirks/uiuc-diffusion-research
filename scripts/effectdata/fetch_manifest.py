#!/usr/bin/env python3
"""Enumerate all Videos/*.zip in ysy31415926/EffectData with size + sha256.
Metadata-only (HF API), no file bytes, no xet. Writes videos_manifest.json into
the dataset dir (data/raw/effectdata/)."""
from pathlib import Path
import json
from huggingface_hub import HfApi

REPO_ID = "ysy31415926/EffectData"
DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "effectdata"

api = HfApi()
items = []
for it in api.list_repo_tree(REPO_ID, path_in_repo="Videos", repo_type="dataset",
                             recursive=True, expand=True):
    if getattr(it, "size", None) is None:      # skip folders
        continue
    if not it.path.lower().endswith(".zip"):
        continue
    lfs = getattr(it, "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    items.append({"path": it.path, "size": it.size, "sha256": sha})

items.sort(key=lambda x: x["path"])
total = sum(x["size"] for x in items)
out = DATA / "videos_manifest.json"
json.dump({"repo": REPO_ID, "n": len(items), "total_bytes": total, "files": items},
          open(out, "w"), indent=1)
print(f"[ok] {out}: {len(items)} files, {total/1e9:.1f} GB, "
      f"{sum(1 for x in items if x['sha256'])} with sha256")
