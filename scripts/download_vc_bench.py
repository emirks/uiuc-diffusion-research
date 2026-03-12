#!/usr/bin/env python3
"""
Download the full VC-Bench dataset from Hugging Face.

Source : Kevinson-lzp/VC-Bench
Target : /workspace/diffusion-research/data/raw/vc-bench-hf/

Content:
  - 1,261 MP4 videos (original full videos, not clips)
  - VC-Bench.csv  (filename, resolution, length, fps, num_scenes,
                   scene_start_frames, category, aesthetic_score, caption)
  - README.md / .gitattributes

Estimated total size: ~38 GB

Resume-safe: snapshot_download caches progress in ~/.cache/huggingface/
and skips already-downloaded files automatically on retry.

If you see "rate limit your IP" from Hugging Face:
  - Log in at https://huggingface.co and create a token at
    https://huggingface.co/settings/tokens (read access is enough).
  - Run with:  HF_TOKEN=your_token python scripts/download_vc_bench.py
  - Or once:   huggingface-cli login   (then this script will use the saved token)

Run with:
    source /workspace/miniforge3/etc/profile.d/conda.sh
    conda activate /workspace/envs/diff
    HF_TOKEN=your_token python scripts/download_vc_bench.py   # if rate limited
    python scripts/download_vc_bench.py
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID   = "Kevinson-lzp/VC-Bench"
LOCAL_DIR = "/workspace/diffusion-research/data/raw/vc-bench-hf"

# Use token to avoid IP rate limiting (anonymous users get limited)
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Downloading {REPO_ID} → {LOCAL_DIR}")
print("1,261 MP4s + CSV. Estimated ~38 GB total.")
print("Resume-safe: re-run if interrupted.")
if not hf_token:
    print("Tip: set HF_TOKEN if you get rate limited (see script docstring).")
print()

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,   # real files, not symlinks into HF cache
    resume_download=True,
    token=hf_token,
)

# Post-download summary
mp4s = sorted(Path(LOCAL_DIR).rglob("*.mp4"))
total_bytes = sum(f.stat().st_size for f in mp4s)
print(f"\n=== Download complete ===")
print(f"  MP4 files   : {len(mp4s)}")
print(f"  Total size  : {total_bytes / 1e9:.1f} GB")
print(f"  Directory   : {LOCAL_DIR}")
