"""exp_058 — cut the prefix condition videos for the training-time validation
samples (design.md §5): water_element_0 (one-sided in-class, prefix-only) and
hero_flight_0 (held-out class, prefix-only). The two-sided validation sample
reuses exp_056's earth_wave cuts. Login-node safe, idempotent.
"""

import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STD = REPO_ROOT / "data/processed/transitions_std121"
OUT = pathlib.Path(__file__).parent / "dataset"
FFMPEG = (
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv/lib/"
    "python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)

CUTS = [  # (class, stem) -> <stem>_start9.mp4
    ("water_element", "water_element_0"),
    ("hero_flight", "hero_flight_0"),
]

OUT.mkdir(exist_ok=True)
for cls, stem in CUTS:
    src = STD / cls / f"{stem}.mp4"
    dst = OUT / f"cond_{stem}_start9.mp4"
    if dst.exists():
        print(f"[skip] {dst.name}")
        continue
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", "select='lt(n,9)',setpts=N/24/TB", "-r", "24",
         "-c:v", "libx264", "-preset", "slow", "-crf", "12",
         "-pix_fmt", "yuv420p", str(dst)],
        check=True)
    print(f"[done] {dst.name}")
