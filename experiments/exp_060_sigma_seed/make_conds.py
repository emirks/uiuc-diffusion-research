"""exp_060 — cut prefix/suffix conditioning clips for the 12 σ_seed clip-A's.

Exactly the exp_057 make_quads recipe: prefix = first 9 frames
(select='lt(n,9)'), suffix = last 9 frames (select='gte(n,112)' of the 121f
std clip), each re-timestamped to 24 fps and re-encoded libx264 crf 12. The
suffix clip carries 9 frames but is consumed with num_frames=8 at generation
(exp_051 causal-VAE rule). Deterministic, idempotent, login-node / CPU only.
"""

import json
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
STD = REPO_ROOT / "data/processed/transitions_std121"
COND = EXP / "dataset/cond"
FFMPEG = (
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv/lib/"
    "python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def clip_path(stem: str) -> pathlib.Path:
    cls = "_".join(stem.split("_")[:-1])
    return STD / cls / f"{stem}.mp4"


def main() -> None:
    COND.mkdir(parents=True, exist_ok=True)
    sel = json.loads((EXP / "dataset/selection.json").read_text())
    clip_as = sorted({it["clip_a"] for it in sel["items"]})
    for e in clip_as:
        src = clip_path(e)
        for name, vf in [(f"{e}_start9.mp4", "select='lt(n,9)'"),
                         (f"{e}_end9.mp4", "select='gte(n,112)'")]:
            dst = COND / name
            if dst.exists():
                continue
            subprocess.run(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                 "-vf", vf + ",setpts=N/24/TB", "-r", "24",
                 "-c:v", "libx264", "-preset", "slow", "-crf", "12",
                 "-pix_fmt", "yuv420p", str(dst)],
                check=True)
            print(f"[cond] {name}")
    print(f"[done] cond clips for {len(clip_as)} endpoint clips -> {COND}")


if __name__ == "__main__":
    main()
