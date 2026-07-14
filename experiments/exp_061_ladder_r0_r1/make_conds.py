"""exp_061 — cut prefix/suffix conditioning clips for the R1 arm items.

Exactly the exp_057/exp_060 recipe: prefix = first 9 frames (select='lt(n,9)'),
suffix = last 9 frames (select='gte(n,112)' of the 121f std clip), each
re-timestamped to 24 fps and re-encoded libx264 crf 12. The suffix clip
carries 9 frames but is consumed with num_frames=8 at generation (exp_051
causal-VAE rule). R0 needs no conditions. Deterministic, idempotent,
login-node / CPU only.
"""

import json
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
COND = EXP / "dataset/cond"
FFMPEG = (
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv/lib/"
    "python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def main() -> None:
    COND.mkdir(parents=True, exist_ok=True)
    sel = json.loads((EXP / "dataset/selection.json").read_text())
    stems = [(it["clip"], it["clip_rel"]) for it in sel["items"]]
    assert len({s for s, _ in stems}) == len(stems), "clip stems not unique"
    for stem, rel in stems:
        src = REPO_ROOT / rel
        for name, vf in [(f"{stem}_start9.mp4", "select='lt(n,9)'"),
                         (f"{stem}_end9.mp4", "select='gte(n,112)'")]:
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
    print(f"[done] cond clips for {len(stems)} item clips -> {COND}")


if __name__ == "__main__":
    main()
