"""exp_062 — cut prefix/suffix conditioning clips for the R2/R3 generation targets
(held-in r2_items + unseen test_items), exp_057/060/061 recipe verbatim: prefix =
first 9 frames (select='lt(n,9)'), suffix = last 9 frames (select='gte(n,112)' of the
121f std clip), re-timestamped to 24 fps, libx264 crf 12. Suffix carries 9 frames,
consumed with num_frames=8 at generation (causal-VAE rule). CPU / login-node only,
idempotent. Writes dataset/cond/<clip>_{start9,end9}.mp4 for all R2/R3 targets.
"""
import json, pathlib, shutil, subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
STD = REPO / "data/processed/transitions_std121"
GRID = REPO / "docs/eval_ladder/ladder_items_v1.json"
COND = EXP / "dataset/cond"
FFMPEG = shutil.which("ffmpeg") or str(pathlib.Path.home() / ".local/bin/ffmpeg")


def cut(src: pathlib.Path, dst: pathlib.Path, vf: str) -> None:
    if dst.exists():
        return
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", vf + ",setpts=N/24/TB", "-r", "24",
         "-c:v", "libx264", "-preset", "slow", "-crf", "12",
         "-pix_fmt", "yuv420p", str(dst)],
        check=True)


def main() -> None:
    COND.mkdir(parents=True, exist_ok=True)
    grid = json.loads(GRID.read_text())["classes"]
    made = 0
    for cls, g in grid.items():
        targets = list(g["r2_items"]) + list(g["test_items"])  # held-in + unseen
        for clip in targets:
            src = STD / cls / f"{clip}.mp4"
            assert src.exists(), f"missing clip: {src}"
            cut(src, COND / f"{clip}_start9.mp4", "select='lt(n,9)'")
            cut(src, COND / f"{clip}_end9.mp4", "select='gte(n,112)'")
            made += 1
    n = len(list(COND.glob("*.mp4")))
    print(f"[done] {made} target clips -> {n} cond files in {COND}")


if __name__ == "__main__":
    main()
