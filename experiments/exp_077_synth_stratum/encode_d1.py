"""exp_077 D1 STAGE 2 (encode) — VAE-encode latents + cond-clean for a SHARD of the D1 clips.

Runs under the TRAINER VENV on an L40S (keeps the H100s free). JOB-ARRAY SHARDABLE + idempotent.
Adapts the proven `build_dataset.py` steps 3 (process_videos VAE latents) and 4a (cond_clean
isolation-encoded suffix anchor). Text embeddings + assembly + mix are done ONCE in the fan-in
`assemble_d1.py` (all D1 clips share ONE neutral prompt, so its embedding is encoded once).

  SHARD/NSHARDS via env. Each shard aggregates ALL render clip manifests (deterministic sort by
  stem), takes its modulo slice, and:
    3.  process_videos.py --resolution-buckets WxHxF -> dataset/latents/<stem>.pt   (skips existing)
    4a. ec.write_cond_clean(...) -> dataset/cond_clean/<stem>.pt                     (skips existing)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))       # encode_conditioning

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
TRAINER = LAB / "LTX-2-cond-bleed-fix/packages/ltx-trainer"
VENV_PY = LAB / "LTX-2-official/.venv/bin/python"


def sh(cmd, cwd, extra_env=None):
    print("[cmd]", " ".join(str(c) for c in cmd), flush=True)
    env = {**os.environ, **(extra_env or {})}
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, check=True)


def all_clip_stems(run: Path) -> list[str]:
    """Aggregate every render shard's clip manifest -> deterministic global stem list."""
    stems: set[str] = set()
    for f in sorted((run / "dataset_manifests").glob("clips_shard*.json")):
        for row in json.loads(f.read_text()):
            stems.add(row["stem"])
    return sorted(stems)


def main() -> None:
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    device = os.environ.get("DEVICE", "cuda")
    cfg = yaml.safe_load((HERE / "config_d1.yaml").read_text())
    model = cfg["model"]["ltx_checkpoint"]
    W, H = cfg["inference"]["width"], cfg["inference"]["height"]
    T = cfg["inference"]["num_frames"]

    run = REPO_ROOT / cfg["outputs"]["dir"] / "d1"
    videos = run / "videos"
    ds = run / "dataset"
    latents_dir = ds / "latents"
    cc_dir = ds / "cond_clean"
    latents_dir.mkdir(parents=True, exist_ok=True)
    cc_dir.mkdir(parents=True, exist_ok=True)

    stems_all = all_clip_stems(run)
    stems = [s for i, s in enumerate(stems_all) if i % nshards == shard]
    print(f"[encode] shard {shard}/{nshards}: {len(stems)} of {len(stems_all)} clips  bucket={W}x{H}x{T}")

    import torch
    from ltx_trainer.video_utils import get_video_frame_count

    # pre-flight: every clip must be exactly T frames (else process_videos SILENTLY skips it)
    bad = [(s, int(get_video_frame_count(videos / f"{s}.mp4"))) for s in stems]
    short = [(s, c) for s, c in bad if c < T]
    if short:
        sys.exit(f"[encode] {len(short)} clips < {T} frames -> would be SILENTLY SKIPPED: {short[:5]}")

    # -- step 3: VAE-encode video latents (only clips whose latent is missing) --
    todo = [s for s in stems if not (latents_dir / f"{s}.pt").exists()]
    print(f"[encode] step3: {len(todo)} latents to encode ({len(stems) - len(todo)} already present)")
    if todo:
        man = videos / f"clips_encode_shard{shard}.json"
        man.write_text(json.dumps([{"video": f"{s}.mp4"} for s in todo], indent=2))
        sh([VENV_PY, "scripts/process_videos.py", str(man),
            "--resolution-buckets", f"{W}x{H}x{T}", "--output-dir", str(latents_dir),
            "--model-path", model, "--video-column", "video", "--device", device],
           cwd=TRAINER, extra_env={"PYTHONPATH": str(TRAINER / "src")})
    for s in stems:
        if not (latents_dir / f"{s}.pt").exists():
            sys.exit(f"[encode] latent missing after encode: {s} (SILENT DROP)")
    print(f"[encode] step3 OK: {len(stems)} latents present for shard {shard}")

    # -- step 4a: cond-clean (isolation-encoded suffix anchor; all D1 clips are two-sided) --
    import encode_conditioning as ec
    todo_cc = [s for s in stems if not (cc_dir / f"{s}.pt").exists()]
    print(f"[encode] step4a: {len(todo_cc)} cond_clean to write ({len(stems) - len(todo_cc)} present)")
    if todo_cc:
        vae = ec.load_vae(model, device=device, dtype=torch.bfloat16)
        rels = []
        for i, s in enumerate(todo_cc):
            r = ec.write_cond_clean(orig_pt=latents_dir / f"{s}.pt", dst_pt=cc_dir / f"{s}.pt",
                                    clip_mp4=videos / f"{s}.mp4", correct_suffix=True,
                                    vae=vae, device=device)
            if not r["corrected"]:
                sys.exit(f"[encode] cond_clean not corrected for {s}")
            rels.append(r["suffix_rel_l2"])
            if (i + 1) % 200 == 0:
                print(f"[encode] cond_clean {i+1}/{len(todo_cc)}", flush=True)
        del vae
        torch.cuda.empty_cache()
        med = sorted(rels)[len(rels) // 2] if rels else float("nan")
        print(f"[encode] step4a OK: {len(todo_cc)} cond_clean (median suffix_rel_l2={med:.4f})")
    for s in stems:
        if not (cc_dir / f"{s}.pt").exists():
            sys.exit(f"[encode] cond_clean missing after write: {s}")
    print(f"[encode] shard {shard} DONE: {len(stems)} latents + cond_clean present")


if __name__ == "__main__":
    main()
