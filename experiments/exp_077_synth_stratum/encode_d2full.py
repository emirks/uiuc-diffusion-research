"""exp_077 D2-FULL STAGE 3 (encode) — VAE-encode latents + cond_clean for a SHARD of D2 clips.

Runs under the TRAINER VENV on an L40S (keeps the H100s free). JOB-ARRAY SHARDABLE + idempotent
(both steps skip work already on disk), adapted from the proven `encode_d1.py`.

  3.  process_videos.py --resolution-buckets WxHxF -> dataset/latents/<stem>.pt   (ALL clips)
  4a. ec.write_cond_clean(...)                    -> dataset/cond_clean/<stem>.pt (TARGET clips only)

cond_clean is the isolation-encoded suffix anchor of the clip being denoised, so only the TARGET of
each tuple needs one — the reference clip is consumed as `reference_latents`. (encode_d1 wrote it
for all 6,144 clips; restricting it to the 3,072 targets is exactly the set `assemble` links, and
halves this stage's VAE work. The assemble step hard-asserts one cond_clean per target.)
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
    subprocess.run([str(c) for c in cmd], cwd=str(cwd),
                   env={**os.environ, **(extra_env or {})}, check=True)


def run_root(cfg) -> Path:
    return REPO_ROOT / cfg["outputs"]["dir"] / cfg["outputs"]["subdir"]


def all_stems(run: Path) -> tuple[list[str], list[str]]:
    """(every clip stem, target-only stems) aggregated over the render shards, deterministic."""
    stems: set[str] = set()
    for f in sorted((run / "dataset_manifests").glob("clips_shard*.json")):
        for row in json.loads(f.read_text()):
            stems.add(row["stem"])
    tgts: set[str] = set()
    for f in sorted((run / "meta").glob("tuples_shard*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                tgts.add(json.loads(line)["target_stem"])
    return sorted(stems), sorted(tgts)


def main() -> None:
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    device = os.environ.get("DEVICE", "cuda")
    cfg = yaml.safe_load((HERE / "config_d2full.yaml").read_text())
    model = cfg["model"]["ltx_checkpoint"]
    W, H = cfg["inference"]["width"], cfg["inference"]["height"]
    T = cfg["inference"]["num_frames"]

    run = run_root(cfg)
    videos = run / "videos"
    ds = run / "dataset"
    latents_dir, cc_dir = ds / "latents", ds / "cond_clean"
    latents_dir.mkdir(parents=True, exist_ok=True)
    cc_dir.mkdir(parents=True, exist_ok=True)

    stems_all, tgts_all = all_stems(run)
    stems = [s for i, s in enumerate(stems_all) if i % nshards == shard]
    tgt_set = set(tgts_all)
    tgts = [s for s in stems if s in tgt_set]
    print(f"[encode] shard {shard}/{nshards}: {len(stems)} of {len(stems_all)} clips "
          f"({len(tgts)} targets of {len(tgts_all)})  bucket={W}x{H}x{T}", flush=True)
    if not stems:
        sys.exit(f"[encode] no clips found under {run} — did the render stage finish?")

    import torch
    from ltx_trainer.video_utils import get_video_frame_count

    # pre-flight: every clip must be exactly T frames (else process_videos SILENTLY skips it)
    counts = [(s, int(get_video_frame_count(videos / f"{s}.mp4"))) for s in stems]
    short = [(s, c) for s, c in counts if c < T]
    if short:
        sys.exit(f"[encode] {len(short)} clips < {T} frames -> SILENT SKIP risk: {short[:5]}")
    print(f"[encode] pre-flight OK: all {len(stems)} clips >= {T} frames "
          f"(min={min(c for _, c in counts)})", flush=True)

    # -- step 3: VAE-encode video latents (only the missing ones) --
    todo = [s for s in stems if not (latents_dir / f"{s}.pt").exists()]
    print(f"[encode] step3: {len(todo)} latents to encode ({len(stems)-len(todo)} present)", flush=True)
    if todo:
        man = videos / f"clips_encode_shard{shard:02d}.json"
        man.write_text(json.dumps([{"video": f"{s}.mp4"} for s in todo], indent=1))
        sh([VENV_PY, "scripts/process_videos.py", str(man),
            "--resolution-buckets", f"{W}x{H}x{T}", "--output-dir", str(latents_dir),
            "--model-path", model, "--video-column", "video", "--device", device],
           cwd=TRAINER, extra_env={"PYTHONPATH": str(TRAINER / "src")})
    miss = [s for s in stems if not (latents_dir / f"{s}.pt").exists()]
    if miss:
        sys.exit(f"[encode] {len(miss)} latents missing after encode (SILENT DROP): {miss[:5]}")
    print(f"[encode] step3 OK: {len(stems)} latents present for shard {shard}", flush=True)

    # -- step 4a: cond_clean for the TARGET clips of this shard --
    import encode_conditioning as ec
    todo_cc = [s for s in tgts if not (cc_dir / f"{s}.pt").exists()]
    print(f"[encode] step4a: {len(todo_cc)} cond_clean to write ({len(tgts)-len(todo_cc)} present)",
          flush=True)
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
    miss_cc = [s for s in tgts if not (cc_dir / f"{s}.pt").exists()]
    if miss_cc:
        sys.exit(f"[encode] cond_clean missing: {miss_cc[:5]}")
    print(f"[encode] shard {shard} DONE: {len(stems)} latents + {len(tgts)} cond_clean")


if __name__ == "__main__":
    main()
