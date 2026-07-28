"""exp_077 TASK 0d (steps 3-5) — build the IC-LoRA training root from synthetic clips.

Runs under the TRAINER VENV (needs ltx_trainer + the LTX-2 VAE/Gemma). Given a render run_dir
(from render_tuples.py) it:

  3. VAE-encodes video latents for every unique synthetic clip via the trainer's own
     process_videos.py (--resolution-buckets WxHxF). RECONCILES counts — the trainer silently
     skips clips shorter than the bucket, so we assert (latents == clips) and that every mp4 is
     exactly num_frames long.
  4. cond-clean: isolation-encodes the suffix anchor for every clip (all synthetic clips are
     two-sided — both endpoints are pinned) via eval_ladder/encode_conditioning.write_cond_clean.
     Text embeddings: a leak-free NEUTRAL prompt (prompts.py render style + arms.yaml token, NO
     operator names) encoded with the trainer's process_captions.py.
  5. Assembles ONE ic_gen-style root (latents / conditions / reference_latents / masks /
     cond_clean_latents) with per-file symlinks like assemble_roots.py::assemble_generalist, and
     hard-asserts equal source counts across all sub-dirs (the silent-drop guard).

Everything lands under <run_dir>/dataset/.  Run:
    PYTHONPATH=<trainer>/src <venv>/bin/python build_dataset.py --run <render_run_dir>
"""

from __future__ import annotations

import argparse
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


def sh(cmd: list[str], cwd: Path, extra_env: dict | None = None) -> None:
    print("[cmd]", " ".join(str(c) for c in cmd), flush=True)
    env = {**os.environ, **(extra_env or {})}
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="render_tuples run_dir")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    model = cfg["model"]["ltx_checkpoint"]
    gemma = cfg["model"]["gemma"]
    W, H = cfg["inference"]["width"], cfg["inference"]["height"]
    T = cfg["inference"]["num_frames"]
    token = cfg["text"]["token"]
    neutral = f"{cfg['text']['neutral_s1']}. {token}. {cfg['text']['neutral_s2']}."

    run = Path(args.run).resolve()
    videos = run / "videos"
    tuples = json.loads((run / "tuples.json").read_text())["tuples"]
    clip_rows = json.loads((videos / "clips_manifest.json").read_text())
    stems = [r["stem"] for r in clip_rows]
    ds = run / "dataset"
    latents_dir = ds / "latents"
    cc_dir = ds / "cond_clean"
    cond_dir = ds / "conditions"
    print(f"[build] run={run}  clips={len(stems)}  tuples={len(tuples)}  neutral={neutral!r}")

    import torch
    from ltx_trainer.video_utils import get_video_frame_count

    # -- pre-flight: every clip must be exactly num_frames (else process_videos silently skips) --
    bad = [(s, int(get_video_frame_count(videos / f"{s}.mp4"))) for s in stems]
    short = [(s, c) for s, c in bad if c < T]
    if short:
        sys.exit(f"[build] {len(short)} clips shorter than {T} frames -> would be SILENTLY SKIPPED: {short}")
    print(f"[build] all {len(stems)} clips have >= {T} frames (min={min(c for _, c in bad)})")

    # -- step 3: VAE-encode video latents ---------------------------------------------------
    bucket = f"{W}x{H}x{T}"
    sh([VENV_PY, "scripts/process_videos.py", str(videos / "clips_manifest.json"),
        "--resolution-buckets", bucket, "--output-dir", str(latents_dir),
        "--model-path", model, "--video-column", "video", "--device", args.device],
       cwd=TRAINER, extra_env={"PYTHONPATH": str(TRAINER / "src")})
    lat_pts = sorted(latents_dir.rglob("*.pt"))
    assert len(lat_pts) == len(stems), \
        f"latent count {len(lat_pts)} != clip count {len(stems)} — clips were SILENTLY DROPPED"
    for s in stems:
        assert (latents_dir / f"{s}.pt").exists(), f"missing latent for {s}"
    print(f"[build] step3 OK: {len(lat_pts)} latents == {len(stems)} clips")

    # -- step 4a: cond-clean (isolation-encoded suffix anchor; all synthetic clips two-sided) --
    import encode_conditioning as ec
    vae = ec.load_vae(model, device=args.device, dtype=torch.bfloat16)
    cc_report = []
    for s in stems:
        r = ec.write_cond_clean(orig_pt=latents_dir / f"{s}.pt", dst_pt=cc_dir / f"{s}.pt",
                                clip_mp4=videos / f"{s}.mp4", correct_suffix=True,
                                vae=vae, device=args.device)
        cc_report.append(r)
    assert all(r["corrected"] for r in cc_report), "a cond_clean write was not corrected"
    (ds / "cond_clean_report.json").write_text(json.dumps(cc_report, indent=2))
    print(f"[build] step4a OK: {len(cc_report)} cond_clean latents (suffix anchor corrected); "
          f"median suffix_rel_l2={sorted(r['suffix_rel_l2'] for r in cc_report)[len(cc_report)//2]:.4f}")
    del vae
    torch.cuda.empty_cache()

    # -- step 4b: leak-free neutral text embeddings -----------------------------------------
    cap_manifest = videos / "captions_manifest.json"
    cap_manifest.write_text(json.dumps(
        [{"caption": neutral, "video": f"{s}.mp4"} for s in stems], indent=2))
    sh([VENV_PY, "scripts/process_captions.py", str(cap_manifest),
        "--output-dir", str(cond_dir), "--model-path", model, "--text-encoder-path", gemma,
        "--caption-column", "caption", "--media-column", "video", "--device", args.device,
        "--overwrite"],
       cwd=TRAINER, extra_env={"PYTHONPATH": str(TRAINER / "src")})
    cond_pts = sorted(cond_dir.rglob("*.pt"))
    assert len(cond_pts) == len(stems), \
        f"text-embed count {len(cond_pts)} != clip count {len(stems)}"
    for s in stems:
        assert (cond_dir / f"{s}.pt").exists(), f"missing text embed for {s}"
    print(f"[build] step4b OK: {len(cond_pts)} text embeddings (neutral prompt, token={token!r})")

    # -- step 5: assemble the ic_gen-style root ---------------------------------------------
    root = ds / "roots" / "synth_smoke"

    def link(src: Path, dst: Path) -> None:
        if not src.exists():
            raise FileNotFoundError(f"missing source: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    mask_cache: dict = {}
    for t in tuples:
        tgt, ref = t["target_stem"], t["reference_stem"]
        rel = f"synth/{tgt}__ref_{ref}.pt"
        link(latents_dir / f"{tgt}.pt", root / "latents" / rel)
        link(cond_dir / f"{tgt}.pt", root / "conditions" / rel)
        link(latents_dir / f"{ref}.pt", root / "reference_latents" / rel)
        link(cc_dir / f"{tgt}.pt", root / "cond_clean_latents" / rel)
        td = torch.load(latents_dir / f"{tgt}.pt", map_location="cpu", weights_only=True)
        f, h, w = int(td["num_frames"]), int(td["height"]), int(td["width"])
        key = (f, h, w)
        if key not in mask_cache:
            m = torch.zeros(f, h, w)
            m[:2] = 1.0                    # prefix anchor (2 latent frames)
            m[-1] = 1.0                    # suffix anchor (two-sided: both endpoints pinned)
            mask_cache[key] = m
        mdst = root / "masks" / rel
        mdst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mask": mask_cache[key].clone()}, mdst)

    subdirs = ["latents", "conditions", "reference_latents", "masks", "cond_clean_latents"]
    counts = {s: sum(1 for _ in (root / s).rglob("*.pt")) for s in subdirs}
    if len(set(counts.values())) != 1:
        sys.exit(f"[build] source counts disagree {counts} — samples would be SILENTLY DROPPED")
    assert counts["latents"] == len(tuples), f"expected {len(tuples)} samples, got {counts}"
    print(f"[build] step5 OK: root={root}  counts={counts}")
    (ds / "build_report.json").write_text(json.dumps(
        {"run": str(run), "root": str(root), "n_clips": len(stems), "n_samples": len(tuples),
         "counts": counts, "bucket": bucket, "neutral_prompt": neutral, "token": token}, indent=2))
    print(f"[build] DONE -> {root}")


if __name__ == "__main__":
    main()
