#!/usr/bin/env python
"""CTT v2 / S1 -- the generator. Consumes S1_GRID.json rows; one row = one clip.

    row  x  its pinned seed  ==  exactly one video at row["out_path"]

Nothing is decided here. The grid already carries the specialist, the endpoints, the
sidedness, the prompt and the output path; this script renders them.

Why not eval_ladder/run_gen.py: that script is keyed to `registry.jsonl` (the EVAL ladder's
rows, whose endpoints are corpus clips) and loads one adapter per process. S1 is 11 adapters
over pool endpoints, so this one:
  * builds conditioning windows for pool clips on the fly (encode_conditioning.cut_windows),
  * caches every prompt embedding and conditioning encode ONCE for the whole task, then
  * loads the 19B transformer ONCE, wraps it in PEFT ONCE, and swaps only the LoRA state dict
    between specialists -- all 11 share rank 32 / alpha 32 / the `attn` target set.

Preemption: outputs are skip-if-exists and each row is independent, so `--requeue` simply
resumes. Chunking is by ARM so a task never pays for an adapter it does not use.

    PY=.../envs/diffusion/bin/python
    $PY scripts/ctt_v2/s1/run_s1_gen.py --grid <S1_GRID.json> --pilot --chunk 0 --num-chunks 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import (
    PrefixConditionConfig,
    SuffixConditionConfig,
    ValidationConfig,
    ValidationSample,
)
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))
import encode_conditioning as ec  # noqa: E402  -- the ONE definition of a conditioning window

# $LAB is /projects/... on the Campus Cluster and /taiga/... on DeltaAI -- the SAME Taiga
# filesystem, mounted at two paths. Honour the env var so one script runs on both; the literal
# keeps the previous cc behaviour exactly when LAB is unset.
LAB = Path(os.environ.get("LAB", "/projects/illinois/eng/cs/jrehg/users/emirkisa"))
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
ARMS = REPO_ROOT / "eval_ladder/arms.yaml"
ADAPTER_TEMPLATE = "outputs/training/ladder2/{arm}/checkpoints/lora_weights_step_{step:05d}.safetensors"
SPEC_STEP = 2000            #: the specialists' pinned checkpoint (arms.yaml)


def conditioning_for(row: dict, conds_dir: Path) -> list:
    """Prefix from endpoint A; suffix from endpoint B's END window on two-sided rows.

    The windows are cut by encode_conditioning.cut_windows -- the same PX_PREFIX / PX_SUFFIX
    rule training and eval use -- so an S1 clip's conditioning is byte-for-byte the shape the
    trainer will later see. B's END window (not its start) is deliberate: it is the window the
    B-role caption was written from, and the window S2 guarantees byte-pure.
    """
    ec.cut_windows(Path(row["endpoint_a_mp4"]), row["endpoint_a"], out_dir=conds_dir)
    conds = [PrefixConditionConfig(video=str(conds_dir / f"{row['endpoint_a']}_start9.mp4"),
                                   num_frames=ec.PX_PREFIX)]
    if row["sided"] == "two":
        ec.cut_windows(Path(row["endpoint_b_mp4"]), row["endpoint_b"], out_dir=conds_dir)
        conds.append(SuffixConditionConfig(video=str(conds_dir / f"{row['endpoint_b']}_end9.mp4"),
                                           num_frames=ec.SUFFIX_GEN_FRAMES))
    return conds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--pilot", action="store_true", help="only rows flagged in_pilot")
    ap.add_argument("--arms", default=None, help="comma/pipe-separated arm allow-list")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve rows, cut windows, verify adapters -- then stop (no GPU)")
    args = ap.parse_args()

    grid = json.loads(Path(args.grid).read_text())
    arms_cfg = yaml.safe_load(ARMS.read_text())
    conds_dir = REPO_ROOT / grid["conds_dir"]
    conds_dir.mkdir(parents=True, exist_ok=True)

    rows = grid["rows"]
    if args.pilot:
        rows = [r for r in rows if r["in_pilot"]]
    if args.arms:
        keep = {a for a in args.arms.replace("|", ",").split(",") if a}
        rows = [r for r in rows if r["arm"] in keep]

    # chunk by ARM: a task then loads only the adapters it actually needs
    all_arms = sorted({r["arm"] for r in rows})
    my_arms = all_arms[args.chunk::args.num_chunks]
    rows = [r for r in rows if r["arm"] in my_arms]
    assert rows, f"no rows for chunk {args.chunk}/{args.num_chunks} (arms={my_arms})"

    for r in rows:
        assert r["prompt"], f"{r['row_id']} has no prompt -- rebuild the grid"
        assert r["prompt"].count(" sksz.") == 1, f"{r['row_id']}: token not exactly once"

    todo = [r for r in rows if not (REPO_ROOT / r["out_path"]).exists()]
    print(f"[s1gen] host={os.uname().nodename} chunk={args.chunk}/{args.num_chunks} "
          f"arms={my_arms}", flush=True)
    print(f"[s1gen] {len(todo)}/{len(rows)} rows to generate "
          f"({len(rows) - len(todo)} already on disk, skip-if-exists)", flush=True)
    if not todo:
        print("[s1gen] nothing to do")
        return

    seeds = {r["seed"] for r in todo}
    assert len(seeds) == 1, f"this task mixes seeds {seeds}; the grid pins one seed per run"
    seed = seeds.pop()

    inf = arms_cfg["inference"]
    vw, vh, vf = arms_cfg["resolution"]
    samples = [ValidationSample(prompt=r["prompt"], conditions=conditioning_for(r, conds_dir))
               for r in todo]

    if args.dry_run:
        for arm in my_arms:
            idxs = [i for i, r in enumerate(todo) if r["arm"] == arm]
            adapter = REPO_ROOT / ADAPTER_TEMPLATE.format(arm=arm, step=SPEC_STEP)
            assert adapter.exists(), f"adapter missing for {arm}: {adapter}"
            print(f"[dry] {arm}: {len(idxs)} rows, adapter OK, "
                  f"conds {[len(samples[i].conditions) for i in idxs]}")
        print(f"[dry] {len(todo)} rows resolved, all windows cut, all adapters present. OK.")
        return

    val_cfg = ValidationConfig(
        samples=samples,
        negative_prompt="worst quality, inconsistent motion, distorted, jittery",
        video_dims=(vw, vh, vf), frame_rate=24.0, seed=seed,
        inference_steps=inf["steps"], interval=1, guidance_scale=inf["guidance_scale"],
        stg_scale=inf["stg_scale"], stg_blocks=inf["stg_blocks"], stg_mode=inf["stg_mode"],
        generate_audio=False,
    )

    device = torch.device("cuda")
    # ValidationRunner.__init__ caches ALL prompt embeddings (Gemma, then unloaded) and ALL
    # conditioning encodes (VAE, then unloaded) up front -- once per task, not once per arm.
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA)

    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    targets = arms_cfg["targets"]["attn"]
    transformer = get_peft_model(transformer, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, target_modules=targets,
        lora_dropout=0.0, init_lora_weights=True))
    transformer = transformer.to(device).eval()
    print(f"[s1gen] transformer loaded, PEFT wrapped ({len(targets)} target modules)", flush=True)

    tmp = REPO_ROOT / grid["output_root"] / "_runner" / f"chunk{args.chunk}of{args.num_chunks}"
    done = 0
    for arm in my_arms:
        idxs = [i for i, r in enumerate(todo) if r["arm"] == arm]
        if not idxs:
            continue
        spec = arms_cfg["arms"][arm]
        assert spec["kind"] == "specialist" and spec["targets"] == "attn", f"{arm} is not a specialist"
        adapter = REPO_ROOT / ADAPTER_TEMPLATE.format(arm=arm, step=SPEC_STEP)
        assert adapter.exists(), f"adapter missing for {arm}: {adapter}"

        sd = {k.replace("diffusion_model.", "", 1): v for k, v in load_file(str(adapter)).items()}
        res = set_peft_model_state_dict(transformer.get_base_model(), sd)
        # Provenance seatbelt. Swapping adapters in a live process is the one way this script
        # could silently generate 11 batches from ONE specialist: PEFT reports a key mismatch
        # instead of raising, so an unnoticed rename would leave the previous arm's weights in
        # place and every clip would be mislabelled. Fail loudly instead.
        unexpected = list(getattr(res, "unexpected_keys", []) or [])
        assert not unexpected, f"{arm}: {len(unexpected)} unexpected adapter keys, e.g. {unexpected[:3]}"
        loaded = sum(1 for k in sd if "lora_" in k)
        assert loaded, f"{arm}: adapter carries no lora_* tensors"
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()[:16]
        print(f"[s1gen] --- {arm}: adapter {adapter.name} sha256:{digest} "
              f"({loaded} lora tensors) swapped in, {len(idxs)} rows", flush=True)

        t0 = time.time()
        saved = runner.run(transformer=transformer, step=done, output_dir=tmp, device=device,
                           progress=TrainingProgress(enabled=True, total_steps=1),
                           work_items=[(i, True) for i in idxs])
        for idx, path in saved:
            dst = REPO_ROOT / todo[idx]["out_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            Path(path).rename(dst)
            done += 1
            print(f"[done] {todo[idx]['row_id']} -> {todo[idx]['out_path']}", flush=True)
        print(f"[s1gen] {arm}: {len(saved)} clips in {time.time() - t0:.0f}s", flush=True)

    print(f"[s1gen] wrote {done}/{len(todo)} clips", flush=True)


if __name__ == "__main__":
    main()
