#!/usr/bin/env python3
"""grid v3 — null-operator guidance (DCG) generation over a STAMPED registry, contract-v2 layout.

Generalises misc/2026-09-02_dcg_dualforce_control/gen_dcg_sweep.py (the driver behind gens/029-032): the SAME
inference stack (src/LTX-2-ctt-v2-train packages, one_way IC-LoRA), the SAME deployed config (text-CFG 4.0,
STG 1.0, 30 steps, crossfade null, dcg_use_null_as_reference=False, dcg_rescale=0), the SAME adapter
(runs/012 @ step 1000, r128/a128) — with the row source, the guidance weight, the seed, the chunking, the
output layout and the frame contract made explicit:

  --registry  a stamped registry (eval_ladder/registry_<arm>.jsonl); rows with arm == --arm are generated
  --w         guidance weight (grid v3 headline: 6.0; 1.0 = plain demo branch)
  --seed / --chunk / --num-chunks   one array task = (seed, chunk); resumable (skip-if-exists)
  --out-root  a store gen subentry (gens/NNN_<arm>/KK_<variant>__dai); clips land flat in videos/ as
              <item_id>__s<seed>.mp4 — exactly run_gen.py --videos-dir
  GEN_FRAMES / GEN_PREFIX_FRAMES   the EffectData tier runs at 81 frames with the frame-0 anchor
              (encode_conditioning.prefix_frames()), the Higgsfield tier at 121 with the 9-frame prefix.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

LAB = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa")
DR = LAB / "diffusion-research"
EVAL = DR / "eval_ladder"
STD = DR / "data/processed/transitions_std121"
sys.path.insert(0, str(EVAL))
import encode_conditioning as ec  # noqa: E402
import prompts as P  # noqa: E402
from ltx_trainer.config import (PrefixConditionConfig, ReferenceConditionConfig,  # noqa: E402
                                SuffixConditionConfig, ValidationConfig, ValidationSample)
from ltx_trainer.model_loader import load_transformer  # noqa: E402
from ltx_trainer.progress import TrainingProgress  # noqa: E402
from ltx_trainer.validation_runner import ValidationRunner  # noqa: E402

MODEL = Path(os.environ.get("LTX_MODEL", LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"))
GEMMA = Path(os.environ.get("LTX_GEMMA", LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"))
ADAPTER = Path(os.environ.get("DCG_ADAPTER", DR / "store/runs/012_dualforce_control/checkpoints/lora_weights_step_01000.safetensors"))
RANK = int(os.environ.get("DCG_RANK", "128"))
ALPHA = int(os.environ.get("DCG_ALPHA", "128"))
GS = float(os.environ.get("DCG_GS", "4.0"))
STG = float(os.environ.get("DCG_STG", "1.0"))
FRAMES = int(os.environ.get("GEN_FRAMES", "121"))
RES = (480, 640, FRAMES)


def ref_clip_path(clip: str) -> str:
    return str(STD / P.clip_class(clip) / f"{clip}.mp4")


def make_sample(row: dict) -> ValidationSample:
    paths = ec.cond_paths(row["endpoint"], row["sided"])
    conds = [PrefixConditionConfig(video=str(paths["prefix"]), num_frames=ec.prefix_frames())]
    if row["sided"] == "two":
        conds.append(SuffixConditionConfig(video=str(paths["suffix"]), num_frames=ec.SUFFIX_GEN_FRAMES))
    conds.append(ReferenceConditionConfig(video=ref_clip_path(row["reference"]), downscale_factor=1, attention="one_way",
                                          temporal_scale_factor=1, include_in_output=False, dcg_null_video=None))
    return ValidationSample(prompt=row["prompt"], conditions=conds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--w", type=float, default=6.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.registry).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["arm"] == args.arm]
    rows.sort(key=lambda r: r["item_id"])
    todo = rows[args.chunk::args.num_chunks]
    out = Path(args.out_root)
    vids = out / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get("DCG_SCRATCH", str(out / "_runner"))) / f"s{args.seed}_c{args.chunk}"
    pending = [r for r in todo if not (vids / f"{r['item_id']}__s{args.seed}.mp4").exists()]
    print(f"[dcg-v3] arm={args.arm} w={args.w} seed={args.seed} chunk={args.chunk}/{args.num_chunks} "
          f"rows={len(todo)} pending={len(pending)} frames={FRAMES} prefix={ec.prefix_frames()} gs={GS} stg={STG} adapter={ADAPTER.name}", flush=True)
    if not pending:
        return

    samples = [make_sample(r) for r in pending]
    cfg = ValidationConfig(samples=samples, video_dims=RES, frame_rate=24.0, seed=args.seed,
                           inference_steps=30, guidance_scale=GS, stg_scale=STG, stg_blocks=[29], stg_mode="stg_v",
                           generate_audio=False, dcg_scale=args.w, dcg_null_kind="crossfade")
    cfg.dcg_use_null_as_reference = False
    cfg.dcg_rescale = 0.0
    device = torch.device("cuda")
    runner = ValidationRunner(config=cfg, model_path=MODEL, text_encoder_path=GEMMA)
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    targets = yaml.safe_load((EVAL / "arms.yaml").read_text())["targets"]["attn_ffn"]
    transformer = get_peft_model(transformer, LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=targets,
                                                         lora_dropout=0.0, init_lora_weights=True))
    sd = {k.replace("diffusion_model.", "", 1): v for k, v in load_file(str(ADAPTER)).items()}
    set_peft_model_state_dict(transformer.get_base_model(), sd)
    transformer = transformer.to(device).eval()

    for i, r in enumerate(pending):
        dst = vids / f"{r['item_id']}__s{args.seed}.mp4"
        if dst.exists():
            continue
        runner._config.seed = args.seed
        saved = runner.run(transformer=transformer, step=0, output_dir=scratch / r["item_id"], device=device,
                           progress=TrainingProgress(enabled=True, total_steps=1), work_items=[(i, True)])
        assert saved, f"no output for {dst.name}"
        tmp = dst.with_suffix(".tmp.mp4")
        shutil.move(str(saved[0][1]), str(tmp))
        os.replace(tmp, dst)
        print(f"[dcg-v3] done {dst.name}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[dcg-v3] chunk complete: {len(pending)} clips", flush=True)


if __name__ == "__main__":
    main()
