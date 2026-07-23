"""ladder2 — the generator. Consumes `registry.jsonl` rows for ONE arm.

    row  x  seed  ==  exactly one video

Nothing is decided here. The row already carries the prompt, the conditioning, the reference
and the arm; this script only loads that arm's adapter once and renders its rows. Selection is
by `--arm` (which model) and optionally `--priority` (which wave), so a generation job is
always "this model, these rows".

    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <repo>/eval_ladder/run_gen.py \
        --arm spec_color_rain --seed 42 [--priority P0] [--chunk 0 --num-chunks 4]

Skip-if-exists on the output path, so preemption + requeue simply continues.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import (
    PrefixConditionConfig,
    ReferenceConditionConfig,
    SuffixConditionConfig,
    ValidationConfig,
    ValidationSample,
)
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
STD = REPO_ROOT / "data/processed/transitions_std121"
REGISTRY = HERE / "registry.jsonl"
ARMS = HERE / "arms.yaml"
OUT_ROOT = REPO_ROOT / "outputs/videos/ladder2"


def build_sample(row: dict) -> ValidationSample:
    """Conditioning is a pure function of the row — the same rule the mask uses at eval."""
    conds = []
    if row.get("conditioning") != "none" and row["endpoint"] is not None:
        paths = ec.cond_paths(row["endpoint"], row["sided"])
        conds.append(PrefixConditionConfig(video=str(paths["prefix"]), num_frames=ec.PX_PREFIX))
        if row["sided"] == "two":
            conds.append(SuffixConditionConfig(video=str(paths["suffix"]),
                                               num_frames=ec.SUFFIX_GEN_FRAMES))
    if row.get("reference"):
        # authoritative clip -> class (clip names do NOT reliably encode the class)
        ref_path = STD / prompts.clip_class(row["reference"]) / f"{row['reference']}.mp4"
        assert ref_path.exists(), f"reference clip not found: {ref_path}"
        conds.append(ReferenceConditionConfig(video=str(ref_path), downscale_factor=1,
                                              temporal_scale_factor=1, include_in_output=False))
    return ValidationSample(prompt=row["prompt"], conditions=conds)


def load_rows(arm: str, priority: str | None, cells: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["arm"] == arm]
    if priority:
        keep = set(priority.split(","))
        rows = [r for r in rows if r["priority"] in keep]
    if cells:
        # NOTE: Slurm's --export is itself comma-separated, so a comma-joined list silently
        # truncates to its first element. Accept "|" (what submit.py sends) as well as ",".
        want = {c for c in re.split(r"[,|]", cells) if c}
        rows = [r for r in rows if r["cell"] in want]
    return rows


def resolve_adapter(arm: str, arms_cfg: dict, step: int | None = None) -> tuple[Path | None, list[str]]:
    """`step` overrides the arm's pinned checkpoint — used ONLY by the convergence diagnostic
    (ckpt-4500 vs ckpt-5000). Claim-bearing generation always uses the pinned step from arms.yaml."""
    spec = arms_cfg["arms"][arm]
    if spec.get("adapter", "unset") is None or spec["kind"] in ("base", "text_floor"):
        return None, []
    path = REPO_ROOT / arms_cfg["adapter_template"].format(arm=arm, step=step or spec["step"])
    assert path.exists(), f"adapter missing for {arm}: {path}"
    return path, arms_cfg["targets"][spec["targets"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--priority", default=None, help="e.g. P0 or P0,P1")
    ap.add_argument("--cells", default=None, help="comma-separated cell names")
    ap.add_argument("--step", type=int, default=None,
                    help="override the pinned checkpoint (convergence diagnostic only)")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()

    arms_cfg = yaml.safe_load(ARMS.read_text())
    assert args.seed in arms_cfg["seeds"], f"seed {args.seed} is not a registered seed"
    rows = load_rows(args.arm, args.priority, args.cells)

    def out_path(r: dict) -> Path:
        # Baseline rows share one canonical video per (endpoint, sided) through
        # video_key = "<dir>/<name>" — the arm never sees the donor, so per-task outputs would
        # be byte-duplicates. Rows without video_key keep row x seed = one video.
        vk = r.get("video_key")
        d, name = vk.split("/", 1) if vk else (r["arm"], r["item_id"])
        suffix = f"__ck{args.step}" if args.step else ""
        return OUT_ROOT / (d + suffix) / f"{name}__s{args.seed}.mp4"

    # dedup by canonical path BEFORE chunk slicing, so two array tasks can never race on the
    # same shared video; then slice. (No-op for arms without video_key.)
    seen_paths: set[Path] = set()
    rows = [r for r in rows
            if (p := out_path(r)) not in seen_paths and not seen_paths.add(p)]
    rows = rows[args.chunk::args.num_chunks]
    assert rows, f"no registry rows for arm={args.arm} priority={args.priority} cells={args.cells}"

    todo = [r for r in rows if not out_path(r).exists()]
    print(f"[gen] arm={args.arm} seed={args.seed} chunk={args.chunk}/{args.num_chunks}: "
          f"{len(todo)}/{len(rows)} to generate")
    if not todo:
        print("[gen] nothing to do")
        return
    for r in todo:
        out_path(r).parent.mkdir(parents=True, exist_ok=True)

    adapter, target_modules = resolve_adapter(args.arm, arms_cfg, args.step)
    inf = arms_cfg["inference"]
    # ValidationSample.video_dims is (WIDTH, HEIGHT, FRAMES) — the corpus is portrait
    # 480x640, so this reads 480 wide by 640 high (same tuple exp_074 generated with).
    vw, vh, vf = arms_cfg["resolution"]
    val_cfg = ValidationConfig(
        samples=[build_sample(r) for r in todo],
        negative_prompt="worst quality, inconsistent motion, distorted, jittery",
        video_dims=(vw, vh, vf), frame_rate=24.0, seed=args.seed,
        inference_steps=inf["steps"], interval=1, guidance_scale=inf["guidance_scale"],
        stg_scale=inf["stg_scale"], stg_blocks=inf["stg_blocks"], stg_mode=inf["stg_mode"],
        generate_audio=False,
    )

    device = torch.device("cuda")
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA)
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    if adapter is not None:
        print(f"[gen] adapter {adapter.name} ({len(target_modules)} target modules)")
        transformer = get_peft_model(transformer, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, target_modules=target_modules,
            lora_dropout=0.0, init_lora_weights=True))
        sd = {k.replace("diffusion_model.", "", 1): v for k, v in load_file(str(adapter)).items()}
        set_peft_model_state_dict(transformer.get_base_model(), sd)
    else:
        print(f"[gen] {args.arm}: no adapter (base weights)")
    transformer = transformer.to(device).eval()

    tmp = OUT_ROOT / "_runner" / f"{args.arm}_s{args.seed}_c{args.chunk}"
    saved = runner.run(transformer=transformer, step=0, output_dir=tmp, device=device,
                       progress=TrainingProgress(enabled=True, total_steps=1))
    for idx, path in saved:
        dst = out_path(todo[idx])
        Path(path).rename(dst)
        print(f"[done] {todo[idx]['item_id']} s{args.seed} -> {dst.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
