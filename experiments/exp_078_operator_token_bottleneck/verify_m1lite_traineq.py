"""M1-lite train==inference check on the COARSE reference — the advisor's flagged pre-read item.

Training fed the model coarse reference latents built by `encode_coarse_refs.py`:
    std121 pixels -> TF.resize BICUBIC to 128x96 -> clamp[0,1] -> *2-1 -> vae(v)

Generation builds its coarse reference inside the trainer instead:
    std121 pixels -> _resize_and_center_crop(128, 96) [BICUBIC + clamp, crop is a no-op at this
    aspect ratio] -> *2-1 -> vae.tiled_encode(v, TilingConfig.default())

The pixel stages are already verified identical by inspection. The one remaining difference is the
VAE call: plain `vae(v)` vs `tiled_encode(...)`. If those disagree materially, M1-lite is being
evaluated on a reference distribution it never trained on, and its read would be invalid — exactly
the failure this check exists to catch. Reports relative L2 between the two latents.

    sbatch ... --wrap "python experiments/exp_078_operator_token_bottleneck/verify_m1lite_traineq.py"
"""

import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))
sys.path.insert(0, str(LAB / "LTX-2-bneck/packages/ltx-trainer/src"))
import encode_conditioning as ec  # noqa: E402

MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
COARSE = EXP / "dataset" / "coarse_ref_latents"
N_CLIPS = 3


def main() -> None:
    from ltx_core.model.video_vae import TilingConfig  # noqa: PLC0415
    from ltx_trainer.validation_runner import ValidationRunner  # noqa: PLC0415

    device = "cuda"
    vae = ec.load_vae(str(MODEL), device=device)
    stored = sorted(COARSE.rglob("*.pt"))[:N_CLIPS]
    print(f"[traineq] checking {len(stored)} clips  gpu={torch.cuda.get_device_name(0)}")

    worst = 0.0
    for pt in stored:
        clip = pt.stem
        src = next((REPO_ROOT / "data/processed/transitions_std121").glob(f"*/{clip}.mp4"), None)
        if src is None:
            print(f"[traineq] no mp4 for {clip}"); continue
        train_lat = torch.load(pt, weights_only=False)["latents"].float()   # what TRAINING saw

        # --- reproduce the INFERENCE path exactly
        px = ec.preprocess(src)                                            # [F,C,H,W] in [0,1]
        pre, _ = ValidationRunner._preprocess_reference(px, 640, 480, downscale_factor=5,
                                                        temporal_scale_factor=1)
        vae.to(device)
        with torch.inference_mode():
            infer_lat = vae.tiled_encode(pre.to(device=device, dtype=torch.bfloat16),
                                         TilingConfig.default()).cpu()[0].float()

        if tuple(train_lat.shape) != tuple(infer_lat.shape):
            print(f"[traineq] {clip}: SHAPE MISMATCH train={tuple(train_lat.shape)} "
                  f"infer={tuple(infer_lat.shape)}  <-- FATAL")
            continue
        rel = float((infer_lat - train_lat).norm() / train_lat.norm().clamp(min=1e-12))
        worst = max(worst, rel)
        print(f"[traineq] {clip:28s} shape={tuple(train_lat.shape)} rel_L2={rel:.5f}")

    print(f"\n[traineq] worst rel_L2 = {worst:.5f}")
    ok = 0.0 < worst < 0.02 or worst == 0.0
    if worst >= 0.02:
        print("[traineq] VERDICT: MISMATCH — the coarse reference at generation differs from what "
              "M1-lite trained on. Its read would be INVALID.")
    else:
        print("[traineq] VERDICT: train==inference (difference is float noise)")
    # Exit non-zero on mismatch so this is a HARD GATE: the generation job is chained with
    # --dependency=afterok, and therefore cannot run on a setup that failed this check.
    raise SystemExit(0 if worst < 0.02 else 1)


if __name__ == "__main__":
    main()
