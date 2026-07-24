"""M1-lite: VAE-encode the 139 reference clips at COARSE 96x128 -> (F=16,H=4,W=3)=192-token latents.

The advisor's M1 successor shrinks the reference substrate 4800->192 tokens using a REAL (not
avgpooled) coarse VAE encode, so the pretrained reading circuit stays in-distribution while the
copy/bleed substrate is diluted. Spatial factor 5 (target latent 20//4 == 15//3 == 5, uniform);
temporal factor 1 (F=16 both).

Reuses encode_conditioning's VAE path (the exact process_videos pipeline), resizing the full-res
pixels to 128H x 96W before the encode. Also runs the pre-registered ENCODE->DECODE ROUNDTRIP sanity
check on the first clip (is the tiny latent a recognizable downscaled clip, not garbage?).

    python experiments/exp_078_operator_token_bottleneck/encode_coarse_refs.py         # GPU
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))
import encode_conditioning as ec  # noqa: E402

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
STD = REPO_ROOT / "data/processed/transitions_std121"
OUT = EXP / "dataset" / "coarse_ref_latents"          # <class>/<clip>.pt
COARSE_H, COARSE_W = 128, 96                            # -> latent (4, 3); factor 5


def unique_reference_clips() -> list[str]:
    import glob
    import os
    refs = set()
    for f in glob.glob(str(EXP / "dataset/roots/b1/reference_latents/**/*.pt"), recursive=True):
        refs.add(os.path.basename(f).split("__ref_")[1][:-3])
    return sorted(refs)


def clip_mp4(clip: str) -> Path:
    hits = list(STD.glob(f"*/{clip}.mp4"))
    if not hits:
        raise FileNotFoundError(f"no corpus mp4 for {clip}")
    return hits[0]


def main() -> None:
    device = "cuda"
    vae = ec.load_vae(str(MODEL), device=device)
    refs = unique_reference_clips()
    print(f"[coarse] {len(refs)} reference clips -> {COARSE_W}x{COARSE_H} coarse latents")

    roundtrip_done = False
    for i, clip in enumerate(refs):
        src = clip_mp4(clip)
        cls = src.parent.name
        dst = OUT / cls / f"{clip}.pt"
        if dst.exists():
            continue
        px = ec.preprocess(src)                              # [F,C,H,W] in [0,1], full res
        px_c = F.interpolate(px, size=(COARSE_H, COARSE_W), mode="bilinear", align_corners=False)
        lat = ec.encode(px_c, vae, device, torch.bfloat16)   # [1,128,16,4,3]
        lat0 = lat[0].to(torch.bfloat16).cpu()
        _, fF, fH, fW = lat0.shape
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"latents": lat0, "num_frames": fF, "height": fH, "width": fW, "fps": 24.0}, dst)

        if not roundtrip_done:
            # pre-registered sanity: encode->decode, report the tiny latent is not garbage
            from ltx_trainer.model_loader import load_video_vae_decoder
            dec = load_video_vae_decoder(str(MODEL), device=device, dtype=torch.bfloat16)
            with torch.inference_mode():
                rec = dec(lat.to(device))
            print(f"[roundtrip] {clip}: coarse latent {tuple(lat0.shape)} (K={fF*fH*fW}); "
                  f"pixel {tuple(px.shape)}->{tuple(px_c.shape)}; decoded {tuple(rec.shape)}; "
                  f"latent absmean {lat0.float().abs().mean():.3f} rms {lat0.float().pow(2).mean().sqrt():.3f} "
                  f"decoded[0,1] range [{rec.float().min():.2f},{rec.float().max():.2f}]")
            del dec
            roundtrip_done = True

        if (i + 1) % 25 == 0:
            print(f"[coarse] {i + 1}/{len(refs)}")

    n = len(list(OUT.rglob("*.pt")))
    print(f"[coarse] done: {n} coarse reference latents in {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
