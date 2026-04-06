# LTX-2 Notes

Notes on the Lightricks LTX-2 video generation model.
Code lives in `src/LTX-2/`.

## Files

| File | Contents |
|---|---|
| [`information_bank.md`](information_bank.md) | **Dense reference:** Diffusers vs vendored LTX-2, two-stage Hub recipe, distilled LoRA role, `LTX2ConditionPipeline` vs `LTX2Pipeline`, C2V indices, pitfalls (PEFT, Stage 2 shapes, offload), exp_020; links to official `ltx2.md`. (Copy lives in `experiments/exp_020_ltx2_c2v_diffusers/LTX2_INFORMATION_BANK.md`.) |
| [`conditioning_mechanism.md`](conditioning_mechanism.md) | Deep notes on the conditioning pipeline: VAE encoding, patchification, position bounding boxes, RoPE, denoise mask, attention mask, `VideoConditionByKeyframeIndex`, two-stage pipeline, frame_idx alignment formula, exp_014 (keyframe interpolation) and exp_015 (C2V clip conditioning), option comparison, future improvements. |

## Quick Reference

- **Temporal scale:** 8 (causal VAE — `F_lat = (F_pix - 1) // 8 + 1`)
- **Spatial scale:** 32 × 32
- **Latent channels:** 128
- **Positions:** pixel-space 3D bounding boxes `(B, 3, N, 2)` → divided by fps → fed to RoPE
- **Causal fix:** shift all temporal coords by `-(8-1)=-7`, clamp at 0 — only for clips/images at `frame_idx=0`
- **End clip frame_idx:** `num_output_frames - num_clip_frames`
- **Denoise mask 1.0** = noisy output token; **0.0** = frozen conditioning token
- **Attention mask** = block matrix isolating conditioning groups from each other
