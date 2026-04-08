# Wan 2.1: Model Overview & C2V Reference

Wan 2.1 is a DiT-based video generation model from Alibaba (1.3B and 14B variants).
It is the **best-performing open-source model** on the VC-Bench benchmark (as of the paper evaluation).
This project uses it primarily via HuggingFace Diffusers for C2V experiments.

---

## Latent geometry

| Quantity | Value / formula |
| -------- | --------------- |
| Temporal VAE scale | **4**: `T_lat = (num_frames - 1) // 4 + 1` |
| Spatial VAE scale | **÷8** per H/W dim |
| `num_frames` constraint | `(num_frames - 1) % 4 == 0` |
| Transformer input shape | `(B, 36, T_lat, H_lat, W_lat)` |
| Latent channels | 16 noisy + 4 mask + 16 VAE condition = **36** |

---

## Conditioning (FLF2V / C2V)

| Topic | Detail |
| ----- | ------ |
| Conditioning channel split | 16 noisy latents \| 4 mask \| 16 VAE condition = 36 |
| Mask build order | pixel-frame ones → repeat first-frame group ×4 → concat → `view(B, T_lat, 4, H, W)` → `transpose(1, 2)` |
| CLIP image embed | Always encodes exactly **two** frames: `[start_frame, end_frame]` |
| `anchor_frames` constraint | `anchor_frames * 2 < num_frames` |

---

## VC-Bench evaluation context

From the VC-Bench paper:
- Wan 2.1 (14B) achieves the **best total score** across all evaluated models.
- Still struggles on **SECS** (start/end pixel consistency) and **TSS** (transition smoothness) relative to VQS.
- Two-scene connecting (semantically distant clips) is harder than same-scene for all models.
- Transfer method: latent mapping (start/end clips → latent anchors, middle filled with noise) + SLERP of features.

---

## Hub IDs

| Variant | HF Hub ID |
| ------- | --------- |
| 1.3B T2V | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| 14B T2V | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| 1.3B I2V | `Wan-AI/Wan2.1-I2V-1.3B-480P-Diffusers` |
| 14B I2V | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` |
