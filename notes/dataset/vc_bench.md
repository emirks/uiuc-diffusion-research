# VC-Bench: Paper, Benchmark & Data — Summary

Summary from visiting all chat links and reading the full paper (arXiv HTML, OpenReview, official repo, Hugging Face).

---

## 1. Links (from chat)

| Resource | URL |
|----------|-----|
| **arXiv** | https://arxiv.org/abs/2601.19236 |
| **PDF** | https://arxiv.org/pdf/2601.19236 |
| **HTML (full paper)** | https://arxiv.org/html/2601.19236v1 |
| **OpenReview (ICLR 2026)** | https://openreview.net/forum?id=Ws8HwWHf8N |
| **Official repo (eval + dataset)** | https://anonymous.4open.science/r/VC-Bench-1B67/ |
| **Hugging Face dataset** | https://huggingface.co/datasets/Kevinson-lzp/VC-Bench |

**Not the same:** VCBench (video cognition, arXiv:2411.09105, `buaaplay/VCBench`) is a different benchmark.

---

## 2. Task: Video Connecting (VC)

- **Input:** Start clip \(V_S\) and end clip \(V_E\) (same resolution & frame rate).
- **Output:** A full video \(V\) such that:
  - Start of \(V\) matches \(V_S\), end of \(V\) matches \(V_E\).
  - Middle is a **generated transition** that is spatio-temporally coherent and smooth.
- **Difference from FLF2V:** VC uses **two video segments** (motion, context), not just first/last frames; start and end can be from **different scenes**, so the task is harder (semantic gap, motion alignment).

---

## 3. Dataset (VC-Bench)

- **Size:** 1,579 high-quality videos (paper); HF release has 929 rows (eval subset); your local CSV has 1,798 rows (likely full + extra).
- **Sources:** Pexels, Pixabay, Mixkit, YouTube.
- **Structure:** 15 main categories, 72 subcategories (BERT-based classification from captions).
- **Pipeline:** Web crawl → aesthetic + motion + periodic filtering → PySceneDetect (1 or 2 scenes) → clip extraction → Qwen2-VL captions.
- **Clip setup:** Total 5 s; start/end clips 2–4 s each; for two-scene videos, transition frames are excluded from start/end.
- **Quality:** ≥720p, aesthetic score ~0.55, no strong periodic motion; caption length 18–57 chars.

---

## 4. Evaluation metrics (9 dimensions → 3 scores)

**Video Quality Score (VQS)**  
VQS = [Q_S + Q_B + (1−Q_F) + Q_A + Q_I] / 5  
- Subject consistency (DINO), background consistency (CLIP), flickering severity (YUV/HSV), aesthetic score (LAION), imaging quality (MUSIQ).

**Start–End Consistency Score (SECS)**  
SECS = [C_P + (1−C_OF)] / 2  
- Pixel consistency (SSIM vs. original start/end), optical flow error (motion alignment).

**Transition Smoothness Score (TSS)**  
TSS = [(1−T_CD) + T_LP] / 2  
- Video connecting distance (DTW + SSIM/d), local perceptual consistency (VGG + LPIPS).

**Total score:** (VQS + SECS + TSS) / 3.  
Human study (30 raters): high correlation with these scores; ICC(2,K) used for rater consistency.

---

## 5. Transfer approach (running OSS models in VC mode)

- **Architecture:** DiT-based video models + 3D VAE.
- **(1) Latent mapping:** Start/end clips → latent start/end; middle filled with noise, denoised by the model.
- **(2) SLERP:** Spherical linear interpolation of start/end features (as in TVG) for the transition.
- **Limitations (paper):** VAE compression loss, patch serialization at boundaries, latent denoising noise → hard to get pixel-perfect start/end match.

**Models evaluated:** Wan-2.1 (1.3B, 14B), CogVideoX (2B, 5B), Open-Sora 2.0 (11B), Ruyi (7B). Conditioning: 1 frame, 1 s, 1.5 s, or 2 s as start/end; 5 s output. Wan-2.1 best overall; all struggle more on SECS/TSS than on VQS; two-scene connecting harder than same-scene.

---

## 6. Official repo (anonymous.4open.science/r/VC-Bench-1B67)

- **Contents:** `evaluate.py`, `evaluate.sh`, `benchmark/`, `asset/`, `CLIP/`, sampled videos; README points to HF dataset.
- **Usage:** `python evaluate.py --videos_path /path/to/folder_or_video/`; multi-GPU via `torchrun`.
- **Dimensions:** subject_consistency, background_consistency, flickering_severity, aesthetic_score, imaging_quality, pixel_consistency, optical_flow_error, connecting_distance, local_perceptual_consistency.
- **Total score:** weighted combination of VQS, SECS, TSS (negative metrics inverted by 1−x).

---

## 7. Hugging Face (Kevinson-lzp/VC-Bench)

- **Content:** 929 rows (train) in the dataset viewer; repo snapshot has **1,261 MP4s** + **VC-Bench.csv** (1 header + 1,797 data rows).
- **CSV columns (local):** `filename`, `resolution`, `length`, `fps`, `num_scenes`, `scene_start_frames`, `category`, `aesthetic_score`, `caption`.
- **Filename format:** Either basename (e.g. `action_1402988_1920x1080.mp4`) or path with backslashes (e.g. `Actions & Activities\action\action_1402988_1920x1080.mp4`) depending on release.
- **vc-bench-hf folder:** If you downloaded the full repo: you have **all metadata** (CSV, 1,797 rows) and **1,261 videos** (MP4s). The CSV lists 1,797 entries; only 1,261 of those have a corresponding MP4 on the HF repo, so **536 CSV rows have no video file** on HF (authors may publish only the “evaluation/sampled” subset as videos).

---

## 8. Takeaways

- **Task:** VC = generate the *middle* between start and end clips; strict start/end consistency and transition smoothness.
- **Data:** 1,579 videos, 15 categories / 72 subcategories, 5 s setup, 1- or 2-scene; captions and metadata in CSV.
- **Metrics:** VQS (quality), SECS (start/end match), TSS (smooth middle); 9 sub-metrics; human-aligned.
- **OSS usage:** Eval code in the repo; run existing DiT models in VC mode via latent mapping + SLERP; no full retrain.
- **Limitations (paper):** Only 5 s videos; only open-source models; two-scene and long-range consistency still hard.

This note is for your own reference; all content is from the linked pages and the paper.
