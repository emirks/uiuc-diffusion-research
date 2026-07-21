#!/usr/bin/env python
"""Precompute cond_clean latents for the causal-VAE endpoint-bleed fix (exp_073).

The suffix intrinsic-conditioning token is trained (in exp_062/exp_064) by pinning the
LAST latent frame of the FULL-video VAE encode. The LTX-2 video VAE is temporally causal,
so that last latent frame has a backward receptive field into the MIDDLE of the clip -> it
carries middle content. At inference the suffix is built from a standalone encode of only
the trailing frames (bleed-free). This script produces "cond_clean" latents in which the
last latent frame is replaced by the last frame of a STANDALONE encode of the trailing 9
pixel frames, so training pins the same bleed-free anchor the model sees at inference.

The trailing cut is sliced from the SAME preprocessed pixel tensor that produced the
original latents (read_video + clamp[0,1] + Normalize(0.5,0.5) == *2-1), NOT from a
re-decoded mp4 (advisor directive; the eval-side end9.mp4 crf12 roundtrip is an accepted
2nd-order residual, matching the prefix's own asymmetry).

Modes:
  --mode gate  : K1/K2 manipulation check in fp32. For each clip, encode full + first-9 +
                 last-9 cuts and report relative-L2 of (standalone cut) vs (full-encode
                 slice) for prefix (K1: must be ~0) and suffix (K2: must be materially >0).
                 Writes gate_stats.json. Writes NO dataset files.
  --mode build : production bf16. base = ORIGINAL stored latents (exact copy); for a
                 TWO-SIDED clip overwrite ONLY the last latent frame with the standalone
                 last-9 encode's last frame. Asserts bitwise-equality to the original
                 everywhere except that one frame. Writes cond_clean latent trees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP = REPO_ROOT / "experiments/exp_073_cond_bleed_fix"
STD = REPO_ROOT / "data/processed/transitions_std121"
MODEL_PATH = "/projects/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"

SP_062 = REPO_ROOT / "experiments/exp_062_ladder_r2r3_specialists/dataset/.precomputed"
IC_064_CLIPS = REPO_ROOT / "experiments/exp_064_ic3_aligned_retrain/dataset/.precomputed_clips"
IC_064_PAIRS = REPO_ROOT / "experiments/exp_064_ic3_aligned_retrain/dataset/pairs.json"

SPECIALIST_TWOSIDED = ["shadow_smoke", "hero_flight"]  # the only two-sided specialists

PX_PREFIX = 9   # first 9 pixel frames -> 2 latent frames
PX_SUFFIX = 9   # last  9 pixel frames -> 2 latent frames, keep the last

sys.path.insert(0, "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-cond-bleed-fix/packages/ltx-trainer/src")
from ltx_trainer.model_loader import load_video_vae_encoder  # noqa: E402
from ltx_trainer.video_utils import read_video  # noqa: E402


def preprocess(clip_path: Path, max_frames: int = 121) -> torch.Tensor:
    """std mp4 -> pixel tensor [F, C, H, W] in [0,1] (same path process_dataset.py used)."""
    video, _ = read_video(clip_path, max_frames=max_frames)  # [F, C, H, W] in [0,1]
    return video


def encode(pixel_fchw: torch.Tensor, vae, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Encode [F,C,H,W] in [0,1] -> latents [1, C', F', H', W'] (clamp01 + *2-1, then vae)."""
    v = pixel_fchw.clamp(0.0, 1.0).mul(2.0).sub(1.0)      # [-1,1], matches Normalize(0.5,0.5)
    v = v.permute(1, 0, 2, 3).unsqueeze(0)                # [1, C, F, H, W]
    v = v.to(device=device, dtype=dtype)
    with torch.inference_mode():
        lat = vae(v)                                       # [1, C', F', H', W']
    return lat


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def _std_path(cls: str, stem: str) -> Path:
    return STD / cls / f"{stem}.mp4"


# --------------------------------------------------------------------------- clip inventory
def specialist_clips() -> list[tuple[str, str, Path]]:
    """(class, stem, original_latents_pt) for every specialist two-sided training clip."""
    out = []
    for cls in SPECIALIST_TWOSIDED:
        latdir = SP_062 / cls / "latents"
        for pt in sorted(latdir.rglob("*.pt")):
            out.append((cls, pt.stem, pt))
    return out


def ic3_target_clips() -> tuple[list[tuple[str, str, Path]], set[str]]:
    """(class, stem, original_clip_latents_pt) for every ic3 target clip + two-sided classes."""
    pairs = json.loads(IC_064_PAIRS.read_text())
    twosided = {p["class"] for p in pairs if p["sidedness"] == "twosided"}
    targets = sorted({(p["class"], p["target"]) for p in pairs})
    out = [(cls, stem, IC_064_CLIPS / "latents" / cls / f"{stem}.pt") for cls, stem in targets]
    return out, twosided


# --------------------------------------------------------------------------- gate (K1/K2)
def run_gate(n_specialist: int, n_ic3: int, device: str) -> None:
    vae = load_video_vae_encoder(MODEL_PATH, device=device, dtype=torch.float32)
    rows = []
    sp = specialist_clips()
    ic, tw = ic3_target_clips()
    ic_tw = [(c, s, p) for (c, s, p) in ic if c in tw]
    sample = sp[:n_specialist] + ic_tw[:n_ic3]
    print(f"[gate] fp32, {len(sample)} clips ({min(n_specialist,len(sp))} specialist + "
          f"{min(n_ic3,len(ic_tw))} ic3 two-sided)")
    for cls, stem, _ in sample:
        px = preprocess(_std_path(cls, stem))
        assert px.shape[0] == 121, f"{cls}/{stem}: expected 121 frames, got {px.shape[0]}"
        full = encode(px, vae, device, torch.float32)
        pre_cut = encode(px[:PX_PREFIX], vae, device, torch.float32)
        suf_cut = encode(px[-PX_SUFFIX:], vae, device, torch.float32)
        pre_rel = rel_l2(pre_cut[:, :, :2], full[:, :, :2])   # prefix: first 2 latent frames
        suf_rel = rel_l2(suf_cut[:, :, -1:], full[:, :, -1:])  # suffix: last latent frame
        rows.append({"clip": f"{cls}/{stem}", "prefix_rel_l2": pre_rel, "suffix_rel_l2": suf_rel})
        print(f"  {cls}/{stem:22s}  prefix_rel={pre_rel:.5f}  suffix_rel={suf_rel:.5f}")

    def med(key):
        vals = sorted(r[key] for r in rows)
        return vals[len(vals) // 2]

    pre_med, suf_med = med("prefix_rel_l2"), med("suffix_rel_l2")
    stats = {
        "n_clips": len(rows), "dtype": "float32",
        "prefix_rel_l2": {"median": pre_med, "max": max(r["prefix_rel_l2"] for r in rows),
                          "min": min(r["prefix_rel_l2"] for r in rows)},
        "suffix_rel_l2": {"median": suf_med, "max": max(r["suffix_rel_l2"] for r in rows),
                          "min": min(r["suffix_rel_l2"] for r in rows)},
        "K1_prefix_identity_pass": bool(pre_med < 0.02 and pre_med < 0.1 * suf_med),
        "K2_suffix_material_pass": bool(suf_med > 0.05),
        "rows": rows,
    }
    (EXP / "gate_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n[gate] prefix median rel_l2 = {pre_med:.5f}  (K1 pass if <0.02 AND <0.1x suffix)")
    print(f"[gate] suffix median rel_l2 = {suf_med:.5f}  (K2 pass if >0.05)")
    print(f"[gate] K1_prefix_identity_pass = {stats['K1_prefix_identity_pass']}")
    print(f"[gate] K2_suffix_material_pass = {stats['K2_suffix_material_pass']}")
    print(f"[gate] wrote {EXP / 'gate_stats.json'}")


# --------------------------------------------------------------------------- build (bf16)
def _write_cond_clean(orig_pt: Path, dst_pt: Path, correct_suffix: bool, cls: str, stem: str,
                      vae, device: str) -> dict:
    base = torch.load(orig_pt, map_location="cpu", weights_only=True)
    lat = base["latents"]  # [C, F', H', W'] bf16
    result = {"clip": f"{cls}/{stem}", "corrected": correct_suffix}
    if correct_suffix:
        px = preprocess(_std_path(cls, stem))
        assert px.shape[0] == 121, f"{cls}/{stem}: expected 121 frames, got {px.shape[0]}"
        suf_cut = encode(px[-PX_SUFFIX:], vae, device, lat.dtype)  # [1,C,2,H,W] in lat.dtype
        new = lat.clone()
        new[:, -1, :, :] = suf_cut[0, :, -1, :, :].to(new.dtype).cpu()
        # bitwise-identical everywhere except the last latent frame (advisor gate)
        assert torch.equal(new[:, :-1], lat[:, :-1]), f"{cls}/{stem}: non-suffix frames changed!"
        assert not torch.equal(new[:, -1:], lat[:, -1:]), f"{cls}/{stem}: suffix frame unchanged?!"
        result["suffix_rel_l2_bf16"] = rel_l2(new[:, -1:], lat[:, -1:])
        base["latents"] = new
    dst_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, dst_pt)
    return result


def run_build(device: str) -> None:
    vae = load_video_vae_encoder(MODEL_PATH, device=device, dtype=torch.bfloat16)
    report = {"specialist": [], "ic3": []}

    # --- specialists: all clips two-sided -> all get suffix correction ---
    for cls, stem, orig_pt in specialist_clips():
        rel = orig_pt.relative_to(SP_062 / cls / "latents")
        dst = EXP / "dataset/specialist" / cls / "cond_clean_latents" / rel
        report["specialist"].append(_write_cond_clean(orig_pt, dst, True, cls, stem, vae, device))
    print(f"[build] specialist: wrote {len(report['specialist'])} cond_clean latents")

    # --- ic3: per-clip tree; two-sided classes corrected, one-sided copied bitwise ---
    ic, twosided = ic3_target_clips()
    for cls, stem, orig_pt in ic:
        dst = EXP / "dataset/ic3_clips/cond_clean" / cls / f"{stem}.pt"
        report["ic3"].append(_write_cond_clean(orig_pt, dst, cls in twosided, cls, stem, vae, device))
    n_corr = sum(1 for r in report["ic3"] if r["corrected"])
    print(f"[build] ic3: wrote {len(report['ic3'])} cond_clean clip latents "
          f"({n_corr} corrected / {len(report['ic3']) - n_corr} bitwise copies)")

    (EXP / "build_report.json").write_text(json.dumps(report, indent=2))
    print(f"[build] wrote {EXP / 'build_report.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "build"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-specialist", type=int, default=16)  # both classes, all clips
    ap.add_argument("--n-ic3", type=int, default=12)
    args = ap.parse_args()
    if args.mode == "gate":
        run_gate(args.n_specialist, args.n_ic3, args.device)
    else:
        run_build(args.device)


if __name__ == "__main__":
    main()
