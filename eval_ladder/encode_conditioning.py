"""ladder2 — the ONE definition of a conditioning window (training AND inference).

Seatbelt #6. The old ladder's worst silent defect was that training and inference disagreed
about what the suffix conditioning token *is*:

* inference built the suffix from a STANDALONE encode of the trailing 9 pixel frames;
* training pinned the LAST latent frame of the FULL-video encode — and the LTX-2 video VAE
  is temporally causal, so that frame reaches backwards into the middle of the clip.

Measured (exp_073, fp32, 28 clips): suffix rel-L2(standalone, full-slice) median **0.280** —
training was conditioning on middle content the model never gets at generation time. The
prefix is clean by causality (median rel-L2 **8.3e-5**), so only the suffix needs correcting
and one-sided clips are untouched.

This module therefore owns:

1. the window rule            — `PX_PREFIX` / `PX_SUFFIX` / `SUFFIX_GEN_FRAMES`
2. inference-side conditioning — `cond_paths()` over `conds/<clip>_{start9,end9}.mp4`
3. training-side conditioning  — `write_cond_clean()`, isolation-encoded, bitwise-asserted
4. `smoke_assert()`            — run at train start; fails loudly if a root's cond_clean tree
                                 is not the isolation encode.

Nothing else in ladder2 may define a window, cut a clip, or pin a latent frame.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STD = REPO_ROOT / "data/processed/transitions_std121"
CONDS = Path(__file__).resolve().parent / "conds"

#: pixel frames taken as the prefix window (-> 2 latent frames; the first is the anchor)
PX_PREFIX = 9
#: pixel frames taken as the suffix window (-> 2 latent frames; the LAST is the anchor)
PX_SUFFIX = 9
#: frames the generator consumes from the suffix clip (causal-VAE rule: 9 cut, 8 consumed)
SUFFIX_GEN_FRAMES = 8
#: total frames in a standardised clip
STD_FRAMES = 121

_FFMPEG = shutil.which("ffmpeg") or str(Path.home() / ".local/bin/ffmpeg")

# ffmpeg selectors — the SAME frame ranges the latent path slices, expressed for mp4 cutting
_SELECT_PREFIX = f"select='lt(n,{PX_PREFIX})'"
_SELECT_SUFFIX = f"select='gte(n,{STD_FRAMES - PX_SUFFIX})'"


# ----------------------------------------------------------------- inference-side windows
def cond_paths(clip: str, sided: str) -> dict[str, Path]:
    """Conditioning mp4s for a generation row. `sided` in {'one','two'}."""
    out = {"prefix": CONDS / f"{clip}_start9.mp4"}
    if sided == "two":
        out["suffix"] = CONDS / f"{clip}_end9.mp4"
    for role, path in out.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {role} window for {clip}: {path} (run cut_windows)")
    return out


def cut_windows(src_mp4: Path, clip: str, out_dir: Path = CONDS) -> int:
    """Cut both windows for one clip (idempotent). Returns the number of files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for name, sel in ((f"{clip}_start9.mp4", _SELECT_PREFIX), (f"{clip}_end9.mp4", _SELECT_SUFFIX)):
        dst = out_dir / name
        if dst.exists():
            continue
        subprocess.run(
            [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src_mp4),
             "-vf", f"{sel},setpts=N/24/TB", "-r", "24", "-c:v", "libx264",
             "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)],
            check=True)
        made += 1
    return made


# ------------------------------------------------------------------ training-side windows
# These need the trainer package (VAE + video reader); imported lazily so the inference-side
# helpers above stay usable in the plain research env.
def _trainer_bits():
    import sys

    tr = os.environ.get(
        "LTX_TRAINER_SRC",
        "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-cond-bleed-fix/packages/ltx-trainer/src",
    )
    if tr not in sys.path:
        sys.path.insert(0, tr)
    from ltx_trainer.model_loader import load_video_vae_encoder  # noqa: PLC0415
    from ltx_trainer.video_utils import read_video  # noqa: PLC0415

    return load_video_vae_encoder, read_video


def load_vae(model_path: str, device: str = "cuda", dtype=None):
    import torch

    load_video_vae_encoder, _ = _trainer_bits()
    return load_video_vae_encoder(model_path, device=device, dtype=dtype or torch.bfloat16)


def preprocess(clip_mp4: Path):
    """std mp4 -> pixel tensor [F,C,H,W] in [0,1] (the exact path process_videos.py used)."""
    _, read_video = _trainer_bits()
    video, _ = read_video(clip_mp4, max_frames=STD_FRAMES)
    return video


def encode(pixel_fchw, vae, device: str, dtype):
    """[F,C,H,W] in [0,1] -> latents [1,C',F',H',W'] (clamp01, *2-1 == Normalize(0.5,0.5))."""
    import torch

    v = pixel_fchw.clamp(0.0, 1.0).mul(2.0).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)
    v = v.to(device=device, dtype=dtype)
    with torch.inference_mode():
        return vae(v)


def rel_l2(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


def write_cond_clean(orig_pt: Path, dst_pt: Path, clip_mp4: Path, correct_suffix: bool,
                     vae, device: str = "cuda") -> dict:
    """Write the isolation-encoded ('cond_clean') latents for one clip.

    Two-sided clip -> copy of the original latents with ONLY the last latent frame replaced
    by the last frame of a standalone encode of the trailing `PX_SUFFIX` pixel frames.
    One-sided clip -> bitwise copy (nothing to correct; the tree stays shape-complete so the
    trainer's file globbing matches the latents tree exactly).
    """
    import torch

    base = torch.load(orig_pt, map_location="cpu", weights_only=True)
    lat = base["latents"]  # [C, F', H', W']
    out = {"clip": clip_mp4.stem, "corrected": bool(correct_suffix)}
    if correct_suffix:
        px = preprocess(clip_mp4)
        assert px.shape[0] == STD_FRAMES, f"{clip_mp4.stem}: {px.shape[0]} frames, want {STD_FRAMES}"
        suffix_cut = encode(px[-PX_SUFFIX:], vae, device, lat.dtype)  # [1,C,2,H,W]
        new = lat.clone()
        new[:, -1, :, :] = suffix_cut[0, :, -1, :, :].to(new.dtype).cpu()
        assert torch.equal(new[:, :-1], lat[:, :-1]), f"{clip_mp4.stem}: non-suffix frames changed"
        assert not torch.equal(new[:, -1:], lat[:, -1:]), f"{clip_mp4.stem}: suffix frame unchanged"
        out["suffix_rel_l2"] = rel_l2(new[:, -1:], lat[:, -1:])
        base["latents"] = new
    dst_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, dst_pt)
    return out


def smoke_assert(root: Path, n_per_side: int = 2) -> str:
    """Train-start check: does this root's cond_clean tree really carry the isolation encode?

    The expectation is PER SAMPLE, not per root: the generalist's tree legitimately mixes
    corrected two-sided clips with bitwise one-sided copies (the copies exist so the trainer's
    per-relative-path source matching stays complete). A two-sided sample must differ from its
    latents in exactly the last latent frame and nowhere else; a one-sided sample must be
    bitwise identical. Sidedness comes from the clip's class, read out of the frozen split.
    """
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prompts  # noqa: PLC0415

    cc_dir, lat_dir = root / "cond_clean_latents", root / "latents"
    if not cc_dir.exists():
        return f"[smoke] {root.name}: no cond_clean tree (one-sided specialist root) — OK"

    sided = prompts.sidedness()
    seen = {"one": 0, "two": 0}
    for cc in sorted(cc_dir.rglob("*.pt")):
        cls = cc.relative_to(cc_dir).parts[0]
        want = sided[cls]
        if seen[want] >= n_per_side:
            continue
        a = torch.load(cc, map_location="cpu", weights_only=True)["latents"]
        b = torch.load(lat_dir / cc.relative_to(cc_dir), map_location="cpu",
                       weights_only=True)["latents"]
        assert torch.equal(a[:, :-1], b[:, :-1]), f"{cc.name}: non-suffix frames differ"
        differs = not torch.equal(a[:, -1:], b[:, -1:])
        assert differs == (want == "two"), (
            f"{cc.name} ({cls}, {want}-sided): suffix anchor "
            f"{'differs' if differs else 'is identical'} — the bleed fix is not in effect")
        seen[want] += 1
        if all(v >= n_per_side for v in seen.values()):
            break
    if not any(seen.values()):
        raise AssertionError(f"{root.name}: cond_clean tree present but no sample could be checked")
    return (f"[smoke] {root.name}: cond_clean verified — {seen['two']} two-sided (suffix anchor "
            f"corrected) + {seen['one']} one-sided (bitwise identical)")
