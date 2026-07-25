"""ctt_v2 — VAE-encode the procedural strata into the ic_gen root format.

Two stages, both idempotent and both shardable across a Slurm array.

  --stage clips   Encode every DISTINCT rendered clip once into a flat cache. A clip is a
                  target in one sample and a reference in several others, so encoding
                  per-sample would redo the same work ~2x. Cache key is the clip id.
                  Also writes the isolation-encoded ('cond_clean') twin, because that is a
                  property of the clip, not of the pairing.

  --stage pairs   Per SAMPLE: link latents[target] / reference_latents[reference] /
                  cond_clean_latents[target] out of the cache, and write the mask.
                  CPU only. Masks are written, never linked: mask = f(conditioning).

The mask convention is copied exactly from eval_ladder/train/assemble_roots.py:
    m[:2] = 1  -> prefix anchor (2 latent frames)
    m[-1] = 1  -> suffix anchor, two-sided only

The isolation encode matters and is not optional: the LTX-2 video VAE is temporally CAUSAL,
so the last latent frame of a full-video encode reaches backwards into the middle of the clip
(measured suffix rel-L2 0.280, exp_073). Training on that frame conditions the model on middle
content it never receives at generation time.

Encoding goes through eval_ladder/encode_conditioning.py, which is the ONE definition of the
window in this campaign, and through the lineage trainer's VAE loader (the default in
`_trainer_bits`) so these latents are byte-consistent with the existing S0 latents.

Usage:
    # GPU, array-shardable
    python scripts/ctt_v2/mix/encode_mix.py --stage clips --manifest <m.json> --shard 0/8
    # CPU
    python scripts/ctt_v2/mix/encode_mix.py --stage pairs --manifest <m.json>

Manifest: {"samples": [{"sid","target","reference","sided","clips":{id: mp4path}}...]} --
see build_mix_manifest.py, which is what turns an S2/S3 delivery manifest into this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"

CACHE = REPO_ROOT / "eval_ladder/dataset/mix_cache"
LAT = CACHE / "latents"
CC = CACHE / "cond_clean"


def load_manifest(path: Path) -> dict:
    man = json.loads(path.read_text())
    if "samples" not in man or not man["samples"]:
        sys.exit(f"[encode] manifest has no samples: {path}")
    return man


def all_clips(man: dict) -> dict[str, Path]:
    """clip_id -> mp4 path, over every clip referenced as a target OR a reference."""
    clips: dict[str, Path] = {}
    for s in man["samples"]:
        for cid, p in s["clips"].items():
            prev = clips.get(cid)
            if prev is not None and str(prev) != str(p):
                sys.exit(f"[encode] clip id {cid!r} maps to two different files:\n  {prev}\n  {p}")
            clips[cid] = Path(p)
    return clips


def stage_clips(man: dict, shard: tuple[int, int], device: str) -> None:
    import torch

    import encode_conditioning as ec  # noqa: PLC0415

    clips = all_clips(man)
    ids = sorted(clips)
    i, n = shard
    mine = ids[i::n]
    todo = [c for c in mine if not (LAT / f"{c}.pt").exists() or not (CC / f"{c}.pt").exists()]
    print(f"[encode] clips total={len(ids)} shard={i}/{n} mine={len(mine)} todo={len(todo)}",
          flush=True)
    if not todo:
        print("[encode] nothing to do")
        return

    # Sidedness is a property of the SAMPLE, but cond_clean correction is a property of the
    # CLIP. A clip used two-sided anywhere needs the corrected suffix; correcting a clip only
    # used one-sided is harmless but wasteful, so take the union.
    two_sided_clips = {s["target"] for s in man["samples"] if s["sided"] == "two"}

    LAT.mkdir(parents=True, exist_ok=True)
    CC.mkdir(parents=True, exist_ok=True)
    vae = ec.load_vae(str(MODEL), device=device)

    for k, cid in enumerate(todo, 1):
        mp4 = clips[cid]
        if not mp4.exists():
            sys.exit(f"[encode] missing render: {mp4}")
        px = ec.preprocess(mp4)
        if px.shape[0] != ec.STD_FRAMES:
            sys.exit(f"[encode] {cid}: {px.shape[0]} frames, expected {ec.STD_FRAMES} — the "
                     f"root format is fixed at 121 frames")
        lat = ec.encode(px, vae, device, torch.bfloat16)[0].cpu()
        data = {"latents": lat, "num_frames": int(lat.shape[1]),
                "height": int(lat.shape[2]), "width": int(lat.shape[3]), "fps": 24.0}
        lat_pt = LAT / f"{cid}.pt"
        torch.save(data, lat_pt)
        ec.write_cond_clean(lat_pt, CC / f"{cid}.pt", mp4,
                            correct_suffix=cid in two_sided_clips, vae=vae, device=device)
        if k % 25 == 0 or k == len(todo):
            print(f"[encode] {k}/{len(todo)}", flush=True)
    print(f"[encode] shard {i}/{n} DONE")


def stage_pairs(man: dict, out_root: Path) -> None:
    import torch

    samples = man["samples"]
    missing = [s["sid"] for s in samples
               if not (LAT / f"{s['target']}.pt").exists()
               or not (LAT / f"{s['reference']}.pt").exists()
               or not (CC / f"{s['target']}.pt").exists()]
    if missing:
        sys.exit(f"[pairs] {len(missing)} samples have un-encoded clips (run --stage clips "
                 f"first): {missing[:5]}")

    def link(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    mask_cache: dict[tuple, "torch.Tensor"] = {}
    for s in samples:
        rel = f"{s['sid']}.pt"
        link(LAT / f"{s['target']}.pt", out_root / "latents" / rel)
        link(LAT / f"{s['reference']}.pt", out_root / "reference_latents" / rel)
        link(CC / f"{s['target']}.pt", out_root / "cond_clean_latents" / rel)

        t = torch.load(LAT / f"{s['target']}.pt", map_location="cpu", weights_only=True)
        f, h, w = int(t["num_frames"]), int(t["height"]), int(t["width"])
        key = (f, h, w, s["sided"])
        if key not in mask_cache:
            m = torch.zeros(f, h, w)
            m[:2] = 1.0                       # prefix anchor (2 latent frames)
            if s["sided"] == "two":
                m[-1] = 1.0                   # suffix anchor (last latent frame)
            mask_cache[key] = m
        mdst = out_root / "masks" / rel
        mdst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mask": mask_cache[key].clone()}, mdst)

    counts = {s: sum(1 for _ in (out_root / s).rglob("*.pt"))
              for s in ("latents", "reference_latents", "cond_clean_latents", "masks")}
    if len(set(counts.values())) != 1:
        sys.exit(f"[pairs] FATAL: counts disagree {counts}")
    print(f"[pairs] wrote {len(samples)} samples -> {out_root}")
    print(f"[pairs] counts {counts}  (conditions/ comes from the text stage)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["clips", "pairs"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None, help="stage=pairs: per-stratum staging dir")
    ap.add_argument("--shard", default="0/1", help="stage=clips: 'i/N' for a Slurm array")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    man = load_manifest(Path(args.manifest))
    if args.stage == "clips":
        i, n = (int(x) for x in args.shard.split("/"))
        stage_clips(man, (i, n), args.device)
    else:
        if not args.out:
            sys.exit("[encode] --out is required for --stage pairs")
        out = Path(args.out)
        stage_pairs(man, out if out.is_absolute() else REPO_ROOT / out)


if __name__ == "__main__":
    main()
