"""ladder2 — training precompute (the only GPU work before training).

Two independent modes, both idempotent:

  --mode cond-clean   isolation-encode the conditioning anchors (the bleed fix). Writes
                      dataset/cond_clean/<class>/<clip>.pt for every training clip:
                      two-sided classes get the corrected suffix anchor, one-sided classes a
                      bitwise copy (so every root's file counts match exactly — the trainer
                      SILENTLY SKIPS samples whose sources disagree, so counts are the guard).
                      Token-independent: can run before the token verdict.

  --mode text         render every training caption with `prompts.render_for_clip()` and
                      encode it with the trainer's own process_captions.py. NOTHING is reused
                      from an earlier precompute: every prior conditions/ tree holds the LEAKY
                      caption ("... The scene transforms into ..."), and reusing one would
                      train the model on text the generator never supplies.

Video latents are NOT recomputed — they are prompt-agnostic and `inventory.py` already proved
all 139 training clips have one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
TRAINER = LAB / "LTX-2-cond-bleed-fix/packages/ltx-trainer"
VENV_PY = LAB / "LTX-2-official/.venv/bin/python"

STD = REPO_ROOT / "data/processed/transitions_std121"
DATASET = HERE.parent / "dataset"
COND_CLEAN = DATASET / "cond_clean"
CONDITIONS = DATASET / "conditions"
CAPTION_MANIFEST = DATASET / "captions" / "dataset_captions.json"
INVENTORY = HERE / "inventory.json"


def inventory() -> dict:
    if not INVENTORY.exists():
        sys.exit("[precompute] run train/inventory.py first")
    return json.loads(INVENTORY.read_text())


def token() -> str:
    import yaml

    return yaml.safe_load((HERE.parent / "arms.yaml").read_text())["token"]


# --------------------------------------------------------------------------- cond-clean
def run_cond_clean(device: str) -> None:
    import torch

    inv = inventory()
    sided = prompts.sidedness()
    clips = sorted({c for v in inv["clips"].values() for c in v})
    todo = [c for c in clips if not (COND_CLEAN / prompts.clip_class(c) / f"{c}.pt").exists()]
    n_two = sum(1 for c in clips if sided[prompts.clip_class(c)] == "two")
    print(f"[cond-clean] {len(clips)} training clips ({n_two} two-sided -> corrected, "
          f"{len(clips) - n_two} one-sided -> bitwise copy); {len(todo)} to write")
    if not todo:
        print("[cond-clean] nothing to do")
        return

    vae = ec.load_vae(str(MODEL), device=device, dtype=torch.bfloat16)
    report = []
    for clip in todo:
        cls = prompts.clip_class(clip)
        orig = REPO_ROOT / inv["latents"][clip]
        dst = COND_CLEAN / cls / f"{clip}.pt"
        report.append(ec.write_cond_clean(
            orig_pt=orig, dst_pt=dst, clip_mp4=STD / cls / f"{clip}.mp4",
            correct_suffix=sided[cls] == "two", vae=vae, device=device))
        tag = "corrected" if report[-1]["corrected"] else "copy"
        extra = f"  rel_l2={report[-1].get('suffix_rel_l2', 0):.4f}" if report[-1]["corrected"] else ""
        print(f"  {clip:32s} {tag}{extra}")
    (DATASET / "cond_clean_report.json").write_text(json.dumps(report, indent=2))
    corrected = sum(1 for r in report if r["corrected"])
    print(f"[cond-clean] wrote {len(report)} ({corrected} corrected) -> {COND_CLEAN}")


# ------------------------------------------------------------------------------- text
def build_caption_manifest() -> Path:
    """Render every training caption through the ONE renderer and write the trainer manifest."""
    inv = inventory()
    tok = token()
    clips = sorted({c for v in inv["clips"].values() for c in v})
    rows = []
    for clip in clips:
        cls = prompts.clip_class(clip)
        rows.append({"caption": prompts.render_for_clip(clip, tok), "video": f"{cls}/{clip}.mp4"})
    # the mp4s must resolve relative to the manifest's own directory (process_captions rule)
    CAPTION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    for cls in sorted({prompts.clip_class(c) for c in clips}):
        link = CAPTION_MANIFEST.parent / cls
        if not link.exists():
            link.symlink_to((STD / cls).resolve())
    CAPTION_MANIFEST.write_text(json.dumps(rows, indent=2))
    leaks = [r for r in rows if prompts.MARKER in r["caption"]]
    assert not leaks, f"{len(leaks)} training captions still carry the outcome marker"
    assert all(f" {tok}." in r["caption"] for r in rows), "a training caption lost the token"
    print(f"[text] rendered {len(rows)} training captions with token {tok!r} -> {CAPTION_MANIFEST}")
    print(f"       sample: {rows[0]['caption'][:110]}...")
    return CAPTION_MANIFEST


def run_text(device: str) -> None:
    manifest = build_caption_manifest()
    CONDITIONS.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PY), "scripts/process_captions.py", str(manifest),
        "--output-dir", str(CONDITIONS),
        "--model-path", str(MODEL),
        "--text-encoder-path", str(GEMMA),
        "--device", device,
        "--overwrite",  # stale embeddings from an earlier token would silently poison training
    ]
    print("[text] " + " ".join(cmd))
    subprocess.run(cmd, cwd=TRAINER, check=True, env={
        **__import__("os").environ, "PYTHONPATH": str(TRAINER / "src")})
    n = len(list(CONDITIONS.rglob("*.pt")))
    print(f"[text] wrote {n} text-embedding files -> {CONDITIONS}")
    (DATASET / "conditions_token.json").write_text(json.dumps({"token": token(), "n": n}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cond-clean", "text", "manifest-only"], required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.mode == "cond-clean":
        run_cond_clean(args.device)
    elif args.mode == "text":
        run_text(args.device)
    else:
        build_caption_manifest()


if __name__ == "__main__":
    main()
