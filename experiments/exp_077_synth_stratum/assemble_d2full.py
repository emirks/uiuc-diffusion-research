"""exp_077 D2-FULL STAGE 4 (assemble + mix, fan-in) — ONE combined IC-LoRA training root.

Runs once after every encode shard (L40S). Adapted from the proven `assemble_d1.py`.

  1. TEXT (once): encode the ONE leak-free NEUTRAL prompt with Gemma, then reuse that embedding
     for every synthetic sample (all synthetic clips share the identical neutral prompt, so
     per-clip encoding would be 3,072 identical Gemma passes).
  2. ASSEMBLE the 3,072 synthetic samples (latents / conditions / reference_latents / masks /
     cond_clean_latents) as per-file symlinks, exactly like assemble_roots.py::assemble_generalist.
  3. MIX with the REAL corpus: eval_ladder's ic_gen root (385 real pairs) replicated x8 = 3,080,
     giving ~50/50 under uniform sampling, in ONE root, with the EQUAL-SOURCE-COUNT assert.
  4. Write the stage-4 section of D2_BUILD_AUDIT.json.

    PYTHONPATH=<trainer>/src <venv>/bin/python  (only for the Gemma subprocess) ; the driver runs
    under the research env.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIT = HERE / "D2_BUILD_AUDIT.json"

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
TRAINER = LAB / "LTX-2-cond-bleed-fix/packages/ltx-trainer"
VENV_PY = LAB / "LTX-2-official/.venv/bin/python"
REAL_ROOT = REPO_ROOT / "eval_ladder/dataset/roots/ic_gen"
SOURCES = ["latents", "conditions", "reference_latents", "masks", "cond_clean_latents"]


def sh(cmd, cwd, extra_env=None):
    print("[cmd]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd),
                   env={**os.environ, **(extra_env or {})}, check=True)


def link(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def merge_audit(update: dict) -> None:
    cur = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    cur.update(update)
    AUDIT.write_text(json.dumps(cur, indent=2))


def main() -> None:
    device = os.environ.get("DEVICE", "cuda")
    cfg = yaml.safe_load((HERE / "config_d2full.yaml").read_text())
    model, gemma = cfg["model"]["ltx_checkpoint"], cfg["model"]["gemma"]
    token = cfg["text"]["token"]
    neutral = f"{cfg['text']['neutral_s1']}. {token}. {cfg['text']['neutral_s2']}."
    replicas = cfg["train"]["real_replicas"]

    run = REPO_ROOT / cfg["outputs"]["dir"] / cfg["outputs"]["subdir"]
    ds = run / "dataset"
    latents_dir, cc_dir = ds / "latents", ds / "cond_clean"

    tuples = [json.loads(l) for f in sorted((run / "meta").glob("tuples_shard*.jsonl"))
              for l in f.read_text().splitlines() if l.strip()]
    n_exp = cfg["d2"]["n_target_pairs"] * cfg["d2"]["ops_per_target"]
    if len(tuples) != n_exp:
        sys.exit(f"[assemble] {len(tuples)} tuples != {n_exp} expected — render incomplete")
    print(f"[assemble] {len(tuples)} tuples  neutral={neutral!r}  real_replicas={replicas}")

    import torch

    # -- guard: every clip has a latent; every TARGET has a cond_clean --
    stems = sorted({s for t in tuples for s in (t["target_stem"], t["reference_stem"])})
    tgt_stems = sorted({t["target_stem"] for t in tuples})
    miss_lat = [s for s in stems if not (latents_dir / f"{s}.pt").exists()]
    miss_cc = [s for s in tgt_stems if not (cc_dir / f"{s}.pt").exists()]
    if miss_lat or miss_cc:
        sys.exit(f"[assemble] encode incomplete: {len(miss_lat)} latents / "
                 f"{len(miss_cc)} cond_clean missing (e.g. {(miss_lat + miss_cc)[:3]})")
    n_lat_files = sum(1 for _ in latents_dir.rglob("*.pt"))
    n_cc_files = sum(1 for _ in cc_dir.rglob("*.pt"))
    assert n_lat_files == len(stems), f"latent files {n_lat_files} != clips {len(stems)}"
    assert n_cc_files == len(tgt_stems), f"cond_clean files {n_cc_files} != targets {len(tgt_stems)}"
    print(f"[assemble] encode complete: {len(stems)} latents ({n_lat_files} files) + "
          f"{len(tgt_stems)} cond_clean ({n_cc_files} files)")

    # -- step 1: the ONE neutral text embedding --
    shared_dir = ds / "conditions_shared"
    shared_pt = shared_dir / "neutral.pt"
    if not shared_pt.exists():
        shared_dir.mkdir(parents=True, exist_ok=True)
        cap = shared_dir / "caption_manifest.json"
        cap.write_text(json.dumps([{"caption": neutral, "video": "neutral.mp4"}], indent=1))
        sh([VENV_PY, "scripts/process_captions.py", str(cap), "--output-dir", str(shared_dir),
            "--model-path", model, "--text-encoder-path", gemma, "--caption-column", "caption",
            "--media-column", "video", "--device", device, "--overwrite"],
           cwd=TRAINER, extra_env={"PYTHONPATH": str(TRAINER / "src")})
    if not shared_pt.exists():
        sys.exit(f"[assemble] neutral text embed not produced at {shared_pt}")
    print(f"[assemble] step1 OK: neutral text embed -> {shared_pt}")

    # -- steps 2+3: ONE combined root --
    root = ds / "roots" / "d2_combined"
    mask_cache: dict = {}
    for t in tuples:
        tid = t["tuple_id"]
        rel = f"synth/d2f_{tid:04d}.pt"
        link(latents_dir / f"{t['target_stem']}.pt", root / "latents" / rel)
        link(shared_pt, root / "conditions" / rel)
        link(latents_dir / f"{t['reference_stem']}.pt", root / "reference_latents" / rel)
        link(cc_dir / f"{t['target_stem']}.pt", root / "cond_clean_latents" / rel)
        td = torch.load(latents_dir / f"{t['target_stem']}.pt", map_location="cpu",
                        weights_only=True)
        key = (int(td["num_frames"]), int(td["height"]), int(td["width"]))
        if key not in mask_cache:
            m = torch.zeros(*key)
            m[:2] = 1.0        # prefix anchor (2 latent frames)
            m[-1] = 1.0        # suffix anchor (two-sided: both endpoints pinned)
            mask_cache[key] = m
        mdst = root / "masks" / rel
        mdst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mask": mask_cache[key].clone()}, mdst)
    n_synth = len(tuples)
    print(f"[assemble] step2 OK: {n_synth} synthetic samples -> {root}")

    # real x replicas
    real_rels = sorted(str(p.relative_to(REAL_ROOT / "latents"))
                       for p in (REAL_ROOT / "latents").rglob("*.pt"))
    if not real_rels:
        sys.exit(f"[assemble] real root empty: {REAL_ROOT}")
    dangling_real = [str(p) for s in SOURCES for p in (REAL_ROOT / s).rglob("*")
                     if p.is_symlink() and not p.exists()]
    if dangling_real:
        sys.exit(f"[assemble] real ic_gen root has {len(dangling_real)} DANGLING symlinks: "
                 f"{dangling_real[:3]}")
    real_counts = {s: sum(1 for _ in (REAL_ROOT / s).rglob("*.pt")) for s in SOURCES}
    if len(set(real_counts.values())) != 1:
        sys.exit(f"[assemble] real root source counts disagree {real_counts}")
    for r in range(replicas):
        for rel in real_rels:
            for s in SOURCES:
                link(REAL_ROOT / s / rel, root / s / f"real_rep{r}" / rel)
    n_real = len(real_rels) * replicas
    print(f"[assemble] step3 OK: {len(real_rels)} real pairs x{replicas} = {n_real} "
          f"(real root verified: {real_counts}, 0 dangling)")

    # -- equal-source-count assert (silent-drop guard) --
    counts = {s: sum(1 for _ in (root / s).rglob("*.pt")) for s in SOURCES}
    if len(set(counts.values())) != 1:
        sys.exit(f"[assemble] source counts DISAGREE {counts} — samples would be SILENTLY DROPPED")
    total = counts["latents"]
    assert total == n_synth + n_real, f"expected {n_synth + n_real}, got {total}"
    dangling = [str(p) for s in SOURCES for p in (root / s).rglob("*")
                if p.is_symlink() and not p.exists()]
    if dangling:
        sys.exit(f"[assemble] combined root has {len(dangling)} dangling symlinks")
    print(f"[assemble] combined root counts={counts} total={total} "
          f"real={n_real} synth={n_synth} mix_real={n_real/total:.4f}")

    merge_audit({"stage4_encode_assemble": {
        "n_clips_encoded": len(stems),
        "n_latents": n_lat_files,
        "n_cond_clean": n_cc_files,
        "encode_counts_reconciled": n_lat_files == len(stems) and n_cc_files == len(tgt_stems),
        "cond_clean_scope": "target clips only (the reference clip is consumed as reference_latents)",
        "n_synthetic_samples": n_synth,
        "n_real_pairs": len(real_rels), "real_replicas": replicas, "n_real_samples": n_real,
        "real_root": str(REAL_ROOT), "real_root_source_counts": real_counts,
        "real_root_dangling_symlinks": 0,
        "combined_root": str(root),
        "combined_root_source_counts": counts,
        "combined_root_counts_equal": True,
        "combined_root_total": total,
        "combined_root_dangling_symlinks": 0,
        "mix_ratio_real": round(n_real / total, 4),
        "mix_ratio_synth": round(n_synth / total, 4),
        "neutral_prompt": neutral, "token": token,
        "text_embedding": "ONE Gemma pass, shared by all synthetic samples (identical prompt)",
    }})
    print(f"[assemble] DONE -> {root}")


if __name__ == "__main__":
    main()
