"""exp_077 D1 STAGE 2 (assemble + mix) — fan-in, runs ONCE after all encode shards (L40S).

  1. TEXT (once): encode the ONE leak-free neutral prompt with Gemma via process_captions, then
     reuse that embedding for every synthetic clip (all share the identical neutral prompt).
  2. ASSEMBLE the synthetic ic_gen-style samples (latents / conditions / reference_latents / masks
     / cond_clean_latents), per-file symlinks, exactly like assemble_roots.py::assemble_generalist.
  3. MIX with the REAL corpus: the existing generalist's 385 real pairs (eval_ladder ic_gen root)
     replicated x8 (~3080) + the 3072 synthetic -> ONE combined root, with the EQUAL-SOURCE-COUNT
     assert (silent-drop guard) -> uniform sampling => ~50/50.
  4. AGGREGATE the render-side audit (gate rejection rate, path-entropy histogram, actual
     per-shader / per-aux-family allocation from the rendered metadata) + combined-root counts +
     mix ratio into D1_BUILD_AUDIT.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIT = HERE / "D1_BUILD_AUDIT.json"

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
    cfg = yaml.safe_load((HERE / "config_d1.yaml").read_text())
    model = cfg["model"]["ltx_checkpoint"]
    gemma = cfg["model"]["gemma"]
    token = cfg["text"]["token"]
    neutral = f"{cfg['text']['neutral_s1']}. {token}. {cfg['text']['neutral_s2']}."
    replicas = cfg["train"]["real_replicas"]

    run = REPO_ROOT / cfg["outputs"]["dir"] / "d1"
    videos = run / "videos"
    ds = run / "dataset"
    latents_dir, cc_dir = ds / "latents", ds / "cond_clean"
    plan = json.loads((HERE / "d1_plan.json").read_text())
    tuples = plan["tuples"]
    print(f"[assemble] {len(tuples)} tuples  neutral={neutral!r}  real_replicas={replicas}")

    import torch

    # -- guard: every clip must have BOTH a latent and a cond_clean (no silent encode drop) --
    stems = set()
    for f in sorted((run / "dataset_manifests").glob("clips_shard*.json")):
        for row in json.loads(f.read_text()):
            stems.add(row["stem"])
    stems = sorted(stems)
    miss_lat = [s for s in stems if not (latents_dir / f"{s}.pt").exists()]
    miss_cc = [s for s in stems if not (cc_dir / f"{s}.pt").exists()]
    if miss_lat or miss_cc:
        sys.exit(f"[assemble] encode incomplete: {len(miss_lat)} latents / {len(miss_cc)} cond_clean missing")
    print(f"[assemble] encode complete: {len(stems)} latents + {len(stems)} cond_clean")

    # -- step 1: neutral text embedding, encoded ONCE, reused for every clip --
    shared_dir = ds / "conditions_shared"
    shared_pt = shared_dir / "neutral.pt"
    if not shared_pt.exists():
        shared_dir.mkdir(parents=True, exist_ok=True)
        cap_man = shared_dir / "caption_manifest.json"
        cap_man.write_text(json.dumps([{"caption": neutral, "video": "neutral.mp4"}], indent=2))
        sh([VENV_PY, "scripts/process_captions.py", str(cap_man),
            "--output-dir", str(shared_dir), "--model-path", model, "--text-encoder-path", gemma,
            "--caption-column", "caption", "--media-column", "video", "--device", device, "--overwrite"],
           cwd=TRAINER, extra_env={"PYTHONPATH": str(TRAINER / "src")})
    if not shared_pt.exists():
        sys.exit(f"[assemble] neutral text embed not produced at {shared_pt}")
    print(f"[assemble] step1 OK: neutral text embed -> {shared_pt}")

    # -- steps 2+3: build ONE combined root (synthetic samples + real x8) --
    root = ds / "roots" / "d1_combined"
    mask_cache: dict = {}
    for t in tuples:
        tid = t["tuple_id"]
        stem_t, stem_r = f"tup{tid:04d}_tgt", f"tup{tid:04d}_ref"
        rel = f"synth/tup{tid:04d}.pt"
        link(latents_dir / f"{stem_t}.pt", root / "latents" / rel)
        link(shared_pt, root / "conditions" / rel)
        link(latents_dir / f"{stem_r}.pt", root / "reference_latents" / rel)
        link(cc_dir / f"{stem_t}.pt", root / "cond_clean_latents" / rel)
        td = torch.load(latents_dir / f"{stem_t}.pt", map_location="cpu", weights_only=True)
        f_, h_, w_ = int(td["num_frames"]), int(td["height"]), int(td["width"])
        key = (f_, h_, w_)
        if key not in mask_cache:
            m = torch.zeros(f_, h_, w_)
            m[:2] = 1.0        # prefix anchor (2 latent frames)
            m[-1] = 1.0        # suffix anchor (two-sided: both endpoints pinned)
            mask_cache[key] = m
        mdst = root / "masks" / rel
        mdst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mask": mask_cache[key].clone()}, mdst)
    n_synth = len(tuples)
    print(f"[assemble] step2 OK: {n_synth} synthetic samples linked into {root}")

    # real x replicas (upweight the 385 real pairs 8x to balance the synthetic)
    real_rels = sorted(str(p.relative_to(REAL_ROOT / "latents"))
                       for p in (REAL_ROOT / "latents").rglob("*.pt"))
    if not real_rels:
        sys.exit(f"[assemble] real root empty: {REAL_ROOT}")
    for r in range(replicas):
        for rel in real_rels:
            for s in SOURCES:
                link(REAL_ROOT / s / rel, root / s / f"real_rep{r}" / rel)
    n_real = len(real_rels) * replicas
    print(f"[assemble] step3 OK: {len(real_rels)} real pairs x{replicas} = {n_real} real samples")

    # -- equal-source-count assert (silent-drop guard) --
    counts = {s: sum(1 for _ in (root / s).rglob("*.pt")) for s in SOURCES}
    if len(set(counts.values())) != 1:
        sys.exit(f"[assemble] source counts DISAGREE {counts} — samples would be SILENTLY DROPPED")
    total = counts["latents"]
    assert total == n_synth + n_real, f"expected {n_synth + n_real}, got {total}"
    mix_ratio = round(n_real / total, 4)
    print(f"[assemble] combined root counts={counts}  total={total}  "
          f"real={n_real} synth={n_synth}  mix real={mix_ratio:.3f}")

    # -- step 4: aggregate the render-side audit from the rendered metadata --
    metas = [json.loads(p.read_text()) for p in sorted((run / "meta").glob("tuple_*.json"))]
    per_shader = Counter(m["operator"]["shader"] for m in metas)
    per_family = Counter(f"{m['operator']['shader']}/{m['operator']['aux_kind']}"
                         for m in metas if m["operator"]["aux_kind"])
    n_aux = sum(1 for m in metas if m["operator"]["aux_kind"])
    # path entropy: ops-per-target-pair + distinct shaders per target pair
    by_target: dict[int, list] = {}
    for m in metas:
        by_target.setdefault(m["target_index"], []).append(m["operator"]["shader"])
    ops_hist = Counter(len(v) for v in by_target.values())
    distinct_hist = Counter(len(set(v)) for v in by_target.values())
    mixed_ok = all(0 < sum(1 for m in metas if m["target_index"] == ti and m["operator"]["aux_kind"]) < 8
                   for ti in by_target)
    # gate rejection rate across all render shards
    tries = rej = 0
    for f in sorted((run / "render_stats").glob("stats_shard*.json")):
        st = json.loads(f.read_text())
        tries += st.get("tries_total", 0)
        rej += st.get("rejections", 0)
    max_ep = max((max(m["target"]["rendered_endpoint_mae"].values()) for m in metas), default=None)

    merge_audit({
        "stage2_encode_assemble": {
            "n_clips_encoded": len(stems),
            "n_synthetic_samples": n_synth,
            "n_real_samples": n_real,
            "real_pairs": len(real_rels),
            "real_replicas": replicas,
            "combined_root": str(root),
            "combined_root_source_counts": counts,
            "combined_root_counts_equal": len(set(counts.values())) == 1,
            "combined_root_total": total,
            "mix_ratio_real": mix_ratio,
            "mix_ratio_synth": round(n_synth / total, 4),
            "neutral_prompt": neutral,
        },
        "stage1_render_audit": {
            "n_tuples_rendered": len(metas),
            "aux_fraction_actual": round(n_aux / len(metas), 4) if metas else None,
            "per_shader_allocation_actual": dict(sorted(per_shader.items())),
            "per_aux_family_allocation_actual": dict(sorted(per_family.items())),
            "ops_per_target_pair_hist": dict(ops_hist),
            "distinct_shaders_per_target_hist": dict(sorted(distinct_hist.items())),
            "all_targets_have_8_ops": set(ops_hist) == {8} and sum(ops_hist.values()) == cfg["d1"]["n_target_pairs"],
            "all_targets_ge6_distinct_shaders": min(distinct_hist) >= cfg["d1"]["min_distinct_shaders"] if distinct_hist else False,
            "all_targets_mixed_aux": mixed_ok,
            "endpoint_identity_gate_rejection_rate": round(rej / tries, 5) if tries else None,
            "gate_total_draws": tries,
            "gate_rejections": rej,
            "max_rendered_endpoint_mae": round(max_ep, 5) if max_ep is not None else None,
            "holdout_shaders_absent_from_data": not (set(plan["holdout_shaders"]) & set(per_shader)),
            "holdout_shader_list": plan["holdout_shaders"],
        },
    })
    print(f"[assemble] audit updated: aux_actual={round(n_aux/len(metas),3)} "
          f"gate_rej={round(rej/tries,4) if tries else None} distinct_hist={dict(distinct_hist)}")
    print(f"[assemble] DONE -> {root}")


if __name__ == "__main__":
    main()
