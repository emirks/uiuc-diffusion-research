"""exp_060 — build the 60-row σ_seed eval manifest + probe-group map.

12 probe items x 5 seeds, adapter arm only. Row contract = SPEC §2 eval
manifest. Conditions point at the FULL clip A (score.py slices first-9 /
last-8), exactly as the certified sibling manifest does — the generation was
conditioned on clip A's start9/end9, so this is the matching scoring contract.

`probe_group` is NOT a valid EvalItemV3 field (manifests_v3.load_eval_manifest
rejects unknown keys, SPEC §2), so it is emitted to a SEPARATE map
(dataset/probe_groups.json, item_id -> class) that compute_sigma.py joins onto
items.jsonl before calling seeds.sigma_seed. score.py is never modified.

generated_video = ABSOLUTE path into the MAIN checkout outputs (scoring runs
from the certified worktree; the videos live here). Corpus clips = repo-
relative (resolve under the worktree's data/ symlinks).
"""

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
CORPUS_REL = "data/processed/transitions_std121"
OUT_REL = "outputs/videos/exp_060_sigma_seed/adapter"
SEEDS = [42, 43, 44, 45, 46]


def clip_rel(stem: str) -> str:
    cls = "_".join(stem.split("_")[:-1])
    return f"{CORPUS_REL}/{cls}/{stem}.mp4"


def main() -> None:
    sel = json.loads((EXP / "dataset/selection.json").read_text())
    manifest, groups = [], {}
    for it in sel["items"]:
        cls = it["class"]
        a, b = it["clip_a"], it["clip_b"]
        a_rel = clip_rel(a)
        for seed in SEEDS:
            item_id = f"sigseed__{cls}__s{seed}"
            gen_abs = str(REPO_ROOT / OUT_REL / f"{item_id}.mp4")
            manifest.append({
                "item_id": item_id,
                "generated_video": gen_abs,
                "reference_video": clip_rel(b),
                "style": cls,
                "n_endpoints": 2,
                "condition_prefix": {"video": a_rel, "num_frames": 9},
                "condition_suffix": {"video": a_rel, "num_frames": 8},
                "arm": "adapter",
                "twin_of": None,
                "notes": f"sigma_seed; cond=A:{a} ref=B:{b}; seed={seed}; {it['stratum']}",
            })
            groups[item_id] = cls
    (EXP / "dataset/eval_manifest.json").write_text(json.dumps(manifest, indent=2))
    (EXP / "dataset/probe_groups.json").write_text(json.dumps(groups, indent=2))
    print(f"[done] {len(manifest)} rows -> dataset/eval_manifest.json "
          f"({len(groups)} item_ids, {len(set(groups.values()))} probe_groups)")


if __name__ == "__main__":
    main()
