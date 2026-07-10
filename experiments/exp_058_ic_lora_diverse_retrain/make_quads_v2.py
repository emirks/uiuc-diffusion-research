"""exp_058 — assemble the v2 eval set (design.md §6).

Items to GENERATE (dataset/quads_v2.json):
  1. suite rerun  — the 40 exp_057 IC quads verbatim (same endpoints/refs/
     prompts/cond cuts/seed), regenerated under the v2 adapter. exp_057's 11
     base twins are adapter-independent and are NOT regenerated.
  2. ic2_prefixonly — prefix-only variants (suffix DROPPED) of in-class quads:
     held-out hero_flight x2 / illustration_scene x2 / gas_transformation x2
     + trained-class portal x1 / shadow x1  (ids prefixed pfx__).
  3. base_prefixonly — 2 base twins of the above (hero_flight, gas).
  4. ic2_raven — held-out two-sided raven_transition in-class x2 + 1 base twin
     (new cond cuts in dataset/cond_q2/, captions reused from exp_056).

Also writes dataset/manifest_ic_v2.json for the scorer (all items above;
prefix-only items omit condition_suffix — EvalItem allows None).
"""

import json
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).parent
E57 = REPO_ROOT / "experiments/exp_057_ic_lora_unseen_eval"
STD = REPO_ROOT / "data/processed/transitions_std121"
OUT_VID = "outputs/videos/exp_058_ic_lora_diverse_retrain"
FFMPEG = (
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv/lib/"
    "python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)

PFX_PICKS = [  # exp_057 in-class quad ids -> prefix-only variants
    "ic_os_inclass__hero_flight_2__ref_hero_flight_4",
    "ic_os_inclass__hero_flight_6__ref_hero_flight_0",
    "ic_os_inclass__illustration_scene_4__ref_illustration_scene_2",
    "ic_os_inclass__illustration_scene_7__ref_illustration_scene_5",
    "ic_os_inclass__gas_transformation_2__ref_gas_transformation_6",
    "ic_os_inclass__gas_transformation_7__ref_gas_transformation_3",
    "ic_os_inclass__portal_12__ref_portal_5",
    "ic_os_inclass__shadow_2__ref_shadow_0",
]
BASE_PFX_PICKS = [
    "ic_os_inclass__hero_flight_2__ref_hero_flight_4",
    "ic_os_inclass__gas_transformation_2__ref_gas_transformation_6",
]
RAVEN_PAIRS = [("raven_transition_0", "raven_transition_1"),
               ("raven_transition_1", "raven_transition_2")]
RAVEN_BASE = [("raven_transition_0", "raven_transition_1")]


def cut(stem: str, cls: str, cond_dir: pathlib.Path) -> None:
    src = STD / cls / f"{stem}.mp4"
    for name, vf in [(f"{stem}_start9.mp4", "select='lt(n,9)'"),
                     (f"{stem}_end9.mp4", "select='gte(n,112)'")]:
        dst = cond_dir / name
        if dst.exists():
            continue
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-vf", vf + ",setpts=N/24/TB", "-r", "24",
             "-c:v", "libx264", "-preset", "slow", "-crf", "12",
             "-pix_fmt", "yuv420p", str(dst)], check=True)
        print(f"[cut] {name}")


def main() -> None:
    quads57 = {q["id"]: q for q in json.loads((E57 / "dataset/quads.json").read_text())}
    caps56 = json.loads(
        (REPO_ROOT / "experiments/exp_056_ltx2_ic_lora_transition_transfer/dataset/captions.json").read_text())
    cond_q57 = "experiments/exp_057_ic_lora_unseen_eval/dataset/cond_q"
    cond_q2_dir = EXP / "dataset/cond_q2"
    cond_q2_dir.mkdir(parents=True, exist_ok=True)
    cond_q2 = "experiments/exp_058_ic_lora_diverse_retrain/dataset/cond_q2"

    items = []  # generation items
    manifest = []  # scorer items

    def add(qid, label, arm, endpoints, ep_class, ref, ref_class, prompt,
            cond_dir, prefix_only):
        gen_rel = f"{OUT_VID}/{label}/quads/{qid}.mp4"
        items.append({
            "id": qid, "label": label, "arm": arm,
            "endpoints": endpoints, "endpoints_class": ep_class,
            "reference": ref, "reference_class": ref_class,
            "prompt": prompt, "cond_dir": cond_dir, "prefix_only": prefix_only,
        })
        entry = {
            "item_id": qid, "generated_video": gen_rel, "style": ref_class,
            "arm": arm,  # EvalItem field; without it report.md/viewer collapse to one blank arm
            "notes": f"endpoints={endpoints} ({ep_class}); reference={ref} ({ref_class})"
                     + ("; PREFIX-ONLY (no end-frame conditioning)" if prefix_only else ""),
            "n_endpoints": 1 if prefix_only else 2,
            "condition_prefix": {"video": f"{cond_dir}/{endpoints}_start9.mp4",
                                  "num_frames": 9},
            "condition_reference": {"video": f"data/processed/transitions_std121/{ref_class}/{ref}.mp4"},
        }
        if not prefix_only:
            entry["condition_suffix"] = {"video": f"{cond_dir}/{endpoints}_end9.mp4",
                                          "num_frames": 8}
        manifest.append(entry)

    # 1. suite rerun: 40 exp_057 IC quads under the v2 adapter
    n_suite = 0
    for q in quads57.values():
        if q["arm"].startswith("base_"):
            continue
        add(q["id"], "ic2", q["arm"], q["endpoints"], q["endpoints_class"],
            q["reference"], q["reference_class"], q["prompt"], cond_q57, False)
        n_suite += 1

    # 2./3. prefix-only variants
    for qid in PFX_PICKS:
        q = quads57[qid]
        add("pfx__" + qid, "ic2", "ic2_prefixonly", q["endpoints"],
            q["endpoints_class"], q["reference"], q["reference_class"],
            q["prompt"], cond_q57, True)
    for qid in BASE_PFX_PICKS:
        q = quads57[qid]
        add("pfx_base__" + qid, "base", "base_prefixonly", q["endpoints"],
            q["endpoints_class"], q["reference"], q["reference_class"],
            q["prompt"], cond_q57, True)

    # 4. raven (held-out two-sided): cuts + items
    for ep, _ in RAVEN_PAIRS:
        cut(ep, "raven_transition", cond_q2_dir)
    for ep, ref in RAVEN_PAIRS:
        prompt = "ICTRANS " + caps56[ep]
        add(f"raven__{ep}__ref_{ref}", "ic2", "ic2_ts_heldout", ep,
            "raven_transition", ref, "raven_transition", prompt, cond_q2, False)
    for ep, ref in RAVEN_BASE:
        prompt = "ICTRANS " + caps56[ep]
        add(f"raven_base__{ep}__ref_{ref}", "base", "base_ts_heldout", ep,
            "raven_transition", ref, "raven_transition", prompt, cond_q2, False)

    (EXP / "dataset/quads_v2.json").write_text(json.dumps(items, indent=1))
    (EXP / "dataset/manifest_ic_v2.json").write_text(json.dumps(manifest, indent=1))
    n_ic = sum(1 for i in items if i["label"] == "ic2")
    n_base = len(items) - n_ic
    print(f"[done] {len(items)} items ({n_suite} suite rerun + "
          f"{len(PFX_PICKS)} pfx + {len(RAVEN_PAIRS)} raven | ic2={n_ic}, "
          f"base={n_base}) -> quads_v2.json / manifest_ic_v2.json")
    # sanity: every cond video must exist
    missing = []
    for m in manifest:
        for k in ("condition_prefix", "condition_suffix", "condition_reference"):
            if k in m and not (REPO_ROOT / m[k]["video"]).exists():
                missing.append(m[k]["video"])
    assert not missing, f"missing cond videos: {missing[:5]}"
    print("[ok] all condition videos present")


if __name__ == "__main__":
    main()
