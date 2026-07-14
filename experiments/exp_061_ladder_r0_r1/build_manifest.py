"""exp_061 — build eval_manifest_r0.json + eval_manifest_r1.json.

Row contract = exp_060 eval_manifest.json (SPEC §2 EvalItemV3, score.py-
compatible). R0/R1 have no in-context demo, so reference_video = the item's
OWN ground-truth corpus clip (the clip whose middle the arm regenerates) —
noted per row; the sidedness re-annotation decides tomorrow how these arms are
scored. condition_prefix/suffix appear only in R1 rows and point at the FULL
item clip (num_frames 9 / 8) exactly like the certified sibling manifest
(score.py slices first-9 / last-8). R1 rows carry twin_of = the R0 item_id
(paired design). generated_video = ABSOLUTE main-checkout path.
"""

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
OUT_REL = "outputs/videos/exp_061_ladder_r0_r1"


def main() -> None:
    sel = json.loads((EXP / "dataset/selection.json").read_text())
    seeds = sel["seeds"]
    for arm in ("r0", "r1"):
        rows = []
        for it in sel["items"]:
            for seed in seeds:
                iid = f"{arm}__{it['class']}__{it['clip']}__s{seed}"
                row = {
                    "item_id": iid,
                    "generated_video": str(REPO_ROOT / OUT_REL / arm / f"{iid}.mp4"),
                    "reference_video": it["clip_rel"],
                    "style": it["class"],
                    "n_endpoints": 2 if arm == "r1" else 0,
                    "arm": arm,
                    "twin_of": (f"r0__{it['class']}__{it['clip']}__s{seed}"
                                if arm == "r1" else None),
                    "notes": (f"ladder {arm}; endpoints_clip={it['clip']} "
                              f"({it['source']}); reference=self (ground-truth "
                              f"clip, no demo in this arm); seed={seed}"),
                }
                if arm == "r1":
                    row["condition_prefix"] = {"video": it["clip_rel"], "num_frames": 9}
                    row["condition_suffix"] = {"video": it["clip_rel"], "num_frames": 8}
                rows.append(row)
        out = EXP / f"dataset/eval_manifest_{arm}.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"[done] {len(rows)} rows -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
