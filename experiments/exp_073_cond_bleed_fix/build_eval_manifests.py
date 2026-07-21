#!/usr/bin/env python
"""Build v4 scoring manifests for the exp_073 regenerated arms (fix/nullA/nullB).

Adapts the exp_066 EvalItemV3 rows (correct reference_video + condition_prefix/suffix, frozen)
by repointing generated_video to the exp_073 per-arm out-roots and setting arm. Item_ids are
kept IDENTICAL (arm-independent, encode seed) so fix/null pair exactly. Three item sets:
  - spec_two_sided : shadow_smoke + hero_flight, condition_suffix, ckpt2000  (F1 = 8 items x3 seeds)
  - ic3_two_sided  : ic3 rows with condition_suffix                          (F2 = 15 items x3 seeds)
  - ic3_control    : the pre-registered n=24 one-sided rows (control_draw.json) x3 seeds
Writes exp_073/dataset/eval/<set>_<arm>.json. Scoring happens after generation.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SRC = REPO / "experiments/exp_066_ladder_v3_scoring/dataset"
OUTD = EXP / "dataset/eval"
VID = "outputs/videos/exp_073_cond_bleed_fix"

SPEC_TWO_SIDED_CLASSES = {"shadow_smoke", "hero_flight"}


def load_src_rows() -> dict:
    """Merge all exp_066 eval_*.json into {item_id: row}."""
    rows = {}
    for f in sorted(SRC.glob("eval_*.json")):
        for r in json.loads(f.read_text()):
            rows.setdefault(r["item_id"], r)
    return rows


def spec_path(arm: str, iid: str) -> str:
    rung = iid.split("__", 1)[0]          # R2 / R3
    return f"{VID}/specialists/{arm}/{rung}/{iid}.mp4"


def ic3_path(arm: str, iid: str) -> str:
    rung = iid.split("__", 1)[0]          # R4A / R4B / R5 / R4X
    return f"{VID}/ic3/{arm}/{rung}/{iid}.mp4"


def write_manifest(name: str, arm: str, item_ids: list[str], src: dict, pathfn) -> None:
    rows = []
    for iid in sorted(item_ids):
        r = dict(src[iid])                 # copy frozen EvalItemV3 row
        r["generated_video"] = pathfn(arm, iid)
        r["arm"] = arm
        rows.append(r)
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"{name}_{arm}.json").write_text(json.dumps(rows, indent=2))
    print(f"  {name}_{arm}.json: {len(rows)} rows")


def main() -> None:
    src = load_src_rows()
    draw = json.loads((EXP / "control_draw.json").read_text())
    control_bases = set(draw["all24"])

    # ---- item sets (by item_id, which includes __s<seed>) ----
    spec_ts = [i for i, r in src.items()
               if r.get("style") in SPEC_TWO_SIDED_CLASSES and r.get("condition_suffix")
               and i.endswith("__ckpt2000")]
    ic3_ts = [i for i, r in src.items()
              if r.get("condition_suffix") and r.get("arm", "").startswith("ic3")]
    # control: exp_066 item_id = <base>__s<seed>; base membership in the 24
    import re
    def base_of(iid):  # strip trailing __s<seed>
        return re.sub(r"__s\d+$", "", iid)
    # base membership uniquely selects control rows (item_ids are unique per arm)
    ic3_control = [i for i in src if base_of(i) in control_bases]

    print(f"[sets] spec_two_sided item_ids={len(spec_ts)} (expect 8x3=24)")
    print(f"[sets] ic3_two_sided item_ids={len(ic3_ts)} (expect 15x3=45)")
    print(f"[sets] ic3_control item_ids={len(ic3_control)} (expect 24x3=72)")
    ss = [i for i in spec_ts if src[i]["style"] == "shadow_smoke"]

    print("[write] specialist two-sided:")
    for arm in ("fix", "nullA"):
        write_manifest("spec_two_sided", arm, spec_ts, src, spec_path)
    write_manifest("spec_two_sided", "nullB", ss, src, spec_path)   # nullB = shadow_smoke only
    print("[write] ic3 two-sided:")
    for arm in ("fix", "nullA", "nullB"):
        write_manifest("ic3_two_sided", arm, ic3_ts, src, ic3_path)
    print("[write] ic3 control (n=24):")
    for arm in ("fix", "nullA", "nullB"):
        write_manifest("ic3_control", arm, ic3_control, src, ic3_path)


if __name__ == "__main__":
    main()
