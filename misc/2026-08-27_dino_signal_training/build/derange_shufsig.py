#!/usr/bin/env python3
"""Add the frozen `signal_source` derangement column to the shufsig registry (PROTOCOL_LOCKED §28).

The pixel `reference` is left UNCHANGED (do NOT reuse `code_source_reference` — that swaps the fed
clip, the wrong control). Only a new `signal_source` column is added, naming the DERANGED clip whose
DINO signal is fed: rotation-by-2 over the 36 lexicographically-sorted unique refs (class-clean;
rotation-1 collides). run_gen.py's SIGNAL_CONFIG hook reads `signal_source` and stamps
signal_id = eval__<signal_source>, while pool_refs still excludes the row's own `reference` — so the
matched and shufsig arms are scored against byte-identical pools and differ only in the fed signal.

The map below is copied VERBATIM from PROTOCOL_LOCKED.md §28 and additionally asserted to equal
rotation-by-2 over the sorted unique refs. Deterministic; safe to re-run (idempotent).

  python derange_shufsig.py <registry.jsonl>   # rewrites the file in place with signal_source added
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Verbatim frozen map from PROTOCOL_LOCKED.md §28 (rotation-by-2 over the 36 sorted unique refs).
FROZEN_MAP: dict[str, str] = {
    "animalization_0": "color_rain_0", "animalization_1": "color_rain_1",
    "color_rain_0": "cotton_cloud_1", "color_rain_1": "display_transition_1",
    "cotton_cloud_1": "earth_element_0", "display_transition_1": "earth_element_4",
    "earth_element_0": "firelava_0", "earth_element_4": "flying_cam_transition_1",
    "firelava_0": "gas_transformation_0", "flying_cam_transition_1": "gas_transformation_6",
    "gas_transformation_0": "hero_flight_0", "gas_transformation_6": "hero_flight_5",
    "hero_flight_0": "illustration_scene_0", "hero_flight_5": "illustration_scene_4",
    "illustration_scene_0": "live_concert_1", "illustration_scene_4": "luminous_gaze_0",
    "live_concert_1": "melt_transition_1", "luminous_gaze_0": "money_rain_0",
    "melt_transition_1": "money_rain_1", "money_rain_0": "monstrosity_0",
    "money_rain_1": "polygon_0", "monstrosity_0": "polygon_4",
    "polygon_0": "portal_0", "polygon_4": "portal_1",
    "portal_0": "raven_transition_0", "portal_1": "saint_glow_0",
    "raven_transition_0": "shadow_0", "saint_glow_0": "shadow_10",
    "shadow_0": "shadow_smoke_0", "shadow_10": "shadow_smoke_1",
    "shadow_smoke_0": "super_fast_run_0", "shadow_smoke_1": "super_fast_run_1",
    "super_fast_run_0": "wireframe_0", "super_fast_run_1": "wireframe_1",
    "wireframe_0": "animalization_0", "wireframe_1": "animalization_1",
}


def main() -> None:
    reg = Path(sys.argv[1])
    rows = [json.loads(line) for line in reg.read_text().splitlines() if line.strip()]

    # Reconstruct rotation-by-2 over the sorted unique refs and assert the frozen map matches it.
    refs = sorted({r["reference"] for r in rows if r.get("reference")})
    assert len(refs) == 36, f"expected 36 unique refs, got {len(refs)}"
    rot2 = {refs[i]: refs[(i + 2) % 36] for i in range(36)}
    assert rot2 == FROZEN_MAP, "frozen map != rotation-by-2 over the sorted unique refs"
    # derangement: no fixed points, and a class-clean permutation (never maps to the same base name)
    assert all(k != v for k, v in FROZEN_MAP.items()), "map has a fixed point (not a derangement)"

    n = 0
    for r in rows:
        ref = r.get("reference")
        if not ref:
            continue  # rows with no reference carry no signal (control path in the runner)
        r["signal_source"] = FROZEN_MAP[ref]
        n += 1

    with reg.open("w") as f:
        for r in sorted(rows, key=lambda r: (r["cell"], r["endpoint"], r.get("reference") or "", r["sided"])):
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"[derange] added signal_source to {n}/{len(rows)} rows in {reg} (rotation-by-2, verified)")


if __name__ == "__main__":
    main()
