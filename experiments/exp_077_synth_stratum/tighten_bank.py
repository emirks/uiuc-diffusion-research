"""exp_077 D1 STAGE 0 — tighten the synthetic endpoint bank.

Operator visual review of the 331-clip bank (`data/processed/synth_endpoints/`) found ~half
marginal (subject-small-in-wide-scene, esp. DAVIS; multi-subject/cluttered; object-only;
a few warped by crop). This re-filters the bank via the ALREADY-COMPUTED per-clip subject
bbox + detector label/score in `manifest.jsonl` (no new detection), keeping only strong,
large, single dominant SUBJECTS:

  KEEP iff subject.present
          AND bbox area (w_frac*h_frac) >= AREA_MIN          (subject not small in a wide scene)
          AND subject.score        >= SCORE_MIN              ("single dominant subject" proxy:
                                                              a confident top detection)
          AND subject.label        in KEEP_LABELS            (person / animal / vehicle; drops
                                                              object-only + still-life scenes)

If the kept count falls below FLOOR (150), AREA_MIN is relaxed in steps to land in ~150-200
(per the D1 spec). The kept list is D1's endpoint source.

Output: `data/processed/synth_endpoints/bank_tightened.json`
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BANK_DIR = REPO_ROOT / "data/processed/synth_endpoints"
MANIFEST = BANK_DIR / "manifest.jsonl"
OUT = BANK_DIR / "bank_tightened.json"
AUDIT = Path(__file__).resolve().parent / "D1_BUILD_AUDIT.json"

AREA_MIN = 0.15          # subject bbox area >= 15% of frame (D1 spec)
SCORE_MIN = 0.70         # confident top detection == "single dominant subject" proxy
FLOOR = 150              # relax AREA_MIN if kept < FLOOR

# COCO person + animals + vehicles = distinctive SUBJECTS. Everything else (furniture, food,
# kitchenware, electronics, indoor objects) is object-only / scene clutter -> dropped.
KEEP_LABELS = {
    "person",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
}


def area(r: dict) -> float:
    s = r.get("subject", {})
    return (s.get("w_frac") or 0.0) * (s.get("h_frac") or 0.0)


def keep(r: dict, area_min: float) -> bool:
    s = r.get("subject", {})
    return bool(
        s.get("present")
        and area(r) >= area_min
        and (s.get("score") or 0.0) >= SCORE_MIN
        and s.get("label") in KEEP_LABELS
    )


def merge_audit(update: dict) -> None:
    cur = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    cur.update(update)
    AUDIT.write_text(json.dumps(cur, indent=2))


def main() -> None:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    area_min = AREA_MIN
    kept = [r for r in rows if keep(r, area_min)]
    relaxed = []
    for relax in (0.12, 0.10, 0.08):
        if len(kept) >= FLOOR:
            break
        area_min = relax
        kept = [r for r in rows if keep(r, area_min)]
        relaxed.append(relax)

    by_source: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for r in kept:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_label[r["subject"]["label"]] = by_label.get(r["subject"]["label"], 0) + 1

    clips = [
        {
            "clip_id": r["clip_id"],
            "mp4": r["mp4"],
            "source": r["source"],
            "label": r["subject"]["label"],
            "score": round(r["subject"]["score"], 4),
            "bbox_area": round(area(r), 4),
        }
        for r in sorted(kept, key=lambda x: x["clip_id"])
    ]

    out = {
        "policy": {
            "area_min": area_min,
            "score_min": SCORE_MIN,
            "keep_labels": sorted(KEEP_LABELS),
            "floor": FLOOR,
            "relaxed_area_steps": relaxed,
            "rule": "subject.present AND bbox_area>=area_min AND score>=score_min AND label in keep_labels",
        },
        "n_input": len(rows),
        "n_kept": len(kept),
        "by_source": by_source,
        "by_label": by_label,
        "clips": clips,
    }
    OUT.write_text(json.dumps(out, indent=2))
    merge_audit({
        "stage0_tighten": {
            "n_input": len(rows), "tightened_bank_count": len(kept),
            "area_min": area_min, "score_min": SCORE_MIN,
            "by_source": by_source, "relaxed_area_steps": relaxed,
            "bank_tightened_path": str(OUT),
        }
    })
    print(f"[tighten] input={len(rows)} kept={len(kept)} area_min={area_min} score_min={SCORE_MIN}")
    print(f"[tighten] by_source={by_source}")
    print(f"[tighten] by_label={by_label}")
    print(f"[tighten] wrote {OUT}")


if __name__ == "__main__":
    main()
