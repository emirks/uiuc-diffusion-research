#!/usr/bin/env python3
"""Build the S6 (EffectData) group spec that `build_inventories.py spec` consumes.

S6 is a one-sided breadth stratum, like S4, but its A-description is PER-SUBJECT, not per-clip:
a subject's first frame is identical across all its clips (Axis-A counterfactual — same start,
different operator), so 2,000 descriptions serve 28,644 clips.  We express that by pointing each
clip's caption at its SUBJECT via an explicit `caption_sources`:

    sided = "one"                     No B endpoint; caption is `{A}. sksz.`.
    caption_sources[clip] = [[subj,"A"]]   The clip draws its A-description from its subject's
                                      key in the store (shelf 004, keyed '<subject>|A'). This
                                      reuses the 2,000-entry store directly — no per-clip copy.
    endpoints[clip] = [subj]          Recorded for provenance/{A} placeholder symmetry with S4.

`endpoint_disjointness = False`: many clips share a subject-endpoint by construction (that is
the whole point of the counterfactual axis), so the A->B-render disjointness property does not
apply — exactly as for S4's self-conditioning.

The prefix is ONE latent frame (video frame 0 alone) via `root_common.prefix_latents` on the
shape — the same frame-0 anchor S4 uses.  Not restated here; the mask store reads it from there.

Usage:
    python scripts/ctt_v2/s6/build_s6_spec.py --out outputs/ctt_v2/inventories/S6_spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
import root_common as rc  # noqa: E402

ROSTER = REPO / "outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json"
SELECTION = REPO / "data/processed/effectdata/selection_top2000.json"
CAPTIONS = REPO / "store/captions/004_effectdata/EFFECTDATA_CAPTION_STORE.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    roster = json.loads(ROSTER.read_text())
    clips = roster["clips"]                       # [{stem, subject, effect, video_path, w,h,latent_fhw}]
    store = json.loads(CAPTIONS.read_text())
    desc = store["descriptions"]                  # '<subject>|A' -> caption

    # every subject a roster clip references must have an A-description
    subjects = sorted({c["subject"] for c in clips})
    missing = [s for s in subjects if f"{s}|A" not in desc]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(subjects)} subjects have no |A description in the "
                         f"store: {missing[:10]}")

    groups: dict[str, dict] = {}
    endpoints: dict[str, list[str]] = {}
    cap_srcs: dict[str, list[list[str]]] = {}
    for c in clips:
        st, subj, eff = c["stem"], c["subject"], c["effect"]
        groups.setdefault(eff, {"class": None, "shader": None, "sided": "one",
                                "clips": []})["clips"].append(st)
        endpoints[st] = [subj]
        cap_srcs[st] = [[subj, "A"]]              # explicit: draw the A-description from the subject
    for g in groups.values():
        g["clips"].sort()

    # shape lookup: all clips are 81f; use any clip's latent_fhw as the representative for
    # prefix_latents (all 4 shapes carry prefix_latents=1 in RULED_SHAPES)
    shape = tuple(clips[0]["latent_fhw"])
    spec = {
        "stratum": "S6",
        "kind": "synthetic_op",
        "endpoint_disjointness": False,
        "endpoint_disjointness_reason":
            "S6 is a one-sided Axis-A stratum: many clips share a subject-endpoint (same first "
            "frame, different operator) by construction. Disjointness is a property of A->B "
            "renders drawing on a shared endpoint pool; it does not apply here, exactly as for S4.",
        "groups": dict(sorted(groups.items())),
        "endpoints": dict(sorted(endpoints.items())),
        "caption_sources": dict(sorted(cap_srcs.items())),
        "provenance": {
            "roster": str(ROSTER.relative_to(REPO)),
            "roster_sha256": rc.sha256_file(ROSTER),
            "selection": str(SELECTION.relative_to(REPO)),
            "selection_sha256": rc.sha256_file(SELECTION),
            "captions": str(CAPTIONS.relative_to(REPO)),
            "captions_content_hash": store["content_hash"],
            "caption_keying": "'<subject>|A' — per-subject (Axis-A shared first frame); 2000 "
                              "descriptions serve 28,644 clips via caption_sources[clip]=[[subject,A]]",
            "sided": "one",
            "sided_authority": "EffectData is A -> A-transformed; there is no B endpoint (roster sided='one')",
            "prefix_latents": rc.prefix_latents(shape),
            "prefix_authority": "frame-0 anchor per S4 precedent; DERIVED from root_common.prefix_latents "
                                "(RULED_SHAPES S6 shapes, prefix_latents=1), never restated here.",
            "group_id_note": "group id = the raw EffectData effect name (underscores); assembly slugs "
                             "it via root_common.slug_group for path safety.",
        },
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1))
    n_clips = sum(len(g["clips"]) for g in groups.values())
    print(f"[ok] {out.relative_to(REPO)}: {len(groups)} effects / {n_clips} clips / {len(subjects)} subjects "
          f"/ sided=one / prefix={rc.prefix_latents(shape)} latent frame(s)")
    print(f"[ok] caption_sources populated for {len(cap_srcs)}/{n_clips} clips (each draws its subject's A)")
    smallest = min(len(g["clips"]) for g in groups.values())
    print(f"[ok] smallest effect-group has {smallest} clip(s)")


if __name__ == "__main__":
    main()
