#!/usr/bin/env python3
"""Build the S4 group spec that `build_inventories.py spec` consumes.

Two things distinguish S4 from every other stratum, and both come from the same fact — S4 is a
**self-conditioned** clip, not a rendered A->B pair:

  sided = "one"      There is no B endpoint.  Only the prefix anchor is conditioned, so the
                     assembled caption is `{A-description}. sksz.` with no suffix sentence.
  endpoints[c] = [c] Every clip is its OWN A endpoint.  S2's endpoints are two *other* clips
                     spliced into a render; S4's conditioned pixels are the clip's own frame 0.
                     That is why the caption store is keyed `<stem>|A` on the clip itself.

The prefix is ONE latent frame for this shape (`root_common.prefix_latents`), i.e. video frame
0 alone — the owner's 2026-07-28 decision.  This script does not restate that; it is a property
of the shape and the mask store reads it from the same place.

Usage:
    python scripts/ctt_v2/s4/build_s4_spec.py --out outputs/ctt_v2/inventories/S4_spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
import root_common as rc  # noqa: E402

ROSTER = REPO / "outputs/ctt_v2/encodes/S4/ROSTER.json"
SELECTION = REPO / "data/processed/s4_refvfx/selection.json"
CAPTIONS = REPO / "outputs/ctt_v2/captions/S4_CAPTION_STORE.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    roster = json.loads(ROSTER.read_text())
    stems = list(roster["stems"])
    effect = {s["k"]: s["effect"] for s in json.loads(SELECTION.read_text())["samples"]}
    store = json.loads(CAPTIONS.read_text())
    desc = store["descriptions"]

    missing_cap = [s for s in stems if f"{s}|A" not in desc]
    if missing_cap:
        raise SystemExit(f"{len(missing_cap)} roster stems have no |A description: "
                         f"{missing_cap[:10]}")

    groups: dict[str, dict] = {}
    for st in stems:
        groups.setdefault(effect[st], {"class": None, "shader": None, "sided": "one",
                                       "clips": []})["clips"].append(st)
    for g in groups.values():
        g["clips"].sort()

    #: EVERY clip is its own A endpoint -- see the module docstring.  Without this the
    #: inventory's `endpoints` are empty, `root_common.caption_sources` returns [], and the
    #: assembled S4 caption silently loses its description sentence.
    endpoints = {st: [st] for st in stems}

    shape = tuple(roster.get("latent_fhw", (5, 14, 26)))
    spec = {
        "stratum": "S4",
        "kind": "synthetic_op",
        "endpoint_disjointness": False,
        "endpoint_disjointness_reason":
            "S4 is SELF-conditioned: endpoints[c] == [c] by construction, so a clip is "
            "trivially 'shared' with itself.  Disjointness is a property of A->B renders "
            "drawing on a shared endpoint pool; it does not apply here and asserting it "
            "would fail on every row.",
        "groups": dict(sorted(groups.items())),
        "endpoints": dict(sorted(endpoints.items())),
        "provenance": {
            "roster": str(ROSTER.relative_to(REPO)),
            "roster_sha256": rc.sha256_file(ROSTER),
            "selection": str(SELECTION.relative_to(REPO)),
            "selection_sha256": rc.sha256_file(SELECTION),
            "captions": str(CAPTIONS.relative_to(REPO)),
            "captions_content_hash": store["content_hash"],
            "sided": "one",
            "sided_authority": "refVFX I2V_LoRA is A -> A-transformed; there is no B endpoint",
            "prefix_latents": rc.prefix_latents(shape),
            "prefix_authority": "owner decision 2026-07-28 — condition on video frame 0 alone "
                                "(= latent frame 0). DERIVED from root_common.prefix_latents, "
                                "never restated here.",
            "group_id_note": "group id = the raw refVFX effect string (spaces and dots "
                             "included); assembly slugs it via root_common.slug_group for "
                             "path safety.",
        },
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1))
    print(f"[ok] {out.relative_to(REPO)}: {len(groups)} groups / {len(stems)} clips / "
          f"sided=one / prefix={rc.prefix_latents(shape)} latent frame(s)")
    print(f"[ok] endpoints populated for {len(endpoints)}/{len(stems)} clips "
          f"(each clip is its own A endpoint)")
    n = min(len(g["clips"]) for g in groups.values())
    print(f"[ok] smallest group has {n} clips -> ring-offset k=min(3, n-1)={min(3, n - 1)}")


if __name__ == "__main__":
    main()
