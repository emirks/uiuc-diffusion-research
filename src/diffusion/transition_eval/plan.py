"""plan — lifecycle phase 1 (SPEC §4): turn a suite design into suite.json,
the generation contract the model side must fulfill and score() verifies
completeness against. The harness never runs the model.

A design is a JSON list of item specs (the bespoke pairing logic — quads,
rotations — stays in experiment scripts, which is fine: they emit designs,
not metrics). plan() adds what the DESIGN layer must never forget:

  - validation: every reference/condition resolves to a corpus clip; styles
    exist; contract fields well-formed (reject loudly, SPEC §2)
  - base twins: auto-emitted for every item marked "twin": true, with
    twin_of linkage (the causal control is not optional per SPEC §4)
  - controls: lerp / static-hold arms are MARKERS here (synthesized at
    scoring from the item's own conditions — no GPU generation involved)

Usage:
    python -m diffusion.transition_eval.plan --design design.json \
        --corpus corpus_manifest.json --out suite.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

from .manifests_v3 import load_corpus_manifest, sidedness_of


def make_suite(design: list[dict], corpus: dict) -> dict:
    items, errors = [], []
    ids = set()
    for spec in design:
        iid = spec.get("item_id")
        if not iid or iid in ids:
            errors.append(f"missing/duplicate item_id: {iid!r}")
            continue
        ids.add(iid)
        style = spec.get("style")
        if style not in corpus["classes"]:
            errors.append(f"{iid}: style {style!r} not in corpus manifest")
            continue
        try:
            side = sidedness_of(style, corpus)
        except ValueError as e:
            errors.append(f"{iid}: {e}")
            continue
        item = dict(spec)
        item["_derived_sidedness"] = side   # informational; score re-derives
        items.append(item)
        if spec.pop("twin", False):
            items.append({**{k: v for k, v in item.items() if k != "twin"},
                          "item_id": f"base__{iid}",
                          "arm": f"base_{item.get('arm', '')}".rstrip("_"),
                          "twin_of": iid,
                          "generated_video": item["generated_video"].replace(
                              ".mp4", "__base.mp4"),
                          "notes": f"auto base twin of {iid}"})
    if errors:
        raise ValueError("plan validation failed:\n  " + "\n  ".join(errors))
    return {"schema": 1, "n_items": len(items), "items": items}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="suite.json")
    args = ap.parse_args()
    corpus = load_corpus_manifest(args.corpus)
    design = json.loads(pathlib.Path(args.design).read_text())
    suite = make_suite(design, corpus)
    pathlib.Path(args.out).write_text(json.dumps(suite, indent=1))
    print(f"[done] {suite['n_items']} planned items -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
