#!/usr/bin/env python
"""Gather the scattered per-chunk caption shards into ONE canonical, hashed store.

The store was filled over many hours in whatever unit of work was affordable/unblocked at
the time -- `chunk1..4`, then gap/tail/regrid patches, then a manual-rewrite shard.  That
layout is an accident of the filling order, not a structure anyone should inherit.  This
script produces the single artifact consumers read, while LEAVING every shard directory on
disk as the audit trail.

Canonical shape (one file):
  * `descriptions`  -- the IN-SCOPE 1,403, keyed "clip|role", the only thing assembly reads
  * `orphans`       -- paid-for (clip, role) descriptions no current grid consumes. KEPT,
                       never deleted, and held OUT of `descriptions` so nothing can
                       accidentally ship them or count them toward coverage
  * `provenance`    -- per description: shard, prompt variant, generator/auditor model +
                       the auditor's echoed model version, raw-response archive paths,
                       acceptance attempt, word count, Tier-2 flags, audit verdict
  * `content_hash`  -- sha256 over the canonical in-scope payload alone, so it is stable
                       against provenance/orphan churn

Hard checks (a consolidation that silently loses or double-counts a description is worse
than no consolidation):
  * a (clip, role) appearing in two shards is a HARD STOP unless the texts are identical
  * coverage against the requirement must be exact; a shortfall is named, never rounded
  * every in-scope description must carry provenance
  * shards contributing ZERO descriptions are recorded explicitly rather than omitted --
    an empty shard is a fact about the run, and omitting it hides a failed patch attempt

Usage
-----
  $PY consolidate_store.py --store-root <dir> --requirement <mass_pairs.json> \
      --out <CAPTION_STORE.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import root_common as rc  # noqa: E402 -- the A16 keyed-store key-shape validators


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--requirement", required=True,
                    help="mass_pairs.json -- the [[clip, role], ...] in-scope requirement")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.store_root)
    required = {tuple(x) for x in json.loads(Path(a.requirement).read_text())}

    descriptions: dict[str, str] = {}
    orphans: dict[str, str] = {}
    provenance: dict[str, dict] = {}
    shard_report = []

    for sd in sorted(p for p in root.iterdir() if p.is_dir()):
        dpath, rpath, mpath = (sd / "descriptions.json", sd / "records.json",
                               sd / "run_meta.json")
        if not dpath.exists():
            shard_report.append({"shard": sd.name, "status": "NO descriptions.json"})
            continue
        desc = json.loads(dpath.read_text())
        recs = json.loads(rpath.read_text()) if rpath.exists() else {}
        meta = json.loads(mpath.read_text()) if mpath.exists() else {}

        # auditor model version, as ECHOED by the API (never assumed from config)
        echoes = set()
        apath = sd / "raw_audit_responses.jsonl"
        if apath.exists():
            for line in apath.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ar = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ar.get("model_version_echo"):
                    echoes.add(ar["model_version_echo"])

        n_shard = 0
        for clip, roles in desc.items():
            for role, text in roles.items():
                key = f"{clip}|{role}"
                if key in provenance:
                    prev = provenance[key]
                    # A16 item 1 — an absent key is an EXCEPTION, never a None. `provenance`
                    # says this key was already written, so exactly one of the two stores must
                    # hold its text; a `.get()`-shaped read would turn a bookkeeping bug into
                    # `prior_text = None`, which then reports a spurious "CONFLICTING
                    # duplicate" against a phantom prior.
                    if key in descriptions:
                        prior_text = descriptions[key]
                    elif key in orphans:
                        prior_text = orphans[key]
                    else:
                        raise SystemExit(
                            f"{key}: recorded in provenance from shard {prev['shard']} but "
                            f"absent from BOTH `descriptions` and `orphans` — the store's "
                            f"bookkeeping disagrees with itself; refusing to interpret the "
                            f"absence as a conflict (A16 keyed-join rule)")
                    if prior_text != text:
                        raise SystemExit(
                            f"{key}: CONFLICTING duplicate -- shard {prev['shard']} and "
                            f"shard {sd.name} hold DIFFERENT text. Refusing to pick one "
                            f"silently.\n  {prev['shard']}: {prior_text!r}\n"
                            f"  {sd.name}: {text!r}")
                    prev.setdefault("also_in_shards", []).append(sd.name)
                    continue
                rec = recs.get(key, {})
                provenance[key] = {
                    "shard": sd.name,
                    "prompt_variant": meta.get("prompt_variant"),
                    "generator_model": meta.get("generator_model"),
                    "auditor_model": meta.get("auditor_model"),
                    "auditor_model_version_echo": sorted(echoes) or None,
                    "auditor_thinking_level": meta.get("auditor_thinking_level"),
                    "accepted_on_attempt": rec.get("accepted_on_attempt"),
                    "words": rec.get("words"),
                    "tier2": rec.get("tier2") or [],
                    "audit_verdict": rec.get("audit"),
                    "bank": rec.get("bank"),
                    "raw_generation_archive": (
                        str(sd / "raw_generation_responses.jsonl")
                        if (sd / "raw_generation_responses.jsonl").exists() else None),
                    "raw_audit_archive": str(apath) if apath.exists() else None,
                    "operator_reason": rec.get("operator_reason"),
                }
                if (clip, role) in required:
                    descriptions[key] = text
                else:
                    orphans[key] = text
                n_shard += 1

        shard_report.append({
            "shard": sd.name, "descriptions_contributed": n_shard,
            "prompt_variant": meta.get("prompt_variant"),
            "generator_model": meta.get("generator_model"),
            "auditor_model": meta.get("auditor_model"),
            "auditor_model_version_echo": sorted(echoes) or None,
            "records": len(recs),
            "queued_unresolved": sum(
                1 for r in recs.values() if r.get("accepted_on_attempt") is None),
            "note": ("CONTRIBUTED NOTHING -- every attempt failed and went to the "
                     "manual-rewrite queue; kept for the audit trail"
                     if n_shard == 0 else None),
        })

    # ---- coverage, exactly ------------------------------------------------
    have = {tuple(k.split("|", 1)) for k in descriptions}
    missing = sorted(required - have)
    if missing:
        raise SystemExit(
            f"consolidation is {len(missing)} description(s) SHORT of the requirement: "
            f"{missing[:20]}. Refusing to write a store that claims coverage it lacks.")
    no_prov = [k for k in descriptions if k not in provenance]
    if no_prov:
        raise SystemExit(f"{len(no_prov)} in-scope description(s) have no provenance: "
                         f"{no_prov[:10]}")

    canonical = json.dumps(descriptions, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode()
    content_hash = sha256_bytes(canonical)
    orphan_payload = json.dumps(orphans, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":")).encode()

    tier2 = {k: provenance[k]["tier2"] for k in descriptions if provenance[k]["tier2"]}
    variants = sorted({provenance[k]["prompt_variant"] for k in descriptions})
    gens = sorted({provenance[k]["generator_model"] for k in descriptions})
    auds = sorted({str(provenance[k]["auditor_model"]) for k in descriptions})

    out = {
        "schema": "ctt_v2_caption_store/v1",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "store_root_audit_trail": str(root),
        "keying": ("'clip_id|role'. A-role describes frames 0-8, B-role frames 112-120. "
                   "One description per (clip, role), never per sample: a clip used as the "
                   "A endpoint of 40 rows has exactly one A-role description."),
        "content_hash": f"sha256:{content_hash}",
        "content_hash_covers": ("the in-scope `descriptions` map only (sorted keys, compact "
                                "separators, ensure_ascii=False) -- stable against orphan "
                                "and provenance churn"),
        "orphans_hash": f"sha256:{sha256_bytes(orphan_payload)}",
        "counts": {
            "in_scope": len(descriptions),
            "required": len(required),
            "coverage": f"{len(descriptions)}/{len(required)}",
            "orphans": len(orphans),
            "total_paid_for": len(descriptions) + len(orphans),
            "tier2_flagged_in_scope": len(tier2),
        },
        "homogeneity": {
            "prompt_variants": variants,
            "generator_models": gens,
            "auditor_models": auds,
            "single_prompt_variant": len(variants) == 1,
            "note": ("A13-b/A14: never mix prompt variants in one store -- 8a names mixed "
                     "prompts as a bug class it detects. A single variant here is the "
                     "assert, not a coincidence."),
        },
        "orphans_policy": ("KEPT, not deleted -- paid-for descriptions the current pinned "
                           "grids do not consume (grid drift). Held out of `descriptions` "
                           "so they cannot ship or inflate coverage."),
        "shards": shard_report,
        "descriptions": descriptions,
        "orphans": orphans,
        "tier2_queue": tier2,
        "provenance": provenance,
    }
    # A16 item 4 — `keying` is MANDATORY on a keyed store artifact, and it must actually
    # DESCRIBE the keys written next to it.  A declaration that drifts from the data is worse
    # than none: it would certify a wrong-shaped lookup as correct.  Proven here, at write
    # time, against the store's own keys.
    rc.require_keying_declaration(out, a.out)
    for probe in (list(descriptions)[:1] or list(orphans)[:1]):
        rc.assert_key_shape(descriptions or orphans, probe,
                            where=f"{a.out}:descriptions", keying=out["keying"])
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"in-scope {len(descriptions)}/{len(required)}  orphans {len(orphans)}  "
          f"total {len(descriptions)+len(orphans)}")
    print(f"prompt variants: {variants}   generators: {gens}   auditors: {auds}")
    print(f"Tier-2 flagged (in-scope): {len(tier2)}")
    print(f"content_hash: sha256:{content_hash}")
    for s in shard_report:
        extra = f"  [{s['note']}]" if s.get("note") else ""
        print(f"  {s['shard']:<10} {s.get('descriptions_contributed', 0):>5}  "
              f"variant={s.get('prompt_variant')} auditor={s.get('auditor_model')}{extra}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
