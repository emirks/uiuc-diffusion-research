#!/usr/bin/env python3
"""Assemble every stratum's TRAINING captions and emit the Gemma-encode input files.

Advisor A21 Q1: the `conditions/` tree is NOT blocked on S1 media.
`ltx-trainer/scripts/process_captions.py` reads `media_path` **only to name its output** and
never opens a media file, so every stratum whose captions are final can be encoded now. All of
them are.

Assembly rule (`root_common.caption_sources` + the render contract, not restated here):

    two-sided   "{A-description}. sksz. {B-description}."
    one-sided   "{A-description}. sksz."
    S0 (corpus) the certified caption verbatim -- already carries ` sksz.`

Output naming is the load-bearing detail. `process_captions._embedding_output_path` computes
`(dataset_file.parent / media_path)`, relativises it, and swaps the suffix for `.pt`. So a
**relative** `media_path` of `<clip>.mp4` yields exactly `<output_dir>/<clip>.pt`. An ABSOLUTE
media_path instead falls through to `Path(*parts[1:])` and buries the output under a mirror of
the filesystem -- a full conditions tree that silently fails to join. Hence: relative, flat,
one output dir per stratum, keyed by the same stem as `encodes/<stratum>/latents/<stem>.pt`.

Usage:
    python scripts/ctt_v2/captions/build_encode_inputs.py --strata S2a,S2b,S4
    python scripts/ctt_v2/captions/build_encode_inputs.py --strata S2a,S2b,S4 --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
import root_common as rc  # noqa: E402

LOCKED = REPO / "outputs/ctt_v2/captions/CAPTION_STORE.json"
S4_STORE = REPO / "outputs/ctt_v2/captions/S4_CAPTION_STORE.json"
S6_STORE = REPO / "store/captions/004_effectdata/EFFECTDATA_CAPTION_STORE.json"
OUTDIR = REPO / "outputs/ctt_v2/conditions_inputs"


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def assembled_for(stratum: str, locked: dict, s4: dict, s6: dict) -> dict[str, str]:
    """{clip_stem: assembled training caption} for one stratum, from its inventory."""
    inv = json.loads((REPO / f"outputs/ctt_v2/inventories/{stratum}.json").read_text())
    kind = inv.get("kind", "synthetic_op")
    # S6 (EffectData) draws A-descriptions from its per-subject store (keyed '<subject>|A',
    # reached via each clip's explicit caption_sources=[[subject,"A"]]); S4 from the S4 store.
    store = {"S4": s4, "S6": s6}.get(stratum, locked)
    out: dict[str, str] = {}
    for stem, entry in inv["clips"].items():
        # The inventory's own `caption` is AUTHORITATIVE when present: it is the exact string
        # assembly will symlink a conditions embed for, so encoding anything else would encode
        # text the trainer never sees. This covers S0 (certified caption) and S1's s0cf layer,
        # whose `caption_sources` is legitimately [] because it draws on the certified 139
        # rather than on the per-(clip, role) store.
        if entry.get("caption"):
            out[stem] = entry["caption"]
            continue
        if kind == "corpus":
            out[stem] = entry["caption"]
            continue
        g = inv["groups"][entry["group"]]
        srcs = rc.caption_sources(entry, g.get("sided", "two"), kind)
        parts = []
        for clip, role in srcs:
            key = f"{clip}|{role}"
            if key not in store:
                raise SystemExit(f"[{stratum}] {stem}: store has no {key!r} -- refusing to "
                                 f"encode a caption assembled from a missing description")
            parts.append(store[key])
        if not parts:
            raise SystemExit(f"[{stratum}] {stem}: caption_sources is EMPTY. That is the "
                             f"`endpoints: {{}}` defect -- the caption would degrade to a bare "
                             f"trigger. Fix the spec, do not encode this.")
        cap = f"{parts[0]}.{rc.TRIGGER_SENTENCE}"
        if len(parts) > 1:
            cap += f" {parts[1]}."
        out[stem] = cap
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", default="S2a,S2b,S4")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    strata = [s.strip() for s in args.strata.split(",") if s.strip()]

    locked_doc = json.loads(LOCKED.read_text())
    locked = locked_doc["descriptions"]
    s4_doc = json.loads(S4_STORE.read_text())
    s4 = s4_doc["descriptions"]
    s6_doc = json.loads(S6_STORE.read_text()) if S6_STORE.exists() else {"descriptions": {}, "content_hash": None}
    s6 = s6_doc["descriptions"]

    filt = rc.leak_filter()
    manifest = {
        "schema": "ctt_v2_conditions_encode_inputs/v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "authority": "advisor A21 Q1 -- the conditions tree is not blocked on S1 media; "
                     "process_captions.py reads media_path only to name its output",
        "description_sources": {
            "locked_store": {"path": str(LOCKED.relative_to(REPO)),
                             "content_hash": locked_doc["content_hash"],
                             "file_sha256": rc.sha256_file(LOCKED)},
            "s4_store": {"path": str(S4_STORE.relative_to(REPO)),
                         "content_hash": s4_doc["content_hash"],
                         "file_sha256": rc.sha256_file(S4_STORE)},
            "s6_store": {"path": str(S6_STORE.relative_to(REPO)),
                         "content_hash": s6_doc["content_hash"],
                         "file_sha256": rc.sha256_file(S6_STORE)} if S6_STORE.exists() else None,
        },
        "encoder": "LTX-2-cond-bleed-fix/packages/ltx-trainer/scripts/process_captions.py",
        "encoder_flags_forbidden": {
            "--lora-trigger": "PREPENDS a trigger; our captions already carry ` sksz.` inline, "
                              "so passing it yields two triggers and fails caption_violations",
            "--remove-llm-prefixes": "would mutate certified caption text",
        },
        "output_naming": "media_path is RELATIVE '<stem>.mp4' => output '<output_dir>/<stem>.pt', "
                         "the same stem as encodes/<stratum>/latents/<stem>.pt. An ABSOLUTE "
                         "media_path would bury outputs under a filesystem mirror and the tree "
                         "would silently fail to join.",
        "leak_filter": filt.source,
        "strata": {},
    }

    total = 0
    for st in strata:
        caps = assembled_for(st, locked, s4, s6)
        bad = {k: v for k, v in ((k, rc.caption_violations(c, filt)) for k, c in caps.items()) if v}
        if bad:
            raise SystemExit(f"[{st}] {len(bad)} assembled caption(s) violate RULING 9, first 3: "
                             + json.dumps(dict(list(bad.items())[:3]), indent=1))
        # ---- CONTENT-ADDRESSED DEDUP -------------------------------------------------------
        # A caption is a pure function of the ENDPOINTS.  Naming the shader is forbidden
        # (RULING 9 -- it is the leak the whole experiment is built to prevent), so every clip
        # sharing an endpoint pair necessarily shares a caption: S2a has 331 distinct pairs
        # across 7,961 clips, S2b 800 across 7,990.  Encoding one embed per CLIP would compute
        # the same tensor ~24x and store it ~24x.  Keying the embed by the sha256 of its own
        # text encodes each distinct caption exactly once and lets many clips point at it --
        # which is already how S0 works (139 embeds serving 385 samples).
        by_hash: dict[str, str] = {}
        clip_to_hash: dict[str, str] = {}
        for stem in sorted(caps):
            h = sha_text(caps[stem])[:16]
            by_hash[h] = caps[stem]
            clip_to_hash[stem] = h
        rows = [{"media_path": f"{h}.mp4", "caption": by_hash[h]} for h in sorted(by_hash)]
        canon = json.dumps({k: caps[k] for k in sorted(caps)}, sort_keys=True)
        rec = {
            "n_clips": len(caps),
            "n_distinct_captions": len(by_hash),
            "dedup_factor": round(len(caps) / max(1, len(by_hash)), 2),
            "dedup_rationale": "caption = f(endpoints); the shader is unnameable by RULING 9, "
                               "so clips sharing an endpoint pair share a caption verbatim",
            "dataset_file": f"outputs/ctt_v2/conditions_inputs/{st}_captions.json",
            "clip_to_caption_hash": f"outputs/ctt_v2/conditions_inputs/{st}_clip_to_hash.json",
            "output_dir": "outputs/ctt_v2/conditions/by_caption",
            "conditions_path_template":
                "outputs/ctt_v2/conditions/by_caption/{caption_sha16}.pt",
            "content_hash": "sha256:" + sha_text(canon),
            "expected_embed_files": len(by_hash),
            "example": rows[0]["caption"][:150],
        }
        manifest["strata"][st] = rec
        total += len(by_hash)
        print(f"[{st}] {len(caps):6,} clips -> {len(by_hash):6,} distinct embeds "
              f"({rec['dedup_factor']:5.2f}x saved)  hash {rec['content_hash'][7:23]}")
        if args.write:
            OUTDIR.mkdir(parents=True, exist_ok=True)
            (OUTDIR / f"{st}_captions.json").write_text(json.dumps(rows, indent=1))
            (OUTDIR / f"{st}_clip_to_hash.json").write_text(
                json.dumps(dict(sorted(clip_to_hash.items())), indent=1))

    manifest["total_distinct_embeds"] = total
    manifest["projected_bytes"] = total * 15_739_149
    manifest["note_audio_half"] = (
        "Half of each 15,739,149-byte file is `audio_prompt_embeds` (1024,3840) bf16, and the "
        "trainer logs audio as DISABLED. It is kept because the format is the trainer's, not "
        "ours -- but it is why the per-file size is twice what the text embed alone needs.")
    print(f"\n[total] {total:,} DISTINCT embeds -> ~{total * 15_739_149 / 1e9:.1f} GB")
    if args.write:
        (OUTDIR / "ENCODE_INPUTS_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        print(f"[ok] wrote {OUTDIR.relative_to(REPO)}/ ({len(strata)} dataset files + manifest)")


if __name__ == "__main__":
    main()
