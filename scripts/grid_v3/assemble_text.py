#!/usr/bin/env python
"""grid v3 — assemble the Lane-B text artifacts for the new endpoints, in the corpus's own shapes.

Inputs (all produced earlier in the grid-v3 lane):
  outputs/ctt_v2/captions/grid_v3/{reserve,higgsfield}/{descriptions,records}.json   gemini-3.6-flash v2 A/B
      descriptions, Layer-2 audited by gemini-3.5-flash-lite (leak NO / inaccurate NO recorded per record)
  store/captions/004_effectdata/EFFECTDATA_CAPTION_STORE.json                         subject|A (S6 store, v2-s4f0)
  eval_ladder/registry_v3_pending.jsonl                                                which clips play the endpoint role

Outputs:
  data/processed/transitions_std121/dataset_grid_v3.json   corpus-format caption source [{video, caption}] with
      caption = "{A}. The scene transforms into {b}" — the marker the renderer splits on; S2 (b) is the B-role
      description for two-sided clips (lowercase, participial) and empty for one-sided clips (never rendered).
      Consumed by prompts.captions() (CAPTION_SOURCES) and prompts.audited_clips() (AUDITED_SOURCES).
  eval_ladder/reserve.yaml                                  the reserve foreign roster in davis.yaml's schema
      (one_sided: clip -> caption; two_sided: hvpair -> prefix/suffix captions; suffix = the B clip's own
      A-description, lowercase-initial, per the DAVIS rule) with `source: humanvid`.
  store/captions/005_grid_v3_endpoints/{CAPTION_STORE.json, meta.yaml}   the tracked canonical: descriptions keyed
      "<clip>|<role>" for the Higgsfield + reserve clips, provenance per description, content_hash over descriptions.
Run:  python scripts/grid_v3/assemble_text.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
STD = REPO / "data/processed/transitions_std121"
CAP = REPO / "outputs/ctt_v2/captions/grid_v3"
PENDING = REPO / "eval_ladder/registry_v3_pending.jsonl"
ED_STORE = REPO / "store/captions/004_effectdata/EFFECTDATA_CAPTION_STORE.json"
OUT_SRC = STD / "dataset_grid_v3.json"
OUT_ROSTER = REPO / "eval_ladder/reserve.yaml"
STORE_DIR = REPO / "store/captions/005_grid_v3_endpoints"
MARKER = "The scene transforms into "
GRID = yaml.safe_load((REPO / "eval_ladder/grid_v3.yaml").read_text())


def lower_initial(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def load_lane(name: str) -> tuple[dict, dict]:
    d = json.loads((CAP / name / "descriptions.json").read_text())
    r = json.loads((CAP / name / "records.json").read_text())
    return d, r


def audited_ok(rec: dict) -> bool:
    h = rec["history"][-1]
    a = h.get("audit") or {}
    return bool(rec.get("description")) and a.get("leak") == "NO" and a.get("inaccurate") == "NO"


def main() -> None:
    pend = [json.loads(l) for l in PENDING.read_text().splitlines()]
    hv_d, hv_r = load_lane("reserve")
    hf_d, hf_r = load_lane("higgsfield")
    ed = json.loads(ED_STORE.read_text())["descriptions"]
    pref = GRID["effectdata"]["clip_prefix"] + "."

    descriptions, provenance = {}, {}
    for lane, d, r in (("reserve", hv_d, hv_r), ("higgsfield", hf_d, hf_r)):
        for key, rec in r.items():
            assert audited_ok(rec), f"{lane}: {key} not audit-clean — refuse to ship"
            descriptions[key] = rec["description"]
            h = rec["history"][-1]
            provenance[key] = {"lane": lane, "prompt_variant": h.get("prompt_variant", "v2"), "generator_model": "gemini-3.6-flash",
                               "auditor_model": "gemini-3.5-flash-lite", "audit": h.get("audit"), "accepted_on_attempt": rec.get("accepted_on_attempt"),
                               "words": rec.get("words"), "N_target": rec.get("N_target"), "strips": "9 anchor frames, re-timed to 8 fps for the API minimum duration (frames identical)",
                               "raw_archive": f"outputs/ctt_v2/captions/grid_v3/{lane}/raw_generation_responses.jsonl"}

    # ---- corpus-format caption source: Higgsfield new endpoints (A [+B]) + EffectData endpoints (subject|A)
    rows, audited = [], []
    hf_eps = {}
    for r in pend:
        ep = r["endpoint"]
        if r["endpoint_source"] in ("heldin_test", "heldout") and ep in hf_d:
            hf_eps[ep] = r["endpoint_class"]
    for ep, cls in sorted(hf_eps.items()):
        a = hf_d[ep]["A"]
        b = lower_initial(hf_d[ep].get("B", ""))
        rows.append({"video": f"{cls}/{ep}.mp4", "caption": f"{a}. {MARKER}{b}"})
        audited.append(ep)
    ed_eps = {}
    for r in pend:
        ep = r["endpoint"]
        if ep.startswith(pref):
            subject = ep.split(".")[2]
            ed_eps[ep] = (r["endpoint_class"], subject)
    for ep, (cls, subject) in sorted(ed_eps.items()):
        a = ed.get(f"{subject}|A")
        assert a, f"no S6 caption for subject {subject} ({ep})"
        rows.append({"video": f"{cls}/{ep}.mp4", "caption": f"{a}. {MARKER}"})
        audited.append(ep)
    OUT_SRC.write_text(json.dumps(rows, indent=1) + "\n")

    # ---- reserve roster (davis.yaml schema + source)
    one, two = {}, {}
    for r in pend:
        ep = r["endpoint"]
        if r["endpoint_source"] != "humanvid":
            continue
        if ep.startswith("hvpair."):
            _, a, b = ep.split(".", 2)
            two[ep] = {"prefix": {"clip": a, "caption": hv_d[a]["A"]}, "suffix": {"clip": b, "caption": lower_initial(hv_d[b]["A"])}}
        else:
            one[ep] = {"clip": ep, "caption": hv_d[ep]["A"]}
    roster = {"source": "humanvid", "note": "the never-trained reserve (CONTENT_POOL_union.json:reserved, bank humanvid); "
              "captions = gemini-3.6-flash v2 A-role descriptions of the clip's first 9 frames, Layer-2 audited "
              "(store/captions/005_grid_v3_endpoints); two-sided pairs follow the DAVIS rule: suffix = first 9 frames of clip B, "
              "captioned by B's own A-description. Generated by scripts/grid_v3/assemble_text.py — do not hand-edit.",
              "one_sided": dict(sorted(one.items())), "two_sided": dict(sorted(two.items()))}
    OUT_ROSTER.write_text("# grid v3 reserve foreign roster — GENERATED (scripts/grid_v3/assemble_text.py); block style only.\n"
                          + yaml.safe_dump(roster, sort_keys=False, width=100, default_flow_style=False))

    # ---- store shelf entry 005 (tracked canonical, hashed over descriptions)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(descriptions, sort_keys=True).encode()
    h = hashlib.sha256(payload).hexdigest()
    store = {"schema": "ctt_v2_caption_store/v1", "written_at": date.today().isoformat(),
             "keying": "<clip>|<role>", "content_hash": f"sha256:{h}", "content_hash_covers": "descriptions",
             "counts": {"descriptions": len(descriptions), "reserve": sum(1 for k in descriptions if k.startswith("humanvid_")),
                        "higgsfield": sum(1 for k in descriptions if not k.startswith("humanvid_"))},
             "descriptions": descriptions, "provenance": provenance}
    (STORE_DIR / "CAPTION_STORE.json").write_text(json.dumps(store, indent=1, sort_keys=True) + "\n")
    (STORE_DIR / "meta.yaml").write_text(f"""id: captions/005_grid_v3_endpoints
seq: 5
shelf: captions
created: {date.today().isoformat()}
role: grid v3 eval endpoints — A/B descriptions for the new Higgsfield corpus clips + A descriptions for the humanvid reserve (foreign roster); Lane B EVAL ONLY (never in training)
keyed: "<clip>|<role>"
coverage: {len(descriptions)} descriptions ({store['counts']['higgsfield']} Higgsfield, {store['counts']['reserve']} reserve)
content_hash: sha256:{h}   # over the `descriptions` map
prompt_variant: v2               # generate_descriptions.py _PROMPT_{{A,B}}_TEMPLATE, per-item length draw over the 171-value corpus list
generator: gemini-3.6-flash (temp 0.7, thinkingLevel minimal, 120 tok)
auditor: gemini-3.5-flash-lite   # Layer-2, every shipped description leak NO + inaccurate NO (CTT_AUDIT_MODEL, the store-validated auditor)
strips: 9-frame A (0-8) / B (112-120) anchors, RE-TIMED TO 8 FPS (frames identical) — the API now rejects 0.375 s clips (HTTP 400); data/processed/caption_strips/strips_index_grid_v3.json
consumers: data/processed/transitions_std121/dataset_grid_v3.json (corpus-format caption source read by eval_ladder/prompts.py) · eval_ladder/reserve.yaml (foreign roster)
effectdata_endpoints: NOT here — their A descriptions are the S6 store (captions/004_effectdata, key subject|A), mapped into dataset_grid_v3.json
source: outputs/ctt_v2/captions/grid_v3/{{reserve,higgsfield}}/  (records.json + raw_*_responses.jsonl archives)
assembled_by: scripts/grid_v3/assemble_text.py
authority: ../../TEXT_LIFECYCLE.md   # §3 Lane B
""")
    print(f"caption source rows {len(rows)} (higgsfield {len(hf_eps)}, effectdata {len(ed_eps)}) -> {OUT_SRC.relative_to(REPO)}")
    print(f"reserve roster: one-sided {len(one)}, two-sided {len(two)} -> {OUT_ROSTER.relative_to(REPO)}")
    print(f"store captions/005: {len(descriptions)} descriptions, sha256 {h[:12]}")


if __name__ == "__main__":
    main()
