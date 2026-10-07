#!/usr/bin/env python
"""Derive S6_r832.json from S6.json by re-pointing every S6 clip's latents/cond_clean at the
r832 re-encode dir (outputs/ctt_v2/encodes/EFFECTDATA_r832/{latents,cond_clean}/<basename>).

Everything else about the inventory is preserved byte-for-byte: every other clip field
(group, conditions, caption, endpoints, caption_sources), the `groups` map, `build_drops`,
`counts`, and the top-level schema/stratum/kind/endpoint_disjointness. Only `provenance` is
extended (append `derived_from`, the r832 roster + sha, encode_dir, grids, spec).

S6 reshape campaign — misc/2026-08-30_s6_reshape DOSSIER Round 1. Additive; writes a NEW file,
never mutates S6.json or the native EFFECTDATA encode dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S6_IN = REPO / "outputs/ctt_v2/inventories/S6.json"
S6_OUT = REPO / "outputs/ctt_v2/inventories/S6_r832.json"
ROSTER_R832 = REPO / "outputs/ctt_v2/encodes/EFFECTDATA_r832/ROSTER.json"
ENC_R = REPO / "outputs/ctt_v2/encodes/EFFECTDATA_r832"
ROSTER_R832_SHA = "c66c64776ac01c809cad7606a7a0e030f048632f6613e8480690a236ca942535"
EXPECT_CLIPS = 28644
EXPECT_GROUPS = 2917
EXPECT_GRIDS = {"[11, 16, 26]": 14523, "[11, 26, 16]": 14121}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    inv = json.load(open(S6_IN))
    assert inv["schema"] == "ctt_v2_stratum_inventory/1", inv.get("schema")
    assert inv["stratum"] == "S6", inv.get("stratum")

    roster = json.load(open(ROSTER_R832))
    got_sha = sha256_file(ROSTER_R832)
    assert got_sha == ROSTER_R832_SHA, f"roster sha {got_sha} != {ROSTER_R832_SHA}"
    grids = Counter(tuple(c["latent_fhw"]) for c in roster["clips"])
    grids_str = {str(list(k)): v for k, v in sorted(grids.items())}
    assert grids_str == EXPECT_GRIDS, f"grids {grids_str} != {EXPECT_GRIDS}"

    clips = inv["clips"]
    assert len(clips) == EXPECT_CLIPS, len(clips)
    assert len(inv["groups"]) == EXPECT_GROUPS, len(inv["groups"])

    lat_dir = ENC_R / "latents"
    cc_dir = ENC_R / "cond_clean"
    rewritten = 0
    exists = 0
    for stem, c in clips.items():
        base_lat = os.path.basename(c["latents"])
        base_cc = os.path.basename(c["cond_clean"])
        new_lat = str(lat_dir / base_lat)
        new_cc = str(cc_dir / base_cc)
        c["latents"] = new_lat
        c["cond_clean"] = new_cc
        rewritten += 2
        assert os.path.isfile(new_lat), f"missing latents: {new_lat}"
        assert os.path.isfile(new_cc), f"missing cond_clean: {new_cc}"
        exists += 2

    s6_in_sha = sha256_file(S6_IN)
    prov = dict(inv["provenance"])
    prov["derived_from"] = {"file": "outputs/ctt_v2/inventories/S6.json", "sha256": s6_in_sha}
    prov["roster"] = "outputs/ctt_v2/encodes/EFFECTDATA_r832/ROSTER.json"
    prov["roster_sha256"] = ROSTER_R832_SHA
    prov["encode_dir"] = "outputs/ctt_v2/encodes/EFFECTDATA_r832"
    prov["grids"] = grids_str
    prov["spec"] = "misc/2026-08-30_s6_reshape DOSSIER Round 1"
    inv["provenance"] = prov

    with open(S6_OUT, "w") as f:
        json.dump(inv, f, indent=1)
        f.write("\n")

    print(f"[derive_r832] clips {len(clips)} / groups {len(inv['groups'])} / "
          f"paths rewritten {rewritten} / exists {exists}")
    print(f"[derive_r832] wrote {S6_OUT}")
    assert len(clips) == EXPECT_CLIPS and len(inv["groups"]) == EXPECT_GROUPS
    assert rewritten == 2 * EXPECT_CLIPS == 57288 and exists == 57288


if __name__ == "__main__":
    sys.exit(main())
