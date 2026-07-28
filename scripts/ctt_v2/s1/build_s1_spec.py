#!/usr/bin/env python3
"""Build S1's group spec + assembled training captions from what actually landed on disk.

S1 arrived as TWO layers (owner, 2026-07-28), 1,417 clips total:

  `ctt_v2_s1`       390 clips on FOREIGN endpoints (pool clips: humanvid / openvid / davis)
  `ctt_v2_s1_s0cf` 1,027 S0-endpoint counterfactuals (the 139 certified corpus clips)

Endpoints are derived from the FILENAME, never from the pinned grid, because 55 of the 390
foreign clips are outside both grids (DOSSIER §A22.2 — they resolve 55/55 against the store's
endpoint universe, including the 76 paid-for orphan descriptions). Deriving from disk is also
the only thing that stays true if the run is extended again.

Filename parsing is done by MATCHING AGAINST THE KNOWN ENDPOINT UNIVERSE, not by splitting on
`__`: openvid ids can themselves contain `__`, and a naive split silently produces corrupt keys
like `openvid|A` (it did, on the first attempt). The arm comes from the DIRECTORY, which is
authoritative, and sidedness from `eval_ladder/registry.jsonl`.

Caption sources differ by layer, and this is the load-bearing distinction:

  foreign   endpoints are pool clips with per-(clip, role) descriptions in the locked store, so
            the caption assembles as `{A}. sksz.` (one-sided) or `{A}. sksz. {B}.` (two-sided).
  s0cf      the endpoint is a CERTIFIED S0 CORPUS CLIP. An S0 clip is a complete transition, so
            its opening is the A anchor and its closing is the B anchor — one endpoint supplies
            BOTH, which is why a two-sided s0cf filename carries only one id. Verified by
            measurement (§A22.3): the two two-sided arms splice their LAST frame from the source
            (MAE 4.2-4.3) while the nine one-sided arms generate it freely (MAE 51-125).
            Its caption is therefore the certified caption VERBATIM — already in `{desc}. sksz.`
            form — and `caption_sources` is explicitly `[]`, exactly as `kind == "corpus"`
            returns for S0 itself, because it draws on the certified 139 and NOT on the
            per-(clip, role) store.

Usage:
    python scripts/ctt_v2/s1/build_s1_spec.py --out outputs/ctt_v2/inventories/S1_spec.json \
        --captions-out outputs/ctt_v2/captions/S1_CAPTIONS_ASSEMBLED.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
sys.path.insert(0, str(REPO / "scripts/ctt_v2/encode"))
import root_common as rc  # noqa: E402

LAYERS = {"foreign": REPO / "outputs/videos/ctt_v2_s1",
          "s0cf": REPO / "outputs/videos/ctt_v2_s1_s0cf"}
LOCKED = REPO / "outputs/ctt_v2/captions/CAPTION_STORE.json"
S0_CAPS = REPO / "eval_ladder/dataset/captions/dataset_captions.json"
S0_INV = REPO / "outputs/ctt_v2/inventories/S0.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--captions-out", required=True)
    args = ap.parse_args()

    import encode_strata as es
    sided_of = es.s1_sidedness()

    locked_doc = json.loads(LOCKED.read_text())
    locked = locked_doc["descriptions"]
    orphans = locked_doc.get("orphans") or {}
    # orphans are paid-for descriptions the pinned grid did not consume; the 55 out-of-grid
    # foreign clips consume exactly them, so they are IN the lookup here and recorded as used.
    lookup = {**orphans, **locked}
    s0_caps = {Path(r["video"]).stem: r["caption"]
               for r in json.loads(S0_CAPS.read_text())}
    s0_inv = json.loads(S0_INV.read_text())["clips"]
    universe = sorted({k.rsplit("|", 1)[0] for k in lookup} | set(s0_caps),
                      key=len, reverse=True)

    downgraded: list[str] = []
    groups: dict[str, dict] = {}
    endpoints: dict[str, list] = {}
    cap_sources: dict[str, list] = {}
    captions: dict[str, str] = {}
    layer_of: dict[str, str] = {}
    used_orphans: set[str] = set()
    problems: list[str] = []

    for layer, root in LAYERS.items():
        for p in sorted(root.glob("spec_*/*.mp4")):
            arm, stem = p.parent.name, p.stem
            sided = sided_of[arm]
            mid = re.sub(r"__s\d+$", "", stem[len(arm) + 2:])
            # longest-first match; for a two-endpoint stem, strip the first hit and match again
            eps: list[str] = []
            rest = mid
            while rest:
                hit = next((e for e in universe if rest.startswith(e)), None)
                if hit is None:
                    break
                eps.append(hit)
                rest = rest[len(hit):].lstrip("_")
            if not eps or rest:
                problems.append(f"{layer}/{stem}: parsed {eps} leftover {rest!r}")
                continue
            # ---- SIDEDNESS IS PER CLIP, NOT PER ARM ----------------------------------------
            # The registry gives the arm's INTENDED sidedness; what a clip actually got is a
            # property of the clip, and each rule below is backed by a splice measurement
            # (last-frame MAE against the endpoint's own source; DOSSIER A22.3/A22.6):
            #   foreign, 2 endpoints  -> two-sided (A and B are distinct pool clips)
            #   foreign, 1 endpoint   -> ONE-sided, measured MAE 71-92 on the tail. 10 clips in
            #                            the two two-sided arms landed here: the runner had no
            #                            second endpoint and fell back, so the tail is freely
            #                            generated. 4 of them even HAVE a B description in the
            #                            store and still did not use it -- which is why the
            #                            store is not evidence and the pixels are.
            #   s0cf, 1 endpoint      -> the arm's registry sidedness, measured 11/11: an S0
            #                            clip is a complete transition, so its own closing
            #                            window serves as the suffix anchor (MAE 4.2 vs 51-125).
            if layer == "foreign":
                clip_sided = "two" if len(eps) == 2 else "one"
            else:
                clip_sided = sided
            if clip_sided != sided:
                downgraded.append(f"{arm}/{stem}: arm={sided} -> clip={clip_sided}")
            sided = clip_sided

            if layer == "s0cf":
                cap = s0_caps.get(eps[0])
                if cap is None:
                    problems.append(f"{stem}: no certified S0 caption for {eps[0]}")
                    continue
                captions[stem] = cap                 # already `{desc}. sksz.`
                cap_sources[stem] = []               # certified corpus, not the (clip,role) store
            else:
                parts = []
                roles = ["A", "B"][:len(eps)] if sided == "two" else ["A"]
                for i, e in enumerate(eps[:len(roles)]):
                    role = roles[i]
                    k = f"{e}|{role}"
                    if k not in lookup:
                        problems.append(f"{stem}: store has no {k!r}")
                        break
                    if k in orphans:
                        used_orphans.add(k)
                    parts.append(lookup[k])
                else:
                    cap = f"{parts[0]}.{rc.TRIGGER_SENTENCE}"
                    if sided == "two" and len(parts) > 1:
                        cap += f" {parts[1]}."
                    captions[stem] = cap
                    cap_sources[stem] = [[e, roles[i]] for i, e in enumerate(eps[:len(roles)])]

            # A group carries ONE sidedness (it drives the mask), and sidedness now varies
            # within an arm, so the group id is (arm, sided). The arm survives in the id.
            gid = arm if sided == sided_of[arm] else f"{arm}__1sided"
            groups.setdefault(gid, {"class": None, "shader": None, "sided": sided,
                                    "clips": []})["clips"].append(stem)
            endpoints[stem] = eps
            layer_of[stem] = layer

    if problems:
        raise SystemExit(f"[s1-spec] {len(problems)} unresolved clip(s); refusing to write a "
                         f"partial spec. First 8:\n  " + "\n  ".join(problems[:8]))

    filt = rc.leak_filter()
    bad = {k: v for k, v in ((k, rc.caption_violations(c, filt))
                             for k, c in captions.items()) if v}
    if bad:
        raise SystemExit(f"[s1-spec] {len(bad)} caption(s) violate RULING 9: "
                         + json.dumps(dict(list(bad.items())[:4]), indent=1))

    for g in groups.values():
        g["clips"].sort()
    spec = {
        "stratum": "S1",
        "kind": "synthetic_op",
        "endpoint_disjointness": False,
        "endpoint_disjointness_reason":
            "the s0cf layer is SELF-endpointed on one S0 clip (its opening is A, its closing is "
            "B), so a clip trivially shares an endpoint with itself; and both layers reuse pool "
            "endpoints across arms by design (11 arms x the same endpoint set is the point).",
        "groups": dict(sorted(groups.items())),
        "endpoints": dict(sorted(endpoints.items())),
        "caption_sources": dict(sorted(cap_sources.items())),
        "provenance": {
            "layers": {k: str(v.relative_to(REPO)) for k, v in LAYERS.items()},
            "n_by_layer": {la: sum(1 for v in layer_of.values() if v == la) for la in LAYERS},
            "sided_authority": "eval_ladder/registry.jsonl per arm; CONFIRMED on disk by "
                               "first/last-frame splice MAE (DOSSIER A22.3): one-sided arms "
                               "51-125 on the tail, two-sided 4.2-4.3",
            "endpoint_derivation": "matched against the known endpoint universe (locked store + "
                                   "76 orphans + 139 certified S0 clips), NOT by splitting on "
                                   "'__' -- openvid ids contain '__' and a split corrupts them",
            "locked_store_content_hash": locked_doc["content_hash"],
            "orphan_descriptions_consumed": sorted(used_orphans),
            "n_orphan_descriptions_consumed": len(used_orphans),
            "per_clip_sidedness_downgrades": downgraded,
            "n_per_clip_sidedness_downgrades": len(downgraded),
        },
    }
    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1))

    cp = Path(args.captions_out)
    cp = cp if cp.is_absolute() else REPO / cp
    cp.write_text(json.dumps({k: captions[k] for k in sorted(captions)}, indent=1))
    h = hashlib.sha256(json.dumps({k: captions[k] for k in sorted(captions)},
                                  sort_keys=True).encode()).hexdigest()

    print(f"[ok] {out.relative_to(REPO)}")
    print(f"     {len(groups)} arms / {len(endpoints)} clips  "
          f"({spec['provenance']['n_by_layer']})")
    two = [a for a, g in groups.items() if g["sided"] == "two"]
    print(f"     two-sided arms: {two} -> {sum(len(groups[a]['clips']) for a in two)} clips")
    print(f"     orphan descriptions consumed: {len(used_orphans)}")
    print(f"     per-clip sidedness downgrades (arm=two -> clip=one): {len(downgraded)}")
    for d in downgraded[:4]:
        print(f"       {d}")
    print(f"[ok] {cp.relative_to(REPO)}: {len(captions)} captions, "
          f"{len(set(captions.values()))} distinct, 0 RULING 9 violations")
    print(f"     content_hash sha256:{h[:16]}")


if __name__ == "__main__":
    main()
