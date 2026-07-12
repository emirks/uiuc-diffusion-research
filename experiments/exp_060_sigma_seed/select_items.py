"""exp_060 — deterministic σ_seed probe-item selection (SPEC §4/§6.4, O6).

Selection rule (frozen here; no cherry-picking — same spirit as the anchor rule):

  1. Restrict to n>=4-eligible classes (the exam's trust convention).
  2. Strata = (twosided, onesided) x (camera-tagged, object-tagged,
     style/untagged), membership by TAG PRESENCE (a class with tags
     [camera,object] is a candidate in BOTH camera and object strata;
     'style/untagged' = 'style' in tags OR no tags at all). Fixed stratum
     order: twosided-{camera,object,style/untagged}, then
     onesided-{camera,object,style/untagged}.
  3. Fill 12 slots by cycling the strata in that fixed order (round-robin),
     each visit taking the FIRST LEXICOGRAPHIC class in that stratum NOT
     already picked globally; skip exhausted/empty strata. (Second visit to a
     stratum therefore takes its second-lexicographic class, etc.)
  4. For each picked class the probe item = the class's BAR PAIR
     (max-endpoint-distance pair) exactly as certification defines it
     (certify/probes.sibling_pairs). Clip A = generated_video of sib__<class>
     (the endpoints / conditioning source), clip B = reference_video (the
     reference). Read from the certified certification artifact so it is
     byte-identical to what the instrument used; bar-pair selection is
     corpus-only + pre-freeze, so the draft.8 artifact == the eval/v3.0.0 tag.

Writes dataset/selection.json (picks + strata + bar pairs + captions +
provenance). Login-node / CPU only.
"""

import collections
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
CORPUS = REPO_ROOT / "data/processed/transitions_std121/corpus_manifest.json"
# certified certification artifact (draft.8 run == v3.0.0 by regrade; bar pairs
# are corpus-only pre-freeze calibration, unchanged between draft.8 and 3.0.0).
SIBLINGS = REPO_ROOT / "outputs/eval/certification/3.0.0-draft.8/manifests/siblings.json"
CAPTION_FILES = [
    "experiments/exp_056_ltx2_ic_lora_transition_transfer/dataset/captions.json",
    "experiments/exp_057_ic_lora_unseen_eval/dataset/captions.json",
    "experiments/exp_058_ic_lora_diverse_retrain/dataset/captions.json",
]

SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    corpus = json.loads(CORPUS.read_text())
    classes = corpus["classes"]
    n_by = collections.Counter(corpus["clips"][k]["class"] for k in corpus["clips"])

    elig = sorted(c for c in classes if n_by[c] >= 4)

    def has(c, tag):
        return tag in classes[c].get("tags", [])

    def sided(c):
        return classes[c]["sidedness"]

    def style_or_untagged(c):
        return has(c, "style") or not classes[c].get("tags")

    strata_defs = [
        ("twosided-camera", lambda c: sided(c) == "twosided" and has(c, "camera")),
        ("twosided-object", lambda c: sided(c) == "twosided" and has(c, "object")),
        ("twosided-style/untagged", lambda c: sided(c) == "twosided" and style_or_untagged(c)),
        ("onesided-camera", lambda c: sided(c) == "onesided" and has(c, "camera")),
        ("onesided-object", lambda c: sided(c) == "onesided" and has(c, "object")),
        ("onesided-style/untagged", lambda c: sided(c) == "onesided" and style_or_untagged(c)),
    ]
    strata = {name: [c for c in elig if fn(c)] for name, fn in strata_defs}

    picked, pick_stratum = [], []
    guard = 0
    while len(picked) < 12 and guard < 50:
        progressed = False
        for name, _fn in strata_defs:
            if len(picked) >= 12:
                break
            cand = [c for c in strata[name] if c not in picked]
            if cand:
                picked.append(cand[0])
                pick_stratum.append(name)
                progressed = True
        guard += 1
        if not progressed:
            break
    assert len(picked) == 12, f"selection produced {len(picked)} picks, expected 12"

    sib = {r["item_id"]: r for r in json.loads(SIBLINGS.read_text())}
    captions = {}
    for f in CAPTION_FILES:
        p = REPO_ROOT / f
        if p.exists():
            captions.update(json.loads(p.read_text()))

    items = []
    for cls, stratum in zip(picked, pick_stratum):
        r = sib[f"sib__{cls}"]
        clip_a = pathlib.Path(r["generated_video"]).stem   # endpoints/conditioning
        clip_b = pathlib.Path(r["reference_video"]).stem    # reference
        items.append({
            "class": cls,
            "stratum": stratum,
            "sidedness": classes[cls]["sidedness"],
            "tags": classes[cls].get("tags", []),
            "n_clips": n_by[cls],
            "clip_a": clip_a,       # conditioning source (prefix9 + suffix8)
            "clip_b": clip_b,       # reference
            "endpoint_dist": r["notes"],
            "caption_a": captions.get(clip_a),   # None if missing (filled by caption_missing.py)
        })

    out = {
        "selection_rule": __doc__,
        "seeds": SEEDS,
        "n_items": len(items),
        "strata_members": strata,
        "eligible_classes_n_ge_4": elig,
        "siblings_artifact": str(SIBLINGS.relative_to(REPO_ROOT)),
        "items": items,
    }
    (EXP / "dataset/selection.json").write_text(json.dumps(out, indent=2))
    print(f"[done] {len(items)} picks -> dataset/selection.json")
    for i, it in enumerate(items, 1):
        cap = "cap:OK" if it["caption_a"] else "cap:MISSING"
        print(f"{i:2d}. {it['class']:22s} [{it['stratum']:24s}] "
              f"A={it['clip_a']:24s} B={it['clip_b']:24s} {cap}")


if __name__ == "__main__":
    main()
