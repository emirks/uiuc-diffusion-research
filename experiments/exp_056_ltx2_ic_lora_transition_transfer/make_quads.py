"""exp_056 — build the post-training IC-LoRA test-quadruple matrix.

A quadruple = (endpoints clip E, reference clip R, prompt = E's type-blind
caption + ICTRANS, generated video). Pairing groups:

  ic_inclass_new     E,R same class, pairing NOT in the training set
                     (possible for classes with n>=5: shadow_smoke, firelava,
                     earth_wave — circulant training used refs i+1..i+3)
  ic_inclass_trained E,R same class, trained pairing (n<=4 classes: all
                     ordered pairs were trained)
  ic_cross           R's class != E's class — every class serves as reference
                     on 2 foreign endpoint sets (the core transfer claim)
  ic_unseen          R = jump_transition_1 (class never trained)
  ic_reverse         E = jump_transition_1 endpoints (never seen as target),
                     R = shadow_smoke
  base_*             base-model twins of a fixed subset (no LoRA) — anchors

Outputs (deterministic, no RNG):
  dataset/quads.json          [{id, arm, endpoints, reference, prompt, ...}]
  dataset/manifest_ic.json    harness manifest (+ condition_reference for the
                              quadruple viewer; style = REFERENCE's class)
  prints the endpoint-source clip list (for cond_*_start9/end9 cuts)
"""

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
STD = REPO_ROOT / "data/processed/transitions_std121"
OUT_REL = "outputs/videos/exp_056_ltx2_ic_lora_transition_transfer"

MAX_REFS = 3  # must match build_dataset.py circulant scheme

captions = json.loads((EXP / "dataset/captions.json").read_text())
classes = {}
for fam_dir in sorted(p for p in STD.iterdir() if p.is_dir()):
    stems = sorted(p.stem for p in fam_dir.glob("*.mp4"))
    if stems:
        classes[fam_dir.name] = stems

fam_of = {s: f for f, stems in classes.items() for s in stems}


def trained_refs(fam: str, target: str) -> set[str]:
    stems = classes[fam]
    n = len(stems)
    i = stems.index(target)
    return {stems[(i + j) % n] for j in range(1, min(MAX_REFS, n - 1) + 1)}


def quad(arm: str, e: str, r: str) -> dict:
    assert e in fam_of and r in fam_of, (e, r)
    return {
        "id": f"{arm}__{e}__ref_{r}",
        "arm": arm,
        "endpoints": e, "endpoints_class": fam_of[e],
        "reference": r, "reference_class": fam_of[r],
        "prompt": "ICTRANS " + captions[e],
    }


quads = []

# --- ic_inclass_new: untrained same-class pairings (n>=5 classes) ---
for e, r in [("shadow_smoke_0", "shadow_smoke_9"), ("shadow_smoke_5", "shadow_smoke_1"),
             ("firelava_0", "firelava_5"), ("firelava_3", "firelava_1"),
             ("earth_wave_0", "earth_wave_4"), ("earth_wave_3", "earth_wave_2")]:
    assert r not in trained_refs(fam_of[e], e) and r != e, (e, r)
    quads.append(quad("ic_inclass_new", e, r))

# --- ic_inclass_trained: one per remaining class (all pairs trained there) ---
for e, r in [("air_bending_1", "air_bending_2"), ("water_bending_0", "water_bending_1"),
             ("raven_transition_0", "raven_transition_1"), ("melt_transition_1", "melt_transition_2"),
             ("flying_cam_transition_1", "flying_cam_transition_2"),
             ("display_transition_1", "display_transition_2"),
             ("flame_transition_0", "flame_transition_1")]:
    assert r in trained_refs(fam_of[e], e), (e, r)
    quads.append(quad("ic_inclass_trained", e, r))

# --- ic_cross: each trained class as reference on 2 foreign endpoint sets ---
CROSS_FAMS = [f for f in classes if f != "jump_transition"]  # 10, sorted
for k, rf in enumerate(CROSS_FAMS):
    r = classes[rf][min(1, len(classes[rf]) - 1)]  # second clip of the class (or only)
    for off in (1, 4):  # two foreign endpoint classes, rotating
        ef = CROSS_FAMS[(k + off) % len(CROSS_FAMS)]
        e = classes[ef][0]
        quads.append(quad("ic_cross", e, r))

# --- ic_unseen: jump reference (never trained) on 4 endpoint sets ---
for e in ["earth_wave_0", "shadow_smoke_0", "firelava_2", "melt_transition_3"]:
    quads.append(quad("ic_unseen", e, "jump_transition_1"))

# --- ic_reverse: jump endpoints (never a target), smoke reference ---
quads.append(quad("ic_reverse", "jump_transition_1", "shadow_smoke_3"))

assert len({q["id"] for q in quads}) == len(quads)

# --- base twins: fixed informative subset, one per group flavour ---
BASE_TWINS = ["ic_inclass_new__shadow_smoke_0__ref_shadow_smoke_9",
              "ic_inclass_new__firelava_0__ref_firelava_5",
              "ic_inclass_trained__air_bending_1__ref_air_bending_2",
              "ic_cross__earth_wave_0__ref_shadow_smoke_1",
              "ic_cross__melt_transition_1__ref_earth_wave_1",
              "ic_cross__shadow_smoke_0__ref_raven_transition_1",
              "ic_unseen__earth_wave_0__ref_jump_transition_1",
              "ic_reverse__jump_transition_1__ref_shadow_smoke_3"]
by_id = {q["id"]: q for q in quads}
for bid in BASE_TWINS:
    src = by_id[bid]  # KeyError = the subset drifted from the matrix; fix here
    b = dict(src)
    b["arm"] = "base_" + src["arm"].removeprefix("ic_")
    b["id"] = b["arm"] + "__" + src["endpoints"] + "__ref_" + src["reference"]
    quads.append(b)

(EXP / "dataset/quads.json").write_text(json.dumps(quads, indent=2))

# --- harness manifest (style = reference's class; + condition_reference) ---
manifest = []
for q in quads:
    ec, e = q["endpoints_class"], q["endpoints"]
    manifest.append({
        "item_id": q["id"],
        "generated_video": f"{OUT_REL}/{'base' if q['arm'].startswith('base_') else 'ic_lora'}/quads/{q['id']}.mp4",
        "style": q["reference_class"],
        "n_endpoints": 2,
        "condition_prefix": {
            "video": f"experiments/exp_056_ltx2_ic_lora_transition_transfer/dataset/cond_q/{e}_start9.mp4",
            "num_frames": 9},
        "condition_suffix": {
            "video": f"experiments/exp_056_ltx2_ic_lora_transition_transfer/dataset/cond_q/{e}_end9.mp4",
            "num_frames": 8},
        "condition_reference": {
            "video": f"data/processed/transitions_std121/{q['reference_class']}/{q['reference']}.mp4"},
        "arm": q["arm"],
        "notes": f"endpoints={e} ({ec}); reference={q['reference']} ({q['reference_class']}); "
                 f"type-blind prompt, transition style only in reference",
    })
(EXP / "dataset/manifest_ic.json").write_text(json.dumps(manifest, indent=2))

ep_clips = sorted({q["endpoints"] for q in quads})
arms = {}
for q in quads:
    arms[q["arm"]] = arms.get(q["arm"], 0) + 1
print(f"[done] {len(quads)} quads -> dataset/quads.json (+manifest_ic.json)")
print(f"[arms] {json.dumps(arms, indent=0)}")
print(f"[endpoint sources needing cond cuts] ({len(ep_clips)}):")
for e in ep_clips:
    print("  ", fam_of[e] + "/" + e)
