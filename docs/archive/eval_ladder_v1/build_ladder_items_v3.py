#!/usr/bin/env python3
"""Grid v3 — the unified tier system (PLAN Amendment 2), tier-first.

Every eval row = (model, tier): base {P, PE-keyed}; specialist {seen,
unseen_own, unseen_foreign}; generalist ic3 {A, B, C, X}. Legacy rung names
kept as output-dir aliases. Emits:
  docs/eval_ladder/ladder_items_v3.json            (grid: meta, classes, r3x 11)
  experiments/exp_065_ladder_v3_grid/dataset/manifest_base_ext.json
  experiments/exp_065_ladder_v3_grid/dataset/manifest_ic3.json      (A+B+C)
  experiments/exp_065_ladder_v3_grid/dataset/manifest_ic3_x.json    (X)
plus any missing cond clips (ffmpeg, exp_061 recipe, idempotent).

Deterministic: all draws seeded (`ladder_v3:donors:{recipient}`), all picks
first-lexicographic. Rerunning reproduces byte-identical manifests.
"""
import json, pathlib, random, subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
STD = REPO / "data/processed/transitions_std121"
E61 = REPO / "experiments/exp_061_ladder_r0_r1"
E65 = REPO / "experiments/exp_065_ladder_v3_grid"
FFMPEG = str(REPO.parent / "LTX-2-official/.venv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")

split = json.loads((STD / "split_v1.1.json").read_text())["classes"]
corpus = json.loads((STD / "corpus_manifest.json").read_text())["classes"]
v2 = json.loads((REPO / "docs/eval_ladder/ladder_items_v2.json").read_text())
sel = {i["clip"]: i for i in json.loads((E61 / "dataset/selection.json").read_text())["items"]}
caps = {}
for p in [REPO / "experiments/exp_058_ic_lora_diverse_retrain/dataset/captions.json",
          REPO / "experiments/exp_060_sigma_seed/dataset/captions_extra.json",
          REPO / "experiments/exp_061_ladder_r0_r1/dataset/captions_extra.json",
          REPO / "experiments/exp_062_ladder_r2r3_specialists/dataset/captions_r2.json"]:
    if p.exists():
        caps.update(json.loads(p.read_text()))
for c, i in sel.items():
    caps.setdefault(c, i["caption"])

HOLDOUT = {"hero_flight", "illustration_scene", "gas_transformation",
           "raven_transition", "hole_transition", "seamless_transition", "jump_transition"}
C5 = ["animalization", "color_rain", "polygon", "portal", "shadow",
      "shadow_smoke", "super_fast_run", "wireframe"]
R1K9 = set(v2["r1k"]) if isinstance(v2.get("r1k"), list) else {
    "shadow", "portal", "super_fast_run", "polygon", "wireframe",
    "animalization", "color_rain", "gas_transformation", "illustration_scene"}
IC3 = "outputs/training/exp_064_ic3_aligned_retrain/ic3/checkpoints/lora_weights_step_05000.safetensors"

two = lambda c: corpus[c]["sidedness"] == "twosided"
test_of = lambda c: sorted(split[c]["test"])
train_of = lambda c: sorted(split[c]["train"])
trained = sorted(c for c in split if c not in HOLDOUT)
test_bearing = sorted(c for c in split if test_of(c))

def ref_of(cls, avoid=()):
    if cls in v2["classes"]:
        return v2["classes"][cls]["reference"]
    return next(t for t in train_of(cls) if t not in avoid)

need_conds = set()
def row(model, tier, rung, cls, clip, ref_cls=None, ref=None, extra=None):
    po = not two(cls)
    cond_dir = "experiments/exp_061_ladder_r0_r1/dataset/cond" if clip in sel \
        else "experiments/exp_065_ladder_v3_grid/dataset/cond"
    if clip not in sel:
        need_conds.add((cls, clip, not po))
    r = {"id": f"{rung}__{cls}__{clip}", "model": model, "tier": tier, "rung": rung,
         "class": cls, "clip": clip, "endpoints": clip, "cond_dir": cond_dir,
         "prefix_only": po, "prompt": "ICTRANS " + caps[clip],
         "reference_class": ref_cls, "reference": ref, "deferred": False}
    if extra: r.update(extra)
    return r

# ---- base extension: PE-keyed (prefix-only) for one_sided test-bearing classes not in R1K-9
base_rows = []
for cls in test_bearing:
    if two(cls) or cls in R1K9:
        continue
    for clip in test_of(cls):
        r = row("base", "PE_keyed", "R1K", cls, clip)
        r["prefix_only"] = True
        base_rows.append(r)

# ---- ic3 tiers A / B / C
a_rows, b_rows, c_rows = [], [], []
for cls in C5:  # tier A designated: first-lex train clip != reference
    ref = ref_of(cls)
    item = next(t for t in train_of(cls) if t != ref)
    a_rows.append(row("ic3", "A", "R4A", cls, item, cls, ref))
for cls in trained:  # tier A stand-ins: trained test-less classes' frozen exp_061 items
    if test_of(cls):
        continue
    item = next(c for c in sel if sel[c]["class"] == cls)
    a_rows.append(row("ic3", "A", "R4A", cls, item, cls, ref_of(cls, avoid={item}),
                      extra={"standin": True}))
for cls in trained:  # tier B
    for clip in test_of(cls):
        b_rows.append(row("ic3", "B", "R4B", cls, clip, cls, ref_of(cls)))
for cls in sorted(HOLDOUT):  # tier C: >=1 train (ref) + >=1 test (endpoints)
    if not test_of(cls) or not train_of(cls):
        continue
    for clip in test_of(cls):
        c_rows.append(row("ic3", "C", "R5", cls, clip, cls, ref_of(cls)))

# ---- tier X: 11 recipients x 4 donors (8 verbatim from v2 R4X; 3 drawn)
r4x_v2 = json.loads((REPO / "experiments/exp_063_ladder_r4r5_generalist/dataset/ladder_r4x.json").read_text())["rows"]
x_rows, r3x_block = [], dict(v2["r3x"]["recipients"])
for r in r4x_v2:
    n = dict(r); n["model"], n["tier"], n["label"] = "ic3", "X", "ic3"
    n["endpoint_seen_by_generalist"] = False  # donors are test-band; ic3 never saw them
    x_rows.append(n)
donor_keys = list(v2["r3x"]["recipients"][next(iter(v2["r3x"]["recipients"]))]["donors"][0].keys())
for rec in ["hero_flight", "shadow_smoke", "super_fast_run"]:
    pool = sorted(c for c in test_bearing if c != rec and two(c) == two(rec))
    donors = random.Random(f"ladder_v3:donors:{rec}").sample(pool, 4)
    dl = []
    for d in donors:
        clip = test_of(d)[0]
        dl.append({k: {"donor_class": d, "donor_clip": clip}.get(k) for k in donor_keys})
        x_rows.append({
            "id": f"R4X__{rec}__{clip}", "model": "ic3", "tier": "X", "rung": "R4X",
            "recipient": rec, "donor_class": d, "class": rec, "clip": clip,
            "endpoints": clip, "cond_dir": "experiments/exp_061_ladder_r0_r1/dataset/cond",
            "prefix_only": True,  # twin-consistent with exp_062 R3X (prefix-only by construction)
            "reference_class": rec,
            "reference": ref_of(rec), "prompt": "ICTRANS " + caps[clip],
            "label": "ic3", "endpoint_seen_by_generalist": False,
            "twin_of": f"R3X__{rec}__{clip}", "deferred": False})
    r3x_block[rec] = {"donors": dl, "extension": "A2.5 seeded draw (sidedness-matched)"}

# ---- build missing conds (idempotent)
(E65 / "dataset/cond").mkdir(parents=True, exist_ok=True)
for cls, clip, needs_end in sorted(need_conds):
    src = STD / cls / f"{clip}.mp4"
    jobs = [(f"{clip}_start9.mp4", "select='lt(n,9)'")]
    if needs_end:
        jobs.append((f"{clip}_end9.mp4", "select='gte(n,112)'"))
    for name, vf in jobs:
        dst = E65 / "dataset/cond" / name
        if dst.exists():
            continue
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i",
                        str(src), "-vf", vf + ",setpts=N/24/TB", "-r", "24", "-c:v",
                        "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt",
                        "yuv420p", str(dst)], check=True)
        print(f"[cond] {name}")

# ---- write
grid = {"version": "v3", "amendment": "PLAN Amendment 2", "frozen": True,
        "derived_from": "ladder_items_v2.json + split_v1.1.json",
        "split_file": "data/processed/transitions_std121/split_v1.1.json",
        "seeds": [42, 43, 44], "generalist_adapter": IC3,
        "holdout": sorted(HOLDOUT), "classes": v2["classes"],
        "r3x": {"recipients": r3x_block},
        "tier_counts": {"base_ext": len(base_rows), "A": len(a_rows),
                        "B": len(b_rows), "C": len(c_rows), "X": len(x_rows)}}
(REPO / "docs/eval_ladder/ladder_items_v3.json").write_text(json.dumps(grid, indent=1))
for name, adapter, rows in [("manifest_base_ext.json", None, base_rows),
                            ("manifest_ic3.json", IC3, a_rows + b_rows + c_rows),
                            ("manifest_ic3_x.json", IC3, x_rows)]:
    (E65 / "dataset" / name).write_text(json.dumps({"adapter": adapter, "rows": rows}, indent=1))
    print(f"[done] {name}: {len(rows)} rows x 3 seeds = {len(rows)*3} videos")
print(f"[counts] {grid['tier_counts']}  (x3 seeds)")
