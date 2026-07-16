"""exp_063 — Amendment 1: build the R4X manifest (generalist twin of R3X, contrast C9).

For each recipient in ladder_items_v2.json r3x block, one row per donor: the generalist
(ic2 step_05000) conditioned on the RECIPIENT's grid reference clip + the DONOR's prefix
endpoint (start9), recipient keying (prefix_only — all B8 recipients are one_sided). Same
donor endpoints as R3X ⇒ the C9 delta (R3X−R4X) is apples-to-apples. No ground truth.

Output rows share run_ic_inference.py's schema (+ recipient/donor/twin_of metadata), so
the same runner generates them: `run_ic_inference.py --manifest dataset/ladder_r4x.json`.
Usage: python experiments/exp_063_ladder_r4r5_generalist/build_r4x_manifest.py
"""
import hashlib, json, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
GRID = REPO / "docs/eval_ladder/ladder_items_v2.json"
SEL = REPO / "experiments/exp_061_ladder_r0_r1/dataset/selection.json"
COND_DIR = "experiments/exp_062_ladder_r2r3_specialists/dataset/cond"  # donor start9 cuts live here
ADAPTER = REPO / "outputs/training/exp_058_ic_lora_diverse_retrain/ic2/checkpoints/lora_weights_step_05000.safetensors"
OUT = EXP / "dataset/ladder_r4x.json"


def main() -> None:
    grid_doc = json.loads(GRID.read_text())
    r3x = grid_doc["r3x"]["recipients"]
    sel = json.loads(SEL.read_text())
    items = sel["items"] if isinstance(sel, dict) else sel
    prompt_by_clip = {it["clip"]: it["caption"] for it in items}

    assert ADAPTER.exists(), f"ic2 adapter missing: {ADAPTER}"
    sha = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()

    rows = []
    for recipient, spec in r3x.items():
        ref = spec["recipient_reference"]
        for d in spec["donors"]:
            clip = d["donor_clip"]
            cap = prompt_by_clip.get(clip)
            assert cap, f"no exp_061 prompt for donor clip {clip}"
            rows.append({
                "id": f"R4X__{recipient}__{clip}",
                "rung": "R4X",
                "recipient": recipient,
                "donor_class": d["donor_class"],
                "class": recipient,               # style/mask = recipient (scoring contract)
                "clip": clip,
                "endpoints": clip,                # donor prefix endpoint
                "cond_dir": COND_DIR,
                "prefix_only": True,              # all B8 recipients are one_sided
                "reference_class": recipient,     # recipient's grid reference clip
                "reference": ref,
                "prompt": f"ICTRANS {cap}",
                "label": "ic2",
                "endpoint_seen_by_ic2": bool(d["endpoint_seen_by_ic2"]),
                "twin_of": f"R3X__{recipient}__{clip}",
                "deferred": False,
            })

    doc = {
        "version": "ladder_r4x_v1",
        "amendment": "PLAN Amendment 1 (C9): generalist twin of R3X",
        "adapter": str(ADAPTER.relative_to(REPO)),
        "adapter_sha256": sha,
        "seeds": grid_doc["seeds"],
        "recipe": "480x640x121@24, 30 steps, CFG 4.0, STG 1.0 stg_v[29]; reference(recipient)+prefix9(donor); ic2 native keyed (prefix_only, recipients one_sided)",
        "cond_dir": COND_DIR,
        "scoring_contract": grid_doc["r3x"]["scoring_contract"],
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"[done] {OUT}  adapter_sha256={sha[:12]}…  rows={len(rows)} (x{len(doc['seeds'])} seeds = {len(rows)*len(doc['seeds'])} videos)")
    for r in rows:
        print(f"  {r['id']:34s} ref={r['reference']:22s} donor={r['donor_class']:20s} seen_ic2={r['endpoint_seen_by_ic2']}")


if __name__ == "__main__":
    main()
