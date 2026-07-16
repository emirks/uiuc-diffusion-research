"""exp_063 — build the R4/R5 generation manifest for the generalist IC-LoRA
(exp_058 ic2 step_05000), per docs/eval_ladder/PLAN.md §1/§3/§5.

R4 = generalist on TRAINED classes, R5 = generalist on HELD-OUT classes (zero-shot).
Targets = each class's 2 split-v1 test clips (== exp_061 items → prompt parity, cond
cuts already on disk in exp_061). Reference = the fixed per-class clip from the frozen
grid. Native keyed mode: suffix condition ON iff sidedness_key is two_sided.
hero_flight R5 rows are marked deferred (wait on sidedness validation, PLAN §B1).

Usage: python experiments/exp_063_ladder_r4r5_generalist/build_manifests.py
Output: dataset/ladder_r4r5.json  (rows consumed by run_ic_inference.py, seeds added at gen).
"""
import hashlib, json, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
GRID = REPO / "docs/eval_ladder/ladder_items_v1.json"
SEL = REPO / "experiments/exp_061_ladder_r0_r1/dataset/selection.json"
COND_DIR = "experiments/exp_061_ladder_r0_r1/dataset/cond"
ADAPTER = REPO / "outputs/training/exp_058_ic_lora_diverse_retrain/ic2/checkpoints/lora_weights_step_05000.safetensors"
OUT = EXP / "dataset/ladder_r4r5.json"


def main() -> None:
    grid_doc = json.loads(GRID.read_text())
    grid = grid_doc["classes"]
    sel = json.loads(SEL.read_text())
    items = sel["items"] if isinstance(sel, dict) else sel
    prompt_by_clip = {it["clip"]: it["caption"] for it in items}

    assert ADAPTER.exists(), f"ic2 adapter missing: {ADAPTER}"
    sha = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()

    rows = []
    for cls, g in grid.items():
        rung = g["generalist_rung"]          # R4 or R5
        if rung not in ("R4", "R5"):
            continue
        prefix_only = (g["sidedness_key"] != "two_sided")
        for clip in g["test_items"]:
            cap = prompt_by_clip.get(clip)
            assert cap, f"no exp_061 prompt for {clip}"
            rows.append({
                "id": f"{rung}__{cls}__{clip}",
                "rung": rung,
                "class": cls,
                "clip": clip,
                "endpoints": clip,
                "cond_dir": COND_DIR,
                "prefix_only": prefix_only,
                "reference_class": cls,
                "reference": g["reference"],
                "prompt": f"ICTRANS {cap}",
                "label": "ic2",
                "endpoint_seen_by_ic2": g["endpoint_seen_by_ic2"][clip],
                "sidedness_key": g["sidedness_key"],
                "sidedness_key_source": g["sidedness_key_source"],
                "deferred": bool(g.get("waits_on_sidedness_validation")),
            })

    doc = {
        "version": "ladder_r4r5_v1",
        "adapter": str(ADAPTER.relative_to(REPO)),
        "adapter_sha256": sha,
        "seeds": grid_doc["seeds"],
        "recipe": "480x640x121@24, 30 steps, CFG 4.0, STG 1.0 stg_v[29]; reference+prefix9+(suffix8 iff two_sided); exp_058 run_ic_inference path, ic2 native keyed",
        "cond_dir": COND_DIR,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2))
    active = [r for r in rows if not r["deferred"]]
    print(f"[done] {OUT}  adapter_sha256={sha[:12]}…")
    print(f"rows={len(rows)} active(now)={len(active)} deferred={len(rows)-len(active)}")
    for r in rows:
        d = " [DEFERRED]" if r["deferred"] else ""
        print(f"  {r['id']:38s} ref={r['reference']:22s} prefix_only={r['prefix_only']} "
              f"seen_ic2={r['endpoint_seen_by_ic2']}{d}")


if __name__ == "__main__":
    main()
