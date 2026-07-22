"""exp_074 — corrected-prompt regeneration manifests (inference-only, no retraining).

Fixes the prompt defects found 2026-07-22 (docs/eval_ladder/PROMPT_REDESIGN.md):
  - one-sided / foreign prompts described the OUTCOME ("The scene transforms
    into <Scene2>") that the conditioning deliberately withholds;
  - foreign (R3X/R4X) prompts were the endpoints clip's OWN full caption,
    directly contradicting the donor transition the arm is graded on.

Corrected prompt rules (prompt states ONLY the endpoints' knowledge):
  prefix-only rows : "ICTRANS <Scene1>"                       (nothing about the
                     transition or the end scene — weights/reference must carry it)
  prefix+suffix    : "ICTRANS <Scene1> <Scene2-capitalized>"  (both scenes are
                     endpoint knowledge; the transition WORDING is removed)

Outputs (dataset/):
  manifest_ic3_cx.json    ic3 adapter — R5 (zero-shot, keyed) + R4X (foreign, prefix-only)
  manifest_r3x_<cls>.json one per specialist class — R3X rows (prefix-only)

Usage: python3 experiments/exp_074_prompt_fix_rerun/build_manifests.py
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
DS65 = REPO / "experiments/exp_065_ladder_v3_grid/dataset"
MARKER = "The scene transforms into "
SPEC_TARGETS = ["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"]
IC3_TARGETS = SPEC_TARGETS + ["ff.net.0.proj", "ff.net.2"]


def load_captions():
    caps = {}
    caps.update(json.loads((REPO / "experiments/exp_058_ic_lora_diverse_retrain/dataset/captions.json").read_text()))
    caps.update(json.loads((REPO / "experiments/exp_062_ladder_r2r3_specialists/dataset/captions_r2.json").read_text()))
    for it in json.loads((REPO / "experiments/exp_061_ladder_r0_r1/dataset/selection.json").read_text())["items"]:
        caps[it["clip"]] = it["caption"]   # exp_061 wins (cross-rung parity, same as exp_062)
    caps.update(json.loads((REPO / "experiments/exp_061_ladder_r0_r1/dataset/captions_extra.json").read_text()))
    return caps


def corrected_prompt(caption: str, prefix_only: bool) -> str:
    assert caption.count(MARKER) == 1, caption[:80]
    s1, s2 = caption.split(MARKER)
    s1 = s1.strip()
    if prefix_only:
        return f"ICTRANS {s1}"
    s2 = s2.strip()
    return f"ICTRANS {s1} {s2[0].upper()}{s2[1:]}"


def main():
    caps = load_captions()
    EXP.joinpath("dataset").mkdir(exist_ok=True)

    # ---- ic3: R5 (zero-shot) + R4X (foreign) under the ic3 adapter ----------
    ic3 = json.loads((DS65 / "manifest_ic3.json").read_text())
    ic3x = json.loads((DS65 / "manifest_ic3_x.json").read_text())
    assert ic3["adapter"] == ic3x["adapter"]
    rows = []
    for r in [x for x in ic3["rows"] if x["rung"] == "R5"] + ic3x["rows"]:
        r = dict(r)
        r["prompt"] = corrected_prompt(caps[r["endpoints"]], r["prefix_only"])
        r.pop("deferred", None)
        rows.append(r)
    doc = {"adapter": ic3["adapter"], "target_modules": IC3_TARGETS, "rows": rows}
    (EXP / "dataset/manifest_ic3_cx.json").write_text(json.dumps(doc, indent=1))
    n5 = sum(1 for r in rows if r["rung"] == "R5")
    print(f"[done] manifest_ic3_cx.json: {len(rows)} rows (R5={n5}, R4X={len(rows) - n5})")

    # ---- specialists: R3X (foreign endpoints, prefix-only, ckpt2000) --------
    # combos from the ORIGINAL eval manifests (ladder_items_v2's r3x section
    # under-covers: 8 classes vs the 11 actually generated)
    import glob
    combos = {}
    for f in glob.glob(str(REPO / "experiments/exp_066_ladder_v3_scoring/dataset/eval_r3x_*.json")):
        for it in json.loads(pathlib.Path(f).read_text()):
            p = it["item_id"].split("__")
            combos.setdefault(p[1], set()).add(p[2])
    for cls, clips in sorted(combos.items()):
        rows = []
        for clip in sorted(clips):
            rows.append({
                "id": f"R3X__{cls}__{clip}", "model": f"spec_{cls}", "tier": "X",
                "rung": "R3X", "class": cls, "clip": clip, "endpoints": clip,
                "cond_dir": "experiments/exp_062_ladder_r2r3_specialists/dataset/cond",
                "prefix_only": True,
                "prompt": corrected_prompt(caps[clip], prefix_only=True),
                "reference_class": None, "reference": None,
            })
        adapter = (f"outputs/training/exp_062_ladder_r2r3_specialists/"
                   f"{cls}_keyed/checkpoints/lora_weights_step_02000.safetensors")
        assert (REPO / adapter).exists(), adapter
        doc = {"adapter": adapter, "target_modules": SPEC_TARGETS, "rows": rows}
        (EXP / f"dataset/manifest_r3x_{cls}.json").write_text(json.dumps(doc, indent=1))
        print(f"[done] manifest_r3x_{cls}.json: {len(rows)} rows")


if __name__ == "__main__":
    main()
