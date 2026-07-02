"""Build the eval manifest for the 24 exp_051 C2V-ladder outputs.

Sample index -> test clip/prompt mapping fixed by exp_051's
run_c2v_inference.py: 1=ew0+trigger, 2=ew0 no-trigger, 3=ew1+trigger,
4=ew1 no-trigger, 5=ew2+trigger, 6=ew2 no-trigger. All arms share the same
condition clips (first 9 / last 8-of-9 frames of the earth_wave sources).
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.transition_eval.manifest import Condition, EvalItem, save_manifest  # noqa: E402

LADDER = REPO_ROOT / "outputs/videos/exp_051_ltx2_lora_c2v_ladder"
COND = REPO_ROOT / "experiments/exp_051_ltx2_lora_c2v_ladder/dataset"
ARMS = ["base", "t2v", "i2v_ff05", "c2v"]
CLIPS = ["ew0", "ew0", "ew1", "ew1", "ew2", "ew2"]  # sample idx 1..6
TRIG = ["trigger", "notrigger"] * 3

items = []
for arm in ARMS:
    samples = sorted((LADDER / arm / "samples").glob("step_*.mp4"))
    assert len(samples) == 6, (arm, samples)
    for idx, path in enumerate(samples):
        clip = CLIPS[idx]
        items.append(EvalItem(
            item_id=f"{arm}_{clip}_{TRIG[idx]}",
            generated_video=str(path.relative_to(REPO_ROOT)),
            style="shadow_smoke",
            n_endpoints=2,
            condition_prefix=Condition(str((COND / f"cond_{clip}_start9.mp4").relative_to(REPO_ROOT)), 9),
            condition_suffix=Condition(str((COND / f"cond_{clip}_end9.mp4").relative_to(REPO_ROOT)), 8),
            arm=arm,
            notes=f"exp_051 ladder; {'SHDWSMK prompt' if TRIG[idx] == 'trigger' else 'no trigger token'}",
        ))

out = pathlib.Path(__file__).parent / "manifest_exp051.json"
save_manifest(items, out)
print(f"[done] {len(items)} items -> {out}")
