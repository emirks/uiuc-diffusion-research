"""Build manifest_adversarial.json — known near-copies from exp_046/048 plus
live-branch negatives from exp_046(tempblend)/exp_047(velocity-guided).

These are the harness's anti-cheating exam: the lerp control covers "too
little transition"; these cover "transition stolen from the reference".
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.transition_eval.manifest import Condition, EvalItem, save_manifest  # noqa: E402

V46 = "outputs/videos/exp_046_smoke_perceptual_inject"
V47 = "outputs/videos/exp_047_smoke_velocity_guide"
V48 = "outputs/videos/exp_048_smoke_self_recon_inject"
SS = "data/processed/transitions/shadow_smoke"


def clip(n: int) -> str:
    return f"{SS}/shadow_smoke.mp4" if n == 0 else f"{SS}/shadow_smoke_{n}.mp4"


# (arm, generated_video, target clip idx, notes)
SPEC = [
    # self-recon decode of the reference clip itself — leakage must be ~max
    ("src_copy", f"{V46}/run_0002/shadow_smoke_0/prior_src.mp4", 0, "recon of ss0 itself"),
    ("src_copy", f"{V46}/run_0002/shadow_smoke_1/prior_src.mp4", 1, "recon of ss1 itself"),
    ("src_copy", f"{V46}/run_0002/shadow_smoke_6/prior_src.mp4", 6, "recon of ss6 itself"),
    ("src_copy", f"{V46}/run_0002/shadow_smoke_9/prior_src.mp4", 9, "recon of ss9 itself"),
    # another reference clip's real smoke latents spliced into the target
    ("donor_pin", f"{V46}/run_0002/shadow_smoke_0/prior_donor:shadow_smoke_2.mp4", 0, "ss2 smoke in ss0"),
    ("donor_pin", f"{V46}/run_0002/shadow_smoke_0/prior_donor:shadow_smoke_8.mp4", 0, "ss8 smoke in ss0"),
    ("donor_pin", f"{V46}/run_0002/shadow_smoke_1/prior_donor:shadow_smoke_5.mp4", 1, "ss5 smoke in ss1"),
    ("donor_pin", f"{V46}/run_0002/shadow_smoke_1/prior_donor:shadow_smoke_7.mp4", 1, "ss7 smoke in ss1"),
    ("donor_pin", f"{V46}/run_0002/shadow_smoke_6/prior_donor:shadow_smoke_0.mp4", 6, "ss0 smoke in ss6"),
    ("donor_pin", f"{V46}/run_0002/shadow_smoke_6/prior_donor:shadow_smoke_2.mp4", 6, "ss2 smoke in ss6"),
    ("donor_pin", f"{V46}/run_0002/shadow_smoke_6/prior_donor:shadow_smoke_8.mp4", 6, "ss8 smoke in ss6"),
    ("donor_pin", f"{V46}/run_0002/shadow_smoke_9/prior_donor:shadow_smoke_5.mp4", 9, "ss5 smoke in ss9"),
    # softened splice (0.7 blend) — reported, no hard bar
    ("donor_blend", f"{V46}/run_0002/shadow_smoke_1/prior_donorblend:shadow_smoke_5:0.7.mp4", 1, "ss5 blend 0.7"),
    ("donor_blend", f"{V46}/run_0002/shadow_smoke_6/prior_donorblend:shadow_smoke_0:0.7.mp4", 6, "ss0 blend 0.7"),
    ("donor_blend", f"{V46}/run_0002/shadow_smoke_9/prior_donorblend:shadow_smoke_5:0.7.mp4", 9, "ss5 blend 0.7"),
    # generative path steered onto the clip's OWN recon — near-copy of itself
    ("self_inject_g1", f"{V48}/run_0002/shadow_smoke_0/regen_g1.0.mp4", 0, "self recon-inject g=1.0"),
    ("self_inject_g1", f"{V48}/run_0002/shadow_smoke_1/regen_g1.0.mp4", 1, "self recon-inject g=1.0"),
    ("self_inject_g1", f"{V48}/run_0002/shadow_smoke_2/regen_g1.0.mp4", 2, "self recon-inject g=1.0"),
    ("self_inject_g08", f"{V48}/run_0002/shadow_smoke_0/regen_g0.8.mp4", 0, "self recon-inject g=0.8"),
    ("self_inject_g08", f"{V48}/run_0002/shadow_smoke_1/regen_g0.8.mp4", 1, "self recon-inject g=0.8"),
    # live-branch negatives: winning recipes of that era, NOT single-clip copies
    ("neg_velguide", f"{V47}/run_0001/shadow_smoke_1/regen_base_cfg4.mp4", 1, "production baseline"),
    ("neg_velguide", f"{V47}/run_0001/shadow_smoke_6/regen_base_cfg4.mp4", 6, "production baseline"),
    ("neg_velguide", f"{V47}/run_0001/shadow_smoke_1/regen_g0.8_cfg4.mp4", 1, "velocity-guided g0.8"),
    ("neg_velguide", f"{V47}/run_0001/shadow_smoke_6/regen_g0.8_cfg4.mp4", 6, "velocity-guided g0.8"),
]


def main():
    items = []
    # tempblend filenames encode blend params — glob rather than hardcode
    for tgt in (1, 6):
        hits = sorted((REPO_ROOT / V46 / "run_0004" / f"shadow_smoke_{tgt}").glob("prior_tempblend*.mp4"))
        if hits:
            SPEC.append(("neg_tempblend", str(hits[0].relative_to(REPO_ROOT)), tgt,
                         "temporal-blend winning recipe"))
    missing = []
    for arm, gen, tgt, notes in SPEC:
        if not (REPO_ROOT / gen).exists():
            missing.append(gen)
            continue
        items.append(EvalItem(
            item_id=f"{arm}_{pathlib.Path(gen).parent.name}_{pathlib.Path(gen).stem}".replace(":", "-"),
            generated_video=gen, style="shadow_smoke", n_endpoints=2,
            condition_prefix=Condition(video=clip(tgt), num_frames=9),
            condition_suffix=Condition(video=clip(tgt), num_frames=8),
            arm=arm, notes=notes))
    if missing:
        print("[error] missing files:")
        for m in missing:
            print("  ", m)
        sys.exit(1)
    out = pathlib.Path(__file__).parent / "manifest_adversarial.json"
    save_manifest(items, out)
    arms = sorted({i.arm for i in items})
    print(f"[done] {len(items)} items -> {out}")
    for a in arms:
        print(f"  {a}: {sum(1 for i in items if i.arm == a)}")


if __name__ == "__main__":
    main()
