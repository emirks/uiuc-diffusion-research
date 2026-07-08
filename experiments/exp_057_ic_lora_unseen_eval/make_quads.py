"""exp_057 — build the broad unseen-class quadruple matrix (design.md arms).

A quadruple = (endpoints clip E, reference clip R, type-blind prompt from E's
caption + ICTRANS, generated video). Arms:

  ic_os_inclass  E,R same UNSEEN one-sided class (12 classes x2, money_rain x1)
  ic_os_to2s     E = trained two-sided endpoints, R = unseen one-sided class
  ic_ts_unseen   two-sided unseen refs (hole in-class + hole/seamless on ew0)
  ic_anchor      exact exp_056 ic_cross items (run link)
  base_*         no-LoRA twins of a fixed 11-item subset

Outputs: dataset/quads.json, dataset/manifest_ic.json, dataset/cond_q/*.mp4
(prefix = first 9 frames, suffix = last 9 frames of the standardized endpoint
clip; suffix consumed with num_frames=8 per the exp_051 causal-VAE rule).
Deterministic, no RNG. Endpoint captions: dataset/captions.json (this exp) +
exp_056 captions for trained-class endpoints.
"""

import json
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
EXP56 = REPO_ROOT / "experiments/exp_056_ltx2_ic_lora_transition_transfer"
STD = REPO_ROOT / "data/processed/transitions_std121"
OUT_REL = "outputs/videos/exp_057_ic_lora_unseen_eval"
FFMPEG = (
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv/lib/"
    "python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)

TAGS = {  # taxonomy for report stratification (primary first), + structure
    "hero_flight": ("camera", "onesided", "novel"),
    "super_fast_run": ("camera", "onesided", "novel"),
    "plasma_explosion": ("camera", "onesided", "novel"),
    "shadow": ("style", "onesided", "cousin:shadow_smoke"),
    "fire_element": ("style", "onesided", "cousin:firelava"),
    "wireframe": ("style", "onesided", "novel"),
    "illustration_scene": ("style", "onesided", "novel"),
    "animalization": ("object", "onesided", "novel"),
    "gas_transformation": ("object", "onesided", "novel-vanish"),
    "portal": ("object", "onesided", "novel-vanish"),
    "giant_grab": ("object", "onesided", "novel-vanish"),
    "money_rain": ("object", "onesided", "novel-degenerate-endpoints"),
    "hole_transition": ("object", "twosided", "novel"),
    "seamless_transition": ("camera", "twosided", "novel"),
}

captions = json.loads((EXP / "dataset/captions.json").read_text())
captions56 = json.loads((EXP56 / "dataset/captions.json").read_text())
fam_of = {}
for fam_dir in sorted(p for p in STD.iterdir() if p.is_dir()):
    for p in fam_dir.glob("*.mp4"):
        fam_of[p.stem] = fam_dir.name


def quad(arm: str, e: str, r: str) -> dict:
    assert e in fam_of and r in fam_of, (e, r)
    cap = captions.get(e) or captions56.get(e)
    assert cap, f"no caption for endpoint clip {e}"
    ref_cls = fam_of[r]
    tax, sided, texture = TAGS.get(ref_cls, ("trained", "twosided", "trained"))
    return {
        "id": f"{arm}__{e}__ref_{r}",
        "arm": arm,
        "endpoints": e, "endpoints_class": fam_of[e],
        "reference": r, "reference_class": ref_cls,
        "ref_taxonomy": tax, "ref_sidedness": sided, "ref_texture": texture,
        "prompt": "ICTRANS " + cap,
    }


quads = []

# --- ic_os_inclass: unseen one-sided class, endpoints + same-class ref ---
OS_INCLASS = [
    ("hero_flight_2", "hero_flight_4"), ("hero_flight_6", "hero_flight_0"),
    ("super_fast_run_2", "super_fast_run_5"), ("super_fast_run_10", "super_fast_run_0"),
    ("plasma_explosion_0", "plasma_explosion_2"), ("plasma_explosion_4", "plasma_explosion_1"),
    ("shadow_10", "shadow_13"), ("shadow_2", "shadow_0"),
    ("fire_element_0", "fire_element_2"), ("fire_element_4", "fire_element_1"),
    ("wireframe_5", "wireframe_4"), ("wireframe_7", "wireframe_2"),
    ("illustration_scene_4", "illustration_scene_2"), ("illustration_scene_7", "illustration_scene_5"),
    ("animalization_0", "animalization_5"), ("animalization_3", "animalization_7"),
    ("gas_transformation_2", "gas_transformation_6"), ("gas_transformation_7", "gas_transformation_3"),
    ("portal_11", "portal_13"), ("portal_12", "portal_5"),
    ("giant_grab_1", "giant_grab_3"), ("giant_grab_4", "giant_grab_2"),
    ("money_rain_2", "money_rain_6"),
]
for e, r in OS_INCLASS:
    quads.append(quad("ic_os_inclass", e, r))

# --- ic_os_to2s: unseen one-sided ref on trained two-sided endpoints ---
EP_ROT = ["earth_wave_0", "melt_transition_1", "water_bending_0"]
OS_REFS = ["hero_flight_4", "super_fast_run_5", "plasma_explosion_2",
           "shadow_13", "wireframe_4", "illustration_scene_2", "fire_element_2",
           "animalization_5", "gas_transformation_6", "portal_13",
           "giant_grab_3", "money_rain_6"]
for k, r in enumerate(OS_REFS):
    quads.append(quad("ic_os_to2s", EP_ROT[k % 3], r))

# --- ic_ts_unseen: two-sided unseen classes (jump analogue) ---
quads.append(quad("ic_ts_unseen", "hole_transition_0", "hole_transition_1"))
quads.append(quad("ic_ts_unseen", "earth_wave_0", "hole_transition_1"))
quads.append(quad("ic_ts_unseen", "earth_wave_0", "seamless_transition_0"))

# --- ic_anchor: exact exp_056 ic_cross items ---
quads.append(quad("ic_anchor", "earth_wave_0", "shadow_smoke_1"))
quads.append(quad("ic_anchor", "melt_transition_1", "earth_wave_1"))

assert len({q["id"] for q in quads}) == len(quads)

# --- base twins ---
BASE_TWINS = [
    "ic_os_inclass__shadow_10__ref_shadow_13",
    "ic_os_inclass__wireframe_5__ref_wireframe_4",
    "ic_os_inclass__animalization_0__ref_animalization_5",
    "ic_os_inclass__gas_transformation_2__ref_gas_transformation_6",
    "ic_os_inclass__hero_flight_2__ref_hero_flight_4",
    "ic_os_inclass__super_fast_run_2__ref_super_fast_run_5",
    "ic_os_to2s__water_bending_0__ref_illustration_scene_2",  # idx 5 -> EP_ROT[2]
    "ic_os_to2s__earth_wave_0__ref_portal_13",
    "ic_os_to2s__earth_wave_0__ref_fire_element_2",
    "ic_ts_unseen__hole_transition_0__ref_hole_transition_1",
    "ic_anchor__earth_wave_0__ref_shadow_smoke_1",
]
by_id = {q["id"]: q for q in quads}
for bid in BASE_TWINS:
    src = by_id[bid]  # KeyError = subset drifted from matrix; fix HERE
    b = dict(src)
    b["arm"] = "base_" + src["arm"].removeprefix("ic_")
    b["id"] = b["arm"] + "__" + src["endpoints"] + "__ref_" + src["reference"]
    quads.append(b)

(EXP / "dataset/quads.json").write_text(json.dumps(quads, indent=2))

# --- cond cuts (prefix first 9f / suffix last 9f of std endpoint clips) ---
cond_dir = EXP / "dataset/cond_q"
cond_dir.mkdir(parents=True, exist_ok=True)
ep_clips = sorted({q["endpoints"] for q in quads})
for e in ep_clips:
    src = STD / fam_of[e] / (e + ".mp4")
    for name, vf in [(f"{e}_start9.mp4", "select='lt(n,9)'"),
                     (f"{e}_end9.mp4", "select='gte(n,112)'")]:
        dst = cond_dir / name
        if dst.exists():
            continue
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-vf", vf + ",setpts=N/24/TB", "-r", "24",
             "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)],
            check=True)
        print(f"[cond] {name}")

# --- harness manifest (style = reference class; viewer gets condition_reference) ---
manifest = []
for q in quads:
    e = q["endpoints"]
    manifest.append({
        "item_id": q["id"],
        "generated_video": f"{OUT_REL}/{'base' if q['arm'].startswith('base_') else 'ic_lora'}/quads/{q['id']}.mp4",
        "style": q["reference_class"],
        "n_endpoints": 2,
        "condition_prefix": {
            "video": f"experiments/exp_057_ic_lora_unseen_eval/dataset/cond_q/{e}_start9.mp4",
            "num_frames": 9},
        "condition_suffix": {
            "video": f"experiments/exp_057_ic_lora_unseen_eval/dataset/cond_q/{e}_end9.mp4",
            "num_frames": 8},
        "condition_reference": {
            "video": f"data/processed/transitions_std121/{q['reference_class']}/{q['reference']}.mp4"},
        "arm": q["arm"],
        "notes": f"endpoints={e} ({q['endpoints_class']}); reference={q['reference']} "
                 f"({q['reference_class']}; {q['ref_taxonomy']}/{q['ref_sidedness']}/{q['ref_texture']})",
    })
(EXP / "dataset/manifest_ic.json").write_text(json.dumps(manifest, indent=2))

arms = {}
for q in quads:
    arms[q["arm"]] = arms.get(q["arm"], 0) + 1
print(f"[done] {len(quads)} quads -> dataset/quads.json (+manifest_ic.json)")
print(f"[arms] {json.dumps(arms)}")
print(f"[endpoint clips] {len(ep_clips)}")
