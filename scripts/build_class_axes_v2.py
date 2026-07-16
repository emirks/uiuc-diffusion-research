#!/usr/bin/env python3
"""Build outputs/taxonomy/class_axes_v2.yaml from the gate-passed Protocol v2 table.

Source of truth: docs/taxonomy/PROTOCOL_v2_PROPOSAL.md §5 (rev.3, PASSED the
fresh-context acceptance gate 2026-07-16). This script embeds that table verbatim
and merges per-class clip lists / subject_anchored metadata from the v1 annotation
record (outputs/taxonomy/class_axes.yaml). sidedness values are carried over from
v1 UNCHANGED (frozen instrument semantics; the 9 tracked conflicts stay pending
owner rulings in the viewer).

Usage: python scripts/build_class_axes_v2.py
"""
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml (research env)")

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/taxonomy/class_axes.yaml"
OUT = ROOT / "outputs/taxonomy/class_axes_v2.yaml"

# class: (mechanism, overlay_direction, scene_swap, camera_defining, stylization,
#         middle_only, note)
# Flags carried separately below. sidedness comes from v1.
T = {
    "air_bending":        ("transform", None, True,  False, False, True,
        "subjects dissolve into cotton-smoke which reforms as B (owner-ruled); cam: owner set y in v1 pass - re-judge under locked-off test"),
    "animalization":      ("transform", None, False, False, False, True,
        "body re-renders in place into animal; transient ring"),
    "color_rain":         ("overlay", "state", False, False, True,  False,
        "liquid drenches subject; grade persists"),
    "cotton_cloud":       ("overlay", "add",   False, False, False, False,
        "cotton fills scene around untouched subject"),
    "display_transition": ("cover",   None,    True,  True,  False, False,
        "held-up screen fills frame; B = screen interior (RULING: ratify interior clause; is monitor in first panel? middle_only provisional n)"),
    "earth_element":      ("transform", None, False, False, False, False,
        "body cracks into rock in place"),
    "earth_wave":         ("transform", None, False, False, False, True,
        "sand wave wraps body (RULING: becomes element -> transform, or hosts it -> overlay-state)"),
    "fire_element":       ("overlay", "state", False, False, False, False,
        "fire manifests on/around posed subject (RULING: accrual or substitution?)"),
    "firelava":           ("cover",   None,    True,  True,  False, True,
        "fire wall sweeps over the silhouette (no visible conversion) during tracked shot; reveal"),
    "flame":              ("cover",   None,    True,  False, False, True,
        "fire -> full-frame whiteout -> recedes off different subject/set"),
    "flying_cam_transition": ("traverse", None, True, True,  False, True,
        "filmstrip-verified: doorway rises from desert sand; camera walks through into living room; no covering matter at handoff"),
    "gas_transformation": ("transform", None, False, False, False, True,
        "body dissolves into gas with correspondence (conversion, dispersal-to-absence)"),
    "giant_grab":         ("overlay", "remove", False, False, False, False,
        "inserted hand drags subject out; scene persists; hand re-enters"),
    "hero_flight":        ("traverse", None,   True,  True,  False, False,
        "camera follows the launch into sustained flight persisting into B window (advisors read two_sided; owner rules)"),
    "hole_transition":    ("traverse", None,   True,  True,  False, False,
        "camera zooms THROUGH an open ring/aperture into B's actual space (passage, not picture)"),
    "illustration_scene": ("transform", None, False, False, True,  False,
        "photoreal -> flat illustration re-render in place"),
    "jump_transition":    ("traverse", None,   True,  True,  False, True,
        "camera follows jump arc to a different place/outfit"),
    "live_concert":       ("traverse", None,   True,  True,  False, False,
        "filmstrip-verified: backstage close-up pulls back through stage haze to festival wide (roster all-dup handled separately)"),
    "luminous_gaze":      ("overlay", "state", False, False, True,  False,
        "eyes ignite; storm + rim-light accrue over persisting subject/scene"),
    "melt_transition":    ("cover",   None,    True,  False, False, True,
        "prop-derived melt covers frame; trace persists into B's first frames (mid=y assumes trace clears before final ~1s window - confirm on video)"),
    "money_rain":         ("overlay", "add",   False, False, False, False,
        "bills fall into persistent scene"),
    "monstrosity":        ("overlay", "add",   False, False, False, False,
        "creature grows in background; scene persists"),
    "mystification":      ("transform", None, False, False, False, True,
        "subject dissolves into colored smoke (conversion, dispersal-to-absence)"),
    "nature_bloom":       ("overlay", "add",   False, False, False, False,
        "flowers grow into background"),
    "plasma_explosion":   ("overlay", "add",   False, True,  False, False,
        "same intersection throughout; explosion cloud added, persists; cam=y arguable under locked-off test - re-judge with air_bending"),
    "polygon":            ("transform", None, False, False, True,  False,
        "whole frame restyles photo -> white low-poly; cam flips to F under locked-off test"),
    "portal":             ("overlay", "remove", False, False, False, True,
        "portal opens at subject, removes them, vanishes; street persists empty (corpus: clip 0 is cartoon - replace exemplar)"),
    "raven_transition":   ("cover",   None,    True,  True,  False, False,
        "raven wall covers; stray birds persist into B"),
    "run_set_on_fire":    ("overlay", "state", False, True,  False, False,
        "runner catches fire mid-run; same run, same alley; cam=y arguable under locked-off test - re-judge with air_bending"),
    "saint_glow":         ("overlay", "add",   False, False, False, False,
        "halo forms around subject; subject unchanged"),
    "sakura_petals":      ("overlay", "remove", False, False, False, True,
        "RULING: conflicting filmstrip reads - external petal swarm covering subject (overlay-remove) vs suit eroding INTO petals (conversion -> transform)"),
    "seamless_transition": ("cut",    None,    True,  False, False, True,
        "hidden edit: walk-out, empty beats, walk-in different room; camera static; sidedness convention needed (effect in neither window)"),
    "shadow":             ("transform", None, False, False, False, False,
        "subject becomes a living shadow (conversion visible)"),
    "shadow_smoke":       ("cover",   None,    True,  True,  False, True,
        "body-derived smoke covers frame, clears to new scene (RULING: merely clears -> cover, or reforms into B -> transform, air_bending precedent)"),
    "super_fast_run":     ("traverse", None,   True,  True,  False, False,
        "sprint + tracking blur carry shot to different environment; still sprinting at end"),
    "water_bending":      ("overlay", "state", False, False, False, False,
        "water manipulated around/onto subject (RULING: accrual or substitution?)"),
    "water_element":      ("transform", None, False, False, False, False,
        "verified exemplar 2 = body->water (transform); other exemplars heterogeneous (RULING: confirm or split class; corpus: maybe not a class)"),
    "wireframe":          ("transform", None, False, False, True,  False,
        "subject+scene restyle to wireframe"),
    "wonderland":         ("transform", None, False, False, True,  False,
        "whole frame restyles to stylized look"),
}

OWNER_RULING = {
    "water_element", "water_bending", "fire_element", "earth_wave",
    "shadow_smoke", "sakura_petals", "display_transition",
}
SIDEDNESS_CONFLICT = {
    "earth_element", "earth_wave", "flying_cam_transition", "giant_grab",
    "hero_flight", "hole_transition", "live_concert", "water_bending",
    "water_element",
}
CAM_RECHECK = {"air_bending", "plasma_explosion", "run_set_on_fire"}

v1 = yaml.safe_load(V1.read_text())["classes"]
assert set(T) == set(v1), (set(T) ^ set(v1)) or "class set mismatch"

out = {"protocol": "v2 rev.3 (gate-passed 2026-07-16; see docs/taxonomy/PROTOCOL_v2_PROPOSAL.md)",
       "classes": {}}
for cls in sorted(T):
    mech, sub, swap, cam, styl, mid, note = T[cls]
    row = {
        "clips_viewed": v1[cls].get("clips_viewed", []),
        "mechanism": mech,
        "overlay_direction": sub if mech == "overlay" else None,
        "scene_swap": swap,
        "sidedness": v1[cls]["sidedness"],          # frozen; carried from v1
        "camera_defining": cam,
        "stylization": styl,
        "middle_only": mid,
        "subject_anchored": v1[cls]["subject_anchored"],  # metadata only in v2
        "owner_ruling": cls in OWNER_RULING,
        "sidedness_conflict": cls in SIDEDNESS_CONFLICT,
        "cam_recheck": cls in CAM_RECHECK,
        "notes": note,
        "v1_mechanism": v1[cls]["mechanism"],
    }
    out["classes"][cls] = row

# sanity: counts must match PROTOCOL_v2 §6
from collections import Counter
mc = Counter(v["mechanism"] for v in out["classes"].values())
assert mc == {"transform": 12, "overlay": 14, "cover": 6, "traverse": 6, "cut": 1}, mc
assert sum(v["owner_ruling"] for v in out["classes"].values()) == 7
assert sum(v["sidedness_conflict"] for v in out["classes"].values()) == 9
assert sum(v["middle_only"] for v in out["classes"].values()) == 14
assert sum(v["scene_swap"] for v in out["classes"].values()) == 14
assert sum(v["stylization"] for v in out["classes"].values()) == 6

OUT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=100))
print(f"wrote {OUT} ({len(out['classes'])} classes) — counts OK: {dict(mc)}")
