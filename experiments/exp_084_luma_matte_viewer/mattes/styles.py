"""Per-operator compositing style for `luma_soft.glsl`.

A single global feather is the wrong abstraction: an ink bleed wants a narrow,
dark, matte boundary, a light leak wants a wide bright one that blooms ahead of
the front, and a paint wipe wants a fat feather with the pigment's own colour
along the stroke. The style is a property of the *operator*, so it is declared
next to the map, not baked into the shader.

Fields map 1:1 onto the shader's uniforms.
"""

from __future__ import annotations

STYLE_PRESETS: dict[str, dict] = {
    # ---- dark pigment, narrow matte edge, barely any bloom -----------------
    "ink": dict(feather=0.035, rimWidth=1.0, rimAmount=0.80,
                rimColor=(0.04, 0.03, 0.05), glowWidth=2.2, glowAmount=0.10,
                glowColor=(0.55, 0.45, 0.40), glowBias=0.9, envEdge=0.07),
    # ---- coloured paint: fat feather, pigment along the stroke -------------
    "paint": dict(feather=0.065, rimWidth=1.1, rimAmount=0.62,
                  rimColor=(0.88, 0.24, 0.16), glowWidth=2.0, glowAmount=0.08,
                  glowColor=(1.00, 0.72, 0.45), glowBias=0.8, envEdge=0.08),
    # ---- light leak / emissive streak: wide bright bloom ahead of the front -
    "leak": dict(feather=0.075, rimWidth=1.3, rimAmount=0.45,
                 rimColor=(1.00, 0.86, 0.58), glowWidth=3.2, glowAmount=0.62,
                 glowColor=(1.00, 0.80, 0.42), glowBias=0.9, envEdge=0.09),
    # ---- cold crystalline edge (frost / shatter) ---------------------------
    "frost": dict(feather=0.045, rimWidth=1.0, rimAmount=0.55,
                  rimColor=(0.80, 0.93, 1.00), glowWidth=2.6, glowAmount=0.34,
                  glowColor=(0.62, 0.84, 1.00), glowBias=0.85, envEdge=0.07),
    # ---- burn / ember: hot narrow line, strong bloom -----------------------
    "burn": dict(feather=0.030, rimWidth=0.9, rimAmount=0.85,
                 rimColor=(0.12, 0.03, 0.01), glowWidth=2.4, glowAmount=0.80,
                 glowColor=(1.00, 0.52, 0.14), glowBias=1.0, envEdge=0.07),
}

# new maps (exp_084) -> style
NEW_MAP_STYLE = {
    "eikonal_ink": "ink",
    "eikonal_burst": "burn",
    "eikonal_streak": "leak",
    "eikonal_drip": "paint",
    "invasion_ink": "ink",
    "invasion_frost": "frost",
    "invasion_finger": "ink",
    "brush_wipe": "paint",
    "brush_scribble": "paint",
    "brush_splat": "ink",
    "edge_draw": "ink",
    "edge_draw_fine": "burn",
}

# the 7 shipped procedural maps (exp_075) -> style. Chosen so the A1/A2 pair is
# a fair fight: each old map gets the treatment its geometry actually suggests.
OLD_MAP_STYLE = {
    "fbm": "ink",
    "radial": "leak",
    "linear": "leak",
    "stripes": "burn",
    "checker": "paint",
    "spiral": "frost",
    "voronoi": "frost",
}


def style_for(map_name: str, is_new: bool) -> tuple[str, dict]:
    table = NEW_MAP_STYLE if is_new else OLD_MAP_STYLE
    key = table[map_name]
    return key, dict(STYLE_PRESETS[key])
