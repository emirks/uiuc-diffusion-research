// luma_soft — a feathered luma-matte compositor.
//
// The shipped gl-transitions `luma.glsl` is
//     mix(getToColor(uv), getFromColor(uv), step(progress, texture2D(luma, uv).r));
// i.e. a HARD threshold on the matte: no feather, no rim, no glow. Every ink /
// paint / light-leak transition that reads as real does so mostly because of what
// happens *at the advancing boundary*, and `step()` has no boundary at all.
//
// This shader keeps the same static-matte plumbing (one greyscale field in the
// `luma` sampler, `progress` sweeps a threshold across it) and only changes the
// compositing:
//   1. smoothstep feather instead of step
//   2. a rim colour painted into the boundary band  (the ink's own pigment)
//   3. an additive glow lobe ahead of the front      (light leaks / emissive edges)
//
// Endpoint identity is exact by construction:
//   * the threshold is remapped to p = progress*(1+2f) - f, so at progress=0 the
//     whole feather band sits below every matte value (alpha == 1 -> pure `from`)
//     and at progress=1 it sits above every matte value (alpha == 0 -> pure `to`);
//   * both the rim and the glow are multiplied by an envelope that is exactly 0
//     at progress 0 and 1.
// That matters: the endpoint blocks of a training clip must reproduce the
// conditioning frames bit-for-bit.

uniform sampler2D luma;
uniform float feather;      // = 0.05
uniform float rimWidth;     // = 1.0
uniform float rimAmount;    // = 0.55
uniform vec3 rimColor;      // = vec3(0.05, 0.04, 0.08)
uniform float glowWidth;    // = 2.6
uniform float glowAmount;   // = 0.35
uniform vec3 glowColor;     // = vec3(1.0, 0.92, 0.76)
uniform float glowBias;     // = 0.8
uniform float envEdge;      // = 0.07

vec4 transition(vec2 uv) {
    float m = texture2D(luma, uv).r;
    float f = max(feather, 1e-4);

    // remapped threshold: sweeps from -f to 1+f so both ends are clean
    float p = progress * (1.0 + 2.0 * f) - f;

    // alpha = 1 -> keep `from`; alpha = 0 -> show `to`
    float alpha = smoothstep(p - f, p + f, m);
    vec3 col = mix(getToColor(uv).rgb, getFromColor(uv).rgb, alpha);

    // signed distance to the front, in feather units (0 exactly on the boundary,
    // >0 = not yet revealed, <0 = already revealed)
    float d = (m - p) / f;

    // envelope: kills rim + glow at both endpoints so the identities hold
    float e = max(envEdge, 1e-4);
    float env = smoothstep(0.0, e, progress) * smoothstep(0.0, e, 1.0 - progress);

    float rw = max(rimWidth, 1e-3);
    float rim = exp(-(d * d) / (rw * rw));
    col = mix(col, rimColor, clamp(rimAmount * rim * env, 0.0, 1.0));

    // glow lobe, biased to the not-yet-revealed side (glowBias in [0,1])
    float gw = max(glowWidth, 1e-3);
    float dg = (d - glowBias * gw) / gw;
    float glow = exp(-(dg * dg));
    col += glowColor * (glowAmount * glow * env);

    return vec4(clamp(col, 0.0, 1.0), 1.0);
}
