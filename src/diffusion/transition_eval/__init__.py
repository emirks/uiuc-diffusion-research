"""Content-invariant evaluation harness for creative video transitions.

A transition is a program that acts on content, not a set of pixels. Every
metric here is computed relative to a video's OWN endpoints and frames, so two
videos that share a transition style but nothing else remain comparable:

- morph:      Morph Profile a(t)/b(t) curves + transformation depth, timing,
              identity hold, core-frame mask (M1)
- motion:     Motion Fidelity via tracklet velocity correlation (M2)
- appearance: effect-medium appearance on core frames + leakage retrieval (M3, M6)
- endpoints:  conditioned-frame fidelity + boundary seam detection (M5)
- judge:      checklist rubric VLM judge, experimental until human-validated (M4)
- controls:   lerp (crossfade) floor synthesis
- report:     floor/ceiling normalization + retrieval-based harness validation

Validated in exp_052 (style-discrimination exam on real clips before any
method decision rests on these numbers).
"""
