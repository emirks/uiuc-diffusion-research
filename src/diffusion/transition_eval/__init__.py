"""Transition eval harness — reference-based creative transition transfer.

SPEC.md in this package is the authoritative, versioned definition (purpose,
input contract, metric formulas, certification, change protocol); VERSION +
versioning.py stamp every result. v3 metric IDs mirror the task anatomy —
*execute the reference's transition (M1) on your own endpoints (M3) without
cheating (M2), and look right overall (M4)*:

- s_structure:   S    curves, sidedness-aware core mask, timing scalars
- m1_transfer:   M1a  appearance-to-reference · M1b camera · M1c object motion
- m2_integrity:  M2a  copy · M2b intrusion (named) · M2c memorization (tiered)
- endpoints:     M3a  endpoint fidelity · M3b seam flag
- judge_gemini:  M4   rubric judge (ADVISORY until human-calibrated)
- controls:      lerp / static-hold degenerate control arms (never divisors)
- manifests_v3:  eval / corpus / training manifests + derivations
- plan, score:   lifecycle CLIs (plan -> external inference -> score)
- certify/:      the health-check system; its application IS certification

Legacy v2 modules (morph, motion, appearance, report) remain as substrate;
legacy metric IDs (old M1–M6) are retired — IDs read per the stamped version.
"""
