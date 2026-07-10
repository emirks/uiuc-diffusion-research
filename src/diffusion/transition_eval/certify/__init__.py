"""certify/ — SPEC §6 as code: the health-check system whose application IS
certification. A certification run executes every module below against a
candidate instrument version, compares against the pre-registered bars in
bars.yaml, writes certifications/v<X.Y>.md, and only a PASS authorizes the
eval/vX.Y tag.

Modules (see each docstring for run contract):
  exam.py      — style-discrimination exam (LOO retrieval) for M1a/M1b/M1c
                 under every contested variant; refreshes per-class trust flags
  probes.py    — adversarial ground truth: splice copies (M2a), cross-labeled
                 references (M2b), degenerate controls (floors/flags)
  stability.py — warm/cold rerun determinism + anchor reproduction
  seeds.py     — sigma_seed measurement (O6) -> minimum detectable effect

STATUS: skeleton-complete, bars are DRAFT (SPEC O3) — the health-design
session freezes numbers BEFORE the first certification run executes.
"""
