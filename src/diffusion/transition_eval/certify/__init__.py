"""certify/ — SPEC §6 as code: the health-check system whose application IS
certification. A certification run executes every module below against a
candidate instrument version, compares against the pre-registered bars in
bars.yaml, writes certifications/v<X.Y>.md, and only a PASS authorizes the
eval/vX.Y tag.

Blocks (SPEC §6.1–6.4; see each docstring for run contract):
  exam.py      — Block A: two-readout validity exam (R1 clip-level LOO 1-NN for
                 M1a/M1b/M1c; R2 pool-level margin classification for M2b) under
                 every contested variant; adoption rules + per-class trust map
  probes.py    — Block B: constructed truth — siblings (max-endpoint-distance
                 bar pairs), controls, copy splices (+1 perturbation level),
                 reversal (sensitive pairs), M3 panel (endpoint-swap, hard-cut)
  (Block C)    — realism pass over the archived exp_056–058 generations:
                 copy-twin bar + descriptive report + v2↔v3 bridge (runner
                 lands with the implementation phase)
  stability.py — Block D: warm rerun bit-identity + anchor reproduction
  seeds.py     — Block D: sigma_seed measurement (O6) -> minimum detectable
                 effect; gates the first model report, not the tag

STATUS: forms locked in bars.yaml (2026-07-13 design review), numbers DRAFT —
the freeze session sets them and flips `frozen` BEFORE the first certification
run executes.
"""
