"""certify/ — SPEC §6 as code: the health-check system whose application IS
certification. A certification run executes every module below against a
candidate instrument version, compares against the pre-registered bars in
bars.yaml, writes certifications/v<X.Y>.md, and only a PASS authorizes the
eval/vX.Y tag.

Blocks (SPEC §6.1–6.4; see each docstring for run contract):
  exam.py      — Block A: two-readout validity exam (R1 clip-level LOO 1-NN for
                 M1a/M1b/M1c; R2 pool-level margin classification for M2b) under
                 every contested variant; pre-registered adoption + trust map
  probes.py    — Block B: constructed truth — siblings (max-endpoint-distance
                 bar pairs + all-pairs content-invariance audit), controls,
                 copy splices (verbatim + one pinned perturbation), reversal
                 (pre-enumerated sensitive pairs), M3 panel (swap, hard-cut)
  blockc.py    — Block C: archived exp_056–058 realism pass — copy-twin bar,
                 v2↔v3 bridge, per-arm distributions, loud exclusions
  stability.py — Block D: rerun comparator (warm tolerance + cold anchors)
  seeds.py     — Block D: sigma_seed (O6) -> MDE; gates the first model
                 report, not the tag
  run_certification.py — the §6.5 driver: freeze-checked, mechanical
                 A→B→C→D, writes certifications/v<version>.md + record.json

STATUS: implemented; runs only against frozen bars (the freeze flip is its
own commit, before the first execution).
"""
