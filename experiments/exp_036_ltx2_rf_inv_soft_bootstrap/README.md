# exp_036 — soft model-bootstrap middle anchors

## Question
exp_035 strength=1.0 bootstrap → catastrophic. exp_034B strength=0 (drop) →
catastrophic. What about strength=0.3 — partial directional guidance from
bootstrap without forcing round-trip through bootstrap-land?

## How to run
```bash
# Pilot (ss0+ss5, full 40 steps)
python experiments/exp_036_ltx2_rf_inv_soft_bootstrap/run.py \
  --config experiments/exp_036_ltx2_rf_inv_soft_bootstrap/config_pilot.yaml

# Full batch — only after pilot CONFIRMS
python experiments/exp_036_ltx2_rf_inv_soft_bootstrap/run.py
```

## Outputs
Same as exp_035 but with `bootstrap.middle_strength` recorded in inv_meta.

## Decision rule
- CONFIRMED: median PSNR across {ss0, ss5} ≥ exp_033 baseline (20.68) + 3 dB.
- REJECTED: regression > 2 dB on either pilot clip.
