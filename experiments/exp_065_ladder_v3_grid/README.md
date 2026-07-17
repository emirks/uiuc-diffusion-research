# exp_065 — generation grid v3 (unified tier system)

## Question
Generate every remaining eval row of PLAN Amendment 2's tier grid — one
manifest per model, all tiers at once: base PE-keyed extension, ic3 tiers
A/B/C/X. (Specialist tiers run from exp_062 with `LADDER_GRID=v3`.)

## Setup
Grid = `docs/eval_ladder/ladder_items_v3.json` (frozen; regenerate with
`build_ladder_items_v3.py` — deterministic). Manifests in `dataset/`:
- `manifest_base_ext.json` — 10 rows: PE-keyed (prefix-only) for one_sided
  test-bearing classes beyond the R1K-9, incl. live_concert_0. adapter=null.
- `manifest_ic3.json` — 55 rows: tier A (15: held-in ceiling), B (33: trained
  class × test-band endpoints), C (7: hero_flight/gas/illustration/raven).
- `manifest_ic3_x.json` — 44 rows: 11 recipients × 4 donors; the 32 B8 twins
  verbatim from grid v2, 12 extension rows prefix-only (twin-consistent with
  exp_062 R3X). adapter = ic3 step_05000.
Contract everywhere: 480×640×121@24, 30 steps, CFG 4, STG 1 stg_v[29],
seeds 42/43/44, skip-if-exists.

## How to run
See the launch sheet in `docs/eval_ladder/PLAN_v3.md`.

## Outputs
`outputs/videos/exp_065_ladder_v3_grid/<rung>/<id>__s<seed>.mp4`
