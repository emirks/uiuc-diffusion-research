# Eval-grid subsets

A subset is a plain JSON list of grid rows, identified by the registry key `(cell, endpoint, reference, sided)` plus the
seed(s) to generate, selected from an EXISTING grid. It never changes the grid, the prompt families (`store/prompts/`)
or their shas: arm registries are stamped from the family as usual (`eval_ladder/stamp_rows.py`) and then filtered to
the subset's keys. Every arm evaluated on a subset uses the identical rows, so arm-vs-arm contrasts are paired.

| file | grid | rows | seeds | design | used by |
|---|---|---|---|---|---|
| `gridv3_sub134_s42.json` | grid v3 neutral tier (384 rows) | 134 | 42 | all 26 seen + 40 unseen + 34 zero-shot Higgsfield + 34 EffectData, classes round-robin (every class kept), task mix 19% two-sided = the grid's | ICLR 2027 rebuttal ablation arms A/B/C (misc/2026-11-02_rebuttal); builder `misc/2026-11-02_rebuttal/armA/build/subset_gridv3.py` (also prints SEGUE's full-grid vs subset check) |

Reporting rule: a subset's numbers are compared WITHIN the subset (every arm on the same rows); they are not quoted next
to full-grid numbers of the paper.
