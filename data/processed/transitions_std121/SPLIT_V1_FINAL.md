# split_v1 — DECLARED FINAL (2026-07-16)

`split_v1.json` is **frozen and final** for the eval-ladder campaign.

- **sha256(`split_v1.json`)** = `f6cc8b5bae7bcc9cf9339e182a758f6ad6ac241dba852ed0853c50b9af4d83e6`
- **git tag:** `split/v1`
- Companion audit: `split_v1_audit.md`.

## Why this is safe to freeze now (before taxonomy validation)

The split rule is **metadata-only**: per-class `n` + a seeded RNG over clip IDs
(stream key `split_v1:{class}`). Taxonomy *labels* — including the sidedness
re-annotation still under owner validation — are **not inputs** to the split, so
label edits cannot move it. The split changes only under a **clip-roster/curation**
change (a clip added, removed, or reassigned).

- **The one open curation flag** is `water_element_5` (visually a wireframe effect,
  not water). `water_element` has `n=5`, is **not** in the R2 roster (roster = n≥8),
  and stays in the same test band (4≤n<8 → 1 test) even if the clip is pulled — so
  pulling it would not change any test *count*, only which clip, and touches no
  roster training.
- The per-class RNG streams localize any future rebuild to the single affected
  class; no other class's train/test membership moves.

Therefore label validation and any single-clip curation are **decoupled** from the
split, and the ladder's training/generation can proceed against this frozen split.

## Roster consequence recorded

`live_concert` (n=8) has **0 test clips** after the dup-audit remediation (all 8
clips were mutual near-duplicates → all-train). It therefore feeds no paired delta
and is excluded from the specialist roster (see `docs/eval_ladder/PLAN.md` §D4).

*Declared under the `/advised` operator/advisor protocol; freeze decision is the
campaign advisor's (fable) D-series ruling, executed by the operator.*
