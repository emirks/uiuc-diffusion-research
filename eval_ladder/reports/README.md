# Ladder run reports

One committed markdown record per **run** of the ladder, written against the bars frozen
beforehand in `../SPEC.md` §9. A run is one set of trained models generated and scored under a
given (design version, instrument version).

Naming: **`v<DESIGN>-R<N>.md`** — design semver from `../VERSION`, run number monotonic and never
reset across design bumps. The instrument version goes *inside* the record, never in the filename.

**The current result is the newest `R<N>` marked `VALID` or `VALID-WITH-AMENDMENTS`.**

A record is never edited to improve an outcome. If a number changes after publication, that is an
**amendment section appended to the same record**, dated and reasoned — the fail-forward rule the
instrument's `certifications/` uses, for the same reason: the honest history is what makes the next
run trustworthy.

---

## Index — newest first

- **`v2.0.0-R1.md` — VALID-WITH-AMENDMENTS, 2026-07-23.** First run of design v2.0.0. 12 models
  (11 c2v specialists + 1 IC-LoRA generalist), 888 generations, instrument `4.0.0-draft.1`.
  Specialists PASS both claim cells at ~100 %/95 % of ceiling (+40 pp, 11/11 donors). The four
  reference-bearing generalist claim readouts were **INVALIDATED mid-run** by the demo-copy
  confound and replaced under Amendment-1 by the transfer index: `G-unseen-cross` PASS
  (ΔTI +3.9 pp, 9/13), `G-zs-cross` FAIL (−8.3 pp, 1/10). No kill rule fired.

---

## Required fields

Nine. All of them, or the record is not valid. This is deliberately writable by hand in twenty
minutes — a record that is expensive to write is a record that does not get written.

1. **Run identity** — run id, date, design `VERSION`, sha256 of `SPEC.md` at run time.
2. **Instrument** — version *and* repo commit sha. If it is a draft (untagged), say so plainly and
   say what has since been certified.
3. **Arms** — the models, their count, and a pointer to the training configs.
4. **Frozen inputs** — sha256 of `registry.jsonl`, its row count, and the hashes of the split /
   caption corpus / DAVIS roster / `arms.yaml`.
5. **Completion** — generations produced vs registry expectation, scored rows, and *anything
   missing, disclosed*. "888/888 verified, 0 missing" or an explicit list.
6. **Per-claim-cell verdicts** — a table of `PASS` / `FAIL` / `INVALID(reason, amendment-ref)` with
   the measured numbers against the pre-frozen bars.
7. **Amendments and deviations** — each labelled `pre-registered` or `outcome-aware`, with the
   reasoning. An outcome-aware amendment is not forbidden; hiding one is.
8. **Kill rules** — which fired, or `none`.
9. **Artifacts and reading** — where the videos/scores/viewer live, the headline table, and one
   paragraph saying what the run means.

Everything beyond those nine is optional detail. Add it when it helps a future reader; do not pad.

---

## Heavy artifacts

Per-run outputs go to `outputs/eval_ladder/R<N>/` — mirroring the instrument's
`outputs/eval/certification/<version>/`. The run's viewer is built into that directory and linked
from the record, so a viewer is always versioned with the run that produced it.

`../REFERENCE.html` is the only always-latest design HTML: it is the face of the **design**, not
of a run, and is regenerated when `VERSION` bumps. For day-to-day reading, a convenience copy of
the **current run's results viewer** is kept at `outputs/reports/ladder_viewer/index.html`
(uncommitted artifact; the versioned copy lives in the run's own directory).

*(R1 predates this rule; its artifacts stay at the paths its record names. The rule binds from R2.)*

## Frozen viewers

- `v2.0.0-R1` → `outputs/eval_ladder/v2.0.0-R1/viewer.html` (frozen 2026-07-23; includes the
  partial v2.1.0 baseline videos that existed when the lane was stopped)
- Stable latest: `outputs/reports/ladder_viewer/index.html` — always rebuilt from current data.
