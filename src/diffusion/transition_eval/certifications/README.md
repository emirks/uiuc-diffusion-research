# Certification records

One committed markdown record per certification attempt, written against the
bars frozen beforehand in `certify/bars.yaml` (SPEC §6). A PASSING record is
what authorizes the matching annotated git tag `eval/vX.Y.Z`; a FAILED record
stays committed as the honest history that drives the next draft (fail-forward
— bars are never adjusted to rescue an outcome). Each record carries: bars
sha256, per-bar verdicts with the measured numbers, provenance (commit, corpus
hash, job), the disclosures the run owes, and pointers to the full artifacts
under `outputs/eval/certification/<version>/` (exam/, analysis/, figures/,
results_explorer.html, cert_*/items.jsonl, record.json).

- **`v3.0.0.md` — CERTIFIED (overall PASS), tag `eval/v3.0.0`, 2026-07-14.**
  Verdicts are a REGRADE of the draft.8 run's artifacts under the owner's
  closed-list bar revision (bar 1 → d-only; bars 2+3 merged); the record
  discloses the outcome-aware edits verbatim and the regrade is reproducible
  via `scripts/regrade_draft8_to_v3.py`.
- `v3.0.0-draft.8.md` — FAILED (6/8 bars pass; the two pre-disclosed bars
  fail). The complete-data run (job 9465002) whose artifacts back v3.0.0.
- `v3.0.0-draft.7.md` — FAILED (record assembled post-hoc after a driver
  crash; diagnosis became the draft.8 fix list).
