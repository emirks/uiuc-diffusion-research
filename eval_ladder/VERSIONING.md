# eval_ladder — versioning guideline

Modeled on `src/diffusion/transition_eval` (the instrument's certification flow), scaled down to
what a *benchmark design* needs. Four moving parts, one rule each. Anything not covered here is
deliberately unversioned — keep it that way rather than inventing ceremony.

## The four parts

| part | file(s) | what it versions |
|---|---|---|
| **design** | `VERSION` + `SPEC.md` | the claim structure: cells, ontology, prompts, arms, seatbelts |
| **runs** | `reports/v<DESIGN>-R<N>.md` | one trained+generated+scored execution of a design |
| **instrument** | named *inside* each run record | the transition-eval version that scored the run |
| **viewer** | `viewer/build.py` + stable output path | presentation only — never versioned itself |

## 1. Design version (`VERSION`, semver)

- **MAJOR** — the claims change meaning: ontology cells added/removed/redefined, prompt scheme
  changed, headline statistic replaced. Old run records stop being comparable.
- **MINOR** — additive: new arms, new cells alongside the old ones, more seeds/donors. Old
  records stay comparable on the shared part.
- **PATCH** — fixes that do not move any claim: a YAML quirk, a path, a build-script bug.
- Bump in the same commit that changes `SPEC.md`. Tooling (viewer, sbatch scripts, monitors)
  **never** bumps the design version.

## 2. Run records (`reports/`)

- Naming: **`v<DESIGN>-R<N>.md`** — `N` is monotonic and never resets across design bumps, so
  "R3" is unambiguous in conversation.
- Status: `VALID`, `VALID-WITH-AMENDMENTS`, or `INVALIDATED`, in the record's header line.
- **The current result is the newest record marked VALID or VALID-WITH-AMENDMENTS.** That is the
  entire "latest valid" mechanism — one sorted directory listing. `reports/README.md` keeps a
  human index; the viewer header prints the same answer automatically.
- **Append-only.** A record is never edited to improve an outcome. Corrections are dated
  amendment sections appended to the same record (the instrument's fail-forward rule, for the
  same reason: the honest history is what makes the next run trustworthy).
- Every record pins: design version + `SPEC.md` hash, instrument version + commit, frozen-input
  hashes (registry, arms, splits), and links its frozen artifacts.

## 3. Heavy artifacts

- Per-run outputs → `outputs/eval_ladder/<run-id>/` (videos stay in their generation trees; this
  holds the run's *frozen viewer* and any analysis exports).
- The **stable latest viewer** lives at `outputs/reports/ladder_viewer/index.html` — always
  rebuilt from current data, always answering "what is the newest state".
- Freezing: `python eval_ladder/viewer/build.py --freeze v<DESIGN>-R<N>` writes an immutable
  copy next to the record's other artifacts once, when the record is written. Old runs stay
  viewable forever at their frozen path; the stable path always shows now.

## 4. What a "run" is (so R-numbers don't inflate)

One set of trained models, generated and scored under one (design, instrument) pair. Re-scoring
the same generations with a new instrument = a **new run record** (cheap, no training).
Adding generations under the same design mid-run = the same record, as an amendment.
Retraining anything = a new run.

## Worked example (this repo, today)

- `VERSION` = 2.1.0 — 2.0.0 was the leak-free ladder2 design; 2.1.0 added the two clean
  baseline arms (additive → MINOR).
- `reports/v2.0.0-R1.md` — VALID-WITH-AMENDMENTS → **it is the current result.**
- The 2.1.0 baseline lane was stopped before generating (owner call, 2026-07-23); if it is ever
  re-run and scored, that becomes `v2.1.0-R2.md`.
- The instrument inside R1 is transition-eval `4.0.0` — its own certification lives with the
  instrument, not here.
