# Coordination note — taxonomy v2 landed (2026-07-16 ~22:30, taxonomy session)

Owner validated ALL 39 classes tonight. Protocol v2 is ADOPTED (`7a10815`), the
sidedness fold is in the corpus manifest (`85023fa`), certification amendment-2 is
on eval/v3-spec-versioning (`26023c2`). **Scoring is UNBLOCKED.** What each
workstream needs to know:

## → Eval-ladder session (generation + scoring driver)

1. **B1 IS ALREADY SUBMITTED — do not resubmit.** hero_flight keyed R2/R3 =
   job **9539197** (`job_gen_keyed.sbatch` new `b1` mode), hero_flight R5 =
   job **9539198** (`job_infer.sbatch` with `EXTRA_FLAGS=--include-deferred`).
   Both harmless to re-run (skip-if-exists) but it wastes queue slots.
   **`git pull` before touching those two sbatch files** (b13a41a changed them).
2. **Scoring rule (amendment-2):** metric code from certified checkout
   `eval/v3.0.0` as always; `--corpus` MUST point at the corrected
   `corpus_manifest.json` (sha256 `348db23dac72d7ed…`). Stale manifest
   (`e7c867a6…`) = error (hero_flight/giant_grab would get the wrong S mask).
3. **First scoring batch must rescore the hero_flight σ_seed item both ways**
   (onesided-mask vs twosided-mask) and report pooled σ_seed under each —
   amendment-2 §2 remedy.
4. **Strata are owner-final** (protocol §5/§6): transform 17 / overlay 12 /
   cover 4 / traverse 6 / cut 0. Pre-registered handling of the two §5.1
   exceptions: portal OUT of pooled new-shot copy tests (pool = 9, needs 8/9);
   plasma_explosion OUT of overlay copy-regime tests. copy_max calibration
   subgroup = stylization=T (6). middle_only split y23/n16. Roster keying
   unaffected (all A_only except shadow_smoke/hero_flight two_sided = as built);
   B8 all 8 unchanged.
5. Don't wait for v4 to score primaries — pre-registration pins primaries to
   v3.0.0(+amendments) regardless.

## → v4-certification session

**HOLD THE STAMP until you consume the corrected manifest.** Specifically:

1. Merge main (or cherry-pick `85023fa`) into eval/v4-metrics:
   `build_corpus_manifest.py` now carries `OWNER_SIDEDNESS_OVERRIDES`
   (giant_grab, hero_flight → twosided). Corrected manifest sha
   `348db23dac72d7ed…` — cite it in the v4.0.0 record.
2. Amendment-2 §3 explicitly hands you: **recompute per-sidedness control
   floors on owner-final labels.** Also recheck anything else
   sidedness-dependent — reference_stats, exam items touching
   giant_grab/hero_flight, the S-mask mode those items get.
3. If the L40 provenance/parity run (9538092) consumed the stale manifest,
   re-run the affected parity checks before stamping.
4. Timing: primaries never wait for you. Your improved M1s enter as
   pre-registered *labeled secondary* only for batches scored AFTER your stamp
   (PLAN §7 rule: "certified before scoring"). If you stamp before tomorrow's
   R2/R3+later scoring, those rungs get v4 secondaries; R0/R1 (scored first)
   can be rescored under v4 only as clearly-labeled post-hoc secondary.

## Unified sequence from here

taxonomy DONE → [ladder session] score R0/R1 now on v3.0.0 + corrected manifest
→ [v4 session] recompute floors on corrected manifest, stamp v4.0.0 →
[ladder session] score remaining rungs as they land (R1K 34/54 done, keyed
train → R2/R3 → R3X; R4X independent; B1 queued), v4 secondaries ride along
post-stamp → contrasts C1–C9 → results table → Friday presentation.
Taxonomy session stands by for: plasma/portal harmonization clicks (owner),
analysis-time strata questions.
