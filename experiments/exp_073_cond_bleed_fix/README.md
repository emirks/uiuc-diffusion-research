# exp_073 — Conditioning-bleed fix (causal-VAE suffix anchor)

Operator/advisor campaign (advisor = fable-advisor). Decision log:
`$LAB/misc/bleed_fix_campaign/DOSSIER.md`. Trainer patch lives on the LTX-2-official
branch `cond-bleed-fix` (off 7809842) — NOT committed to this public repo.

## Question

The LTX-2 endpoint-conditioning trainer pins the **suffix** conditioning token to the
**last latent frame of the FULL-video VAE encode**. The video VAE is temporally causal, so
that latent frame has a backward receptive field into the **middle** of the clip and thus
carries middle content. At inference the suffix is built from a standalone encode of only
the trailing frames (bleed-free). So **training suffix ≠ inference suffix**. Does aligning
training to the inference-time (standalone-cut) suffix anchor improve endpoint-conditioned
generation — measured on the suffix endpoint — without harming anything else?

Scope: only TWO-SIDED items have a suffix. 2/11 specialists (shadow_smoke, hero_flight);
95/403 ic3 training pairs (10 two-sided classes). One-sided items have no suffix →
untouched by the fix → **built-in negative control**.

## Setup (recipe-identical except the fix)

- Fix mechanism: optional `cond_clean_latents_dir` on the video modality
  (`FlexibleStrategy`). When set, intrinsic conditions (prefix/suffix/mask) pin their CLEAN
  latent from that precomputed source; unset ⇒ bitwise-identical to the original trainer.
  `cond_clean` = the original stored latents with ONLY the last latent frame replaced by the
  last frame of a standalone encode of the trailing 9 pixel frames (two-sided clips). Prefix
  is NOT substituted (identity by causality — gate K1). Trailing cut sliced from the same
  preprocessed pixel tensor as the original latents (no mp4 roundtrip).
- Trainer commit base = `7809842` (the exact commit the originals trained on).
- Specialists: rank32/α32, attn-only, lr 1e-4 linear, 2000 steps, ckpt/250, bf16, seed 42,
  ICTRANS, 480x640x121. ic3: rank32/α32, attn+FFN, lr 2e-4, 5000 steps, ckpt/500, bf16,
  seed 42, reference + per-pair mask. ic3 run as a SINGLE block (no resume chain, to avoid
  resume-point RNG artifacts).

### Comparison design (PRIMARY = fix vs NULL, not fix vs original)

Every retrained model has a **null twin**: identical recipe/seed/branch, `cond_clean_latents_dir`
UNSET. `fix − nullA` isolates the anchor change; `nullA − nullB` (a duplicated null) measures
the pure run-to-run nondeterminism floor. The original (resume-chained) arm is DEMOTED to
context/diagnostics (rescored, never regenerated). 8 training jobs:
shadow_smoke/hero_flight/ic3 × {fix, nullA} + {shadow_smoke, ic3} × nullB.

## Pre-registration (committed BEFORE any score is unblinded)

Instrument: v4 scorer (`.claude/worktrees/eval-v4-cert`), one pinned commit for ALL arms
(fix, null, original). Uncertified v4 is acceptable for paired deltas. Verdict engine:
`compare_arm.py` logic re-keyed on suffix_lpips (copied into the campaign dir; the
misc/advised_method_impl copy is the parent's — untouched).

**Manipulation check (gate, before training):**
- K1 (prefix identity, fp32): median per-clip prefix rel-L2(standalone-first9, full[:2]) < 0.02
  AND < 0.1× the suffix rel-L2. If FAIL → STOP + reconsult (scope would become prefix+suffix
  + all 11 specialists). RESULT: **PASS** — median prefix rel-L2 = 8.3e-5 (max 1.2e-4), 28
  clips (job 9604698, gate_stats.json). Prefix is bit-clean; §14-b's sub-clip≠slice worry
  does not hold for the frame-aligned prefix → one-sided items genuinely untouched.
- K2 (suffix material, fp32): median suffix rel-L2(standalone-last9-last, full[-1]) > 0.05.
  If trivial (<0.02) → report back (bug cannot matter; contradicts exp_051).
  RESULT: **PASS** — median suffix rel-L2 = 0.280 (range 0.186–0.416). The bug is large.

**Predictions:**
- P1 (PRIMARY): on two-sided eval items, `suffix_lpips` improves (↓) in the fix arm vs nullA.
- P2: `suffix_seam_z` improves (↓); `suffix_dino` improves (↑) or flat. (secondary/directional)
- P3: `prefix_lpips`/`prefix_dino`/`prefix_seam_z` on two-sided items — no change beyond the
  floor (within-item control).
- P4: one-sided ic3 eval items (incl. ic3_x) — fix-vs-nullA deltas within the nullA-vs-nullB
  floor (negative control).
- P5: `margin`, `app_ref`, `cam_zpr`, `obj_csls` flat; `near_copy`/`copy_max` do NOT increase
  (proxy-gaming guard).
- P6 (context): nullA vs original nonzero (retraining noise), larger for ic3. Reported, not judged.
- NULL IS A REPORTABLE OUTCOME. Expected effect is modest (only the last 1 of 16 latent frames
  changes, on 95/403 ic3 + 2/11 specialists).

**Primary metric + analysis rule:**
- Metric = `suffix_lpips`. Per-item delta = median over the 3 gen seeds of
  `[lpips(nullA) − lpips(fix)]`; positive = fix better. Item = clip×rung (seeds are correlated
  replicates, NOT independent n). Ties dropped; exact two-sided binomial SIGN test on the rest.
- Families: **F2 = ic3 two-sided eval items pooled across tiers (POWERED)**; F1 = specialist
  two-sided items (shadow_smoke+hero_flight, step-2000 only). If F1 n<10 items → F1 descriptive-only.
  n(F1) = [FILLED AFTER MANIFESTS]; n(F2) = [FILLED AFTER MANIFESTS].
- Material bar: median improvement ≥ max(P75 of |nullA−nullB| item deltas on the same
  items/metric, 0.01 LPIPS absolute).

**Verdict tiers (pre-committed):**
- IMPROVED (A): F2 sign test p<0.05 favorable AND material bar met AND F1 majority (≥60%)
  favorable AND guards pass (P3/P4/P5 within floor; secondary not significantly worse).
- WEAK (B): F2 p<0.20 favorable, or material bar met without significance; guards pass.
- NULL (C): deltas within floor — "no measurable effect at this scale."
- HARMFUL: significant degradation on primary or any guard.
- No metric shopping; no extra seeds/reruns after unblinding.

**Kill rules:** K1/K2 (above); K3 unit test (c) bitwise-identity (PASSED: 5/5 tests, commit
7d2ddb8); K4 fix-arm loss diverges >2× vs null/original at matched steps → halt; K5 (post-eval)
nullA−nullB floor comparable to fix−nullA effect → INDETERMINATE.

## How to run

```
# Phase 0 gate (K1/K2):
sbatch --partition=<gpu> --account=<acct> --gres=gpu:1 --export=ALL,MODE=gate \
    experiments/exp_073_cond_bleed_fix/job_precompute.sbatch
# Phase 0 build (cond_clean latents):  ... --export=ALL,MODE=build ...
# then assemble datasets, then 8 trainings (job_train_*.sbatch), gen, score.
```

## Outputs

- `gate_stats.json` (K1/K2), `build_report.json` (cond_clean provenance).
- Adapters: `outputs/training/exp_073_cond_bleed_fix/{shadow_smoke,hero_flight,ic3}_{fix,nullA,nullB}`.
- Videos: `outputs/videos/exp_073_cond_bleed_fix/...` (new out-roots; originals never overwritten).
- v4 scores + verdict: `outputs/eval/exp_073_*` + campaign verdict tables.
