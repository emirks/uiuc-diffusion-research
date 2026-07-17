# Eval Ladder v3 — Summary & Conclusions

*(compact presentation companion to `REPORT_v3.md` · certified instrument
`eval/v3.0.0` + amendments 1–3 · campaign 2026-07-16/17 · final aggregation:
2,134 certified rows, 20/20 labels, 0 error rows)*

## The question and the answer shape

**Where does transition ability come from?** Four candidate sources, one model
family (LTX-2 19B), one frozen corpus (222 clips / 39 classes), one certified
instrument. Every claim is a *paired delta* on identical items with a
cross-class sign test and a seed-noise MDE gate — never an absolute score.

| Source of ability | Arm | Verdict (one line) |
|---|---|---|
| The base prior | base·P | contains the effect weakly; conditioning-starved |
| + endpoint conditioning | base·PE | **large, near-universal gain** (+0.229 app, 21/22 cls) |
| + per-class weights | specialists | **coherence, not class-likeness** (seam −10.8σ); collapses on foreign content |
| + in-context demo | ic3 generalist | **margin parity with the specialist on identical unseen items — by synthesis (3% near-copy), not memorization (100%)**; seam much better than base (21/25 cls), worse than specialist; zero-shot & foreign endpoints remain open |

## Design in one table

Tier = (was the class trained?) × (endpoint band). No adapter ever trains on a
test-band clip; sidedness owner-final; seeds 42/43/44; paired items everywhere.

| Model | Tiers (volumes) |
|---|---|
| base (frozen) | P 150 · PE 150+54+30 |
| 11 specialist LoRAs | SEEN 132 · UNSEEN·own 132 · UNSEEN·foreign 132 (prefix-only) |
| ic3 IC-LoRA generalist (retrained this campaign, split-aligned) | A held-in 45 · B unseen 99 · C zero-shot 21 · X foreign 132 (twins of specialist-foreign) |
| ic2 (frozen, contaminated) | comparison only, never headlines |

## Presentation Table 1 — key contrasts (certified v3.0.0)

| Contrast (A−B, same items) | Channel | Δ | classes+ (p) | ≥MDE | Reading |
|---|---|---|---|---|---|
| **C1** conditioning (PE−P) | app_ref | **+0.229** | 21/22 (<0.001) | YES | conditioning is the single largest lever |
| | margin | +0.114 | 18/22 (0.004) | YES | |
| **C4** specialist−basePE | app_ref | −0.020 | (0.18) | no | appearance parity — |
| | max_seam_z | **−10.8** | 10/11 (0.012) | YES | — but transitions become *coherent* |
| **C5 (PRIMARY)** ic3·B−specialist | **margin** | **−0.018** | 2/5 (1.0) | no | **parity on identical unseen items** |
| | copy_max | −0.647 | 0/6 (0.031) | YES | generalist synthesizes; specialist memorizes |
| | max_seam_z | +18.0 | 4/6 (0.69) | YES* | specialist keeps its seam edge (*sign-weak, heavy-tailed) |
| **C8** ic3·B−basePE | margin | −0.016 | 7/19 (0.36) | no | not "more class-like" than conditioning — |
| | max_seam_z | **−19.4** | 21/25 impr. (0.001) | YES | — but coherent, |
| | copy_max | −0.591 | 0/25 (<0.001) | YES | and synthesized (3% vs 100% near-copy) |
| **C9** specialist-foreign−ic3·X (twins) | app_ref | **+0.042** | **6/6 (0.031)** | YES | pre-registered direction CONFIRMED: |
| | margin | **+0.094** | **6/6 (0.031)** | YES | both collapse; specialist keeps a small residue |
| **C11** ic3 A−B (descriptive) | margin | 0.240 vs 0.187 | — | — | tiny held-in gap — no specialist-style overfit (C3: 0.306→0.237) |

## Presentation Table 2 — tier table (trusted-class channel means)

| model·tier | app_ref | margin | cam_dtw↓ | seam_z↓ | near-copy |
|---|---|---|---|---|---|
| base·P | 0.445 | 0.073 | 1.184 | 6.4 | 4% |
| base·PE | 0.659 | 0.175 | 1.059 | 7.5 | 100% |
| spec·SEEN@2000 | **0.845** | **0.306** | 0.698 | 4.2 | 98% |
| spec·UNSEEN@2000 | 0.706 | 0.237 | 1.019 | **0.04** | 100% |
| spec·FOREIGN | 0.205 | −0.141 | 1.197 | 2.0 | 0% |
| ic3·A held-in | 0.347† | 0.240 | 1.204 | **0.02** | 0% |
| ic3·B unseen | 0.408† | 0.187 | 1.051 | 4.4 | 3% |
| ic3·C zero-shot | 0.287† | 0.038 | 1.113 | −0.1 | 0% |
| ic3·X foreign | 0.162† | −0.240 | 1.237 | 7.6 | 0% |
| ic2·R4 (frozen, contaminated) | 0.321† | 0.221 | 1.195 | 1.3 | 0% |
| CONTROL hold | 0.358 | −0.010 | 1.071 | (outlier-dominated) | 47% |
| CONTROL lerp | 0.262 | −0.031 | 1.145 | −1.2 | 46% |

† ic-arm app_ref scores against the *demo reference* (pre-registered
convention) — not level-comparable with own-GT arms, whose app_ref also rides
near-copy inflation. **Compare arms on `margin`.** Full table (all channels,
n per cell): `outputs/eval/ladder_v3/_contrasts/tier_table.md`.

Reading guide: `app_ref` = appearance similarity to the reference (M1a, DINO
mean-of-max over core windows); `margin` = target-minus-intruder separation
(the cross-family-comparable column); `cam_dtw` lower=better; `max_seam_z`
lower=better (seam integrity); `copy_max`≥0.858 flags near-copy. Absolute
levels are only comparable *within* a reference convention — deltas are the
inferential unit.

## Presentation Table 3 — near-copy diagnostic (τ=0.858)

| Arm family | near-copy rate | interpretation |
|---|---|---|
| own-item arms (base·PE, spec·SEEN/UNSEEN) | ~98–100% | definitional: their GT is in the corpus |
| specialist FOREIGN | 0% | no content paste-on onto foreign endpoints |
| ic2 tiers | 0% | generalists synthesize |
| ic3 tiers | A 0% · B 3% · C 0% · X 0% | the aligned generalist synthesizes everywhere — even on held-in items |
| controls | ~47% | the detector flags the degenerate arms |

## Presentation Table 4 — v4.0.0 cross-comparison (NOT re-certified)

The metric-search upgrades (S3 / D_ZPR / CSLS), deployed as instrument v4.0.0;
reference population rebuilt for the corrected corpus per owner directive
2026-07-17. v3 stays the headline; v4 is the robustness cross-check.

- Bridge check (built-in): v4 carries the raw v3 appearance metric —
  |app_ref_v3 − certified v3 app_ref| = **0.00000** over shared H100 rows
  (mixed-GPU insurance lane drift 7e-5 ≈ MDE/300 — quantifies the
  H100-purity doctrine).
- ⟨key contrasts under v4 metrics: C1/C4/C5/C9 rows from
  `ladder_v4h/_contrasts/contrasts_v4.md` — agreement/disagreement notes⟩

## Conclusions

1. **Endpoint conditioning is the foundation** — the largest, most universal
   effect in the study (C1).
2. **Per-class specialist training does not make outputs more class-like on
   unseen content — it makes them *mechanically coherent*** (C4: seam
   integrity transformed, appearance null), and its value is **content-bound**:
   on foreign endpoints the specialist falls below the degenerate-control
   floor (0.205 < 0.36).
3. **One in-context generalist matches eleven specialists on their own
   unseen items (margin parity, C5) — and does it by synthesis, not
   memorization** (near-copy 3% vs 100%; no specialist-style overfit gap,
   C11). What it gives up: the specialist's seam-integrity edge (C5) and a
   small appearance residue on foreign endpoints (C9, 6/6, confirmed). What
   it keeps over plain conditioning: coherent seams (21/25 classes) and
   non-degenerate synthesis (C8). Zero-shot onto unseen classes is not yet
   real (C6: below conditioned base).
4. **Decontaminating the training split cost nothing** — ic3 on genuinely
   unseen items ties ic2 on items ic2 had trained on (C10b margin −0.002,
   null; descriptive by pre-registration).
5. Measurement discipline made these claims cheap to defend: certified
   instrument, pre-registered contrasts, paired deltas, MDE gates,
   fail-forward, and a second instrument (v4) as a free robustness check.

## Caveats (complete list in REPORT_v3 §8)

- obj/cam channels are †-heavy (tiny trusted-class counts) — claims rest on
  app_ref/margin/seam.
- C6/C7/C10 descriptive (n small or contaminated baseline); C9 extension
  exploratory.
- v4 numbers are cross-checks only (instrument not re-certified for the
  rebuilt reference).
