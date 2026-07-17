# Eval Ladder v3 — Summary & Conclusions

*(compact presentation companion to `REPORT_v3.md` · certified instrument
`eval/v3.0.0` + amendments 1–3 · campaign 2026-07-16/17)*

⟨…⟩ = fills from the final aggregation (running now).

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
| + in-context demo | ic3 generalist | ⟨C5/C8/C9 verdict lines⟩ |

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
| **C5 (PRIMARY)** ic3·B−specialist | app_ref | ⟨…⟩ | ⟨…/8 (…)⟩ | ⟨…⟩ | ⟨generalist vs specialist on identical unseen items⟩ |
| | margin | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | |
| | max_seam_z | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | |
| **C8** ic3·B−basePE | app_ref | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨what the demo buys over conditioning⟩ |
| **C9** specialist-foreign−ic3·X (twins) | app_ref | ⟨…⟩ | ⟨…/8 (…)⟩ | ⟨…⟩ | ⟨pre-registered direction: specialist collapses harder⟩ |
| **C11** ic3 A−B (overfit gap) | app_ref | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨generalist's memorization gap vs C3's 0.845→0.706⟩ |

## Presentation Table 2 — tier table (trusted-class channel means)

⟨paste final `outputs/eval/ladder_v3/_contrasts/tier_table.md` — one row per
arm: base·P, base·PE, spec·SEEN/UNSEEN/FOREIGN @250/2000, ic3·A/B/C/X,
ic2·R4/R5, CONTROL hold/lerp⟩

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
| ic3 tiers | ⟨…⟩ | ⟨…⟩ |

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
3. ⟨**The in-context generalist verdict** — C5/C8/C9 in one sentence⟩
4. ⟨**Alignment value** — C10 descriptive note (ic2 contamination caveat)⟩
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
