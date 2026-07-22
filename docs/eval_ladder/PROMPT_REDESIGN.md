# Prompt redesign — defect record + retrain proposal (owner-directed, 2026-07-22)

Handoff brief for the next campaign agent. Direct and compact; evidence files
inline. Read together with `POOL_YARDSTICK.md` (reporting rules) and the
`exp-eval` skill (v4 instrument default).

## 1 · The defect

Every caption in the corpus has the form **"Scene1. The scene transforms into
Scene2."** and every arm — training AND inference, specialist AND generalist —
was prompted with `ICTRANS <full caption>` (trigger prepended at preprocess;
verified zero mismatches across exp_061/062/064/065 manifests). "Type-blind"
removed the class *name*, not the *outcome*. Three consequences:

1. **One-sided leak.** Keyed conditioning withholds the visual end anchor,
   then the prompt hands the model the complete end state in words. "The
   transition is embedded in the token/weights (or reference)" is confounded
   with "Gemma read the caption."
2. **Foreign contradiction.** R3X/R4X prompts are the *endpoints clip's own
   full caption* — text instructs the recipient's own transition while
   weights/reference demand the donor's. The arms are graded on the donor
   effect they were textually told not to perform.
3. **Trigger placement.** `ICTRANS` sits as a global prefix while the natural
   marker "The scene transforms into" — which the base model already
   understands — keeps doing the transition work mid-caption. The token
   likely binds to "transition-dataset sample", not to *the transition event*.

## 2 · Evidence (2026-07-22, all reproducible)

- **Foreign forensic** (`apps_top3` top-1 on ladder_v4h rows): r3x looks like
  the recipient class **100/132 (76%)** vs donor 19 (14%); ic3_x recipient
  **109/132 (83%)** vs donor 10 (8%). (Endpoints are also recipient-class —
  exp_074 disentangles text from endpoints.)
- **r0 (prompt-only)**: 63% overall / 70% trusted-classes of the GT appearance
  ceiling, and top-1 = own class 71% — the text channel alone carries most of
  the appearance yardstick.
- **Cracked-vs-keyed base**: one-sided r1 (saw end anchor) 98% vs r1k
  (prefix-only) 95% — the visual end anchor adds ~nothing *given the textual
  one*. The honest one-sided baseline does not exist yet.
- Scope notes: appearance-% is the most text-vulnerable metric (margin/2AFC
  less so). Paired same-prompt contrasts (C1/C4/C5/C8) remain valid as
  comparisons; their interpretation narrows to "value of visual conditioning
  GIVEN outcome text". F-001/F-002 untouched (no prompts involved). exp_073
  (bleed-fix) unaffected — both arms share prompts.

## 3 · Immediate patch (running): exp_074, inference-only

Corrected prompts on the EXISTING adapters — prompt states only the
endpoints' knowledge: one-sided/foreign → `ICTRANS <Scene1>` (V1); two-sided →
`ICTRANS <Scene1> <Scene2>` (transition wording removed). Plus a V2
marker-control lane (`… The scene transforms.` — phrase kept, outcome
removed) because current adapters never trained without the phrase: V2−orig
isolates outcome removal, V1−V2 isolates the format shift. R5 + R4X + R3X,
561 gens twin-matched to originals, then v4 scoring + pool yardstick +
forensic. See `experiments/exp_074_prompt_fix_rerun/README.md` (predictions
pre-registered there).

## 4 · Retrain proposal (the real fix — next campaign)

**Caption format** (train = inference, always):

| sidedness | caption |
|---|---|
| one-sided | `<Scene1> <TOKEN>` — nothing about the transition or the end scene |
| two-sided | `<Scene1> <TOKEN> <Scene2>` — token in the transition's temporal slot, no "transforms into" wording |

- The token replaces the natural transition marker *positionally*, so it must
  carry the event semantics instead of riding along as a global prefix.
- **Token choice — research task for the campaign agent**: `ICTRANS` is not a
  reserved token (Gemma-3 splits it into ordinary subwords; the text encoder
  is frozen — the DiT LoRA keys on the embedding pattern). Research and pick a
  better trigger: criteria = tokenizes to 1–2 stable subwords, near-zero
  semantic prior in Gemma-3, no collision with corpus vocabulary, survives
  mid-sentence placement grammatically. Report candidates + rationale before
  training.
- **Controllability preserved by caption-segment dropout**: with probability
  p (start at 0.5) train on the full two-scene caption, else the leak-free
  form above. Keeps text→outcome coupling alive for later steering ("the
  smoke is red") without text being the guaranteed workhorse. Controllability
  gets its OWN small probe lane (fixed endpoints/reference + attribute-modifier
  prompts; score: did only the named attribute change) — never mixed into the
  ladder claims.
- **Generalist (ic3)**: reference carries the transition; run a keyword-free
  arm (reference-presence = the switch, cleaner control contract) vs keyword
  arm before committing.
- **Conditioning**: keyed sidedness stays; adopt the exp_073 bleed-fix encode
  (separate prefix/suffix VAE encodes) if its frozen verdict passes — do NOT
  fold prompt and conditioning changes into one comparison; sequence them.

## 5 · Before ANY retraining: pre-flight audit (thorough, owner-mandated)

One pass over the CURRENT stack, findings written down, each item signed off:

1. **Training inputs**: dump 5 actual (caption, mask, conditioning, reference)
   tuples per model type from the preprocessed datasets — eyeball that what we
   *think* the model sees is what it sees (this doc exists because nobody did
   this for prompts).
2. **Inference inputs**: same dump from the gen manifests — prompt, anchors,
   reference, sampler settings; diff against training format.
3. **Eval semantics**: per arm — row reference kind (own-GT/demo/donor), pool
   coverage, ceiling, trust; confirm the scoring manifests inherit the
   corrected prompts.
4. **Known-defect regression list**: causal-VAE suffix bleed (exp_073), prompt
   outcome leak (this doc), copy_max reference semantics, ic2↔ic3_c item-id
   collision — each with its check command.

## 6 · Next ladder: simpler architecture (owner requirement)

The v3 ladder scattered ~7 training runs, 5 gen scripts, 20 eval labels and 4
manifest formats (r2/r3/r3x/r1k/ic2/ic3/ic3_x…). The next run is ONE
campaign experiment with data-driven arms:

- **One item registry** (single JSON: item_id, class, clip, seed, sidedness,
  tier, arm, adapter, conditioning, prompt) — generated by one builder from
  the split + corpus. Arms are ROWS, not experiments.
- **One training recipe file per model type** (specialist, generalist) —
  classes/datasets parameterized, configs generated, one output tree
  `outputs/training/<campaign>/<model>/`.
- **One gen runner** (exp_074's `run_gen.py` is the template: manifest-driven,
  adapter+targets per manifest, skip-if-exists, chunked arrays).
- **One scoring flow**: eval manifests derived FROM the registry (no hand-kept
  parallel copies), v4 instrument, pool yardstick computed in the same pass,
  viewer built from the same rows.
- Every artifact path derivable from item_id alone. If a new arm needs a new
  script, the design is wrong.

## 7 · Sequencing

1. exp_074 scores land → update foreign/zero-shot numbers + forensic (this doc §2).
2. exp_073 verdict (bleed-fix) → conditioning encode decision.
3. Pre-flight audit (§5) → sign-off.
4. Token research (§4) → owner picks token + dropout p.
5. Retrain ablation (2 specialist classes + 1 ic3) old-vs-new format → gate.
6. Full simplified ladder (§6) on the winning format.
