"""IC-LoRA trainings — build the multi-run results viewer.

    python eval_ladder/viewer/build_runs.py [--out outputs/reports/iclora_runs/index.html]

ONE page that holds EVERY IC-LoRA training run, with a chip per run. Adding the next training is
one entry in RUNS below — nothing else changes.

Provenance: forked from `eval_ladder/viewer/build.py` (+ `template.html`), which builds the ladder2
results viewer. That page is the published record of the ladder2 campaign and its numbers bind to
pre-registered claims, so it is NOT touched and NOT extended — this is a sibling. The fork is
deliberate: a shared template would couple that page's evolution to this one's.

WHAT IS DIFFERENT FROM build.py
  * a run is a first-class dimension. `build.py` has one hardcoded `generalist` tier; here each
    entry in RUNS gets its own tier column, and the chips multi-select over them. Both runs are on
    by default, so the DEFAULT VIEW IS THE COMPARISON.
  * scores carry their INSTRUMENT. `report_full.SCORES` is a module-level constant pointing at
    `outputs/eval/ladder2` and it ignores $LADDER_SCORES — so importing it and calling load_scored()
    silently reads the stale-artifact score set. That is the exact cross-instrument trap this
    campaign was bitten by, so here every score set is loaded explicitly, by path, and every
    generation is tagged with which artifact scored it. Nothing is implicit.
  * every join is asserted with counts. A silent no-op that exits 0 cost this campaign four
    debugging windows; see `check()`.

THE CARD MODEL is inherited verbatim and is still two levels with no view-time joins:

    card = one INPUT   (donor_class + endpoint + sidedness; ic_gen and ctt_v2 share `input_key`
                        field-for-field, so both runs' answers land in the SAME card natively)
    gen  = one arm's answer to that input, already averaged pool-refs -> seeds
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
REPO_ROOT = LADDER.parents[0]
LAB = REPO_ROOT.parent
sys.path.insert(0, str(LADDER))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402
import report_full as rf  # noqa: E402
import run_eval  # noqa: E402

STD = REPO_ROOT / "data/processed/transitions_std121"
SEEDS = (42, 43)

# --------------------------------------------------------------------------------- the runs
#: One entry per IC-LoRA training. `gen_dir` is EXPLICIT (repo-relative) rather than derived from
#: $LADDER_GEN_STEP — that env var is what made the scorer look in `ctt_v2/` instead of
#: `ctt_v2__ck10000/` and silently score nothing. A checkpoint is part of the run's identity, so
#: a second checkpoint of the same adapter is simply a second entry.
RUNS = [
    {"id": "ic_gen", "arm": "ic_gen", "checkpoint": None,
     "label": "IC-LoRA generalist", "sub": "ladder2 · the incumbent",
     "gen_dir": "store/gens/001_ic_gen/videos", "registry": None},
    {"id": "ctt_v2", "arm": "ctt_v2", "checkpoint": 10000,
     "label": "CTT v2 · neutral prompt", "sub": "step 10,000 · rank 128 · one-way ref attention",
     "gen_dir": "store/gens/002_ctt_v2/videos",
     "registry": "eval_ladder/registry_ctt_v2.jsonl"},
]
RUN_BY_ARM = {r["arm"]: r for r in RUNS}
RUN_TIER = {r["arm"]: f"run_{r['id']}" for r in RUNS}

# -------------------------------------------------------------- arms that bring their own prompt
#: Arms run over THIS page's grid that are NOT IC-LoRA trainings of this campaign — prior-work
#: baselines, and our own re-runs under a changed prompt. Exactly like `specialist` and `copier`
#: they are CONTEXT tiers: they get a column and they enter the per-arm aggregate tables, but they
#: never join the paired Δ or the sign test. Adding the next one is one entry here and nothing else.
#:
#: They join on `item_id`: these arms ran the ctt_v2 registry's own 152 rows at the same two
#: seeds, so a source row names a registry row and the card follows from it. An arm here never
#: creates a card — this page is about the trainings.
#:
#: THE ONE THING THEY NEED THAT THE RUNS DO NOT. Every run shares the registry prompt; each of
#: these carries its OWN prompt, per row, and the difference between the two is the reason they
#: exist. So each brings its source rows, and the page shows the prompt that produced the clip it
#: is showing (`prompt`/`prompt_hi` on the generation, `alt_prompts` on the card).
#:
#: PER-ENTRY FIELDS THAT CARRY A SEMANTIC GUARD — set them, never infer them at view time:
#:   `frames`  the arm's own clip length. 121 = our geometry; anything else earns the `†`
#:             absolute-frame caveat on core_degenerate / copy_max (see WINDOW_CAVEAT).
#:   `no_twin` the arm has NO base twin, so it contributes a LEVEL and never a margin — the `‡`
#:             caveat. Toggling a column changes visibility; it never changes either caveat.
#:   `rows`    where the per-row prompts come from: a refVFX-style manifest (one row per item AND
#:             seed, with `out_name`) or a registry (one row per item; clips are `<item>__s<seed>`).
#:
#: `scores` is a SLOT. It is scored separately, later; until that directory holds harness output
#: the arm renders as "unscored — video only" and contributes no numbers anywhere. Nothing is
#: estimated, borrowed from another arm, or filled in.
EXTERNAL = [
    #: 2026-07-31 — THE PAGE'S TWO BASELINES, and the only two. Both are the LTX-2 base weights
    #: with no adapter, over this page's own 152 rows at the same two seeds and the same
    #: 480x640x121f geometry, so they are per-card comparable with every column here. They are
    #: what the `no_baseline_note` used to say did not exist.
    #:
    #: Their prompt is the leaky prompt with the trained `sksz.` token REMOVED: the token means
    #: nothing to weights that never saw it, and a baseline handed a nonsense string in the exact
    #: slot where the treatment arms carry their learned behaviour is a strawman. What is left
    #: states the transition in plain language — the only channel a no-reference model has.
    #:
    #: NEITHER IS SHOWN THE DEMO (`use_reference: false` on the row). The reference still sits on
    #: the row because `run_eval.pool_refs()` bans it from the GT pool, so removing it would hand
    #: these arms a different pool than the arms they are read against. That is also why `ref` is
    #: cleared in external_gen rather than here: it is scoring identity, not an input.
    {"id": "base_prompt_ctt", "score_id": "base_v4", "kind": "baseline", "frames": 121,
     "no_twin": True,
     "label": "⓪ BASE · prompt only",
     "sub": "no adapter, no anchors, no demo · the floor",
     "src": REPO_ROOT / "store/gens/006_base_prompt_ctt/videos",
     "media": "outputs/videos/base_arms/base_prompt_ctt",
     "rows": ("registry", REPO_ROOT / "store/gens/006_base_prompt_ctt/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/002_base_arms__dai__2026-07-31/base_prompt_ctt",
     "prompt_kind": "base weights, text only. The transition is described in plain language (the "
                    "effect-prompt arm's effect clause, with the trained <span class='mono'>sksz</span> "
                    "token removed) and nothing else is given — no prefix, no suffix, no demo. "
                    "This is what the prompt alone is worth.",
     "doc": "misc/base_arms/README.md"},
    {"id": "base_cond_ctt", "score_id": "base_v4", "kind": "baseline", "frames": 121,
     "no_twin": True,
     "label": "① BASE · prompt + endpoints",
     "sub": "no adapter, our anchors, no demo · the honest no-adapter twin",
     "src": REPO_ROOT / "store/gens/007_base_cond_ctt/videos",
     "media": "outputs/videos/base_arms/base_cond_ctt",
     "rows": ("registry", REPO_ROOT / "store/gens/007_base_cond_ctt/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/002_base_arms__dai__2026-07-31/base_cond_ctt",
     "prompt_kind": "the same text-only prompt as ⓪, plus the endpoint conditioning every "
                    "training arm receives (prefix 9f, and suffix on two-sided rows). Still no "
                    "demo — a base model handed a reference is a copier, not a baseline (owner "
                    "ruling 2026-07-23). This is the level an adapter has to beat.",
     "doc": "misc/base_arms/README.md"},
    {"id": "ctt_v2_leaky", "score_id": "leaky_v4", "kind": "ours", "frames": 121, "no_twin": True,
     "label": "⑥ CTT v2 · effect prompt",
     "sub": "our adapter, prompt also describes the transition · level, not a margin",
     "src": REPO_ROOT / "store/gens/005_ctt_v2_leaky/videos",
     "media": "outputs/videos/ctt_v2_leaky/clips",
     "rows": ("registry", REPO_ROOT / "store/gens/005_ctt_v2_leaky/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/001_five_arm__dai__2026-07-30/ctt_v2_leaky",
     "prompt_kind": "our own ctt_v2 adapter, same weights / references / endpoint conditioning / "
                    "geometry / seeds — the ONLY change is the prompt, which now also describes "
                    "the transition (the effect clause is inserted after the trained `sksz.` "
                    "token). The mirror of Ⓐ: our model handed the same text budget.",
     "doc": "misc/ctt_v2_leaky/DOSSIER.md"},
    #: 2026-08-02 — campaign `bneck_coupling`. THESE TWO ARE A PAIR AND ARE ONLY MEANINGFUL AS ONE.
    #: Same adapter file, same rows, same seeds, byte-identical GT pool; the ONLY difference is
    #: which clip the FROZEN certified encoder saw. Their paired difference is the campaign's
    #: measurement; their LEVELS carry no bar (advisor A11 forbids level claims here, including
    #: reading ⑧'s higher G-unseen-same level as "shuffled codes help").
    {"id": "bneck_frozen", "score_id": "bneck_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "⑦ BNECK · frozen code (matched)",
     "sub": "raw demo REPLACED by 72 frozen operator tokens · treatment",
     "src": REPO_ROOT / "store/gens/008_bneck_frozen/videos",
     "media": "outputs/videos/bneck_coupling/bneck_frozen",
     "rows": ("registry", REPO_ROOT / "store/gens/008_bneck_frozen/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/003_bneck_coupling__dai__2026-08-02/bneck_frozen",
     "prompt_kind": "the ctt_v2 recipe with ONE content change: the reference channel carries 72 "
                    "operator tokens from the certified transition encoder, held FROZEN (verified "
                    "bitwise at step 10,000), instead of the raw full-resolution demo. Same prompt "
                    "as ctt_v2. Compare ONLY against ⑧, its shuffled-code twin.",
     "doc": "misc/bneck_coupling/DOSSIER.md"},
    {"id": "bneck_frozen_shufcode", "score_id": "bneck_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "⑧ BNECK · shuffled code (corpse)",
     "sub": "SAME adapter file, deliberately WRONG code · the control",
     "src": REPO_ROOT / "store/gens/009_bneck_frozen_shufcode/videos",
     "media": "outputs/videos/bneck_coupling/bneck_frozen_shufcode",
     "rows": ("registry", REPO_ROOT / "store/gens/009_bneck_frozen_shufcode/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/003_bneck_coupling__dai__2026-08-02/bneck_frozen_shufcode",
     "prompt_kind": "byte-identical to ⑦ in every field — same adapter file, same prompt, same "
                    "reference, same seeds — except that the encoder is fed a DIFFERENT clip via "
                    "`code_source_reference` (a class-level derangement; 152/152 mapped, 0 fixed "
                    "points). The row's own `reference` is untouched, so both twins are scored "
                    "against a byte-identical GT pool. ⑧ scoring the same as ⑦ is the finding.",
     "doc": "misc/bneck_coupling/DOSSIER.md"},
    #: 2026-08-05 — campaign `bneck_redesign`, the operator-token REDESIGN 2x2 (rep x residual/raw).
    #: Each arm is a matched/deranged PAIR read exactly like ⑦/⑧: same adapter file, same rows, same
    #: seeds, byte-identical GT pool, the ONLY difference being which clip the code encoder saw
    #: (152/152 class-level derangement reused verbatim from bneck_coupling). Their paired Δapp_ref is
    #: the measurement; LEVELS carry no bar. Recalibrated arm bars (band-setter): G-unseen-same ≥9/13,
    #: G-unseen-cross ≥8/13, P2 ≥ +0.10. Scored on DeltaAI, v4 instrument sha 459fd9a7 (UNCERTIFIED
    #: by design for v4). Added as they score; HRC-residual first.
    {"id": "hrc_coupling", "score_id": "redesign_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "Ⓗ HRC-resid · matched code",
     "sub": "raw demo REPLACED by 72 HRC residual operator tokens · treatment",
     "src": REPO_ROOT / "outputs/videos/bneck_redesign/hrc_coupling__ck10000",
     "media": "outputs/videos/bneck_redesign_arms/hrc_coupling",
     "rows": ("registry", REPO_ROOT / "misc/bneck_redesign/build/registry_hrc_coupling.jsonl"),
     "scores": REPO_ROOT / "misc/bneck_redesign/eval/scores/hrc_coupling",
     "prompt_kind": "the ctt_v2 recipe with ONE content change: the reference channel carries 72 "
                    "residual operator tokens from the from-scratch native-basis HRC encoder "
                    "(trained on the endpoint-subtracted residual, λ=1.0), coupled at step 10,000, "
                    "instead of the raw full-resolution demo. Same prompt / endpoints / seeds as "
                    "ctt_v2. Compare ONLY against its deranged-code twin.",
     "doc": "misc/bneck_redesign/DOSSIER.md"},
    {"id": "hrc_coupling_shufcode", "score_id": "redesign_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "Ⓗ HRC-resid · deranged code",
     "sub": "SAME adapter file, deliberately WRONG code · the control",
     "src": REPO_ROOT / "outputs/videos/bneck_redesign/hrc_coupling_shufcode__ck10000",
     "media": "outputs/videos/bneck_redesign_arms/hrc_coupling_shufcode",
     "rows": ("registry", REPO_ROOT / "misc/bneck_redesign/build/registry_hrc_coupling_shufcode.jsonl"),
     "scores": REPO_ROOT / "misc/bneck_redesign/eval/scores/hrc_coupling_shufcode",
     "prompt_kind": "byte-identical to the matched HRC arm in every field except the encoder is fed "
                    "a DIFFERENT clip via `code_source_reference` (the same 152/152 class-level "
                    "derangement, 0 fixed points, reused from bneck_coupling). The row's own "
                    "`reference` is untouched, so both twins score against a byte-identical GT pool. "
                    "Scoring the same as the matched arm is the finding (clean null: P2 −0.003).",
     "doc": "misc/bneck_redesign/DOSSIER.md"},
    {"id": "vjepa_coupling", "score_id": "redesign_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "Ⓥ V-JEPA-resid · matched code",
     "sub": "raw demo REPLACED by 144 V-JEPA2 residual tokens · treatment",
     "src": REPO_ROOT / "outputs/videos/bneck_redesign/vjepa_coupling__ck10000",
     "media": "outputs/videos/bneck_redesign_arms/vjepa_coupling",
     "rows": ("registry", REPO_ROOT / "misc/bneck_redesign/build/registry_vjepa_coupling.jsonl"),
     "scores": REPO_ROOT / "misc/bneck_redesign/eval/scores/vjepa_coupling",
     "prompt_kind": "the ctt_v2 recipe with the reference channel replaced by 144 residual operator "
                    "tokens from a FROZEN pretrained V-JEPA2-ViT-L backbone (bitwise-frozen) through "
                    "a trainable projector into the DiT's 128-ch latent basis, trained on the "
                    "endpoint-subtracted residual trajectory and jointly coupled with the LoRA "
                    "(reference probability 0.9), instead of the raw demo. Same prompt / endpoints / "
                    "seeds as ctt_v2. Compare ONLY against its deranged-code twin.",
     "doc": "misc/bneck_redesign/DOSSIER.md"},
    {"id": "vjepa_coupling_shufcode", "score_id": "redesign_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "Ⓥ V-JEPA-resid · deranged code",
     "sub": "SAME adapter file, deliberately WRONG code · the control",
     "src": REPO_ROOT / "outputs/videos/bneck_redesign/vjepa_coupling_shufcode__ck10000",
     "media": "outputs/videos/bneck_redesign_arms/vjepa_coupling_shufcode",
     "rows": ("registry", REPO_ROOT / "misc/bneck_redesign/build/registry_vjepa_coupling_shufcode.jsonl"),
     "scores": REPO_ROOT / "misc/bneck_redesign/eval/scores/vjepa_coupling_shufcode",
     "prompt_kind": "byte-identical to the matched V-JEPA arm in every field except the encoder is "
                    "fed a DIFFERENT clip via `code_source_reference` (the same 152/152 class-level "
                    "derangement, 0 fixed points). The row's own `reference` is untouched, so both "
                    "twins score against a byte-identical GT pool. Scoring the same as the matched "
                    "arm is the finding (clean null: P2 +0.011, both claim cells below bar).",
     "doc": "misc/bneck_redesign/DOSSIER.md"},
    #: 2026-08-06 — campaign `bneck_redesign`, the CLEAN Idea-1 arm. Registered in the STORE
    #: (runs/006, gens/013+014, evals/004) — src/rows/scores are store paths, exactly like ⑦/⑧.
    #: Read like every other bottleneck pair: same adapter file, same rows, same seeds,
    #: byte-identical GT pool (measured pool-identity TRUE), the ONLY difference being which clip the
    #: frozen code encoder saw (the same 152/152 class-level derangement reused from bneck_coupling).
    #: The redesign vs ⑦: the 72 operator tokens are compressed by a co-trained ContextAdapter to
    #: K'=16 tokens and injected as cross-attention CONTEXT (inject=context), not concatenated as
    #: reference tokens. Their paired Δapp_ref is the measurement; LEVELS carry no bar. Band-setter
    #: bars: G-unseen-same ≥9/13, G-unseen-cross ≥8/13, P2 ≥ +0.10. Scored on DeltaAI, v4 sha
    #: 459fd9a7 (UNCERTIFIED by design for v4). MEASURED NULL: 6/13 & 6/13, P2 −0.008.
    {"id": "bneck_ctx_v2", "score_id": "redesign_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "Ⓒ CTX-v2 · matched code",
     "sub": "raw demo REPLACED by 16 frozen context tokens (K'=16) · treatment",
     "src": REPO_ROOT / "store/gens/013_bneck_ctx_v2/videos",
     "media": "outputs/videos/bneck_redesign_arms/bneck_ctx_v2",
     "rows": ("registry", REPO_ROOT / "store/gens/013_bneck_ctx_v2/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/004_bneck_ctx_v2__dai__2026-08-06/bneck_ctx_v2",
     "prompt_kind": "the ctt_v2 recipe with ONE content change: the reference channel carries the "
                    "certified transition encoder's 72 operator tokens, held FROZEN, compressed by a "
                    "co-trained ContextAdapter to 16 context tokens and injected as cross-attention "
                    "CONTEXT (inject=context) rather than concatenated as reference tokens (⑦). Same "
                    "prompt / endpoints / seeds as ctt_v2. Compare ONLY against its deranged-code twin.",
     "doc": "misc/bneck_redesign/DOSSIER.md"},
    {"id": "bneck_ctx_v2_shufcode", "score_id": "redesign_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "Ⓒ CTX-v2 · deranged code",
     "sub": "SAME adapter file, deliberately WRONG code · the control",
     "src": REPO_ROOT / "store/gens/014_bneck_ctx_v2_shufcode/videos",
     "media": "outputs/videos/bneck_redesign_arms/bneck_ctx_v2_shufcode",
     "rows": ("registry", REPO_ROOT / "store/gens/014_bneck_ctx_v2_shufcode/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/004_bneck_ctx_v2__dai__2026-08-06/bneck_ctx_v2_shufcode",
     "prompt_kind": "byte-identical to the matched CTX-v2 arm in every field except the encoder is "
                    "fed a DIFFERENT clip via `code_source_reference` (the same 152/152 class-level "
                    "derangement, 0 fixed points, reused from bneck_coupling). The row's own "
                    "`reference` is untouched, so both twins score against a byte-identical GT pool. "
                    "Scoring the same as the matched arm is the finding (clean null: P2 −0.008, both "
                    "claim cells below bar).",
     "doc": "misc/bneck_redesign/DOSSIER.md"},
    {"id": "surg1_wsd", "score_id": "redesign_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "Ⓢ SURG-1 · matched code",
     "sub": "V-JEPA 144-tok code · objective surgery (WSD) · treatment",
     "src": REPO_ROOT / "store/gens/015_surg1_wsd/surg1_wsd__ck4500",
     "media": "outputs/videos/surg1_arms/surg1_wsd",
     "rows": ("registry", REPO_ROOT / "store/gens/015_surg1_wsd/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/006_surg1_wsd__dai__2026-08-11/surg1_wsd",
     "prompt_kind": "the ctt_v2 recipe with the reference channel REPLACED by a compact 144-token "
                    "V-JEPA transition code (one-way, backbone-free gen: trained projector from the "
                    "step-4500 checkpoint). Objective surgery = high-sigma timestep mixture + code-swap "
                    "contrastive gap loss. Same prompt/endpoints/seeds. Gate B: reads-but-weakly "
                    "(matched > twin, P1 cross 9/13, P2 +0.020 << 0.1016). Compare ONLY vs its twin.",
     "doc": "misc/2026-08-10_encoder_branch_redteam/DOSSIER.md"},
    {"id": "surg1_wsd_shufcode", "score_id": "redesign_v4", "kind": "bottleneck", "frames": 121,
     "no_twin": True, "same_prompt_by_design": True,
     "label": "Ⓢ SURG-1 · deranged code",
     "sub": "SAME adapter, cross-class WRONG code · the must-fail control",
     "src": REPO_ROOT / "store/gens/016_surg1_wsd_shufcode/surg1_wsd_shufcode__ck4500",
     "media": "outputs/videos/surg1_arms/surg1_wsd_shufcode",
     "rows": ("registry", REPO_ROOT / "store/gens/016_surg1_wsd_shufcode/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/006_surg1_wsd__dai__2026-08-11/surg1_wsd_shufcode",
     "prompt_kind": "byte-identical to the matched SURG-1 arm except the V-JEPA code source is a "
                    "DIFFERENT manner class via `code_source_reference`; the row's own `reference` is "
                    "untouched (byte-identical GT pool). It scored LOWER than matched in both claim "
                    "cells — that IS the read signal (pool-% Δpp +2.7 same, +2.8 cross).",
     "doc": "misc/2026-08-10_encoder_branch_redteam/DOSSIER.md"},
    {"id": "refvfx_A", "score_id": "refvfx_v4", "kind": "prior-work", "frames": 33,
     "label": "Ⓐ refVFX · effect prompt",
     "sub": "external baseline · prompt describes the effect",
     "src": REPO_ROOT / "store/gens/003_refvfx_A/videos",
     "media": "outputs/videos/refvfx_baseline/refvfx_A",
     "rows": ("manifest", REPO_ROOT / "store/gens/003_refvfx_A/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/001_five_arm__dai__2026-07-30/refvfx_A",
     "prompt_kind": "refVFX's own convention — the prompt NAMES the effect the demo shows, so "
                    "text and demo agree. Their model at its strongest; NOT text-matched to ours.",
     "doc": "misc/refvfx_baseline/RECORD.md"},
    {"id": "refvfx_B", "score_id": "refvfx_v4", "kind": "prior-work", "frames": 33,
     "label": "Ⓑ refVFX · neutral prompt",
     "sub": "external baseline · no transition information in text",
     "src": REPO_ROOT / "store/gens/004_refvfx_B/videos",
     "media": "outputs/videos/refvfx_baseline/refvfx_B",
     "rows": ("manifest", REPO_ROOT / "store/gens/004_refvfx_B/grid.jsonl"),
     "scores": REPO_ROOT / "store/evals/001_five_arm__dai__2026-07-30/refvfx_B",
     "prompt_kind": "our arms' text budget, in their vocabulary — a class-agnostic effect clause "
                    "in place of our `sksz`. Same weights, same seeds, same geometry as Ⓐ; the "
                    "prompt is the only field that differs.",
     "doc": "misc/refvfx_baseline/PROMPT_DESIGN_ours.md"},
]
#: kind -> (heading, the paragraph that governs how that kind may be read). Rendered per kind, in
#: this order, above that kind's arms — a single paragraph covering both would have to be vague
#: about which caveat belongs to which arm.
ARM_KINDS = [
    ("baseline",
     "The two baselines.",
     "⓪ and ① are the LTX-2 base weights — <b>no adapter</b> — over these same 152 rows, seeds "
     "and geometry (480×640×121f @ 24fps). They differ from each other in exactly one field: ① "
     "receives the endpoint conditioning (prefix 9f, suffix 8f on two-sided rows) and ⓪ receives "
     "nothing but text, so <b>⓪ vs ① prices the anchors</b> and either against a training prices "
     "the adapter. Both are handed the transition <b>in the prompt</b>, since a model with no "
     "demo and no adapter has no other way to know what to produce; the clause is the same one ⑥ "
     "and Ⓐ carry, with the trained <span class='mono'>sksz</span> token removed. <b>Neither is "
     "ever shown the in-context demo</b> — a base model handed a reference copies it, which is "
     "why the old BASE+DEMO column was a warning rail and not a baseline, and why it is no longer "
     "on this page. They are read as <b>levels</b>: the page's paired Δ and sign test remain "
     "between the two trainings, which are the only columns that answer an identical input.",
     ),
    ("ours",
     "Our effect-prompt arm.",
     "⑥ is the SAME ctt_v2 adapter as ⑤ — same weights, references, endpoint conditioning, "
     "geometry (480×640×121f @ 24fps) and seeds. One field changed: the prompt now also describes "
     "the transition, the effect clause inserted straight after the trained <span class='mono'>"
     "sksz.</span> token. It is the mirror of Ⓐ, and the comparison it was built for is <b>⑥ vs ⑤ "
     "as levels</b>, joined on the registry row. <b>It has no base twin, by design</b> — the "
     "honest control would be the base model handed the effect prompt, and that generation does "
     "not exist; pairing it against a base video rendered from the PLAIN prompt would attribute "
     "the prompt change to the adapter. So it never enters the paired Δ or the sign test, and its "
     "column is marked ‡. Read it knowing this prompt shape is <b>out of distribution</b> for the "
     "adapter: no training caption ever named an effect, so a drop is as consistent with OOD "
     "prompt shape as with the description being unhelpful.",
     ),
    ("bottleneck",
     "The operator-token bottleneck pair — a measured NEGATIVE.",
     "⑦ and ⑧ are <b>one experiment, not two arms</b>, and neither number means anything alone. "
     "Both are generated from the <b>same adapter file</b> over the same 152 rows and seeds; the "
     "only difference is which clip the <b>frozen</b> certified transition encoder was shown. In "
     "⑦ it sees the row's own demo; in ⑧ a deliberately wrong one, delivered through a separate "
     "<span class='mono'>code_source_reference</span> field so the row's own reference — and "
     "therefore the GT pool both are scored against — stays byte-identical. <b>The measurement is "
     "their paired difference, and it came out at chance</b>: 6/13 and 7/13 donor classes positive "
     "against a pre-registered bar of ≥11/13, pooled median Δapp_ref −0.002 against a bar of "
     "+0.05, with a 95% CI whose upper bound (+0.007) sits <b>below</b> the bar — so the design "
     "could have detected the effect and did not. This is a <b>null, not an underpowered study</b>. "
     "Separately measured and also true: the channel is <b>causally live</b> — swapping the code "
     "moves the generated pixels at ~44% of a raw demo's leverage, against a constant-code floor of "
     "exactly zero. <b>The coupling transmits but does not instruct.</b> Read ⑦ and ⑧ only against "
     "each other: their <b>levels carry no bar</b>, and ⑧'s slightly higher level on G-unseen-same "
     "must <b>not</b> be read as “shuffled codes help”.",
     ),
    ("prior-work",
     "External baselines.",
     "Two external arms, one prior-work model (refVFX, arXiv:2601.07833, unofficial CMU "
     "reimplementation), over this page's own 152 rows at the same two seeds. They share every "
     "weight, input, hyper-parameter and seed and differ in exactly one manifest field — the "
     "PROMPT — so Ⓐ vs Ⓑ isolates what the text budget is worth to them. Their geometry is their "
     "own (a first-frame anchor, 33f, duration-matched to our 121f@24fps), so the prefix/suffix "
     "bar is replaced by a statement of their contract rather than redrawn as if it were ours.",
     ),
]

# --------------------------------------------------------------------------- the score sets
#: Ordered by preference. A generation takes its numbers from the FIRST set that scored it, and
#: carries that set's id in `instr` so the page can badge it. `primary` is the comparison-valid
#: instrument: ic_gen and ctt_v2 were both rescored under it on eps, which is the only reason a
#: cross-run comparison is meaningful at all.
#:
#: `path` is one directory; `paths` is a list of them, merged in order, for a set that lives in one
#: directory PER ARM. That is the shape the DeltaAI re-score of ic_gen/ctt_v2 lands in, so pointing
#: the run columns at it is a single new entry here — see misc/refvfx_baseline/VIEWER_NOTES.md.
#: Paths may be repo-relative or absolute; `../misc/...` reaches $LAB.
SCORE_SETS = [
    #: 2026-07-30 — the run columns moved HERE, onto the box the external arms were scored on, so
    #: the comparison this page exists to show is single-machine. `base` (the copier tier) was not
    #: part of that re-score and still falls through to `rebuilt222`, which is why the machine note
    #: does not disappear: it narrows to the context tiers.
    #:
    #: 🔴 ONE DIRECTORY PER ARM, AND ONLY THESE TWO. `ctt_v2_leaky`, `refvfx_A` and `refvfx_B` sit
    #: in sibling directories under the same parent and are MEASURED to reuse the ctt_v2 registry's
    #: eval item_ids EXACTLY — 1,842 of 1,842 collide, treatment rows and control rows alike, and
    #: the harness `arm` field is the ONLY thing that distinguishes them. Adding one of those
    #: directories to this list would not raise: `load_all_scores` merges `paths` by eval id, so
    #: their rows would be concatenated into the ctt_v2 column and averaged with it. `assert_arms`
    #: below now makes that fail loudly, and this comment says why it exists. Those arms load
    #: through EXTERNAL, each into its own dict, which is what keeps the columns separate.
    {"id": "dai222", "primary": True, "corpus": "dc2e139a",
     "paths": ["store/evals/001_five_arm__dai__2026-07-30/ic_gen",
               "store/evals/001_five_arm__dai__2026-07-30/ctt_v2"],
     "short": "reference_v4 · 222-clip corpus dc2e139a · rescored on DeltaAI",
     "label": "reference_v4 on the pinned 222-clip corpus (dc2e139a), both adapters rescored on "
              "DeltaAI alongside the refVFX arms, 2026-07-30 — run and external columns therefore "
              "come off one machine"},
    {"id": "rebuilt222", "primary": False,
     "path": "outputs/eval/ctt_v2_compare", "corpus": "dc2e139a",
     "short": "rebuilt reference_v4 · 222-clip corpus dc2e139a",
     "label": "reference_v4 rebuilt on the pinned 222-clip corpus (dc2e139a); both adapters "
              "rescored together on eps, 2026-07-30"},
    {"id": "stale223", "primary": False,
     "path": "outputs/eval/ladder2", "corpus": "aa28c6d5",
     "short": "as-published ladder2 · 223-clip reference_v4",
     "label": "the as-published ladder2 scores, computed under the superseded 223-clip "
              "reference_v4 (aa28c6d5) before live_concert_2 was quarantined"},
]

#: Owner instruction 2026-07-30: the specialist generations must be visible here, as they are in
#: the ladder2 results viewer. Specialists were never rescored on eps (no GPU budget, and the owner
#: forbade new scoring), so their numbers exist ONLY under the superseded artifact.
#:
#: The advisor's two-class instrument policy (after the cross-build error was measured): a
#: stale-scored tier appears WITH its numbers, badged, and may enter the aggregate panel as its own
#: column — never merged into a run's column. The justification is on the page: the same 304 ic_gen
#: videos scored under BOTH artifacts differ by 0.09pp mean / 0.31pp max at CELL level, which is the
#: regime aggregates live in. Per-GENERATION the tail reaches 8.3pp, which is why no per-card
#: specialist-vs-run delta is ever drawn — that comparison happens only in the aggregate.
SECONDARY_IN_STATS = True

#: metric -> (label, direction, decimals, group). Groups become the collapsible tables.
METRICS = [
    ("app_ref", "M1a app_ref", "up", 3, "ref"),
    ("cam_zpr", "M1b cam_zpr", "down", 3, "ref"),
    ("obj_csls", "M1c obj_csls", "down", 3, "ref"),
    ("copy_max", "M2a copy_max", "down", 3, "ref"),
    ("cam_dtw", "cam_dtw", "up", 3, "ref"),
    ("cam_corr", "cam_corr", "up", 3, "ref"),
    ("obj_match", "obj_match", "up", 3, "ref"),
    ("app_ref_v3", "app_ref_v3", "up", 3, "ref"),
    ("cross", "cross", "info", 3, "ref"),
    ("margin", "M2b margin", "up", 3, "gen"),
    ("app_target", "app_target", "up", 3, "gen"),
    ("prefix_dino", "M3a prefix_dino", "up", 3, "gen"),
    ("prefix_lpips", "M3a prefix_lpips", "down", 4, "gen"),
    ("max_seam_z", "M3b max_seam_z", "down", 2, "gen"),
    ("scalar_depth", "depth", "info", 3, "gen"),
    ("scalar_depart", "depart", "info", 3, "gen"),
    ("scalar_arrive", "arrive", "info", 3, "gen"),
    ("scalar_core_frac", "core_frac", "info", 3, "gen"),
]
FLAGS = [("near_copy", "near_copy"), ("cross_high", "cross_high"),
         ("app_saturated", "app_sat"), ("core_degenerate", "core_degen"),
         ("intruder", "intruder")]

NOVELTY_ORDER = ["seen", "unseen", "zero_shot"]
CONTENT_ORDER = ["same", "cross", "foreign"]
NOVELTY_LABEL = {"seen": "seen<br><span>held-in training sample</span>",
                 "unseen": "unseen<br><span>held-in test sample</span>",
                 "zero_shot": "zero-shot<br><span>held-out sample</span>"}
CONTENT_LABEL = {"same": "same<br><span>test sample from reference's class</span>",
                 "cross": "cross<br><span>test sample from other class</span>",
                 "foreign": "foreign<br><span>DAVIS endpoints</span>"}

#: Context tiers bracket the runs. `specialist` does not participate in the run chips — it is the
#: invariant upper yardstick, and its numbers come from a superseded artifact besides.
#:
#: 🔴 THE PAGE HAS EXACTLY TWO BASELINES, `base_prompt_ctt` and `base_cond_ctt` (owner call
#: 2026-07-31), and they are EXTERNAL entries above, so they toggle like every other arm. Three
#: things that used to stand in for a baseline are deliberately gone:
#:   * `copier` (⚠ BASE + DEMO) — a no-adapter model handed a reference copies it; that was the
#:     owner's 2026-07-23 ruling and the column carried the warning in its own label. With real
#:     baselines on the page a column that is not one, sitting where one would sit, is worse than
#:     absent, so it was REMOVED rather than relabelled.
#:   * the old `base_prompt`/`base_cond` rows in registry.jsonl — a different (ladder2) roster,
#:     never generated. The new arms deliberately carry different names; see arms.yaml.
#:   * the two roster-confounded substitutes (eps `base:SP-*` rows, exp_072's base·PE) that
#:     differed from each other by ~19pp on composition alone — more than the effect this page
#:     shows. Neither is drawn, and now neither is needed.
#:
#: `text_floor` is still absent, for a structural reason rather than a policy one: all 12 of its
#: rows carry `endpoint: None` — it is a per-DONOR-CLASS floor (prompt with no anchors at all),
#: not a per-card row, so it cannot join a (donor, endpoint, sided) card. It joined 0/139 and was
#: dropped on the advisor's pre-stated condition. Its numbers live in the ladder2 results viewer.
#: ⓪/① replace what it was reached for: they ARE the per-card prompt-only floor.
CONTEXT_TIERS_BEFORE = ["specialist"]
CONTEXT_TIERS_AFTER = [a["id"] for a in EXTERNAL]
TIER_LABEL = {
    "specialist": ["② SPECIALIST", "transition baked into the weights"],
    **{a["id"]: [a["label"].upper(), a["sub"]] for a in EXTERNAL},
}
#: tiers whose numbers come from the superseded artifact (no eps rescore exists)
BADGED_TIERS = {"specialist"}

#: The pair the "instrument" panel differences, pinned BY ID rather than by position. That panel
#: exists to price the specialists' stale223 badge against the artifact it is read against, and it
#: is a cross-BUILD quantity — inserting a new primary set at the front of SCORE_SETS must not
#: silently repoint it at a cross-MACHINE one. The machine term is stated separately (PROBE).
IDELTA_PAIR = ("stale223", "rebuilt222")


def tier_of(r: dict) -> str | None:
    arm = r["arm"]
    if arm in RUN_TIER:
        return RUN_TIER[arm]
    # The old ladder2 baseline rows and `base` (the removed BASE+DEMO copier tier) get no column:
    # this page's baselines are the two `base_*_ctt` arms, which arrive through EXTERNAL.
    if arm in ("base", "base_prompt", "base_cond", "text_floor"):
        return None                         # see the tier note above
    if arm.startswith("spec_"):
        return "specialist"
    return None


def novelty_view(r: dict) -> str:
    if r["arm"].startswith("spec_"):
        return "seen" if r["cell"] == "SP-fit" else "unseen"
    return r["ref_novelty"]


def rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def clip_video(clip: str) -> str | None:
    if prompts.is_davis(clip):
        return None
    p = STD / prompts.clip_class(clip) / f"{clip}.mp4"
    return rel(p) if p.exists() else None


def video_paths(row: dict) -> dict[str, str]:
    """Where a row's mp4 live. Baseline rows share one canonical video per (endpoint, sided) via
    video_key; a run's videos come from ITS OWN explicit gen_dir."""
    vk = row.get("video_key")
    if vk:
        d, name = vk.split("/", 1)
        base = REPO_ROOT / "outputs/videos/ladder2" / d
    else:
        run = RUN_BY_ARM.get(row["arm"])
        base = (REPO_ROOT / run["gen_dir"]) if run else REPO_ROOT / "outputs/videos/ladder2" / row["arm"]
        name = row["item_id"]
    out = {}
    for s in SEEDS:
        p = base / f"{name}__s{s}.mp4"
        if p.exists():
            out[str(s)] = rel(p)
    return out


# ------------------------------------------------------------------------ scoring provenance
def score_paths(ss: dict) -> list[Path]:
    """Every directory a score set is spread over, in preference order."""
    raw = ss.get("paths") or [ss["path"]]
    return [(REPO_ROOT / p).resolve() for p in raw]


def score_env(paths: list[Path]) -> dict:
    """The MACHINE a score set was produced on, read out of the harness's own results.json.

    Same discipline the page already applies to `certified` and the corpus hash: report what the
    artifact says, never what this file believes. It is load-bearing here — a cross-machine probe
    (misc/refvfx_baseline/probe/PROBE.md) FAILED the project's pre-registered reproduction bar, so
    two columns scored on different boxes are not automatically like-for-like."""
    for base in paths:
        for f in sorted(base.glob("*/results.json")) + sorted(base.glob("results.json")):
            e = json.loads(f.read_text()).get("provenance", {}).get("env", {}) or {}
            plat = e.get("platform") or ""
            return {"platform": plat, "python": e.get("python"),
                    "torch": (e.get("packages") or {}).get("torch"),
                    "arch": ("aarch64" if "aarch64" in plat else
                             "x86_64" if "x86_64" in plat else "?")}
    return {}


#: The measured cost of putting columns from two machines in one table. Numbers are quoted from
#: PROBE.md §"Delta table" and §"BIAS or NOISE?" — not re-derived here.
PROBE = {
    "doc": "misc/refvfx_baseline/probe/PROBE.md",
    "text": "eps and DeltaAI do <b>not</b> reproduce each other row by row on this instrument, even "
            "with an identical reference_v4 and corpus. The probe (178 rows) <b>FAILED</b> the "
            "pre-registered bar of max |Δ| &lt; 0.005 at <b>0.046</b> on app_ref (36% of rows "
            "breaching); the full-set diagnostic is <b>worse</b> — app_ref max |Δ| <b>0.185</b>, "
            "<b>54%</b> of 1,841 rows breaching. It is unbiased numerical noise rather than a "
            "calibratable offset: <b>zero</b> gate flips at τ=0.858, zero core_degenerate or tier "
            "changes, and <b>cell means reproduce the eps table to ≤0.4 pp</b>. Both halves matter "
            "— per-row disagreement is severe, aggregates are sound. So never read a single clip's "
            "number across machines, and never read a few-tenths-of-a-point cell difference as an "
            "effect.",
}

#: `core_degenerate` (and, more weakly, `copy_max`) are measured against ABSOLUTE frame counts, so
#: they are not comparable between a 121f arm and a 33f one. Marked on the page, never silently
#: averaged into a model difference.
WINDOW_CAVEAT = {
    "mark": "†",
    # driven by the arm's OWN declared clip length, never by "is it external" — ⑥ is an external
    # tier on this page but runs our 121f geometry, so it must not inherit refVFX's caveat
    "tiers": [a["id"] for a in EXTERNAL if a.get("frames") != 121],
    "flags": ["core_degenerate"],
    "metrics": ["copy_max"],
    "text": "<b>not comparable across clip lengths.</b> "
            "<span class='mono'>FALLBACK_MIN_FRAMES = 8</span> (s_structure.py:23) is an "
            "<b>absolute</b> frame count, and <span class='mono'>mid_mask</span> "
            "(m2_integrity.py:33) excludes a fixed 9-frame prefix and 8-frame suffix whatever the "
            "clip length. A refVFX clip is 33 frames, so its scored window is <b>24 frames "
            "(one-sided) or 16 (two-sided)</b> against <b>112 / 104</b> on our 121-frame arms — "
            "the same 8-frame bar is 4–7× harder to clear there. Read a raised "
            "<span class='mono'>core_degen</span> rate on the external columns as clip length, "
            "not as a model difference. <span class='mono'>copy_max</span> carries a weaker form "
            "of the same effect: it searches a 16/24-frame window instead of a 104/112-frame one.",
}

#: The second structural caveat, marked the same way as `†` and for the same reason: it must be
#: visible ON the number, not only in a paragraph. An arm with no base twin has no legitimate
#: paired Δ, so its column is a LEVEL — the page never differences it against anything.
TWIN_CAVEAT = {
    "mark": "‡",
    "tiers": [a["id"] for a in EXTERNAL if a.get("no_twin")],
    "text": "<b>no per-row base twin — a LEVEL, never a margin.</b> These arms carry a deliberately "
            "new <span class='mono'>input_key</span> (the prompt changed, so the input changed; "
            "the original hash is kept as <span class='mono'>input_key_base</span>), so none of "
            "their 152 rows resolves a <span class='mono'>twin_of</span> in the harness. For ⓪ "
            "and ① that is definitional — they <i>are</i> the baselines, and a baseline is not "
            "joined to one. For ⑥ it is by construction: the honest control is the base model "
            "handed the same effect-describing prompt, and pairing it instead against a base "
            "video rendered from the PLAIN prompt would silently attribute the prompt change to "
            "the adapter. (⓪/① are that control as a <b>column</b> — same clause, minus the "
            "trained token — which is why they are on the page; they are still not a per-row "
            "twin, because the token differs.) So these columns never enter the paired Δ, the "
            "per-card Δ badge or the donor-class sign test, all of which stay between the two "
            "trainings. <span class='mono'>M2b margin</span> is unaffected and is shown: it is "
            "app(target) − best other class, computed from the generation alone, with no twin in "
            "it. Toggling a column changes visibility only — the caveat holds either way. "
            "See <span class='mono'>misc/base_arms/README.md</span> and "
            "<span class='mono'>misc/ctt_v2_leaky/DOSSIER.md</span> "
            "§“READ BEFORE SCORING THIS ARM”.",
}


# --------------------------------------------------------------------------- external arms
def ensure_external_media() -> None:
    """Point outputs/videos/refvfx_baseline/<arm> at each arm's finished output directory.

    The sources are finished experimental data that other work reads: they are never moved,
    copied or written to. A symlink is enough — the page's paths are repo-root-relative and the
    viewer's static server follows links — and rebuilding it here means a wiped `outputs/` costs
    one rerun of this script, which is the repo's rule for anything under `outputs/`."""
    for a in EXTERNAL:
        src, link = a["src"], REPO_ROOT / a["media"]
        if not src.is_dir():
            raise SystemExit(f"external arm '{a['id']}': no video directory at {src}")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if os.readlink(link) == str(src):
                continue
            link.unlink()
        elif link.exists():
            raise SystemExit(f"external arm '{a['id']}': {a['media']} exists and is not a symlink")
        link.symlink_to(src)
        print(f"[media] {a['media']} -> {src}")


def arm_rows(a: dict) -> dict[str, dict[int, dict]]:
    """item_id -> {seed: source row} for one arm, from whichever shape its campaign wrote.

    `manifest` — refVFX's shape: one row per (item, seed), carrying its own `out_name`,
    `num_frames` and `end_image`. Read verbatim; nothing is inferred from the item_id.
    `registry`  — our shape: one row per ITEM, no seed, the clips named `<item_id>__s<seed>.mp4`
    (`video_key` on the row says the same). The per-seed rows are synthesised here so both shapes
    reach the rest of this file identically; the PROMPT still comes from the row, never rebuilt."""
    kind, src = a["rows"]
    out: dict[str, dict[int, dict]] = {}
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if kind == "manifest":
            out.setdefault(row["item_id"], {})[int(row["seed"])] = row
            continue
        for s in SEEDS:
            out.setdefault(row["item_id"], {})[s] = {
                **row, "seed": s, "out_name": f"{row['item_id']}__s{s}.mp4",
                "num_frames": a["frames"], "end_image": row["sided"] == "two"}
    return out


#: MEASURED 2026-07-30, and the reason every join below carries an arm: `ctt_v2`, `ctt_v2_leaky`,
#: `refvfx_A` and `refvfx_B` all ran the ctt_v2 registry's rows, so their eval item_ids are
#: IDENTICAL — 1,842 of 1,842, controls included. `ic_gen` pulls the other way: its item_ids embed
#: its own arm name, so a raw item_id join against any of the four returns ZERO rows. One page,
#: both traps, opposite failure modes — silent merge on one side, silent nothing on the other. The
#: only field that separates the colliding four is the harness's own `arm`, so it is asserted here
#: rather than assumed, at the point where rows are read.
def assert_arms(rows: list[dict], expect: str, where: Path) -> None:
    """Every treatment row read out of `where` must be stamped with the arm we think we are reading.

    This is the seatbelt for the collision above: point a column at the wrong directory — or add a
    colliding directory to a score set's `paths` — and the ids will join perfectly and the numbers
    will be wrong. The arm is the only thing that notices."""
    got = collections.Counter(r.get("arm") for r in rows if not str(r.get("arm", "")).startswith("control_"))
    wrong = {a: n for a, n in got.items() if a != expect}
    if wrong:
        raise SystemExit(f"[arms] {where} carries rows stamped {wrong} but is being read as "
                         f"'{expect}' — these arms share eval item_ids, so this would have merged "
                         f"silently instead of failing. Check the score directory wiring.")


def load_external_scores(path: Path, registry: dict, ceil: dict, expect_arm: str) -> tuple[dict, dict]:
    """(item_id -> seed-averaged metrics, provenance) for one external arm; ({}, {}) if unscored.

    Same shape and the same `rf.collapse` the runs use, so an external column means exactly what a
    run column means. The harness writes `<dir>/*/items.jsonl` (a flat `items.jsonl` is accepted
    too) with ids `<registry item_id>__s<seed>__ref_<pool clip>` — that is the whole contract; drop
    a scored directory in and rerun this script. An absent directory returns nothing and the arm
    renders as unscored: no placeholder, no borrowed number.

    Rows that do not name a registry item (the harness scores control_lerp / control_hold twins
    into the same file) fall out on the registry join, so no floor row is ever read as an arm's.

    `expect_arm` is not decoration. This arm's eval ids are byte-identical to ctt_v2's (and to the
    other two colliding arms'), so a mis-wired `scores` path would join perfectly and silently
    render another arm's numbers in this column. The harness's own `arm` stamp is the only thing
    that separates them, and it is checked before a single number is read."""
    if not path.is_dir():
        return {}, {}
    files = sorted(path.glob("*/items.jsonl")) + sorted(path.glob("items.jsonl"))
    seen: set[str] = set()
    per: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for f in files:
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        assert_arms(rows, expect_arm, f)
        for r in rows:
            if r["item_id"] in seen:
                continue
            seen.add(r["item_id"])
            head, _, _ref = r["item_id"].rpartition("__ref_")
            item, _, seed = head.rpartition("__s")
            if not seed.isdigit():
                continue
            per[(item, int(seed))].append(r)
    acc: dict[str, dict[str, list]] = {}
    for (item, _seed), rows in per.items():
        t = registry.get(item)
        if t is None:
            continue
        d = acc.setdefault(item, collections.defaultdict(list))
        c = rf.collapse(rows)
        for k, v in c.items():
            d[k].append(v)
        cls = t["gt_pool_class"]
        if cls in ceil and "app_ref" in c:
            d["pct"].append(c["app_ref"] / ceil[cls])
    # a column that carries numbers declares which artifact produced them — for an external arm
    # that declaration comes out of the harness's own results.json, not out of this file
    prov: dict = {}
    rj = sorted(path.glob("*/results.json")) + sorted(path.glob("results.json"))
    if rj:
        p = json.loads(rj[0].read_text()).get("provenance", {})
        prov = {"harness": p.get("harness"), "certified": bool(p.get("certified")),
                "corpus": (p.get("corpus_sha256") or "")[:8],
                "spec": (p.get("spec_sha256") or "")[:8],
                "reasons": p.get("uncertified_reasons") or [],
                "env": score_env([path])}
        prov["same_corpus_as_primary"] = prov["corpus"] == SCORE_SETS[0].get("corpus")
        pe = SCORE_SETS[0].get("env") or {}
        prov["same_machine_as_primary"] = bool(pe) and (
            prov["env"].get("arch"), prov["env"].get("torch"), prov["env"].get("python")
        ) == (pe.get("arch"), pe.get("torch"), pe.get("python"))
    return {item: {m: rf.mean_or_nan(v) for m, v in d.items()} for item, d in acc.items()}, prov


def diff_span(base: str, other: str) -> list[int]:
    """[start, end) of the stretch of `other` that differs from `base`, snapped to word edges.

    Both external arms differ from our rendered prompt in one clause, and from each other in the
    same clause — so highlighting that stretch against ONE baseline (our prompt) puts Ⓐ's and Ⓑ's
    difference in the same place on the page, which is what makes the contrast readable."""
    if base == other:
        return [0, 0]
    n, i = min(len(base), len(other)), 0
    while i < n and base[i] == other[i]:
        i += 1
    j = 0
    while j < n - i and base[len(base) - 1 - j] == other[len(other) - 1 - j]:
        j += 1
    s, e = i, len(other) - j
    while s > 0 and other[s - 1] not in " \n":
        s -= 1
    while e < len(other) and other[e] not in " \n":
        e += 1
    return [s, min(e, len(other))]


def external_gen(a: dict, r: dict, per_seed: dict, m: dict | None, ceil: dict,
                 our_prompt: str) -> dict:
    """One external arm's answer to one registry row — same entry shape as a run's."""
    vids = {}
    for s in SEEDS:
        row = per_seed.get(s)
        if row and (REPO_ROOT / a["media"] / row["out_name"]).exists():
            vids[str(s)] = f"{a['media']}/{row['out_name']}"
    row = per_seed.get(SEEDS[0]) or next(iter(per_seed.values()))
    m = m or {}
    e = {
        "id": r["item_id"], "arm": a["id"], "cell": r["cell"], "videos": vids,
        "novelty": novelty_view(r), "content": r["content"], "donor": r["donor_class"],
        "pct_type": r["pct_type"], "cond": "external", "ref": r.get("reference"),
        "mismatched_ref": bool(r.get("mismatched_reference")),
        "ceil": ceil.get(r["gt_pool_class"]), "tier": a["id"],
        "scored": bool(m), "instr": a["score_id"] if m else None, "stat": False,
        # their contract, stated rather than drawn as our prefix/suffix bar: a single anchor
        # FRAME (plus a last frame on two-sided rows) and their native 33f, duration-matched
        "cond_note": ("1st+last frame" if row.get("end_image") else "1st frame")
                     + f" + demo → {row.get('num_frames', '?')}f",
        "prompt": row["prompt"], "prompt_hi": diff_span(our_prompt, row["prompt"]),
    }
    # an arm of OURS conditions the way our runs do, so it gets the real prefix/suffix bar — the
    # "their contract" box is for a model whose geometry is not ours. Which conditioning it
    # actually got is read off the ARM'S OWN row (`row`), never off the registry row `r`: ⓪ and ①
    # share every field of `r` and differ only here, so taking it from `r` would draw the prefix
    # bar on the prompt-only baseline.
    if a.get("kind") in ("ours", "baseline"):
        e["cond"] = ("none" if row.get("conditioning") == "none"
                     else "prefix+suffix" if r["sided"] == "two" else "prefix")
        e.pop("cond_note")
    # `use_reference: false` means the demo was never given to this arm. The reference stays on the
    # row as SCORING identity (pool_refs bans it), so it must be cleared here or the baselines
    # would draw the in-context-demo ribbon and claim an input they never received.
    if row.get("use_reference") is False:
        e["ref"] = None
        e["mismatched_ref"] = False
    e["m"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 6))
              for k, _l, _d, _dp, _g in METRICS}
    e["f"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 4))
              for k, _l in FLAGS}
    e["pct"] = (None if m.get("pct") is None or m.get("pct") != m.get("pct")
                else round(m["pct"], 6))
    if e["ref"]:
        e["ref_class"] = prompts.clip_class(e["ref"])
        e["ref_video"] = clip_video(e["ref"])
    return e


def attach_external(cards: dict, registry: dict, ceil: dict) -> list[dict]:
    """Hang every external arm's clips on the cards the runs already built, and collect each
    card's per-arm prompts so the page can show them side by side with ours."""
    stats = []
    for a in EXTERNAL:
        by_item = arm_rows(a)
        metrics, prov = load_external_scores(a["scores"], registry, ceil, a["id"])
        joined = vids = scored = off_grid = same_as_ours = 0
        for item, per_seed in sorted(by_item.items()):
            r = registry.get(item)
            if r is None:
                off_grid += 1
                continue
            card = cards.get(f"{r['donor_class']}|{r['endpoint']}|{r['sided']}")
            if card is None:                      # a row of the grid no run answers — not a card
                continue
            g = external_gen(a, r, per_seed, metrics.get(item), ceil, card["prompt"])
            card["slots"][a["id"]].append(g)
            # the prompt belongs to (arm, reference): two rows can share a card with different
            # demos, and arm Ⓐ's prompt is written from the demo, so it differs between them
            card.setdefault("alt_prompts", []).append(
                {"tier": a["id"], "label": a["label"], "kind": a["prompt_kind"],
                 "ref": r.get("reference"), "text": g["prompt"], "hi": g["prompt_hi"]})
            joined += 1
            vids += len(g["videos"])
            scored += bool(g["scored"])
            same_as_ours += g["prompt"] == card["prompt"]
        stats.append({"id": a["id"], "label": a["label"], "sub": a["sub"],
                      "kind": a.get("kind", "prior-work"), "frames": a.get("frames"),
                      "no_twin": bool(a.get("no_twin")),
                      "score_id": a["score_id"], "prov": prov,
                      "prompt_kind": a["prompt_kind"], "doc": a["doc"],
                      "media": a["media"], "manifest": str(a["rows"][1].relative_to(LAB)),
                      "scores_slot": str(a["scores"].relative_to(LAB)),
                      "rows": len(by_item), "gens": joined, "videos": vids, "scored": scored,
                      "off_grid": off_grid, "same_as_ours": same_as_ours,
                      # carried through so the prompt-identity seatbelt can exempt the arms whose
                      # claim REQUIRES an identical prompt (the ⑦/⑧ bottleneck pair)
                      "same_prompt_by_design": a.get("same_prompt_by_design", False)})
    return stats


# ------------------------------------------------------------------------------- score loading
def load_all_scores() -> tuple[dict, dict]:
    """item_id -> metrics (from the most-preferred set that scored it), and item_id -> set id.

    Each set is loaded by EXPLICIT path. `report_full.SCORES` is a module constant that ignores
    $LADDER_SCORES; assigning it here is the whole reason this function can be trusted."""
    ceil = run_eval.ceilings()
    registry = {r["item_id"]: r for r in run_eval.load_registry()}
    metrics: dict[str, dict] = {}
    instr: dict[str, str] = {}
    per_set_counts: dict[str, collections.Counter] = {}

    for ss in SCORE_SETS:
        paths = score_paths(ss)
        for path in paths:
            if not path.is_dir():
                raise SystemExit(f"score set '{ss['id']}' missing: {path}")
        ss["env"] = score_env(paths)           # which MACHINE produced these numbers
        scored: dict = {}
        origin: dict[tuple, Path] = {}         # which directory supplied each (item, seed)
        for path in paths:                     # a set may live in one directory per arm
            run_eval.SCORES = path
            rf.SCORES = path                   # module-level and hardcoded — must be overridden
            assert rf.SCORES == path, "report_full.SCORES did not take the override"
            for k, v in rf.load_scored().items():
                # 🔴 the colliding-id trap. `extend` would CONCATENATE two arms' rows for the same
                # eval id and rf.collapse would average them into one number, with nothing to see.
                # ctt_v2 / ctt_v2_leaky / refvfx_* share every id, so this must be an error, not a
                # merge — the arm stamp below then says which arm actually turned up.
                if k in origin:
                    assert_arms(v, registry[k[0]]["arm"] if k[0] in registry else "?", path)
                    raise SystemExit(
                        f"[scores] set '{ss['id']}': eval id {k} is supplied by BOTH "
                        f"{origin[k]} and {path}. These directories share item_ids; merging them "
                        f"would average two arms into one column. Give the second one its own "
                        f"EXTERNAL entry instead of adding it to this set's `paths`.")
                origin[k] = path
                scored[k] = list(v)
                assert_arms(v, registry[k[0]]["arm"] if k[0] in registry else v[0].get("arm"), path)

        acc: dict[str, dict[str, list]] = {}
        counts = collections.Counter()
        for (item, _seed), rows in scored.items():
            r = registry.get(item)
            if r is None or item in metrics:   # a more-preferred set already supplied it
                continue
            d = acc.setdefault(item, collections.defaultdict(list))
            c = rf.collapse(rows)
            for k, v in c.items():
                d[k].append(v)
            cls = r["gt_pool_class"]
            if cls in ceil and "app_ref" in c:
                d["pct"].append(c["app_ref"] / ceil[cls])
        for item, d in acc.items():
            metrics[item] = {m: rf.mean_or_nan(v) for m, v in d.items()}
            instr[item] = ss["id"]
            counts[registry[item]["arm"]] += 1
        per_set_counts[ss["id"]] = counts
        ss["arms"] = sorted(counts)            # which columns actually took numbers from this set

    print("\n[scores] which instrument supplied each arm (first set wins):")
    for ss in SCORE_SETS:
        c = per_set_counts[ss["id"]]
        if not c:
            print(f"   {ss['id']:11s} —")
            continue
        print(f"   {ss['id']:11s} " + "  ".join(f"{a}={n}" for a, n in sorted(c.items())))
    return metrics, instr


#: the two degenerate references every transition must beat, as scored pseudo-arms
FLOORS = {"control_lerp": ("crossfade", "a linear dissolve between this card's own endpoints"),
          "control_hold": ("freeze", "hold the first frame — no motion at all")}
#: shown under the floor line so nobody equates these with POOL_YARDSTICK's headline
FLOOR_NOTE = ("Recomputed from this campaign's own ladder2 control rows, where each control is "
              "built from the card's own endpoints. Not comparable to POOL_YARDSTICK's 48% / 22%, "
              "which are exp_072's separately-constructed control arms (that lane re-aggregates "
              "to 43.6% / 18.9%). Roster is not the difference — the lane is. See RUN_RECORD §20.")


def control_floors(registry: dict) -> dict:
    """Recompute the crossfade / freeze floors FROM THIS PAGE'S OWN LANE.

    POOL_YARDSTICK.md's headline 48% / 22% are NOT quoted, and the reason is stronger than a
    roster difference — it is a DIFFERENT LANE. Those figures come from exp_072
    (outputs/eval/exp_072_pool_v4/), which builds its own control_lerp/control_hold arms over its
    own pairing; re-aggregating that lane reproduces them (43.6% / 18.9%). ladder2's control_lerp
    is instead constructed per-card from that card's own endpoints, and gives 30.1% / 17.4% here.
    (These follow SCORE_SETS[0]. Moving the primary set from the eps rescore to the DeltaAI one on
    2026-07-30 moved them 30.33 → 30.13 and 17.46 → 17.45, on the IDENTICAL control rows — n=160
    and n=448 either way — so the shift is the machine term PROBE.md priced at 0.1–0.4 pp on an
    aggregate, and nothing about the roster changed.)
    Both are correct for their own lane; they are different quantities and must never be quoted
    against each other. Checked before concluding: roster does NOT explain the gap — the same eps
    control rows on the SP-* roster give 31.7% / 21.3%, and no roster × score-set combination
    reaches 48% (max 32.6%). See RUN_RECORD.md §20.

    A control is derived from the INPUT clips, so the same control content is scored once against a
    treatment row and again against its base twin — deduped here on (kind, item, seed) via the
    accumulator key, with `base:`-prefixed cells dropped so a floor is never counted twice."""
    ceil = run_eval.ceilings()
    acc: dict[str, dict[tuple, list]] = {k: {} for k in FLOORS}
    # floors are primary-instrument only, so they follow the primary set wherever it lives
    for f in sorted(g for p in score_paths(SCORE_SETS[0]) for g in p.glob("*/items.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            kind = r.get("arm", "")
            if kind not in FLOORS or "app_ref" not in r:
                continue
            head, _, _ref = r["item_id"].split("__", 1)[1].rpartition("__ref_")
            item, _, seed = head.rpartition("__s")
            t = registry.get(item)
            if t is None or t["cell"].startswith("base:") or t["gt_pool_class"] not in ceil:
                continue
            acc[kind].setdefault((item, seed, t["cell"]), []).append(
                r["app_ref"] / ceil[t["gt_pool_class"]])
    out = {}
    for kind, (label, blurb) in FLOORS.items():
        per: dict[str, list] = collections.defaultdict(list)
        for (_i, _s, cell), vs in acc[kind].items():
            per[cell].append(st.mean(vs))
        allv = [v for vs in per.values() for v in vs]
        if not allv:
            continue
        out[kind] = {
            "label": label, "blurb": blurb,
            "global": round(st.mean(allv) * 100, 2), "n": len(allv),
            # n>=8 per the advisor's rule: below that a per-cell floor is noise dressed as a line
            "cells": {c: {"pct": round(st.mean(v) * 100, 2), "n": len(v)}
                      for c, v in sorted(per.items()) if len(v) >= 8},
        }
    return out


def instrument_delta(registry: dict) -> dict:
    """ic_gen is scored in BOTH sets — the SAME video files. Their difference IS the cost of
    showing a stale-instrument number next to a rebuilt-instrument one, so the page states it
    instead of asserting the badge is harmless."""
    ceil = run_eval.ceilings()
    per: dict[str, dict] = {}
    for ss in SCORE_SETS:
        out = {}
        for path in score_paths(ss):
            run_eval.SCORES = path
            rf.SCORES = path
            for (item, seed), rows in rf.load_scored().items():
                r = registry.get(item)
                if r is None or r["arm"] != "ic_gen":
                    continue
                c = rf.collapse(rows)
                cls = r["gt_pool_class"]
                if cls in ceil and "app_ref" in c:
                    out[(item, seed)] = c["app_ref"] / ceil[cls]
        per[ss["id"]] = out
    a, b = per[IDELTA_PAIR[0]], per[IDELTA_PAIR[1]]               # stale, rebuilt — by id, not index
    shared = [k for k in a if k in b]
    if not shared:
        raise SystemExit(f"[instrument] {IDELTA_PAIR[0]} × {IDELTA_PAIR[1]} joined ZERO ic_gen "
                         f"generations — the panel would silently report nothing")
    by_cell: dict[str, list] = collections.defaultdict(list)
    for k in shared:
        by_cell[registry[k[0]]["cell"]].append((b[k] - a[k]) * 100)
    cells = {c: st.mean(v) for c, v in by_cell.items()}
    pg = [(b[k] - a[k]) * 100 for k in shared]
    return {"n": len(shared), "pair": list(IDELTA_PAIR),
            "cell_mean_abs": round(st.mean([abs(v) for v in cells.values()]), 3),
            "cell_max_abs": round(max(abs(v) for v in cells.values()), 3),
            "gen_mean_abs": round(st.mean([abs(v) for v in pg]), 3),
            "gen_max_abs": round(max(abs(v) for v in pg), 3),
            "gen_over_1pp": sum(1 for v in pg if abs(v) > 1)}


# ------------------------------------------------------------------------------------- build
def build() -> dict:
    ensure_external_media()
    run_eval.EXTRA_REGISTRY = None
    extras = [r["registry"] for r in RUNS if r["registry"]]
    if extras:
        # load_registry() concatenates ONE extra file; more than one would need a loop upstream.
        assert len(extras) == 1, f"only one extra registry is wired: {extras}"
        run_eval.EXTRA_REGISTRY = REPO_ROOT / extras[0]
        assert run_eval.EXTRA_REGISTRY.exists(), f"missing {run_eval.EXTRA_REGISTRY}"

    rows = run_eval.load_registry()
    registry = {r["item_id"]: r for r in rows}
    ceil = run_eval.ceilings()
    metrics, instr = load_all_scores()
    idelta = instrument_delta(registry)
    floors = control_floors(registry)

    def gen_entry(r: dict) -> dict | None:
        vids = video_paths(r)
        m = metrics.get(r["item_id"])
        if m is None and not vids:
            return None
        m = m or {}
        cond = ("none" if r.get("conditioning") == "none" or not r["endpoint"]
                else "prefix+suffix" if r["sided"] == "two" else "prefix")
        iid = instr.get(r["item_id"])
        primary = SCORE_SETS[0]["id"]
        e = {
            "id": r["item_id"], "arm": r["arm"], "cell": r["cell"], "videos": vids,
            "novelty": novelty_view(r), "content": r["content"], "donor": r["donor_class"],
            "pct_type": r["pct_type"], "cond": cond, "ref": r.get("reference"),
            "mismatched_ref": bool(r.get("mismatched_reference")),
            "ceil": ceil.get(r["gt_pool_class"]), "tier": tier_of(r),
            "scored": bool(m), "instr": iid,
            # a gen only feeds the recomputed statistics if its instrument is the primary one —
            # this is what keeps every aggregate on the page single-instrument
            "stat": bool(m) and (iid == primary or SECONDARY_IN_STATS),
        }
        e["m"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 6))
                  for k, _l, _d, _dp, _g in METRICS}
        e["f"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 4))
                  for k, _l in FLAGS}
        e["pct"] = (None if m.get("pct") is None or m.get("pct") != m.get("pct")
                    else round(m["pct"], 6))
        return e

    all_tiers = CONTEXT_TIERS_BEFORE + [RUN_TIER[r["arm"]] for r in RUNS] + CONTEXT_TIERS_AFTER
    carded = set(all_tiers)

    def new_card(r: dict) -> dict:
        ep, sided = r["endpoint"], r["sided"]
        paths = ec.cond_paths(ep, sided)
        return {
            "key": f"{r['donor_class']}|{ep}|{sided}",
            "donor": r["donor_class"], "endpoint": ep, "sided": sided, "prompt": r["prompt"],
            "endpoint_class": r.get("endpoint_class"), "endpoint_split": r.get("endpoint_split"),
            "prefix_video": rel(paths["prefix"]),
            "suffix_video": rel(paths["suffix"]) if sided == "two" else None,
            "endpoint_video": clip_video(ep),
            "slots": {t: [] for t in all_tiers},
        }

    cards: dict[str, dict] = {}
    for r in rows:
        if tier_of(r) not in carded:
            continue
        g = gen_entry(r)
        if g is None:
            continue
        key = f"{r['donor_class']}|{r['endpoint']}|{r['sided']}"
        card = cards.get(key) or cards.setdefault(key, new_card(r))
        if r.get("reference"):
            g["ref_class"] = prompts.clip_class(r["reference"])
            g["ref_video"] = clip_video(r["reference"])
        card["slots"][g["tier"]].append(g)

    # a card exists only if at least one RUN answers it — this page is about the trainings
    run_tiers = [RUN_TIER[r["arm"]] for r in RUNS]
    cards = {k: c for k, c in cards.items() if any(c["slots"][t] for t in run_tiers)}

    # the external baselines hang on the cards that already exist; they never make one
    ext = attach_external(cards, registry, ceil)
    ext_tiers = [a["id"] for a in EXTERNAL]

    # INPUTS band owns every input; output boxes only INDICATE what they received
    for card in cards.values():
        refs: dict[str, dict] = {}
        for slot in run_tiers + ext_tiers:
            for g in card["slots"][slot]:
                if not g.get("ref"):
                    continue
                e = refs.setdefault(g["ref"], {
                    "clip": g["ref"], "cls": g.get("ref_class"), "video": g.get("ref_video"),
                    "mismatched": g["mismatched_ref"], "tiers": []})
                if slot not in e["tiers"]:
                    e["tiers"].append(slot)
        card["refs"] = sorted(refs.values(), key=lambda e: e["clip"])
        # per-arm prompts, deduped on (arm, text): two registry rows can share a card, and arm Ⓐ's
        # prompt is written from the demo, so those two rows do not always agree
        seen_p: set[tuple] = set()
        card["alt_prompts"] = [p for p in card.get("alt_prompts", [])
                               if not (p["tier"], p["text"]) in seen_p
                               and not seen_p.add((p["tier"], p["text"]))]

    # per-card head-to-head: the LAST run minus the FIRST, in pool-% points
    a_t, b_t = run_tiers[0], run_tiers[-1]
    for card in cards.values():
        a = [g["pct"] for g in card["slots"][a_t] if g.get("pct") is not None]
        b = [g["pct"] for g in card["slots"][b_t] if g.get("pct") is not None]
        card["delta"] = round((st.mean(b) - st.mean(a)) * 100, 2) if a and b else None

    nrank = {n: i for i, n in enumerate(reversed(NOVELTY_ORDER))}
    crank = {c: i for i, c in enumerate(reversed(CONTENT_ORDER))}

    def card_sort(c):
        t = [g for tt in run_tiers for g in c["slots"][tt]]
        return (min((nrank.get(g["novelty"], 9) for g in t), default=9),
                min((crank.get(g["content"], 9) for g in t), default=9), c["key"])
    ordered = sorted(cards.values(), key=card_sort)

    treatments = [g for c in ordered for t in run_tiers for g in c["slots"][t]]
    matrix = collections.Counter((g["novelty"], g["content"]) for g in treatments)

    n_spec = sum(1 for c in ordered if c["slots"]["specialist"])
    n_vid = sum(len(g["videos"]) for c in ordered for s in c["slots"].values() for g in s)

    tier_label = dict(TIER_LABEL)
    for i, r in enumerate(RUNS):
        tier_label[RUN_TIER[r["arm"]]] = [f"{'④⑤'[i]} {r['label'].upper()}", r["sub"]]

    #: Owner request 2026-07-30: EVERY arm toggles, not just the trainings, so any subset can be
    #: put side by side. One list, in column order — the chips are built from it, and so is the
    #: set of columns a toggle may hide. `specialist` stays out: it is the invariant
    #: yardstick the page brackets everything with, not arms under comparison. A toggle controls
    #: VISIBILITY only; `run` is what decides whether an arm may enter the paired Δ / sign test,
    #: and `no_twin` / the 33f `†` hold whether the column is shown or not.
    arm_chips = ([{"tier": RUN_TIER[r["arm"]], "label": f"{'④⑤'[i]} {r['label']}",
                   "sub": r["sub"], "run": True, "no_twin": False}
                  for i, r in enumerate(RUNS)]
                 + [{"tier": a["id"], "label": a["label"], "sub": a["sub"], "run": False,
                     "no_twin": bool(a.get("no_twin"))} for a in EXTERNAL])

    # which MACHINE produced each column's numbers, grouped — the pool-% table puts these side by
    # side, and PROBE.md says a cross-machine comparison is not free
    def set_who(ss: dict) -> str:
        arms = ss.get("arms") or []
        shown = ", ".join(arms[:3]) + (f" +{len(arms) - 3} more" if len(arms) > 3 else "")
        return f"{ss['id']} ({shown or 'no arms'})"

    by_machine: dict[tuple, dict] = {}
    for who, e in ([(set_who(ss), ss.get("env")) for ss in SCORE_SETS]
                   + [(a["id"], (a.get("prov") or {}).get("env")) for a in ext]):
        if not e:
            continue
        m = by_machine.setdefault((e["arch"], e["torch"], e["python"]),
                                  {**e, "cols": []})
        m["cols"].append(who)
    pe = SCORE_SETS[0].get("env") or {}
    pkey = (pe.get("arch"), pe.get("torch"), pe.get("python"))
    for k, m in by_machine.items():
        m["primary"] = k == pkey
    machines = sorted(by_machine.values(), key=lambda m: not m["primary"])

    return {
        "meta": {
            "title": "IC-LoRA trainings — results",
            "design_version": (LADDER / "VERSION").read_text().strip(),
            "instrument": "transition_eval 4.0.0 (m1a_S3)",
            "score_sets": SCORE_SETS,
            "primary_set": SCORE_SETS[0]["id"],
            "instrument_delta": idelta,
            "floors": floors, "floor_note": FLOOR_NOTE,
            "badged_tiers": sorted(BADGED_TIERS),
            # which score set the badged tiers actually took their numbers from — derived, because
            # "the non-primary one" stopped being unique the moment a third set appeared
            "badged_sets": sorted({g["instr"] for c in ordered for t in BADGED_TIERS
                                   for g in c["slots"].get(t, []) if g.get("instr")}),
            "no_baseline_note":
                "This page now has a same-roster, same-geometry baseline: ⓪ BASE · prompt only "
                "and ① BASE · prompt + endpoints, generated 2026-07-31 on these exact 152 rows at "
                "seeds 42/43 and scored on the same machine as the run columns. They replace the "
                "three things that used to stand in for one — the BASE+DEMO copier column (a "
                "copier, not a baseline), the ungenerated ladder2 base_prompt/base_cond rows, and "
                "the two roster-confounded candidates that differed from each other by ~19pp on "
                "composition alone. They are read as levels; the paired Δ and the sign test "
                "remain between the two trainings.",
            "registry_rows": len(rows), "generations": n_vid, "cards": len(ordered),
            "seeds": list(SEEDS), "px_prefix": ec.PX_PREFIX,
            "suffix_gen_frames": ec.SUFFIX_GEN_FRAMES, "frames": 121,
            "tiers": all_tiers, "run_tiers": run_tiers,
            "arm_tiers": [c["tier"] for c in arm_chips], "arm_chips": arm_chips,
            "context_before": CONTEXT_TIERS_BEFORE, "context_after": CONTEXT_TIERS_AFTER,
            "spec_cards": n_spec,
            "machines": machines, "probe": PROBE, "window_caveat": WINDOW_CAVEAT,
            "twin_caveat": TWIN_CAVEAT,
            "external": ext, "external_tiers": ext_tiers,
            "arm_kinds": [{"id": k, "title": t, "note": n} for k, t, n in ARM_KINDS
                          if any(a["kind"] == k for a in ext)],
            "record": "misc/ctt_v2_training/RUN_RECORD.md §19",
            "absent_tiers": [
                ["base + demo (the old ⚠ copier column)",
                 "removed 2026-07-31 — a no-adapter model handed a reference copies it, so it was "
                 "never a baseline; ⓪/① are, and a non-baseline sitting in a baseline's place is "
                 "worse than an absent one"],
                ["text floor (prompt only)",
                 "all 12 rows have no endpoint — a per-donor-class floor, not a per-card row, so "
                 "it joins 0 of these cards. It is in the ladder2 results viewer."],
            ],
        },
        "runs": [{"id": r["id"], "arm": r["arm"], "tier": RUN_TIER[r["arm"]],
                  "label": r["label"], "sub": r["sub"], "checkpoint": r["checkpoint"],
                  "gens": sum(1 for g in treatments if g["arm"] == r["arm"])} for r in RUNS],
        "tier_label": tier_label,
        "metrics": [{"k": k, "label": l, "dir": d, "dp": dp, "group": g}
                    for k, l, d, dp, g in METRICS],
        "flags": [{"k": k, "label": l} for k, l in FLAGS],
        "novelty_order": NOVELTY_ORDER, "content_order": CONTENT_ORDER,
        "novelty_label": NOVELTY_LABEL, "content_label": CONTENT_LABEL,
        "matrix": {f"{n}|{c}": matrix.get((n, c), 0) for n in NOVELTY_ORDER for c in CONTENT_ORDER},
        "ceilings": {k: round(v, 6) for k, v in ceil.items()},
        "verdict": VERDICT,
        "cards": ordered,
    }


#: RUN_RECORD.md §19, the advisor's own three-register wording. On the page verbatim: a page that
#: showed only the favourable pool-% table would misrepresent this run, and one that led with
#: "GATE: FAIL" would misrepresent it the other way. Both registers, at once.
VERDICT = {
    "headline": "a null-to-modest result — not a failure of the adapter, not a success of the method",
    "registers": [
        {"k": "GATE", "v": "FAIL as written",
         "d": "clause (a) at the run-local fallback τ=0.2134. <b>Arm-invariant</b> (base 86.5/97.5, "
              "ic_gen 59.6/67.5, ctt_v2 67.3/77.5) and diagnosed as a fallback-calibration defect, "
              "not a property of the checkpoint. At the certified τ=0.858 <b>ctt_v2's copy rate is "
              "0.0%</b> and its maximum copy_max is 0.648 — nothing approaches the threshold. "
              "Clause (c) passed; clause (b) skipped by rule (info: +0.211)."},
        {"k": "PRIMARY CLAIM", "v": "NOT MET, direction consistent",
         "d": "+0.054 on G-unseen-cross against a committed +0.10, with all four cross/foreign "
              "cells positive at half to a fifth of the bar. Same-ontology no-regression holds. "
              "G-ref-control regressed −0.071 and is reported as a cost."},
        {"k": "MECHANISM", "v": "UNRESOLVED",
         "d": "The leak index is unmoved (G-unseen-cross +0.392 → +0.378). The training improved "
              "cross-ontology margin <b>without</b> reducing the reference-appearance leakage the "
              "campaign targeted."},
    ],
    "guards": [
        "“all arms fail the copy gate” must <b>not</b> be read as “the models copy” — at the "
        "calibrated τ=0.858 nobody copies, and the inline canary independently showed the adapter "
        "stopped reproducing demo content by step 1,000 (output-vs-demo MAD 0.08 → 0.42, flat to 10k).",
        "the +0.05 margins must <b>not</b> be sold as the predicted effect; they are half the "
        "committed bar.",
    ],
}


def check(data: dict) -> None:
    """Loud joins. Every failure mode this campaign actually hit exits 0 while producing nothing,
    so the build refuses to emit a page it cannot vouch for."""
    m, runs, cards = data["meta"], data["runs"], data["cards"]
    print("\n[join] run × card × generation")
    print(f"   {'run':10s} {'gens':>6s} {'cards':>7s} {'scored':>7s} {'in-stats':>9s}")
    bad = []
    for r in runs:
        t = r["tier"]
        gs = [g for c in cards for g in c["slots"][t]]
        nc = sum(1 for c in cards if c["slots"][t])
        ns = sum(1 for g in gs if g["scored"])
        nst = sum(1 for g in gs if g["stat"])
        print(f"   {r['id']:10s} {len(gs):6d} {nc:7d} {ns:7d} {nst:9d}")
        if not gs:
            bad.append(f"run '{r['id']}' produced ZERO generations")
        if ns != len(gs):
            bad.append(f"run '{r['id']}': {len(gs) - ns} of {len(gs)} generations are unscored")
    # ── the join trap ──────────────────────────────────────────────────────────────────────────
    # A registry item_id EMBEDS the arm (`G-fit__ic_gen__…` vs `G-fit__ctt_v2__…`), so joining two
    # runs on item_id returns ZERO rows and exits 0 — the exact silent failure this build refuses
    # to ship. The cards are keyed arm-free (donor|endpoint|sided); this asserts that the rows the
    # paired Δ actually differences agree on the arm-free key (cell, endpoint, reference, sided),
    # and that the paired set is non-empty.
    if len(runs) > 1:
        a_t, b_t = runs[0]["tier"], runs[-1]["tier"]
        raw_overlap = len({g["id"] for c in cards for g in c["slots"][a_t]}
                          & {g["id"] for c in cards for g in c["slots"][b_t]})
        paired, mismatched = 0, 0
        for c in cards:
            ka = {(g["cell"], c["endpoint"], g["ref"], c["sided"]) for g in c["slots"][a_t]}
            kb = {(g["cell"], c["endpoint"], g["ref"], c["sided"]) for g in c["slots"][b_t]}
            if not ka or not kb:
                continue
            paired += 1
            mismatched += ka != kb
        print(f"\n[join-key] {a_t} × {b_t}: paired on the arm-free key for {paired}/{len(cards)} "
              f"cards, {mismatched} disagree · raw item_id overlap {raw_overlap} "
              f"(0 is expected — item_ids embed the arm, which is why they are never joined)")
        if not paired:
            bad.append(f"paired Δ joined ZERO cards between {a_t} and {b_t}")
        if mismatched:
            bad.append(f"{mismatched} cards pair rows that differ on (cell, endpoint, reference, "
                       f"sided) — the paired Δ would compare unlike inputs")
    # the 1:1 grid is the reason a card can hold two runs at once — verify it, don't assume it
    sets = {r["tier"]: {c["key"] for c in cards if c["slots"][r["tier"]]} for r in runs}
    keys = list(sets)
    for i in range(len(keys) - 1):
        a, b = sets[keys[i]], sets[keys[i + 1]]
        if a != b:
            bad.append(f"cards for {keys[i]} and {keys[i+1]} are not 1:1 "
                       f"(+{len(b - a)} / −{len(a - b)})")
    for t in m["context_before"] + m["context_after"]:
        gs = [g for c in cards for g in c["slots"][t]]
        nc = sum(1 for c in cards if c["slots"][t])
        badge = "  ← badged (stale223)" if t in m["badged_tiers"] else ""
        for a in m["external"]:
            if a["id"] == t:
                badge = f"  ← {a['kind']} arm, own prompt (context tier)"
        print(f"   {t:10s} {len(gs):6d} {nc:7d} {sum(1 for g in gs if g['scored']):7d} "
              f"{sum(1 for g in gs if g['stat']):9d}{badge}")
        # the owner asked to SEE the specialists; an empty specialist join is the failure that
        # would look like a working page. Say the count out loud either way.
        if t == "specialist" and nc == 0:
            bad.append("specialist tier joined ZERO cards — the owner asked to see these")
    if not cards:
        bad.append("ZERO cards")
    # An external arm with no clips looks exactly like a working page, so say it out loud. Being
    # UNSCORED is not a failure — the scoring runs separately and lands later.
    print("\n[arms] the columns that bring their own prompt — videos now, numbers when scoring lands")
    for a in m["external"]:
        p = a.get("prov") or {}
        state = (f"scored {a['scored']}/{a['gens']} · {p.get('harness')} · corpus {p.get('corpus')}"
                 f"{' (same as the runs)' if p.get('same_corpus_as_primary') else ' ⚠ DIFFERENT CORPUS'}"
                 f"{'' if p.get('certified') else ' · uncertified run'}" if a["scored"]
                 else f"UNSCORED — drop harness output in $LAB/{a['scores_slot']}/ and rebuild")
        marks = ("†" if a["id"] in (m["window_caveat"]["tiers"] or []) else " ") + \
                ("‡" if a["id"] in (m["twin_caveat"]["tiers"] or []) else " ")
        print(f"   {a['id']:13s}{marks} {a['kind']:10s} {a['frames']}f · {a['gens']:4d} gens · "
              f"{a['videos']:4d} videos · {a['rows']} source rows · {state}")
        if a["scored"] and a["scored"] != a["gens"]:
            print(f"              ⚠ {a['gens'] - a['scored']} of {a['gens']} rows have no score")
        if not a["gens"]:
            bad.append(f"arm '{a['id']}' joined ZERO cards — check item_id against the registry")
        if a["videos"] != a["gens"] * len(SEEDS):
            bad.append(f"arm '{a['id']}': {a['gens'] * len(SEEDS) - a['videos']} of "
                       f"{a['gens'] * len(SEEDS)} mp4 are missing on disk")
        if a["off_grid"]:
            bad.append(f"arm '{a['id']}': {a['off_grid']} source rows are not in the registry")
        # the whole reason these arms carry their own prompt is that it DIFFERS from ours. A row
        # whose prompt equals the card's is either the wrong source file or a silently plain re-run
        # — EXCEPT for arms that declare `same_prompt_by_design`. The bottleneck pair (⑦/⑧) holds
        # the prompt byte-identical to ctt_v2 on purpose: its whole claim is that the ONLY change is
        # the content of the reference channel, so an identical prompt is the controlled contrast,
        # not a mis-wire. Those two are still separated from ctt_v2 and from each other by the
        # clip-identity check above (measured: 0 shared clips across all 152 rows).
        if a["same_as_ours"] and not a.get("same_prompt_by_design"):
            bad.append(f"arm '{a['id']}': {a['same_as_ours']} of {a['gens']} rows carry a prompt "
                       f"IDENTICAL to ours — that arm is not the arm it claims to be")
    print(f"   † absolute-frame caveat on {m['window_caveat']['tiers'] or '—'} · "
          f"‡ no-base-twin (level, never a margin) on {m['twin_caveat']['tiers'] or '—'} "
          f"— both hold whether or not the column is toggled on")
    # ── the OTHER join trap: colliding ids ────────────────────────────────────────────────────
    # The mirror image of the one above. `ic_gen` embeds its arm in every item_id, so joining it to
    # anything on item_id gives ZERO rows. `ctt_v2`, `ctt_v2_leaky`, `refvfx_A` and `refvfx_B` all
    # ran the ctt_v2 registry, so their item_ids are IDENTICAL and a join on item_id SILENTLY
    # MERGES. Both failure modes exit 0 and produce a page. So: assert that (arm, item_id) is
    # unique across everything rendered — i.e. no column absorbed another — and prove the colliding
    # columns are still carrying their own clips and their own numbers, rather than one arm's twice.
    pairs = [(g["arm"], g["id"]) for c in cards for s in c["slots"].values() for g in s]
    per_arm = collections.Counter(a for a, _ in pairs)
    if len(set(pairs)) != len(pairs):
        dup = [k for k, n in collections.Counter(pairs).items() if n > 1]
        bad.append(f"{len(dup)} (arm, item_id) pairs are rendered more than once, e.g. {dup[:3]} — "
                   f"two sources merged into one column")
    print(f"\n[ids] (arm, item_id) pairs rendered: {len(pairs)} rows over {len(per_arm)} arms, "
          f"{len(set(pairs))} distinct — {'MATCH' if len(set(pairs)) == len(pairs) else 'COLLISION'}")
    by_arm: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for c in cards:
        for s in c["slots"].values():
            for g in s:
                by_arm[g["arm"]][g["id"]] = g
    arms = sorted(by_arm)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            shared = set(by_arm[a]) & set(by_arm[b])
            if not shared:
                continue
            same_vid = sum(1 for k in shared
                           if set(by_arm[a][k]["videos"].values()) & set(by_arm[b][k]["videos"].values()))
            same_num = sum(1 for k in shared
                           if by_arm[a][k]["scored"] and by_arm[b][k]["scored"]
                           and by_arm[a][k]["m"] == by_arm[b][k]["m"])
            print(f"      {a:14s} × {b:14s} share {len(shared):3d} item_ids · "
                  f"{same_vid} share a clip · {same_num} share an identical metric vector")
            if same_vid:
                bad.append(f"'{a}' and '{b}' share {same_vid} CLIPS on colliding item_ids — one "
                           f"column is rendering the other's videos")
            if same_num and same_num == len(shared):
                bad.append(f"'{a}' and '{b}' have identical metrics on all {len(shared)} colliding "
                           f"item_ids — the two columns were merged")
    # every column that carries numbers must declare which artifact produced them, or a future run
    # silently inherits 'eps' without saying so
    undeclared = {g["arm"] for c in cards for s in c["slots"].values() for g in s
                  if g["scored"] and not g.get("instr")}
    if undeclared:
        bad.append(f"arms with numbers but no instrument tag: {sorted(undeclared)}")
    if bad:
        for b in bad:
            print(f"   ✗ {b}")
        raise SystemExit("[join] refusing to emit — fix the wiring above")
    nspec = sum(1 for c in cards if c["slots"]["specialist"])
    print(f"   ✓ every run scored, card sets 1:1, every scored arm declares its instrument")
    print(f"   ✓ specialists join {nspec}/{len(cards)} cards "
          f"({len(cards) - nspec} without — zero-shot donors have no specialist by design)")
    print("\n[machine] which box produced each column's numbers (from results.json, not asserted)")
    for mm in m["machines"]:
        print(f"   {mm['arch']:8s} torch {str(mm['torch']):12s} py {str(mm['python']):8s} "
              f"← {', '.join(mm['cols'])}")
    prim = [mm for mm in m["machines"] if mm["primary"]]
    ext_off = [a["id"] for a in m["external"]
               if a["scored"] and not (a.get("prov") or {}).get("same_machine_as_primary")]
    if prim and not ext_off:
        print("   ✓ runs and external arms are on ONE machine — the comparison this page exists "
              "for is single-machine")
    if len(m["machines"]) > 1:
        others = [c for mm in m["machines"] if not mm["primary"] for c in mm["cols"]]
        print(f"   ⚠ {len(m['machines']) - 1} other machine(s) still supply context columns "
              f"({', '.join(others)}) — stated on the page by the pool-% table")
    f = m.get("floors") or {}
    for k, v in f.items():
        print(f"   floor {v['label']:10s} {v['global']:5.1f}%  n={v['n']:4d}  "
              f"per-cell where n≥8: {len(v['cells'])} cells")


def emit(data: dict, out: Path) -> None:
    depth = len(out.parent.relative_to(REPO_ROOT).parts)
    data["meta"]["rel"] = "../" * depth
    tpl = (HERE / "template_runs.html").read_text()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reports/iclora_runs/index.html")
    args = ap.parse_args()
    data = build()
    check(data)
    out = REPO_ROOT / args.out
    emit(data, out)
    m = data["meta"]
    print(f"\n[viewer] {m['cards']} cards · {m['generations']} videos · "
          f"{len(m['arm_tiers'])} toggleable arms ({len(data['runs'])} runs + "
          f"{len(m['external'])} own-prompt) -> {args.out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"[viewer] serve:  python3 scripts/viewers/viewerctl.py serve   (port 8017)")


if __name__ == "__main__":
    main()
