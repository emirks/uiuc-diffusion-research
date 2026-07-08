# exp_057 — broad unseen-class eval of the exp_056 IC-LoRA (no retraining)

**COMPLETE 2026-07-08.** Inference 9396109/10 (base) + 9396277-81 (ic; first
round 9396104-45 failed on a relative LoRA path — sbatch `cd`s into the
trainer, pass absolute paths). Scored `outputs/eval/exp_057/quads/run_0001`
(51 items, eval job 9396609, 26 min incl. 110 fresh corpus clips); taxonomy
tables + floor-ceiling gap audit in `analysis.md`; viewer
`outputs/eval/exp_057/viewer` (PASS 351/351). **Findings: in-context transfer
survives unseen classes and one-sided/vanish structure; style > object >
camera; texture-cousins beat novel textures (0.45 vs 0.30 raw cross-target);
base twins copy on 11/11; lerp-floor normalization provably broken for 7/16
one-sided styles (floor≥ceiling) — see
`notes/exp/exp_057_ic_lora_unseen_eval.md`.**

## Question

exp_056 established in-context transition transfer but probed *unseen classes*
with a single class (jump, camera-type, n=1 clip) — an anecdote, and a
distributionally far one. Using the user-labeled corpus
(`data/processed/transitions/{onesided,twosided}_transitions/`, taxonomy in
dir names: **object** = a new object forms, **camera** = the camera move is
the transition, **style** = same object re-rendered), evaluate the SAME frozen
adapter (step 3000) on 14 unseen classes stratified by taxonomy, structure
(one-sided vs two-sided; training was 100% two-sided), and texture familiarity
(cousins of trained classes vs genuinely novel). Also: can it produce
**one-sided** targets (subject transforms in place / vanishes) although every
training target was a scene-A→scene-B clip?

## Setup

See `design.md` for full selection rationale, arms, and pre-registered
metric-validity caveats. Summary:

- Corpus filter (`inventory.py` → `dedup_report.md`): 339 labeled clips, 0
  exact dups; excluded 17 short (<121f) clips, 3 same-take regenerations
  (`giant_grab_5`, `super_fast_run_11`, `plasma_explosion_3`), and all 320px
  low-res classes.
- 110 new clips standardized (`standardize_new.py`) into
  `transitions_std121/` — extends the harness reference corpus to 25 styles.
- 51 quadruples (`make_quads.py` → `dataset/quads.json`): 23 ic_os_inclass,
  12 ic_os_to2s, 3 ic_ts_unseen, 2 ic_anchor (exp_056 repro), 11 base twins.
  Type-blind endpoint captions (`dataset/captions.json`, written from the
  standardized first/last frames); `ICTRANS` trigger; exp_056 recipe (seed 42,
  480×640×121@24, 30 steps, CFG 4, STG 1 stg_v [29], prefix 9f/suffix 8f).
- Scoring: `run_score_ic.py` (exp_056 fork) + `eval_config.yaml` with
  `transitions_root` → `transitions_std121` (raw tree was reorganized; new
  classes live in std121). `analyze_taxonomy.py` post-processes items.jsonl
  into taxonomy/texture-strata tables + normalization-reliability flags.

## How to run

```bash
cd $LAB/diffusion-research
python experiments/exp_057_ic_lora_unseen_eval/inventory.py          # corpus scan
python experiments/exp_057_ic_lora_unseen_eval/standardize_new.py    # std 110 clips
python experiments/exp_057_ic_lora_unseen_eval/make_quads.py         # quads + cond cuts
LORA=outputs/training/exp_056_ltx2_ic_lora_transition_transfer/ic/checkpoints/lora_weights_step_03000.safetensors
for i in 0 1 2 3 4; do sbatch -p secondary -A campusclusterusers --gres=gpu:H100:1 --requeue \
  experiments/exp_057_ic_lora_unseen_eval/job_infer.sbatch ic_lora $i 5 $LORA; done
for i in 0 1; do sbatch -p secondary -A campusclusterusers --gres=gpu:H100:1 --requeue \
  experiments/exp_057_ic_lora_unseen_eval/job_infer.sbatch base $i 2; done
sbatch experiments/exp_057_ic_lora_unseen_eval/job_eval.sbatch       # after inference
python experiments/exp_057_ic_lora_unseen_eval/analyze_taxonomy.py \
  --items outputs/eval/exp_057/quads/run_0001/items.jsonl
```

## Expected outcome (pre-registered)

(a) Cousin-texture styles (shadow, fire_element) transfer appearance best —
if so, exp_056's "unseen transfer" needs the texture-familiarity control.
(b) Novel styles: motion/timing imitation with prior-flavored rendering
(the exp_056 jump pattern) — wireframe/illustration are the hardest
appearance asks. (c) One-sided targets: endpoint anchoring should hold (the
mechanism is endpoint-agnostic), but the model may inject a scene change
mid-clip (trained prior) instead of transforming in place. (d) Vanish classes
(gas/portal/giant_grab): the empty-suffix endpoint is the least informative —
the purest in-context test in the suite. (e) Base twins: copy regime
(leak≥0.9) everywhere, replicating exp_056. (f) Metric validity: lerp floor
high (gap small) for one-sided classes; money_rain degenerate; camera-class
M3 ill-defined — per-style gap table decides which numbers are readable.

## Outputs

- `outputs/videos/exp_057_ic_lora_unseen_eval/{ic_lora,base}/quads/*.mp4`
- `outputs/eval/exp_057/quads/run_NNNN/` — items.jsonl, report.md,
  ceilings.json, scatter.png, analysis.md (taxonomy tables)
- `outputs/eval/exp_057/viewer/` — quadruple viewer (exp_055/056 tooling)
- W&B `creative-transition-transfer`, run `exp057_quads`
