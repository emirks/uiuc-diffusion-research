# exp_056 — IC-LoRA in-context transition transfer (one adapter, 10 classes, reference carries the style)

**Status: COMPLETE — training, 46-quadruple inference suite, harness scoring and
interactive viewer all done overnight 2026-07-07→08. In-context transition
transfer works: the adapter reads the transition style off the reference video
and re-applies it to foreign endpoints; the base model instead copies the
reference's content nearly verbatim.**

## What was run

- **Training** (job 9383890, H200 ccc0481, 3:21 h, 3000 steps @ ~2.6 s/step,
  peak 50.3 GB): official V2V IC-LoRA (`reference` condition, rank 32/α32,
  attention+FFN targets, lr 2e-4) composed with the exp_051 C2V endpoint
  conditions (`prefix tb=2` + `suffix tb=1`), all p=1.0. Data = 131 circulant
  same-class pairs over 46 standardized clips (480×640×121@24; 2× decimation
  for 242f clips), **type-blind captions** ("<scene A>. The scene transforms
  into <scene B>." — leak-checked; transition type observable ONLY in the
  reference), trigger `ICTRANS`. `jump_transition` (singleton) fully held out.
- **Quadruple suite**: 46 generations — arms `ic_inclass_new` (6, untrained
  same-class pairings), `ic_inclass_trained` (7), `ic_cross` (20 = each class
  as reference on 2 foreign endpoint sets), `ic_unseen` (4, jump reference),
  `ic_reverse` (1, jump endpoints + smoke ref), + 8 `base_*` twins.
  `run_ic_inference.py` = trainer's ValidationRunner + exact PEFT load,
  chunked, skip-if-exists.
- **Scoring**: exp_053-v2 harness fork (`run_score_ic.py`, singleton-ceiling
  guard) against the exp_054 47-clip validation run →
  `outputs/eval/exp_056/quads/run_0002` (41 normalizable items + 5 jump-style
  raw-only). W&B run `exp056_quads` (1tcopqle).
- **Viewer**: `build_viewer_ic.py` (exp_055 fork adding the in-context
  reference video slot per quadruple) → `outputs/eval/exp_056/viewer`
  (381/381 assets, 46/46 filmstrips; serve with `python -m http.server`).

## Results

**The base model treats in-context reference tokens as content to reproduce;
the IC-LoRA treats them as a style demonstration.** Reference tokens share the
target grid's RoPE positions, so the untrained model replays the reference
(bookstore woman → suddenly the reference's rollerblading man → its
skateboarder — then a hard snap to the pinned end anchor). Every metric axis
agrees:

| axis | base (n=7 normalizable) | ic (n=34) |
|---|---|---|
| leak max sim (≥~0.9 = near-copy) | **0.95–0.98 — copy regime** | 0.58–0.81 |
| appearance (norm, 0=lerp/1=real) | 1.00 (by copying) | 0.63–1.00 (honest) |
| max seam z (<0 = seamless) | +2.4…+7.1 (snap cuts) | **−0.47…−0.64** |
| endpoint DINO | 0.86–0.94 | **0.96–0.98** |
| depart→arrive (timing) | 0.07→0.93 (replay spans clip) | 0.30→0.63 (mid-clip transition) |

- **Cross-class transfer (the core claim, n=20)**: normalized appearance
  0.65±0.35 with leak 0.61±0.13 — the reference's morphology is imposed on
  foreign endpoints without copying its content. Visual: smoke ref on
  earth_wave endpoints → the bookstore woman herself is wrapped by the
  ink-black billow; earth-wave ref on melt endpoints → a rolling wave sweeps
  the subject, material-adapted to the scene (hay). High appearance variance =
  some style×endpoint combos transfer more faithfully than others — per-item
  inspection in the viewer.
- **Unseen class (jump, never trained)**: the generated subject crouches,
  launches airborne, lands as scene B — transition SEMANTICS read in-context
  from one demo. Quantitatively: leak 0.21–0.36 (no copy), seams negative,
  profile-DTW to the jump reference low (0.08–0.23 raw), but raw appearance to
  jump's frames only 0.15–0.29 — imitation lives in motion/timing while the
  visual rendering stays in the trained prior (smoky dust). Base twin: leak
  0.978 = verbatim replay.
- **In-class**: strongest anchors + appearance 1.00 (new pairings) — but
  in-class leak is higher (0.77–0.81), partly legitimate (generated smoke
  should resemble the smoke corpus) and partly real content leakage: the ss0
  rung shows the reference's briefcase/skateboard elements blending into
  scene-B's unpinned approach frames.
- **Endpoint anchoring + seamlessness**: IC arms hold anchors better than base
  everywhere (0.96–0.98 DINO) with uniformly negative seam z — conditioning-
  matched training buys continuity, replicating exp_051's finding under the
  IC regime.

## Mechanics worth reusing

- `reference` + `prefix` + `suffix` compose cleanly in the flexible strategy
  (intrinsic first, then concat; loss on `pred[:, -target_len:]`) AND in
  ValidationRunner samples. See memory `ltx2-ic-lora-mechanics` and
  `notes/exp/exp_050/051` for the underlying conditioning facts.
- **Pair datasets need the symlink trick**: preprocessing names every output
  after the row's TARGET path → encode unique clips once, assemble
  `latents/conditions/reference_latents` pair trees as symlinks
  (`build_dataset.py --link`); PrecomputedDataset follows them.
- **Concurrent inference chunks must not share an output dir**: the
  ValidationRunner writes `samples/step_XXXXXX_N.mp4` — four parallel chunk
  jobs raced and mangled renames (first suite run scrapped, 3/4 jobs FAILED).
  Fix: per-chunk `chunk{i}of{n}/` dirs, rename into a shared `quads/` at end.
- Harness: `run_score_ic.py` guards singleton styles (no LOO ceiling → raw
  metrics only) — the exp_054 caveat, now handled; viewer-only manifest keys
  must be stripped before `load_manifest` (EvalItem forbids extras).
- Scoring 46 items took 8:39 on an H100 thanks to the shared content-keyed
  feature cache (`outputs/eval/cache`).

## Open questions / next steps

- **Checkpoint ladder**: suite ran at step 3000 only; the 13-rung validation
  ladder suggests capability lands early — score 250/500/1000 checkpoints to
  find the sweet spot (drift vs transfer fidelity).
- **In-class content leakage**: quantify per-frame (leak argmax frames are in
  items.jsonl) — does p<1.0 reference probability or reference dropout reduce
  scene-B element bleed?
- **Unseen-class appearance**: motion transfers, appearance doesn't — is this
  a one-demo limit or a prior-strength issue? Try 2-3 in-context demos
  (multiple reference conditions) or unseen-class references closer to the
  training distribution.
- Human validation of the harness's IC readings (viewer makes this easy).
- Compare against exp_046/047 injection recipes on identical quadruples.
