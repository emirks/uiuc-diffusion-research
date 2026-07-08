# exp_057 — broad unseen-class eval of the IC-LoRA (14 classes, taxonomy-stratified)

**Status: COMPLETE 2026-07-08.** Frozen exp_056 adapter (step 3000), no
retraining. 51 quadruples over 14 unseen classes; harness run
`outputs/eval/exp_057/quads/run_0001` (+ `analysis.md` taxonomy tables);
viewer `outputs/eval/exp_057/viewer` (validate_bundle PASS 351/351).

## 1. What the adapter under test was trained on (recap)

- 46 clips, 10 two-sided transition classes (scene A → effect → scene B),
  standardized 480×640×121@24. **131 training pairs**, circulant within-class
  (target *i* ← refs *i+1..i+3* mod n): each pair = full reference clip
  (in-context, loss-excluded) + target endpoints (first 9 / last 8 px frames,
  clean) + **type-blind caption** ("\<scene A\>. The scene transforms into
  \<scene B\>." — no mechanism words) + trigger `ICTRANS`. Loss on the
  target's middle ~13/16 latent frames only. 3000 steps, rank 32 attn+FFN,
  lr 2e-4, one H200, 3.3 h. Everything the adapter knows about *which*
  transition to make comes from the reference video.
- Notably: **every training target was two-sided**; no vanish/in-place
  transformations, no camera-arc classes beyond flying_cam, all textures from
  the 10 trained families.

## 2. Eval corpus and filtering (user-labeled tree)

`data/processed/transitions/{onesided,twosided}_transitions/` — user's own
taxonomy in dir names: **object** (a new object forms), **camera** (camera
move is the transition), **style** (same object re-rendered), multi-tags
allowed. 36 one-sided + 13 two-sided classes, 339 clips.

Filter report (`inventory.py`, `dedup_report.md`):
- 0 exact (md5) duplicates, 0 undecodable clips;
- 17 clips <121f — removes `mouth_in` (13/16 short) and 4/12 of `eyes_in`;
- near-dups: aHash flagged `giant_grab_0≈5`, `sakura_petals_0≈1`,
  `wonderland_0≈1`; montage inspection added same-take regenerations
  `super_fast_run_1≈11` and `plasma_explosion_0≈3`. One of each pair
  **excluded from the eval corpus** so LOO ceilings aren't inflated by twins.
- 8 classes at 320px source (disintegration, head_explosion, set_on_fire,
  turning_metal×3, eyes_in, earth_zoom_out, thunder_god) **excluded**: 2.6×
  upscale would systematically depress/blur appearance similarity; vanish
  coverage comes from high-res classes instead.
- `portal` keeps 4 cartoon-domain clips in its corpus (ceiling mixes cartoon
  + live-action — noted; endpoint clips chosen live-action).

**Selected: 14 classes / 110 corpus clips** (standardized into
`transitions_std121/`, corpus now 25 styles): camera hero_flight(10),
super_fast_run(12), plasma_explosion(4); style shadow(15)+fire_element(4) as
deliberate *texture-cousins* of trained shadow_smoke/firelava, wireframe(9)+
illustration_scene(10) as *novel* textures; object animalization(8),
gas_transformation(10), portal(13), giant_grab(5), money_rain(7); two-sided
unseen hole_transition(2), seamless_transition(1).

## 3. Test design (51 generations, exp_056 recipe, seed 42)

| arm | n | construction |
|---|---|---|
| ic_os_inclass | 23 | unseen one-sided class: endpoints clip X + ref clip Y≠X (12 classes ×2; money_rain ×1) |
| ic_os_to2s | 12 | unseen one-sided ref applied to trained two-sided endpoints (ew0/melt1/wb0 rotation) |
| ic_ts_unseen | 3 | two-sided unseen refs (hole in-class; hole & seamless on ew0) — jump analogues |
| ic_anchor | 2 | exact exp_056 ic_cross items (run link) |
| base_* | 11 | no-LoRA twins on identical inputs |

24 new type-blind endpoint captions written from the standardized first/last
frames. One-sided caption caveat (pre-registered): the *end state* of a
one-sided clip partially reveals the effect's terminal appearance (a man in
flames is described as visible content) — but the endpoints are given as
pixels anyway; only vanish classes (empty last frame) are fully
appearance-blind, which makes them the purest in-context probes.

## 4. Results

### 4a. Twin pairs are the load-bearing evidence (identical inputs, ±adapter)

All 11 base twins are in the **copy regime**: leak 0.97–0.995, they visibly
replay the *reference's* scenes between the pinned endpoints, then snap
(seam z +4.8 … **+228**, the gas twin's cut to the empty café). The IC twins
on identical inputs: leak −0.05…−0.60 lower, seams near/below 0, endpoint
DINO higher on **11/11 twins** (+0.04..+0.10). The adapter, not the prompt or
conditioning, converts reference-copying into reference-*reading*.

### 4b. Taxonomy (arm × user's labels; full tables in analysis.md)

- **style** transfers appearance best: in-class raw appearance 0.75±0.08;
  on foreign endpoints (os_to2s) 0.37±0.11 with norm 0.50±0.38 —
  wireframe→melt renders the melting hay subject as a neon grid;
  illustration→water_bending comic-panels the swimmer. Style is carried by
  texture, which survives re-targeting.
- **object** works in-class (0.58±0.15 raw; the vanish trio all resolve to
  clean empty scenes) but on foreign endpoints the *formed object* often
  shrinks to a token gesture (portal appears small and swallows nothing:
  app 0.39, leak 0.46).
- **camera** is the weakest cross-target stratum (os_to2s 0.30±0.10): a
  camera arc demonstrated on one scene rarely re-executes on foreign
  endpoints — hero_flight→earth_wave produced a mild dolly, not a takeoff
  (worst cross item, app 0.21). In-class it *does* work (hero_flight_2
  reproduces ground→aerial→first-person-arm on the right subject).
- **texture familiarity gradient (pre-registered prediction (a): confirmed
  where it can be measured).** On foreign endpoints, cousins transfer at
  0.45±0.10 raw / 0.80±0.27 norm vs novel 0.30±0.07 / 0.27±0.24. exp_056's
  "cross-class transfer" numbers were partly riding trained textures; novel-
  texture transfer is real but roughly 2/3 the raw appearance and far below
  the class ceiling.
- **structure (one-sided targets, trained only on two-sided): works.** The
  in-place/vanish structure is executed (subject transforms or dissolves in
  its own scene; empty-scene suffixes honored, endpoint DINO 0.96±0.02), with
  a caveat: renderings drift toward trained textures (gas dissolution comes
  out darker/smokier than the white-vapor demo; illustration_7 detours
  through a black-shroud + fire phase — trained-prior intrusion mid-clip).
- **two-sided unseen (fair jump-analogues)**: hole in-class transfers the
  *semantics* (approach → pass-through → scene B; leak 0.69 vs base twin
  0.99) but not the donut-object's appearance (raw 0.34 vs ceiling 0.60).
  seamless (wall-wipe pan) mostly fails to reproduce the pan (raw 0.13,
  singleton class, raw-only). Replicates exp_056's jump finding: **semantics
  transfer in-context; appearance stays near the prior.**

### 4c. Anchors — run-to-run reproducibility

exp_056 items re-generated + re-scored under the new corpus: raw appearance
0.494→0.496 / 0.429→0.469, seam −0.573→−0.568 / −0.793→−0.777, suffix DINO
±0.001. Raw metrics reproduce essentially exactly; leak (+0.06) and
normalized values shift with the corpus/ceiling change — exactly why
raw+twins, not normalized means, carry the conclusions here.

## 5. Which metrics can you trust here? (critical reading)

- **Normalized appearance is BROKEN for 7/16 styles** — the pre-registered
  lerp-floor concern is confirmed *quantitatively*: for one-sided classes the
  lerp control stays inside the scene (high similarity to class core frames)
  while the LOO ceiling is LOW (each clip is a different scene/subject), so
  **floor ≥ ceiling** for animalization (−0.09), gas (−0.16), giant_grab
  (−0.48), money_rain (−0.24), shadow (−0.23), super_fast_run (−0.12),
  wireframe (−0.22). Any normalized number there is noise; analysis.md gates
  them out. Two-sided styles (hole +0.34, shadow_smoke +0.24, earth_wave
  +0.21) keep healthy gaps — the normalization scheme is *specifically* a
  two-sided-transition instrument.
- **Raw appearance is copy-confounded**: base twins score raw 0.81–0.93 *by
  copying the reference* (leak 0.99). Raw appearance is only meaningful
  jointly with leak — high app + low leak = transfer; high app + leak>0.95 =
  plagiarism. Neither alone ranks systems.
- **Leak (max over frames) is the right statistic for copy detection** (a
  single replayed segment must trip it — a mean would dilute it), but
  **in-class leak is inflated legitimately**: portal_12 hits leak 0.96 with
  zero content copied (green swirl ≈ portal corpus frames). In-class leak
  separates copy-from-transfer ONLY via the base-twin contrast; cross-class
  leak (os_to2s 0.46±0.08) is interpretable directly.
- **Max seam z is the right endpoint-fidelity alarm** (a cut is a point
  event; max finds it, mean would hide it) but it conflates *failure* snaps
  (base twins, +5..+228) with *handoff* artifacts on successful items
  (portal_11 transfers the effect yet scores +8.6 because the pinned suffix
  portal jumps in scale at the boundary). Read it with the filmstrip.
- **Camera classes**: M3 appearance is close to ill-defined (the "effect" is
  the scene itself in motion); super_fast_run's floor>ceiling and
  hero_flight's ok-gap are both driven by scene composition, not effect
  texture. M1 profile + M2 motion + the viewer carry camera-class judgments;
  M2's trust flags are all † here (the exam never saw these classes) so even
  that is advisory.
- **money_rain (degenerate endpoints)** behaved as predicted: endpoints
  nearly identical → floor 0.58 > ceiling 0.34, seam/profile scores
  uninformative — kept as a designed control, excluded from claims.
- **Ceiling trust**: fire_element/plasma (n=4), giant_grab (n=5) below or at
  min_ceiling_clips; hole (n=2) †; seamless/jump singletons raw-only. Portal
  ceiling mixes cartoon+live-action. Treat per-style normalized numbers as
  screening, not measurement.

## 6. Honest summary

One adapter trained on 131 two-sided pairs with type-blind captions
**reads transitions off a single in-context demo and re-executes them under
distribution shift it never saw**: unseen classes (14), unseen structure
(one-sided/vanish), and — with reduced fidelity — unseen textures. The
strength ordering is texture-style > object-semantics > camera-arc; content
is essentially never copied (unlike the base model, which copies ~verbatim on
all 11 twins); endpoint anchoring survives everything (0.96–0.99). The
correct quantitative instruments at this distribution shift are raw
appearance × leak, base-twin deltas, and endpoint/seam scores — the
normalization layer (lerp floor / LOO ceiling), designed for two-sided
in-distribution evals, is provably unreliable for 7/16 styles here and was
gated out rather than reported.

## 7. Artifacts

- `outputs/eval/exp_057/quads/run_0001/` — items.jsonl, report.md,
  analysis.md (taxonomy tables + gap table + twins), ceilings.json, scatter.
- `outputs/eval/exp_057/viewer/` — 51 quadruples with in-context reference
  videos, filmstrips, per-item metrics (serve `outputs/eval/` for 056+057).
- `experiments/exp_057_ic_lora_unseen_eval/` — design.md (pre-registration),
  inventory/dedup, standardizer, quads builder, captions, analysis script.
- W&B `creative-transition-transfer` run `exp057_quads`.

## Open questions

- Camera-arc cross-target failure: is it conditioning (arc conflicts with
  pinned foreign endpoints) or capacity? Try camera refs on endpoint pairs
  whose scenes permit the arc (start portrait → end aerial).
- Novel-texture gap: multi-demo (2–3 references) or a texture-diverse
  retrain (include wireframe/illustration-like classes) — does the gradient
  flatten?
- Low-res classes were excluded — a small side-run could quantify how much
  source resolution alone moves appearance metrics (calibration for future
  corpus curation).
- The 7 broken-normalization styles need a one-sided-aware floor (e.g.
  static-frame control instead of endpoint lerp) if normalized reporting is
  ever wanted there.
