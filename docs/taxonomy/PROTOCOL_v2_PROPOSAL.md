# Transition Taxonomy — Protocol v2 (PROPOSAL)

**Status:** rev.3 — **PASSED the fresh-context acceptance gate 2026-07-16** (independent evaluator:
zero material defects; 12/12 mandated re-derivations reproduce the table; 5 minor wording fixes
applied in this revision, none changing any assignment or stratum). rev.1 had FAILED the same gate
on two material wording defects (T1 trigger; sakura_petals flag). Remaining step: **owner sign-off**
via the §7 rulings in the v2 viewer.
**Date:** 2026-07-16. **Supersedes:** `outputs/taxonomy/PROTOCOL.md` (v1) — *descriptive layer only*.
**Instrument impact: NONE.** `sidedness` semantics, mask S, the certified harness (eval/v3.0.0), split_v1, and all training rosters are untouched by this proposal. v1→v2 conversion is mechanical given the table in §5.

**Provenance:** direct filmstrip audit of 14/39 classes; two independent fresh-context reviews
(an unanchored "architect" derivation and an adversarial stress-test that force-assigned all 39
classes under both schemas), reconciled by the operator; final gate = fresh-context evaluation
(this document is the artifact under evaluation).

---

## 1. Why v1's descriptive layer fails (evidence)

1. **`mechanism=morph` ⇔ `scene_swap=False` in 21/21 classes.** The elastic clause ("in-place
   restyle or state accumulation with no underlying cut") made morph the complement of
   scene_swap: one field masquerading as two, and a 54%-of-corpus stratum with no
   discriminative power for cross-class sign tests.
2. **Annotation noise where it hurts:** hard-call flags on `sidedness` 17/39, `mechanism` 13/39.
3. **Verified misassignments** (filmstrip): `portal` and `sakura_petals` labeled *occlusion* though
   nothing is frame-covered and nothing is revealed from behind (the subject is *removed*, the
   scene persists); `polygon` labeled *traversal* for an in-place photo→low-poly restyle (keyed
   on incidental camera drift); `plasma_explosion` carried the incoherent pair occlusion+swap=F.
4. **`dressed_cut` empty (n=0)** while its archetype (`seamless_transition`, a hidden edit behind
   a doorway walk-out/walk-in) sat inside traversal.
5. **`inserted_content` annotated contra its own definition on ≥5 classes** (cotton_cloud,
   money_rain, monstrosity, nature_bloom marked True though the content IS in the B endpoint;
   display's screen is in the A endpoint). Annotators de facto annotated "content absent from A"
   — a shadow of the missing overlay category.

## 2. Design principle (everything follows from this)

In our C2V setup the **endpoints are given as conditioning; the model synthesizes the middle.**
A descriptive field carries signal only if it describes
(a) **the operation the middle must synthesize** (the bridge / the endpoint delta), or
(b) **the evidence the conditioning already contains.**
**Apparatus (smoke, rings, petals, hands, blur) never determines classification** — it is what
the effect is *made of*, not what the effect *does*.

## 3. Schema v2 — six annotated fields + one metadata field

| # | Field | Values | One-line test | Consumer (why it earns its bit) |
|---|-------|--------|---------------|--------------------------------|
| 1 | `mechanism` | cover / transform / overlay / traverse / cut | §4 decision procedure | The pooling unit for all cross-class sign tests; each value = a distinct generative capability (§6) |
| 1a | `overlay_direction` | add / remove / state | what the overlay does to the persistent scene | descriptive sub-tag, not a stratum; free (falls out of the §4 test) |
| 2 | `scene_swap` | yes / no | first vs last frames: different shot/world, or same scene changed? | copy_max regime switch (swap=F: background copying is *partially correct*; swap=T: copying is failure); seam_z applicability; **consistency check on mechanism** (§5.1) |
| 3 | `sidedness` | A_only / B_only / two_sided | **FROZEN v1 semantics** — which endpoint's frames the effect visibly alters | mask S (certified) + prefix/suffix training recipe. Per-class relabels only, via the owner's viewer pass; the 9 open conflicts are unchanged by v2 |
| 4 | `camera_defining` | yes / no | "Replace the camera path with a locked-off tripod shot — does the effect still function as the same effect?" No → yes. In doubt → **no**, unless the class's identity IS an ego-motion/scale-reveal move | cam_dtw validity gate (compute cam_dtw only where yes) |
| 5 | `stylization` | yes / no | frame-wide appearance treatment extending beyond the effect region at any point (class-majority across exemplars) | m1a mask-validity + copy-detection gate; carves the restyle calibration subgroup (transform ∧ styl) |
| 6 | `middle_only` | yes / no | looking at the FIRST and LAST ~1s panels only: is any effect matter/treatment visible in either? None → yes | **conditioning-evidence bit**: yes ⇒ conditioning contains zero evidence of the effect ⇒ pre-registered headline split for R1−R0 suppression and R2 gains |
| 7 | `subject_anchored` | yes / no (metadata) | effect originates from / targets / tracks one endpoint entity | **metadata only** — no metric consumes it; retained for exploratory splits (e.g. R2−R3 re-binding), excluded from pre-registered claims |

Dropped from v1: **`inserted_content`** (provenance ≠ signal; its consumers are covered by
`middle_only` and `overlay_direction`). **`mechanism` v1 values** replaced wholesale.
`scene_swap` demoted from independent axis to annotate-and-verify. `subject_anchored` demoted
to metadata.

## 4. `mechanism` decision procedure (apply IN ORDER — the order is the tie-breaker)

- **TB0.** Mentally strip frame-wide treatment and motion blur *when a residual effect remains
  underneath*; a pure frame-wide restyle is judged at T1, never stripped away. Judge compounds
  at the maximal-effect (handoff) frame. Apparatus never decides.
- **T1 — transform.** Pre-existing content **undergoes visible conversion** with correspondence:
  it deforms; dissolves into matter derived from itself — *with or without* that matter
  reforming into B content (dispersal-to-absence still counts); undergoes substance
  substitution; or re-renders/restyles in place. External matter that covers, extracts, or
  deletes content **without visible conversion** is never transform. *Conversion beats coverage*
  (air_bending: subjects dissolve into the smoke which reforms as B — owner-ruled from video
  motion; gas_transformation/mystification: dissolve-and-disperse, no reformation, still T1).
- **T2 — overlay.** B is the **same scene** as A, changed only by content **added / removed /
  accrued as state** while everything surviving keeps tracker-confirmable identity.
  *Same-scene beats coverage* (sakura_petals covers only the subject; plasma's cloud persists).
  Sub-tag: `add` (content in B not A), `remove` (content in A not B), `state` (matter/energy
  accrues ON surviving content).
- **T3 — cover.** The frame is substantially blocked by effect matter/flash at the handoff, and a
  **different shot** is handed off at clearance — or inside the occluder's interior (screens,
  held portals showing B). Residue bleeding into B's first frames is allowed. Translucent media
  passed *through* during continuous ego-motion (haze, spray) are not blocking. *Covered beats
  camera* (firelava: tracked shot, but the fire wall does the work).
- **T4 — traverse.** The **camera/view itself travels** — through space or an open aperture
  (hole, doorway, stage haze) — into B's place. A subject traveling while the camera stays is
  **never** traverse. *Discriminator vs T3:* camera exits through the far side of an aperture
  into continuous space → traverse; B lives *inside* the occluder object (a picture/screen) →
  cover.
- **T5 — cut.** None of the above; a discontinuity underneath, staged/dressed.

## 5. Full 39-class assignment (v1 → v2)

Sub = overlay_direction. Swap/styl per §3 tests. mid = middle_only. Flags: ⚑ = owner ruling
needed (§7); ◆ = sidedness conflict already pending in the owner's viewer pass (unchanged).

| class | v1 mech | **v2 mech** | sub | swap | side | cam | styl | mid | behavior (one line) |
|---|---|---|---|---|---|---|---|---|---|
| air_bending | occlusion | **transform** | — | T | two | F* | F | y | subjects dissolve into cotton-smoke which reforms as B (owner-ruled); *cam: owner set y in v1 pass — re-judge under locked-off test |
| animalization | morph | **transform** | — | F | A | F | F | y | body re-renders in place into animal; transient ring |
| color_rain | morph | **overlay** | state | F | A | F | T | n | liquid drenches subject; grade persists |
| cotton_cloud | morph | **overlay** | add | F | A | F | F | n | cotton fills scene around untouched subject |
| display_transition | occlusion | **cover** ⚑ | — | T | two | T | F | ⚑ | held-up screen fills frame; B = screen interior (ratify interior clause; is the monitor in the first panel?) |
| earth_element | morph | **transform** | — | F | A ◆ | F | F | n | body cracks into rock in place |
| earth_wave | morph | **transform** ⚑ | — | F | A ◆ | F | F | y | sand wave wraps body — becomes element, or hosts it? |
| fire_element | morph | **overlay** ⚑ | state | F | A | F | F | n | fire manifests on/around posed subject — accrual or substitution? |
| firelava | occlusion | **cover** | — | T | two | T | F | y | fire wall sweeps over the silhouette (no visible conversion) during tracked shot; reveal |
| flame | occlusion | **cover** | — | T | two | F | F | y | fire → full-frame whiteout → recedes off different subject/set |
| flying_cam_transition | occlusion | **traverse** | — | T | two ◆ | T | F | y | filmstrip-verified: doorway rises from desert sand; camera walks through into living room; no covering matter at handoff |
| gas_transformation | morph | **transform** | — | F | A | F | F | y | body dissolves into gas with correspondence (conversion, not extraction) |
| giant_grab | morph | **overlay** | remove | F | two ◆ | F | F | n | inserted hand drags subject out; scene persists; hand re-enters |
| hero_flight | traversal | **traverse** | — | T | two ◆ | T | F | n | camera follows the launch into sustained flight persisting into B window (advisors read two_sided; owner rules) |
| hole_transition | occlusion | **traverse** | — | T | A ◆ | T | F | n | camera zooms *through* an open ring/aperture into B's actual space (passage, not picture) |
| illustration_scene | morph | **transform** | — | F | A | F | T | n | photoreal → flat illustration re-render in place |
| jump_transition | traversal | **traverse** | — | T | two | T | F | y | camera follows jump arc to a different place/outfit |
| live_concert | traversal | **traverse** | — | T | A ◆ | T | F | n | filmstrip-verified: backstage close-up pulls back through stage haze to festival wide (roster all-dup handled separately) |
| luminous_gaze | morph | **overlay** | state | F | A | F | T | n | eyes ignite; storm + rim-light accrue over persisting subject/scene |
| melt_transition | occlusion | **cover** | — | T | two | F | F | y | prop-derived melt covers frame; trace persists into B's first frames (residue allowed; mid=y assumes trace clears before the final ~1s window — confirm on video, else flip to n and update §6 counts) |
| money_rain | morph | **overlay** | add | F | A | F | F | n | bills fall into persistent scene |
| monstrosity | morph | **overlay** | add | F | A | F | F | n | creature grows in background; scene persists |
| mystification | morph | **transform** | — | F | A | F | F | y | subject dissolves into colored smoke (conversion) |
| nature_bloom | morph | **overlay** | add | F | A | F | F | n | flowers grow into background |
| plasma_explosion | occlusion | **overlay** | add | F | A | T* | F | n | filmstrip-verified: same intersection throughout; explosion cloud added, persists; *cam=y arguable under locked-off test — re-judge with air_bending |
| polygon | traversal | **transform** | — | F | A | F* | T | n | whole frame restyles photo → white low-poly; *cam flips to F under locked-off test |
| portal | occlusion | **overlay** | remove | F | A | F | F | y | portal opens at subject, removes them, vanishes; street persists empty |
| raven_transition | occlusion | **cover** | — | T | two | T | F | n | raven wall covers; stray birds persist into B |
| run_set_on_fire | morph | **overlay** | state | F | A | T* | F | n | runner catches fire mid-run; same run, same alley; *cam=y arguable under locked-off test — re-judge with air_bending |
| saint_glow | morph | **overlay** | add | F | A | F | F | n | halo forms around subject; subject unchanged |
| sakura_petals | occlusion | **overlay** ⚑ | remove | F | A | F | F | y | conflicting filmstrip reads: external petal swarm covering the subject (→ overlay-remove) vs the suit visibly eroding INTO petal clusters (→ conversion, T1 transform) — owner rules convert-vs-extract |
| seamless_transition | traversal | **cut** | — | T | two ⚑ | F* | F | y | hidden edit: walk-out, empty beats, walk-in different room; *cam flips (camera static); sidedness convention needed (effect in neither window) |
| shadow | morph | **transform** | — | F | A | F | F | n | subject becomes a living shadow (conversion visible) |
| shadow_smoke | occlusion | **cover** ⚑ | — | T | two | T | F | y | body-derived smoke covers frame, clears to new scene — unless it *reforms* into B (air_bending precedent) → transform |
| super_fast_run | traversal | **traverse** | — | T | A | T | F | n | sprint + tracking blur carry shot to different environment; still sprinting at end (v1 hard-call on sidedness, not a tracked conflict) |
| water_bending | morph | **overlay** ⚑ | state | F | A ◆ | F | F | n | water manipulated around/onto subject — accrual or substitution? |
| water_element | morph | **transform** ⚑ | — | F | A ◆ | F | F | n | verified exemplar 2 = body→water (transform); other exemplars heterogeneous — confirm or split class |
| wireframe | morph | **transform** | — | F | A | F | T | n | subject+scene restyle to wireframe |
| wonderland | morph | **transform** | — | F | A | F | T | n | whole frame restyles to stylized look |

### 5.1 Consistency table (annotate-and-verify)

`mechanism ∈ {cover, traverse, cut}` ⇒ `scene_swap = yes`. `mechanism = overlay` ⇒
`scene_swap = no`. `mechanism = transform` spans both (air_bending T; animalization F).
`mechanism = traverse` ⇒ `camera_defining = yes`. Any violation = annotation error, escalate.
(This check applied to v1 would have caught polygon.)

## 6. Strata, power, and pre-registration impact

Counts: **transform 12 · overlay 14 · cover 6 · traverse 6 · cut 1** (39 ✓).

- **Confirmatory sign-test strata:** transform (12; 10/12 → p≈.019) and overlay (14; 11/14 →
  p≈.029). For copy-sensitive claims use **transform ∧ ¬stylization (8)** — a crossfade is
  approximately a *correct* answer for restyles, so the 4 restyle classes are the pre-declared
  **copy_max calibration subgroup**, excluded from copy-confound sign tests.
- **Pooled confirmatory stratum "new-shot handoff" = cover ∪ traverse ∪ cut (13)** — coherent
  (all synthesize a passage to a different shot) and absorbs the cut singleton. cover (6) and
  traverse (6) alone are descriptive (p<.05 only if unanimous); pre-register them as such.
  Do not gerrymander assignments to inflate strata.
- **Recomputation policy:** 4 of the 7 owner rulings (§7) sit on the transform/overlay boundary,
  so rulings can shift the 12/14 split (e.g., sakura→transform ⇒ transform 13 / overlay 13).
  Stratum membership and this section's arithmetic are recomputed once §7 rulings land, BEFORE
  any scoring consumes the strata; the pre-registration records post-ruling counts.
- **Capability → metric mapping:** cover = occluder synthesis + reveal (seam_z; copy_max guards
  freeze-fade cheating) · transform = correspondence morphing (copy_max/lerp is THE confound;
  m1a, obj_match) · overlay = local synthesis over a copyable background (copy_max
  region-restricted; obj_match survivor persistence) · traverse = ego-motion continuity
  (cam_dtw, m1b) · cut = **no cut-alone claims**; contributes only via the pooled
  new-shot-handoff stratum.
- **middle_only (y=14 / n=25):** pre-registered headline split for R1−R0 (conditioning contains
  zero effect evidence ⇒ predict strongest suppression, largest R2 gains). Count note:
  display_transition is provisionally n pending its §7 ruling.
- **scene_swap (T14/F25):** copy_max regime + seam_z applicability split.
- Sidedness stays {A_only 27, two_sided 12} pending the owner's 9 conflict rulings; B_only
  remains empty ⇒ all suppression analyses are A-side-weighted (note in report).

## 7. Owner ruling list (exactly 7 mechanism calls) + existing sidedness work

Mechanism rulings (watch the videos, apply §4): **water_element** (do the other exemplars match
the verified body→water transform? else split/exclude), **water_bending**, **fire_element**,
**earth_wave** (each: does the body *become* the element → transform, or *host* it → overlay-state),
**shadow_smoke** (does the smoke merely clear → cover, or reform into B → transform),
**sakura_petals** (external swarm extracting the subject → overlay-remove, or suit eroding into
petals → transform; filmstrip reads conflict),
**display_transition** (ratify the cover-interior clause; check monitor visibility in first panel
for `middle_only`).
Unchanged and still owed: the **9 sidedness conflicts** (◆) in the viewer pass — plus one new
convention call: sidedness for classes whose effect appears in *neither* endpoint window
(seamless_transition, flying_cam_transition).

## 8. Corpus repairs (independent of schema)

1. **water_element** — 3 exemplars = 3 different effects; not a class as-is. Exclude from
   specialist rungs or re-curate.
2. **portal clip 0** — fully cartoon-animated; medium heterogeneity poisons DINO/m1a stats;
   replace exemplar.
3. **live_concert** — roster all-dup (already tracked); fix before any rung consumes it.
4. **flying_cam_transition / live_concert notes were empty** in v1 — behaviors now documented
   (§5) from filmstrip inspection.
5. **v1 taxonomy files are untracked** (`outputs/taxonomy/` not in git) — migrate PROTOCOL.md,
   class_axes.yaml (+v2 successors) into tracked paths with the v2 adoption commit.

## 9. What does NOT change

- `sidedness` semantics, mask S construction, every certified metric, τ_copy, σ_seed numbers,
  certification v3.0.0 status: **untouched.** (A proposed *operationalization* of the sidedness
  test — "effect visible in first/last ~1s window" — is recorded as a candidate but must be
  checked against SPEC.md's S-mask definition before anyone annotates with it. Caution: it would
  contradict current labels for every `middle_only=y ∧ two_sided` class — 8 classes, e.g.
  firelava, melt_transition — so if pursued it is a *relabel proposal* requiring owner + §7-style
  review, never a "clarification.")
- split_v1 (FINAL, tag split/v1), training rosters, all submitted jobs (exp_062/exp_063),
  R0/R1 generations: **untouched.**
- v1 labels remain the annotation of record until owner sign-off; conversion is the §5 table.
