# Transition Taxonomy — Protocol v2 (ADOPTED)

**Status:** rev.4 — **ADOPTED. Owner-validated 2026-07-16, 39/39 classes signed off** in the
viewer (two correction exports folded the same evening; §5 is the owner-final table, mirrored
byte-for-byte by `scripts/build_class_axes_v2.py` → `outputs/taxonomy/class_axes_v2.yaml`).
Two standing §5.1 exceptions (plasma_explosion, portal) carry pre-registered conservative
handling until harmonized. Gate history: rev.3 PASSED the fresh-context acceptance gate
2026-07-16 (zero material defects; 12/12 re-derivations reproduce the table); rev.1 had FAILED
on two material wording defects (T1 trigger; sakura_petals flag).
**Date:** 2026-07-16. **Supersedes:** `outputs/taxonomy/PROTOCOL.md` (v1) — *descriptive layer only*.
**Instrument impact:** `sidedness` *semantics*, mask S construction, the certified harness
(eval/v3.0.0), split_v1, and all training rosters are untouched. The owner's validation did
flip two per-class sidedness *labels* (giant_grab, hero_flight → twosided) — folded into
`corpus_manifest.json` via the documented owner-override (§6, last bullet; amendment-2).

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

## 5. Full 39-class assignment (v1 → v2) — OWNER-FINAL

All 39 classes reviewed and signed off by the owner in the viewer on 2026-07-16 (two
correction exports, both folded; audit trail in `outputs/taxonomy/fold_report_v2_*.md`).
Sub = overlay_direction. ⚠ = standing §5.1 exception, owner-validated as written but
inconsistent with the implication table (see §5.1) — 2 classes.

| class | v1 mech | **v2 mech** | sub | swap | side | cam | styl | mid | behavior (one line) |
|---|---|---|---|---|---|---|---|---|---|
| air_bending | occlusion | **transform** | — | T | two | T | F | y | subjects dissolve into cotton-smoke which reforms as B (owner-ruled); cam: owner set y in v1 pass - re-judge under locked-off test |
| animalization | morph | **transform** | — | F | A | F | F | y | body re-renders in place into animal; transient ring |
| color_rain | morph | **overlay** | state | F | A | F | T | y | liquid drenches subject; grade persists |
| cotton_cloud | morph | **overlay** | add | F | A | F | F | y | cotton fills scene around untouched subject |
| display_transition | occlusion | **cover** | — | T | two | T | F | y | held-up screen fills frame; B = screen interior (RULING: ratify interior clause; is monitor in first panel? middle_only provisional n) |
| earth_element | morph | **transform** | — | F | A | T | F | y | body cracks into rock in place |
| earth_wave | morph | **transform** | — | T | two | T | F | y | sand wave wraps body (RULING: becomes element -> transform, or hosts it -> overlay-state) |
| fire_element | morph | **transform** | — | F | A | F | F | n | fire manifests on/around posed subject (RULING: accrual or substitution?) |
| firelava | occlusion | **transform** | — | T | two | T | F | y | fire wall sweeps over the silhouette (no visible conversion) during tracked shot; reveal |
| flame | occlusion | **cover** | — | T | two | F | F | y | fire -> full-frame whiteout -> recedes off different subject/set |
| flying_cam_transition | occlusion | **traverse** | — | T | two | T | F | n | filmstrip-verified: doorway rises from desert sand; camera walks through into living room; no covering matter at handoff |
| gas_transformation | morph | **transform** | — | F | A | F | F | y | body dissolves into gas with correspondence (conversion, dispersal-to-absence) |
| giant_grab | morph | **overlay** | remove | F | two | F | F | n | inserted hand drags subject out; scene persists; hand re-enters |
| hero_flight | traversal | **traverse** | — | T | two | T | F | n | camera follows the launch into sustained flight persisting into B window (advisors read two_sided; owner rules) |
| hole_transition | occlusion | **cover** | — | T | two | T | F | y | camera zooms THROUGH an open ring/aperture into B's actual space (passage, not picture) |
| illustration_scene | morph | **overlay** | state | F | A | F | T | n | photoreal -> flat illustration re-render in place |
| jump_transition | traversal | **traverse** | — | T | two | T | F | y | camera follows jump arc to a different place/outfit |
| live_concert | traversal | **traverse** | — | T | A | T | F | n | filmstrip-verified: backstage close-up pulls back through stage haze to festival wide (roster all-dup handled separately) |
| luminous_gaze | morph | **transform** | — | F | A | F | T | n | eyes ignite; storm + rim-light accrue over persisting subject/scene |
| melt_transition | occlusion | **transform** | — | T | two | F | F | y | prop-derived melt covers frame; trace persists into B's first frames (mid=y assumes trace clears before final ~1s window - confirm on video) |
| money_rain | morph | **overlay** | add | F | A | F | F | n | bills fall into persistent scene |
| monstrosity | morph | **overlay** | add | F | A | T | F | n | creature grows in background; scene persists |
| mystification | morph | **transform** | — | F | A | F | F | y | subject dissolves into colored smoke (conversion, dispersal-to-absence) |
| nature_bloom | morph | **overlay** | add | F | A | F | F | y | flowers grow into background |
| plasma_explosion | occlusion | **overlay** ⚠ | add | T | A | T | F | y | / same intersection throughout; explosion cloud added, persists; cam=y arguable under locked-off test - re-judge with air_bending |
| polygon | traversal | **overlay** | state | F | A | F | T | n | whole frame restyles photo -> white low-poly; cam flips to F under locked-off test |
| portal | occlusion | **cover** ⚠ | — | F | A | F | F | y | / portal opens at subject, removes them, vanishes; street persists empty (corpus: clip 0 is cartoon - replace exemplar) |
| raven_transition | occlusion | **transform** | — | T | two | T | F | y | raven wall covers; stray birds persist into B |
| run_set_on_fire | morph | **transform** | — | F | A | T | F | n | runner catches fire mid-run; same run, same alley; cam=y arguable under locked-off test - re-judge with air_bending |
| saint_glow | morph | **overlay** | add | F | A | F | F | n | halo forms around subject; subject unchanged |
| sakura_petals | occlusion | **transform** | — | F | A | F | F | y | RULING: conflicting filmstrip reads - external petal swarm covering subject (overlay-remove) vs suit eroding INTO petals (conversion -> transform) |
| seamless_transition | traversal | **traverse** | — | T | two | T | F | n | hidden edit: walk-out, empty beats, walk-in different room; camera static; sidedness convention needed (effect in neither window) |
| shadow | morph | **transform** | — | F | A | F | F | y | subject becomes a living shadow (conversion visible) |
| shadow_smoke | occlusion | **transform** | — | T | two | T | F | y | body-derived smoke covers frame, clears to new scene (RULING: merely clears -> cover, or reforms into B -> transform, air_bending precedent) |
| super_fast_run | traversal | **traverse** | — | T | A | T | F | n | sprint + tracking blur carry shot to different environment; still sprinting at end |
| water_bending | morph | **transform** | — | T | two | T | F | y | water manipulated around/onto subject (RULING: accrual or substitution?) |
| water_element | morph | **transform** | — | F | A | F | F | y | verified exemplar 2 = body->water (transform); other exemplars heterogeneous (RULING: confirm or split class; corpus: maybe not a class) |
| wireframe | morph | **overlay** | state | F | A | F | T | n | subject+scene restyle to wireframe |
| wonderland | morph | **overlay** | state | F | A | F | T | n | whole frame restyles to stylized look |

### 5.1 Consistency table (annotate-and-verify)

`mechanism ∈ {cover, traverse, cut}` ⇒ `scene_swap = yes`. `mechanism = overlay` ⇒
`scene_swap = no`. `mechanism = transform` spans both. `mechanism = traverse` ⇒
`camera_defining = yes`. A violation is an annotation error — with two standing
owner-validated exceptions (2026-07-16 sign-off): **plasma_explosion** (overlay + swap=T)
and **portal** (cover + swap=F). Pre-registered handling until the owner harmonizes each
(one click: change mechanism or swap): each is EXCLUDED from any stratum whose
interpretation rests on the violated implication — portal from the pooled new-shot copy
tests (its scene persists ⇒ copying is partially correct there, unlike true covers),
plasma_explosion from overlay-stratum copy-regime tests (its swap=T breaks the
"background copying is partially correct" reading). Both keep their per-class rows.

## 6. Strata, power, and pre-registration impact — POST-VALIDATION COUNTS

Counts (owner-final): **transform 17 · overlay 12 (add 6 / state 5 / remove 1) · cover 4 ·
traverse 6 · cut 0** (39 ✓). The owner's rulings moved every borderline conversion INTO
transform (raven, melt, firelava, shadow_smoke, sakura, water_bending, earth_wave,
fire_element, luminous_gaze, run_set_on_fire) and emptied cut (seamless → traverse).

- **Confirmatory sign-test strata:** transform (17; 13/17 → p≈.025) and overlay (12; 10/12 →
  p≈.019). For copy-sensitive claims use **transform ∧ ¬stylization (16)**; the
  **copy_max calibration subgroup is now `stylization=T` (6: 5 overlay-state + luminous_gaze
  transform)** — a crossfade approximates a correct answer for restyles, so these 6 are
  excluded from copy-confound sign tests. overlay ∧ ¬stylization = 7 (descriptive-leaning;
  9/9–7/7 unanimity zones only if further split).
- **Pooled confirmatory stratum "new-shot handoff" = cover ∪ traverse (10)** — cut is empty.
  Power note: n=10 needs **9/10** for p<.05 (8/10 → p≈.055). Portal's §5.1 exception (above)
  makes the effective copy-test pool **9** (needs 8/9, p≈.020) until harmonized. cover (4)
  and traverse (6) alone are descriptive; pre-registered as such.
- **Capability → metric mapping:** unchanged in kind — transform = correspondence
  morphing (copy_max/lerp THE confound; m1a, obj_match) · overlay = local synthesis over a
  copyable background · cover = occluder synthesis + reveal (seam_z) · traverse = ego-motion
  continuity (cam_dtw, m1b). Camera-transfer secondary contrast (pre-registered while still
  outcome-blind): **traverse (6) vs cover (4)** — both demand a new shot; only traverse
  demands synthesized camera travel. camera_defining (18/39) cross-cuts all four mechanisms
  and is the marginal stratum for any future camera/temporal-LoRA transfer claim.
- **middle_only (y=23 / n=16):** pre-registered headline split for R1−R0.
- **scene_swap (T17 / F22):** copy_max regime + seam_z applicability split.
- **Sidedness (owner-final): A_only 24 · two_sided 15 · B_only 0** ⇒ all suppression
  analyses remain A-side-weighted (report note). The 9 tracked conflicts were resolved by
  the owner 2026-07-16; net instrument-side relabels: **giant_grab and hero_flight
  onesided→twosided** (corpus_manifest updated via owner-override in
  `build_corpus_manifest.py`; certification amendment-2 records the σ_seed caveat —
  hero_flight was a σ_seed roster item drawn as onesided).

## 7. Owner rulings — ALL RESOLVED (2026-07-16 viewer sign-off)

The 7 mechanism rulings, the 9 sidedness conflicts, the 3 camera re-checks, and the
neither-window sidedness convention were all settled by the owner's full-table review
(39/39 validated). The owner-final record is `outputs/taxonomy/class_axes_v2.yaml`,
reproducible from `scripts/build_class_axes_v2.py`; the §5 table above mirrors it.
Notable outcomes: dissolve/erode/host borderline classes uniformly → **transform**;
seamless_transition → traverse (camera ruled moving); display_transition cover-interior
clause ratified with middle_only=y; plasma_explosion & portal left as the two ⚠ §5.1
exceptions (§5.1 pre-registered handling applies until harmonized).

## 8. Corpus repairs (independent of schema)

1. **water_element** — 3 exemplars = 3 different effects; not a class as-is. Exclude from
   specialist rungs or re-curate.
2. **portal clip 0** — fully cartoon-animated; medium heterogeneity poisons DINO/m1a stats;
   replace exemplar.
3. **live_concert** — ~~roster all-dup~~ **[CORRECTED 2026-07-22: this "all-dup" claim is stale/wrong.
   PLAN.md Amendment 2 (owner visual review 2026-07-16) found exactly ONE true duplicate
   (`live_concert_2`, quarantined to `_removed/`); the remaining clips are distinct. live_concert is a
   USABLE class — 7 clips (6 train / 1 test), mechanism=traverse, sidedness=A_only. Used as a held-out
   zero-shot class in ladder2 split v1.2. `live_concert_2` is barred from ref+endpoint roles.]**
4. **flying_cam_transition / live_concert notes were empty** in v1 — behaviors now documented
   (§5) from filmstrip inspection.
5. ~~v1 taxonomy files are untracked~~ **DONE** — v1 archived at `docs/taxonomy/v1_PROTOCOL.md`
   + `docs/taxonomy/v1_class_axes.yaml`; v2 reproducible from `scripts/build_class_axes_v2.py`.

## 9. What does NOT change

- `sidedness` semantics, mask S construction, every certified metric, τ_copy, σ_seed numbers,
  certification v3.0.0 status: **untouched.** Per-class sidedness *labels* were owner-validated
  2026-07-16; the two label flips (giant_grab, hero_flight → twosided) are folded into
  `corpus_manifest.json` and documented in certification amendment-2 (hero_flight σ_seed-roster
  caveat included). (A proposed *operationalization* of the sidedness test — "effect visible in
  first/last ~1s window" — remains a candidate only; it would contradict validated labels for
  every `middle_only=y ∧ two_sided` class, so if pursued it is a *relabel proposal* requiring
  owner + §7-style review, never a "clarification.")
- split_v1 (FINAL, tag split/v1), training rosters, all submitted jobs (exp_062/exp_063),
  R0/R1 generations: **untouched.**
- v1 labels remain the annotation of record until owner sign-off; conversion is the §5 table.
