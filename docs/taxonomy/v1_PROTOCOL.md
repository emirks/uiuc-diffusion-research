# Transition taxonomy annotation protocol — v1 (FROZEN 2026-07-14, before any labeling)

Six fields per class. Labels are judged from the actual video frames, never from
the class name or its text description (known divergence example: shadow_smoke's
smoke does NOT cover the whole frame — it is a morphing, circularly traveling
smoke mass mid-screen; names lie, frames don't).

Annotation unit: the CLASS. Watch 2–3 exemplars per class (first / middle / last
in lexicographic clip order; all clips for n ≤ 2). If any field disagrees between
exemplars, the class is HETEROGENEOUS: record per-clip labels for it and set
`heterogeneous: true`. Do not average.

Labels are assigned blind to all metric scores, trust maps, and ladder results.

## The handoff

Every transition has a handoff: the interval where A-content is last visible and
B-content first visible. Several fields are judged AT this moment, not over the
whole clip.

## Fields

### 0. scene_swap — `yes` / `no`
Are the two endpoints different shots (different location/scene/subject
framing), or is B the SAME shot in a transformed state (subject removed,
scene restyled/drenched/emptied)? Different shots → yes. Same-shot
state change → no. Judge from first vs last frame.

### 1. sidedness — `A_only` / `B_only` / `two_sided`
Which endpoint's frames does the effect visibly alter? Effect activity touching
only the departure from A → A_only. Only the arrival into B → B_only. Both →
two_sided.
INSTRUMENT-CRITICAL: this field is consumed by the certified core mask S. Any
label that disagrees with the corpus manifest's current sidedness, and any
B_only finding at all (S cannot represent it), is escalated to the owner —
flag it `sidedness_conflict: true` and do NOT resolve it yourself.

### 2. mechanism — `occlusion` / `morph` / `traversal` / `dressed_cut`
What carries the handoff?
- occlusion: the handoff frame is substantially covered by effect-generated
  content (smoke wall, flock, fire, money); B is revealed from behind it. No
  A↔B correspondence needed.
- morph: continuous spatial or representational correspondence — A's content
  deforms, decomposes, or restyles into B's; real scene content visible through
  the handoff. INCLUDES continuous physical rearrangement/removal of scene
  content by an effect agent (rigid correspondence counts — e.g. a giant hand
  dragging the subject off-frame) and in-place restyle/state accumulation when
  there is no underlying cut (photo→illustration, rain progressively tinting
  the whole scene).
- traversal: camera motion in scene space carries the view from A's world to
  B's (whip, fly-through, zoom-into) with real scene content visible during the
  move.
- dressed_cut: none of the above carries the swap — underneath the effect is
  essentially a cut or dissolve, dressed with an overlay or treatment.

Tie-breakers (frozen):
- T1 occlusion vs traversal: if the handoff frame is substantially covered by
  effect content, occlusion wins — even if the camera is flying into it.
  (portal: mechanism=occlusion, camera_defining=yes.)
- T2 frame-wide treatment at handoff (restyles): judge with the treatment
  removed — dissolve/cut underneath → dressed_cut; continuous deformation
  underneath → morph. (A stylized dissolve is dressed_cut + stylization=yes.)
- T3 compounds: judge the handoff frame only. (gas_transformation: subject
  morphs to gas, gas then fills frame — if the handoff frame is covered,
  occlusion.)

### 3. camera_defining — `yes` / `no`
Is deliberate camera work part of the effect's identity (whip, orbit, fly,
push)? Incidental handheld drift = no. Note: traversal ⇒ camera_defining
(consistency rule).

### 4. inserted_content — `yes` / `no`
Does the effect introduce content present in NEITHER endpoint and NOT derived
from endpoint content by transformation? Origin test: enters ex nihilo or from
off-frame → yes (ravens, money, petals, rain). Transforms out of a visible
endpoint entity → no (a subject's gas form, motion streaks behind a runner,
melt drips of the existing scene).

### 5. stylization — `yes` / `no`
Pause a mid-effect frame: is a frame-wide appearance treatment evident (color
grade, illustration rendering, wireframe shading), independent of localized
effect content? Localized effect however dramatic = no.

### 6. subject_anchored — `yes` / `no`
Does the effect physically originate from or attach to a specific endpoint
entity (a person, a face, a single subject), or would it work unchanged on
arbitrary content? Anchored → yes (animalization, luminous_gaze, giant_grab).
Content-agnostic → no (color_rain, generic wipes).

## Per-field uncertainty
Every field may carry `hard_call: [field, ...]` — flag rather than force
confidence. A hard call is still labeled (best judgment).

## Output format (one YAML document, outputs/taxonomy/class_axes.yaml)
```yaml
classes:
  <class_name>:
    clips_viewed: [<clip>, ...]
    sidedness: A_only|B_only|two_sided
    mechanism: occlusion|morph|traversal|dressed_cut
    camera_defining: true|false
    inserted_content: true|false
    stylization: true|false
    subject_anchored: true|false
    hard_call: []          # subset of field names
    heterogeneous: false   # true → add per_clip: {<clip>: {..fields..}}
    sidedness_conflict: false  # true when disagreeing with corpus manifest
    notes: ""              # one line, only if something needs saying
```

## Consistency sweep (run after full annotation; violations get re-watched, not auto-fixed)
- HARD: mechanism=traversal ⇒ camera_defining=true.
- HARD: mechanism=dressed_cut ⇒ (inserted_content=true ∨ stylization=true).
- SOFT: mechanism=occlusion ∧ inserted_content=false is rare but legal (cover
  derived from endpoint content, e.g. a subject's own shadow expanding).

## Amendment log
Definitions may be amended ONLY during the pilot phase (6 stress classes:
portal, gas_transformation, illustration_scene, super_fast_run, giant_grab,
color_rain; plus shadow_smoke as the known name-vs-reality divergence case).
Every amendment is logged here with a reason. After the pilot, definitions are
frozen for the full pass.

(no amendments yet)
