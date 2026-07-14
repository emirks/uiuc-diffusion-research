# Advisor consultation log (OPERATIONS §8)

Every consultation: timestamp, question asked, direction received, action taken.
The advisor steers EXECUTION. It cannot override a frozen gate, kill rule,
threshold, or adoption rule, authorize a rescue variant, or make the §7 call;
those are owner-reserved and stop the track. Advisor input never softens or
reframes a committed record — verdicts stay rule text + numbers.

**Channel note (2026-07-14 10:55).** The Claude Code `advisor` tool is
unavailable in this session (API-level). The advisor channel is therefore
implemented as a Fable reasoning consult through the Agent tool, given the
RUNBOOK, OPERATIONS, the deployed seams, and the run's intermediate results.
Same may/may-not boundary; same logging requirement.

---

## C1 — 2026-07-14 11:00 · pre-work plan + step-0 findings

**Asked.** (Q1) RUNBOOK A4: which reading of "within-stratum recall" gates §3.6,
given that the drafted 0.62/0.44 reproduce under NO definition I can construct,
and is that a stop? (Q2) endpoint/whitening mechanics — RUNBOOK §1.2 ("mean
whitened embedding") vs OPERATIONS §5 (`endpoint_vecs` then whiten), which
differ by a nonlinear L2-renorm. (Q3) GPU cache-build design: dense-flow storage
(~15 GB) vs downsampling; ZCA fit population. (Q4) run E1 immediately after the
cache job, before Phase 1's acceptance tests?

**Direction received.**

- **Q1 — pin definition (iv), `macro_per_class_eligible`; proceed on artifact
  truth; NOT an owner-stop.** A4's "the same stratum-grouping code that will
  judge the candidates" is a *symmetry* requirement (identical code both sides),
  not a pointer to `diagnostics.tag_accuracy`. §3.5 says the incumbent's
  *recorded* stratum recalls, and the only stratum recall the artifact actually
  recorded is `o7_conditional.camera_stratum_mean_recall` — which (iv)
  reproduces bit-for-bit. RUNBOOK §6 ("n≤3: reported, never gating") positively
  forbids the clip-pooled variants, which let ineligible clips move a gating
  number. 0.62/0.44 were never validly pinned — A4 pre-registered their
  replacement before any candidate ran, so this is A4 operating as designed (the
  A1 pattern), not baseline re-pinning. Requirements: freeze the definition and
  the targets in gates.yaml before any candidate descriptor is computed;
  enumerate the eligible class lists as frozen corpus facts; record all five
  definitions; pin the NaN-support edge case pre-run; disclose the 0.62/0.44
  finding loudly in baselines.json, the step-0 commit, and WORKBENCH_REPORT.md.
  Hard line carried forward: if a candidate's §3.6 verdict flips between (iv)
  and any recorded alternative, the frozen (iv) cell decides the verdict and the
  sensitivity is reported neutrally beside it — re-adjudicating the cell choice
  after candidate numbers exist is owner-only.
- **Q2 — follow OPERATIONS: `endpoint_vecs` verbatim, whiten after.** Not a
  science conflict: §1.2's phrase is underdetermined on renorm, and the
  conditioned windows *are* the flanking stable frames outside the S-mask.
  `endpoint_vecs` is the deployed endpoint definition everywhere the certified
  instrument reasons about endpoints; a divergent workbench anchor would be
  worse than the renorm wart. The renorm cancels in E1 (no anchors in the
  formula) and is self-consistent in the min-D guard; it bites only E2's
  coordinates. Plus two E1 choices to pre-commit: (1) pool v_null over **the
  clip's own core frame indices**, not the null's own S-mask (a lerp's envelope
  stays ≥ threshold throughout, so its strict core is degenerate by construction
  and the fallback valley would pick an arbitrary sliver); (2) low_D excludes
  from E2's D-normalized channels, NOT from E1 (E1's delta contains no D).
- **Q3 — dense float16 flow at full res, ~15 GB, `savez_compressed`.** Anything
  less makes the §3.3 ε recalibration a GPU re-run, which is what the front-load
  rule exists to prevent. **Trap flagged:** torchvision RAFT needs H,W divisible
  by 8, and the deployed `RAFTExtractor._resize` rounds to /8 **only when it
  downscales**; decoding at short_side=320 gives 320×427 and passes through
  unrounded. Feed native 480×640 with `max_size=432` — the wrapper's own
  arithmetic then yields exactly 320×432. Cache all 120 pairs per clip (seam
  exclusion is descriptor-time, not cache-time). Pin resolution, weights enum,
  `num_flow_updates`. Never edit `signals/flow.py`; batch in `flowcache.py` if
  needed. **ZCA fits on S-mask core frames** — §1.1's own parenthetical ("S-mask
  frame selection") pins it; fitting on all frames would be quiet drift on a
  frozen scientific choice. Deterministic eigenvalue floor, recorded, frozen
  pre-candidate. Nulls: deployed `make_lerp` verbatim, endpoints decoded at
  short_side=256, lerp null for one-sided clips too (not `make_static_hold`),
  own `lerpnull-v1` tag into `$WB_CACHE`; make the two-cache split structural so
  a corpus-side miss inside the GPU job still raises rather than writes.
- **Q4 — E1 immediately after the cache job: endorsed.** §6 sanctions
  interleaving; E1 gates ~4 days of downstream work and depends only on nulls +
  ZCA + the frozen kernel.
- **Step-0 stress tests.** Stub extractor confirmed necessary (`need_frames=False`
  alone does NOT protect the cache: on a miss the pipeline decodes and
  `array_features` computes *and writes*). Add a before/after cache
  `mtime_ns` audit so non-pollution is a recorded fact, not an argument. Pin BLAS
  threads for the bitwise round-trip as a pre-declared determinism control.

**Action taken.** All directions adopted; none required overriding a frozen rule.
gates.yaml pins definition (iv) with targets 0.34623 / 0.03427, the
`misretrieved` convention, the NaN-support convention, and the two E1 choices,
all before any candidate computation. baselines.json enumerates the frozen
eligible class lists and records all five stratum definitions plus the
0.62/0.44 disclosure. bundles.py gains the cache mtime_ns audit and the BLAS
thread pin (the round-trip had already come out bitwise-exact without it; the
pin is now declared rather than discovered). The ZCA OOD concern is parked in
IDEAS_NEXT_CYCLE.md, unregistered and non-authoritative.
