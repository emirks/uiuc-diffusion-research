# Ideas — next cycle

**NON-AUTHORITATIVE. This file is the single sanctioned non-neutral file
(OPERATIONS §8): a parking lot the executor may append to. It is never referenced
by any record or verdict, it gates nothing, and nothing in it revises a recorded
result. Ideas are parked here precisely BECAUSE acting on them this cycle would
violate the pre-registration.**

---

## Whitening regularization (parked from consultation C1; sharpened by C3)

RUNBOOK §1.1 mandates ZCA whitening for all downstream inner-product geometry and
justifies it as "raw DINO is anisotropic; unwhitened chords and angles are measured
with a bent ruler". It does **not** pin the regularization. The executor chose an
eigenvalue floor of 1e-6·λ_max and froze it before any candidate ran.

Measured this cycle (`e1_floor_sensitivity.json`, diagnostics only): DINO's
core-frame covariance spans λ ∈ [1.85e-9, 3.64e-2]. That floor floors 1 of 768
dims, so ZCA divides by √λ across ~700 near-null directions. The whitened
no-subtraction control — a representation containing no endpoint-normalization at
all — scores 0.0628 accuracy (chance 0.067) with hubness entropy 0.042, while the
same representation unwhitened scores 0.6054 with entropy 0.882.

Parked questions for a future pre-registration (NOT acted on here):
- Should whitening be regularized by a *fraction of the mean eigenvalue* or by
  retaining the top-k principal directions, rather than a fraction of λ_max? The
  latter is scale-free with respect to the spectrum's tail, which is where the
  problem lives.
- Should the ZCA fit population include the frames the map is APPLIED to? §1.1's
  parenthetical pins the fit to S-mask core frames, but the map is then applied to
  conditioned-window endpoint frames and to rendered-null curves, both off that
  manifold (measured: ‖whitened null‖ 35.5 vs ‖whitened clip‖ 14.6). C1 ruled that
  changing the fit population would be spec drift, correctly. A future cycle could
  register the choice explicitly, either way.
- Is whitening the right instrument at all for a *retrieval* metric? Whitening
  equalizes variance across directions, which is what you want for measuring angles
  and chords, and is arguably the opposite of what you want for nearest-neighbour
  retrieval, where high-variance directions carry the class signal. §1.1 assumes the
  first framing; the appearance ladder is graded on the second.

## Motion

- SEA-RAFT emits per-pixel uncertainty (`info` / `nf` heads) which this cycle
  discards. A future M1b could use it directly as the definedness signal instead of
  the inlier-fraction proxy, and as a principled weight in the IRLS.
- Huber is bounded-influence but low-breakdown; the measured bias vs contaminated
  area is in `m1b_flow.fit_similarity`'s docstring. A high-breakdown estimator
  (LMedS / RANSAC-seeded IRLS) would change the fit's behaviour on large effect
  regions. §3.2 pins Huber, so this is next-cycle only.
- `certify.diagnostics.clip_tags` parses tags out of source directory names by
  splitting on "_", so the hyphenated `onesided_object-monstrosity` silently loses
  its `object` tag. Harmless today (n=3, ineligible, and the gating code reads
  manifest tags), but it is a live bug in a descriptive table of the certified
  instrument and will eventually bite a class that matters.
