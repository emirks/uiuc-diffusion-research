# BRIEF P3c — feature-store follow-ups from the P3b report (implementer: opus48)

Branch `feature-store` @ af4488c or later; LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa; python `$LAB/envs-aarch64/ltx2/bin/python`
(never bare `python3`). No GPU, no sbatch/srun. Read: `misc/2026-09-17_feature_store/P3b_REPORT.md` §6–§7, `store/FEATURES.md`,
`src/diffusion/feature_store.py`, `src/diffusion/transition_eval/store_io.py`, `src/diffusion/transition_eval/certify/run_certification.py`,
`scripts/store_features.py`. The migration chain has finished: `feature_store.py` and `store_features.py` are no longer frozen.

## Deliverables
1. `src/diffusion/feature_store.py`
   a. Public control-variant write: `put(video, ns, arrays, meta, *, variant=None, …)` writes `<ns>.ctl-<variant>.npz/.json` (the form
      `store_io` currently reimplements); `has/get/read_meta/sidecar/path` accept `variant=` too. `store_io.py` then calls the library
      (delete its private writer). Same atomic tmp+rename, sidecar-last.
   b. `coverage`/`fsck` treat `*.ctl-*` files as first-class: fsck validates their pair + sidecar exactly like namespaces (stale-check
      against the GEN's video identity), coverage reports them in a separate `controls` column (`<ns>` cell counts only real-namespace files).
   c. `put(..., video_sha256=None)`: when given, skip re-hashing; add `FeatureStore.sha_from_sums(video)` that reads `videos/SHA256SUMS`
      (or the clip folder's SHA256SUMS) and returns the sha or None. `store_features.py extract` and `store_io` use it (P4 throughput).
   d. Keep every existing test green; add tests for a–c.
2. `scripts/store_features.py` startup: `--help` takes ~2 min because `import diffusion` runs the package `__init__` (torch). Make the CLI
   import `feature_store` WITHOUT triggering `diffusion/__init__.py` (e.g. load `src/diffusion/feature_store.py` by file path via
   `importlib.util`, since it is numpy+stdlib only) — do NOT edit `diffusion/__init__.py`. `extract` may still import torch lazily.
   Verify `--help` returns in < 5 s.
3. `src/diffusion/transition_eval/certify/run_certification.py`: it still passes the removed `--cache-dir`/`--lpips-cache` to `score.py`.
   Port it to the store model: warm rerun = the shared store root (REPO_ROOT); cold anchors = an EMPTY temporary store root (so the six
   anchors re-extract everything, as before); no other semantic change. Unit-test the argument construction (no GPU). This is what P4's
   bar-8 run will execute — say in the report exactly how to invoke it.
4. `judge_gemini.py`: its `cache_dir` is a Gemini RESPONSE cache keyed by item_id (not a (video, ns) array) — leave it; add one sentence
   to SPEC.md §9 saying so.
5. Update `store/FEATURES.md` (controls column, sha_from_sums) and SPEC.md §9 minimally. Report `misc/2026-09-17_feature_store/P3c_REPORT.md`
   (`git add -f`), with the CHANGELOG entry text (do NOT commit CHANGELOG.md). Tests: `tests/test_feature_store.py`, `tests/test_transition_eval*.py`,
   `tests/test_versioning.py`, `tests/test_certify_v3.py` all green (PYTHONPATH=$PWD/src).

## Rules
No GPU. Do not touch the pre-existing dirty files (~66; includes `store/prompts/01{0,2}_*/meta.yaml`), `.claude/worktrees/*`, `CHANGELOG.md`.
Git identity `emirks <emirks88@gmail.com>`; stage by pathspec; push after every commit; message prefix `infra(feature-store):`/`eval(4.0.1-draft.1):`.
Reply with a concise summary: files, tests, how P4 invokes the certification driver, open questions.
