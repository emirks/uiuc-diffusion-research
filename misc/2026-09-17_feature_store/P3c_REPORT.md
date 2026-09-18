# P3c — feature-store follow-ups: implementation report

**Author:** opus48 implementer · **Date:** 2026-09-18 · **Branch:** `feature-store` · **Machine:** DeltaAI login (gh-login, aarch64) · **No GPU, no sbatch/srun.**
Commits (both pushed, `ahead=0`): `a92238e` `infra(feature-store): public control-variant put + ctl-aware fsck/coverage + sha_from_sums; fast CLI import` · `4c45e70` `eval(4.0.1-draft.1): port certification bar 8 to the feature-store model`.
Authority followed: `BRIEF_P3c.md`, `P3b_REPORT.md` §6–§7, the merged P3b store code, and an advisor ruling on the cold-anchor store semantics (see §5).

## 1. Files changed (absolute paths)

- `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/src/diffusion/feature_store.py`
- `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/src/diffusion/transition_eval/store_io.py`
- `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/scripts/store_features.py`
- `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/src/diffusion/transition_eval/certify/run_certification.py`
- `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/store/FEATURES.md`
- `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/src/diffusion/transition_eval/SPEC.md`
- `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/tests/test_feature_store.py` (+5 tests)
- `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/tests/test_certify_v3.py` (+3 tests)

`judge_gemini.py`, `diffusion/__init__.py`, and every measurement/grader `.py` are **untouched**.

## 2. What changed, per deliverable

**1a — public control-variant `put`.** `FeatureStore.put(video, ns, arrays, meta, *, variant=None, link_from=None, video_sha256=None)`. When `variant` is set it writes `<ns>.ctl-<variant>.npz` + `.json` (same atomic `.tmp-<pid>`+rename, sidecar-last), stamps `origin` from `meta` (`control:<name>`), and adds a `control` field; the sidecar carries the **gen video's** identity (`video`=gen relpath, size/mtime from `gen.stat()`, sha reused). `has/get/read_meta/sidecar/path` all gained `variant=`. `store_io.HarnessStore` now delegates control writes/reads to the library (`store.put(gen, ns, …, variant=name)`); its private `_put_control` / `_control_paths` / `_gen_video_identity` / `_now_iso` / `_primary_array` are **deleted** — no I/O is reimplemented in the bridge any more.

**1b — ctl-aware `fsck` / `coverage`.** A file basename containing `.ctl-` is a control (the infix never occurs in a real namespace tag). `fsck`: control pair/sidecar/stale-checked exactly like a namespace (they carry the gen identity, so the item-folder mp4 stat validates them and a `--rehash` sha match is clean), counted in a new `n_controls`, but **kept out of the per-namespace host tracking** (so a control never triggers a spurious `mixed_hosts`/`legacy_ns`). `coverage`: a new `controls` key `{have, of, names}`; the `<ns>` cells count only real-namespace files. The CLI shows a `controls` column and a `(+N controls)` note in `fsck`.

**1c — precomputed-sha put + `sha_from_sums`.** `put(..., video_sha256=…)` (or `meta["video_sha256"]`) skips the whole-file re-hash. `FeatureStore.sha_from_sums(video)` reads the `SHA256SUMS` next to the clip (`videos/SHA256SUMS` for gens, the class-folder `SHA256SUMS` for corpus) and returns the sha or `None`. Wired into `store_features.py extract` (`video_sha256=store.sha_from_sums(v)`) and `store_io.HarnessStore.put` (real videos via `sha_from_sums`; controls reuse the gen's own sidecar sha, falling back to `sha_from_sums`).

**2 — fast CLI import.** `scripts/store_features.py` loads `feature_store.py` **by file path** via `importlib.util` (module name `diffusion_feature_store`) instead of `from diffusion.feature_store import …`, so it never triggers `diffusion/__init__.py` (torch). `diffusion/__init__.py` is not edited. `extract` still imports `diffusion.feature_extractors.REGISTRY` lazily inside `cmd_extract` (the only torch path). Measured: `store_features.py --help` returns in **0.508 s** (was ~2 min); `torch` is **not** in `sys.modules` after importing the CLI module.

**3 — certification bar 8 ported to the store model.** `certify/run_certification.py` no longer passes the removed `--cache-dir`/`--lpips-cache`. A pure `build_score_cmd(...)` emits `--store-root`; `start_score(..., store_root)` launches it. Warm rerun + all three first-pass runs pass `--store-root REPO_ROOT`. The driver's own corpus / reversed-ref bundles now go through `FeatureStore(REPO_ROOT)` (removed the legacy `outputs/eval/cache` backend — no split-brain). Cold anchors: `stage_cold_anchors()` hard-links (copy fallback, never symlink) each of the six anchor gens into an empty `out/cold_store/<item_id>/videos/`, rewrites **only** the `generated_video` field, and passes `--store-root out/cold_store`; guards `cold_store_populated()` (G2), `run_health()` (G3), a store-root-mismatch label (G4) and staging invariants (G1) are recorded, and `bar8.pass` now additionally requires the cold store to be populated. Rationale in §5.

**4 — `judge_gemini.py` left alone.** Verified its `cache_dir` writes `<item_id>.json` Gemini response text (keyed by `item_id`), not a `(video, ns)` array — it does not map onto the store and is off `score.py`'s numeric path. One sentence added to SPEC §9 saying so.

**5 — docs.** `store/FEATURES.md`: controls column + `sha_from_sums` + control-write-through-the-library. `SPEC.md` §9: controls first-class / `sha_from_sums` / bar-8 store port / judge note; a P3c follow-up clause on the `4.0.1-draft.1` changelog entry (no version bump).

## 3. Tests (CPU, `$LAB/envs-aarch64/ltx2/bin/python`)

Authoritative final run of the brief's required set:
```
PYTHONPATH=$PWD/src pytest -q tests/test_feature_store.py tests/test_transition_eval*.py \
    tests/test_versioning.py tests/test_certify_v3.py
97 passed, 19 warnings in 27.89s
```
(= 89 pre-existing + 5 new `test_feature_store.py` + 3 new `test_certify_v3.py`.) New tests: control-variant put/coverage/fsck first-class, control orphan + gen-identity stale, `sha_from_sums`+skip-rehash (video and clip-folder), `build_score_cmd` flags (`--store-root`, no `--cache-dir`/`--lpips-cache`), `stage_cold_anchors`+`cold_store_populated` real behavior (only-gen-path rewrite, hard link/byte-equal, not-symlink, under cold_root, cold miss not warm re-hit, `_relvideo` no-raise, writes land under cold_root, warm store untouched), `run_health` (error rows + missing ids).

CLI smoke (no GPU): `store_features.py coverage/fsck` on `store/gens/013_dualforce_control/03_neutral_v3__dai` renders the new `controls` column (`0`, no control files on disk yet — controls land at P4 GPU scoring) and `fsck` reports `1692 features over 564 videos` unchanged.

## 4. Exact P4 invocation of the certification driver (bar-8 run)

```
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research    # cwd MUST be repo root
PYTHONPATH=$PWD/src $LAB/envs-aarch64/ltx2/bin/python \
  -m diffusion.transition_eval.certify.run_certification \
  --corpus data/processed/transitions_std121/corpus_manifest.json \
  --main-root /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
```
Requires (unchanged): `certify/bars.yaml` frozen; cwd = repo root; corpus clips resolvable under `<repo>/<corpus_root>`; a GPU for cache-miss featurization. New store-model **P4 pre-flight** (advisor flag, see §6): before the run, ensure the corpus clips and the Block-C archive gens are **warm in the store** (`store_features.py fsck`/`coverage` clean on the corpus) and set `--main-root` = the repo root (so the exp_056/057/058 archive gens are colocated-warm). The cold store lands at `out/cold_store` and is persisted for audit. `--out` defaults to `outputs/eval/certification/<version>`.

## 5. Why the cold anchors are STAGED (design fork resolved by advisor)

The brief said "cold anchors = an EMPTY temporary store root". Verified empirically: P3b's `FeatureStore.path()` derives the feature location from the VIDEO's own path (`base/features/stem`), **not** from the store root — so a bare empty store root re-**HITS** the warm colocated features (no real cold test), and `_relvideo` raises `ValueError` on any write for a video outside the temp root (which per-item isolation would turn into an error row the comparator silently drops → a non-reproducing cold run could pass bar 8). The advisor's ruling (Option A, narrow): stage ONLY the six anchors' `generated_video` (hard link) into `out/cold_store/<item_id>/videos/`, rewrite that one field, `--store-root out/cold_store`. Only the gen path touches the store per item (DINO/tracks/lpips_t + the synthesized control, all keyed by the gen path); reference clips are read-only warm and condition clips never touch the store (endpoint LPIPS is recomputed fresh, control frames decode via `load_frames`). So the anchor gen + its control re-extract from nothing while references stay warm (identical files → identical numbers); the metric math is untouched. `compare_runs` compares only headline metrics over shared `item_id`s, and control row ids are path-free, so relocating the gen path changes no compared quantity. Three of the six anchors (`sib__*`) ARE corpus clips, so cold-vs-warm on them directly tests "stored corpus feature == fresh extraction".

## 6. CHANGELOG entry text (do NOT commit CHANGELOG.md — it is dirty)

```
- `HH:MM` **Feature store P3c — control-variant put in the library, ctl-aware fsck/coverage, sha_from_sums, fast CLI, and certification bar-8 ported to the store model.** feature_store.py: `put(..., variant="<name>")` writes `<ns>.ctl-<name>` (store_io deletes its private control writer and delegates); fsck/coverage treat controls as first-class (separate `controls` column, kept out of per-namespace host tracking); `put(..., video_sha256=)` + new `sha_from_sums(video)` skip re-hashing (wired into extract + store_io). scripts/store_features.py loads feature_store BY FILE PATH so `--help` returns in 0.5 s (was ~2 min) without triggering diffusion/__init__.py (torch); diffusion/__init__.py untouched. certify/run_certification.py: --cache-dir/--lpips-cache dropped; warm rerun + first-pass runs use --store-root REPO_ROOT; cold anchors STAGE the six anchor gens (hard link) into an empty out/cold_store/<item_id>/videos and pass --store-root out/cold_store so they re-extract from nothing (references/conditions warm), with cold-populated/run-health/store-root-mismatch guards closing the silent-pass mode; driver corpus/reversed-ref bundles now use FeatureStore(REPO_ROOT). judge_gemini.py left as-is (item_id-keyed response cache; SPEC §9 note). Tests: +5 feature_store, +3 certify; required suite 97 pass. No GPU. This is the driver the P4 bar-8 run executes.
```

## 7. Open questions / flags for P4

- **P4 pre-flight (advisor flag).** `score.py`'s `_ref_bundle_cache` runs OUTSIDE the per-item try: a corpus clip that is a store **miss** and resolves outside `--store-root` would crash score.py outright, and a Block-C gen under a foreign `--main-root` yields error rows. So P4 must first confirm store fsck/coverage clean on the corpus and `--main-root == REPO_ROOT` (or those archive gens warm). The new `run_health` guard converts a stray error row into a loud bar-8 fail instead of a silent pass, but the pre-flight avoids the crash.
- **`bars.yaml["frozen"]`** must be true for the driver to run at all (SPEC §6.5) — confirm before P4.
- **Reference pin / corpus for P4.** The certification driver passes `--corpus` to `score.py` but **not** `--reference-corpus`; the grid-v3 amendment path (evals/028+030 used a 677-clip corpus + 222-clip pin) is not exercised by this driver. If P4's bar-8 must certify against the amendment reference, that's a driver change beyond P3c — flag to the owner. As shipped, the driver certifies against whatever `--corpus` names, with the committed `reference_v4.npz` (`459fd9a7`) verified for that corpus's sha.
- **First real controls on disk.** No `<ns>.ctl-<name>` files exist in the store yet (they land at P4 GPU scoring). Watch that they don't drift `fsck`'s `manifest_drift` on the paper variants — rebuild the manifest (`fsck --rebuild-manifest`) after the first scoring pass; the control sidecars are now included in the manifest by design.
- **`certifications/v4.0.0.md`** still records the historical `e6ea4011` reference sha (the 4.0.0 cert record); unchanged, as in P3b.
