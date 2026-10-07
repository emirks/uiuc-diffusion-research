# P3a — feature-store library + CLI + tests: implementation report

**Author:** opus48 implementer · **Date:** 2026-09-17 · **Branch:** `feature-store` · **Machine:** DeltaAI login (gh-login01, aarch64) · **No GPU used.**

Authority followed: `store/FEATURES.md` (frozen registry), `misc/2026-09-17_feature_store/PROPOSAL.md` §2–§5/§8, `store/README.md` clause 10, and the certified `transition_eval` cache code.

## 1. What was built

| file | what |
|---|---|
| `src/diffusion/feature_store.py` | `FeatureStore(repo_root)` — `path`/`sidecar`/`has`/`get`/`read_meta`/`put(…, link_from=)`/`coverage`/`fsck(rehash=)`/`rebuild_manifest`/`write_meta_block`/`iter_videos`; module-level `legacy_key`, `legacy_filenames`, `sha256_file`, `NAMESPACES`, `NS_ARRAYS`, `MIGRATABLE`. numpy + stdlib only at import (verified: a direct-file import pulls no torch/transformers/av). |
| `src/diffusion/feature_extractors.py` | `REGISTRY` of 7 extractor factories. Three REUSE `transition_eval` verbatim (`DinoExtractor.extract`; `Tracker.track` at grid 20 / max-side 384, query frames 0+mid, backward tracking; `temporal_lpips` via `LpipsScorer("alex")`). Four competitor lenses (`clip_b32@r256`, `clip_l14@r224`, `videoprism@f16r288`, `raft_mag@r256`) ported faithfully from `their_metrics/{clip_sim,videoprism_sim,dynamic_degree}.py` per `SCORERS.json`. Backbones imported lazily inside `__init__`. |
| `scripts/store_features.py` | CLI `coverage`/`fsck`/`sha256sums`/`migrate`/`extract` with shared `--gens/--corpus/--conds` targets; importable workers `migrate_videos`, `extract_videos`, `write_sha256sums`. Records `host` (`socket.gethostname()`), `code_sha` (git HEAD short sha, `+dirty` iff the two feature-store files are modified — observed `c188167+dirty`), `created` (ISO-8601 UTC). |
| `tests/test_feature_store.py` | 13 CPU-only tests, fake extractor injected. |
| `store/FEATURES.md` | 3 minimal edits (see §4). |

Path rule implemented exactly: `root = video.parent.parent if video.parent.name=="videos" else video.parent`; `root/"features"/video.stem/"<ns>.npz"` (+ `.json` sidecar). Atomic writes: `<file>.tmp-<pid>` → `os.replace`; the `.json` is written LAST. Migrated npz is an `os.link` of the legacy file (same inode). Sidecar `video` path is relative to repo root.

## 2. pytest

```
$LAB/envs-aarch64/ltx2/bin/python -m pytest tests/test_feature_store.py -q
13 passed, 15 warnings in 106.19s
```
(The warnings are torch NVML/JIT-deprecation noise from the pre-existing `diffusion/__init__.py`, which eagerly imports torch when the package is imported by name; `feature_store.py` itself imports nothing but numpy+stdlib.) Tests cover: path rule (store-gen + corpus cases), atomic put + sidecar-last + lone-npz→orphan, hard-link migrate (same `st_dev,st_ino`) with a hand-computed legacy key, dry-run counts-no-link, extract fills-misses-only (fake extractor), manifest rebuild == sidecars, fsck stale (stat drift) + mixed-hosts + legacy, coverage counts + legacy, `write_meta_block` text/comment preservation + idempotence, sha256sums skip-by-size+mtime.

## 3. Acceptance on real data (CPU only)

Legacy dirs searched, in order: `misc/refvfx_baseline/probe/cache`, `outputs/eval/cache`. One filesystem confirmed (`stat -c %d` equal → hard links valid).

**3a. `migrate --dry-run --gens store/gens/013_dualforce_control/03_neutral_v3__dai` (564 videos)**

| namespace | hit / of | miss |
|---|---|---|
| dino_cls@dinov2b-r256 | 564 / 564 | 0 |
| cotracker3@g20-m384-v2 | 564 / 564 | 0 |
| lpips_t@alex-r256 | 564 / 564 | 0 |

Matches the brief's ≈564/564 expectation exactly (recipe correct).

**3b. `migrate` (real) on that one variant** — final state after the re-migrate that rewrote sidecars to the corrected host schema (idempotent; npz hard links unchanged — a sample npz kept inode `1116896540477098219`, `nlink=3`, shared with `misc/refvfx_baseline/probe/cache/dino_arr_*.npz`):

| namespace | linked | hit / of | miss |
|---|---|---|---|
| dino_cls@dinov2b-r256 | 564 | 564 / 564 | 0 |
| cotracker3@g20-m384-v2 | 564 | 564 / 564 | 0 |
| lpips_t@alex-r256 | 564 | 564 / 564 | 0 |

`coverage --gens …` → `dino_cls 564/564  cotracker3 564/564  lpips_t 564/564  clip_b32 0/564  videoprism 0/564  raft_mag 0/564  clip_l14 0/564`.
`fsck --gens …` → `[OK] … 1692 features over 564 videos | legacy=['cotracker3','dino_cls','lpips_t']`, **rc=0** (all migrated ⇒ reported `legacy`, not mixed-host). Sidecar example: `host: null`, `migrated_by_host: gh-login01.delta.ncsa.illinois.edu`, `origin: migrated:misc/refvfx_baseline/probe/cache/dino_arr_b3bb08feaaec0e4e.npz`, `shape:[121,768]`, `dtype:float32`. `meta.yaml` features block: `dino_cls@dinov2b-r256: {have: 564, of: 564, hosts: [], legacy: 564}` (× the three).

**3c. `migrate --corpus --dry-run` (4,536 clips over 398 classes)**

| namespace | hit / of | miss |
|---|---|---|
| dino_cls@dinov2b-r256 | 4535 / 4536 | 1 |
| cotracker3@g20-m384-v2 | 4535 / 4536 | 1 |
| lpips_t@alex-r256 | 144 / 4536 | 4392 |

DINO/tracks are essentially complete (1 clip of 4,536 missing both — identity not pinned down within the round; harmless for a dry run, `extract` fills it in P4). Temporal-LPIPS is sparse on the corpus (144/4,536), exactly as the proposal's probe predicted (LPIPS-t is a generated-video seam metric, not a reference-clip metric).

## 4. `store/FEATURES.md` changes (minimal)

1. `cotracker3@g20-m384-v2` arrays: `vis [T,N] bool` → **`vis [T,N] f32`**. Verified: legacy `tracks_*.npz` store `vis` as float32, and the verbatim `Tracker.track` reuse emits float32; migrated files are hard links, so the registry must state f32.
2. Sidecar paragraph: `host` = the EXTRACTING machine; migrated files have unknown extraction host → `host: null` plus an extra `migrated_by_host` (the node that ran `migrate`). (Per coordinator correction.)
3. Host-rule paragraph: mixed-host guard counts only KNOWN extraction hosts; an all-migrated namespace is reported `legacy`, not a mixed-host warning.

## 5. Deviations from the brief

- **Host semantics changed after first migrate** (coordinator correction): `host` now means the extracting host. Migrated sidecars carry `host: null` + `migrated_by_host`; `coverage` returns a per-ns `legacy` count; `fsck` returns `legacy_ns` and excludes null from the mixed-host guard. The one allowed variant was re-migrated so all 1,692 sidecars, the manifest, and the meta block reflect this.
- **CHANGELOG.md NOT committed.** It is a pre-existing dirty file (one of the ~66) already carrying 5 unrelated uncommitted 2026-09-17 entries from parallel work; staging it by pathspec would bundle those into my commit, violating the "do not touch the pre-existing dirty files / stage by pathspec only" rule. The intended entry is in §7 for the operator to append when they land the surrounding work. `CLAUDE.md`'s changelog mandate is therefore deferred, deliberately.
- **Four lens extractors are code-complete but NOT GPU-verified** (no-GPU round). Verified by code reading against the `their_metrics` sources + the fake-extractor tests only; the operator verifies them at P4 extraction. `clip_l14@r224` has no `their_metrics` source scorer — it mirrors the `clip_sim` `ClipEmbedder` pattern with `openai/clip-vit-large-patch14` (768-d), decoding at short-side 256 and letting the CLIP-L/14 processor resize to 224.

## 6. Open questions / recommendations

- **Migrated host provenance.** The true extracting host of the legacy features (ghx4/`dai`) is not recoverable from the caches, so migrated sidecars record `host: null` + `migrated_by_host: <login node>`. If stamping the real extraction host is wanted, add an optional `--host` to `migrate` (default `socket.gethostname()`); I did not, to keep to the brief's "record `socket.gethostname()`".
- **.gitignore for corpus/conds sidecars.** Under `/store/**` the `<ns>.json` sidecars, `manifest.jsonl`, and `.SHA256SUMS.stat.json` are already ignored. For corpus (`data/processed/transitions_std121/…`) and conds (`eval_ladder/conds/…`) the `.npz` and `manifest.jsonl` are ignored but the `<ns>.json` sidecars and `.SHA256SUMS.stat.json` are NOT — they would be git-tracked. Before P2/P3b migrates those trees, add e.g. `**/features/*/*.json` and `**/features/.SHA256SUMS.stat.json` to `.gitignore`. Not edited here (outside my write scope; does not affect this store-only + dry-run acceptance).
- **On-disk acceptance artifact.** The paper arm's `features/` (1,692 hard-linked npz + sidecars + `manifest.jsonl`, zero extra bytes, gitignored) and its `meta.yaml` features block are left in place as the acceptance result. P2's full migrate is idempotent and will treat this variant as already-done. Remove with `rm -rf store/gens/013_dualforce_control/03_neutral_v3__dai/features` + `git checkout` the meta.yaml if a pristine tree is preferred (but the coordinator asked to commit the meta.yaml).
- **1 corpus clip** misses dino_cls+cotracker3 (4535/4536). Likely never scored, or a stat change since extraction. Not chased down (dry run only).

## 7. CHANGELOG entry to append (operator; blocked — see §5)

```
- `23:50` **Feature store P3a — library + CLI + tests (`diffusion.feature_store`, `scripts/store_features.py`, `diffusion.feature_extractors`).** Path-is-identity per-video feature storage (contract v2 clause 10): `features/<item>/<ns>.npz` + `<ns>.json` sidecar next to `videos/` (gens) or inside the clip folder (corpus/conds); atomic tmp+rename, sidecar-last; migrated files are `os.link` hard links of the legacy caches (same inode). CLI `coverage|fsck|sha256sums|migrate|extract`; legacy-key recipe matches `transition_eval` byte-for-byte. Extractors: DINO/tracks/lpips_t reuse `transition_eval` verbatim, 4 competitor lenses ported from `their_metrics` (GPU-verify at P4). Sidecar `host`=extracting host (null for migrated + `migrated_by_host`). 13 pytest pass. Acceptance (CPU): `013_dualforce_control/03_neutral_v3__dai` migrated 564/564 on dino_cls/cotracker3/lpips_t (hard-linked, fsck OK, 1692 features); corpus dry-run 4535/4536 dino+tracks, 144/4536 lpips_t. `store/FEATURES.md` vis dtype bool→f32 + host-semantics note. No GPU.
```
