# ctt_v2plus_s6reshape — the CODE-SIDE root format (2026-08-30)

Authority for the physical form of `outputs/ctt_v2/roots/ctt_v2plus_s6reshape_mix`. This is the
**successor of `004_ctt_v2plus`'s `CODESIDE_FORMAT.md`** — the general code-side contract (row schema,
`_src` portability, the JSONL-vs-independent-sources verification battery) is unchanged from 004 and
is documented there; **this doc states only the DELTAS.** Round 4 of `misc/2026-08-30_s6_reshape`.

---

## Deltas vs 004

| aspect | 004_ctt_v2plus | 005_ctt_v2plus_s6reshape |
|---|---|---|
| total pairs | 114,215 | **138,147** |
| S6 pairs | 57,847 (26,266 targets) | **81,779 (28,552 targets)** — 2,286 targets returned |
| S6 grids | **4** native (11,22,33)/(11,22,39)/(11,33,22)/(11,39,22), 7,986–9,438 tok | **2** re-encoded (11,16,26)/(11,26,16), **4,576 tok each** |
| S6 dropped | 2,378 shape-singletons | **92** shape-singletons |
| mask store | 7 files | **5** files (S6 collapses 4 → 2) |
| S6 latents | `_src/…/encodes/EFFECTDATA/{latents,cond_clean}/` | `_src/…/encodes/**EFFECTDATA_r832**/{latents,cond_clean}/` |
| signal root | `armA_signals/feat` (NORM_dino_v3) | **`armA_signals_005/feat`** (**NORM_dino_v4**) |
| VERSION | `3.0.0-ctt_v2plus-codeside` | `3.1.0-ctt_v2plus_s6reshape-codeside` |
| contract id | `003_ctt_v2plus` | `005_ctt_v2plus_s6reshape` |
| verify battery | `verify_code_side.py` | **`verify_code_side_005.py`** |
| `samples_sha256` | `5a73eb3c…` | `048d1ef45d8ec98664291c8bce1c8c4b1f756435ae180abc89979ba323747d98` |

**Non-S6 rows (S0/S1/S2a/S2b/S4) are a MULTISET-IDENTICAL copy of 004** (full-row JSON, `id`
included — verified by `misc/2026-08-30_s6_reshape/r4/compare_004_005.py`). Only S6 changed.

## The 5 masks

```
_mask_store/
├── f16_h20_w15_p2_twosided.pt    corpus S0 (two-sided)
├── f16_h20_w15_p2_onesided.pt    corpus S1/S2a/S2b (one-sided)
├── f5_h14_w26_p1_onesided.pt     S4
├── f11_h16_w26_p1_onesided.pt    S6 landscape (832×512)   frame-0 plane sum 416 = 16×26
└── f11_h26_w16_p1_onesided.pt    S6 portrait  (512×832)   frame-0 plane sum 416 = 26×16
```
Both S6 masks: shape (11,H,W), frame-0 plane all-1 (sum 416), frames 1–10 all-0 (prefix_latents=1).

## `_src` bring-up — now also needs `EFFECTDATA_r832`

The device must present, under `_src`, the same relative layout 004 needs **PLUS** the reshaped
S6 encode dir:

```
_src/outputs/ctt_v2/encodes/EFFECTDATA_r832/{latents,cond_clean}/   # NEW (57,288 tensors, ~67 GB)
_src/datasets/ctt_v2/encodes/{S1,S2a,S2b,S4}/…                      # as 004 (realpath-collapsed)
_src/experiments/exp_058…/…, _src/eval_ladder/dataset/…            # as 004
```
(As in 004, paths are realpath-derived, so `outputs/ctt_v2/encodes → datasets/ctt_v2/encodes` is
collapsed and the on-disk form is `_src/datasets/ctt_v2/encodes/EFFECTDATA_r832/…`.)

Bring-up ritual (unchanged from 004 §4): rsync the small root, re-point `_src`, assert
`sha256sum samples.jsonl == 048d1ef4…` (also in `ROOT_MANIFEST.samples_sha256`), then
`SampleListDataset(verify_files=True)`.

> **samples_sha256 / samples_rows:** `scripts/ctt_v2/assemble_root.py`'s code-side branch now emits
> these two keys **NATIVELY** (since the Round-6 `data(005)` commit — locate it with
> `git log --oneline -- scripts/ctt_v2/assemble_root.py`): `samples_sha256 = sha256(samples.jsonl bytes)`,
> `samples_rows = row count`. The 004 and 005 ROOT_MANIFESTs on disk were **stamped post-assembly** in
> Round 4; Round 6's determinism proof re-emitted them natively into a scratch root and verified them
> **EQUAL** (scratch reproduced samples.jsonl sha `048d1ef4…` and the manifest key set, differing only in
> {created, root, mask_store.dir, shapes.note}). The config / eps generators read
> `ROOT_MANIFEST.samples_sha256` and assert it against the file.

## Verify

```bash
python misc/2026-08-24_flow_signal_conditioning/armA/verify_code_side_005.py         # invariants 1-5
python misc/2026-08-24_flow_signal_conditioning/armA/validate_training_ready_005.py  # V1/V6/V8
python misc/2026-08-30_s6_reshape/r4/compare_004_005.py                              # 004-parity of non-S6 + S6 deltas
```
All three PASS in Round 4 (2026-08-30). See the reports named in `README.md` and `meta.yaml:health`.

## Rebuild

```bash
python scripts/ctt_v2/s6/derive_inventory_r832.py         # S6_r832.json inventory (28,644 clips)
python misc/2026-08-30_s6_reshape/r4/seed_shape_cache.py  # seed _shape_cache.json (100% hit -> fast)
python scripts/ctt_v2/assemble_root.py \
  --manifest outputs/ctt_v2/strata_manifest_005_ctt_v2plus_s6reshape.json \
  --contract 005_ctt_v2plus_s6reshape --sampler-mix --code-side \
  --prereg-inline-ood misc/ctt_v2_final/PREREG_inline_ood_ops_s2a.json
# ROOT_MANIFEST.json now carries samples_sha256/samples_rows NATIVELY (Round-6 assemble_root commit);
# the 004/005 manifests on disk were stamped post-assembly (Round 4) and verified EQUAL to native
# emission by Round 6's determinism proof. No separate stamp step is needed for a fresh build.
```
(`--prereg-inline-ood` is required on DeltaAI/Taiga because the default `PREREG_INLINE_OOD` path in
`root_common.py` uses the Campus-Cluster-only `/projects` prefix; point it at the frozen file above,
sha `8e9eefc0`.) The S6 latents/cond_clean + DINO signal fields are built upstream (see `BUILD.md`).

## `_shape_cache.json` (regenerable; excluded from ship)

`root/_shape_cache.json` is a **local build cache** the assembler uses to avoid re-`torch.load`-ing every
latent to read its `(F,H,W)`. Keys are `f"{realpath}|{st_size}|{int(st_mtime)}"` → `[F,H,W]`; on a 100% hit
the assembler does **not** rewrite it. It is **regenerable** (`misc/2026-08-30_s6_reshape/r4/seed_shape_cache.py`,
or a live crawl) and is **NOT shipped** to eps — the eps root omits it (verified in the eps listing). It is
not part of the dataset contract; `samples_sha256` / `ROOT_MANIFEST` are.

## eps ship (shipped 2026-08-30, Round 5)

The 005 root, its S6 tensors, and its signals live on eps (`fal-h100`, `/storage/ozgur/dino_signal/`) in
**NEW sibling dirs** beside 004's — 004 and the live A0-004 run are untouched:

| eps path | contents | count / sha |
|---|---|---|
| `.../ctt_v2/datasets/ctt_v2/encodes/EFFECTDATA_r832/{latents,cond_clean}` | S6 r832 tensors | 28,644 + 28,644, 0 `.tmp`; ROSTER `c66c6477` |
| `datasets/ctt_v2plus_s6reshape_mix/` | the code-side root | `samples.jsonl` `048d1ef4`, `ROOT_MANIFEST` `ea824bf1`; **no `_shape_cache.json`**; `_src` → `/storage/ozgur/dino_signal/srcroot` (== 004's) |
| `signals_005/feat` | 44-ch DINO fields + norm + pca | 47,667 (S6__ 28,644 real; 19,023 non-S6 **hardlinked** to 004's `signals/feat`); NORM_dino_v4 `db47be88`, pca `4d59539b` |
| `campaign/configs_005/` | rendered eps configs (no `/taiga`) | PINS byte-identical to Taiga `configs_005/PINS.env` |
| `campaign/env.sh` | `SIGNALS` knob | `export SIGNALS="${SIGNALS:-$DS/signals}"` (default = 004; set `SIGNALS=$DS/signals_005` for 005) |

**eps verification (Round 5, all PASS):** full S6 sha manifest `sha256sum -c` 0 FAILED; root `sha256sum -c` 0
FAILED; existence 0 missing of 690,735 paths; `tensor_smoke.py` 7 (stratum,shape) groups PASS; `assert_pins` ×5
(sha `048d1ef4` rows 138147 norm `db47be88`); `verify_signal_store` ×5 v4 PASS; `bin/01_preflight.sh`
`CONFIGS=configs_005` → `01 OK` (fsck S6 81,779 / 0 miss). Ship with rsync temp+rename (never edit in place;
never `--delete`). See `misc/2026-08-30_s6_reshape/DOSSIER.md` Round 5 + `r5/eps_preflight_005.log`.

The S6 latents/cond_clean + DINO signal fields are built upstream (see `BUILD.md`).
