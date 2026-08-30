# ctt_v2plus — the CODE-SIDE root format (2026-08-29)

Authority for the physical form of `outputs/ctt_v2/roots/ctt_v2plus_mix` after the 2026-08-29
rework (S1 restored · S6 re-paired same-shape · symlink trees → code-side). This doc lives **in the
store** at `store/datasets/004_ctt_v2plus/` alongside `meta.yaml` (the registry entry), `BUILD.md`
(the S6 source build), and `README.md` (the entry point that routes here). A symlink remains at the
old `misc/2026-08-28_effectdata_s6/CODESIDE_FORMAT.md` path. fable-advisor GO-WITH-CHANGES
2026-08-29, invariants 1–8.

---

## 1 · What changed and why

The old root materialized **one symlink per training slot** — 114,215 rows × 5 trees ≈ **571k
symlinks** — so the realized dataset was "countable off disk." That form is slow to build, heavy,
and — critically — **not portable**: the symlinks (and any absolute paths) hardcode one machine's
mount, so a different device with a different store root cannot read it without re-materializing.

The trainer never needed the trees. `SampleListDataset` reads a `samples.jsonl` and, per row,
resolves `dataset_root / paths[tree]` and `torch.load`s the tensor. So the root is now **code-side**:

- **`samples.jsonl` carries each row's paths directly** — no per-row symlink trees.
- **Paths are ROOT-RELATIVE and portable**: source tensors as `_src/<repo-relative>/…`, masks as
  `_mask_store/<name>`. Nothing hardcodes a machine. Paths are **realpath-derived** (symlink layers
  like `exp_064→exp_058` and the moved-in-encodes compat symlink `outputs/ctt_v2/encodes→datasets/ctt_v2`
  are collapsed, so a device needs only real files, not the source symlink graph).

The mix was never in the trees to begin with — 004 has been **sampler-mix** since assembly (the
mix is a number in `mix.json` realized by `StratifiedEpochSampler`, not baked into the data). So
going code-side gives up only a redundant off-disk copy of the base-pair set; the anti-drift job
moves to a JSONL-vs-independent-sources verification battery (§6) and a pinned sha (§4).

---

## 2 · Layout — the entire dataset

```
outputs/ctt_v2/roots/ctt_v2plus_mix/
├── samples.jsonl        114,215 rows — the dataset. One row per training pair. (§3)
├── mix.json             per-stratum sampler weights (S0 12 / S1 4.8 / S2 55.2 / S4 8 / S6 20)
├── ROOT_MANIFEST.json   provenance: form=code_side, pairing rule, S6 per-shape census + 2,378 drops,
│                        samples_sha256, samples_rows
├── CAPTIONS.json        content-addressed caption text (sha16 → text); captions.json → symlink
├── VERSION              3.0.0-ctt_v2plus-codeside
├── _mask_store/         7 REAL mask tensors — the ONLY materialized source artifact (masks are
│                        generated, not sourced). Travels with the dataset.
└── _src ──────────────► the repo (../../../.. on Taiga) — the ONE re-pointable bridge (§4)
```

**No `latents/ reference_latents/ cond_clean_latents/ conditions/ masks/` per-row trees.**

---

## 3 · Row schema (`samples.jsonl`)

```json
{
  "id": "S6/accretion_eye_vortices/<target>__ref_<reference>",
  "stratum": "S6", "group": "...", "group_slug": "...",
  "target": "<stem>", "reference": "<stem>", "sided": "one",
  "caption_key": "<sha16>",                 // content-address into CAPTIONS.json
  "shape": [11, 22, 33],                    // the TARGET's latent (F, H, W)
  "paths": {
    "latents":            "_src/datasets/ctt_v2/encodes/EFFECTDATA/latents/<target>.pt",
    "reference_latents":  "_src/datasets/ctt_v2/encodes/EFFECTDATA/latents/<reference>.pt",
    "cond_clean_latents": "_src/datasets/ctt_v2/encodes/EFFECTDATA/cond_clean/<target>.pt",
    "conditions":         "_src/datasets/ctt_v2/conditions/by_caption/<sha16>.pt",
    "masks":              "_mask_store/f11_h22_w33_p1_onesided.pt"
  },
  "endpoints": [...], "caption_sources": [["<stem>", "A"]]
}
```

Every path is either `_src/…` (relative-under-root, no absolute, no `..`) or `_mask_store/…`.
The trainer config points at this file with **one line**: `data.sample_list: <root>/samples.jsonl`.

### Strata (114,215 pairs)

| stratum | pairs | clips | sided | shape(s) |
|---|--:|--:|---|---|
| S0 | 385 | 139 | mixed | (16,20,15) |
| S1 | 3,675 | 1,225 | mostly one | (16,20,15) |
| S2a | 22,731 | 7,577 | two | (16,20,15) |
| S2b | 23,577 | 7,859 | two | (16,20,15) |
| S4 | 6,000 | 2,000 | one | (5,14,26) |
| **S6** | **57,847** | **26,266** | one | (11,{22,33,39}²) — 4 native shapes |
| **total** | **114,215** | | | |

### S6 same-shape pairing + the 2,378 dropped clips

S6 has 4 native resolutions, and a reference is a **different-subject** demonstration of the same
effect. Pairing is restricted to **same shape** (target and reference share the grid) so
reference-spatially-aligned conditioning works: an effect's clips are sub-grouped by shape and
ring-paired within each `(effect, shape)`. **2,378 clips are the lone subject of their shape for
their effect → dropped** (no same-shape same-effect partner). Every remaining S6 pair is same-shape
by construction.

> **The 2,378 dropped clips are NOT trained.** They belong to well-represented effects (4–13
> subjects each) and remain in the signal cache and in EffectData — i.e. **free unseen-subject eval
> material** (a trained effect on an untrained subject). Do not describe them as trained.

---

## 4 · Portability — `_src` + the sha pin

The dataset is a small, device-agnostic unit: `samples.jsonl` (~145 MB) + `mix.json` +
`ROOT_MANIFEST.json` + `CAPTIONS.json` + `_mask_store/` (7 files, ~KB) + the `_src` symlink.
**`samples.jsonl` hardcodes nothing about Taiga** — the only per-device knob is `_src`.

- `_src` → the repo (the tensors live under `<repo>/datasets/ctt_v2/encodes/…`,
  `<repo>/experiments/exp_058…/dataset/…`, `<repo>/eval_ladder/dataset/…`).
- **sha pin:** `ROOT_MANIFEST.json:samples_sha256 = 5a73eb3c24e274d021e8f47a32a0bfa1ed4f0051395f6cabaa77f768040b380e`.
  This is now the primary cross-device anti-drift mechanism — assert it at run start.

### Per-device bring-up ritual (run on each training device)

```bash
# 1. copy the small root to the device (samples.jsonl + mix + manifest + captions + _mask_store)
rsync -a <src>/ctt_v2plus_mix/  <device>/ctt_v2plus_mix/     # (exclude nothing; it's small)

# 2. point _src at THIS device's repo (where the tensors live in the mirrored layout)
ln -sfn <device_repo>  <device>/ctt_v2plus_mix/_src

# 3. assert the JSONL is the intended one
test "$(sha256sum <device>/ctt_v2plus_mix/samples.jsonl | cut -d' ' -f1)" \
   = 5a73eb3c24e274d021e8f47a32a0bfa1ed4f0051395f6cabaa77f768040b380e

# 4. let the trainer stat everything (fails closed on any missing path)
#    SampleListDataset(..., verify_files=True) runs _verify_files() automatically.

# 5. sampled tensor-load smoke: ~200 rows spanning ALL strata and ALL 4 S6 shapes —
#    torch.load each path, assert latent (F,H,W) == row["shape"] for the target and the
#    reference keyed to its OWN shape (S6 target and reference can differ in orientation only
#    across DROPPED clips; kept pairs are same-shape by construction).

# 6. pin the sha in the trainer config/launch and assert it at run start, then train:
#    data.sample_list: <device>/ctt_v2plus_mix/samples.jsonl
```

The device must present the tensors under `_src` in the **same relative layout**
(`datasets/ctt_v2/encodes/…`, `experiments/exp_058…/dataset/…`, `eval_ladder/dataset/…`). If a
device ships a flattened layout instead, add per-source `_src_*` symlinks or a path-map — but the
clean default is "mirror the sub-tree, point `_src` at it."

---

## 5 · Build / regenerate

```bash
source $LAB/envs-aarch64/activate
python scripts/ctt_v2/assemble_root.py \
  --manifest outputs/ctt_v2/strata_manifest_003_ctt_v2plus.json \
  --contract 003_ctt_v2plus --sampler-mix \
  --prereg-inline-ood misc/ctt_v2_final/PREREG_inline_ood_ops_s2a.json \
  --code-side
python misc/2026-08-24_flow_signal_conditioning/armA/verify_code_side.py   # §6 battery, must PASS
```

`--code-side` branches only at the materialize step; pairing/gates/plan-only run the **identical**
code path as the physical form (single pairing authority). `002_ctt_v2` stays physical — code-side
is a per-dataset choice, recorded as `form: code_side` in `ROOT_MANIFEST.json`.

---

## 6 · Verification battery (replaces "countable off disk")

`misc/2026-08-24_flow_signal_conditioning/armA/verify_code_side.py` → `CODESIDE_VERIFY.md`.
Every check compares `samples.jsonl` against **independent** sources (inventories / ROSTER /
certified counts), never against itself:

1. per-stratum counts == inventory/ROSTER predictions (S6/S1/S0 re-derived via the ring formula;
   S2a/S2b/S4 vs certified); total 114,215; S6 26,266 targets + 2,378 drops == 28,644. **PASS**
2. 0 duplicate `(stratum,target,reference)`; S1/S6 stem-sets == predicted. **PASS**
3. shared-stub detector: each `latents`/`reference_latents` path → exactly one clip; distinct
   `conditions` per stratum == distinct `caption_keys` (content-address dedup, never a stub). **PASS**
4. path-scheme: every path `_src/…` or `_mask_store/…`, 0 absolute, 0 `..`. **PASS**
5. existence (FULL — all 95,731 distinct paths) + shape (sampled, target & reference keyed to each
   stem's OWN shape). **PASS**

Prune bar (§ invariant 6): the old 571k-symlink tree deleted in full; final root == exactly the 8
entries in §2; mask census: 7 masks, each referenced only by matching `(shape, sided)` rows, no
orphans. **PASS**

---

## 7 · EPS MIGRATION — for the eps agent (do NOT run from Taiga)

A **stale** copy of the OLD form was already shipped to eps/fal-h100 and must be actively killed —
it is the exact silent-drift artifact this rework must not leave alive. It is stale in three ways:
**138,625 rows** (not 114,215), **S1 absent**, **S6 cross-shape paired**, and **symlink-tree form**.

1. **Tombstone/delete** on eps:
   - `/storage/ozgur/dino_signal/datasets/ctt_v2plus_mix/samples.jsonl`
   - `/storage/ozgur/dino_signal/datasets/ctt_v2plus_mix/samples.deltaai.jsonl`
   Leave a `STALE_SUPERSEDED.txt` pointer: superseded 2026-08-29 by the code-side root
   (`samples_sha256 5a73eb3c…`, 114,215 rows).
2. **Bring up the new code-side root** on eps per §4 (mirror the tensors under `_src`, run
   `_verify_files`, sampled tensor-load, assert the sha).
3. **Update the DINO-signal arm configs** `misc/2026-08-27_dino_signal_training/{configs_004,eps/configs_004}/*.yaml`:
   they pin `data.sample_list → .../ctt_v2plus_mix/samples.jsonl` with the comment "138,625 rows,
   S1 absent" — repoint to the new root, correct the comment to **114,215 rows (S0/S1/S2a/S2b/S4/S6)**,
   and add the `samples_sha256` assertion at run start.
4. **Re-ship any newly-needed tensors**: S1 latents/cond_clean/conditions (1,225 clips) were not in
   the earlier eps shipment (S1 was absent) — they must be shipped for the S1 rows to resolve.

The DINO 44-ch signal (norm v3) is a separate input joined by `(stratum, stem)`; it is unchanged by
the pairing/form rework (only S1 rejoins the norm; the 2,378 dropped clips' feat stays cached-but-
unconsumed). See `store/datasets/003_dino_signals/meta.yaml`.

## 8 · OPEN RESIDUALS (owner-visible; fable-advisor 2026-08-29, at close)

The 004 rebuild + norm v3 are complete, verified, and committed (`81437f2`); consumers still empty.
Two items remain OPEN and are the owner's / a downstream agent's to close — record here so no run
launches against a stale state:

1. **eps stale-root kill + config sha-repoint is documented, not executed** (§7). Until the eps agent
   runs it, a stale **138,625-row, S1-less, mis-paired-S6, symlink-form** `samples.jsonl` is live at
   `eps:/storage/ozgur/dino_signal/datasets/ctt_v2plus_mix/`. **No training run may launch against 004
   from eps before §7 executes.**
2. **The trainer-config sha-pin is not yet enforced anywhere** — it is a documented step (§4.6). The
   first consumer (the paired-arm gate, 004 vs 002) MUST verify `samples_sha256 == 5a73eb3c…` is present
   in the config and asserted at run start before its runs count.

Housekeeping done at close: the pre-S1 root backups were moved OUT of the shippable root to
`misc/2026-08-28_effectdata_s6/pre_s1_root_backups/`; the bring-up rsync excludes `*.bak`.
