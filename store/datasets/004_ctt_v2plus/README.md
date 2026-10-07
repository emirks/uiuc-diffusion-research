# 004_ctt_v2plus — dataset entry point

**cttv2 + EffectData (S6), CODE-SIDE form.** A **114,215-pair** LTX-2 transition-training set across
six strata, portable across devices via one `_src` symlink. This directory is **self-sufficient** —
everything needed to understand, use, move, rebuild, and verify the dataset is here or reachable
through the links below.

## Quick facts

| | |
|---|---|
| pairs | **114,215** (S0 385 / S1 3,675 / S2a 22,731 / S2b 23,577 / S4 6,000 / S6 57,847) |
| mix (sampler weights) | S0 12 / S1 4.8 / S2a 27.10 / S2b 28.10 / S4 8 / S6 20 — sums to 100 |
| form | **code-side** — `samples.jsonl` with root-relative paths; no per-row symlink trees |
| `samples_sha256` | `5a73eb3c24e274d021e8f47a32a0bfa1ed4f0051395f6cabaa77f768040b380e` (pin at run start) |
| signal + norm | 44-ch DINO operator signal, `NORM_dino_v3` → **`../003_dino_signals/`** |
| version | `3.0.0-ctt_v2plus-codeside` |

## Read this — in order

| doc | what it is | when to read |
|---|---|---|
| **[`CODESIDE_FORMAT.md`](CODESIDE_FORMAT.md)** | **the primary doc** — layout, row schema, strata, S6 same-shape pairing, `_src` portability + per-device bring-up, rebuild command, verification battery, open residuals | to use / move / rebuild the dataset |
| [`meta.yaml`](meta.yaml) | the store registry record. **Append-only** — read the `correction_2026_08_29` block at the bottom; it supersedes the top lines | for the canonical registry entry |
| [`BUILD.md`](BUILD.md) | the **S6 source build** (selection · native shapes · encode · captions · conditions). §1–7 authoritative; **§8 superseded** by the code-side rebuild | to trace where the S6 clips came from |
| `root/README.md` + `root/ROOT_MANIFEST.json` | the physical root's bring-up card + machine-readable **ground-truth** numbers/sha/shapes (via the `root` symlink) | for exact counts / provenance |
| **`../003_dino_signals/meta.yaml`** | the 44-ch DINO signal + `NORM_dino_v3` (a sibling store entry — not duplicated here) | for the signal & normalization |

## Layout of this store entry

```
store/datasets/004_ctt_v2plus/
├── README.md            ← you are here (entry point / index)
├── CODESIDE_FORMAT.md   the primary format + usage + verify doc
├── BUILD.md             the S6 source build (§1–7 authority; §8 superseded)
├── meta.yaml            registry record (read the correction_2026_08_29 block)
└── root ──────────────► ../../../outputs/ctt_v2/roots/ctt_v2plus_mix
                         (samples.jsonl · mix.json · ROOT_MANIFEST.json · CAPTIONS.json
                          · _mask_store/ · _src · VERSION · README.md)
```

`CODESIDE_FORMAT.md` and `BUILD.md` are the canonical copies; symlink stubs remain at their old
`misc/2026-08-28_effectdata_s6/` paths so older references still resolve.

## Train

```yaml
data:
  sample_list: <device>/ctt_v2plus_mix/samples.jsonl   # SampleListDataset does the rest
```
Bring the root up on a new device with the ritual in `CODESIDE_FORMAT.md §4`; assert the
`samples_sha256` above at run start.

## Two open residuals (not yet closed)

Recorded in full in `CODESIDE_FORMAT.md §8`:
1. **eps stale-root kill** — a stale 138,625-row, S1-less, mis-paired-S6, symlink-form
   `samples.jsonl` is still live on eps; no run may launch against 004 from eps until it is killed
   and re-shipped code-side (the eps agent's job).
2. **sha-pin enforcement** — documented (§4.6) but not yet wired; the first consumer (the paired-arm
   gate, 004 vs 002) must assert the sha before its runs count.
