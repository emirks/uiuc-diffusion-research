# Moved — this filename was ambiguous

There were two `DATASET.md` files in this repo and they are **different documents**:

| now at | what it is |
|---|---|
| **`data/DATASET.md`** | the **CTT v2 training dataset SPEC** — the stamp target, consumed by `scripts/ctt_v2/make_stamp.py`. If you are looking for the frozen dataset definition, mix weights, holdouts, or the STAMP block, it is here. |
| **`docs/DATASET_REGISTRY.md`** | the broader **dataset registry** — strata inventory, endpoint pool, S3 state, eval stock, dataloader contracts, scoreboard, viewers. This file used to be `docs/DATASET.md`. |

The collision was a real hazard, not a cosmetic one: a status check during the
CTT v2 finalization reported "DATASET.md ✓" against `docs/DATASET.md` while the
campaign's actual spec — the one carrying the STAMP — was `data/DATASET.md`.
Renamed 2026-07-28 during branch consolidation. Nothing referenced
`docs/DATASET.md`, so no links needed updating.
