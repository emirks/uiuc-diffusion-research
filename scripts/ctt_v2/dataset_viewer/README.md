# ctt_v2 dataset viewer — refVFX + VFXMaster

Browse both external corpora along their counterfactual axes, streaming videos **directly
out of the WebDataset tar shards** (nothing is extracted; the index stores byte offsets).

## Datasets covered

| tab | corpus | samples | real axes |
|---|---|---|---|
| refVFX · code | `data/raw/refvfx/data/code_based_edits` (16 tars, 375 GB) | 136,800 | effect (2,736 ops × ~50 contents), spatial family, temporal family, content* |
| refVFX · LoRA | `data/raw/refvfx/data/I2V_LoRA` (1 tar, 12 GB) | 6,995 | effect (48 triggers). Content axis **degenerate** — 0 reused inputs (measured) |
| VFXMaster | `data/raw/vfxmaster/extracted` (6.6 GB) | ~9.9k | class (241 effects × ~38 contents). Cross-effect content axis does not exist (19/8,119 hash collisions only — measured) |

*whether the code-subset content axis is real is measured by `build_index.py` (`[axis-check]`
line) from input-bytes fingerprints, not assumed.

## Run

```bash
# one-time (already done; ~5 min):
nice -n 19 python scripts/ctt_v2/dataset_viewer/build_index.py --workers 4

# serve (login node fine — pure I/O):
nice -n 19 python scripts/ctt_v2/dataset_viewer/serve.py --port 8799
```

From your laptop:

```bash
ssh -L 8799:<login-node>:8799 cc     # e.g. cc-login5 if the server runs there
# open http://localhost:8799
```

## UI

- Tabs = dataset; pills = axis. **By effect / class** = same operator × many contents
  (the counterfactual diagonal). **By content** = same base video × many operators
  (the other diagonal; only shown where it exists).
- Every card cross-links: `op · N` jumps to all N contents with that exact operator;
  `content · N ops` jumps to that base video under its N operators; spatial/temporal
  family chips jump to the coarse-family view.
- refVFX code cards show input | output side-by-side with a `mask` toggle on the output
  pane; prompts expand on click. Videos stream with seek support (Range requests into
  the tar at member offset).
