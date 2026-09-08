# Collapse-to-the-endpoint-line probe (base LTX-2, no adapter)

A controlled probe of "collapse to the endpoint line". For 40 prompts (30 high-drama scene
changes + 10 in-place controls) on a real 9-frame START clip:

- **R1** — start anchor only + full prompt (start caption + change clause + end caption). Seeds 42, 43, 44.
- **R2** — start anchor + END anchor cut from R1's own output (its last 9 frames), same full prompt, same seed.
- **R3** — same anchors as R2, but the NEUTRAL prompt (start caption + end caption, no change clause), same seed.

R1→R2 differ only in the end anchor; R2→R3 differ only in the text. Middles are then scored for
closeness to a frame-wise blend of the two anchors with the existing null-family instrument
(`misc/2026-09-08_collapse_remeasure/instrument.py`).

## Generation contract (verified against `eval_ladder/run_gen.py`)

- **Arm:** `base_cond_neutral` (a `kind: base`, no-adapter arm).
- **Verbatim prompt:** `run_gen.build_sample()` returns
  `ValidationSample(prompt=row["prompt"], conditions=conds)` — the row's `prompt` field is used
  **verbatim**; nothing is stripped or re-rendered at generation time. (The `base_cond_*` prompt
  rules live in `build_registry.py`, which builds the FROZEN `registry.jsonl`; an
  `--extra-registry` row carries its own final prompt.) So each row's `prompt` is set to the exact
  `full_prompt` (R1, R2) or `neutral_prompt` (R3), and `preflight.py` proves it by calling the real
  `build_sample`.
- **Conditioning** is a pure function of the row: `conditioning != "none"` attaches the prefix
  window `conds/<endpoint>_start9.mp4` (9 px); `sided == "two"` additionally attaches
  `conds/<endpoint>_end9.mp4` (9 px cut, 8 consumed). `use_reference: false`; no `reference` field
  (we never run `run_eval` on these).
- **New clip ids need no generator change:** windows are resolved by clip NAME only, so R2/R3 use a
  new endpoint id `probe_<prompt_id>_s<seed>` whose `_start9.mp4` is a byte copy of the real start
  window and whose `_end9.mp4` is cut from the R1 output.
- **Recipe** (from `store/gens/005_base_cond/02_neutral__dai/meta.v1.yaml` / `arms.yaml`): steps 30,
  guidance 4.0, stg 1.0, 480x640x121, prefix 9f / suffix 8f, no reference.
- **Output path:** `<out-root>/base_cond_neutral/<item_id>__s<seed>.mp4`.

## Files

| file | purpose |
|---|---|
| `_probe_common.py` | shared row schema, item-id scheme, paths, and the generator prompt-render proof |
| `build_r1.py` | `prompts.jsonl` -> `reg/r1.jsonl` (40 R1 rows; sided one; prompt = full_prompt) |
| `splice_r1.py` | after R1: cut end anchors from R1 outputs, write `reg/r2_s<seed>.jsonl`, `reg/r3_s<seed>.jsonl`, `reg/splice_manifest.csv` |
| `preflight.py` | CPU gate: windows exist, rendered prompt == full/neutral (via `build_sample`), banned words absent, sbatch paths exist; `--post-r1` validates splice outputs |
| `score_probe.py` | scores R1/R2/R3 with the certified instrument; `results/{per_clip.csv,paired.csv,TABLES.md,fig_probe.png}`; `--smoke` for a stand-in dry run |
| `job_r1.sbatch` | array = seeds 42..44, generates R1 |
| `job_r2r3.sbatch` | array = seeds; per seed runs splice then R2 then R3 |
| `prompts/prompts.jsonl` | the 40 prompts (written by the prompt-writer agent) |
| `prompts/_example.jsonl` | 2-row example for development / the score smoke test |

## Launch sequence

```bash
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-08_collapse_probe
source /taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/activate
export OPENBLAS_NUM_THREADS=2

# 1) build R1 registry and gate it
python build_r1.py
python preflight.py                       # must print ALL CHECKS PASSED

# 2) generate R1 (seeds 42,43,44)
R1=$(sbatch --parsable job_r1.sbatch)

# 3) generate R2+R3 after R1 succeeds (splice runs inside the job, per seed)
sbatch --dependency=afterok:$R1 job_r2r3.sbatch

# 4) after R2/R3 finish, gate the splice and score
python preflight.py --post-r1             # windows, 121-frame checks, R2/R3 render == full/neutral
python score_probe.py                     # -> results/
```

`--dependency=afterok:$R1` holds `job_r2r3` until **every** R1 array task exits 0. Both jobs are
resumable: `run_gen` skips outputs that already exist, and `splice_r1.py` / `cut_windows` are
idempotent. Account is `bgjg-dtai-gh` (swap to `bhwp-dtai-gh` by FairShare per the DeltaAI
throughput skill).

## Analysis-time selection guard (do NOT filter at generation)

Every prompt x seed is generated. At analysis, keep a pair only if its **R1 realized scene change**
is large enough to make "collapse to the line" a meaningful question — i.e. threshold on
`realized_dino` (R1 DINOv2-base CLS distance between frame 8 and the last frame). A sensible cut is
the low-tier (in-place control) distribution: report the full paired table, then re-report the
high-tier deltas restricted to `realized_dino >= <threshold>` (e.g. the low-tier 95th percentile).
The guard is applied in the CSVs, never by dropping generations.

## Known caveats

- **Synthetic end anchor.** R2/R3's end anchor is R1's own last 9 frames after a VAE round-trip, not
  a real target clip — so R2/R3 ask "given the endpoint you produced, does re-conditioning on it (and
  removing the text) pull the middle onto the line?", not "given a ground-truth endpoint".
- **Same-seed sharing.** R1, R2, R3 for a prompt share one seed, so the noise draw is held fixed
  across the three; differences are attributable to the changed factor (end anchor R1→R2, text
  R2→R3), not to noise, but the three are not independent samples.
- **Start anchor identity.** R2/R3 `_start9.mp4` is a byte copy of the real endpoint start window, so
  R1 and R2/R3 share the identical start anchor (verified by sha in `splice_manifest.csv`).
