# exp_082 — S2b: the second S2 batch over the UNION endpoint bank

## Question

S2a (`exp_081_s2_stratum`, main tree) delivered 7,990 clips / 799 exact operators over a
single 291-clip *synth* endpoint bank. Every S2 clip therefore shows a shader transition
between two stock/DAVIS-style scenes, and the eval content is people. **Does the stratum
still hold when half its endpoint content is people, and does the gate accept person clips
at the same rate as synth clips?**

S2b renders a *second, disjoint* batch — 800 NEW operators under shader policy v2 — over the
merged 1,146-clip union of the synth bank and the 855 surviving HumanVid Pexels clips, with a
pre-registered bank quota inside the pairing and a pre-registered by-bank rejection audit.

Authority: `misc/ctt_v2_final/advisors/A5_SYNTHESIS_RULING_VERBATIM.md`, **RULING 4** (the S2b
bullet). Pipeline spec: `misc/ctt_v2_final/REF_s2_pipeline.md`. Pool audit: DOSSIER §6.

## Setup

| | S2a (exp_081) | **S2b (this)** |
|---|---|---|
| endpoint pool | `CONTENT_POOL.json`, 291 synth | `CONTENT_POOL_union.json`, **1,146** (291 synth + 855 humanvid) |
| shader policy | `D2_POLICY_FINAL.json` — 72 keep, **62** trainable | `D2_POLICY_V2.json` — 66 keep, **56** trainable |
| operators | 800 planned (799 delivered) | **800 NEW**, seed `20260727` (S2a used `20260725`) |
| content pairs | 333 (~24 ops each) | **800** (10 ops each) |
| bank quota | n/a | **25% synth-synth / 50% cross / 25% humanvid-humanvid**, + no bank-pure op |
| output dir | `outputs/videos/ctt_v2_s2` | `outputs/videos/ctt_v2_s2_humanvid` |
| frame cache | `endpoint_frames_s2.npz` | `endpoint_frames_s2_union.npz` |

Everything else — the exact-op definition, the four gates and their bars, endpoint
disjointness, incidence integrity (exactly 10 clips/op or the op is dropped and resampled),
the retry-by-pair-swap rider, the 10 `HOLDOUT_S2` shader families, the 120 reserved endpoint
clips — is **byte-identical to S2a**.

CPU-only throughout (EGL/llvmpipe software GL). This pipeline never touches a GPU.

## How to run

```bash
sbatch job_plan.sbatch                     # freeze PLAN_S2_UNION.json (~25 min, builds the frame cache)
MODE=smoke python render_s2.py             # 3 ops end-to-end -> smoke/SMOKE.json — must PASS first
sbatch job_render_array.sbatch             # 20-way array on `secondary`, NSHARDS MUST stay 20
python accept_s2.py --stage verify         # hard invariants + gates + overdraw -> S2_ACCEPTANCE.json
python bank_rejection_audit.py             # by-bank rejection differential -> BANK_REJECTION_AUDIT.json
python accept_s2.py --stage sheets --n 64  # blind audit media + AUDIT_KEY.json (bar: <= 3 BAD)
```

`render_s2.py` has no argparse — it is driven by `MODE` / `SHARD` / `NSHARDS` env vars plus
`config_s2.yaml`. It is **not idempotent across a plan change**: a third batch needs its own
`outputs.dir`, or the resume logic will read this one's manifests as done-state.

## Outputs

```
outputs/videos/ctt_v2_s2_humanvid/full/
  videos/      s2_{op_index:04d}_c{slot:02d}.mp4      8,000 clips
  filmstrips/  s2_{op_index:04d}_c{slot:02d}.jpg      16-frame strips
  meta/        clips_shard{NN}.jsonl                  the training manifest (full GT op params)
               ops_shard{NN}.jsonl                    one row per finalised op + its rejects
               summary_shard{NN}.json
  audit_sheets/, AUDIT_KEY.json                       blind n=64 audit media
```

Manifests in this directory: `PLAN_S2_UNION.json` (the frozen grid, with per-pair bank labels
and the realised quota), `HOLDOUT_S2_UNION.json`, `S2_ACCEPTANCE.json`,
`BANK_REJECTION_AUDIT.json`.
