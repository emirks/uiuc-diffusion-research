# exp_081 — ctt_v2 masked retrain (T6(a) one-way reference attention + diversity mix)

## Question

Does removing the reference-block leakage path (one-way attention) **together with** a ~43×
increase in training-task diversity (26 → ~1,126 operators) move the two CROSS cells of the
frozen v4 instrument, at same-cell parity?

Pre-registered pass bars, fixed from the D0 baselines BEFORE any candidate was scored:

| cell | baseline (ic_gen) | bar |
|---|---|---|
| G-unseen-cross | 72.9% (proxy) | **≥ 78.9** |
| G-zs-cross | 72.8% (proxy) | **≥ 78.8** |
| G-unseen-same | 88.7% (same) | **≥ 86.7** (≤2pp regression) |
| G-zs-same | 90.8% (same) | **≥ 88.8** (≤2pp regression) |

Plus the anti-routing rule: **positive liveness on the cross cells**. A candidate that lifts
cross % without positive cross liveness is routing, not transferring, and does not pass.

## Setup

Two deltas ride this retrain, both advisor-ratified (round 6, 2026-07-25):

1. **T6(a) one-way reference attention.** The IC-LoRA reference block was fully bidirectional
   on both paths — the trainer built its `Modality` with no `attention_mask` at all, and the
   inference conditioning called the documented no-op `update_attention_mask(None)`. So the
   *noisy* target wrote into the clean reference tokens at every layer. The mask closes
   reference→noisy and leaves target→reference, reference→reference and noisy→noisy open.
   Implemented in `$LAB/LTX-2-bneck` (LOCAL ONLY) as
   `ltx_trainer/reference_attention.py`, config-gated `attention: bidirectional|one_way`,
   **default bidirectional** so every certified baseline path stays bit-identical.
2. **T6(c) training half — reference dropout p=0.1.** No new code: `probability` already is
   reference dropout, so `probability: 0.9`. It rides this retrain because it is an *enabler
   with an expiry date* — without it, the λ_ref CFG knob can never be tested on this
   checkpoint without paying for a second ~13 H100-h retrain.

**Mix (advisor Q1/Q2):** S0 + S2 + S3 only. S4 (refVFX I2V-LoRA) is **deferred, not killed** —
it adds ~4% to the operator count while importing a tempo-rewrite or an unvalidated
mixed-length trainer path into the same retrain that carries the mask. Sampling stream
weights **S0 15% / S2 ≈69% / S3 ≈16%**; 10,000 steps; the **PRIMARY checkpoint is
pre-committed as the final one (10k)** — intermediates are diagnostics only and are never
scored on the frozen cells (that would close the loop on the instrument).

**Throughput rule (pre-registered):** the one-way mask is a dense (1, 9600, 9600) tensor that
pushes SDPA off the flash path. ≤30% step-time slowdown and fits memory → ship the dense mask.
>30% or OOM → build the two-call split (target rows over all keys, reference rows over
reference keys only; 25% *fewer* attention pairs than the bidirectional baseline, keeps flash),
but it may not ride the retrain until it passes a numerical-equivalence gate against the dense
implementation.

## How to run

```bash
# throughput probes (30 steps each, identical but for `attention`) — H100-class memory REQUIRED:
# the bidirectional BASELINE OOMs on a 44 GiB L40S, so this is not a mask cost.
sbatch --job-name=ctt081_thr_bidir  --partition=secondary --account=campusclusterusers \
       --gres=gpu:H100:1 --time=01:30:00 --export=ALL,CONFIG=thr_bidir \
       experiments/exp_081_ctt_v2_masked_retrain/job_train.sbatch

# the gates behind the surgery (must be green before any retrain):
cd $LAB/LTX-2-bneck/packages/ltx-trainer
PYTHONPATH=src $LAB/LTX-2-official/.venv/bin/python -m pytest tests/ -q
```

`job_train.sbatch` always runs the private `bneck` trainer, echoes the trainer commit and the
reference attention topology into the log, and hard-fails on unequal source counts — the
trainer pairs its five data dirs by identical relative path and **silently skips** any sample
missing from one of them.

## Outputs

- `D0_baselines_v4.txt` — the frozen baseline table the bars above are derived from.
- `configs/thr_{bidir,oneway}.yaml` — throughput probes.
- Retrain config and results land here once S2/S3 delivery completes.
