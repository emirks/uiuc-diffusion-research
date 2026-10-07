# Archived: v1 arm-comparison viewer builder (`iclora_neutral_effect`)

Archived 2026-09-22 when the v2 builder (`eval_ladder/viewer/build_neutral_effect_v2.py`, page
`outputs/reports/iclora_neutral_effect_v2/{index.html,data.js}`, registry slug
`iclora_neutral_effect_v2`) became the registered, featured arm-comparison viewer.

The two files here are the v1 builder and template exactly as last used (their final state includes
the 2026-09-21 additions: DCG guidance-weight sweep arms, TEG two-endpoint baselines, renamed
categories, the sidedness filter). They were moved here with `git mv`; bytes unchanged:

```
c8cc6d81e744e74fa8a5377c0713348498ee580569f0a81c443a543d1707d7e2  build_neutral_effect.py
4edc7db2696303f9ca0fa8cfb914e160c415715d01f271c0c79bb3a554471403  template_neutral_effect.html
```

The v1 PAGE it last produced is kept at `outputs/reports/iclora_neutral_effect/index.html`
(one fused 35 MB file; sha256 a65c126e…) and stays openable from the dashboard's
"Earlier versions & archive" table (registry entry `iclora_neutral_effect`, `archived`).
Its embedded JSON is byte-identical to the v2 page's `data.js` payload (35,455,253 B,
sha256 870112c5…), measured 2026-09-22; see `../../NOTES_v2.md`.

## Why keep it

Owner instruction: "the current version should be kept for any case." v2 is a copy of v1 plus
three changes (data.js split, `--mode`, per-arm disk cache); the metric/card code is the same.
If v2 ever misbehaves, v1 is the reference build.

## How to run it again

v1 resolves `eval_ladder/` from its own location and imports sibling modules from there, so it
cannot run from this folder. Use the wrapper, which copies the two files back under their original
names only for the duration of the build (and refuses if a live file with that name exists):

```
bash eval_ladder/viewer/archive/neutral_effect_v1_2026-09-22/run_v1.sh            # default --out
bash eval_ladder/viewer/archive/neutral_effect_v1_2026-09-22/run_v1.sh --out outputs/reports/_scratch/v1.html
```

Expect ~10 minutes on a login node (no cache in v1). Do not edit the files in this folder; if a
change is needed, make it in the v2 builder.
