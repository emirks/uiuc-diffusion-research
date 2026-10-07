#!/usr/bin/env bash
# Rebuild the ARCHIVED v1 arm-comparison page exactly as it was built before 2026-09-22.
#
# v1 locates eval_ladder/ from its own file location (HERE.parent) and imports run_eval, prompts,
# report_full and encode_conditioning from there, so it only runs from eval_ladder/viewer/.
# This script copies the two archived files back under their original names for the duration of
# the build and removes them afterwards, so the archived copies stay byte-identical and the live
# viewer directory keeps only the v2 builder.
#
# Usage (from anywhere):  bash eval_ladder/viewer/archive/neutral_effect_v1_2026-09-22/run_v1.sh [--out PATH]
#   default --out is outputs/reports/iclora_neutral_effect/index.html (the archived page's location).
#   PY=<python> overrides the interpreter (default: $LAB/envs-aarch64/ltx2/bin/python).
set -euo pipefail
ARCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWER="$(cd "$ARCH/../.." && pwd)"           # eval_ladder/viewer
REPO="$(cd "$VIEWER/../.." && pwd)"           # diffusion-research
PY="${PY:-$REPO/../envs-aarch64/ltx2/bin/python}"
for f in build_neutral_effect.py template_neutral_effect.html; do
  if [ -e "$VIEWER/$f" ]; then echo "refusing: $VIEWER/$f already exists (v1 must not overwrite a live file)" >&2; exit 1; fi
done
cleanup() { rm -f "$VIEWER/build_neutral_effect.py" "$VIEWER/template_neutral_effect.html"; }
trap cleanup EXIT
cp "$ARCH/build_neutral_effect.py" "$ARCH/template_neutral_effect.html" "$VIEWER/"
cd "$REPO"
exec_rc=0
"$PY" eval_ladder/viewer/build_neutral_effect.py "$@" || exec_rc=$?
exit $exec_rc
