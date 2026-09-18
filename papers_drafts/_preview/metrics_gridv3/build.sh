#!/usr/bin/env bash
# Rebuild preview.pdf for the grid v3 metric tables (Op-5).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PAPER="/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/papers_drafts/ctt_iclr2027"
export PATH="/taiga/illinois/eng/cs/jrehg/users/emirkisa/texlive/bin/aarch64-linux:$PATH"
export TEXINPUTS=".:$PAPER:$PAPER//:"
export BIBINPUTS="$TEXINPUTS"; export BSTINPUTS="$TEXINPUTS"
cd "$HERE"
latexmk -pdf -interaction=nonstopmode -file-line-error preview.tex
latexmk -c preview.tex >/dev/null 2>&1 || true   # drop aux, keep pdf
echo "OK -> $HERE/preview.pdf"
