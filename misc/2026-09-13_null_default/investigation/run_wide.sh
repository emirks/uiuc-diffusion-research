#!/bin/bash
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
source /taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/activate
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec python misc/2026-09-13_null_default/investigation/wide_pix.py
