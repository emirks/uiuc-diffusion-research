#!/bin/bash
#SBATCH --job-name=ep2_build
#SBATCH --partition=scavenger
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=outputs/synth_endpoints_logs/build_v2-%j.out
#SBATCH --error=outputs/synth_endpoints_logs/build_v2-%j.err

# BUILD pass: apply the QC accept/reject cascade to the v2 candidates and standardise the
# survivors to the std121 contract (480x640, 121f, 24fps, libx264 crf14) — byte-compatible with
# the round-1 bank so the two pools are interchangeable as procedural contents.
# Writes ONLY under data/processed/ctt_v2_strata/endpoints_v2/. The round-1 bank is READ-ONLY.
#
# Deliberately NOT chained with --dependency=afterok: a preempted scavenger job would leave the
# dependent in DependencyNeverSatisfied forever (that exact failure cost ~4 h in the dsx
# campaign). The watchdog sequences this instead, and re-submits whatever actually died.
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
cd $LAB/diffusion-research
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="/u/emirkisa/.local/bin:$PATH"
$LAB/envs/diffusion/bin/python data/processed/ctt_v2_strata/endpoints_v2/_work/build_v2.py
echo "=== EXIT $? ==="
