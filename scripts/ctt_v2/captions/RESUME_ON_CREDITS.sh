#!/usr/bin/env bash
# =============================================================================
# RESUME ON CREDITS — CTT v2 caption pipeline
#
# Everything below is BLOCKED until the Gemini prepayment credits are topped up.
# Verified 2026-07-28 ~00:30: gemini-3.6-flash AND gemini-3.5-flash both return
#   HTTP 429  RESOURCE_EXHAUSTED  "Your prepayment credits are depleted."
# This is BILLING exhaustion, not a rate limit — backoff and lower concurrency
# cannot get through it. Only the owner can top it up.
#
# Budget context for the morning: the credits were spent on tonight's measurement
# work, not on a leak or runaway loop — ~2,300 successful flash calls for the M3
# pilot (2 x 400 descriptions + 2 x 400 Layer-2 audits + 391 mismatch control +
# 460 gate-5 participial calls) plus ~1,500 from a parallel agent's S4
# adjudication. Both returned decisive results.
#
# STEP 0 — confirm credits are back with ONE cheap call before spending anything:
#   source $LAB/secrets/gemini_transition.env
#   curl -s -o /dev/null -w '%{http_code}\n' \
#     -H "x-goog-api-key: $GEMINI_API_KEY" -H 'Content-Type: application/json' \
#     -d '{"contents":[{"parts":[{"text":"hi"}]}]}' \
#     "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
#   # 200 => proceed.  429 => still depleted, STOP.
#
# Then run this script step by step (it does NOT auto-run; each step is guarded).
# =============================================================================
set -euo pipefail

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
WT=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
CAP=$WT/scripts/ctt_v2/captions
OUT=$WT/outputs/ctt_v2/captions
PY=$LAB/envs/diffusion/bin/python            # generation: requests, no sklearn
PYSK=$LAB/envs/nichescout/bin/python         # gates: sklearn 1.7.2

source $LAB/secrets/gemini_transition.env    # never echo $GEMINI_API_KEY
cd "$CAP"

step() { echo; echo "=============== $* ==============="; }

# -----------------------------------------------------------------------------
step "STEP 1 — ROUND 3 (v3): 400-description pilot   [~25 s]"
# v3 = v2 + the be-verb fix, A-role only. Consumes A4's 3rd and LAST round.
# Pre-committed: v3 does NOT need to reach 0.65 and will not. Report the number.
# -----------------------------------------------------------------------------
: <<'RUN1'
$PY generate_descriptions.py --sample-per-cell 100 --seed 42 --workers 120 \
    --prompt-variant v3 --out $OUT/pilot_m3_round3
RUN1

# -----------------------------------------------------------------------------
step "STEP 2 — score round 3 against the RE-PINNED bars   [~50 s]"
#   8a corpus-vs-new   <= 0.73   HARD (drift guard)
#   8b synth-vs-humanvid <= 0.60 HARD (load-bearing, stratum-internal blindness)
#   8c original <= 0.65          RECORD-ONLY (superseded, never quietly replaced)
#   plus all 11 other gates must PASS.
# Round 2 reference: 8a 0.7066 | 8b 0.5518 | all 12 PASS, HARD failures none.
# -----------------------------------------------------------------------------
: <<'RUN2'
$PYSK gate_battery.py --store $OUT/pilot_m3_round3 \
    --out $CAP/pilot_m3/round3/gate_report.json --llm-participial
RUN2

# -----------------------------------------------------------------------------
step "STEP 3 — Layer-2 leak audit of the 139 CERTIFIED corpus captions   [~20 s]"
# Needs ffmpeg job 9686559 to have landed:
#   $WT/data/processed/corpus_anchors/corpus_anchors_index.json
# ANY hit escalates to the OWNER. NEVER silently edit a certified caption.
# -----------------------------------------------------------------------------
: <<'RUN3'
test -f $WT/data/processed/corpus_anchors/corpus_anchors_index.json \
  || { echo "corpus anchors not built yet (job 9686559)"; exit 1; }
$PY audit_corpus_139.py --out $CAP/pilot_m3/corpus_139_layer2_audit.json   # TO WRITE
RUN3

# -----------------------------------------------------------------------------
step "STEP 4 — MASS GENERATION for the clips the pinned grids actually use"
# A4 Q7.3: caption only clips actually used. Build the pair list from the pinned
# grids FIRST (PLAN_S2_UNION.json pairs -> A-role for every A endpoint, B-role for
# every B endpoint; plus S1 once its grid is pinned).
#
# HONOR THE ROLE-SCOPED EXCLUSION recorded in
#   data/processed/ctt_v2_strata/POOL_DROPS_M3_ADJUDICATION.json
#   -> openvid_T1MiFx98l3g_0_50to156 : NO A-ROLE DESCRIPTION (blank A-anchor,
#      std 0.79). Its B-role is legitimate and is used by 10 rendered S2b clips.
#
# Measured cost: ~16 descriptions/s end-to-end at 120-way concurrency including
# the 100% Layer-2 audit  =>  ~5,600 descriptions in ~6 MINUTES.
# -----------------------------------------------------------------------------
: <<'RUN4'
$PY build_mass_pair_list.py \
    --plan $WT/experiments/exp_082_s2_humanvid/PLAN_S2_UNION.json \
    --exclusions $WT/data/processed/ctt_v2_strata/POOL_DROPS_M3_ADJUDICATION.json \
    --out $OUT/mass_pairs.json                                             # TO WRITE
$PY generate_descriptions.py --pairs $OUT/mass_pairs.json --seed 42 --workers 120 \
    --prompt-variant v3 --out $OUT/mass_store
RUN4

# -----------------------------------------------------------------------------
step "STEP 5 — gate battery on the MASS store   [~1-2 min]"
# Same re-pinned bars. 8a > 0.73 => STOP: cannot be the known fingerprint, so it
# means mixed prompts / wrong round's store / contamination.
# -----------------------------------------------------------------------------
: <<'RUN5'
$PYSK gate_battery.py --store $OUT/mass_store \
    --out $CAP/mass_store_gate_report.json --llm-participial
RUN5

# -----------------------------------------------------------------------------
step "STEP 6 — auditor mismatch CONTROL on the mass store   [~3 min]"
# Validates the Layer-2 instrument on the real store. Round-1 reference:
# 100% (391/391) inaccurate=YES on wrong-clip pairings vs 5.75% matched.
# -----------------------------------------------------------------------------
: <<'RUN6'
$PY pilot_m3/audit_mismatch_control.py $OUT/mass_store/records.json \
    $CAP/mass_store_mismatch_control.json
RUN6

# -----------------------------------------------------------------------------
step "STEP 7 — Tier-2 review queue for the mass store   [seconds, no API]"
# Pilot rate 7-8% of descriptions => expect ~420-460 rows at 5,600.
# The queue ships with the 23 CERTIFIED-corpus flags as calibration rows
# (verdict pre-filled SCENE_CONTENT) so the reviewer calibrates before judging.
# -----------------------------------------------------------------------------
: <<'RUN7'
$PY tier2_queue.py --store $OUT/mass_store \
    --out $CAP/mass_store_tier2_queue.json --tsv $CAP/mass_store_tier2_queue.tsv
RUN7

# =============================================================================
# GOVERNANCE PINNED BY THE ROUND-9 RULING — check the store against these
# =============================================================================
#  first-pass rate (prompt-controllable failures: format + Tier-1 + leak)  >= 97%
#      round 1 measured 99.25%  |  round 2 measured 99.5%          -> PASS
#  first-pass inaccurate=YES                                        <= 8%  REVIEW
#      round 1 measured 5.75%   |  round 2 measured 7.25%
#  unresolved inaccurate in the FINAL store                          = 0   HARD
#      via regenerate-once then operator-rewrite. Expected queue
#      ~= 5,600 x 0.0575^2  ~= 19 rewrites.
#  Tier-1 hits in the final store                                    = 0   HARD
#      round 1 and round 2 both measured 0/400.
#  unresolved leak=YES                                               = 0   HARD
#      EXCEPT the new content-borne rule: a leak=YES that REPRODUCES across
#      regeneration goes to an operator view of the 9-frame window; if the
#      flagged language describes byte-pure visible content, mark
#      `leak_content_borne`, KEEP it, log it. If it describes change or onset,
#      DROP the clip. (Both pilot leaks were content-borne; neither clip is in
#      the training pool.)
#  gate 8a  <= 0.73  HARD   |   gate 8b <= 0.60  HARD   |   8c record-only
#  gate 9   AUC >= 0.80 => dump top-20; content-dominated => accept and record.
#      Rounds 1 and 2 both ACCEPT (34-35 of 40 top features are content; zero
#      style features outside gate-8's known fingerprint).
#
# STILL NOT WRITTEN (needed for steps 3 and 4):
#   audit_corpus_139.py        — Layer-2 over the certified 139 using corpus anchors
#   build_mass_pair_list.py    — pinned-grid -> (clip, role) list, honouring exclusions
#   Both are small; they were not written because they cannot be tested without
#   credits and an untested script in the runbook is worse than an explicit TODO.
# =============================================================================
echo
echo "This script is a guarded runbook: every step is inside a heredoc so nothing"
echo "runs by accident. Un-comment a step (delete its ': <<'RUNn'' and 'RUNn' lines)"
echo "or copy-paste the command. Do STEP 0 first."
