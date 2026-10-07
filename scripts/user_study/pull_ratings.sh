#!/usr/bin/env bash
# Pull the study's ratings from the Fly.io deployment into the repo (source of truth stays on the Fly volume).
#   scripts/user_study/pull_ratings.sh            # -> misc/2026-09-22_user_study/state_fly/ratings.jsonl (+ timestamped copy)
# Needs $LAB/secrets/user_study_admin.env (STUDY_ADMIN_TOKEN). Run by hand or on a loop; safe to repeat.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
URL="${STUDY_URL:-https://segue-study.fly.dev}"
OUT="$REPO/misc/2026-09-22_user_study/state_fly"
source "${LAB:?set LAB}/secrets/user_study_admin.env"
mkdir -p "$OUT/history"
tmp="$(mktemp)"
curl -sf -m 60 -H "X-Admin-Token: $STUDY_ADMIN_TOKEN" "$URL/api/export" -o "$tmp"
n=$(grep -c . "$tmp" || true)
cp "$tmp" "$OUT/ratings.jsonl"
cp "$tmp" "$OUT/history/ratings_$(date +%Y%m%d_%H%M%S).jsonl"
rm -f "$tmp"
curl -sf -m 30 -H "X-Admin-Token: $STUDY_ADMIN_TOKEN" "$URL/api/status" > "$OUT/status.json"
echo "[pull] $n rating lines -> $OUT/ratings.jsonl ($(date '+%Y-%m-%d %H:%M'))"
