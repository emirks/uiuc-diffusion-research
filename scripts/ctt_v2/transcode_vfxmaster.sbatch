#!/bin/bash
# ctt_v2 — VFXMaster ships as MPEG-4 Part 2 (DivX-era), which no browser can decode.
# Transcode the extracted tree to H.264 (+faststart, aac) into a parallel tree
#   extracted_h264/data/<class>/<name>.mp4
# so the dataset viewer can stream it. Idempotent: existing outputs are skipped.
#
#SBATCH --job-name=ctt_vfx_h264
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --cpus-per-task=24
#SBATCH --mem=16g
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/raw/_dl_logs/%x_%j.out

set -eo pipefail
FF=/u/emirkisa/.local/bin/ffmpeg
SRC=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/raw/vfxmaster/extracted/data
DST=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/raw/vfxmaster/extracted_h264/data

echo "[h264] host=$(hostname) start=$(date '+%F %T')"
cd "$SRC"
find . -name '*.mp4' | sed 's|^\./||' > /tmp/vfx_list.txt
total=$(wc -l < /tmp/vfx_list.txt); echo "[h264] $total files"

transcode() {
  rel="$1"
  out="$DST/$rel"
  [ -s "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  $FF -hide_banner -loglevel error -y -i "$SRC/$rel" \
      -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
      -c:a aac -movflags +faststart -threads 2 "$out.tmp.mp4" \
    && mv "$out.tmp.mp4" "$out" \
    || { rm -f "$out.tmp.mp4"; echo "FAIL $rel"; }
}
export -f transcode; export FF SRC DST

xargs -a /tmp/vfx_list.txt -d '\n' -P 12 -I{} bash -c 'transcode "$@"' _ {}

done_n=$(find "$DST" -name '*.mp4' | wc -l)
echo "[h264] done: $done_n/$total at $(date '+%F %T')"
[ "$done_n" -ge $((total - 20)) ] && echo "[h264] TRANSCODE-COMPLETE" || echo "[h264] SHORTFALL — rerun to fill gaps"
