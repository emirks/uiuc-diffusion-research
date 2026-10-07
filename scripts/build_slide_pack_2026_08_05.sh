#!/usr/bin/env bash
# Build outputs/presentation/slide_pack_2026-08-05/ — per-slide media for the deck upload.
set -euo pipefail
FF=/u/emirkisa/.local/bin/ffmpeg
REPO=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
P=$REPO/outputs/presentation/slide_pack_2026-08-05
G=$REPO/store/gens/_legacy   # v2 migration 2026-08-13: legacy shims resolve the old flat ids
T=$REPO/data/processed/transitions_std121
E=$REPO/datasets/ctt_v2/encodes
DAV=$REPO/data/raw/DAVIS/JPEGImages/480p
TD=$REPO/outputs/presentation/01_task_definition/01_shadow_smoke_0__ref_1
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

cpv() { # lossless copy, muted, faststart
  [ -f "$1" ] || { echo "MISSING: $1" >&2; exit 1; }
  $FF -hide_banner -loglevel error -y -i "$1" -c:v copy -an -movflags +faststart "$2"
}
cut_head() { # first $3 frames, re-encoded
  [ -f "$1" ] || { echo "MISSING: $1" >&2; exit 1; }
  $FF -hide_banner -loglevel error -y -i "$1" -vf "select='lt(n,$3)',setpts=N/24/TB" -r 24 -frames:v "$3" -an -crf 16 -movflags +faststart "$2"
}
cut_tail() { # last $3 frames
  [ -f "$1" ] || { echo "MISSING: $1" >&2; exit 1; }
  local n off
  n=$($FF -hide_banner -i "$1" -vf select=1 -f null - 2>&1 | grep -oE 'frame=\s*[0-9]+' | tail -1 | grep -oE '[0-9]+')
  off=$((n-$3))
  $FF -hide_banner -loglevel error -y -i "$1" -vf "select='gte(n,$off)',setpts=N/24/TB" -r 24 -frames:v "$3" -an -crf 16 -movflags +faststart "$2"
}
davis_head() { # first 24 frames of a DAVIS sequence, portrait crop, 24fps
  local seq=$1 out=$2
  $FF -hide_banner -loglevel error -y -framerate 24 -pattern_type glob -i "$DAV/$seq/*.jpg" \
    -vf "select='lt(n,24)',setpts=N/24/TB,scale=-2:640,crop=480:640" -r 24 -frames:v 24 -an -crf 16 -movflags +faststart "$out"
}

mkdir -p $P/{02_coldopen,03_lerp,04_refvfx,06_dataset,07_counterfactual,12_iid,13_zeroshot}

# ---------------- 02_coldopen ----------------
cpv "$TD/endpoints/start_9f.mp4"   $P/02_coldopen/02_coldopen__start.mp4
cpv "$TD/endpoints/end_24f.mp4"    $P/02_coldopen/02_coldopen__end.mp4
cpv "$TD/reference_full.mp4"       $P/02_coldopen/02_coldopen__reference.mp4
cpv "$G/002_ctt_v2/videos/G-memo-probe__ctt_v2__shadow_smoke_0__ref_shadow_smoke_1__s42.mp4" $P/02_coldopen/02_coldopen__ctt_v2.mp4

# ---------------- 03_lerp ----------------
cpv "$G/007_base_cond_ctt/videos/G-zs-cross__ctt_v2__water_bending_3__ref_firelava_0__s42.mp4"                 $P/03_lerp/03_lerp__dissolve_1.mp4
cpv "$G/007_base_cond_ctt/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_raven_transition_0__s42.mp4" $P/03_lerp/03_lerp__dissolve_2.mp4

# ---------------- 04_refvfx ----------------
IT=G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_firelava_0
cpv "$G/004_refvfx_B/videos/${IT}__seed42.mp4"   $P/04_refvfx/04_refvfx__refvfx_promptA.mp4
cpv "$G/003_refvfx_A/videos/${IT}__seed43.mp4"   $P/04_refvfx/04_refvfx__refvfx_promptB.mp4
cpv "$G/002_ctt_v2/videos/${IT}__s42.mp4"        $P/04_refvfx/04_refvfx__cttv2_promptA.mp4
cpv "$G/005_ctt_v2_leaky/videos/${IT}__s43.mp4"  $P/04_refvfx/04_refvfx__cttv2_promptB.mp4

# ---------------- 06_dataset ----------------
cpv "$E/S0/clips/sakura_petals/sakura_petals_0.mp4"           $P/06_dataset/06_dataset__s0_1.mp4
cpv "$E/S0/clips/water_bending/water_bending_1.mp4"           $P/06_dataset/06_dataset__s0_2.mp4
cpv "$E/S0/clips/illustration_scene/illustration_scene_0.mp4" $P/06_dataset/06_dataset__s0_3.mp4
cpv "$E/S0/clips/giant_grab/giant_grab_0.mp4"                 $P/06_dataset/06_dataset__s0_4.mp4
cpv "$E/S1/clips/spec_portal__humanvid_7712292__s42.mp4"                        $P/06_dataset/06_dataset__s1_1.mp4
cpv "$E/S1/clips/spec_wireframe__vcbench_teacher_33674801_2160x3840__s42.mp4"   $P/06_dataset/06_dataset__s1_2.mp4
cpv "$E/S1/clips/spec_gas_transformation__humanvid_8117113__s42.mp4"            $P/06_dataset/06_dataset__s1_3.mp4
cpv "$E/S1/clips/spec_super_fast_run__davis_car-roundabout__s42.mp4"            $P/06_dataset/06_dataset__s1_4.mp4
cpv "$E/S2a/clips/s2_0693_c06.mp4" $P/06_dataset/06_dataset__s2_1.mp4
cpv "$E/S2b/clips/s2_0165_c04.mp4" $P/06_dataset/06_dataset__s2_2.mp4
cpv "$E/S2b/clips/s2_0620_c06.mp4" $P/06_dataset/06_dataset__s2_3.mp4
cpv "$E/S2b/clips/s2_0114_c08.mp4" $P/06_dataset/06_dataset__s2_4.mp4
cpv "$E/S4/clips/005860.mp4" $P/06_dataset/06_dataset__s4_1.mp4
cpv "$E/S4/clips/006099.mp4" $P/06_dataset/06_dataset__s4_2.mp4
cpv "$E/S4/clips/000060.mp4" $P/06_dataset/06_dataset__s4_3.mp4
cpv "$E/S4/clips/001117.mp4" $P/06_dataset/06_dataset__s4_4.mp4

# ---------------- 07_counterfactual ----------------
# S3 (3D depth-parallax) shared-operator grid from exp_076 runs — each op row is one
# operator instance (same seed) rendered on the same 3 endpoint pairs.
# opA=roll_crossfade_fog  opB=orbit_depth_wipe_sphere_focus  opC=crane_crossfade
# ep1=animalization_4__shadow_11  ep2=gas_transformation_4__super_fast_run_8  ep3=polygon_0__illustration_scene_3
E76=$REPO/outputs/videos/exp_076_depth3d_transitions
eps76=(animalization_4__shadow_11 gas_transformation_4__super_fast_run_8 polygon_0__illustration_scene_3); epl=(ep1 ep2 ep3)
for j in 0 1 2; do
  cpv "$E76/run_0005/videos/sharedop0__${eps76[$j]}__roll_crossfade_fog__687661.mp4"              "$P/07_counterfactual/07_counterfactual__opA_${epl[$j]}.mp4"
  cpv "$E76/run_0005/videos/sharedop1__${eps76[$j]}__orbit_depth_wipe_sphere_focus__828682.mp4"   "$P/07_counterfactual/07_counterfactual__opB_${epl[$j]}.mp4"
  cpv "$E76/run_0001/videos/sharedop1__${eps76[$j]}__crane_crossfade__155414.mp4"                 "$P/07_counterfactual/07_counterfactual__opC_${epl[$j]}.mp4"
done

# ---------------- shared endpoint cuts ----------------
davis_head tennis      $TMP/tennis24.mp4
davis_head snowboard   $TMP/snowboard24.mp4
davis_head lucia       $TMP/lucia24.mp4
davis_head mallard-water $TMP/mallard24.mp4

# ---------------- 12_iid ----------------
I=$P/12_iid
# row1: davis_tennis_snowboard <- shadow_smoke (two-sided, unseen-foreign)
cp $TMP/tennis24.mp4    $I/12_iid__row1_start.mp4
cp $TMP/snowboard24.mp4 $I/12_iid__row1_end.mp4
cpv "$T/shadow_smoke/shadow_smoke_0.mp4" $I/12_iid__row1_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-unseen-foreign__ctt_v2__davis_tennis_snowboard__ref_shadow_smoke_0__s42.mp4" $I/12_iid__row1_base.mp4
cpv "$G/003_refvfx_A/videos/G-unseen-foreign__ctt_v2__davis_tennis_snowboard__ref_shadow_smoke_0__seed42.mp4"   $I/12_iid__row1_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-unseen-foreign__ctt_v2__davis_tennis_snowboard__ref_shadow_smoke_0__s42.mp4"        $I/12_iid__row1_cttv2.mp4
# row2: davis_lucia <- animalization (one-sided, unseen-foreign)
cp $TMP/lucia24.mp4 $I/12_iid__row2_start.mp4
cpv "$T/animalization/animalization_0.mp4" $I/12_iid__row2_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-unseen-foreign__ctt_v2__davis_lucia__ref_animalization_0__s42.mp4" $I/12_iid__row2_base.mp4
cpv "$G/003_refvfx_A/videos/G-unseen-foreign__ctt_v2__davis_lucia__ref_animalization_0__seed42.mp4"   $I/12_iid__row2_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-unseen-foreign__ctt_v2__davis_lucia__ref_animalization_0__s42.mp4"        $I/12_iid__row2_cttv2.mp4
# row3: davis_mallard_water <- super_fast_run (one-sided, unseen-foreign)
cp $TMP/mallard24.mp4 $I/12_iid__row3_start.mp4
cpv "$T/super_fast_run/super_fast_run_0.mp4" $I/12_iid__row3_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-unseen-foreign__ctt_v2__davis_mallard_water__ref_super_fast_run_0__s42.mp4" $I/12_iid__row3_base.mp4
cpv "$G/003_refvfx_A/videos/G-unseen-foreign__ctt_v2__davis_mallard_water__ref_super_fast_run_0__seed42.mp4"   $I/12_iid__row3_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-unseen-foreign__ctt_v2__davis_mallard_water__ref_super_fast_run_0__s42.mp4"        $I/12_iid__row3_cttv2.mp4
# row4: gas_transformation_6 <- earth_element (one-sided, unseen-cross)
cut_head "$T/gas_transformation/gas_transformation_6.mp4" $I/12_iid__row4_start.mp4 24
cpv "$T/earth_element/earth_element_4.mp4" $I/12_iid__row4_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-unseen-cross__ctt_v2__gas_transformation_6__ref_earth_element_4__s42.mp4" $I/12_iid__row4_base.mp4
cpv "$G/003_refvfx_A/videos/G-unseen-cross__ctt_v2__gas_transformation_6__ref_earth_element_4__seed42.mp4"   $I/12_iid__row4_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-unseen-cross__ctt_v2__gas_transformation_6__ref_earth_element_4__s42.mp4"        $I/12_iid__row4_cttv2.mp4
# row5: davis_lucia <- earth_element (one-sided, unseen-foreign)
cp $TMP/lucia24.mp4 $I/12_iid__row5_start.mp4
cpv "$T/earth_element/earth_element_4.mp4" $I/12_iid__row5_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-unseen-foreign__ctt_v2__davis_lucia__ref_earth_element_4__s42.mp4" $I/12_iid__row5_base.mp4
cpv "$G/003_refvfx_A/videos/G-unseen-foreign__ctt_v2__davis_lucia__ref_earth_element_4__seed42.mp4"   $I/12_iid__row5_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-unseen-foreign__ctt_v2__davis_lucia__ref_earth_element_4__s42.mp4"        $I/12_iid__row5_cttv2.mp4
# row6: earth_element_6 <- earth_element (one-sided, unseen-SAME)
cut_head "$T/earth_element/earth_element_6.mp4" $I/12_iid__row6_start.mp4 24
cpv "$T/earth_element/earth_element_4.mp4" $I/12_iid__row6_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-unseen-same__ctt_v2__earth_element_6__ref_earth_element_4__s42.mp4" $I/12_iid__row6_base.mp4
cpv "$G/003_refvfx_A/videos/G-unseen-same__ctt_v2__earth_element_6__ref_earth_element_4__seed42.mp4"   $I/12_iid__row6_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-unseen-same__ctt_v2__earth_element_6__ref_earth_element_4__s42.mp4"        $I/12_iid__row6_cttv2.mp4

# ---------------- 13_zeroshot ----------------
Z=$P/13_zeroshot
# row1: davis_tennis_snowboard <- firelava (two-sided, zs-foreign)
cp $TMP/tennis24.mp4    $Z/13_zeroshot__row1_start.mp4
cp $TMP/snowboard24.mp4 $Z/13_zeroshot__row1_end.mp4
cpv "$T/firelava/firelava_0.mp4" $Z/13_zeroshot__row1_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_firelava_0__s42.mp4" $Z/13_zeroshot__row1_base.mp4
cpv "$G/003_refvfx_A/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_firelava_0__seed43.mp4"   $Z/13_zeroshot__row1_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_firelava_0__s43.mp4"        $Z/13_zeroshot__row1_cttv2.mp4
# row2: shadow_smoke_7 <- firelava (two-sided, zs-cross)
cut_head "$T/shadow_smoke/shadow_smoke_7.mp4" $Z/13_zeroshot__row2_start.mp4 16  # 24f leaks the smoke onset
cut_tail "$T/shadow_smoke/shadow_smoke_7.mp4" $Z/13_zeroshot__row2_end.mp4 24
cpv "$T/firelava/firelava_0.mp4" $Z/13_zeroshot__row2_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-zs-cross__ctt_v2__shadow_smoke_7__ref_firelava_0__s42.mp4" $Z/13_zeroshot__row2_base.mp4
cpv "$G/003_refvfx_A/videos/G-zs-cross__ctt_v2__shadow_smoke_7__ref_firelava_0__seed42.mp4"   $Z/13_zeroshot__row2_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-zs-cross__ctt_v2__shadow_smoke_7__ref_firelava_0__s42.mp4"        $Z/13_zeroshot__row2_cttv2.mp4
# row3: davis_lucia <- saint_glow (one-sided, zs-foreign)
cp $TMP/lucia24.mp4 $Z/13_zeroshot__row3_start.mp4
cpv "$T/saint_glow/saint_glow_0.mp4" $Z/13_zeroshot__row3_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-zs-foreign__ctt_v2__davis_lucia__ref_saint_glow_0__s42.mp4" $Z/13_zeroshot__row3_base.mp4
cpv "$G/003_refvfx_A/videos/G-zs-foreign__ctt_v2__davis_lucia__ref_saint_glow_0__seed42.mp4"   $Z/13_zeroshot__row3_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-zs-foreign__ctt_v2__davis_lucia__ref_saint_glow_0__s42.mp4"        $Z/13_zeroshot__row3_cttv2.mp4
# row4: davis_tennis_snowboard <- display_transition (two-sided, zs-foreign)
cp $TMP/tennis24.mp4    $Z/13_zeroshot__row4_start.mp4
cp $TMP/snowboard24.mp4 $Z/13_zeroshot__row4_end.mp4
cpv "$T/display_transition/display_transition_1.mp4" $Z/13_zeroshot__row4_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_display_transition_1__s42.mp4" $Z/13_zeroshot__row4_base.mp4
cpv "$G/003_refvfx_A/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_display_transition_1__seed42.mp4"   $Z/13_zeroshot__row4_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_display_transition_1__s42.mp4"        $Z/13_zeroshot__row4_cttv2.mp4
# row5: davis_tennis_snowboard <- raven_transition (two-sided, zs-foreign)
cp $TMP/tennis24.mp4    $Z/13_zeroshot__row5_start.mp4
cp $TMP/snowboard24.mp4 $Z/13_zeroshot__row5_end.mp4
cpv "$T/raven_transition/raven_transition_0.mp4" $Z/13_zeroshot__row5_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_raven_transition_0__s42.mp4" $Z/13_zeroshot__row5_base.mp4
cpv "$G/003_refvfx_A/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_raven_transition_0__seed42.mp4"   $Z/13_zeroshot__row5_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-zs-foreign__ctt_v2__davis_tennis_snowboard__ref_raven_transition_0__s43.mp4"        $Z/13_zeroshot__row5_cttv2.mp4
# row6: hero_flight_5 <- display_transition (two-sided, zs-cross)
cut_head "$T/hero_flight/hero_flight_5.mp4" $Z/13_zeroshot__row6_start.mp4 24
cut_tail "$T/hero_flight/hero_flight_5.mp4" $Z/13_zeroshot__row6_end.mp4 24
cpv "$T/display_transition/display_transition_1.mp4" $Z/13_zeroshot__row6_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-zs-cross__ctt_v2__hero_flight_5__ref_display_transition_1__s42.mp4" $Z/13_zeroshot__row6_base.mp4
cpv "$G/003_refvfx_A/videos/G-zs-cross__ctt_v2__hero_flight_5__ref_display_transition_1__seed43.mp4"   $Z/13_zeroshot__row6_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-zs-cross__ctt_v2__hero_flight_5__ref_display_transition_1__s42.mp4"        $Z/13_zeroshot__row6_cttv2.mp4
# row7: money_rain_3 <- live_concert (one-sided, zs-cross)
cut_head "$T/money_rain/money_rain_3.mp4" $Z/13_zeroshot__row7_start.mp4 24
cpv "$T/live_concert/live_concert_1.mp4" $Z/13_zeroshot__row7_reference.mp4
cpv "$G/007_base_cond_ctt/videos/G-zs-cross__ctt_v2__money_rain_3__ref_live_concert_1__s42.mp4" $Z/13_zeroshot__row7_base.mp4
cpv "$G/003_refvfx_A/videos/G-zs-cross__ctt_v2__money_rain_3__ref_live_concert_1__seed42.mp4"   $Z/13_zeroshot__row7_refvfx.mp4
cpv "$G/005_ctt_v2_leaky/videos/G-zs-cross__ctt_v2__money_rain_3__ref_live_concert_1__s42.mp4"        $Z/13_zeroshot__row7_cttv2.mp4

echo "=== file count per folder ==="
for d in $P/*/; do echo "$(basename $d): $(ls $d | wc -l)"; done
echo BUILD_OK
