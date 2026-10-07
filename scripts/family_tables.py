#!/usr/bin/env python
"""Metric-family tables for grid v3, one standard format everywhere:
  Table 1  own arms by tier (seen / unseen / zero-shot; neutral prompt; HF-121f + ED-81f rows pooled per tier)
  Table 2  prior works vs ours on the shared one-sided zero-shot set (identical rows per arm)
Rows and scopes never change between families; only the columns do. A cell is '--' when the metric is not
measurable for that arm/scope (one-frame conditioning has no given motion; a one-sided set has no B side; a
pending eval). Thin cells (n < 10) render italic with a dagger and never bold.

Families (each its own preview folder under papers_drafts/_preview/):
  input_fidelity_gridv3      identity (endpoint / hand-off, A / B) + motion (endpoint / hand-off, A / B)
  transition_fidelity_gridv3 transport, motion fidelity vs ref, VideoPrism / CLIP ref-sim, copy, seam
  quality_gridv3             motion smoothness (matched spacing + native), dynamic degree, aesthetic
  metrics_v2_gridv3          the full table: input fidelity | transition fidelity | quality, in one

Sources (store evals over the feature store; nothing is computed here beyond means / derived px/s):
  043 endpoint identity   044 endpoint motion   045 smooth matched   046 viclip text
  047 transport v5 per_gen (Transport = bl_333, the equal-thirds blend; the weight-grid columns)
  048 seam (seam_free, seam z; recomputed from the stored temporal LPIPS at the physical windows)
  049 flow+action (Flow MSE from flow_u32@raft-r256-g24x32, Action KL from action@swin3db-k400-u32; whole video vs reference)
  038 handoff (hand-off identity_A/B, motion_A/B, fps)   039 copy (copy_max, near_copy)
  040 lenses (det_motion_fidelity, videoprism_sim_ref, clip_sim_ref, dynamic_degree_mean_mag, aesthetic)
  028/030/041/042 per_gen (transport_v4 = the v4 transport_pct, for the semantic_transport families only)
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from datetime import date
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from store_eval_common import parse_stem, harness_arm_of, load_jsonl, grid_of  # noqa: E402

EVALS = REPO_ROOT / "store" / "evals"
PREVIEW = REPO_ROOT / "papers_drafts" / "_preview"
PAPER = REPO_ROOT / "papers_drafts" / "ctt_iclr2027"
TEXBIN = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/texlive/bin/aarch64-linux"
MIN_N = 10
PRIOR_BASELINE = False   # owner 2026-09-18: the prior-works table compares the prior works with SEGUE w/o guidance and SEGUE only
WITH_BASE = False        # owner 2026-09-18: the own-arms table lists baseline LoRA, SEGUE w/o guidance, SEGUE (no LTX-2 no-reference row)
CLEAN = False            # --clean: no per-cell n, no dagger, no n column, simple headers (the version sent to the PI)
CLEAN_HEADER = {"ep_id_A": "Start", "ep_id_B": "End", "ho_id_A": "Start hand-off", "ho_id_B": "End hand-off",
                "ep_mot_A": "Start", "ep_mot_B": "End", "ho_mot_A": "Start hand-off", "ho_mot_B": "End hand-off",
                "transport": "Transport", "motfid": "Motion fid.", "vp_ref": "Ref.\\ sim.", "clip_ref": "Ref.\\ sim.\\ (CLIP)",
                "copy_pct": "Copy \\%", "copy_max": "Copy max", "seam_free": "Seam-free \\%", "seam_med": "Seam $z$",
                "flow_mse": "Flow MSE", "action_kl": "Action KL",
                "smooth_native": "Smoothness", "smooth8": "Smoothness (8\\,fps)", "dyn_pxs": "Dynamics (px/s)", "aesthetic": "Aesthetic",
                "text_own": "Consistency", "text_effect": "Effect-prompt consistency",
                "ep_psnr_A": "PSNR start", "ep_psnr_B": "PSNR end", "ep_ssim_A": "SSIM start", "ep_ssim_B": "SSIM end",
                "ep_lpips_A": "LPIPS start", "ep_lpips_B": "LPIPS end"}

# ---- table roster (metrics v5, D6): every table row comes from scripts/metrics_v5/roster.json.
# roles: own_base -> the --with-base rows; own (+ sweep:true) -> Table 1 own arms; prior -> Table 2;
# prior_teg -> Table 3. label / tex / gens per entry; order = roster order.
_ROSTER = json.loads((REPO_ROOT / "scripts/metrics_v5/roster.json").read_text())


def _roster_rows(role, sweep=None):
    out = []
    for a in _ROSTER["arms"]:
        if a["role"] != role:
            continue
        if sweep is False and a.get("sweep"):
            continue
        if sweep is True and not a.get("sweep"):
            continue
        out.append((a["label"], a["id"]))
    return out


SWEEP = False            # --sweep: also list the roster sweep:true rows (DCG w=1.5, w=3)
SUFFIX = ""              # --suffix: write to <family>[_sweep][_clean]<suffix>/
OWN_BASE = _roster_rows("own_base")               # LTX-2 (no ref., effect prompt) — the --with-base / Table-3 same-backbone row
OWN_ARMS = _roster_rows("own", sweep=False)       # ic_gen, dualforce_control, dualforce_dcg_w6 (roster order)
EXT = _roster_rows("prior")                       # Table 2: prior works (author-native + their neutral twins)
EXT_TEG = _roster_rows("prior_teg")               # Table 3: two-endpoint baselines (effect + neutral twins)
ALL_ARMS = [(a["label"], a["id"]) for a in _ROSTER["arms"]]
TEX = {a["label"]: a.get("tex", a["label"]) for a in _ROSTER["arms"]}
GENS = {a["id"]: tuple(a["gens"]) for a in _ROSTER["arms"]}


def own_arms():
    """Own-arm rows in roster order; the sweep arms (w=1.5, w=3) appear only under --sweep."""
    return _roster_rows("own") if SWEEP else OWN_ARMS

# ----------------------------------------------------------------------------- column catalogue
# key -> (header, direction, decimals, how)   direction: max | min | none ; how: mean | median
CAT = {
    "ep_id_A":  ("Endpoint A", "max", 3, "mean"), "ho_id_A": ("Hand-off A", "max", 3, "mean"),
    "ep_id_B":  ("Endpoint B", "max", 3, "mean"), "ho_id_B": ("Hand-off B", "max", 3, "mean"),
    "ep_mot_A": ("Endpoint A", "max", 3, "mean"), "ho_mot_A": ("Hand-off A", "max", 3, "mean"),
    "ep_mot_B": ("Endpoint B", "max", 3, "mean"), "ho_mot_B": ("Hand-off B", "max", 3, "mean"),
    "transport": ("Transport", "max", 1, "mean"), "transport_v4": ("Transport", "max", 1, "mean"), "motfid": ("Motion fid.\\ (vs ref)", "max", 3, "mean"),
    "vp_ref": ("Ref sim.\\ (VP)", "max", 3, "mean"), "clip_ref": ("Ref sim.\\ (CLIP)", "max", 3, "mean"),
    "copy_pct": ("Copy \\%", "min", 1, "mean"), "copy_max": ("Copy max", "min", 3, "mean"),
    "seam_free": ("Seam-free \\%", "max", 1, "mean"), "seam_med": ("Seam $z$ med.", "min", 2, "median"),
    "flow_mse": ("Flow MSE", "min", 2, "mean"), "action_kl": ("Action KL", "min", 2, "mean"),
    "smooth8": ("Smooth.\\ (8\\,fps)", "max", 3, "mean"), "smooth_native": ("Motion smooth.", "max", 3, "mean"),
    "dyn_pxs": ("Dyn.\\ (px/s)", "none", 1, "mean"), "aesthetic": ("Aesthetic", "max", 3, "mean"),
    "text_own": ("Own prompt", "max", 3, "mean"), "text_effect": ("Effect prompt", "max", 3, "mean"),
    # ---- pixel fidelity of the given endpoint frames (metrics v5, eval 050; frame-level only, window means stay in 050)
    "ep_psnr_A": ("PSNR start", "max", 1, "mean"), "ep_ssim_A": ("SSIM start", "max", 3, "mean"), "ep_lpips_A": ("LPIPS start", "min", 3, "mean"),
    "ep_psnr_B": ("PSNR end", "max", 1, "mean"), "ep_ssim_B": ("SSIM end", "max", 3, "mean"), "ep_lpips_B": ("LPIPS end", "min", 3, "mean"),
    # ---- semantic-transport metric trials (ours; misc/2026-09-02_temporal_dynamics_metric/score_v3_mf.py per-gen records)
    "look_u": ("Look$_u$", "max", 1, "mean"), "s3_chk": ("S3 (re-scored)", "max", 1, "mean"),
    "mf_pool": ("MF pooled", "max", 1, "mean"), "px_pool": ("Pixel pooled", "max", 1, "mean"),
    "look_px_w25": ("Look$_u$+Pixel (25\\%)", "max", 1, "mean"), "look_px_avg": ("Look$_u$+Pixel (50\\%)", "max", 1, "mean"),
    "bl_333": (".33/.33/.33", "max", 1, "mean"), "bl_255": (".25/.25/.50", "max", 1, "mean"), "bl_244": (".20/.40/.40", "max", 1, "mean"),
    "bl_235": (".20/.30/.50", "max", 1, "mean"), "bl_1455": (".10/.45/.45", "max", 1, "mean"), "bl_055": ("0/.50/.50", "max", 1, "mean"),
    "bl_064": ("0/.60/.40", "max", 1, "mean"), "bl_d": ("D only", "max", 1, "mean"), "bl_px": ("Pixel only", "max", 1, "mean"),
    "look_mf_w25": ("Look$_u$+MF (25\\%)", "max", 1, "mean"), "look_mf_avg": ("Look$_u$+MF (50\\%)", "max", 1, "mean"),
    "look_mf_p05": ("Look$_u$ $\\times$ MF$^{0.5}$", "max", 1, "mean"), "look_mf_p025": ("Look$_u$ $\\times$ MF$^{0.25}$", "max", 1, "mean"),
}
KEY = {  # one line per metric for the compact metric key
    "ep_id_A": "Endpoint identity A/B: DINO cosine between the output frames that should be the given start/end frames and the given frames (mean over the given window; 1 frame on ED rows / prior works).",
    "ho_id_A": "Hand-off identity A/B: the first/last $K$ generated frames vs.\\ the adjacent given frame, $K=\\mathrm{fps}/3$.",
    "ep_mot_A": "Endpoint motion A/B: dense RAFT flow of the given window, output vs.\\ given clip, agreement $1-\\sum\\|\\Delta F\\|/\\sum(\\|F_\\mathrm{out}\\|+\\|F_\\mathrm{given}\\|)$ (needs a 9-frame clip; NaN if the clip is static).",
    "ho_mot_A": "Hand-off motion A/B: flow continuity across the hand-off (pending the whole-video flow pass).",
    "transport": "Transport: the equal-thirds blend of appearance (Mu), semantic transport (D) and pixel transport (CoTracker3 tracks) rank-similarities vs.\\ the reference class, re-ranked against the same-weight corpus population, \\% of the class ceiling (capped at 100).",
    "motfid": "Motion fid.: Yatim et al.\\ tracklet velocity-direction correlation, output vs.\\ reference (NaN where nothing moves).",
    "vp_ref": "Ref sim.\\ (VP / CLIP): per-frame mean-of-max cosine between output and reference embeddings (VideoPrism: refVFX headline; CLIP-B/32: VAP).",
    "copy_pct": "Copy \\% / Copy max: M2a copy score of the output's mid frames vs.\\ the reference's own non-core frames (rate at $\\tau=0.858$; mean of the max).",
    "seam_free": "Seam-free \\% / Seam $z$ med.: temporal-LPIPS robust $z$ of the hand-off step vs.\\ the video's own steps (prefix seam; both seams on two-sided rows); share with $z\\le 3$ and the median $z$.",
    "smooth8": "Smooth.\\ (8\\,fps / native): mean cosine between CLIP-B/32 embeddings of frames $\\approx$125\\,ms apart (stride round(fps/8)), and of consecutive frames (frame-rate confounded).",
    "dyn_pxs": "Dyn.\\ (px/s): mean RAFT flow magnitude per step at 256\\,px times fps; descriptive, no direction.",
    "aesthetic": "Aesthetic: LAION aesthetic predictor on CLIP-L/14 frames, mean.",
}
KEY_ORDER = ["ep_id_A", "ho_id_A", "ep_mot_A", "ho_mot_A", "transport", "motfid", "flow_mse", "action_kl", "vp_ref", "copy_pct", "seam_free", "smooth8", "dyn_pxs", "aesthetic"]
SHORT = {  # per column: how it is computed; a citation ONLY where the whole metric is inherited (backbones are not cited)
    "ep_id_A": "Identity Endpoint A/B: DINOv2-B CLS cosine, output frames of the given window vs.\\ the given start/end frames",
    "ho_id_A": "Identity Hand-off A/B: DINOv2-B CLS cosine, first/last $K{=}$fps/3 generated frames vs.\\ the adjacent given frame",
    "ep_mot_A": "Motion Endpoint A/B: RAFT flow of the given window, output vs.\\ given clip, velocity agreement; needs a 9-frame clip",
    "ho_mot_A": "Motion Hand-off A/B: flow continuity across the hand-off (pending)",
    "transport": "Transport: the equal-thirds three-channel blend -- appearance (Mu), semantic transport (D) and pixel transport (CoTracker3 tracks), each a rank-similarity vs.\\ the 222-corpus population, weighted 1/3 each, re-ranked, \\% of the class ceiling (ours)",
    "transport_v4": "Transport (v4): S3 semantic transport on DINOv2-B features vs.\\ the reference class, \\% of the certified ceiling (ours; the deployed metric)",
    "motfid": "Motion fid.\\ (vs ref): CoTracker3 tracklets, velocity-direction correlation, output vs.\\ reference -- the Motion Fidelity metric of \\citet{yatim2024spacetime}",
    "vp_ref": "Ref sim.\\ (VP): VideoPrism per-frame mean-of-max cosine, output vs.\\ reference -- refVFX's reference-similarity metric~\\citep{refvfx2026}",
    "clip_ref": "Ref sim.\\ (CLIP): CLIP-B/32 per-frame mean-of-max cosine, output vs.\\ reference -- VAP's reference-similarity metric~\\citep{vap2025}",
    "copy_pct": "Copy \\%: share of outputs with a mid frame within DINOv2-B cosine $\\tau{=}0.858$ of a non-core reference frame",
    "copy_max": "Copy max: mean over outputs of the max DINOv2-B cosine between a mid frame and a non-core reference frame",
    "seam_free": "Seam-free \\%: temporal LPIPS at the hand-off step, robust $z$ vs.\\ the video's own steps, share with $z\\le3$",
    "seam_med": "Seam $z$ med.: median of that hand-off $z$",
    "flow_mse": "Flow MSE: whole-video optical-flow MSE, generated vs.\\ reference -- RAFT-large flow over 32 uniform steps reduced to a 24$\\times$32 grid of the frame-diagonal fraction, mean squared difference in (\\% of frame diagonal)$^2$ (lower = closer motion field; ours)",
    "action_kl": "Action KL: KL divergence of the Video Swin-B Kinetics-400 action-class distribution, reference $\\parallel$ generated, over 32 sampled frames (lower = closer action signature; ours)",
    "smooth_native": "Motion smooth.: CLIP-B/32 cosine of consecutive frames at native rate (ours 24\\,fps, prior works 6.5--9.7\\,fps) -- the motion-smoothness metric as reported by VAP and refVFX~\\citep{vap2025,refvfx2026}",
    "smooth8": "Smooth.\\ (8\\,fps): the same between frames $\\approx$125\\,ms apart, stride round(fps/8) (frame-rate control)",
    "dyn_pxs": "Dyn.\\ (px/s): mean RAFT flow magnitude per step at 256\\,px $\\times$ fps -- VBench's dynamic degree~\\citep{huang2024vbench}, in px/s (descriptive)",
    "aesthetic": "Aesthetic: LAION aesthetic predictor on CLIP-L/14 frame embeddings, mean over frames -- VBench's aesthetic quality~\\citep{huang2024vbench}",
    "text_own": "Text (own prompt): ViCLIP video--text cosine between the output and the prompt it was conditioned on (trigger token removed) -- VBench's overall consistency~\\citep{huang2024vbench}",
    "text_effect": "Text (effect prompt): the same against the effect description of the row, identical for every arm",
    "ep_psnr_A": "Pixel PSNR start / end: peak signal-to-noise ratio (dB) of the output's first / last frame vs.\\ the given start9[0] / end9[8] frame, over all pixels+3 channels; capped by the H.264 encoding of both clips (ours)",
    "ep_ssim_A": "Pixel SSIM start / end ($\\times$100): structural similarity (Wang et al.\\ 2004, 11$\\times$11 Gaussian $\\sigma$1.5) of the output's first / last frame vs.\\ the given start / end frame (ours)",
    "ep_lpips_A": "Pixel LPIPS start / end ($\\times$100, $\\downarrow$): AlexNet perceptual distance of the output's first / last frame vs.\\ the given start / end frame (ours)",
    "ep_psnr_B": "Pixel PSNR end: PSNR of the output's last frame vs.\\ the given end9[8] frame",
    "ep_ssim_B": "Pixel SSIM end: SSIM of the output's last frame vs.\\ the given end frame",
    "ep_lpips_B": "Pixel LPIPS end: AlexNet perceptual distance of the output's last frame vs.\\ the given end frame",
    "look_u": "Look$_u$: our size-free semantic transport -- per-frame DINOv2-B CLS resampled to 32 steps, endpoint plane removed, appearance bag (Mu) + optimal transport over the residual change (D), rank-fused; \\% of the class ceiling (ours)",
    "s3_chk": "S3 (re-scored): the Transport column re-derived from the stored pair scores through the trial script's own pooling (agrees with Transport to within 1\\,pp; the residual is the harness's per-generation pooling)",
    "mf_pool": "MF pooled: the Motion Fidelity of \\citet{yatim2024spacetime} scored like ours -- output vs.\\ every clip of the reference class, ranked against the corpus population, \\% of the class ceiling",
    "look_mf_w25": "Look$_u$+MF: rank-average of Look$_u$ and the Motion-Fidelity rank-similarity with the given MF weight, re-ranked; \\% of ceiling",
    "px_pool": "Pixel pooled: pixel-space transition descriptor -- CoTracker3 tracks summarised per step on a 32-step normalized-time grid (moving share, speed, direction histogram, divergence / curl / shear of the moving field, spread; $z$-scored), cosine of the ordered sequence, output vs.\\ every clip of the reference class, ranked against the corpus population, \\% of the class ceiling (ours; the strongest pixel-motion channel of MOTION\\_DESCRIPTORS)",
    "look_px_w25": "Look$_u$+Pixel: rank-average of Look$_u$ and the pixel-descriptor rank-similarity with the given weight, re-ranked; \\% of ceiling",
    "bl_333": "$w_\\mathrm{app}/w_\\mathrm{sem}/w_\\mathrm{pix}$ columns: the Look$_u$ rule generalised to three channels -- rank-similarities of appearance (Mu), semantic transport (D) and pixel transport, weighted sum, re-ranked against the same-weight corpus population, within-class ceilings; (.50/.50/0) is Look$_u$; \\% of ceiling",
    "look_mf_p05": "Look$_u$ $\\times$ MF$^{p}$: power fusion $s_{\\mathrm{Look}}\\cdot s_{\\mathrm{MF}}^{p}$ of the two rank-similarities, re-ranked; \\% of ceiling",
}
SIMPLE = {  # --clean: plain-language one-liners, source named in words, no citations
    "ep_id_A": r"Identity Start / End: are the given start/end frames kept? DINO similarity between the output's given-window frames and the given frames (1 = identical)",
    "ho_id_A": r"Identity Start / End hand-off: does the subject look the same right after (before) the given frames? DINO similarity of the first (last) generated frames to the adjacent given frame",
    "ep_mot_A": r"Motion Start / End: is the given clip's motion kept? Optical-flow agreement between the output's given window and the given clip (1 = identical motion; needs a 9-frame clip)",
    "ho_mot_A": r"Motion Start / End hand-off: does the given motion continue into the generated frames? Flow continuity across the hand-off (pending)",
    "transport": r"Transport: does the output perform the reference's transition? Appearance, semantic transport and pixel transport blended at equal thirds, as \% of the class ceiling. Ours",
    "transport_v4": r"Transport (v4): semantic transport vs.\ the reference class, \% of the class ceiling. Ours (the deployed metric)",
    "motfid": r"Motion fid.: does the output move like the reference? Correlation of tracked-point motion between output and reference. The Motion Fidelity metric of Yatim et al.",
    "vp_ref": r"Ref.\ sim.: does the output look like the reference? VideoPrism embedding similarity between output and reference. From refVFX",
    "clip_ref": r"Ref.\ sim.\ (CLIP): the same with CLIP embeddings. From VAP",
    "copy_pct": r"Copy \%: share of outputs that replay a frame of the reference's own scenes",
    "copy_max": r"Copy max: how close the most reference-like output frame gets to the reference's own scenes",
    "seam_free": r"Seam-free \%: is there a visible cut at the hand-off? Share of outputs whose hand-off step is no larger than the video's normal frame-to-frame change (LPIPS)",
    "seam_med": r"Seam $z$: the typical size of that hand-off step, in units of the video's normal frame-to-frame change",
    "flow_mse": r"Flow MSE: does the output's motion match the reference's? Optical-flow field of the whole video, generated vs.\ reference, mean squared difference (lower is closer). Ours",
    "action_kl": r"Action KL: does the output read as the same kind of action as the reference? KL divergence of a video action-classifier's class distribution, reference vs.\ generated (lower is closer). Ours",
    "smooth_native": r"Smoothness: how smooth is the video frame to frame? CLIP similarity of consecutive frames at native rate (ours 24\,fps, prior works 6.5--9.7\,fps). Motion smoothness as reported by VAP and refVFX",
    "smooth8": r"Smoothness (8\,fps): the same between frames 125\,ms apart, so all arms are compared at the same temporal spacing",
    "dyn_pxs": r"Dynamics (px/s): how much moves? Mean optical-flow speed in pixels per second (descriptive). Dynamic degree from VBench",
    "aesthetic": r"Aesthetic: how good do the frames look? LAION aesthetic score of the frames. Aesthetic quality from VBench",
    "text_own": r"Text: how consistent is the output with its own prompt? ViCLIP video--text similarity (trigger token removed). Overall consistency from VBench",
    "text_effect": r"Text (effect prompt): the same against the row's effect description, identical for every arm",
    "ep_psnr_A": r"Pixel PSNR start / end: how close are the output's first / last frames to the given start / end frames, pixel for pixel? Peak signal-to-noise ratio in dB (higher = closer; capped by video compression). Ours",
    "ep_ssim_A": r"Pixel SSIM start / end ($\times$100): structural similarity of the output's first / last frames to the given frames (100 = identical). Ours",
    "ep_lpips_A": r"Pixel LPIPS start / end ($\times$100): perceptual (AlexNet) distance of the output's first / last frames to the given frames (0 = identical, lower is better). Ours",
    "ep_psnr_B": r"Pixel PSNR end: PSNR of the output's last frame vs.\ the given end frame",
    "ep_ssim_B": r"Pixel SSIM end: SSIM of the output's last frame vs.\ the given end frame",
    "ep_lpips_B": r"Pixel LPIPS end: perceptual distance of the output's last frame vs.\ the given end frame",
    "look_u": r"Look$_u$: does the output perform the reference's transition? Size-free semantic transport (32-step resample, endpoint plane removed), \% of the class ceiling. Ours",
    "s3_chk": r"S3 (re-scored): the Transport column re-derived on the same rows (check)",
    "mf_pool": r"MF pooled: Motion Fidelity of Yatim et al.\ scored like ours, against the whole reference class, \% of ceiling",
    "look_mf_w25": r"Look$_u$+MF: Look$_u$ averaged with Motion Fidelity at the given weight, \% of ceiling",
    "px_pool": r"Pixel pooled: does the output's tracked motion unfold like the reference class's? Ordered pixel-motion descriptor from point tracks, \% of ceiling. Ours",
    "look_px_w25": r"Look$_u$+Pixel: Look$_u$ averaged with the pixel-motion descriptor at the given weight, \% of ceiling",
    "bl_333": r"$w_\mathrm{app}/w_\mathrm{sem}/w_\mathrm{pix}$: appearance, semantic transport and pixel transport blended at the given weights, \% of ceiling",
    "look_mf_p05": r"Look$_u{\cdot}$MF$^{p}$: Look$_u$ multiplied by Motion Fidelity to the power $p$, \% of ceiling",
}
FAMILY_OF = {  # column -> explanation group
    **{k: "Input fidelity" for k in ("ep_id_A", "ho_id_A", "ep_mot_A", "ho_mot_A")},
    **{k: "Transition fidelity" for k in ("transport", "transport_v4", "motfid", "flow_mse", "action_kl", "vp_ref", "clip_ref", "copy_pct", "copy_max", "seam_free", "seam_med")},
    **{k: "Quality" for k in ("smooth_native", "smooth8", "dyn_pxs", "aesthetic")},
    **{k: "Input fidelity" for k in ("text_own", "text_effect")},
    **{k: "Input fidelity" for k in ("ep_psnr_A", "ep_ssim_A", "ep_lpips_A", "ep_psnr_B", "ep_ssim_B", "ep_lpips_B")},
    **{k: "Transition fidelity" for k in ("look_u", "s3_chk", "mf_pool", "look_mf_w25", "look_mf_avg", "look_mf_p05", "look_mf_p025", "px_pool", "look_px_w25", "look_px_avg",
                                          "bl_333", "bl_255", "bl_244", "bl_235", "bl_1455", "bl_055", "bl_064", "bl_d", "bl_px")},
}
PREVIEW_REFS = r"""% metric source not in the paper bibliography (preview only)
@inproceedings{yatim2024spacetime,
  title     = {Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer},
  author    = {Yatim, Danah and Fridman, Rafail and Bar-Tal, Omer and Kasten, Yoni and Dekel, Tali},
  booktitle = {CVPR}, year = {2024}}
"""
KEY_ORDER = ["ep_id_A", "ho_id_A", "ep_mot_A", "ho_mot_A", "transport", "motfid", "flow_mse", "action_kl", "vp_ref", "copy_pct", "seam_free", "smooth8", "dyn_pxs", "aesthetic"]

FAMILIES = {
    "input_fidelity_gridv3": dict(title="input fidelity", groups=[
        ("Identity $\\uparrow$", ["ep_id_A", "ho_id_A", "ep_id_B", "ho_id_B"]),
        ("Motion $\\uparrow$", ["ep_mot_A", "ho_mot_A", "ep_mot_B", "ho_mot_B"]),
        ("Text $\\uparrow$", ["text_own", "text_effect"]),
        ("Pixel (given frames)", ["ep_psnr_A", "ep_ssim_A", "ep_lpips_A", "ep_psnr_B", "ep_ssim_B", "ep_lpips_B"])]),
    "transition_fidelity_gridv3": dict(title="transition fidelity", groups=[
        ("Ours", ["transport"]), ("Reference fidelity", ["motfid", "vp_ref", "clip_ref"]),
        ("Disentanglement", ["copy_pct", "copy_max"]), ("Seam", ["seam_free", "seam_med"])]),
    "quality_gridv3": dict(title="quality", groups=[
        ("Motion", ["smooth_native", "smooth8", "dyn_pxs"]), ("Frame", ["aesthetic"])]),
    "metrics_v2_gridv3": dict(title="full metric table (v2)", groups=[      # main table: endpoint columns only (hand-off lives in the input-fidelity family view)
        ("Identity $\\uparrow$", ["ep_id_A", "ep_id_B"]),
        ("Motion $\\uparrow$", ["ep_mot_A", "ep_mot_B"]),
        ("Text $\\uparrow$", ["text_own"]),
        ("Transition fidelity", ["transport", "motfid", "flow_mse", "action_kl", "vp_ref", "seam_free"]),
        ("Quality", ["smooth_native", "dyn_pxs", "aesthetic"])]),
    "semantic_transport_gridv3": dict(title="semantic transport (ours) -- metric trials", groups=[   # our metric trials, rendered like the other families
        ("Deployed", ["transport_v4"]),
        ("Size-free", ["look_u"]),
        ("+ Pixel transport (tracks)", ["px_pool", "look_px_w25", "look_px_avg"]),
        ("+ Motion Fidelity (Yatim et al.)", ["mf_pool", "look_mf_w25", "look_mf_avg"]),
        ("Reference", ["motfid"])]),
    "semantic_transport_blend_gridv3": dict(title="semantic transport (ours) -- channel-weight grid", groups=[   # appearance / semantic transport / pixel transport weights
        ("Deployed", ["transport_v4"]),
        ("Look$_u$ (.50/.50/0)", ["look_u"]),
        ("$w_\\mathrm{app}/w_\\mathrm{sem}/w_\\mathrm{pix}$", ["bl_333", "bl_255", "bl_244", "bl_235", "bl_1455", "bl_055", "bl_064"]),
        ("Single channel", ["bl_d", "bl_px"])]),
}
MF_PERGEN = REPO_ROOT / "misc/2026-09-02_temporal_dynamics_metric/V3CELLS_MF_pergen.jsonl"   # score_v3_mf.py, MF_MODE=cells (semantic_transport family only)
# metrics v5: the blend columns now come from eval 047 (per_gen.jsonl, keyed like the other store evals).
BLEND_COLS = {"bl_333": ".33/.33/.33", "bl_255": ".25/.25/.50", "bl_244": ".20/.40/.40", "bl_235": ".20/.30/.50", "bl_1455": ".10/.45/.45",
              "bl_055": "0/.50/.50", "bl_064": "0/.60/.40", "bl_d": "0/1/0 (D only)", "bl_px": "0/0/1 (PX only)"}
MF_COLS = {"look_u": "Look_u", "s3_chk": "S3", "mf_pool": "MF_only", "look_mf_w25": "Look+MF_w25", "look_mf_avg": "Look+MF_avg",
           "look_mf_p05": "Look*MF^0.5", "look_mf_p025": "Look*MF^0.25",
           "px_pool": "PX_only", "look_px_w25": "Look+PX_w25", "look_px_avg": "Look+PX_avg"}


def _find_eval(prefix: str) -> Path | None:
    """The newest numbered eval with this prefix (metrics v5: the draft fallback is gone; 043-048 are finalized)."""
    c = sorted(EVALS.glob(prefix + "*"))
    return c[-1] if c else None


def _rows_by_arm(eval_dir: Path | None, fname: str = "rows.jsonl") -> dict:
    out = {}
    if eval_dir is None:
        return out
    for d in eval_dir.iterdir():
        p = d / fname
        if p.exists():
            for r in load_jsonl(p):
                out[(d.name, r["item_id"], int(r["seed"]))] = r
    return out


def _fin(x):
    return x if isinstance(x, (int, float)) and x is not None and math.isfinite(x) else None


def collect(handoff_motion: str = "none") -> dict:
    """-> {(harness_arm, item_id, seed): record with all metric keys (None where undefined)}.

    metrics v5 (D6): endpoint identity/motion/smooth/text from the numbered evals 043-046; the
    Transport column (`transport` = bl_333) and every blend column from eval 047 (per_gen.jsonl);
    seam-free / seam z from eval 048 (recomputed at the physical windows). The v4 transport_pct is
    kept as `transport_v4` for the semantic_transport families only. 038 supplies the hand-off
    identity/motion and fps; 039 copy; 040 the lenses; MF_PERGEN the Motion-Fidelity trials."""
    e038 = _rows_by_arm(_find_eval("038_handoff_gridv3")); e039 = _rows_by_arm(_find_eval("039_copy_gridv3"))
    e040 = _rows_by_arm(_find_eval("040_lenses_gridv3"))
    e043 = _rows_by_arm(_find_eval("043_endpoint_identity_gridv3"))   # endpoint identity (ep_id_A/B)
    e044 = _rows_by_arm(_find_eval("044_endpoint_motion_gridv3"))     # endpoint motion (ep_mot_A/B)
    e045 = _rows_by_arm(_find_eval("045_smooth_matched_gridv3"))      # smoothness (smooth8/native)
    e046 = _rows_by_arm(_find_eval("046_viclip_text_gridv3"))         # text (text_own/effect)
    e048 = _rows_by_arm(_find_eval("048_seam_gridv3"))                # seam z / seam-free (physical windows)
    e049 = _rows_by_arm(_find_eval("049_transition_flow_action_gridv3"))   # flow MSE + action KL (whole-video)
    e050 = _rows_by_arm(_find_eval("050_endpoint_pixel_gridv3"))      # pixel PSNR/SSIM/LPIPS of the given endpoint frames (frame-level)
    bl = _rows_by_arm(_find_eval("047_transport_v5_gridv3"), "per_gen.jsonl")   # the blend (transport + weight grid)
    mf = {(r["harness_arm"], r["item_id"], int(r["seed"])): r for r in (load_jsonl(MF_PERGEN) if MF_PERGEN.exists() else [])}
    pg = {}                                                           # v4 transport_pct -> transport_v4 (semantic_transport families)
    for p in (list(EVALS.glob("028_grid_v3_paper_arms*/*/per_gen.jsonl")) + list(EVALS.glob("030_external_zs_authornative*/*/per_gen.jsonl"))
              + list(EVALS.glob("041_grid_v3_dcg_w_sweep*/*/per_gen.jsonl")) + list(EVALS.glob("042_teg_zs_baselines*/*/per_gen.jsonl"))):
        for r in load_jsonl(p):
            pg[(p.parent.name, r["item_id"], int(r["seed"]))] = r
    recs = {}
    for label, base in ALL_ARMS:
        for vrel in GENS[base]:
            vdir = REPO_ROOT / vrel
            if not (vdir / "grid.jsonl").exists():      # not registered yet -> its cells stay '--'
                print(f"[collect] {vrel}: not registered, skipped", file=sys.stderr); continue
            grid = grid_of(vdir)
            for v in sorted((vdir / "videos").glob("*.mp4")):
                item_id, seed = parse_stem(v.stem); g = grid.get(item_id)
                if not g:
                    continue
                k = (harness_arm_of(item_id), item_id, seed)
                h, c, l, m, i, s, p = (e038.get(k, {}), e039.get(k, {}), e040.get(k, {}), e044.get(k, {}), e043.get(k, {}), e045.get(k, {}), pg.get(k, {}))
                tx = e046.get(k, {}); sm = e048.get(k, {}); br = bl.get(k, {}); fa = e049.get(k, {}); px = e050.get(k, {})
                fps = _fin(h.get("fps"))
                seam_free = _fin(sm.get("seam_free")); seam_z = _fin(sm.get("seam_z"))
                recs[k] = dict(
                    label=label, base=base, tier=g.get("ref_novelty"), sided=g.get("sided", "one"), cell=g.get("cell"),
                    endpoint=g["endpoint"], reference=g.get("reference"),
                    ep_id_A=_fin(i.get("A_given_clip_mean")), ep_id_B=_fin(i.get("B_given_clip_mean")),
                    ho_id_A=_fin(h.get("identity_A")), ho_id_B=_fin(h.get("identity_B")),
                    ep_mot_A=_fin(m.get("A_agree")), ep_mot_B=_fin(m.get("B_agree")),
                    ho_mot_A=(_fin(h.get("motion_A")) if handoff_motion == "tracks" else None),
                    ho_mot_B=(_fin(h.get("motion_B")) if handoff_motion == "tracks" else None),
                    transport=_fin(br.get("bl_333")), transport_v4=_fin(p.get("transport_pct")),
                    motfid=_fin(l.get("det_motion_fidelity")),
                    vp_ref=_fin(l.get("videoprism_sim_ref")), clip_ref=_fin(l.get("clip_sim_ref")),
                    copy_pct=(100.0 * float(c["near_copy"]) if c.get("near_copy") is not None else None), copy_max=_fin(c.get("copy_max")),
                    seam_free=(100.0 * seam_free if seam_free is not None else None), seam_med=seam_z,
                    flow_mse=_fin(fa.get("flow_mse")), action_kl=_fin(fa.get("action_kl")),
                    smooth8=_fin(s.get("smooth_matched")), smooth_native=_fin(s.get("smooth_native")),
                    dyn_pxs=((l["dynamic_degree_mean_mag"] * fps) if (_fin(l.get("dynamic_degree_mean_mag")) is not None and fps) else None),
                    aesthetic=_fin(l.get("aesthetic")),
                    text_own=_fin(tx.get("text_own")), text_effect=_fin(tx.get("text_effect")),
                    ep_psnr_A=_fin(px.get("psnr_A")), ep_ssim_A=_fin(px.get("ssim_A")), ep_lpips_A=_fin(px.get("lpips_A")),
                    ep_psnr_B=_fin(px.get("psnr_B")), ep_ssim_B=_fin(px.get("ssim_B")), ep_lpips_B=_fin(px.get("lpips_B")))
                mr = mf.get(k, {})
                recs[k].update({col: _fin(mr.get(src)) for col, src in MF_COLS.items()})
                recs[k].update({col: _fin(br.get(src)) for col, src in BLEND_COLS.items()})
                if recs[k].get("look_u") is None:                      # the blend grid's (.50/.50/0) column IS Look_u (same channels, populations, ceilings)
                    recs[k]["look_u"] = _fin(br.get("Look_u (.50/.50/0)"))
    return recs


# ----------------------------------------------------------------------------- aggregation + rendering
def agg(vals, how):
    v = [x for x in vals if x is not None]
    if not v:
        return (None, 0)
    return ((float(np.median(v)) if how == "median" else float(np.mean(v))), len(v))


def cells_for(rows, cols):
    return [agg([r.get(c) for r in rows], CAT[c][3]) for c in cols]


def fmt_cell(m, n, n_all, key, best, md=False):
    if m is None:
        return "--"
    _, d, dec, _ = CAT[key]
    s = f"{m:.{dec}f}"
    if CLEAN:
        if best is not None and abs(m - best) <= 1e-9 and d != "none":
            s = ("**" + s + "**") if md else (r"\textbf{" + s + "}")
        return s
    thin = n < MIN_N
    if thin:
        s = (f"_{s}_ †") if md else (r"\textit{" + s + r"}$^{\dagger}$")
    elif best is not None and abs(m - best) <= 1e-9 and d != "none":
        s = ("**" + s + "**") if md else (r"\textbf{" + s + "}")
    if n != n_all:
        s += (f" (n={n})" if md else r"{\tiny\,($n{=}" + str(n) + "$)}")
    return s


def render_block(cells, labels, n_all, cols, md=False):
    best = []
    for ci, c in enumerate(cols):
        d = CAT[c][1]
        vals = [cells[l][ci][0] for l in labels if cells[l][ci][0] is not None and cells[l][ci][1] >= MIN_N]
        best.append(None if (d == "none" or not vals) else (max(vals) if d == "max" else min(vals)))
    out = []
    for l in labels:
        parts = [fmt_cell(cells[l][ci][0], cells[l][ci][1], n_all[l], c, best[ci], md) for ci, c in enumerate(cols)]
        out.append((l, parts, n_all[l]))
    return out


def header_rows(groups):
    cols = [c for _, cs in groups for c in cs]
    top = " & " + " & ".join(r"\multicolumn{" + str(len(cs)) + "}{c}{" + g + "}" for g, cs in groups) + ("" if CLEAN else " &") + r" \\"
    mids, start = [], 2
    for _, cs in groups:
        mids.append(r"\cmidrule(lr){" + f"{start}-{start + len(cs) - 1}" + "}"); start += len(cs)
    arrow = lambda c: ("" if CAT[c][1] == "none" or "uparrow" in dict(groups_flat(groups)).get(c, "") else (r"\,$\uparrow$" if CAT[c][1] == "max" else r"\,$\downarrow$"))
    name = lambda c: (CLEAN_HEADER.get(c, CAT[c][0]) if CLEAN else CAT[c][0])
    second = "arm & " + " & ".join(name(c) + arrow(c) for c in cols) + ("" if CLEAN else r" & $n$") + r" \\"
    return cols, top + "\n" + "".join(mids) + "\n" + second


def groups_flat(groups):
    return [(c, g) for g, cs in groups for c in cs]


def render_family(recs, folder: str, spec: dict, out_root: Path) -> Path:
    out = out_root / (folder + ("_sweep" if SWEEP else "") + ("_clean" if CLEAN else "") + SUFFIX); out.mkdir(parents=True, exist_ok=True)
    groups = spec["groups"]; cols, hdr = header_rows(groups)
    ncol = len(cols) + (1 if CLEAN else 2)
    tabspec = "l " + " ".join("c" * len(cs) for _, cs in groups) + ("" if CLEAN else " c")
    ncell = (lambda n: "") if CLEAN else (lambda n: f" & {n}")
    md = [f"# {spec['title'].capitalize()} tables (grid v3 preview)", "", f"_Generated {date.today().isoformat()} by `scripts/family_tables.py`._", ""]
    md_hdr = "| arm | " + " | ".join(CAT[c][0].replace("\\", "").replace("$", "") + (" ↑" if CAT[c][1] == "max" else (" ↓" if CAT[c][1] == "min" else "")) for c in cols) + " | n |"
    # ---- Table 1: own arms by tier
    L = [r"\begin{table}[H]", r"\centering",
         (r"\caption{\textbf{Own arms by tier.} Neutral prompt, grid v3 (HF-121f and ED-81f rows pooled per tier). Bold: best per column and tier; -- not measurable. "
          r"End columns exist on two-sided rows only (8 / 64 / 76 per arm and tier); Motion columns on rows whose 9-frame given clip actually moves (about half of the HF rows)." + (r" Guidance sweep: \segue{} w/o guidance is $w{=}1$, \segue{} is $w{=}6$." if SWEEP else "") + "}" if CLEAN else
          r"\caption{\textbf{" + spec["title"].capitalize() + r", own arms by tier.} Neutral prompt, grid v3, HF-121f and ED-81f rows pooled per tier. "
          r"Bold: best per column and tier; a per-cell $n$ where a column is defined on fewer rows; $n<" + str(MIN_N) + r"$ in italics$^{\dagger}$ (indicative); -- not measurable." + (r" Guidance sweep: \segue{} w/o guidance is $w{=}1$, \segue{} is $w{=}6$." if SWEEP else "") + "}"),
         r"\label{tab:" + folder + r"_own}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{" + tabspec + "}", r"\toprule", hdr, r"\midrule"]
    md += ["## Table 1 -- own arms by tier", md_hdr, "|---" * (len(cols) + 2) + "|"]
    for tier, tname in (("seen", "Seen"), ("unseen", "Unseen"), ("zero_shot", "Zero-shot")):
        own = (OWN_BASE if WITH_BASE else []) + own_arms()
        cells, n_all = {}, {}
        for label, _ in own:
            rs = [r for r in recs.values() if r["label"] == label and r["tier"] == tier]
            cells[label] = cells_for(rs, cols); n_all[label] = len(rs)
        tlabel = tname + (f" ($n{{=}}{max(n_all.values())}$ per arm)" if CLEAN else "")
        L.append(r"\multicolumn{" + str(ncol) + r"}{l}{\textit{" + tlabel + r"}} \\")
        md.append(f"| **{tname}** |" + " |" * (len(cols) + 1))
        for l, parts, n in render_block(cells, [l for l, _ in own], n_all, cols):
            L.append(TEX[l] + " & " + " & ".join(parts) + ncell(n) + r" \\")
        for l, parts, n in render_block(cells, [l for l, _ in own], n_all, cols, md=True):
            md.append(f"| {l} | " + " | ".join(parts) + f" | {n} |")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out / "tab_1.tex").write_text("\n".join(L) + "\n")
    # ---- Table 2: shared one-sided zero-shot set
    arms = EXT + (OWN_BASE if WITH_BASE else []) + (own_arms() if PRIOR_BASELINE else own_arms()[1:])
    per = {l: {(r["cell"], r["endpoint"], r["reference"], k[2]): r for k, r in recs.items()
               if r["label"] == l and r["tier"] == "zero_shot" and r["sided"] == "one"} for l, _ in arms}
    shared = None
    for l, _ in arms:
        ks = set(per[l]); shared = ks if shared is None else shared & ks
    cells = {l: cells_for([per[l][k] for k in shared], cols) for l, _ in arms}; n_all = {l: len(shared) for l, _ in arms}
    L = [r"\begin{table}[H]", r"\centering",
         (r"\caption{\textbf{Prior works vs.\ ours.} Shared one-sided zero-shot set, $n{=}" + str(len(shared)) + r"$ identical rows per arm "
          r"(prior works at their author-native prompt, ours neutral). Bold: best per column; -- not measurable for that arm (one-frame conditioning has no given motion; a one-sided set has no end endpoint).}" if CLEAN else
          r"\caption{\textbf{" + spec["title"].capitalize() + r", prior works vs.\ ours.} Shared one-sided zero-shot set, $n{=}" + str(len(shared)) +
          r"$ identical rows per arm; prior works at their author-native prompt, ours neutral. Bold: best per column; -- not measurable for that arm.}"),
         r"\label{tab:" + folder + r"_prior}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{" + tabspec + "}", r"\toprule", hdr, r"\midrule"]
    md += ["", f"## Table 2 -- prior works vs ours, shared one-sided zero-shot set (n = {len(shared)})", md_hdr, "|---" * (len(cols) + 2) + "|"]
    for l, parts, n in render_block(cells, [l for l, _ in arms], n_all, cols):
        L.append(TEX[l] + " & " + " & ".join(parts) + ncell(n) + r" \\")
    for l, parts, n in render_block(cells, [l for l, _ in arms], n_all, cols, md=True):
        md.append(f"| {l} | " + " | ".join(parts) + f" | {n} |")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out / "tab_2.tex").write_text("\n".join(L) + "\n")
    # ---- Table 3: shared two-sided zero-shot set (both endpoints given; the TEG block of the paper's main table)
    arms3 = OWN_BASE + EXT_TEG + own_arms()
    per3 = {l: {(r["cell"], r["endpoint"], r["reference"], k[2]): r for k, r in recs.items()
                if r["label"] == l and r["tier"] == "zero_shot" and r["sided"] == "two"} for l, _ in arms3}
    shared3 = None
    for l, _ in arms3:
        if not per3[l]:                                   # an arm without scored rows shows '--' and does not shrink the shared set
            print(f"[table3] {l}: no two-sided zero-shot rows yet", file=sys.stderr); continue
        ks = set(per3[l]); shared3 = ks if shared3 is None else shared3 & ks
    shared3 = shared3 or set()
    cells3 = {l: cells_for([per3[l][k] for k in shared3 if k in per3[l]], cols) for l, _ in arms3}; n_all3 = {l: len(shared3) for l, _ in arms3}
    L = [r"\begin{table}[H]", r"\centering",
         (r"\caption{\textbf{Both endpoints given.} Shared two-sided zero-shot set, $n{=}" + str(len(shared3)) + r"$ identical rows per arm. "
          r"The adapted baselines are two-endpoint models that cannot take a reference (refVFX can), so they receive the effect as an effect prompt; ours the neutral prompt. "
          r"Bold: best per column; -- not measurable for that arm (one-frame conditioning has no given motion; the VACE clip carries 6 start / 4 end frames at 16\,fps).}" if CLEAN else
          r"\caption{\textbf{" + spec["title"].capitalize() + r", both endpoints given (TEG).} Shared two-sided zero-shot set, $n{=}" + str(len(shared3)) +
          r"$ identical rows per arm; adapted baselines at the effect prompt (refVFX also its reference), ours neutral. Bold: best per column; -- not measurable for that arm.}"),
         r"\label{tab:" + folder + r"_teg}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{" + tabspec + "}", r"\toprule", hdr, r"\midrule"]
    md += ["", f"## Table 3 -- both endpoints given (TEG), shared two-sided zero-shot set (n = {len(shared3)})", md_hdr, "|---" * (len(cols) + 2) + "|"]
    for l, parts, n in render_block(cells3, [l for l, _ in arms3], n_all3, cols):
        L.append(TEX[l] + " & " + " & ".join(parts) + ncell(n) + r" \\")
    for l, parts, n in render_block(cells3, [l for l, _ in arms3], n_all3, cols, md=True):
        md.append(f"| {l} | " + " | ".join(parts) + f" | {n} |")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out / "tab_3.tex").write_text("\n".join(L) + "\n")
    # ---- metric key (compact) + preview
    pair = {"ep_id_B": "ep_id_A", "ho_id_B": "ho_id_A", "ep_mot_B": "ep_mot_A", "ho_mot_B": "ho_mot_A",
            "ep_psnr_B": "ep_psnr_A", "ep_ssim_B": "ep_ssim_A", "ep_lpips_B": "ep_lpips_A"}
    keys = []
    for c in cols:
        k = pair.get(c, c)
        if k in SHORT and k not in keys: keys.append(k)
    CLEAN_NOTE = {"Identity Endpoint A/B": "Identity Start / End", "Identity Hand-off A/B": "Identity Start / End hand-off",
                  "Motion Endpoint A/B": "Motion Start / End", "Motion Hand-off A/B": "Motion Start / End hand-off",
                  "Ref sim.\\ (VP)": "Ref.\\ sim.", "Ref sim.\\ (CLIP)": "Ref.\\ sim.\\ (CLIP)", "Seam $z$ med.": "Seam $z$",
                  "Motion smooth.": "Smoothness", "Smooth.\\ (8\\,fps)": "Smoothness (8\\,fps)", "Dyn.\\ (px/s)": "Dynamics (px/s)",
                  "Text": "Consistency (text)"}
    def _item(k):   # "Name: how" -> inline bold name + compact explanation (+ source)
        name, how = (SIMPLE if CLEAN else SHORT)[k].split(":", 1)
        name = name.strip()
        if CLEAN: name = CLEAN_NOTE.get(name, name)
        return r"\item \textbf{" + name + r"} --- " + how.strip()
    blocks = []
    for fam in ("Input fidelity", "Transition fidelity", "Quality"):
        ks = [k for k in keys if FAMILY_OF.get(k) == fam]
        if ks:
            blocks += [r"\textbf{" + fam + r"}", r"\begin{itemize}[leftmargin=*, nosep, itemsep=1pt]"] + [_item(k) for k in ks] + [r"\end{itemize}", r"\smallskip"]
    key_tex = "\n".join([(r"\newpage" if CLEAN else ""), r"\noindent\footnotesize", (r"\textbf{Columns}\par\smallskip" if CLEAN else ""), r"\begin{multicols}{2}"] + blocks + [r"\end{multicols}"])
    # preview-only bib entries: drop any key the paper's references.bib already carries (bibtex rejects repeated entries)
    import re as _re
    have = set(_re.findall(r"@\w+\{\s*([^,\s]+)\s*,", (PAPER / "references.bib").read_text())) if (PAPER / "references.bib").exists() else set()
    def _key(e):
        m = _re.match(r"@\w+\{\s*([^,\s]+)", e); return m.group(1) if m else None      # None = a leading comment chunk (kept)
    keep = [e for e in _re.split(r"(?=@\w+\{)", PREVIEW_REFS) if e.strip() and _key(e) not in have]
    (out / "preview_refs.bib").write_text("".join(keep))
    md += ["", "## Columns"] + [f"- {SHORT[k]}" for k in keys]
    (out / "TABLES.md").write_text("\n".join(md).replace("\\\\", "\\") + "\n")
    tex = "\n".join([r"% GENERATED by scripts/family_tables.py", r"\documentclass{article}", r"\usepackage{natbib}", r"\input{preamble}",
                     r"\usepackage{geometry}", r"\geometry{landscape, margin=1.5cm}", r"\usepackage{enumitem}", r"\usepackage{multicol}", (r"\hypersetup{hidelinks}" if CLEAN else ""), r"\begin{document}", r"\begin{center}",
                     (r"{\Large\bfseries \segue{} -- metric tables (grid v3)}\\[3pt]" if CLEAN else r"{\Large\bfseries \segue{} -- " + spec["title"] + r" (grid v3 preview)}\\[3pt]"),
                     (r"\small " + date.today().isoformat() + r". Two seeds per row; levels, no confidence intervals." if CLEAN else
                      r"\small Generated " + date.today().isoformat() + r" from the store evals over the feature store; a preview for owner review, not the paper."),
                     r"\end{center}", r"\vspace{0.3em}", r"\input{tab_1}", r"\vspace{0.6em}", r"\input{tab_2}", r"\vspace{0.6em}", r"\input{tab_3}", r"\vspace{0.6em}", key_tex,
                     ("" if CLEAN else r"\bibliographystyle{iclr2027_conference}"), ("" if CLEAN else r"\bibliography{references,preview_refs}"), r"\end{document}", ""])
    (out / "preview.tex").write_text(tex)
    env = dict(os.environ); env["PATH"] = TEXBIN + ":" + env.get("PATH", "")
    env["TEXINPUTS"] = f".:{PAPER}:{PAPER}//:"; env["BIBINPUTS"] = env["TEXINPUTS"]; env["BSTINPUTS"] = env["TEXINPUTS"]
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error", "preview.tex"], cwd=out, env=env, capture_output=True, text=True, timeout=600)
    subprocess.run(["latexmk", "-c", "preview.tex"], cwd=out, env=env, capture_output=True, text=True)
    print(f"[{folder}] build", "OK" if r.returncode == 0 else "FAILED", "->", out / "preview.pdf")
    if r.returncode != 0:
        print("\n".join(r.stdout.splitlines()[-25:]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES), help="subset of family folders to build")
    ap.add_argument("--handoff-motion", choices=["none", "tracks"], default="none",
                    help="fill hand-off motion from the tracks-based eval 038 (default: leave '--' until the flow version exists)")
    ap.add_argument("--out-root", default=str(PREVIEW))
    ap.add_argument("--prior-baseline", action="store_true", help="also list the LTX-2 baseline LoRA in the prior-works table")
    ap.add_argument("--with-base", action="store_true", help="also list the LTX-2 no-reference rows (neutral + effect prompt = text-only inbetweening) in both tables")
    ap.add_argument("--clean", action="store_true", help="clean version: no per-cell n / dagger / n column, simple headers; writes <family>_clean/")
    ap.add_argument("--sweep", action="store_true", help="add the DCG guidance-weight rows (SEGUE w=1.5, w=3) to both tables; writes <family>_sweep[_clean]/")
    ap.add_argument("--suffix", default="", help="write to <family>[_sweep][_clean]<suffix>/ (e.g. _withbase for an alternate-roster variant)")
    args = ap.parse_args(argv)
    global PRIOR_BASELINE, WITH_BASE, CLEAN, SWEEP, SUFFIX
    PRIOR_BASELINE = args.prior_baseline; WITH_BASE = args.with_base; CLEAN = args.clean; SWEEP = args.sweep; SUFFIX = args.suffix
    recs = collect(args.handoff_motion)
    print(f"[collect] {len(recs)} generations")
    for f in args.families:
        render_family(recs, f, FAMILIES[f], Path(args.out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
