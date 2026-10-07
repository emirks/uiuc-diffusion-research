# Null-transition line of work — ledger of what was done and what was measured (2026-08-24 → 2026-09-13)

Written 2026-09-13 at the owner's request before a fresh start on "the best null solution of the CTT task".
Facts and numbers only; no readings, verdicts or recommendations are carried. Each campaign folder keeps its own
attributed readings; treat those as claims to re-test. Every number below is quoted from the file named in its section.

## 0. Instrument and terms (what the numbers measure)

- Endpoint-line (interpolation) family: x(t) = (1−α(t))·A + α(t)·B. α a step = cut; α linear = dissolve; α constant = freeze.
- DR = median, over interior frames, of the distance to the A→B line, divided by the anchor gap; pixel space, 128 px, blur 1.0.
  On-line convention DR ≤ 0.12 (θ_guard set in the Aug-24 Bar I). Sensitivity to θ is in re-measure TABLES §B.
- τ = projection coordinate on the A→B line (0 at A, 1 at B). M = share of interior frames with τ ∈ [0.25, 0.75]
  (synthetic dissolve 0.50, cut/freeze ≈ 0). R = p95−p5 of τ. S = Spearman(τ, frame index).
- Aug-24 classes: STATIC (gap < 0.12); on-line branch DR ≤ 0.12 → DISSOLVE (M ≥ 0.25, S ≥ S_thr), CUT (M < 0.25, R ≥ 0.5),
  FREEZE (M < 0.25, R < 0.5); else REAL. M_thr 0.25, S_thr 1.0, calibrated on synthetic floors from 223 GT endpoint pairs.
- Windows: 121 frames at 24 fps; prefix 9 (start anchor a_idx 8), suffix 8 (end anchor b_idx 113 when two-sided);
  start-only clips are scored against their own last frame (b_idx 120).
- Abrupt-cut detector (2026-09-12): over frames a..b, step_share = largest adjacent-frame step / total path;
  step/gap = largest step / anchor gap. abrupt := step_share > 0.06 AND step/gap > 0.56 (p95 of 240 random GT clips;
  GT medians 0.03 and 0.28).
- jump3 = largest rise of distance-to-A over 3 frames, in gap units. transit frames = interior frames with distance-to-A in (0.15, 0.85) gap.
- Endpoint distances: DINO = 1−cos DINOv2 CLS; CLIP = 1−cos ViT-B/32; pixel = gap_rel (anchor-to-anchor L2 in instrument units).
- Code: `misc/2026-09-08_collapse_remeasure/instrument.py` (load_matrix, window_indices, measure); Aug-24 `lerp_metrics.py`, `phase2_classify.py`.

## 1. Aug-24 campaign — `misc/2026-08-24_lerp_collapse/` (REPORT.md, DOSSIER.md; advised)

Done: Phase 0 instrument calibration; Phase 1 stratified DR on store generations (both-anchor vs start-only rows within each arm);
Phase 2 classifier + class table; Tier-2 paired regeneration; Step-C one-step x0-hat probe.
Data: CTT eval grid v2 = 152 rows (40 two-sided = prefix 9 + suffix 8; 112 one-sided = prefix 9), seeds 42/43. Arms: base_cond
(base LTX-2, no adapter), ctt_v3, refvfx (Wan2.1-FLF2V + refVFX LoRA + CausVid), vap (Wan I2V), vfxmaster (CogVideoX-Fun I2V),
base_prompt (no anchors). Prompt: V-NEUTRAL = the start-scene sentence only ("{S1}."), no sksz token, no effect clause, no end-scene text
(`store/gens/005_base_cond/02_neutral__dai/meta.v1.yaml`).

1.1 Bar I: synthetic-dissolve DR p95 0.016; GT-real DR p5 0.240 (median 0.528); Cliff's δ 1.0; θ_guard 0.12; 0 of 223 GT clips are dissolves.
1.2 Phase 1 (C = DR calibrated to floor 0 / ceiling 1; neutral prompt): base_cond both n=80 C median 0.673 [0.358, 1.079], 18.8 % at C ≤ 0.33, DR 0.358, PR 2.82;
    base_cond start n=220 C 1.093, DR 0.576; ctt_v3 both 1.209 / start 0.886; base_prompt 0.950; refvfx both 0.882; vap 0.795; vfxmaster 0.822.
    Pre-registered Claim-1 bar (C median ≤ 0.25) not met.
1.3 Classifier certification: synthetic dissolve 223/223 → DISSOLVE, cut 223/223 → CUT, freeze 223/223 → FREEZE, GT-real 221/223 → REAL;
    known misfire: DISSOLVE on near-static low-gap I2V clips.
1.4 Class table (non-foreign cells, neutral): base_cond both n=52: DIS 0 / CUT 15 / FRZ 0 / REAL 37, shortcut 28.8 % [18.3, 42.3];
    base_cond start n=160: CUT 8, 5.0 % [2.6, 9.6]; base_prompt n=52: CUT 5, 9.6 %; refvfx both n=212: 0; ctt_v3 both n=52: 0 [0, 6.9];
    ctt_v3 start n=160: CUT 4 / FRZ 1, 3.1 %. C1′ both − start = +23.8 pp, Newcombe 95 % CI [12.4, 37.5]; all cells 18.75 % vs 5.0 % (+13.7 pp).
    Foreign share of base two-sided cells 28/80 = 35 %.
1.5 Tier-2 (the 40 two-sided rows regenerated start-only: suffix dropped, prefix byte-identical, seeds 42/43; 52 non-foreign pairs as counted then):
    shortcut 28.8 % → 9.6 %; McNemar discordant 10 / 0; one-sided p 0.00098. `tier2_report.json`.
1.6 Step-C (1-step Euler x0-hat, 20 clean two-sided rows, seeds 42/43): both-anchor DR 0.589, M 0.755, S 0.249; start-only DR 0.484, M 0.018, S 0.118;
    fraction closer to lerp(start,end) than to the 40-step sample 1.00. Bars: S ≥ 0.8 not met; DR < 0.12 not met; closer-to-lerp met. `stepc_montage.png`.
1.7 Owner correction 2026-08-31: the refvfx arm is a trained pipeline; no cross-model claim is made from it.

## 2. Sep-08 re-measurement of store generations — `misc/2026-09-08_collapse_remeasure/` (TABLES.md §A–G, REPORT.md, per_clip.csv, endpoint_covariates.csv)

Done: the continuous instrument run over every stored base and trained generation; md5 de-duplication; foreign/static exclusion;
endpoint covariates (DINO/CLIP/pixel/motion); paired and unpaired contrasts with endpoint-cluster bootstraps; regressions; Tier-2 re-check.
Scope: 6663 rows scored; 584 byte-identical duplicates; 2186 foreign (davis); 143 static; 4031 unique clean clips. Arms base_cond,
dualforce_control, dualforce_dcg w ∈ {1, 1.5, 3, 6}; grids v2, v3, v3ed81; prompts neutral / effect; conditioning both / start.

2.1 base_cond levels (unique clean; DR median, M median, on-line share, classes DIS/CUT/FRZ):
    v2 neutral both n=30: 0.211 [0.136, 0.306], 0.06, 26.7 % (8), 0/8/0 · v2 neutral start n=80: 0.455, 0.27, 5.0 % (4), 0/4/0
    v2 effect both n=52: 0.385, 0.25, 5.8 % (3), 0/3/0 · v2 effect start n=160: 0.310, 0.23, 10.6 % (17), 0/17/0
    v3 neutral both n=38: 0.211, 0.06, 21.1 % (8), 0/8/0 · v3 neutral start n=119: 0.506, 0.34, 3.4 % (4)
    v3 effect both n=94: 0.385, 0.23, 8.5 % (8), 1/7/0 · v3 effect start n=277: 0.337, 0.26, 7.9 % (22)
    v3ed81 effect start n=136: 0.219, 0.16, 26.5 % (36), 1/35/0 · v3ed81 neutral start n=32 (70 static excluded): 0.100, 0.09, 53.1 % (17), 2/3/12
2.2 Trained levels: dualforce_control both cells DR 0.616–0.693, on-line 0 % in every both cell; start cells 0.446–0.480, on-line 1.8–2.5 %.
    dcg w6 both 0.828–0.865, on-line 0 %; start 0.573–0.621, on-line 0–2.5 %.
2.3 θ sensitivity, base v2 neutral both (n=30): θ 0.06 → 7 %, 0.08 → 7 %, 0.10 → 10 %, 0.12 → 27 %, 0.16 → 27 %, 0.20 → 43 %, 0.25 → 57 %;
    v2 neutral start (n=80): 0 %, 1 %, 4 %, 5 %, 6 %, 8 %, 14 %.
2.4 Paired contrasts (same endpoint and seed; ΔDR median [cluster CI]; on-line first → second; flips first-only/second-only):
    END ANCHOR base both − start (v2 neutral, Tier-2 regen) 30 pairs / 15 endpoints: −0.052 [−0.175, 0.018], Cliff −0.33, Wilcoxon 0.0197, 26.7 % → 10.0 %, 5/0.
    PROMPT base effect − neutral: v2 both 52 pairs +0.139 [0.106, 0.173], 5.8 % vs 28.8 %, flips 1/13 (p 0.002); v3 both 94 pairs +0.148, 8.5 % vs 26.6 %, 2/19;
    v2 start 156 pairs −0.120, 9.0 % vs 5.1 %, 10/4; v3 start 271 pairs −0.119, 6.3 % vs 3.0 %, 14/5; v3ed81 start 37 pairs +0.079, 32.4 % vs 56.8 %, 3/12.
    TRAINING base − control: v2 neutral both 52 pairs −0.394 [−0.453, −0.285], Cliff −1.00, 28.8 % → 0 %, 15/0; v3 neutral both 94 pairs −0.394, 26.6 % → 0 %, 25/0;
    v2 effect both −0.265, 5.8 % → 0 %; v3 effect both −0.265, 8.5 % → 0 %; v2 neutral start +0.005 (p 0.71), 5.1 % → 1.3 %; v3 neutral start −0.003 (p 0.93);
    v2 effect start −0.103, 10.6 % → 2.5 %; v3 effect start −0.088; v3ed81 neutral start −0.333, 56.8 % → 2.7 %, 20/0; v3ed81 effect start −0.217, 26.5 % → 2.2 %, 34/1.
    GUIDANCE dcg − control (v2 neutral both, 26 pairs): w1 +0.001 (p 0.80); w1.5 +0.052; w3 +0.133; w6 +0.211 [0.165, 0.325]; v3 neutral both w6 +0.253; v2/v3 effect both w6 +0.173 / +0.169;
    start cells w6 +0.110 to +0.157; on-line 0 % → 0 % in every both cell.
2.5 Unpaired both vs start within base (different endpoints per stratum): v2 neutral 0.211 / 0.455, Δ −0.244 [−0.328, −0.148], Cliff −0.62, on-line +21.7 pp [0.4, 45.0];
    v3 neutral 0.211 / 0.506, −0.296 [−0.346, −0.202], +17.7 pp [1.6, 36.6]; v2 effect 0.385 / 0.310, +0.075 [−0.041, 0.185], −4.9 pp; v3 effect +0.048, +0.6 pp.
    Trained arms: both minus start +0.149 to +0.253 in every cell, on-line 0 % vs 1.2–2.5 %.
2.6 Per-endpoint propensity, base both neutral (2 seeds): v2 15 endpoints: 33 % with ≥ 1 on-line generation, 20 % with all; v3 19 endpoints: 26 %, 16 %; median per-endpoint share 0 %.
2.7 Covariates (Spearman with DR): base both neutral v2 DINO −0.48 (p 0.007), pixel −0.41, CLIP −0.22 (n.s.), motion_prefix +0.44; v3 DINO −0.39 (0.016), pixel −0.43, CLIP −0.08;
    base both effect v2 DINO −0.38 (0.005), v3 −0.19 (0.065); start cells DINO −0.15 to +0.06 (n.s.), motion_prefix +0.21 to +0.47.
    On-line by DINO tercile: v2 neutral both 20/10/50 %; v3 neutral both 0/25/42 %; v2 effect both 0/0/19 %; v3 effect both 0/6/20 %.
    Regression R1 (neutral, n=157, 79 endpoints): both-endpoint −0.226 [−0.307, −0.151]; DINO +0.027 [−0.116, +0.161]; motion_prefix +0.464.
    R2 (both prompts, n=554): both +0.097 [−0.015, +0.206]; effect −0.099 [−0.148, −0.035]; both×neutral −0.312 [−0.418, −0.215]; DINO −0.052 (n.s.).
    R3 (within base both neutral, n=38): DINO −0.342 [−0.743, +0.069]; motion_suffix +0.742 [−0.221, +1.760].
2.8 §G Tier-2 re-check (2026-09-12; md5 dedup: 36 unique generations per side over 18 endpoints, 3 davis endpoints foreign; `tier2_pairs.csv`):
    clean 30 pairs / 15 endpoints: DR 0.211 → 0.345; ΔDR −0.052 [−0.173, +0.018]; 20/30 pairs negative; Wilcoxon one-sided 0.010; on-line 26.7 % (8) → 10.0 % (3);
    flips 5/0 (sign test 0.031); classes CUT 8 + REAL 22 → CUT 3 + REAL 27; abrupt 50 % (15) → 53 % (16), flips 5/6.
    all incl. davis 36 pairs: ΔDR −0.025 [−0.158, +0.058], p 0.093, on-line 22.2 % → 8.3 %, flips 5/0; davis only 6 pairs: ΔDR +0.199, 0 % on-line both sides.
    No DISSOLVE or FREEZE on either side; M median of on-line clips 0.06 / 0.05. Flip endpoints: air_bending_1 ×2, melt_transition_2, hero_flight_5, shadow_smoke_7;
    shadow_smoke_2 on-line in all four clips.

## 3. Sep-08 from-scratch probe — `misc/2026-09-08_collapse_probe/` (results/REPORT.md, TABLES.md §G–I, results/*.csv)

Done: 30 real 9-frame start clips (in-place classes, ≤ 2 per class); 40 three-part prompts (start caption verbatim from the grid + change clause + end caption):
30 scene changes through 30 distinct named mechanisms, 10 in-place controls; seeds 42/43/44; base LTX-2 with the base_cond_neutral recipe (30 steps, guidance 4.0,
stg 1.0, 480×640×121). R1 = start anchor + full prompt; R2 = start + R1's own last frames as end anchor + full prompt; R3 = same anchors + captions only.
All three runs scored against the same two frames (start anchor, R1's frame 120). Jobs 3115053 / 3115054 / 3115382, ≈ 5 GPU-h. Viewer `outputs/viewers/collapse_probe/`.

3.1 Levels (DR median [IQR], M median, on-line): scene change n=90 — R1 0.367 [0.234, 0.499], 0.23, 6.7 %; R2 0.334 [0.234, 0.448], 0.21, 6.7 %; R3 0.206 [0.112, 0.291], 0.06, 27.8 %.
    in-place n=30 — R1 0.278, 0.24, 0 %; R2 0.291, 0.21, 0 %; R3 0.286, 0.18, 0 %. Realized DINO change of the R1 witnesses: scene change 0.978 [0.958, 1.008]; in-place 0.451 [0.290, 0.637].
3.2 Paired (same prompt and seed): R2−R1 scene −0.008 (55/90 negative, p 0.013), in-place −0.025 (20/30, 0.033); R3−R2 scene −0.070 (79/90, 6e−14), in-place −0.005 (23/30, 0.041);
    R3−R1 scene −0.100 (78/90, 1e−13), in-place −0.029 (22/30, 0.004). R3−R2 per seed: −0.094 (28/30), −0.038 (24/30), −0.068 (27/30); 22/30 prompts toward the line in all three seeds, 0 away.
3.3 Grid vs probe levels: grid base neutral both DR 0.211 / M 0.06 / on-line 21–27 %; probe R3 scene 0.206 / 0.06 / 27.8 %; grid effect both 0.385; probe R2 0.334.
3.4 §G distances: tier medians DINO 0.98 [0.96, 1.01] scene vs 0.45 [0.29, 0.64] in-place; CLIP 0.50 vs 0.22; pixel 0.93 vs 0.75.
    All 120 pairs, Spearman with DR: R3 DINO −0.24 (0.009), CLIP −0.19 (0.034), pixel −0.34 (<0.001); R2 −0.08 / −0.02 / −0.26; R1 −0.07 / −0.01 / −0.23;
    ΔDR(R3−R2) DINO −0.24, CLIP −0.28, pixel +0.07; ΔDR(R2−R1) +0.06 / +0.10 / −0.16.
    Scene change only (90): R3 −0.12 (n.s.) / −0.04 / −0.29; R2 −0.33 / −0.21 / −0.32; R1 −0.36 / −0.22 / −0.31. In-place (30): R3 +0.23 / +0.20 / −0.18.
    R3 on-line by tercile, pooled 120: DINO 5 / 28 / 30 %; CLIP 10 / 30 / 22 %; pixel 5 / 25 / 32 %. Within scene change by CLIP tercile: 33 / 20 / 30 %, DR 0.16 / 0.24 / 0.20.
3.5 §H members: scene-change cut-like (M < 0.15) R1 1 %, R2 2 %, R3 26 %; mixed 6 / 4 / 2 %; off-line 93 / 93 / 72 %; in-place 0 %.
    R3 scene on-line n=25: M median 0.05 [0.02, 0.07]; 92 % cut-like; 0 % dissolve-like (M > 0.30); τ_med median 0.99 [0.87, 1.00]; first τ > 0.5 at 0.34 of the interior [0.23, 0.40];
    near-A share 0.30, near-B share 0.64; mean 10-bin τ profile 0.03 0.14 0.21 0.54 0.77 0.91 0.98 0.99 1.00 1.00; 0/25 hold A to the end; 3/25 jump immediately.
    Off-line R3 scene clips (65): crossing 0.27, near-B 0.69, DR 0.19–0.33. R1/R2 on-line clips (n=6 each) M ≈ 0.2. ρ(M, DINO) −0.02, ρ(M, CLIP) −0.20 (0.06), ρ(M, pixel) −0.04.
3.6 §I abrupt detector: scene change abrupt R1 2 %, R2 4 %, R3 40 % (R3: on-line 28 %; abrupt & on-line 12 %; abrupt & off-line 28 %; on-line & not abrupt 16 %); in-place 0 / 10 / 10 %.
    Paired scene 90: R2→R3 33 become abrupt / 1 back; R1→R2 2 / 0. R3 abrupt clips cross at 0.23 of the interior; abrupt & off-line DR 0.28, path/gap 7.4; on-line-not-abrupt step_share 0.08, M 0.06;
    12/30 prompts abrupt in ≥ 2 seeds, 2/30 in all 3. Spearman(step_share, distance) within scene change: R3 DINO +0.19 (0.07), CLIP +0.15, pixel +0.03; R2 DINO +0.18 (0.09); R1 pixel +0.22 (0.04).
    Abrupt by DINO tercile R3 37 / 37 / 47 %; by CLIP 37 / 43 / 40 %.
3.7 Not run in the probe: start-only + captions-only (R0); end-only; trained arms on these inputs; any text condition other than full / captions-only.

## 4. Sep-12/13 distance-vs-cut addendum — `misc/2026-08-24_lerp_collapse/distance_vs_cut/` (TABLES.md, cutstats.csv, both_anchor_with_dist.csv, both_neutral_cut_profiles.csv, transit_profiles.csv, strips/)

Done: abrupt detector over all 1238 unique base_cond clips; the 222 unique both-anchor base clips (v2/v3 × neutral/effect) joined to endpoint distances;
per-frame distance-to-A / distance-to-B profiles; transit frames and jump3 against the GT clips of the same 19 clean endpoints; frame strips. CPU only.

4.1 Clean unique clips (on-line CUT class / abrupt / DR median): both neutral v2 n=30: 27 % (8) / 50 % (15) / 0.21 · both neutral v3 n=8: 0 % / 62 % (5) / 0.22 ·
    both effect v2 n=52: 6 % (3) / 10 % (5) / 0.39 · both effect v3 n=46: 9 % (4) / 9 % (4) / 0.39 · start neutral v2 n=82: 5 % (4) / 13 % (11) / 0.46 · start neutral v3 n=40: 0 % / 22 % (9) / 0.56 ·
    start effect v2 n=160: 11 % (17) / 2 % (4) / 0.31 · start effect v3 n=140: 6 % (8) / 6 % (8) / 0.38 · start effect v3ed81 n=136: 26 % (35) / 7 % (10) / 0.22 · start neutral v3ed81 n=102: 3 % (3) / 5 % (5) / 0.67 ·
    Tier-2 start regen neutral v2 n=30: 10 % (3) / 53 % (16) / 0.34. All cells: both neutral v3 n=34: CUT 29 % (10), abrupt 71 % (24).
    Abrupt among on-line clips: both neutral v2 5/8; among off-line: 10/22. Class split over all 222 both-anchor base clips: REAL 194, CUT 27, DISSOLVE 1; M of on-line clips 0.06–0.16 per cell.
4.2 Paired Tier-2 abrupt flag (30 clean pairs): both-only 5, start-only 6, both 10, neither 9. On-line flips 5/0.
4.3 Transit geometry (median transit frames; share transit ≤ 3; jump3 median; share jump3 ≥ 0.7; frames near A; near B):
    GT clips n=19: 26; 0 %; 0.24; 0 %; 0.06; 0.04 · both/effect n=98: 17; 8 %; 0.31; 11 %; 0.06; 0.19 · both/neutral n=38: 16; 26 %; 0.45; 29 %; 0.08; 0.18 ·
    start regen/neutral n=30 (own-gap units): 7; 40 %; 0.69; 47 %; 0.08; 0.09.
4.4 Profiles, both/neutral clean (38 clips): post-cut segment distance-to-A 0.98–1.19 gap in every clip; pre-cut distance-to-A 0.00–0.83; post-cut distance-to-B 0.06–0.12 for CUT-class clips,
    0.03–0.76 for REAL-class clips; suffix distance-to-B 0.01–0.42. Largest-step position (neutral abrupt clips, n=20): quartiles 0.17 / 0.37 / 0.43 of the interior; 26 of the 29 abrupt clean
    both-anchor clips have it inside (0.1, 0.9). Number of steps > half the largest: off-line abrupt 1 in 9/15 clips; on-line abrupt {1, 6, 9, 9, 14}.
4.5 DINO terciles (edges 0.84 / 0.94), both clean, CUT / abrupt / DR: neutral close 20 clips, 10 endpoints: 15 % / 50 % / 0.26; mid 8, 4: 0 % / 50 % / 0.26; far 10, 5: 50 % / 60 % / 0.15;
    effect close 54, 10: 2 % / 6 % / 0.43; mid 18, 4: 0 % / 6 % / 0.40; far 26, 5: 23 % / 19 % / 0.27.
    Spearman(DR, DINO) neutral −0.39 (0.016), effect −0.22 (0.032). Median DINO of CUT vs other clips: neutral 0.94 vs 0.77 (MW 0.039); effect 0.94 vs 0.83 (0.028).
    Abrupt vs other: neutral 0.84 vs 0.80 (0.61); effect 0.95 vs 0.83 (0.019). CLIP: CUT vs other 0.41 vs 0.33 (0.023) neutral, effect n.s.; pixel: Spearman(DR) −0.43 (0.006) neutral, −0.29 (0.003) effect,
    CUT vs other n.s. Distance coverage: DINO 0.50–1.03 over 19 endpoints, 2 seeds.
4.6 Strips: display_transition_1 s42 both = A held to frame 48, one-frame cut, B's scene with motion to the anchored end pose (post-cut distance-to-B 0.44, suffix 0.13); its start-only twin cuts near
    frame 25 and again near frame 80 into two unrelated scenes. hero_flight_5 s42 both = cut at 0.08 of the interior, B held (CUT class). Multithreaded cv2 decode on the login node produced corrupt strips; use cv2.setNumThreads(1).
4.7 Veo 3.1 documentation: no "distant frames → abrupt cut" sentence found on the Gemini API Veo page, the Vertex first-and-last-frames page or the Cloud prompting guide; third-party guides state it.

## 5. Text conditions that exist in the data

- V-NEUTRAL (grid `*neutral*` variants; Aug-24 headline and Tier-2): start-scene sentence only.
- effect (grid `*effect*` variants): the effect clause stated.
- Probe full: start caption + change clause + end caption. Probe captions-only: start caption + end caption.
- Not present anywhere: empty prompt; both captions with an abstract cue ("transitions into"); start-only + captions-only; end-only.

## 6. Design ideas raised by the owner in this line, not run

- Pick two endpoints independently → generate with both anchors → regenerate with the end anchor dropped, same seeds (exists only as the grid-v2 neutral Tier-2, seeds 42/43).
- Endpoint-distance spectrum: same-video pairs through GT transition endpoints to cross-video pairs; measure jump / on-line / transit against DINO, CLIP and pixel distance.
- Text conditions for the "default": captions only vs abstract cue vs empty prompt.
- Measure abrupt cuts separately from on-line lerp, and use that failure mode to inform the null-operator choice.
- Stated concern: R2/R3 use R1's own ending as the end anchor.

## 7. Code, data and compute facts

- Store generations: `store/gens/005_base_cond/{01_effect,02_neutral,04_neutral_v3,05_neutral_v3ed81,06_effect_v3,07_effect_v3ed81,tier2_start}__dai/videos/`; base_cond ignores the reference, so rows sharing an endpoint are byte-identical (dedupe by md5).
- GT clips: `data/processed/transitions_std121/<class>/<class>_<k>.mp4` (4176 clips, class subfolders; glob recursively).
- Generation: `eval_ladder/run_gen.py` (conditioning is a pure function of the row's `sided`; dropping the suffix = start-only). Probe pipeline: `misc/2026-09-08_collapse_probe/{_probe_common.py, build_r1.py, splice_r1.py, job_*.sbatch, score_probe.py, scripts/distance_corr.py}`; gotchas in memory `collapse-probe-campaign.md`.
- Analysis env: `source $LAB/envs-aarch64/activate` (login python3 is 3.6). Viewer server: tmux session `viewer`, port 8017, repo root.
- Commits: ad0064e (probe viewer), be691e7 (probe §G), 79a49aa / 9a5d72d (probe §H), dcf835b (probe §I), 62247ba (re-measure §G), plus the 2026-09-12 distance_vs_cut addendum on branch bneck_redesign.
- Base LTX-2 generation cost on DeltaAI GH200: ≈ 45 s per 121-frame clip.
