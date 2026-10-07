#!/usr/bin/env python
"""paper_tables.py — fill the paper's five result tables from the store (metrics v5, D3).

Renders, in Ozgur's exact layout, the measured cells of
  papers_drafts/ctt_iclr2027/tables/{tab_main, tab_ablation, tab_ablationtiers, tab_weight, tab_isolation}.tex
from the store evals over the feature store, reusing scripts/family_tables.py collect() + means so every
cell equals the corresponding metrics_v2_gridv3(_clean/_sweep_clean) cell:

  Transport   = eval 047  (bl_333, the .33/.33/.33 channel blend)
  Motion fid. = eval 040  (det_motion_fidelity)
  Ref sim.    = eval 040  (videoprism_sim_ref)
  Seam-free % = eval 048  (seam_free, physical windows)
  Text cons.  = eval 046  (text_own)   -- lower is better in the paper tables
  Action KL   -- do not exist: kept as \\ph{--}.

N = neutral prompt arm, E = effect / author-native prompt arm, on the SAME rows; Delta = |E - N| of the printed
(rounded) values; unrounded levels in LEVELS.md. The `own_effect` roster arms supply the E columns of our arms. Rows / key sets:
  tab_main TEG   : shared two-sided zero-shot set (family_tables Table 3, n=76)
  tab_main VET   : shared one-sided zero-shot set (family_tables Table 2, n=366); ALL of tab_isolation too
  tab_ablation   : zero-shot tier, HF+ED pooled, one- and two-sided together (Table 1 logic, n=442)
  tab_ablationtiers : seen (n=52) / unseen (n=274), Table 1 logic
  tab_weight     : Table 1 logic per tier, for w in {1,1.5,3,6}

Each table's leading `%` comment block is kept verbatim; one dated block (marker never duplicated on
re-runs) is appended. Only the numeric value cells of the recognised data rows change; captions, column
specs, \\tabsec gray rows, \\ph placeholders, the dashed rule and the conditioning-1 row stay byte-for-byte.

Writes papers_drafts/_preview/paper_tables_v5/{five .tex} + LEVELS.md + preview.pdf.
`--apply` backs the five current files up to misc/2026-09-23_metrics_v5/paper_tables_before_fill/ and copies
the rendered files over papers_drafts/ctt_iclr2027/tables/. CPU only.
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import family_tables as FT  # noqa: E402

PAPER = REPO / "papers_drafts" / "ctt_iclr2027"
TABLES = PAPER / "tables"
PREVIEW = REPO / "papers_drafts" / "_preview" / "paper_tables_v5"
BACKUP = REPO / "misc" / "2026-09-23_metrics_v5" / "paper_tables_before_fill"
TEXBIN = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/texlive/bin/aarch64-linux"
TODAY = date.today().isoformat()

DEC = {"transport": 1, "motfid": 1, "flow_mse": 2, "action_kl": 2, "vp_ref": 1, "seam_free": 1, "text_own": 1}
# Ozgur 2026-09-23 23:36 ("0.xxx olanlari 100 le carp, (x100) koy yanina"): the cosine-type 0.xxx metrics are printed
# x100 with one decimal and "(x100)" in the header; the store levels stay unscaled (LEVELS.md lists both).
SCALE = {"motfid": 100.0, "vp_ref": 100.0, "text_own": 100.0}
DIRN = {"transport": "max", "motfid": "max", "flow_mse": "min", "action_kl": "min",
        "vp_ref": "max", "seam_free": "max", "text_own": "min"}
MARKER = "% FILLED 2026-09-23 (metrics v5, scripts/metrics_v5/paper_tables.py):"

# ---- pixel-endpoint preview table (R6, --pixel-endpoint-table; NOT applied to the paper) ------------
# Mirrors tables/tab_endpoint.tex rows/style; prior works at their effect/author-native arm, SEGUE = the
# neutral dualforce_dcg_w6, on the shared zero-shot sets (shared3 n=76 / shared2 n=366). Base LTX-2 = the
# effect-prompt base (base_cond_effect), verified against tab_endpoint (text 23.8 / smooth 97.9 / aesth 4.668).
PIXEL_TEG = [(r"Base \ltx{}", "base_cond_effect"), ("VACE", "wan_vace"), (r"\refvfx{}", "refvfx_teg"),
             ("Wan2.1 FLF2V", "wan_flf2v"), (r"\segue{}", "dualforce_dcg_w6")]
PIXEL_VET = [(r"\vap{}", "vap_author_native"), (r"\vfxmaster{}", "vfxmaster_author_native"),
             (r"\refvfx{}", "refvfx_author_native"), (r"\segue{}", "dualforce_dcg_w6")]
PIXEL_ADDLINE = {"dualforce_dcg_w6"}      # \addlinespace[2pt] before SEGUE in each block (as tab_endpoint)
PIXEL_COLS = ["ep_psnr_A", "ep_ssim_A", "ep_lpips_A", "ep_psnr_B", "ep_ssim_B", "ep_lpips_B"]
PIXEL_SCALE = {"ep_psnr_A": 1.0, "ep_ssim_A": 100.0, "ep_lpips_A": 100.0, "ep_psnr_B": 1.0, "ep_ssim_B": 100.0, "ep_lpips_B": 100.0}
PIXEL_DEC = {c: 1 for c in PIXEL_COLS}
PIXEL_DIR = {"ep_psnr_A": "max", "ep_ssim_A": "max", "ep_lpips_A": "min", "ep_psnr_B": "max", "ep_ssim_B": "max", "ep_lpips_B": "min"}
# window-mean levels (LEVELS_pixel.md) reported for the clip-conditioned arms only
PIXEL_WINDOW_ARMS = [("SEGUE HF (dualforce_dcg_w6, v3)", "dualforce_dcg_w6", "HF"),
                     ("SEGUE ED (dualforce_dcg_w6, ed81)", "dualforce_dcg_w6", "ED"),
                     ("VACE (wan_vace)", "wan_vace", "VACE16")]

# column layout of the two table families (Action KL now filled from eval 049, both lower-better)
COLS6 = ["transport", "motfid", "action_kl", "vp_ref", "seam_free"]   # owner 2026-09-23 23:30: Flow MSE removed from the paper tables           # tab_main / ablation / ablationtiers (N,E,Delta each)
COLS7 = COLS6 + ["text_own"]


# ----------------------------------------------------------------------------- store levels
class Levels:
    """family_tables collect() means, plus the shared zero-shot key sets, keyed by roster arm id."""

    def __init__(self):
        FT.CLEAN = True
        FT.SWEEP = False
        FT.WITH_BASE = False
        FT.PRIOR_BASELINE = False
        FT.SUFFIX = ""
        self.recs = FT.collect("none")
        self.id2label = {a["id"]: a["label"] for a in FT._ROSTER["arms"]}
        arms3 = [aid for _, aid in FT.OWN_BASE + FT.EXT_TEG + FT.own_arms()]
        arms2 = [aid for _, aid in FT.EXT + FT.own_arms()[1:]]
        self.shared3 = self._shared(arms3, "two")
        self.shared2 = self._shared(arms2, "one")
        self.records = []  # (table, row, col, prompt, mean, n) for LEVELS.md

    def _per(self, label, sided):
        return {(r["cell"], r["endpoint"], r["reference"], k[2]): r
                for k, r in self.recs.items()
                if r["label"] == label and r["tier"] == "zero_shot" and r["sided"] == sided}

    def _shared(self, arm_ids, sided):
        sh = None
        for aid in arm_ids:
            p = self._per(self.id2label[aid], sided)
            if not p:
                continue
            ks = set(p)
            sh = ks if sh is None else (sh & ks)
        return sh or set()

    # ---- the two row selectors
    def rows_shared(self, arm_id, sided):
        shared = self.shared2 if sided == "one" else self.shared3
        p = self._per(self.id2label[arm_id], sided)
        return [p[k] for k in shared if k in p]

    def rows_tier(self, arm_id, tier):
        label = self.id2label[arm_id]
        return [r for r in self.recs.values() if r["label"] == label and r["tier"] == tier]

    @staticmethod
    def means(rows):
        out = {}
        for m in DEC:
            vals = [r.get(m) for r in rows if r.get(m) is not None]
            out[m] = (float(np.mean(vals)), len(vals)) if vals else (None, 0)
        return out


# ----------------------------------------------------------------------------- cell formatting
def _fmt(mean, metric, best):
    if mean is None:
        return r"\ph{--}"
    s = f"{mean * SCALE.get(metric, 1.0):.{DEC[metric]}f}"
    if best is not None and abs(mean - best) <= 1e-9:
        s = r"\textbf{" + s + "}"
    return s


def _delta_val(nmean, emean, metric):
    """Delta = the difference of the PRINTED values (owner's precedent in the tab_main comments), so every row is
    internally consistent (E - N as the reader sees them); the unrounded levels are in LEVELS.md."""
    if nmean is None or emean is None:
        return None
    k = SCALE.get(metric, 1.0)
    return round(round(emean * k, DEC[metric]) - round(nmean * k, DEC[metric]), DEC[metric])


def _delta(nmean, emean, metric, best_abs=None):
    d = _delta_val(nmean, emean, metric)
    if d is None:
        return r"\ph{--}"
    body = f"{abs(d):.{DEC[metric]}f}"                        # owner 2026-09-23 23:05: Delta printed as |E - N| (no sign)
    if best_abs is not None and abs(abs(d) - best_abs) <= 1e-9:   # owner 2026-09-23: the smallest gap is the good one
        body = r"\textbf{" + body + "}"
    return body


def _best(mean_lists, metric):
    vals = [m for m in mean_lists if m is not None]
    if not vals:
        return None
    return max(vals) if DIRN[metric] == "max" else min(vals)


def _best6(data):
    """Per-column bests of an (N, E, Delta) block: (metric, "N") / (metric, "E") = best measured value of that column
    (max, or min for lower-better metrics); (metric, "D") = the smallest |Delta| over the rows of the block
    (owner 2026-09-23: "bold all of the good scores for each column, minimum amount of delta is good")."""
    best = {}
    for m in COLS6:
        if m is None:
            continue
        best[(m, "N")] = _best([nm[m][0] for nm, _, _, _ in data.values() if nm], m)
        best[(m, "E")] = _best([em[m][0] for _, em, _, _ in data.values() if em], m)
        ds = [abs(d) for d in (_delta_val(nm[m][0] if nm else None, em[m][0] if em else None, m)
                               for nm, em, _, _ in data.values()) if d is not None]
        best[(m, "D")] = min(ds) if ds else None
    return best


def _cells6(nm, em, best):
    """15 value cells for a Transport|Motion|Action KL|Ref sim.|Seam-free (N,E,Delta) row (Flow MSE dropped by the owner 2026-09-23 23:30; it stays in eval 049 and the metrics_v2 previews)."""
    cells = []
    for m in COLS6:
        if m is None:
            cells += [r"\ph{--}", r"\ph{--}", r"\ph{--}"]
            continue
        n = nm[m][0] if nm else None
        e = em[m][0] if em else None
        cells += [_fmt(n, m, best.get((m, "N"))), _fmt(e, m, best.get((m, "E"))), _delta(n, e, m, best.get((m, "D")))]
    return cells


def _cells7(rm, best, cols=None):
    """Single-value row cells for the given columns (default COLS7; tab_weight uses COLS6 since 2026-09-26)."""
    cells = []
    for m in (cols or COLS7):
        if m is None:
            cells.append(r"\ph{--}")
            continue
        cells.append(_fmt(rm[m][0] if rm else None, m, best.get(m)))
    return cells


# ----------------------------------------------------------------------------- row substitution
_TERM = re.compile(r"(\s*\\\\(?:\[[^\]]*\])?)\s*$")


def _split_row(line):
    """(label_part, [cells], terminator) for a `... & ... \\` data line, else None."""
    m = _TERM.search(line)
    if not m:
        return None
    head = line[:m.start()]
    parts = head.split(" & ")
    if len(parts) < 2:
        return None
    return parts[0], parts[1:], m.group(1)


def _rebuild(label_part, cells, terminator):
    return label_part + " & " + " & ".join(cells) + terminator


def _fill6(line, nm, em, best):
    """Rewrite a Transport|Motion|Flow MSE|Action KL|Ref sim.|Seam-free (N,E,Delta) data row."""
    label_part, cells_in, term = _split_row(line)
    cells = _cells6(nm, em, best)
    assert len(cells) == len(cells_in), (label_part, len(cells), len(cells_in))
    return _rebuild(label_part, cells, term)


def _fill7(line, keep_cols, rm, best, cols=None):
    """Rewrite a single-value row, keeping the first `keep_cols` label columns and replacing the value cells (COLS7, or `cols`)."""
    label_part, cells_in, term = _split_row(line)
    parts = [label_part] + cells_in
    newcells = parts[1:keep_cols] + _cells7(rm, best, cols)
    assert len(newcells) == len(cells_in), (label_part, len(newcells), len(cells_in))
    return _rebuild(label_part, newcells, term)


def _fill_file(path, comment_lines, handler):
    """Keep the leading % block verbatim (dropping any prior appended block), append `comment_lines`,
    then pass every body line through handler(line, ctx) -> line (context tracking + row fills)."""
    lines = path.read_text().split("\n")
    body_start = next(i for i, ln in enumerate(lines) if not ln.startswith("%"))
    lead = lines[:body_start]
    body = lines[body_start:]
    if MARKER in lead:                       # drop a previously appended block so re-runs never duplicate the marker
        lead = lead[:lead.index(MARKER)]
    ctx = {}
    out_body = [handler(ln, ctx) for ln in body]
    path.write_text("\n".join(lead + comment_lines + out_body))


# ----------------------------------------------------------------------------- comment blocks
_COMMON = [
    MARKER,
    r"%   Sources: Transport = eval 047 (bl_333, equal-thirds blend); Motion fid. = eval 040 (det_motion_fidelity);",
    r"%   Ref sim. = eval 040 (videoprism_sim_ref); Seam-free = eval 048 (physical windows); Text cons. = eval 046 (text_own).",
    r"%   N (neutral prompt) and E (effect / author-native prompt) are measured on the SAME rows with the current metrics:",
    r"%   the neutral twins of the prior works (gens 2026-09-21/22) and the SEGUE effect-prompt runs on grid v3 (own_effect roster arms).",
    r"%   Delta = |E - N| of the printed (rounded) values (owner 2026-09-23: absolute gap, smallest bold); unrounded levels in",
    r"%   papers_drafts/_preview/paper_tables_v5/LEVELS.md. The blue \pnum provisional",
    r"%   Deltas (former Table 5, deployed metric / older grid) are replaced. Action KL (Video Swin-B Kinetics-400 class",
    r"%   probabilities over 32 uniformly sampled frames of the whole video, KL(reference || generated), lower is better) is",
    r"%   FILLED from eval 049. Flow MSE was REMOVED from the paper tables by the owner (2026-09-23 23:30); its values stay in",
    r"%   eval 049 (flow_mse) and in the metrics_v2 previews. The SEGUE w/ conditioning-1 row stays a placeholder.",
    r"%   Motion fid., Ref sim. and Text cons. are printed x100 with one decimal, '(x100)' in the header (Ozgur 2026-09-23 23:36);",
    r"%   the unscaled levels are in LEVELS.md. Bold (owner 2026-09-23): the best value of each column in the block --",
    r"%   N and E separately -- and, in the Delta columns, the smallest |Delta| (the least text dependence); ties all bold.",
]
_SPECIFIC = {
    "tab_main.tex": [
        r"%   Rows: TEG block on the shared two-sided zero-shot set (n=76); VET block on the shared one-sided zero-shot set (n=366).",
        r"%   CORRECTION: the VET seam-free was the 9-frame planned prefix (old draft VAP 76.0 / VFXMaster 85.5 / refVFX 93.2),",
        r"%   now the physical given window (eval 048). The refVFX-TEG CHECK is resolved: its prompt is the base_cond effect grid",
        r"%   prompt (prompts/011 effect), so it fills the E column of the refVFX TEG row.",
    ],
    "tab_ablation.tex": [
        r"%   N = Table-1 zero-shot pooled (HF+ED, one- and two-sided together, n=442 per arm); E = the effect-prompt own twin on",
        r"%   the same rows; Delta = |E - N|. (Supersedes the earlier ``E: placeholders'' note above -- the effect twins are now scored.)",
    ],
    "tab_ablationtiers.tex": [
        r"%   N = Table-1 seen (n=52) / unseen (n=274) pooled; E = the effect-prompt own twin on the same tier rows; Delta = |E - N|.",
        r"%   (Supersedes the earlier ``E and Delta: placeholders'' note above; the effect twins cover every tier on grid v3.)",
    ],
    "tab_weight.tex": [
        r"%   Table-1 logic per tier (n = 52 seen / 274 unseen / 442 zero-shot). CORRECTION: Transport is now the .33/.33/.33 blend,",
        r"%   and the w=1.5 / w=3 Motion fid. and Ref sim. now come from the finished lens pass (eval 040): the 2026-09-20 16:14 fill",
        r"%   for those two arms preceded 040 (17:18). The w=1 / w=6 rows now equal the SEGUE w/o NRG / SEGUE rows of tab_ablation and",
        r"%   tab_ablationtiers (same pipeline) -- the KNOWN disagreement noted in those tables is gone.",
    ],
    "tab_isolation.tex": [
        r"%   One metric and one n for every row: the shared one-sided zero-shot set, start endpoint given (n=366), the same rows as",
        r"%   the VET block of tab_main. The old-grid transport-only numbers (deployed metric, n=48/768) are replaced.",
    ],
}


def _comment(name):
    return _COMMON + _SPECIFIC[name]


# ----------------------------------------------------------------------------- the five tables
def render(lv: Levels):
    PREVIEW.mkdir(parents=True, exist_ok=True)

    def record(table, row, prompt, means):
        for m in DEC:
            mean, n = means[m]
            lv.records.append((table, row, m, prompt, mean, n))

    # ---- tab_main -----------------------------------------------------------
    teg = [(r"Base \ltx{}", "base_cond_neutral", "base_cond_effect"),
           ("VACE", "wan_vace_neutral", "wan_vace"),
           (r"\refvfx{}", "refvfx_teg_neutral", "refvfx_teg"),
           ("Wan2.1 FLF2V", "wan_flf2v_neutral", "wan_flf2v"),
           (r"\segue{}", "dualforce_dcg_w6", "dualforce_dcg_w6_effect")]
    vet = [(r"\vap{}", "vap_neutral", "vap_author_native"),
           (r"\vfxmaster{}", "vfxmaster_neutral", "vfxmaster_author_native"),
           (r"\refvfx{}", "refvfx_neutral", "refvfx_author_native"),
           (r"\segue{}", "dualforce_dcg_w6", "dualforce_dcg_w6_effect")]

    def block6(rows, sided):
        data = {}
        for label, nid, eid in rows:
            nm = lv.means(lv.rows_shared(nid, sided))
            em = lv.means(lv.rows_shared(eid, sided))
            data[label] = (nm, em, nid, eid)
        return data, _best6(data)

    teg_data, teg_best = block6(teg, "two")
    vet_data, vet_best = block6(vet, "one")
    for label, (nm, em, nid, eid) in teg_data.items():
        record("tab_main/TEG/" + label, nid, "N", nm); record("tab_main/TEG/" + label, eid, "E", em)
    for label, (nm, em, nid, eid) in vet_data.items():
        record("tab_main/VET/" + label, nid, "N", nm); record("tab_main/VET/" + label, eid, "E", em)

    def main_filler(line, ctx):
        if r"\tabsec" in line:
            if "both endpoints given" in line:
                ctx["block"] = "TEG"
            elif "start endpoint given" in line:
                ctx["block"] = "VET"
            return line
        if _split_row(line) is None:
            return line
        key = line.split(" & ")[0].strip()
        data, best = ((teg_data, teg_best) if ctx.get("block") == "TEG"
                      else (vet_data, vet_best) if ctx.get("block") == "VET" else (None, None))
        if data is not None and key in data:
            nm, em, _, _ = data[key]
            return _fill6(line, nm, em, best)
        return line

    _render_to(TABLES / "tab_main.tex", PREVIEW / "tab_main.tex", _comment("tab_main.tex"), main_filler)

    # ---- tab_ablation (zero-shot) ------------------------------------------
    abl = [("Plain LoRA", "ic_gen", "ic_gen_effect"),
           (r"\segue{} w/o NRG", "dualforce_control", "dualforce_control_effect"),
           (r"\segue{}", "dualforce_dcg_w6", "dualforce_dcg_w6_effect")]

    def tierblock6(rows, tier):
        data = {}
        for label, nid, eid in rows:
            nm = lv.means(lv.rows_tier(nid, tier))
            em = lv.means(lv.rows_tier(eid, tier))
            data[label] = (nm, em, nid, eid)
        return data, _best6(data)

    abl_data, abl_best = tierblock6(abl, "zero_shot")
    for label, (nm, em, nid, eid) in abl_data.items():
        record("tab_ablation/" + label, nid, "N", nm); record("tab_ablation/" + label, eid, "E", em)

    def abl_filler(line, ctx):
        if _split_row(line) is None:
            return line
        key = line.split(" & ")[0].strip()
        if key in abl_data:                      # the conditioning-1 row and everything else stay verbatim
            nm, em, _, _ = abl_data[key]
            return _fill6(line, nm, em, abl_best)
        return line

    _render_to(TABLES / "tab_ablation.tex", PREVIEW / "tab_ablation.tex", _comment("tab_ablation.tex"), abl_filler)

    # ---- tab_ablationtiers (seen / unseen) ---------------------------------
    seen_data, seen_best = tierblock6(abl, "seen")
    unseen_data, unseen_best = tierblock6(abl, "unseen")
    for label, (nm, em, nid, eid) in seen_data.items():
        record("tab_ablationtiers/Seen/" + label, nid, "N", nm); record("tab_ablationtiers/Seen/" + label, eid, "E", em)
    for label, (nm, em, nid, eid) in unseen_data.items():
        record("tab_ablationtiers/Unseen/" + label, nid, "N", nm); record("tab_ablationtiers/Unseen/" + label, eid, "E", em)

    def tiers_filler(line, ctx):
        if r"\tabsec" in line:
            if "{Seen}" in line:
                ctx["tier"] = "seen"
            elif "{Unseen}" in line:
                ctx["tier"] = "unseen"
            return line
        if _split_row(line) is None:
            return line
        key = line.split(" & ")[0].strip()
        data, best = ((seen_data, seen_best) if ctx.get("tier") == "seen"
                      else (unseen_data, unseen_best) if ctx.get("tier") == "unseen" else (None, None))
        if data is not None and key in data:
            nm, em, _, _ = data[key]
            return _fill6(line, nm, em, best)
        return line

    _render_to(TABLES / "tab_ablationtiers.tex", PREVIEW / "tab_ablationtiers.tex", _comment("tab_ablationtiers.tex"), tiers_filler)

    # ---- tab_weight (per tier, per w) --------------------------------------
    WARM = {"$w = 1$": "dualforce_control", "$w = 1.5$": "dualforce_dcg_w1p5",
            "$w = 3$": "dualforce_dcg_w3", "$w = 6$": "dualforce_dcg_w6"}
    TIERK = {"Seen": "seen", "Unseen": "unseen", "Zero-shot": "zero_shot",
             "Held-out instance": "unseen", "Held-out class": "zero_shot"}   # paper tier names since 2026-09-25
    weight_means = {}   # (tier_label, w_label) -> means
    weight_best = {}    # tier_label -> {metric: best}
    for tlabel, tk in TIERK.items():
        rows_by_w = {}
        for w, arm in WARM.items():
            rows_by_w[w] = lv.means(lv.rows_tier(arm, tk))
            weight_means[(tlabel, w)] = rows_by_w[w]
            record(f"tab_weight/{tlabel}/{w}", arm, "-", rows_by_w[w])
        best = {}
        for m in DEC:
            best[m] = _best([rows_by_w[w][m][0] for w in WARM], m)
        weight_best[tlabel] = best

    def weight_filler(line, ctx):
        parsed = _split_row(line)
        if parsed is None:
            return line
        label_part, cells_in, _ = parsed
        parts = [label_part] + cells_in
        if len(parts) < 3:
            return line
        tier_tok = parts[0].strip()
        w_tok = parts[1].strip()
        if tier_tok in TIERK:
            ctx["tier"] = tier_tok
        tier = ctx.get("tier")
        if w_tok in WARM and tier is not None:            # keep Tier + w label columns (2), replace the 7 value cells
            return _fill7(line, 2, weight_means[(tier, w_tok)], weight_best[tier], COLS6)   # Text cons. dropped 2026-09-26
        return line

    _render_to(TABLES / "tab_weight.tex", PREVIEW / "tab_weight.tex", _comment("tab_weight.tex"), weight_filler)

    # ---- tab_isolation (neutral / effect, shared one-sided set) ------------
    iso = [(r"\vap{}", "vap_neutral", "vap_author_native"),
           (r"\vfxmaster{}", "vfxmaster_neutral", "vfxmaster_author_native"),
           (r"\refvfx{}", "refvfx_neutral", "refvfx_author_native"),
           ("Plain LoRA", "ic_gen", "ic_gen_effect"),
           (r"\segue{} w/o NRG", "dualforce_control", "dualforce_control_effect"),
           (r"\segue{}", "dualforce_dcg_w6", "dualforce_dcg_w6_effect")]
    iso_means = {}   # (system, prompt) -> means
    for label, nid, eid in iso:
        iso_means[(label, "neutral")] = lv.means(lv.rows_shared(nid, "one"))
        iso_means[(label, "effect")] = lv.means(lv.rows_shared(eid, "one"))
        record("tab_isolation/" + label, nid, "neutral", iso_means[(label, "neutral")])
        record("tab_isolation/" + label, eid, "effect", iso_means[(label, "effect")])
    iso_best = {}
    for prompt in ("neutral", "effect"):
        b = {}
        for m in DEC:
            b[m] = _best([iso_means[(lab, prompt)][m][0] for lab, _, _ in iso], m)
        iso_best[prompt] = b

    def iso_filler(line, ctx):
        parsed = _split_row(line)
        if parsed is None:
            return line
        label_part, cells_in, _ = parsed
        parts = [label_part] + cells_in
        if len(parts) < 3:
            return line
        sys_tok = parts[0].strip()
        prompt_tok = {"N": "neutral", "w/o": "neutral", "E": "effect", "w/": "effect"}.get(parts[1].strip(), parts[1].strip())   # N / E since 2026-09-25 (owner)
        if sys_tok:
            ctx["sys"] = sys_tok
        sysname = ctx.get("sys")
        if prompt_tok in ("neutral", "effect") and (sysname, prompt_tok) in iso_means:  # keep system + prompt columns (2)
            return _fill7(line, 2, iso_means[(sysname, prompt_tok)], iso_best[prompt_tok])
        return line

    _render_to(TABLES / "tab_isolation.tex", PREVIEW / "tab_isolation.tex", _comment("tab_isolation.tex"), iso_filler)

    _write_levels(lv)


def _render_to(src, dst, comment_lines, row_filler):
    """Fill the current on-disk table `src` and write the result to the preview copy `dst`."""
    dst.write_text(src.read_text())           # start from the current on-disk table
    _fill_file(dst, comment_lines, row_filler)


def _write_levels(lv: Levels):
    lines = ["# paper_tables_v5 levels (metrics v5, scripts/metrics_v5/paper_tables.py)",
             f"# generated {TODAY}. Every level = the mean over its rows, 4 decimals, with n.",
             "# Delta printed in a table = |round(E, dec) - round(N, dec)| at the metric's decimals (absolute difference of the printed values).",
             "# columns: table | row (roster id) | metric | prompt | mean(4dp) | n", ""]
    for table, row, metric, prompt, mean, n in lv.records:
        mv = f"{mean:.4f}" if mean is not None else "--"
        lines.append(f"{table} | {row} | {metric} | {prompt} | {mv} | {n}")
    (PREVIEW / "LEVELS.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- R6 pixel-endpoint table
def _pmean(rows, key):
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float)) and r.get(key) is not None]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def _per_k(recs, label, sided):
    """(cell,endpoint,reference,seed) -> (k, rec) over the zero-shot rows of one arm/sidedness (k = the recs key)."""
    return {(r["cell"], r["endpoint"], r["reference"], k[2]): (k, r)
            for k, r in recs.items()
            if r["label"] == label and r["tier"] == "zero_shot" and r["sided"] == sided}


def _pfmt(mean, metric, best):
    if mean is None:
        return "--"
    s = f"{mean * PIXEL_SCALE[metric]:.{PIXEL_DEC[metric]}f}"
    if best is not None and abs(mean - best) <= 1e-9:
        s = r"\textbf{" + s + "}"
    return s


def render_pixel_endpoint_table(lv: "Levels"):
    """tab_endpoint_pixel.tex (mirrors tables/tab_endpoint.tex) + LEVELS_pixel.md; NOT applied to the paper."""
    PREVIEW.mkdir(parents=True, exist_ok=True)
    e050 = FT._rows_by_arm(FT._find_eval("050_endpoint_pixel_gridv3"))    # (arm,item_id,seed) -> full 050 row (window means)
    level_lines = ["# tab_endpoint_pixel levels (metrics v5 R6, scripts/metrics_v5/paper_tables.py --pixel-endpoint-table)",
                   f"# generated {TODAY}. Frame-level means over the family_tables shared zero-shot sets (TEG n={len(lv.shared3)}, VET n={len(lv.shared2)}),",
                   "# 4 decimals unscaled (the table prints PSNR as-is, SSIM/LPIPS x100 at 1 decimal). Window means (clip-conditioned) below.",
                   "# columns: block | row | arm | metric | mean(4dp) | n", ""]

    def block(rows_map, sided):
        shared = lv.shared3 if sided == "two" else lv.shared2
        data = {}
        for label, arm in rows_map:
            pk = _per_k(lv.recs, lv.id2label[arm], sided)
            rows = [pk[key][1] for key in shared if key in pk]           # rec dicts (frame-level ep_* keys)
            data[label] = (arm, {c: _pmean(rows, c) for c in PIXEL_COLS})
        best = {}
        for c in PIXEL_COLS:
            vals = [d[1][c][0] for d in data.values() if d[1][c][0] is not None]
            best[c] = (None if not vals else (max(vals) if PIXEL_DIR[c] == "max" else min(vals)))
        return data, best

    def emit(rows_map, data, best, sec_tag, sided):
        lines = [r"        \tabsec{7}{" + sec_tag + r"} \\"]
        for label, arm in rows_map:
            if arm in PIXEL_ADDLINE:
                lines.append(r"        \addlinespace[2pt]")
            cells = [_pfmt(data[label][1][c][0], c, best[c]) for c in PIXEL_COLS]
            lines.append("        " + label + " & " + " & ".join(cells) + r" \\")
            for c in PIXEL_COLS:
                mean, n = data[label][1][c]
                level_lines.append(f"{sec_tag[:3]} | {label} | {arm} | {c} | {mean:.4f} | {n}" if mean is not None
                                   else f"{sec_tag[:3]} | {label} | {arm} | {c} | -- | {n}")
        return lines

    teg_data, teg_best = block(PIXEL_TEG, "two")
    vet_data, vet_best = block(PIXEL_VET, "one")
    teg_lines = emit(PIXEL_TEG, teg_data, teg_best, "Transition effect generation: both endpoints given", "two")
    vet_lines = emit(PIXEL_VET, vet_data, vet_best, "Visual effect transfer: start endpoint given", "one")

    caption = (r"\textbf{Pixel fidelity of the given endpoint frames} of the systems in Table~\ref{tab:main}, "
               r"zero-shot tier: the output's first frame against the given start frame and, with both endpoints given, "
               r"its last frame against the given end frame. --: not applicable.")
    lead = [
        r"% !TEX root = ../main.tex",
        r"% GENERATED (PREVIEW ONLY, NOT applied to the paper) by scripts/metrics_v5/paper_tables.py --pixel-endpoint-table.",
        r"% Protocol (brief R6): frame-level = out frame 0 vs the given start9[0] frame and, on two-sided rows, out frame T-1",
        r"% vs the given end9[8] frame; start9/end9 mp4 for EVERY grid type (the cross-system comparison). PSNR/SSIM/LPIPS on",
        r"% uint8 [0,255], full native resolution (PyAV rgb24, no resize). SSIM Wang 2004 (11x11 Gaussian sigma1.5); LPIPS alex.",
        r"% Rows/sources/n as tab_endpoint.tex: prior works at their effect/author-native arm, SEGUE = dualforce_dcg_w6 neutral;",
        r"% shared two-sided zero-shot set (n=" + str(len(lv.shared3)) + r") for TEG, shared one-sided zero-shot set (n=" + str(len(lv.shared2)) + r") for VET.",
        r"% CAVEAT: PSNR is capped by the H.264 encoding of BOTH the output and the given clips (~40-45 dB ceiling; a perfect copy",
        r"% through a VAE lands lower -- see the R6 RECORD re-encoding reference). The window means for the clip-conditioned systems",
        r"% (own HF/ED, VACE, from the lossless PNGs) are in eval 050 and papers_drafts/_preview/paper_tables_v5/LEVELS_pixel.md.",
        r"% SSIM and LPIPS printed x100; PSNR in dB. Bold: best per column and block (PSNR/SSIM max, LPIPS min).",
    ]
    tex = "\n".join(lead + [
        r"\begin{table}[H]", r"    \centering",
        r"    \caption{" + caption + "}", r"    \label{tab:endpoint_pixel}",
        r"    \footnotesize", r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{tabular}{lcccccc}", r"        \toprule",
        r"        & \multicolumn{3}{c}{Start frame} & \multicolumn{3}{c}{End frame} \\",
        r"        \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"        & \shortstack{PSNR\\$\uparrow$} & \shortstack{SSIM\,$\uparrow$\\($\times$100)} & \shortstack{LPIPS\,$\downarrow$\\($\times$100)}"
        r" & \shortstack{PSNR\\$\uparrow$} & \shortstack{SSIM\,$\uparrow$\\($\times$100)} & \shortstack{LPIPS\,$\downarrow$\\($\times$100)} \\",
        r"        \midrule",
    ] + teg_lines + [r"        \midrule"] + vet_lines + [
        r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""])
    (PREVIEW / "tab_endpoint_pixel.tex").write_text(tex)

    # ---- window-mean levels for the clip-conditioned arms (own HF+ED, VACE) over the same shared sets
    level_lines += ["", "# window means (SECONDARY): mean over the whole given window (HF 9/8, VACE16 6/4 from lossless PNGs, ED 1/1)",
                    "# frame-level (A/B) vs window (Aw/Bw) on the shared zero-shot set; window fields from eval 050 rows.",
                    "# columns: arm-desc | field | mean(4dp) | n"]
    for desc, arm, gt in PIXEL_WINDOW_ARMS:
        sided = "two" if gt == "VACE16" else None
        for sd in (["two", "one"] if sided is None else ["two"]):
            pk = _per_k(lv.recs, lv.id2label[arm], sd)
            shared = lv.shared3 if sd == "two" else lv.shared2
            ks = [pk[key][0] for key in shared if key in pk]     # (harness_arm,item_id,seed) recs keys
            wr = [e050[k] for k in ks if k in e050]
            # keep only rows of the requested grid type (HF vs ED for the SEGUE arm)
            wr = [r for r in wr if r.get("gtype") == gt]
            if not wr:
                continue
            for fld in ("psnr_A", "ssim_A", "lpips_A", "psnr_Aw", "ssim_Aw", "lpips_Aw",
                        "psnr_B", "ssim_B", "lpips_B", "psnr_Bw", "ssim_Bw", "lpips_Bw"):
                vals = [r.get(fld) for r in wr if isinstance(r.get(fld), (int, float)) and r.get(fld) is not None]
                if vals:
                    level_lines.append(f"{desc} [{sd}] | {fld} | {np.mean(vals):.4f} | {len(vals)}")
    (PREVIEW / "LEVELS_pixel.md").write_text("\n".join(level_lines) + "\n")
    print(f"[pixel-table] wrote {PREVIEW/'tab_endpoint_pixel.tex'} + LEVELS_pixel.md (TEG n={len(lv.shared3)}, VET n={len(lv.shared2)})")
    return teg_data, teg_best, vet_data, vet_best


def build_preview_pixel():
    preview = "\n".join([
        r"% GENERATED by scripts/metrics_v5/paper_tables.py --pixel-endpoint-table", r"\documentclass{article}",
        r"\usepackage{natbib}", r"\input{preamble}", r"\usepackage{geometry}",
        r"\geometry{landscape, margin=1.5cm}", r"\usepackage{booktabs}", r"\usepackage{graphicx}",
        r"\hypersetup{hidelinks}", r"\begin{document}", r"\begin{center}",
        r"{\Large\bfseries \segue{} -- pixel endpoint fidelity (grid v3 preview)}\\[3pt]",
        r"\small " + TODAY + r". Zero-shot tier; levels, two seeds per row, no confidence intervals. PREVIEW ONLY (not in the paper).",
        r"\end{center}", r"\vspace{0.4em}", r"\input{tab_endpoint_pixel}", r"\end{document}", ""])
    (PREVIEW / "preview_pixel.tex").write_text(preview)
    env = dict(os.environ)
    env["PATH"] = TEXBIN + ":" + env.get("PATH", "")
    env["TEXINPUTS"] = f".:{PAPER}:{PAPER}//:"
    env["BIBINPUTS"] = env["TEXINPUTS"]; env["BSTINPUTS"] = env["TEXINPUTS"]
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error", "preview_pixel.tex"],
                       cwd=PREVIEW, env=env, capture_output=True, text=True, timeout=600)
    subprocess.run(["latexmk", "-c", "preview_pixel.tex"], cwd=PREVIEW, env=env, capture_output=True, text=True)
    print(f"[pixel-table] preview build {'OK' if r.returncode == 0 else 'FAILED'} -> {PREVIEW / 'preview_pixel.pdf'}")
    if r.returncode != 0:
        print("\n".join(r.stdout.splitlines()[-30:]))
    return r.returncode


# ----------------------------------------------------------------------------- preview pdf
def build_preview():
    preview = "\n".join([
        r"% GENERATED by scripts/metrics_v5/paper_tables.py", r"\documentclass{article}",
        r"\usepackage{natbib}", r"\input{preamble}", r"\usepackage{geometry}",
        r"\geometry{landscape, margin=1cm}", r"\usepackage{booktabs}", r"\usepackage{graphicx}",
        r"\hypersetup{hidelinks}", r"\begin{document}", r"\begin{center}",
        r"{\Large\bfseries \segue{} -- paper result tables (grid v3 preview)}\\[3pt]",
        r"\small " + TODAY + r". Two seeds per row; levels, no confidence intervals.",
        r"\end{center}", r"\vspace{0.4em}",
        r"\input{tab_main}", r"\vspace{0.5em}", r"\input{tab_ablation}", r"\vspace{0.5em}",
        r"\input{tab_ablationtiers}", r"\vspace{0.5em}", r"\input{tab_weight}", r"\vspace{0.5em}",
        r"\input{tab_isolation}", r"\vspace{0.5em}", r"\input{tables/tab_endpoint}", r"\end{document}", ""])
    (PREVIEW / "preview.tex").write_text(preview)
    env = dict(os.environ)
    env["PATH"] = TEXBIN + ":" + env.get("PATH", "")
    env["TEXINPUTS"] = f".:{PAPER}:{PAPER}//:"
    env["BIBINPUTS"] = env["TEXINPUTS"]
    env["BSTINPUTS"] = env["TEXINPUTS"]
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error", "preview.tex"],
                       cwd=PREVIEW, env=env, capture_output=True, text=True, timeout=600)
    subprocess.run(["latexmk", "-c", "preview.tex"], cwd=PREVIEW, env=env, capture_output=True, text=True)
    print(f"[paper-tables] preview build {'OK' if r.returncode == 0 else 'FAILED'} -> {PREVIEW / 'preview.pdf'}")
    if r.returncode != 0:
        print("\n".join(r.stdout.splitlines()[-30:]))
    return r.returncode


# ----------------------------------------------------------------------------- apply + paper build
def apply_and_build():
    BACKUP.mkdir(parents=True, exist_ok=True)
    names = ["tab_main.tex", "tab_ablation.tex", "tab_ablationtiers.tex", "tab_weight.tex", "tab_isolation.tex"]
    backed = 0
    for n in names:
        if not (BACKUP / n).exists():                 # back up ONCE: the pre-fill originals, never a later fill
            shutil.copy2(TABLES / n, BACKUP / n); backed += 1
        shutil.copy2(PREVIEW / n, TABLES / n)
    print(f"[paper-tables] backed up {backed} tables -> {BACKUP} (existing backups kept); copied rendered tables -> {TABLES}")
    env = dict(os.environ)
    env["PATH"] = TEXBIN + ":" + env.get("PATH", "")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error", "main.tex"],
                       cwd=PAPER, env=env, capture_output=True, text=True, timeout=900)
    log = (PAPER / "main.log").read_text() if (PAPER / "main.log").exists() else ""
    errs = len(re.findall(r"^! |LaTeX Error", log, flags=re.MULTILINE))
    print(f"[paper-tables] paper build rc={r.returncode}; '^! |LaTeX Error' count in main.log = {errs}")
    if r.returncode != 0:
        print("\n".join(r.stdout.splitlines()[-30:]))
    return r.returncode, errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="back up + copy the rendered tables into the paper and build main.tex")
    ap.add_argument("--no-preview", action="store_true", help="skip the preview.pdf build")
    ap.add_argument("--pixel-endpoint-table", action="store_true",
                    help="R6: render ONLY the pixel-endpoint preview table (tab_endpoint_pixel.tex + preview_pixel.pdf + "
                         "LEVELS_pixel.md); NOT applied to the paper, the five result tables untouched")
    args = ap.parse_args(argv)
    lv = Levels()
    print(f"[paper-tables] collect(): {len(lv.recs)} generations; |shared3|={len(lv.shared3)} |shared2|={len(lv.shared2)}")
    if args.pixel_endpoint_table:
        render_pixel_endpoint_table(lv)
        rc = 0 if args.no_preview else build_preview_pixel()
        print("[paper-tables] pixel-endpoint preview done (paper tables untouched)")
        return rc
    render(lv)
    print(f"[paper-tables] rendered 5 tables + LEVELS.md -> {PREVIEW}")
    if not args.no_preview:
        build_preview()
    if args.apply:
        rc, errs = apply_and_build()
        return 1 if (rc != 0 or errs) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
