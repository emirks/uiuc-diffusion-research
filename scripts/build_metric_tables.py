#!/usr/bin/env python
"""build_metric_tables.py -- Op-5 metric table builder for the CTT / SEGUE paper (grid v3).

Joins four per-generation feature sources on ``(item_id, seed)`` and emits three
LaTeX tables (plus a markdown twin and a standalone preview PDF) in the paper's
booktabs style, written ONLY under ``papers_drafts/_preview/metrics_gridv3/``.  It
never touches Ozgur's ``papers_drafts/ctt_iclr2027/tables/*.tex``.

Input row sets (any may be partially present; a missing column renders ``\\ph{--}``):
  1. v4 per-gen rows (Op-2):  store/evals/028*/<arm>/per_gen.jsonl
                              store/evals/030*/<arm>/per_gen.jsonl
     keys: item_id, seed, arm, variant_dir, gen_video, tier, sided, cell, content,
           pct_type, endpoint, reference, ref_class, n_frames, transport_pct,
           transport_raw, transport_ceiling, transport_capped, max_seam_z,
           prefix_seam_z, suffix_seam_z, prefix_dino, ...
           (its copy_max / near_copy are POOL-scored and NOT used -- see source 4).
  2. hand-off rows (Op-3):    store/evals/*_handoff_gridv3__*/<arm>/rows.jsonl
     keys: item_id, seed, arm, n_pre, n_suf, K, fps, identity_A, identity_B,
           motion_A, motion_B, seam_free, missing
  3. lens rows (later):       store/evals/*_lenses_gridv3__*/<arm>/rows.jsonl
     keys: item_id, seed, arm, motion_smoothness, videoprism_sim_ref,
           det_motion_fidelity, clip_sim_ref, dynamic_degree_mean_mag, aesthetic
  4. copy rows (Op-6):        store/evals/*_copy_gridv3__*/<arm>/rows.jsonl
     keys: item_id, seed, arm, copy_max, near_copy, copy_gen_frame, copy_ref_frame,
           ref_core_frac, n_mid, missing   -- M2a vs the gen's OWN reference; drives Copy rate.

A cell whose defining n is below MIN_N (=10) renders "n/a" (with its n) and never bolds.

Tables (fixed column definitions -- see the module-level TABLE_* specs):
  A  own arms x {seen, unseen, zero_shot}, neutral prompt (HF + ED pooled)
  B  prior works, zero-shot, the shared one-sided set (183 triples x 2 seeds = 366/arm)
  C  text dependency (neutral vs effect) on the same shared set
  D  ablation w-sweep -- NOT built tonight (arms not on grid v3): a one-line note.

Run (regenerate everything against the live store, once the inputs land):
  $LAB/envs-aarch64/ltx2/bin/python scripts/build_metric_tables.py --strict

Author: Op-5 (emirks).  CPU only.  Pure python + numpy over jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(REPO_ROOT, "papers_drafts", "ctt_iclr2027")
DEFAULT_OUT = os.path.join(REPO_ROOT, "papers_drafts", "_preview", "metrics_gridv3")
TEXLIVE_BIN = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/texlive/bin/aarch64-linux"

# A cell whose defining n is below this renders "n/a" (with its n) and is never
# bold (coordinator rule 2026-09-18): thin cells (e.g. seen-tier Identity/Motion B)
# are not reportable point estimates.
MIN_N = 10

# ---------------------------------------------------------------------------
# arm / variant identity
# ---------------------------------------------------------------------------
# base_arm -> LaTeX row label (macro) used in the paper.  \citep tacked on in
# Table B only, mirroring tab_main.tex.
ARM_LABEL = {
    "base_cond": r"\ltx{} (no reference)",
    "ic_gen": r"\ltx{} baseline LoRA",
    "dualforce_control": r"\segue{} w/o guidance",
    "dualforce_dcg_w6": r"\segue{}",
    "vap": r"\vap{}",
    "vfxmaster": r"\vfxmaster{}",
    "refvfx": r"\refvfx{}",
}
ARM_CITE = {
    "vap": r"~\citep{vap2025}",
    "vfxmaster": r"~\citep{vfxmaster2025}",
    "refvfx": r"~\citep{refvfx2026}",
}
# plain-text names for the markdown twin (the .tex uses the macros above)
ARM_MD = {
    "base_cond": "LTX-2 (no reference)",
    "ic_gen": "LTX-2 baseline LoRA",
    "dualforce_control": "SEGUE w/o guidance",
    "dualforce_dcg_w6": "SEGUE",
    "vap": "Video-As-Prompt",
    "vfxmaster": "VFXMaster",
    "refvfx": "refVFX",
}
OWN_ARMS = ["base_cond", "ic_gen", "dualforce_control", "dualforce_dcg_w6"]
EXTERNAL_ARMS = ["vap", "vfxmaster", "refvfx"]
# Table A own-arm order and Table B/C orders follow the paper's tab_main / tab_isolation.
TABLE_A_ARMS = ["base_cond", "ic_gen", "dualforce_control", "dualforce_dcg_w6"]
TABLE_B_ARMS = ["vap", "vfxmaster", "refvfx", "ic_gen", "dualforce_control", "dualforce_dcg_w6"]
TABLE_C_OWN = ["base_cond", "ic_gen", "dualforce_control", "dualforce_dcg_w6"]
TIERS = ["seen", "unseen", "zero_shot"]
TIER_LABEL = {"seen": "Seen", "unseen": "Unseen", "zero_shot": "Zero-shot"}

# known variant tails on the eval arm directories
_VARIANT_TAILS = [
    ("neutral_v3ed81", "neutral", "ed"),
    ("effect_v3ed81", "effect", "ed"),
    ("neutral_v3", "neutral", "hf"),
    ("effect_v3", "effect", "hf"),
    ("author_native", "effect", "external"),
]


def parse_arm(arm_dir: str) -> Optional[tuple[str, str, str]]:
    """arm-directory name -> (base_arm, variant_kind, family) or None if unknown.

    e.g. base_cond_neutral_v3ed81 -> (base_cond, neutral, ed)
         vap_author_native        -> (vap, effect, external)
    """
    for tail, kind, family in _VARIANT_TAILS:
        suff = "_" + tail
        if arm_dir.endswith(suff):
            base = arm_dir[: -len(suff)]
            if base in ARM_LABEL:
                return base, kind, family
    return None


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _key(item_id, seed) -> tuple[str, int]:
    try:
        s = int(seed)
    except (TypeError, ValueError):
        s = seed
    return (str(item_id), s)


def load_per_gen(roots: list[str]) -> tuple[list[dict], list[str]]:
    """Load Op-2 per_gen.jsonl under the given eval roots (globs allowed).

    Returns (records, arm_dirs_seen).  Each record is annotated with base_arm,
    variant_kind, family, arm_label.
    """
    records: list[dict] = []
    arm_dirs: list[str] = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*", "per_gen.jsonl"))):
            arm_dir = os.path.basename(os.path.dirname(path))
            parsed = parse_arm(arm_dir)
            if parsed is None:
                continue
            base, kind, family = parsed
            arm_dirs.append(arm_dir)
            for row in _read_jsonl(path):
                row = dict(row)
                row["base_arm"] = base
                row["variant_kind"] = kind
                row["family"] = family
                row["_arm_dir"] = arm_dir
                records.append(row)
    return records, arm_dirs


def load_side(globs: list[str], keys: list[str]) -> dict[tuple, dict]:
    """Load hand-off / lens rows keyed by (item_id, seed); keep only `keys`."""
    out: dict[tuple, dict] = {}
    for g in globs:
        for path in sorted(glob.glob(g)):
            for row in _read_jsonl(path):
                k = _key(row.get("item_id"), row.get("seed"))
                out[k] = {kk: row.get(kk) for kk in keys}
    return out


HANDOFF_KEYS = ["identity_A", "identity_B", "motion_A", "motion_B", "seam_free",
                "n_pre", "n_suf", "K", "fps", "missing"]
LENS_KEYS = ["motion_smoothness", "videoprism_sim_ref", "det_motion_fidelity",
             "clip_sim_ref", "dynamic_degree_mean_mag", "aesthetic"]
# M2a copy against the generation's OWN reference (Op-6).  We do NOT use per_gen's
# near_copy/copy_max: in evals/028/030 M2a was scored against POOL clips and is
# identical across arms on 100% of pool rows (not a generation property).
COPY_KEYS = ["copy_max", "near_copy", "copy_gen_frame", "copy_ref_frame",
             "ref_core_frac", "n_mid", "missing"]


def join_records(per_gen: list[dict], handoff: dict, lens: dict,
                 copy: Optional[dict] = None) -> list[dict]:
    """Attach hand-off, lens and copy fields to each per-gen record by (item_id, seed).

    The copy eval's near_copy / copy_max land under `copy_near_copy` / `copy_copy_max`
    so per_gen's (invalid, pool-scored) near_copy is never read by the tables.
    """
    copy = copy or {}
    for rec in per_gen:
        k = _key(rec.get("item_id"), rec.get("seed"))
        h = handoff.get(k)
        if h:
            for kk in HANDOFF_KEYS:
                if kk not in ("missing",):
                    rec[kk] = h.get(kk)
        l = lens.get(k)
        if l:
            for kk in LENS_KEYS:
                rec[kk] = l.get(kk)
        c = copy.get(k)
        if c:
            rec["copy_near_copy"] = c.get("near_copy")
            rec["copy_copy_max"] = c.get("copy_max")
    return per_gen


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _fin(vals) -> list[float]:
    out = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, bool):
            out.append(1.0 if v else 0.0)
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        out.append(f)
    return out


@dataclass
class Cell:
    """One aggregated table cell: a mean over `n` finite values, or a placeholder."""
    value: Optional[float] = None
    n: int = 0
    bold: bool = False

    @property
    def present(self) -> bool:
        return self.value is not None


def agg(records: list[dict], field_name: str, scale: float = 1.0) -> Cell:
    vals = _fin([r.get(field_name) for r in records])
    if not vals:
        return Cell(None, 0)
    return Cell(float(np.mean(vals)) * scale, len(vals))


# ---------------------------------------------------------------------------
# column specification
# ---------------------------------------------------------------------------
@dataclass
class Col:
    key: str                       # short id
    header: str                    # LaTeX header (\shortstack{...})
    fmt: str                       # 'pct1' | 'sim3' | 'dpct1' | 'dsim3'
    direction: str                 # 'max' | 'min' | 'absmin' | 'none'
    extract: Callable[[list[dict]], Cell]


def _fmt_num(value: float, fmt: str) -> str:
    if fmt == "pct1":
        return f"{value:.1f}"
    if fmt == "sim3":
        return f"{value:.3f}"
    if fmt == "dpct1":
        return f"{value:+.1f}"
    if fmt == "dsim3":
        return f"{value:+.3f}"
    return f"{value:.3f}"


# ---------------------------------------------------------------------------
# bolding
# ---------------------------------------------------------------------------
def bold_block(rows_cells: list[list[Cell]], cols: list[Col], eps: float = 1e-9) -> None:
    """Mark best cell(s) per column across the rows of one block, in place."""
    for ci, col in enumerate(cols):
        if col.direction == "none":
            continue
        # thin cells (n < MIN_N) render "n/a" and are excluded from the bold contest
        present = [(ri, rc[ci].value) for ri, rc in enumerate(rows_cells)
                   if rc[ci].present and rc[ci].n >= MIN_N]
        if not present:
            continue
        if col.direction == "max":
            best = max(v for _, v in present)
            for ri, v in present:
                if abs(v - best) <= eps:
                    rows_cells[ri][ci].bold = True
        elif col.direction == "min":
            best = min(v for _, v in present)
            for ri, v in present:
                if abs(v - best) <= eps:
                    rows_cells[ri][ci].bold = True
        elif col.direction == "absmin":
            best = min(abs(v) for _, v in present)
            for ri, v in present:
                if abs(abs(v) - best) <= eps:
                    rows_cells[ri][ci].bold = True


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
def cell_tex(cell: Cell, fmt: str, block_n: Optional[int] = None) -> str:
    if not cell.present:
        return r"\ph{--}"
    if cell.n < MIN_N:                       # too thin to report -> n/a (with its n)
        return r"n/a{\tiny\,($n{=}" + str(cell.n) + "$)}"
    s = _fmt_num(cell.value, fmt)
    if cell.bold:
        s = r"\textbf{" + s + "}"
    if block_n is not None and cell.n != block_n:
        s += r"{\tiny\,($n{=}" + str(cell.n) + "$)}"
    return s


def cell_md(cell: Cell, fmt: str, block_n: Optional[int] = None) -> str:
    if not cell.present:
        return "--"
    if cell.n < MIN_N:
        return f"n/a (n={cell.n})"
    s = _fmt_num(cell.value, fmt)
    if cell.bold:
        s = "**" + s + "**"
    if block_n is not None and cell.n != block_n:
        s += f" (n={cell.n})"
    return s


# ---------------------------------------------------------------------------
# Table A -- own arms x tiers, neutral prompt
# ---------------------------------------------------------------------------
TABLE_A_COLS = [
    Col("identity_a", r"\shortstack{Identity\\A}", "sim3", "max",
        lambda rs: agg(rs, "identity_A")),
    Col("identity_b", r"\shortstack{Identity\\B}", "sim3", "max",
        lambda rs: agg(rs, "identity_B")),
    Col("motion_a", r"\shortstack{Motion\\A}", "sim3", "max",
        lambda rs: agg(rs, "motion_A")),
    Col("motion_b", r"\shortstack{Motion\\B}", "sim3", "max",
        lambda rs: agg(rs, "motion_B")),
    Col("seam", r"\shortstack{Seam-\\free \%}", "pct1", "max",
        lambda rs: agg(rs, "seam_free", scale=100.0)),
    Col("smooth", r"\shortstack{Motion\\smooth.}", "sim3", "max",
        lambda rs: agg(rs, "motion_smoothness")),
    Col("copy", r"\shortstack{Copy\\rate \%}", "pct1", "min",
        lambda rs: agg(rs, "copy_near_copy", scale=100.0)),
    Col("transport", r"\shortstack{Transport\\ours}", "pct1", "max",
        lambda rs: agg(rs, "transport_pct")),
]


def scope_neutral(records, base_arm, tier):
    return [r for r in records
            if r.get("base_arm") == base_arm
            and r.get("variant_kind") == "neutral"
            and r.get("tier") == tier]


@dataclass
class TableA:
    # per tier: list of (base_arm, block_n, [Cell per col])
    blocks: dict = field(default_factory=dict)


def build_table_a(records) -> TableA:
    ta = TableA()
    for tier in TIERS:
        rows = []
        cells_only = []
        for base in TABLE_A_ARMS:
            scope = scope_neutral(records, base, tier)
            block_n = len(scope)
            cells = [c.extract(scope) for c in TABLE_A_COLS]
            rows.append((base, block_n, cells))
            cells_only.append(cells)
        bold_block(cells_only, TABLE_A_COLS)
        ta.blocks[tier] = rows
    return ta


# ---------------------------------------------------------------------------
# shared one-sided zero-shot set (Tables B & C)
# ---------------------------------------------------------------------------
def _match_key(r: dict) -> tuple:
    return (r.get("endpoint"), r.get("reference"), r.get("cell"), _key("", r.get("seed"))[1])


def variant_scope(records, base_arm, variant_kind):
    """One-sided, zero-shot records for a base arm + variant kind."""
    return [r for r in records
            if r.get("base_arm") == base_arm
            and r.get("variant_kind") == variant_kind
            and r.get("sided") == "one"
            and r.get("tier") == "zero_shot"]


def external_scope(records, base_arm):
    return [r for r in records
            if r.get("base_arm") == base_arm
            and r.get("family") == "external"
            and r.get("sided") == "one"
            and r.get("tier") == "zero_shot"]


def _arm_bench_scope(records, base_arm):
    """The scope used to define the shared set for a Table B arm."""
    if base_arm in EXTERNAL_ARMS:
        return external_scope(records, base_arm)
    return variant_scope(records, base_arm, "neutral")


def build_shared_set(records) -> tuple[set, dict]:
    """Intersection of (endpoint, reference, cell, seed) keys across the 6 Table-B arms.

    Returns (shared_keys, diag) where diag reports per-arm coverage and lost rows.
    """
    per_arm_keys: dict[str, set] = {}
    for base in TABLE_B_ARMS:
        scope = _arm_bench_scope(records, base)
        per_arm_keys[base] = {_match_key(r) for r in scope}
    nonempty = [ks for ks in per_arm_keys.values() if ks]
    if not nonempty:
        shared: set = set()
    else:
        shared = set.intersection(*nonempty)
    diag = {"per_arm_n": {b: len(k) for b, k in per_arm_keys.items()},
            "shared_n": len(shared),
            "lost": {b: len(k - shared) for b, k in per_arm_keys.items()}}
    return shared, diag


# ---------------------------------------------------------------------------
# Table B -- prior works, shared one-sided zero-shot set
# ---------------------------------------------------------------------------
TABLE_B_COLS = [
    Col("identity_a", r"\shortstack{Identity\\A}", "sim3", "max",
        lambda rs: agg(rs, "identity_A")),
    Col("seam", r"\shortstack{Seam-\\free \%}", "pct1", "max",
        lambda rs: agg(rs, "seam_free", scale=100.0)),
    Col("smooth", r"\shortstack{Motion\\smooth.}", "sim3", "max",
        lambda rs: agg(rs, "motion_smoothness")),
    Col("copy", r"\shortstack{Copy\\rate \%}", "pct1", "min",
        lambda rs: agg(rs, "copy_near_copy", scale=100.0)),
    Col("transport", r"\shortstack{Transport\\ours}", "pct1", "max",
        lambda rs: agg(rs, "transport_pct")),
    Col("vpref", r"\shortstack{Ref sim.\\(VideoPrism)}", "sim3", "max",
        lambda rs: agg(rs, "videoprism_sim_ref")),
    Col("motfid", r"\shortstack{Motion fid.\\(vs ref)}", "sim3", "max",
        lambda rs: agg(rs, "det_motion_fidelity")),
    Col("aes", r"\shortstack{Aesthetic}", "sim3", "max",
        lambda rs: agg(rs, "aesthetic")),
]


@dataclass
class TableB:
    rows: list = field(default_factory=list)   # (base_arm, n, [Cell])
    diag: dict = field(default_factory=dict)


def build_table_b(records) -> TableB:
    shared, diag = build_shared_set(records)
    tb = TableB(diag=diag)
    cells_only = []
    for base in TABLE_B_ARMS:
        scope = [r for r in _arm_bench_scope(records, base) if _match_key(r) in shared]
        n = len(scope)
        cells = [c.extract(scope) for c in TABLE_B_COLS]
        tb.rows.append((base, n, cells))
        cells_only.append(cells)
    bold_block(cells_only, TABLE_B_COLS)
    return tb


# ---------------------------------------------------------------------------
# Table C -- text dependency on the shared set
# ---------------------------------------------------------------------------
@dataclass
class TableC:
    rows: list = field(default_factory=list)   # (base_arm, n, cells-dict)
    shared_n: int = 0


def build_table_c(records) -> TableC:
    shared, _ = build_shared_set(records)
    tc = TableC(shared_n=len(shared))

    def in_shared(scope):
        return [r for r in scope if _match_key(r) in shared]

    rows = []
    cells_only = []  # for bolding on transport_effect, transport_delta, vp_effect, vp_delta
    for base in TABLE_C_OWN:
        neu = in_shared(variant_scope(records, base, "neutral"))
        eff = in_shared(variant_scope(records, base, "effect"))
        t_neu = agg(neu, "transport_pct")
        t_eff = agg(eff, "transport_pct")
        v_neu = agg(neu, "videoprism_sim_ref")
        v_eff = agg(eff, "videoprism_sim_ref")
        t_d = Cell(t_eff.value - t_neu.value, min(t_neu.n, t_eff.n)) if (t_neu.present and t_eff.present) else Cell()
        v_d = Cell(v_eff.value - v_neu.value, min(v_neu.n, v_eff.n)) if (v_neu.present and v_eff.present) else Cell()
        n = max(t_neu.n, t_eff.n, v_neu.n, v_eff.n)
        rows.append((base, n, dict(t_neu=t_neu, t_eff=t_eff, t_d=t_d, v_neu=v_neu, v_eff=v_eff, v_d=v_d)))
        cells_only.append([t_eff, t_d, v_eff, v_d])
    for base in EXTERNAL_ARMS:
        eff = in_shared(external_scope(records, base))
        t_eff = agg(eff, "transport_pct")
        v_eff = agg(eff, "videoprism_sim_ref")
        n = max(t_eff.n, v_eff.n)
        rows.append((base, n, dict(t_neu=Cell(), t_eff=t_eff, t_d=Cell(), v_neu=Cell(), v_eff=v_eff, v_d=Cell())))
        cells_only.append([t_eff, Cell(), v_eff, Cell()])
    # bold: transport effect (max), transport delta (absmin), vp effect (max), vp delta (absmin)
    bold_block(cells_only, [
        Col("", "", "pct1", "max", None),
        Col("", "", "dpct1", "absmin", None),
        Col("", "", "sim3", "max", None),
        Col("", "", "dsim3", "absmin", None),
    ])
    tc.rows = rows
    return tc


# ---------------------------------------------------------------------------
# strict n assertions (Tables B & C)
# ---------------------------------------------------------------------------
def check_strict(tb: TableB, tc: TableC) -> list[str]:
    """Return --strict violations: any present Table B/C column with mixed n across rows.

    (The shared-set intersection dropping own-arm ED / two-sided rows is BY DESIGN --
    the externals set the one-sided HF frontier -- so that is reported, not failed.)
    """
    problems = []
    # Table B: every present column must share one n across rows (all should equal shared_n).
    for ci, col in enumerate(TABLE_B_COLS):
        ns = {cells[ci].n for (_b, _n, cells) in tb.rows if cells[ci].present}
        if len(ns) > 1:
            problems.append(f"Table B column '{col.key}' has mixed n across rows: {sorted(ns)}")
    # Table C: each present transport/refsim column shares one n across rows.
    for keyname in ("t_neu", "t_eff", "v_neu", "v_eff"):
        ns = {cd[keyname].n for (_b, _n, cd) in tc.rows if cd[keyname].present}
        if len(ns) > 1:
            problems.append(f"Table C column '{keyname}' has mixed n across rows: {sorted(ns)}")
    return problems


def shared_set_warnings(tb: TableB) -> list[str]:
    """Non-fatal coverage notes about the shared one-sided set (always surfaced)."""
    notes = []
    shared_n = tb.diag.get("shared_n", 0)
    if shared_n != 366:
        notes.append(f"shared-set n = {shared_n}, not the expected 366 (183 triples x 2 seeds) "
                     f"-- coverage gap among the Table-B arms.")
    losers = {b: v for b, v in tb.diag.get("lost", {}).items() if v}
    if losers:
        notes.append(f"rows present in an arm's bench but dropped by the intersection "
                     f"(own arms lose ED / non-shared rows by design): {losers}")
    return notes


# ===========================================================================
# LaTeX rendering
# ===========================================================================
def _row_label(base_arm: str, n: Optional[int], cite: bool = False) -> str:
    lbl = ARM_LABEL[base_arm]
    if cite and base_arm in ARM_CITE:
        lbl = lbl + ARM_CITE[base_arm]
    if n is not None:
        lbl = lbl + r" {\tiny\color{gray}$n{=}" + str(n) + "$}"
    return lbl


def render_table_a(ta: TableA) -> str:
    L = []
    L.append(r"% GENERATED by scripts/build_metric_tables.py -- Table A (own arms x tiers, neutral prompt).")
    L.append(r"% Do not hand-edit; rerun the builder.  Style mirrors tables/tab_main.tex.")
    L.append(r"\begin{table}[H]")
    L.append(r"    \centering")
    L.append(r"    \caption{\textbf{Own arms across tiers (neutral prompt).} Grid v3, "
              r"HF-121f and ED-81f pooled per tier. Input fidelity is the clip hand-off "
              r"(identity and motion at the given/generated seam), plus seam-free rate and "
              r"motion smoothness; Copy rate is lower-is-better; Transport is our metric. "
              r"Best value per column and tier in bold; $n$ per row is the block size, with a "
              r"per-cell $n$ where a column is defined on fewer rows (Identity/Motion B: "
              r"two-sided only; Motion A: a clip was given).}")
    L.append(r"    \label{tab:gridv3_own}")
    L.append(r"    \footnotesize")
    L.append(r"    \setlength{\tabcolsep}{2.5pt}")
    L.append(r"    \begin{tabular}{l" + "c" * len(TABLE_A_COLS) + "}")
    L.append(r"        \toprule")
    L.append(r"        & \multicolumn{4}{c}{Input fidelity\,$\uparrow$} & "
              r"Seam-free\,$\uparrow$ & Smoothness\,$\uparrow$ & Copy\,$\downarrow$ & Fidelity\,$\uparrow$ \\")
    L.append(r"        \cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}\cmidrule(lr){8-8}\cmidrule(lr){9-9}")
    L.append("        & " + " & ".join(c.header for c in TABLE_A_COLS) + r" \\")
    L.append(r"        \midrule")
    for ti, tier in enumerate(TIERS):
        L.append(r"        \multicolumn{" + str(len(TABLE_A_COLS) + 1) +
                 r"}{l}{\emph{" + TIER_LABEL[tier] + r"}} \\")
        for base, block_n, cells in ta.blocks[tier]:
            bn = block_n if block_n else None
            label = _row_label(base, bn)
            tex_cells = [cell_tex(cells[i], TABLE_A_COLS[i].fmt, block_n=block_n)
                         for i in range(len(TABLE_A_COLS))]
            L.append("        " + label + " & " + " & ".join(tex_cells) + r" \\")
        if ti != len(TIERS) - 1:
            L.append(r"        \midrule")
    L.append(r"        \bottomrule")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def render_table_b(tb: TableB) -> str:
    L = []
    L.append(r"% GENERATED by scripts/build_metric_tables.py -- Table B (prior works, shared one-sided zero-shot set).")
    L.append(r"% Do not hand-edit; rerun the builder.  Style mirrors tables/tab_main.tex.")
    L.append(r"\begin{table}[H]")
    L.append(r"    \centering")
    L.append(r"    \caption{\textbf{Comparison with previous approaches (zero-shot, shared "
              r"one-sided set).} Every system scored on the same one-sided zero-shot "
              r"generations, matched across arms by (endpoint, reference, cell, seed). "
              r"Externals use their author-native prompt; our arms use the neutral prompt. "
              r"Best value per column in bold; Copy rate is lower-is-better; $n$ per row.}")
    L.append(r"    \label{tab:gridv3_prior}")
    L.append(r"    \footnotesize")
    L.append(r"    \setlength{\tabcolsep}{2.5pt}")
    L.append(r"    \begin{tabular}{l" + "c" * len(TABLE_B_COLS) + "}")
    L.append(r"        \toprule")
    L.append(r"        & Input fid.\,$\uparrow$ & Seam-free\,$\uparrow$ & Smoothness\,$\uparrow$ & "
              r"Copy\,$\downarrow$ & Fidelity\,$\uparrow$ & \multicolumn{3}{c}{Reference fidelity\,$\uparrow$} \\")
    L.append(r"        \cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}"
              r"\cmidrule(lr){6-6}\cmidrule(lr){7-9}")
    L.append("        & " + " & ".join(c.header for c in TABLE_B_COLS) + r" \\")
    L.append(r"        \midrule")
    for i, (base, n, cells) in enumerate(tb.rows):
        if base == "ic_gen":   # separate externals from our arms
            L.append(r"        \addlinespace[2pt]")
        label = _row_label(base, n if n else None, cite=True)
        tex_cells = [cell_tex(cells[j], TABLE_B_COLS[j].fmt) for j in range(len(TABLE_B_COLS))]
        L.append("        " + label + " & " + " & ".join(tex_cells) + r" \\")
    L.append(r"        \bottomrule")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def render_table_c(tc: TableC) -> str:
    L = []
    L.append(r"% GENERATED by scripts/build_metric_tables.py -- Table C (text dependency, shared one-sided zero-shot set).")
    L.append(r"% Do not hand-edit; rerun the builder.  Style mirrors tables/tab_isolation.tex.")
    L.append(r"\begin{table}[H]")
    L.append(r"    \centering")
    L.append(r"    \caption{\textbf{Text dependency (shared one-sided zero-shot set).} Each "
              r"system with the neutral and with an effect prompt; $\Delta =$ effect $-$ "
              r"neutral is how much it leans on the text. Externals have no neutral prompt "
              r"(author-native $=$ their effect row). Smallest $|\Delta|$ per metric in bold; "
              r"$n$ per row.}")
    L.append(r"    \label{tab:gridv3_text}")
    L.append(r"    \footnotesize")
    L.append(r"    \setlength{\tabcolsep}{3pt}")
    L.append(r"    \begin{tabular}{lccccccc}")
    L.append(r"        \toprule")
    L.append(r"        & \multicolumn{3}{c}{Transport\,$\uparrow$} & "
              r"\multicolumn{3}{c}{Ref sim. (VideoPrism)\,$\uparrow$} & \\")
    L.append(r"        \cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    L.append(r"        & neutral & effect & $\Delta$ & neutral & effect & $\Delta$ & $n$ \\")
    L.append(r"        \midrule")
    for i, (base, n, cd) in enumerate(tc.rows):
        if base == "vap":   # separate externals from our arms
            L.append(r"        \midrule")
        label = ARM_LABEL[base]
        cells = [
            cell_tex(cd["t_neu"], "pct1"),
            cell_tex(cd["t_eff"], "pct1"),
            cell_tex(cd["t_d"], "dpct1"),
            cell_tex(cd["v_neu"], "sim3"),
            cell_tex(cd["v_eff"], "sim3"),
            cell_tex(cd["v_d"], "dsim3"),
            (str(n) if n else r"\ph{--}"),
        ]
        L.append("        " + label + " & " + " & ".join(cells) + r" \\")
    L.append(r"        \bottomrule")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


# ===========================================================================
# markdown twin
# ===========================================================================
def _placeholder_report(ta: TableA, tb: TableB, tc: TableC) -> list[str]:
    """List every column that renders entirely as \\ph{--} and why."""
    lines = []
    # Table A
    for tier in TIERS:
        for base, block_n, cells in ta.blocks[tier]:
            missing = [TABLE_A_COLS[i].key for i in range(len(TABLE_A_COLS)) if not cells[i].present]
            if missing:
                lines.append(f"- A / {TIER_LABEL[tier]} / {base}: {', '.join(missing)}"
                             + ("" if block_n else "  (no rows in scope)"))
    # Table B
    for base, n, cells in tb.rows:
        missing = [TABLE_B_COLS[i].key for i in range(len(TABLE_B_COLS)) if not cells[i].present]
        if missing:
            lines.append(f"- B / {base}: {', '.join(missing)}" + ("" if n else "  (no rows in shared set)"))
    # Table C
    for base, n, cd in tc.rows:
        missing = [k for k in ("t_neu", "t_eff", "t_d", "v_neu", "v_eff", "v_d") if not cd[k].present]
        if missing:
            lines.append(f"- C / {base}: {', '.join(missing)}")
    return lines


def _md_table_a(ta: TableA) -> list[str]:
    hdr = ["arm", "Id A", "Id B", "Mot A", "Mot B", "Seam%", "Smooth", "Copy%", "Transport", "n"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for tier in TIERS:
        out.append(f"| **{TIER_LABEL[tier]}** | | | | | | | | | |")
        for base, block_n, cells in ta.blocks[tier]:
            cvals = [cell_md(cells[i], TABLE_A_COLS[i].fmt, block_n=block_n) for i in range(len(TABLE_A_COLS))]
            out.append("| " + ARM_MD[base] + " | " + " | ".join(cvals) +
                       f" | {block_n or '--'} |")
    return out


def _md_table_b(tb: TableB) -> list[str]:
    hdr = ["arm", "Id A", "Seam%", "Smooth", "Copy%", "Transport", "RefSim(VP)", "MotFid", "Aesth", "n"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for base, n, cells in tb.rows:
        cvals = [cell_md(cells[i], TABLE_B_COLS[i].fmt) for i in range(len(TABLE_B_COLS))]
        out.append("| " + ARM_MD[base] + " | " + " | ".join(cvals) + f" | {n or '--'} |")
    return out


def _md_table_c(tc: TableC) -> list[str]:
    hdr = ["arm", "T neutral", "T effect", "T Δ", "VP neutral", "VP effect", "VP Δ", "n"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for base, n, cd in tc.rows:
        cvals = [cell_md(cd["t_neu"], "pct1"), cell_md(cd["t_eff"], "pct1"), cell_md(cd["t_d"], "dpct1"),
                 cell_md(cd["v_neu"], "sim3"), cell_md(cd["v_eff"], "sim3"), cell_md(cd["v_d"], "dsim3")]
        out.append("| " + ARM_MD[base] + " | " + " | ".join(cvals) + f" | {n or '--'} |")
    return out


def render_markdown(ta, tb, tc, meta) -> str:
    L = []
    L.append("# Grid v3 metric tables (preview twin)")
    L.append("")
    L.append(f"_Generated {meta['generated']} by `scripts/build_metric_tables.py`"
             + (f" -- **{meta['banner']}**" if meta.get("banner") else "") + "._")
    L.append("")
    L.append("These are the same numbers as `tab_A.tex` / `tab_B.tex` / `tab_C.tex`, in markdown, "
             "for review. Bold marks the best value per column per block (smallest for Copy rate and for "
             "the text-dependency $\\Delta$). Numbers: transport and rates to 1 decimal, similarities to 3. "
             f"A cell whose defining $n < {MIN_N}$ renders `n/a` (with its $n$) and never bolds.")
    L.append("")
    L.append("Inputs joined on `(item_id, seed)`:")
    for k in ("per_gen", "handoff", "lens", "copy"):
        st = meta["inputs"][k]
        L.append(f"- **{k}**: {st['files']} file(s), {st['rows']} rows"
                 + (f" from `{st['example']}`" if st.get("example") else " -- **ABSENT** (columns render `--`)"))
    L.append("")
    L.append("## Table A -- own arms across tiers (neutral prompt)")
    L += _md_table_a(ta)
    L.append("")
    L.append("## Table B -- comparison with previous approaches (shared one-sided zero-shot set)")
    L.append(f"Shared set (intersection over the 6 Table-B arms): **n = {tb.diag.get('shared_n', 0)}** "
             f"(target 366 = 183 triples x 2 seeds).")
    L.append(f"Per-arm bench coverage before intersection: `{tb.diag.get('per_arm_n', {})}`; "
             f"rows lost in the match per arm: `{tb.diag.get('lost', {})}`.")
    L.append("")
    L.append("**Copy rate source & frame-count caveat.** Copy rate is `100·mean(near_copy)` from the "
             "M2a copy eval (`*_copy_gridv3*`), which scores each generation against its OWN reference "
             "-- NOT per_gen's `near_copy`, which was scored against pool clips and is identical across "
             "arms (not a generation property). M2a takes the max over the generation's mid frames, so a "
             "longer generation has more chances to match: our clips are 121 f vs the externals' 49 f "
             "(VAP/VFXMaster) / 33 f (refVFX), which under-estimates the externals' copy rate -- disclosed, "
             "not corrected.")
    L += _md_table_b(tb)
    L.append("")
    L.append("## Table C -- text dependency (same shared set)")
    L.append(f"Shared set: **n = {tc.shared_n}**.")
    L += _md_table_c(tc)
    L.append("")
    L.append("## Table D -- ablation (guidance w-sweep)")
    L.append("**Not built tonight.** The w-sweep arms (w = 1, 1.5, 3) were not generated on grid v3, "
             "so there are no grid-v3 rows to aggregate. Table D stays as the paper's `tab_ablation.tex` "
             "draft until the sweep is regenerated on grid v3.")
    L.append("")
    ph = _placeholder_report(ta, tb, tc)
    L.append("## Placeholder (`\\ph{--}`) cells and why")
    if ph:
        L.append("Each line is a row whose listed columns render as `--` because the backing input "
                 "row set is absent or empty:")
        L.append("")
        L += ph
    else:
        L.append("None -- every column has data.")
    L.append("")
    L.append("## Shared-set / `--strict` checks")
    probs = meta.get("strict_problems") or []
    notes = meta.get("coverage_notes") or []
    if probs:
        L.append("`--strict` violations (a present Table B/C column with mixed $n$ across rows):")
        L.append("")
        for p in probs:
            L.append(f"- {p}")
    else:
        L.append("- No `--strict` violations: every present Table B/C column has a single $n$ across its rows.")
    if notes:
        L.append("")
        L.append("Non-fatal coverage notes:")
        L.append("")
        for w in notes:
            L.append(f"- {w}")
    L.append("")
    L.append("### Known column-choice caveat")
    L.append("Our Transport rows carry Op-2's `transport_pct` (the S3 / `app_ref` column). The paper "
             "draft's refVFX zero-shot value 75.5 is from the `Look_u` column, not S3; the S3 refVFX "
             "value is ~62. Compare a system with itself, and only across rows on the same metric column.")
    L.append("")
    return "\n".join(L) + "\n"


# ===========================================================================
# preview.tex + build
# ===========================================================================
PREVIEW_TEX = r"""% GENERATED by scripts/build_metric_tables.py -- standalone preview of the grid v3 tables.
% Uses the paper's real macros (preamble.tex from ctt_iclr2027) + the vendored ICLR bib style,
% so \ltx{}, \segue{}, \ph{}, \citep{} and booktabs all render exactly as in the paper.
% Build:  ./build.sh   (sets TEXINPUTS to the paper folder, runs latexmk).
\documentclass{article}
\usepackage{natbib}          % \citep -- the conference style provides it in the real paper
\input{preamble}             % paper packages + macros (\ltx \segue \vap \vfxmaster \refvfx \ph ...)
\begin{document}
\begin{center}
{\Large\bfseries CTT / \segue{} metric tables --- grid v3 preview}\\[3pt]
%%BANNER%%
\small Generated %%DATE%% by \texttt{scripts/build\_metric\_tables.py}. A preview for owner review,
not the paper. Numbers join three per-generation feature sources on \texttt{(item\_id, seed)}.
Bold marks the best value per column per block.
\end{center}
\vspace{1em}
\input{tab_A}
\vspace{0.5em}
\input{tab_B}
\vspace{0.5em}
\input{tab_C}
\vspace{1em}
\noindent\textbf{Table D (ablation, guidance $w$-sweep).} Not built here: the $w$-sweep arms are not on
grid v3, so the paper's \texttt{tab\_ablation.tex} draft stands until they are regenerated.
\bibliographystyle{iclr2027_conference}
\bibliography{references}
\end{document}
"""

BUILD_SH = r"""#!/usr/bin/env bash
# Rebuild preview.pdf for the grid v3 metric tables (Op-5).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PAPER="%%PAPER%%"
export PATH="%%TEXBIN%%:$PATH"
export TEXINPUTS=".:$PAPER:$PAPER//:"
export BIBINPUTS="$TEXINPUTS"; export BSTINPUTS="$TEXINPUTS"
cd "$HERE"
latexmk -pdf -interaction=nonstopmode -file-line-error preview.tex
latexmk -c preview.tex >/dev/null 2>&1 || true   # drop aux, keep pdf
echo "OK -> $HERE/preview.pdf"
"""


def write_preview_and_build(out_dir: str, banner: Optional[str], do_build: bool) -> tuple[bool, str]:
    generated = date.today().isoformat()
    banner_tex = ("\\par\\medskip\\fcolorbox{red}{red!6}{\\parbox{0.9\\linewidth}{\\centering\\small"
                  "\\textbf{" + banner + "}}}\\par\\medskip") if banner else ""
    tex = PREVIEW_TEX.replace("%%BANNER%%", banner_tex).replace("%%DATE%%", generated)
    with open(os.path.join(out_dir, "preview.tex"), "w") as fh:
        fh.write(tex)
    build_sh = BUILD_SH.replace("%%PAPER%%", PAPER_DIR).replace("%%TEXBIN%%", TEXLIVE_BIN)
    bsh = os.path.join(out_dir, "build.sh")
    with open(bsh, "w") as fh:
        fh.write(build_sh)
    os.chmod(bsh, 0o755)
    if not do_build:
        return True, "skipped (--no-build)"
    env = dict(os.environ)
    env["PATH"] = TEXLIVE_BIN + ":" + env.get("PATH", "")
    env["TEXINPUTS"] = ".:" + PAPER_DIR + ":" + PAPER_DIR + "//:"
    env["BIBINPUTS"] = env["TEXINPUTS"]
    env["BSTINPUTS"] = env["TEXINPUTS"]
    try:
        r = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error", "preview.tex"],
            cwd=out_dir, env=env, capture_output=True, text=True, timeout=600)
        subprocess.run(["latexmk", "-c", "preview.tex"], cwd=out_dir, env=env,
                       capture_output=True, text=True, timeout=120)
        ok = os.path.exists(os.path.join(out_dir, "preview.pdf")) and r.returncode == 0
        tail = "\n".join(r.stdout.splitlines()[-25:])
        return ok, tail
    except FileNotFoundError:
        return False, "latexmk not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "latexmk timed out"


# ===========================================================================
# synthetic fixture (for tests and for a populated demo when inputs are absent)
# ===========================================================================
def write_fixture(base_dir: str) -> dict:
    """Write a small, deterministic 4-source fixture under base_dir/evals/*.

    Returns the roots dict the loaders expect.  Designed so:
      * own + external arms, all three tiers, one/two-sided, HF + ED families exist,
      * blocks are healthy (>= MIN_N one-sided rows) while the two-sided subset is
        deliberately thin (n < MIN_N) -> Identity/Motion B render "n/a" (min-n rule),
      * lens rows are present for only SOME arms (so a column goes placeholder),
      * a copy eval (M2a vs own reference) drives Copy rate,
      * a shared one-sided zero-shot set of a known size can be intersected.
    """
    d028 = os.path.join(base_dir, "evals", "028_fixture__dai__2026-09-18")
    d030 = os.path.join(base_dir, "evals", "030_fixture_ext__dai__2026-09-18")
    dhand = os.path.join(base_dir, "evals", "090_handoff_gridv3__dai__2026-09-18")
    dlens = os.path.join(base_dir, "evals", "091_lenses_gridv3__dai__2026-09-18")
    dcopy = os.path.join(base_dir, "evals", "092_copy_gridv3__dai__2026-09-18")

    per_gen: dict[str, list[dict]] = {}
    handoff: list[dict] = []
    lens: list[dict] = []
    copy: list[dict] = []

    # deterministic value knobs per base arm (so tables are legible + bolding testable)
    knob = {
        "base_cond": 0.55, "ic_gen": 0.62, "dualforce_control": 0.88, "dualforce_dcg_w6": 0.93,
        "vap": 0.81, "vfxmaster": 0.86, "refvfx": 0.62,
    }

    def variant_tail(kind, family):
        if family == "external":
            return "author_native"
        suffix = "v3ed81" if family == "ed" else "v3"
        return f"{kind}_{suffix}"

    def emit(base, kind, family, tier, sided, endpoint, reference, cell, seed, lens_present):
        arm_dir = f"{base}_{variant_tail(kind, family)}"
        item_id = f"G-fit__{arm_dir}__{endpoint}__ref_{reference}__s{seed}"
        k = knob[base]
        row = {
            "item_id": item_id, "seed": seed, "arm": arm_dir,
            "variant_dir": f"gens/{base}", "gen_video": f"gens/{base}/{item_id}.mp4",
            "tier": tier, "sided": sided, "cell": cell, "content": "same",
            "pct_type": "same", "endpoint": endpoint, "reference": reference,
            "ref_class": reference.split("_")[0], "n_frames": 121 if family == "hf" else 81,
            "transport_pct": round(100 * k + (2 if kind == "effect" else 0) + seed - 42, 2),
            "transport_raw": round(k, 4), "transport_ceiling": 0.9, "transport_capped": False,
            "copy_max": round(0.30 - 0.1 * k, 4), "near_copy": (k < 0.6 and seed == 42),
            "max_seam_z": round(2.0 - k, 3), "prefix_seam_z": round(1.5 - k, 3),
            "suffix_seam_z": round(1.2 - k, 3) if sided == "two" else None,
            "prefix_dino": round(0.90 + 0.05 * k, 4), "prefix_lpips": 0.01,
            "cam_zpr": 0.2, "obj_csls": 0.15, "core_degenerate": False,
            "cross_high": False, "ref_in_v4_population": True,
        }
        which = per_gen.setdefault(arm_dir, [])
        which.append(row)
        # handoff (present for all fixture gens)
        n_pre = 9 if family == "hf" else 1
        n_suf = 8 if sided == "two" else 0
        handoff.append({
            "item_id": item_id, "seed": seed, "arm": arm_dir, "n_pre": n_pre, "n_suf": n_suf,
            "K": 8, "fps": 24.0,
            "identity_A": round(0.85 + 0.1 * k, 4),
            "identity_B": round(0.80 + 0.1 * k, 4) if sided == "two" else float("nan"),
            "motion_A": round(0.5 + 0.4 * k, 4) if n_pre >= 9 else float("nan"),
            "motion_B": round(0.45 + 0.4 * k, 4) if (sided == "two" and n_pre >= 9) else float("nan"),
            "seam_free": 1 if k > 0.6 else 0, "missing": [],
        })
        # copy eval (M2a vs the gen's OWN reference) -- present for all fixture gens
        copy.append({
            "item_id": item_id, "seed": seed, "arm": arm_dir,
            "copy_max": round(0.30 - 0.1 * k, 4), "near_copy": (k < 0.6 and seed == 42),
            "copy_gen_frame": 61, "copy_ref_frame": 15,
            "ref_core_frac": 0.4, "n_mid": 40, "missing": [],
        })
        if lens_present:
            lens.append({
                "item_id": item_id, "seed": seed, "arm": arm_dir,
                "motion_smoothness": round(0.98 + 0.01 * (k - 0.6), 4),
                "videoprism_sim_ref": round(0.94 + 0.03 * k, 4),
                "det_motion_fidelity": round(0.12 + 0.12 * k, 4),
                "clip_sim_ref": round(0.6 + 0.1 * k, 4),
                "dynamic_degree_mean_mag": round(1.0 + k, 4),
                "aesthetic": round(4.5 + k, 4),
            })

    # endpoint/reference pools (distinct triples -> distinct match keys)
    seen_tr = [(f"seenA_{i}", f"seenR_{i}", "seen|same") for i in range(6)]        # 6 x2 = 12/arm
    unseen_one = [(f"unsA_{i}", f"unsR_{i}", "unseen|same") for i in range(6)]      # 6 x2 = 12/arm
    unseen_two = [(f"unsTwoA_{i}", f"unsTwoR_{i}", "unseen|cross") for i in range(2)]  # 2 x2 = 4/arm (THIN)
    zs_hf = [(f"zsA_{i}", f"zsR_{i}", "zero_shot|same") for i in range(8)]          # 8 x2 = 16/arm (shared)
    zs_ed = [(f"ed.Zs_{i}", f"ed.ZsR_{i}", "zero_shot|same") for i in range(2)]     # 2 x2 = 4/arm (ED only)

    for base in OWN_ARMS:
        for seed in (42, 43):
            for ep, ref, cell in seen_tr:
                for kind in ("neutral", "effect"):
                    emit(base, kind, "hf", "seen", "one", ep, ref, cell, seed, lens_present=True)
            for ep, ref, cell in unseen_one:
                for kind in ("neutral", "effect"):
                    emit(base, kind, "hf", "unseen", "one", ep, ref, cell, seed, lens_present=True)
            for ep, ref, cell in unseen_two:                       # two-sided -> thin Identity/Motion B
                emit(base, "neutral", "hf", "unseen", "two", ep, ref, cell, seed, lens_present=True)
            for ep, ref, cell in zs_hf:                            # part of the shared set
                for kind in ("neutral", "effect"):
                    emit(base, kind, "hf", "zero_shot", "one", ep, ref, cell, seed,
                         lens_present=(base != "base_cond"))
            for ep, ref, cell in zs_ed:                            # ED, dropped by the intersection
                for kind in ("neutral", "effect"):
                    emit(base, kind, "ed", "zero_shot", "one", ep, ref, cell, seed, lens_present=True)

    # Externals: author_native (=effect), one-sided zero_shot, on the same HF triples.
    for base in EXTERNAL_ARMS:
        for seed in (42, 43):
            for ep, ref, cell in zs_hf:
                emit(base, "author", "external", "zero_shot", "one", ep, ref, cell, seed,
                     lens_present=(base != "refvfx"))

    def dump(evdir, arm_rows):
        for arm_dir, rows in arm_rows.items():
            ad = os.path.join(evdir, arm_dir)
            os.makedirs(ad, exist_ok=True)
            with open(os.path.join(ad, "per_gen.jsonl"), "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

    own_rows = {a: r for a, r in per_gen.items() if not a.startswith(("vap", "vfxmaster", "refvfx"))}
    ext_rows = {a: r for a, r in per_gen.items() if a.startswith(("vap", "vfxmaster", "refvfx"))}
    dump(d028, own_rows)
    dump(d030, ext_rows)

    os.makedirs(os.path.join(dhand, "all"), exist_ok=True)
    with open(os.path.join(dhand, "all", "rows.jsonl"), "w") as fh:
        for r in handoff:
            fh.write(json.dumps(r) + "\n")
    os.makedirs(os.path.join(dlens, "all"), exist_ok=True)
    with open(os.path.join(dlens, "all", "rows.jsonl"), "w") as fh:
        for r in lens:
            fh.write(json.dumps(r) + "\n")
    os.makedirs(os.path.join(dcopy, "all"), exist_ok=True)
    with open(os.path.join(dcopy, "all", "rows.jsonl"), "w") as fh:
        for r in copy:
            fh.write(json.dumps(r) + "\n")

    return {
        "per_gen_roots": [d028, d030],
        "handoff_globs": [os.path.join(base_dir, "evals", "*_handoff_gridv3__*", "*", "rows.jsonl")],
        "lens_globs": [os.path.join(base_dir, "evals", "*_lenses_gridv3__*", "*", "rows.jsonl")],
        "copy_globs": [os.path.join(base_dir, "evals", "*_copy_gridv3__*", "*", "rows.jsonl")],
    }


# ===========================================================================
# driver
# ===========================================================================
def build_all(records) -> tuple[TableA, TableB, TableC]:
    return build_table_a(records), build_table_b(records), build_table_c(records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--per-gen-root", action="append", default=None,
                    help="eval root(s) holding <arm>/per_gen.jsonl (default: store/evals/028*, 030*)")
    ap.add_argument("--handoff-glob", action="append", default=None)
    ap.add_argument("--lens-glob", action="append", default=None)
    ap.add_argument("--copy-glob", action="append", default=None)
    ap.add_argument("--source", choices=["auto", "real", "fixture"], default="auto",
                    help="auto: real store if per_gen exists, else fixture (banner). "
                         "real: real store only. fixture: synthetic demo (banner).")
    ap.add_argument("--strict", action="store_true",
                    help="fail (exit 1) if any Table B/C column has mixed n")
    ap.add_argument("--no-build", action="store_true", help="write .tex/.md but skip latexmk")
    ap.add_argument("--emit-fixture", default=None,
                    help="write the synthetic fixture to this dir and exit (for tests/inspection)")
    args = ap.parse_args(argv)

    if args.emit_fixture:
        roots = write_fixture(args.emit_fixture)
        print("fixture written to", args.emit_fixture)
        print(json.dumps(roots, indent=1))
        return 0

    os.makedirs(args.out_dir, exist_ok=True)

    # resolve inputs
    real_per_gen = args.per_gen_root or [
        os.path.join(REPO_ROOT, "store", "evals", "028*"),
        os.path.join(REPO_ROOT, "store", "evals", "030*"),
    ]
    # expand any glob roots
    def expand(roots):
        out = []
        for r in roots:
            hits = sorted(glob.glob(r))
            out.extend(hits if hits else [r])
        return out
    real_per_gen = expand(real_per_gen)
    real_handoff = args.handoff_glob or [
        os.path.join(REPO_ROOT, "store", "evals", "*_handoff_gridv3__*", "*", "rows.jsonl")]
    real_lens = args.lens_glob or [
        os.path.join(REPO_ROOT, "store", "evals", "*_lenses_gridv3__*", "*", "rows.jsonl")]
    real_copy = args.copy_glob or [
        os.path.join(REPO_ROOT, "store", "evals", "*_copy_gridv3__*", "*", "rows.jsonl")]

    banner = None
    fixture_dir = None
    if args.source == "fixture":
        fixture_dir = os.path.join(args.out_dir, "_fixture_inputs")
        roots = write_fixture(fixture_dir)
        per_gen_roots, handoff_globs, lens_globs, copy_globs = (
            roots["per_gen_roots"], roots["handoff_globs"], roots["lens_globs"], roots["copy_globs"])
        banner = "SYNTHETIC FIXTURE DATA -- not real results. Demonstrates layout, join, bolding, placeholders."
    else:
        per_gen_roots, handoff_globs, lens_globs, copy_globs = (
            real_per_gen, real_handoff, real_lens, real_copy)
        have_per_gen = any(glob.glob(os.path.join(r, "*", "per_gen.jsonl")) for r in per_gen_roots)
        if not have_per_gen and args.source == "auto":
            fixture_dir = os.path.join(args.out_dir, "_fixture_inputs")
            roots = write_fixture(fixture_dir)
            per_gen_roots, handoff_globs, lens_globs, copy_globs = (
                roots["per_gen_roots"], roots["handoff_globs"], roots["lens_globs"], roots["copy_globs"])
            banner = ("SYNTHETIC FIXTURE DATA -- Op-2/Op-3/lens/copy inputs not yet present. "
                      "Rerun without --source fixture once they land.")

    # load + join
    per_gen, arm_dirs = load_per_gen(per_gen_roots)
    handoff = load_side(handoff_globs, HANDOFF_KEYS)
    lens = load_side(lens_globs, LENS_KEYS)
    copy = load_side(copy_globs, COPY_KEYS)
    records = join_records(per_gen, handoff, lens, copy)

    def _count_files(gl):
        n = 0
        for g in gl:
            n += len(glob.glob(g))
        return n

    meta = {
        "generated": date.today().isoformat(),
        "banner": banner,
        "inputs": {
            "per_gen": {"files": len(arm_dirs), "rows": len(per_gen),
                        "example": arm_dirs[0] if arm_dirs else None},
            "handoff": {"files": _count_files(handoff_globs), "rows": len(handoff),
                        "example": "rows.jsonl" if handoff else None},
            "lens": {"files": _count_files(lens_globs), "rows": len(lens),
                     "example": "rows.jsonl" if lens else None},
            "copy": {"files": _count_files(copy_globs), "rows": len(copy),
                     "example": "rows.jsonl" if copy else None},
        },
    }

    ta, tb, tc = build_all(records)

    # strict n-assertions + non-fatal coverage notes
    problems = check_strict(tb, tc)
    warnings = shared_set_warnings(tb)
    meta["strict_problems"] = problems
    meta["coverage_notes"] = warnings
    if problems:
        print("[strict] n-assertion findings:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
    if warnings:
        print("[coverage] shared-set notes:", file=sys.stderr)
        for w in warnings:
            print("  - " + w, file=sys.stderr)

    # write outputs
    with open(os.path.join(args.out_dir, "tab_A.tex"), "w") as fh:
        fh.write(render_table_a(ta))
    with open(os.path.join(args.out_dir, "tab_B.tex"), "w") as fh:
        fh.write(render_table_b(tb))
    with open(os.path.join(args.out_dir, "tab_C.tex"), "w") as fh:
        fh.write(render_table_c(tc))
    with open(os.path.join(args.out_dir, "TABLES.md"), "w") as fh:
        fh.write(render_markdown(ta, tb, tc, meta))

    ok, tail = write_preview_and_build(args.out_dir, banner, not args.no_build)

    # summary to stdout
    print(f"[build_metric_tables] out-dir: {args.out_dir}")
    print(f"  per_gen: {meta['inputs']['per_gen']['files']} arm files, {len(per_gen)} rows"
          + (f"  ({banner})" if banner else ""))
    print(f"  handoff: {meta['inputs']['handoff']['rows']} rows | "
          f"lens: {meta['inputs']['lens']['rows']} rows | copy: {meta['inputs']['copy']['rows']} rows")
    print(f"  Table B shared-set n={tb.diag.get('shared_n', 0)} (target 366); per-arm bench "
          f"{tb.diag.get('per_arm_n', {})}")
    print(f"  preview.pdf build: {'OK' if ok else 'FAILED'}")
    if not ok:
        print("  --- latexmk tail ---")
        print(tail)

    if args.strict and problems:
        print("[strict] FAIL: Table B/C column n mismatch (see findings above).", file=sys.stderr)
        return 1
    if not ok and not args.no_build:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
