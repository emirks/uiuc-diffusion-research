#!/usr/bin/env python
"""metrics v5 A10 — every number in the paper-style table equals the corresponding cell of
metrics_v2_gridv3_clean/TABLES.md (Table 2 for the transfer block, Table 3 for the TEG block).

Parses papers_drafts/_preview/paper_table_v5/tab_main.tex and papers_drafts/_preview/
metrics_v2_gridv3_clean/TABLES.md, matches the paper rows to the metrics_v2 rows by roster id ->
label, and compares each of the 12 metric cells. Reports mismatches (must be 0) and any paper row
with no metrics_v2 counterpart (e.g. Plain LoRA in the transfer block -- metrics_v2 Table 2 does not
list the baseline LoRA; its value is computed over the identical shared set and is therefore
unverifiable against that table, not a mismatch).

Run:  PYTHONPATH=<repo>/src $LAB/envs-aarch64/ltx2/bin/python scripts/metrics_v5/paper_table_check.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
import v5  # noqa: E402

PAPER = REPO / "papers_drafts/_preview/paper_table_v5/tab_main.tex"
CLEAN = REPO / "papers_drafts/_preview/metrics_v2_gridv3_clean/TABLES.md"
# the metrics_v2 md column order (family_tables metrics_v2 groups) -> metric key. Since metrics v5 R4
# the Transition-fidelity group carries Flow MSE + Action KL (eval 049) between Motion fid. and Ref sim.,
# so the md table now has 14 metric columns; A10 matches the 12 old paper-table columns by KEY, so the two
# extra md keys are simply not looked up (the old paper_table_v5 keeps its 12 columns; brief R4 D4).
MD_COLS = ["ep_id_A", "ep_id_B", "ep_mot_A", "ep_mot_B", "text_own", "transport",
           "motfid", "flow_mse", "action_kl", "vp_ref", "seam_free", "smooth_native", "dyn_pxs", "aesthetic"]
PAPER_KEYS = [k for k, _ in v5.PAPER_COLS]


def _num(cell):
    """A table cell string -> float or None ('--'); strips \\textbf{} / ** and tiny-n markup."""
    s = cell.strip()
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace("**", "").replace(r"\,", "").strip()
    if s in ("--", "—", ""):
        return None
    m = re.match(r"-?\d+\.?\d*", s)
    return float(m.group(0)) if m else None


def _parse_paper():
    """{'teg': {roster_id: {key: val}}, 'vet': {...}} from tab_main.tex, positionally."""
    lines = PAPER.read_text().splitlines()
    blocks, cur = {"teg": [], "vet": []}, None
    for ln in lines:
        s = ln.strip()
        if "both endpoints given" in s:
            cur = "teg"; continue
        if "start endpoint given" in s:
            cur = "vet"; continue
        if cur is None or not s.endswith(r"\\") or s.startswith(("\\multicolumn", "\\cmidrule", "\\midrule",
                                                                  "\\toprule", "\\bottomrule", "\\addlinespace",
                                                                  "&", "\\shortstack")):
            continue
        cells = [c.strip() for c in s[:-2].split("&")]
        if len(cells) != 13:
            continue
        blocks[cur].append([_num(c) for c in cells[1:]])
    order = {"teg": [a for a in v5.PAPER_TEG_ROWS if a != "__addlinespace__"],
             "vet": [a for a in v5.PAPER_VET_ROWS if a != "__addlinespace__"]}
    out = {}
    for blk in ("teg", "vet"):
        rows = blocks[blk]
        assert len(rows) == len(order[blk]), f"{blk}: parsed {len(rows)} rows, expected {len(order[blk])}"
        out[blk] = {aid: {PAPER_KEYS[i]: rows[r][i] for i in range(12)} for r, aid in enumerate(order[blk])}
    return out


def _parse_md_table(header):
    """{label: {key: val}} for the metrics_v2 md table under `header`."""
    text = CLEAN.read_text().splitlines()
    i = next(k for k, ln in enumerate(text) if ln.startswith(header))
    rows = {}
    for ln in text[i + 1:]:
        if ln.startswith("## "):
            break
        if not ln.startswith("|") or ln.startswith("| arm") or set(ln.strip()) <= set("|-"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        label = cells[0]
        if not label or label.startswith("**"):
            continue
        vals = [_num(c) for c in cells[1:1 + len(MD_COLS)]]
        rows[label] = {MD_COLS[j]: vals[j] for j in range(len(MD_COLS))}
    return rows


def main() -> int:
    id2label = {a["id"]: a["label"] for a in v5.roster()["arms"]}
    paper = _parse_paper()
    md = {"teg": _parse_md_table("## Table 3"), "vet": _parse_md_table("## Table 2")}
    mism, unmatched, checked = [], [], 0
    for blk in ("teg", "vet"):
        for aid, cells in paper[blk].items():
            label = id2label[aid]
            mrow = md[blk].get(label)
            if mrow is None:
                unmatched.append(f"{blk}:{aid} ({label}) -- no metrics_v2 {blk} row")
                continue
            for key in PAPER_KEYS:
                pv, mv = cells[key], mrow.get(key)
                checked += 1
                if pv is None and mv is None:
                    continue
                if pv is None or mv is None or abs(pv - mv) > 1e-9:
                    mism.append(f"{blk}:{aid}.{key}: paper={pv} metrics_v2={mv}")
    print(f"[A10] checked {checked} matched cells; mismatches {len(mism)}; unmatched paper rows {len(unmatched)}")
    for u in unmatched:
        print("   UNMATCHED:", u)
    for m in mism:
        print("   MISMATCH :", m)
    print(f"[A10] {'PASS' if not mism else 'FAIL'} (every matched paper number == its metrics_v2 cell)")
    return 0 if not mism else 1


if __name__ == "__main__":
    sys.exit(main())
