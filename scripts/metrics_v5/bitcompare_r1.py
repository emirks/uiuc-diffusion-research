#!/usr/bin/env python
"""A6 bit-compare: diff each rebuilt metrics_v2 TABLES.md against the frozen baseline in
misc/2026-09-23_metrics_v5/before/, per table, per cell, aligned by (table, tier, arm).
Every cell except Transport and Seam-free must be identical; new rows (the neutral twins,
rendering '--') are reported as expected additions. Writes BITCOMPARE_R1.md."""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BEFORE = REPO / "misc/2026-09-23_metrics_v5/before"
PREVIEW = REPO / "papers_drafts/_preview"
OUT = REPO / "misc/2026-09-23_metrics_v5/BITCOMPARE_R1.md"
VARIANTS = ["metrics_v2_gridv3", "metrics_v2_gridv3_clean", "metrics_v2_gridv3_sweep", "metrics_v2_gridv3_sweep_clean"]


def parse(md_text: str):
    """-> {section: {"header": [cols], "rows": {(tier, arm): [cells]}}} for Table 1/2/3."""
    out = {}
    section = None
    tier = None
    header = None
    for ln in md_text.splitlines():
        m = re.match(r"## (Table \d)", ln)
        if m:
            section = m.group(1); tier = None; header = None
            out[section] = {"header": None, "rows": {}}
            continue
        if ln.startswith("## "):        # left the table sections (e.g. "## Columns")
            section = None
            continue
        if section is None or not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] == "arm":
            header = cells; out[section]["header"] = cells; continue
        if set("".join(cells)) <= set("-"):   # separator row
            continue
        label = cells[0]
        mt = re.match(r"\*\*(.+)\*\*", label)   # tier row: **Seen** etc.
        if mt and all(c == "" for c in cells[1:]):
            tier = mt.group(1); continue
        out[section]["rows"][(tier, label)] = cells
    return out


def compare(old_md, new_md):
    old, new = parse(old_md), parse(new_md)
    report = []
    ok = True
    for sec in ("Table 1", "Table 2", "Table 3"):
        o, n = old.get(sec, {"rows": {}, "header": None}), new.get(sec, {"rows": {}, "header": None})
        hdr = n["header"] or o["header"] or []
        def idx(name_sub):
            for i, c in enumerate(hdr):
                if name_sub in c:
                    return i
            return None
        i_transport = idx("Transport")
        i_seam = idx("Seam-free")
        okeys, nkeys = set(o["rows"]), set(n["rows"])
        shared = [k for k in n["rows"] if k in okeys]   # roster order
        added = [k for k in n["rows"] if k not in okeys]
        removed = [k for k in okeys if k not in nkeys]
        report.append(f"### {sec}")
        report.append(f"- shared rows: {len(shared)}; added rows (new, expect neutral twins as `--`): {len(added)}; removed rows: {len(removed)}")
        cell_fail = []
        transport_changes = []
        seam_changes = []
        for k in shared:
            oc, nc = o["rows"][k], n["rows"][k]
            L = min(len(oc), len(nc))
            for i in range(L):
                if oc[i] == nc[i]:
                    continue
                if i == i_transport:
                    transport_changes.append((k, oc[i], nc[i]))
                elif i == i_seam:
                    seam_changes.append((k, oc[i], nc[i]))
                else:
                    cell_fail.append((k, hdr[i] if i < len(hdr) else f"col{i}", oc[i], nc[i]))
            if len(oc) != len(nc):
                cell_fail.append((k, "COLUMN COUNT", str(len(oc)), str(len(nc))))
        if cell_fail:
            ok = False
            report.append(f"- **UNEXPECTED cell diffs (FAIL): {len(cell_fail)}**")
            for k, col, a, b in cell_fail[:40]:
                report.append(f"    - {k} [{col}]: '{a}' -> '{b}'")
        else:
            report.append("- non-Transport/non-Seam cells: IDENTICAL on every shared row")
        report.append(f"- Transport changes: {len(transport_changes)}")
        for k, a, b in transport_changes:
            report.append(f"    - {k[0]} / {k[1]}: {a} -> {b}")
        report.append(f"- Seam-free changes: {len(seam_changes)}")
        for k, a, b in seam_changes:
            report.append(f"    - {k[0]} / {k[1]}: {a} -> {b}")
        if added:
            report.append(f"- added rows: " + "; ".join(f"{k[0]}/{k[1]}" for k in added))
        if removed:
            ok = False
            report.append(f"- **REMOVED rows (FAIL): " + "; ".join(f"{k[0]}/{k[1]}" for k in removed) + "**")
        report.append("")
    return ok, report


def main():
    lines = ["# metrics v5 — A6 bit-compare (Round 1)", "",
             "Rebuilt `papers_drafts/_preview/metrics_v2_gridv3*` TABLES.md vs the frozen baseline in "
             "`misc/2026-09-23_metrics_v5/before/`. Rule: every cell except **Transport** and **Seam-free** identical "
             "on the shared (pre-existing) rows; the neutral prior-work rows are new and render `--` (Round 1, no features).", ""]
    all_ok = True
    for v in VARIANTS:
        old = (BEFORE / f"{v}.TABLES.md")
        new = (PREVIEW / v / "TABLES.md")
        lines.append(f"## {v}")
        if not old.exists() or not new.exists():
            lines.append(f"- MISSING: old={old.exists()} new={new.exists()}"); all_ok = False; lines.append(""); continue
        ok, rep = compare(old.read_text(), new.read_text())
        all_ok = all_ok and ok
        lines += rep
    lines.append(f"## VERDICT: {'PASS' if all_ok else 'FAIL'} (non-Transport/non-Seam cells identical, no removed rows)")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[bitcompare] wrote {OUT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
