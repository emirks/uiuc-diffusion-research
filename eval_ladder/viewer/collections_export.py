#!/usr/bin/env python3
"""Export the arm-comparison viewer's collections to Markdown + JSONL — the durable path.

    /usr/bin/python3.12 eval_ladder/viewer/collections_export.py --all
    /usr/bin/python3.12 eval_ladder/viewer/collections_export.py --collection teg_user_study
    /usr/bin/python3.12 eval_ladder/viewer/collections_export.py --collection teg_user_study --stdout
    /usr/bin/python3.12 eval_ladder/viewer/collections_export.py --check

Reads the ONE tracked record `eval_ladder/viewer/collections/neutral_effect_collections.json`
(everything an export needs is snapshotted there — this never touches data.js) and writes
`outputs/reports/iclora_neutral_effect_v2/collections/<id>.{md,jsonl}`.

The Markdown / JSONL wording is intentionally IDENTICAL to the browser's own "Export Markdown" /
"Export JSONL" buttons (the template mirrors these exact string operations), so the durable CLI path
and the convenience in-page path produce byte-for-byte the same file for the same data.

stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COLLECTIONS_FILE = HERE / "collections" / "neutral_effect_collections.json"
OUT_DIR = REPO_ROOT / "outputs/reports/iclora_neutral_effect_v2/collections"


# ---------------------------------------------------------------- shared formatting helpers
def _col_name(col: dict) -> str:
    """Human column name: the arm label, plus the entry label when it adds information.

    Same rule as the template's colName(): append the pill/entry label only when it is not already a
    substring of the arm label (so "Ⓐ refVFX · effect prompt" stays as-is, but a bare "neutral" pill
    on an arm whose label lacks the word becomes "… · neutral")."""
    lbl = col.get("arm_label") or col.get("arm") or "?"
    ent = col.get("label") or ""
    if ent and ent.lower() not in lbl.lower():
        return f"{lbl} · {ent}"
    return lbl


def _inputs_line(inp: dict) -> str:
    parts = []
    if inp.get("prefix_video"):
        parts.append("start (prefix)")
    if inp.get("sided") == "two" and inp.get("suffix_video"):
        parts.append("end (suffix)")
    for r in (inp.get("refs") or []):
        parts.append(f"reference ({r.get('cls') or '?'})")
    return " · ".join(parts) if parts else "—"


def _columns_line(columns: list) -> str:
    """Columns grouped by category, in appearance order; only the FIRST present column names its unit
    ("(N clips)"), the rest use "(N)"; absent columns read "— (no entry)". Matches the template."""
    groups = []          # [(category_label, [rendered column, ...])]
    index = {}
    first_present = [True]

    def render(col):
        name = _col_name(col)
        if col.get("present"):
            n = len(col.get("gens") or [])
            if first_present[0]:
                first_present[0] = False
                return f"{name} ✓ ({n} clips)"
            return f"{name} ✓ ({n})"
        return f"{name} — (no entry)"

    for col in columns:
        clabel = col.get("category_label") or col.get("category") or "?"
        if clabel not in index:
            index[clabel] = len(groups)
            groups.append((clabel, []))
        groups[index[clabel]][1].append(render(col))
    return " | ".join(f"{clabel}: " + " · ".join(cols) for clabel, cols in groups)


def _notes_body(notes: str) -> str:
    n = notes or ""
    if n.strip() == "":
        return "—"
    return n.replace("\n", "\n  ")


def _tier_export(tier: dict) -> str:
    """The row's tier from the snapshot's RAW ids — mirrors the template's tierExport() byte-for-byte.
    Items saved before per-reference rows lack the field and render with no tier segment."""
    if not tier:
        return ""
    nov = "+".join(tier.get("novelty") or [])
    con = "+".join(tier.get("content") or [])
    cel = " ".join(tier.get("cells") or [])
    return " · ".join(x for x in [nov, con, cel] if x)


def export_markdown(col: dict) -> str:
    items = col.get("items") or []
    date = (col.get("updated") or "")[:10]
    title = col.get("title") or col.get("id")
    lines = [f"# {title} ({len(items)} rows) — {date}"]
    cnotes = col.get("notes") or ""
    if cnotes.strip() != "":
        lines.append("")
        lines.append(cnotes)
    for i, item in enumerate(items, 1):
        inp = item.get("inputs") or {}
        view = item.get("view") or {}
        tstr = _tier_export(inp.get("tier"))
        lines.append("")
        lines.append(f"## row {i} — {inp.get('endpoint') or '?'} · {inp.get('donor') or '?'} · "
                     f"{inp.get('sided') or '?'}-sided" + (f" · {tstr}" if tstr else "")
                     + f" · seed {view.get('seed') or '?'}")
        lines.append(f"inputs: {_inputs_line(inp)}")
        lines.append(f"columns: {_columns_line(item.get('columns') or [])}")
        lines.append(f"notes: {_notes_body(item.get('notes'))}")
    return "\n".join(lines) + "\n"


def export_jsonl(col: dict) -> str:
    out = []
    for i, item in enumerate(col.get("items") or [], 1):
        inp = item.get("inputs") or {}
        refs = [r.get("video") for r in (inp.get("refs") or [])]
        base = {
            "row_index": i,
            "card_key": item.get("card_key"),
            "prefix_video": inp.get("prefix_video"),
            "suffix_video": inp.get("suffix_video"),
            "endpoint_video": inp.get("endpoint_video"),
            "refs": refs,
            "row_tier": inp.get("tier"),   # named distinctly: per-column "tier" (arm id) also merges into each row
        }
        for col_ in (item.get("columns") or []):
            colfields = {
                "category": col_.get("category"),
                "tier": col_.get("tier"),
                "arm": col_.get("arm"),
                "variant": col_.get("variant"),
                "label": col_.get("label"),
                "present": bool(col_.get("present")),
            }
            gens = col_.get("gens") or []
            if col_.get("present") and gens:
                for g in gens:
                    row = {**base, **colfields,
                           "seed": g.get("seed"), "video": g.get("video"),
                           "ref_video": g.get("ref_video"), "scored": g.get("scored"),
                           "pct": g.get("pct")}
                    out.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            else:
                row = {**base, **colfields,
                       "seed": None, "video": None, "ref_video": None,
                       "scored": None, "pct": None}
                out.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(out) + ("\n" if out else "")


# ---------------------------------------------------------------- load / check
def load_doc() -> dict:
    if not COLLECTIONS_FILE.exists():
        sys.exit(f"[export] no collections file at {COLLECTIONS_FILE}")
    with COLLECTIONS_FILE.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc


def cmd_check(doc: dict) -> int:
    problems = []
    if not isinstance(doc.get("schema"), int) or isinstance(doc.get("schema"), bool):
        problems.append("top-level 'schema' missing or not an integer")
    if not isinstance(doc.get("collections"), list):
        problems.append("top-level 'collections' missing or not a list")
        for p in problems:
            print(f"  SCHEMA {p}")
        return 1
    n_cols = n_items = 0
    checked = missing = 0
    missing_examples = []

    def check_path(p, where):
        nonlocal checked, missing
        if not p:
            return
        checked += 1
        if not (REPO_ROOT / p).exists():
            missing += 1
            if len(missing_examples) < 10:
                missing_examples.append(f"{where}: {p}")

    for col in doc["collections"]:
        n_cols += 1
        if not col.get("id"):
            problems.append("a collection has no id")
        if "items" not in col or not isinstance(col["items"], list):
            problems.append(f"collection {col.get('id')} has no items list")
            continue
        for i, item in enumerate(col["items"], 1):
            n_items += 1
            where = f"{col.get('id')}#{i}"
            if not item.get("card_key"):
                problems.append(f"{where}: no card_key")
            inp = item.get("inputs") or {}
            check_path(inp.get("prefix_video"), where + " prefix")
            check_path(inp.get("suffix_video"), where + " suffix")
            check_path(inp.get("endpoint_video"), where + " endpoint")
            for r in (inp.get("refs") or []):
                check_path(r.get("video"), where + " ref")
            for col_ in (item.get("columns") or []):
                for g in (col_.get("gens") or []):
                    check_path(g.get("video"), where + " gen")
                    check_path(g.get("ref_video"), where + " gen-ref")
    print(f"[check] schema={doc.get('schema')} viewer={doc.get('viewer')}")
    print(f"[check] {n_cols} collections · {n_items} items")
    print(f"[check] video paths: {checked} checked · {missing} missing")
    for e in missing_examples:
        print(f"        MISSING {e}")
    for p in problems:
        print(f"        PROBLEM {p}")
    return 1 if (missing or problems) else 0


def write_exports(col: dict, to_stdout: bool) -> None:
    md, jsonl = export_markdown(col), export_jsonl(col)
    if to_stdout:
        print(f"===== {col['id']}.md =====")
        sys.stdout.write(md)
        print(f"===== {col['id']}.jsonl =====")
        sys.stdout.write(jsonl)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{col['id']}.md").write_text(md, encoding="utf-8")
    (OUT_DIR / f"{col['id']}.jsonl").write_text(jsonl, encoding="utf-8")
    print(f"[export] {col['id']}: {len(col.get('items') or [])} rows -> "
          f"{(OUT_DIR / (col['id'] + '.md')).relative_to(REPO_ROOT)} + .jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", help="collection id to export")
    ap.add_argument("--all", action="store_true", help="export every collection")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing files")
    ap.add_argument("--check", action="store_true",
                    help="validate the schema and that every snapshotted video path exists (exit 1 on any miss)")
    args = ap.parse_args()

    doc = load_doc()
    if args.check:
        sys.exit(cmd_check(doc))

    cols = doc.get("collections") or []
    by_id = {c["id"]: c for c in cols}
    if args.all:
        targets = cols
    elif args.collection:
        if args.collection not in by_id:
            sys.exit(f"[export] no collection '{args.collection}' (have: {', '.join(by_id)})")
        targets = [by_id[args.collection]]
    else:
        sys.exit("[export] pass --collection <id>, --all, or --check")
    for col in targets:
        write_exports(col, args.stdout)


if __name__ == "__main__":
    main()
