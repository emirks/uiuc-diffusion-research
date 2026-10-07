#!/usr/bin/env python3
"""stamp_rows — derive an arm-stamped registry from a canonical prompt family (contract v2).

The prompts/ shelf holds ARM-FREE rows; this is the ONLY sanctioned way to turn a family
into a per-arm registry (hand-writing one is a contract violation — store/README.md §5).

  python eval_ladder/stamp_rows.py --family 002_ctt152_effect --arm ctt_v4_effect \
      --out eval_ladder/registry_ctt_v4_effect.jsonl [--set conditioning=prefix] [--set use_reference=false]

- `arm` should be the new harness_arm grammar `<arm>_<variant>` (single underscore).
- item_id is rendered with the frozen grammar: <cell>__<arm>__<endpoint>[__ref_<reference>]
  (same as eval_ladder/build_registry.py) so old and new artifacts stay shape-compatible.
- The family's prompt_corpus_sha is re-verified before stamping; print it, pin it in the
  gen's meta.yaml as prompt_sha.
"""
import argparse, hashlib, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = REPO_ROOT / "store/prompts"


def key(r):
    return (r["cell"], r["endpoint"], r.get("reference") or "", r["sided"])


def corpus_sha(rows):
    uniq = {key(r): r for r in rows}
    blob = "".join(r["prompt"] for r in sorted(uniq.values(), key=key))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def parse_val(v: str):
    if v in ("true", "false"):
        return v == "true"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, help="prompts/ entry, e.g. 002_ctt152_effect")
    ap.add_argument("--arm", required=True, help="harness_arm to stamp (grammar: <arm>_<variant>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="arm-contract field applied to every row (conditioning=prefix, use_reference=false, no_twin=true, ...)")
    ap.add_argument("--strip-token", action="store_true",
                    help="base-arm transform: remove ' sksz.' from prompts (sksz means nothing to base weights)")
    ap.add_argument("--cells", default=None,
                    help="OPTIONAL comma-separated cell filter -- a row SUBSET of the family, "
                         "prompts byte-identical, yielding a DERIVED corpus sha to pin in meta. "
                         "Default None => the whole family, byte-identical to every prior stamp.")
    ap.add_argument("--token", default=None, metavar="TEXT",
                    help="replace the literal 'sksz' with TEXT (external-model text budget, refvfx-B precedent)")
    args = ap.parse_args()
    assert not (args.strip_token and args.token), "--strip-token and --token are mutually exclusive"

    fam = PROMPTS / args.family
    rows = [json.loads(l) for l in (fam / "grid.jsonl").read_text().splitlines() if l.strip()]
    declared = next((l.split(":", 1)[1].split("#")[0].strip()
                     for l in (fam / "meta.yaml").read_text().splitlines()
                     if l.startswith("prompt_corpus_sha:")), None)
    sha = corpus_sha(rows)
    assert sha == declared, f"family sha drifted: computed {sha} != declared {declared}"

    def transform(text):
        if args.strip_token:
            return text.replace(" sksz.", "", 1)
        if args.token:
            return text.replace("sksz", args.token)
        return text
    if args.strip_token or args.token:
        for r in rows:
            r["prompt"] = transform(r["prompt"])
            if "prompt_base" in r:
                r["prompt_base"] = transform(r["prompt_base"])
        sha = corpus_sha(rows)  # the DERIVED sha — pin THIS in the gen meta

    if args.cells:
        want = {c.strip() for c in args.cells.split(",") if c.strip()}
        before = len(rows)
        rows = [r for r in rows if r["cell"] in want]
        assert rows, f"cell filter {sorted(want)} selected 0 of {before} rows"
        sha = corpus_sha(rows)  # DERIVED sha over the subset -- pin THIS in the gen meta
        print(f"[stamp] cell filter {sorted(want)}: {len(rows)}/{before} rows, derived sha {sha}")

    extra = dict(kv.split("=", 1) for kv in args.set)
    out = Path(args.out)
    with out.open("w") as f:
        for r in sorted(rows, key=key):
            s = dict(r)
            s["arm"] = args.arm
            ref = s.get("reference")
            s["item_id"] = f"{s['cell']}__{args.arm}__{s['endpoint']}" + (f"__ref_{ref}" if ref else "")
            for k, v in extra.items():
                s[k] = parse_val(v)
            f.write(json.dumps(s, sort_keys=True) + "\n")
    print(f"[stamp] {len(rows)} rows -> {out}  arm={args.arm}  prompt_sha={sha} (pin this in the gen meta)")


main()
