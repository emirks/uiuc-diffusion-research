#!/usr/bin/env python3
"""store_register — close out a gen into the store (contract v2). Ends the no-writer era.

  python scripts/store_register.py gen store/gens/009_ctt_v3/05_neutral__dai \
      --registry eval_ladder/registry_ctt_v4_neutral.jsonl \
      --run runs/008_ctt_v3_wsd --step 6000 --code "src/LTX-2-surg @ a4033230" [--notes "..."]

Reads arm/variant/machine from the subentry path (gens/NNN_<arm>/KK_<variant>__<machine>),
copies the registry in as grid.jsonl, verifies the prompt sha against the prompts/ shelf,
counts videos/, writes meta.yaml (REFUSES to overwrite — entries are immutable), and prints
the INDEX.md row to paste. Register = this + INDEX row + CHANGELOG, one commit, at close.
"""
import argparse, hashlib, json, shutil, subprocess, sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = REPO_ROOT / "store"


def key(r):
    return (r["cell"], r["endpoint"], r.get("reference") or "", r["sided"])


def corpus_sha(rows):
    uniq = {key(r): r for r in rows}
    blob = "".join(r["prompt"] for r in sorted(uniq.values(), key=key))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def family_of(sha):
    for fam in sorted((STORE / "prompts").iterdir()):
        meta = fam / "meta.yaml"
        if meta.exists() and sha in meta.read_text():  # matches prompt_corpus_sha OR a derived: transform sha
            return f"prompts/{fam.name}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["gen"])
    ap.add_argument("subentry", help="store/gens/NNN_<arm>/KK_<variant>__<machine>")
    ap.add_argument("--registry", required=True, help="the exact stamped rows the gen consumed")
    ap.add_argument("--run", required=True, help="runs/<id> (or 'base weights (no adapter)')")
    ap.add_argument("--step", default=None)
    ap.add_argument("--code", required=True, help="code pin: <checkout> @ <commit>")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    sub = Path(args.subentry).resolve()
    arm_dir, sub_slug = sub.parent.name, sub.name
    assert sub.parent.parent == (STORE / "gens").resolve(), f"{sub} is not a gens subentry"
    kk, _, rest = sub_slug.partition("_")
    variant, sep, machine = rest.rpartition("__")[0], "__", rest.rpartition("__")[2]
    assert kk.isdigit() and variant and machine in ("cc", "eps", "dai"), \
        f"subentry slug must be KK_<variant>__<cc|eps|dai>, got {sub_slug}"

    meta_p, grid_p, videos = sub / "meta.yaml", sub / "grid.jsonl", sub / "videos"
    assert not meta_p.exists(), f"{meta_p} exists — entries are immutable; next KK for a re-run"
    assert videos.is_dir(), f"{videos} missing — generate with --out-root {sub} --videos-dir first"

    reg = Path(args.registry)
    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    harness = {r["arm"] for r in rows}
    assert len(harness) == 1, f"registry stamps multiple arms: {harness}"
    harness = harness.pop()
    sha = corpus_sha(rows) if all("prompt" in r for r in rows) else None
    family = family_of(sha) if sha else None
    n_mp4 = len(list(videos.glob("*.mp4")))

    if not grid_p.exists():
        shutil.copy2(reg, grid_p)
    meta = f"""id: gens/{arm_dir}/{sub_slug}
seq: {int(kk)}
shelf: gens
arm: {arm_dir[4:]}
harness_arm: {harness}   # frozen stamp inside items.jsonl/item_ids — joins use THIS
variant: {variant}
machine: {machine}
created: {date.today().isoformat()}
inputs: {{run: {args.run}{', step: ' + args.step if args.step else ''}}}
code: {args.code}
prompt_family: {family or 'none (probe/custom rows)'}
prompt_sha: {sha or 'n/a'}
grid_rows: {len(rows)}
videos: {n_mp4}
source: scripts/store_register.py over {reg}
"""
    if args.notes:
        meta += f"notes: {args.notes}\n"
    if sha and not family:
        print(f"!! prompt sha {sha} matches NO prompts/ family — hand-written registry? "
              f"stamp_rows.py is the sanctioned path", file=sys.stderr)
        sys.exit(1)
    meta_p.write_text(meta)
    print(f"[register] {meta_p} written  harness_arm={harness} videos={n_mp4} rows={len(rows)} prompt_sha={sha}")
    print(f"[register] INDEX row (paste under the arm's line, then CHANGELOG + commit):\n"
          f"  `{sub_slug}` — {args.run}{'@' + args.step if args.step else ''}, {variant}, {machine}")
    print("[register] now run: python scripts/store_fsck.py")


main()
