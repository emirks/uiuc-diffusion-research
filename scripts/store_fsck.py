#!/usr/bin/env python3
"""store_fsck — validate the artifact store against contract v2. Run at every registration.

Checks: gens subentry schema + grid/video counts + prompt-sha pins, prompts/ family shas,
_legacy shim resolution, runs/ checkpoints presence, evals/ scorer layout. Exit 1 on any FAIL.
"""
import hashlib, json, sys
from pathlib import Path

STORE = Path(__file__).resolve().parents[1] / "store"
REQ_GEN_KEYS = ("id:", "seq:", "arm:", "harness_arm:", "variant:", "machine:",
                "created:", "inputs:", "prompt_family:", "prompt_sha:", "grid_rows:", "videos:")
fails, warns = [], []


def key(r):
    return (r["cell"], r["endpoint"], r.get("reference") or "", r["sided"])


def _row_text(r):
    # two-channel shelves (e.g. 008_ext112_authorcfg) declare target_text/ref_text; single-channel
    # shelves + gen rows carry `prompt` (the target/generation text). Match each shelf's own sha convention.
    if "target_text" in r:
        return r["target_text"] + "\x1f" + (r.get("ref_text") or "")
    return r["prompt"]


def corpus_sha(rows):
    uniq = {key(r): r for r in rows}
    blob = "".join(_row_text(r) for r in sorted(uniq.values(), key=key))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def meta_val(text, k):
    for line in text.splitlines():
        if line.startswith(k):
            return line.split(":", 1)[1].split("#")[0].strip()
    return None


def check_gens():
    for arm_dir in sorted((STORE / "gens").iterdir()):
        if not arm_dir.is_dir() or arm_dir.name == "_legacy":
            continue
        for sub in sorted(arm_dir.iterdir()):
            tag = f"gens/{arm_dir.name}/{sub.name}"
            meta_p = sub / "meta.yaml"
            if not meta_p.exists():
                if (sub / ".inflight").exists():
                    warns.append(f"{tag}: in-flight (generation running, not yet registered)")
                else:
                    fails.append(f"{tag}: no meta.yaml")
                continue
            meta = meta_p.read_text()
            for k in REQ_GEN_KEYS:
                if k not in meta:
                    fails.append(f"{tag}: meta missing `{k}`")
            void = meta_val(meta, "void:") == "true"
            grid = sub / "grid.jsonl"
            if not grid.exists():
                fails.append(f"{tag}: no grid.jsonl"); continue
            rows = [json.loads(l) for l in grid.read_text().splitlines() if l.strip()]
            if str(len(rows)) != meta_val(meta, "grid_rows:"):
                fails.append(f"{tag}: grid_rows {meta_val(meta,'grid_rows:')} != actual {len(rows)}")
            n = len(list((sub / "videos").glob("*.mp4"))) if (sub / "videos").is_dir() else -1
            if n < 0 and not void:
                fails.append(f"{tag}: no videos/")
            elif str(n) != meta_val(meta, "videos:"):
                fails.append(f"{tag}: videos {meta_val(meta,'videos:')} != actual {n}")
            sc = sub / "scores"
            if sc.is_symlink() and not sc.resolve().is_dir():
                fails.append(f"{tag}: broken scores link")
            fam = meta_val(meta, "prompt_family:")
            if fam and fam.startswith("prompts/") and all("prompt" in r for r in rows):
                sha = corpus_sha(rows)
                if sha != meta_val(meta, "prompt_sha:"):
                    fails.append(f"{tag}: prompt_sha {meta_val(meta,'prompt_sha:')} != computed {sha}")


def check_prompts():
    for fam in sorted((STORE / "prompts").iterdir()):
        meta = (fam / "meta.yaml").read_text()
        if "probe" in (meta_val(meta, "family:") or ""):
            continue  # probe sets share row keys across conditions; pinned by file sha in meta
        rows = [json.loads(l) for l in (fam / "grid.jsonl").read_text().splitlines() if l.strip()]
        sha = corpus_sha(rows)
        if sha != meta_val(meta, "prompt_corpus_sha:"):
            fails.append(f"prompts/{fam.name}: sha drift {sha} != {meta_val(meta,'prompt_corpus_sha:')}")


def check_legacy():
    for shim in sorted((STORE / "gens/_legacy").iterdir()):
        if not shim.resolve().is_dir():
            fails.append(f"gens/_legacy/{shim.name}: broken shim")


def check_runs():
    for run in sorted((STORE / "runs").iterdir()):
        if not run.is_dir():
            continue
        if not (run / "meta.yaml").exists():
            fails.append(f"runs/{run.name}: no meta.yaml")
        ck = run / "checkpoints"
        has_ck = ck.exists() and any(ck.iterdir())
        # encoder/projector runs (e.g. runs/005) ship .pt artifacts at entry root
        has_root_model = any(run.glob("*.pt")) or any(run.glob("*.safetensors"))
        if not has_ck and not has_root_model:
            (warns if "external" in (run / "meta.yaml").read_text() else fails).append(
                f"runs/{run.name}: no checkpoints/ and no root model artifact")


def check_evals():
    for ev in sorted((STORE / "evals").iterdir()):
        if not ev.is_dir():
            continue
        if not (ev / "meta.yaml").exists():
            fails.append(f"evals/{ev.name}: no meta.yaml"); continue
        for arm in sorted(p for p in ev.iterdir() if p.is_dir()):
            if not list(arm.glob("*/results.json")):
                warns.append(f"evals/{ev.name}/{arm.name}: no <label>/results.json")


check_gens(); check_prompts(); check_legacy(); check_runs(); check_evals()
for w in warns:
    print(f"  warn: {w}")
for f in fails:
    print(f"  FAIL: {f}")
print(f"store_fsck: {'PASS' if not fails else 'FAIL'} ({len(fails)} fails, {len(warns)} warns)")
sys.exit(1 if fails else 0)
