"""CTT v2 — build the tiny MIXED-FORMAT root the mandatory smoke gate runs on (A9 §3 item 3).

A9: *"The A1b mixed-format smoke gate (mandatory, 1-2 GPU-h) additionally asserts per-format
finite comparable loss and realized shifts in {1.120, 2.302} exactly"*, and *"placeholder
captions suffice — it tests mechanics, not caption quality"*.

Design decisions, all of them made to keep FORMAT the only thing that varies
---------------------------------------------------------------------------
* **Three arms, not two.**  The primary comparison is `S1_121f_one` vs `S4_33f_one`:
  sidedness is held CONSTANT at one-sided across it, so the conditioned-token fraction is
  the only remaining geometric difference (and that difference is itself a consequence of
  the format — see below).  `S2a_121f_two` is carried as a third arm purely to exercise the
  two-sided mask reshape at 121f in the same process.
* **One single placeholder `conditions/` embedding, shared by every sample.**  Captions are
  Gemini-blocked (HTTP 429, prepayment credits depleted) and A9 permits placeholders.  Using
  the *same* pre-encoded Gemma embedding for all arms is stronger than using different
  placeholders: the text condition becomes a constant and therefore cannot contribute any
  per-format difference in loss.  The embedding is an existing certified `ic_gen` conditions
  file; its caption does not describe any of these clips and is never read.
  ⚠ This root is a MECHANICS FIXTURE ONLY.  It must never be trained on for a real run.
* **Masks come from `scripts/ctt_v2/masks/regen_masks.py`** — regenerated per geometry,
  proven bit-identical to `assemble_root.ensure_mask`, never reused across formats.
* **Reference latents are same-stratum, same-group** (A9's "S4 references are S4-native,
  ring within op, so reference and target share the ~2 s span — no cross-span mismatch").
  The builder ASSERTS reference geometry == target geometry per sample.

A geometric fact the gate must be read against (not a defect, a consequence)
---------------------------------------------------------------------------
The prefix anchor is a fixed **2 latent frames** (`assemble_root.ensure_mask`).  At 121f that
is 2/16 = 12.5% of tokens conditioned; at S4's 5 latent frames it is 2/5 = **40%**.  So S4's
loss is computed over 60% of its tokens vs 87.5% for a one-sided 121f sample.  The trainer's
loss is normalised by the mask mean (`flexible.py:_compute_modality_loss`), so this does not
scale the loss, but it does mean the two arms average over different fractions of their
sequence.  Recorded here so the gate's numbers are not over-read.

    python scripts/ctt_v2/smoke/build_smoke_root.py            # CPU, seconds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTT = HERE.parent
REPO_ROOT = HERE.parents[2]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MAIN = LAB / "diffusion-research"

ENC = MAIN / "outputs/ctt_v2/encodes"
MASK_STORE = MAIN / "outputs/ctt_v2/masks/_mask_store"
IC_GEN = MAIN / "eval_ladder/dataset/roots/ic_gen"
OUT = MAIN / "outputs/ctt_v2/smoke/root_mixed"

SELECTION = REPO_ROOT / "data/processed/s4_refvfx/selection.json"
ROOT_DIRS = ("latents", "conditions", "cond_clean_latents", "masks", "reference_latents")

SHIFT_M, SHIFT_B = 1.1 / 3072, 0.95 - (1.1 / 3072) * 1024


def shift_for(tokens: int) -> float:
    return SHIFT_M * tokens + SHIFT_B


def log(m: str) -> None:
    print(f"[smoke-root] {m}", flush=True)


# --------------------------------------------------------------------------------------
def ring_pairs(stems: list[str], max_refs: int = 3) -> list[tuple[str, str]]:
    """Byte-for-byte the rule in `root_common.ring_pairs` (RULING 4)."""
    n = len(stems)
    if n < 2:
        return []
    k = min(max_refs, n - 1)
    return [(t, stems[(i + j) % n]) for i, t in enumerate(stems) for j in range(1, k + 1)]


def s1_groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted((ENC / "S1/latents").glob("*.pt")):
        out.setdefault(p.stem.split("__", 1)[0], []).append(p.stem)
    return out


def s1_sidedness() -> dict[str, str]:
    reg = REPO_ROOT / "eval_ladder/registry.jsonl"
    out: dict[str, str] = {}
    for line in reg.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if isinstance(r.get("arm"), str) and r["arm"].startswith("spec_") and r.get("sided") in ("one", "two"):
            out[r["arm"]] = r["sided"]
    return out


def s2_groups(stratum: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted((ENC / stratum / "latents").glob("*.pt")):
        out.setdefault(p.stem.rsplit("_c", 1)[0], []).append(p.stem)
    return out


def s4_groups() -> dict[str, list[str]]:
    """S4's pairing group is the EFFECT (42 of them) — `selection.json` is the authority."""
    have = {p.stem for p in (ENC / "S4/latents").glob("*.pt")}
    sel = json.loads(SELECTION.read_text())
    out: dict[str, list[str]] = {}
    for r in sel["samples"]:
        if r["k"] in have:
            out.setdefault(r["effect"], []).append(r["k"])
    return {k: sorted(v) for k, v in out.items()}


# --------------------------------------------------------------------------------------
def pick_arms(n_per_arm: int) -> list[dict]:
    """Deterministic, smallest-name-first selection so the fixture is reproducible."""
    arms = []
    sided_map = s1_sidedness()

    g1 = s1_groups()
    one_sided = sorted(a for a in g1 if sided_map.get(a) == "one" and len(g1[a]) >= 2)
    if not one_sided:
        raise SystemExit("no one-sided S1 arm with >= 2 encoded clips")
    arm = one_sided[0]
    arms.append({"arm": "S1_121f_one", "stratum": "S1", "group": arm, "sided": "one",
                 "pairs": ring_pairs(g1[arm])[:n_per_arm]})

    g4 = s4_groups()
    eff = sorted((e for e in g4 if len(g4[e]) >= 2), key=lambda e: (-len(g4[e]), e))[0]
    arms.append({"arm": "S4_33f_one", "stratum": "S4", "group": eff, "sided": "one",
                 "pairs": ring_pairs(g4[eff])[:n_per_arm]})

    g2 = s2_groups("S2a")
    op = sorted(o for o in g2 if len(g2[o]) >= 2)[0]
    arms.append({"arm": "S2a_121f_two", "stratum": "S2a", "group": op, "sided": "two",
                 "pairs": ring_pairs(g2[op])[:max(2, n_per_arm // 2)]})
    return arms


def placeholder_conditions() -> Path:
    """One certified `ic_gen` conditions embedding, reused for every sample (see docstring)."""
    cands = sorted((IC_GEN / "conditions").glob("*/*.pt"))
    if not cands:
        raise SystemExit(f"no conditions embeddings under {IC_GEN/'conditions'}")
    return cands[0]


def build(n_per_arm: int, out: Path) -> dict:
    import torch

    arms = pick_arms(n_per_arm)
    cond = placeholder_conditions()
    log(f"placeholder conditions embedding (SHARED by every sample): "
        f"{cond.relative_to(IC_GEN)}")

    for d in ROOT_DIRS:
        (out / d).mkdir(parents=True, exist_ok=True)

    rows, geoms = [], {}
    for a in arms:
        lat_dir = ENC / a["stratum"] / "latents"
        cc_dir = ENC / a["stratum"] / "cond_clean"
        for tgt, ref in a["pairs"]:
            rel = f"{a['stratum']}_r00/{a['group']}/{tgt}__ref_{ref}.pt"
            tp, rp, cp = lat_dir / f"{tgt}.pt", lat_dir / f"{ref}.pt", cc_dir / f"{tgt}.pt"
            for p in (tp, rp, cp):
                if not p.exists():
                    raise SystemExit(f"missing source {p}")
            dt = torch.load(tp, map_location="cpu", weights_only=True)
            dr = torch.load(rp, map_location="cpu", weights_only=True)
            f, h, w = int(dt["num_frames"]), int(dt["height"]), int(dt["width"])
            rf, rh, rw = int(dr["num_frames"]), int(dr["height"]), int(dr["width"])
            if (f, h, w) != (rf, rh, rw):
                raise SystemExit(f"{rel}: reference geometry ({rf},{rh},{rw}) != target "
                                 f"({f},{h},{w}) — cross-span RoPE mismatch, refusing")
            mask = MASK_STORE / f"f{f}_h{h}_w{w}_{a['sided']}sided.pt"
            if not mask.exists():
                raise SystemExit(f"{rel}: no regenerated mask at {mask} — run "
                                 f"scripts/ctt_v2/masks/regen_masks.py first")
            md = torch.load(mask, map_location="cpu", weights_only=True)["mask"]
            if md.numel() != f * h * w:
                raise SystemExit(f"{rel}: mask numel {md.numel()} != {f*h*w}")

            for sub, src in (("latents", tp), ("reference_latents", rp),
                             ("cond_clean_latents", cp), ("conditions", cond),
                             ("masks", mask)):
                dst = out / sub / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(os.path.realpath(src))

            tok = f * h * w
            geoms.setdefault((f, h, w), 0)
            geoms[(f, h, w)] += 1
            rows.append({
                "rel": rel, "arm": a["arm"], "stratum": a["stratum"], "group": a["group"],
                "target": tgt, "reference": ref, "sided": a["sided"],
                "format": f"{'121f' if f == 16 else f'{(f-1)*8+1}f'}",
                "latent_fhw": [f, h, w], "tokens": tok,
                "expected_shift": shift_for(tok),
                "fps": float(dt["fps"]),
                "mask_cond_fraction": float(md.mean().item()),
                "mask_sha_name": mask.name,
            })
            del dt, dr, md

    # prune anything not in this build
    want = {f"{sub}/{r['rel']}" for sub in ROOT_DIRS for r in rows}
    removed = 0
    for sub in ROOT_DIRS:
        for p in (out / sub).glob("**/*.pt"):
            k = f"{sub}/{p.relative_to(out / sub)}"
            if k not in want:
                p.unlink()
                removed += 1

    shifts = sorted({r["expected_shift"] for r in rows})
    man = {
        "schema": "ctt_v2_smoke_root/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "authority": "A9 §3 item 3 — the mandatory mixed-format smoke gate",
        "root": str(out),
        "WARNING": "MECHANICS FIXTURE ONLY. `conditions/` is ONE placeholder Gemma embedding "
                   "shared by every sample (Gemini captions are credit-blocked; A9 permits "
                   "placeholders for this gate). Never train a real run on this root.",
        "placeholder_conditions_source": str(cond),
        "n_samples": len(rows),
        "arms": {a["arm"]: sum(1 for r in rows if r["arm"] == a["arm"]) for a in arms},
        "geometries": {f"{k[0]},{k[1]},{k[2]}": v for k, v in sorted(geoms.items())},
        "distinct_expected_shifts": shifts,
        "pruned_stale_links": removed,
        "samples": rows,
    }
    (out / "SMOKE_ROOT_MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")

    # hard count assert across the five trees
    counts = {sub: len(list((out / sub).glob("**/*.pt"))) for sub in ROOT_DIRS}
    if len(set(counts.values())) != 1 or counts["latents"] != len(rows):
        raise SystemExit(f"five-tree count mismatch {counts} vs {len(rows)} samples")
    log(f"{len(rows)} samples, five trees at {counts['latents']} each")
    for a in arms:
        n = man["arms"][a["arm"]]
        ex = [r for r in rows if r["arm"] == a["arm"]][0]
        log(f"  {a['arm']:14s} n={n} group={a['group']!r} latent={ex['latent_fhw']} "
            f"tokens={ex['tokens']} fps={ex['fps']} sided={ex['sided']} "
            f"shift={ex['expected_shift']:.6f} mask_cond={ex['mask_cond_fraction']:.4f}")
    log(f"distinct expected shifts: {[round(s, 6) for s in shifts]}")
    log(f"manifest -> {out/'SMOKE_ROOT_MANIFEST.json'}")
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-arm", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    build(args.n_per_arm, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
