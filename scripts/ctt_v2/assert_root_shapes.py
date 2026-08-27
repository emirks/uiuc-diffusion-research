"""CTT v2 — the TWO-SHAPE extension to the root assert battery (A9 §3, last paragraph).

A9: *"the five-tree set-equality assert and the 'Fast index: N of N' gate must count S4's
6,000."*  This module is that extension.

WHY IT IS A SEPARATE MODULE
---------------------------
`assert_root.py` is owned by another agent this session, so nothing here edits it.  The split
is also the right design: `assert_root.py`'s A1 asserts set-equality of relative PATHS, which
is a filesystem property and already shape-agnostic.  Everything a second shape adds is a
property of the TENSOR PAYLOADS, which A1 never opens.  Those are exactly the checks below,
and they are the ones that catch the mixed-format hazards `REF_mixed_length.md` ranks 2 and 3.

    from assert_root_shapes import assert_two_shapes
    rc = assert_two_shapes(root)          # 0 = pass; writes SHAPE_ASSERT_REPORT.json

WHAT A1 CANNOT SEE, AND EACH CHECK THAT DOES
--------------------------------------------
    B1  per-shape five-tree set equality.  A1 proves the five path sets are equal ONCE, over
        the whole root; it cannot notice that the S4 subtree has 6,000 entries in `latents/`
        and 6,000 in `masks/` that are the WRONG 6,000.  B1 re-runs set equality inside each
        shape class, so a cross-shape swap cannot cancel out in the global count.
    B2  per-sample geometry agreement across all five trees.  This is the hazard: a sample
        whose five paths all exist but whose `masks/` entry is the 121f mask.  numel 4,800 vs
        1,820 makes `flexible.py:533`'s reshape raise — LOUDLY, which is the good case — but
        only when that sample is finally drawn, possibly thousands of steps in.  B2 finds it
        before launch.  Also asserts reference geometry == target geometry (A9's RoPE
        decision: S4 references are S4-native, so no cross-span mismatch) and
        cond_clean geometry == target geometry.
    B3  the set of shape classes is EXACTLY the expected one, and each stratum maps to
        exactly one class.  Guards against a third geometry appearing from a re-encode.
    B4  per-stratum and per-shape sample counts, S4's 6,000 included, checked against the
        root manifest's own replica arithmetic.
    B5  the "Fast index: N valid samples from N total" gate, with N = the two-shape total.
        `datasets.py` logs it at DEBUG on the happy path and drops samples SILENTLY, so this
        is the only direct evidence that the trainer saw every sample of both shapes.
    B6  the realized logit-normal shift per shape class equals the analytic value from
        `scripts/ctt_v2/sigma/sigma_schedule.py`, so the archived sigma table and the root on
        disk can never disagree.
    B7  no two shape classes share a token count.  If they did, a cross-shape mask would
        reshape SILENTLY and condition the wrong tokens — the one failure mode that is not
        loud.  (16,20,15)=4,800 and (5,14,26)=1,820 are safely distinct; the check exists so
        a future geometry cannot quietly collide.

🔴 On A9's "(5,20,15)": it is not achievable and it is not a separate mask parameterisation.
`assemble_root.ensure_mask(path, f, h, w, sided)` does `torch.zeros(f, h, w)` where (f,h,w)
comes verbatim from the TARGET LATENT's `(num_frames, height, width)`, and `flexible.py:533`
reshapes that mask to `(B, seq_len)` with `seq_len = F*H*W`.  The mask triple IS the latent
triple; there is no independent mask geometry.  So (5,20,15) can only be read as a claim
about S4's latent grid, and it is wrong for the same reason the 1,500-token figure is wrong:
832x464 is not VAE-legal (464/32 = 14.5), the delivered bucket is 832x448x33, and the real
grid is (5,14,26) = 1,820 tokens (DOSSIER §10.9).

    python scripts/ctt_v2/assert_root_shapes.py --root <root>
    python scripts/ctt_v2/assert_root_shapes.py --self-test      # prove every check FIRES
    python scripts/ctt_v2/assert_root_shapes.py --root <root> --train-log <log>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")

ROOT_DIRS = ("latents", "conditions", "cond_clean_latents", "masks", "reference_latents")
#: the trees whose payload carries a video geometry.  `conditions/` is a (1024,3840) Gemma
#: embedding — fixed for every sample and every format — so it is deliberately NOT geometric.
GEOMETRIC_DIRS = ("latents", "cond_clean_latents", "reference_latents")

#: A9 §4 / DOSSIER §12 / §10.9 — the two shape classes the mix may contain, and nothing else.
#: `prefix_latents` is INDEPENDENTLY RESTATED here, not imported from `root_common`.  That is
#: the point of this checker: if the assembler's shape table drifts, the disagreement is a
#: failure rather than something both sides inherit.  2 at 121f; 1 for S4, whose conditioning
#: is video frame 0 alone (owner decision 2026-07-28).
EXPECTED_SHAPE_CLASSES = {
    (16, 20, 15): {"name": "121f", "px_whf": [480, 640, 121], "fps": 24.0, "tokens": 4800,
                   "strata": ["S0", "S1", "S2a", "S2b"], "prefix_latents": 2},
    (5, 14, 26): {"name": "33f", "px_whf": [832, 448, 33], "fps": 16.0, "tokens": 1820,
                  "strata": ["S4"], "prefix_latents": 1},
    # EffectData (S6): 4 native VAE-legal shapes, 81f/24fps, one-sided frame-0 anchor (prefix 1).
    # Two transpose pairs => two token-count collisions (9438, 7986); B7 is generalized below to
    # allow them because each shape gets a DISTINCT mask file. Mirrors root_common.RULED_SHAPES.
    (11, 22, 39): {"name": "effd_1248x704_81f", "px_whf": [1248, 704, 81], "fps": 24.0, "tokens": 9438,
                   "strata": ["S6"], "prefix_latents": 1},
    (11, 39, 22): {"name": "effd_704x1248_81f", "px_whf": [704, 1248, 81], "fps": 24.0, "tokens": 9438,
                   "strata": ["S6"], "prefix_latents": 1},
    (11, 33, 22): {"name": "effd_704x1056_81f", "px_whf": [704, 1056, 81], "fps": 24.0, "tokens": 7986,
                   "strata": ["S6"], "prefix_latents": 1},
    (11, 22, 33): {"name": "effd_1056x704_81f", "px_whf": [1056, 704, 81], "fps": 24.0, "tokens": 7986,
                   "strata": ["S6"], "prefix_latents": 1},
}


def expected_prefix(f: int, h: int, w: int) -> int:
    return EXPECTED_SHAPE_CLASSES.get((f, h, w), {}).get("prefix_latents", 2)

#: RULING 4 pairing; S4 = 2,000 clips over 42 effects, every group n>=4 => 3n pairs = 6,000
EXPECTED_BASE_PAIRS = {"S4": 6000}

SHIFT_M, SHIFT_B = 1.1 / 3072, 0.95 - (1.1 / 3072) * 1024

_re = __import__("re")
RE_INDEX = _re.compile(
    r"Fast index:\s*(\d+)\s*valid samples from\s*(\d+)\s*total(?:\s*\((\d+)\s*skipped\))?")
#: The trainer logs through `RichHandler`: every number arrives wrapped in SGR colour codes
#: and OSC-8 hyperlinks (`Fast index: \x1b[1;36m10\x1b[0m valid samples from ...`).  Parsing a
#: raw capture matches NOTHING and B5 reports a spurious FAIL on a perfectly healthy run —
#: observed on the first mixed smoke run and fixed here.
RE_SGR = _re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
RE_OSC = _re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def strip_ansi(s: str) -> str:
    return RE_SGR.sub("", RE_OSC.sub("", s))


def shift_for(tokens: int) -> float:
    return SHIFT_M * tokens + SHIFT_B


# --------------------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.results: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "", offenders=None) -> bool:
        offenders = list(offenders or [])
        self.results.append({"name": name, "ok": bool(ok), "detail": detail,
                             "n_offenders": len(offenders), "offenders": offenders[:200]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        for o in offenders[:10]:
            print(f"        - {o}")
        if len(offenders) > 10:
            print(f"        ... and {len(offenders) - 10} more")
        return ok

    @property
    def failed(self) -> list[str]:
        return [r["name"] for r in self.results if not r["ok"]]


def rel_paths(root: Path, sub: str) -> set[str]:
    base = root / sub
    return {str(p.relative_to(base)) for p in base.glob("**/*.pt")} if base.is_dir() else set()


def stratum_of(rel: str) -> str:
    """'S2a_r03/op/a__ref_b.pt' -> 'S2a'.  Same scheme as root_common.parse_replica."""
    first = rel.split("/", 1)[0]
    return first.rsplit("_r", 1)[0] if "_r" in first else first


def read_geo(path: Path) -> tuple[int, int, int, float, tuple]:
    import torch

    d = torch.load(path, map_location="cpu", weights_only=True)
    if "mask" in d and "latents" not in d:
        m = d["mask"]
        return (*tuple(int(x) for x in m.shape), float("nan"), tuple(m.shape))
    lat = d["latents"]
    return (int(d["num_frames"]), int(d["height"]), int(d["width"]),
            float(d.get("fps", float("nan"))), tuple(int(x) for x in lat.shape))


# --------------------------------------------------------------------------------------
def assert_two_shapes(root: Path, train_log: Path | None = None,
                      expected_classes: dict | None = None,
                      report_path: Path | None = None,
                      max_samples: int | None = None) -> int:
    """The two-shape battery.  Returns 0 on pass, 1 on any hard failure."""
    import math

    expected_classes = expected_classes or EXPECTED_SHAPE_CLASSES
    root = Path(root)
    r = Report()
    t0 = time.time()

    sets = {sub: rel_paths(root, sub) for sub in ROOT_DIRS}
    missing_dirs = [s for s in ROOT_DIRS if not (root / s).is_dir()]
    if missing_dirs:
        r.check("B0_five_dirs_exist", False, f"missing root dirs: {missing_dirs}")
        return _finish(r, root, report_path, t0)
    r.check("B0_five_dirs_exist", True, f"all five trees present "
            f"({{{', '.join(f'{k}:{len(v)}' for k, v in sets.items())}}})")

    rels = sorted(sets["latents"])
    if not rels:
        r.check("B0b_nonempty", False, "latents/ is empty")
        return _finish(r, root, report_path, t0)

    # ---- classify every sample by the geometry of its own latents ----------------------
    pool = rels if max_samples is None else rels[:max_samples]
    geo_by_rel: dict[str, tuple] = {}
    per_sample_bad: list[str] = []
    for rel in pool:
        try:
            f, h, w, fps, shape = read_geo(root / "latents" / rel)
        except Exception as exc:  # noqa: BLE001
            per_sample_bad.append(f"{rel}: latents unreadable ({exc!r})")
            continue
        geo_by_rel[rel] = (f, h, w)
        tok = f * h * w
        cls = expected_classes.get((f, h, w))

        # B2 — every geometric tree must agree, and the mask must have exactly tok elements
        for sub in GEOMETRIC_DIRS[1:]:
            if rel not in sets[sub]:
                per_sample_bad.append(f"{rel}: absent from {sub}")
                continue
            try:
                gf, gh, gw, gfps, _ = read_geo(root / sub / rel)
            except Exception as exc:  # noqa: BLE001
                per_sample_bad.append(f"{rel}: {sub} unreadable ({exc!r})")
                continue
            if (gf, gh, gw) != (f, h, w):
                per_sample_bad.append(
                    f"{rel}: {sub} geometry ({gf},{gh},{gw}) != latents ({f},{h},{w})"
                    + ("  [reference/target span mismatch -> RoPE covers a different "
                       "number of seconds]" if sub == "reference_latents" else ""))
            if sub == "reference_latents" and not math.isnan(gfps) and not math.isnan(fps) \
                    and abs(gfps - fps) > 1e-6:
                per_sample_bad.append(f"{rel}: reference fps {gfps} != target fps {fps}")
        if rel in sets["masks"]:
            try:
                mf, mh, mw, _, mshape = read_geo(root / "masks" / rel)
                if (mf, mh, mw) != (f, h, w):
                    per_sample_bad.append(
                        f"{rel}: mask shape {(mf, mh, mw)} != latent ({f},{h},{w}); "
                        f"numel {mf*mh*mw} vs required {tok} -> flexible.py:533 reshape")
            except Exception as exc:  # noqa: BLE001
                per_sample_bad.append(f"{rel}: mask unreadable ({exc!r})")
        else:
            per_sample_bad.append(f"{rel}: absent from masks")
        if cls is not None and not math.isnan(fps) and abs(fps - cls["fps"]) > 1e-6:
            per_sample_bad.append(f"{rel}: fps {fps} != {cls['fps']} for shape class "
                                  f"{cls['name']} — RoPE is scaled to ABSOLUTE SECONDS, so a "
                                  f"wrong fps is a silent training defect")

    r.check("B2_per_sample_geometry_agreement", not per_sample_bad,
            f"all {len(geo_by_rel)} samples: latents == cond_clean == reference geometry, "
            f"mask numel == F*H*W, fps matches the shape class"
            if not per_sample_bad else
            f"{len(per_sample_bad)} per-sample geometry problems", per_sample_bad)

    classes: dict[tuple, list[str]] = {}
    for rel, g in geo_by_rel.items():
        classes.setdefault(g, []).append(rel)

    # ---- B1 per-shape five-tree set equality ------------------------------------------
    b1_bad = []
    for g, members in sorted(classes.items()):
        ms = set(members)
        for sub in ROOT_DIRS:
            if sub == "latents":
                continue
            sym = (ms - sets[sub])
            for x in sorted(sym)[:50]:
                b1_bad.append(f"shape {g}: {x} present in latents but missing from {sub}")
    # and the reverse direction A1 also covers, restated per shape so counts cannot cancel
    extra = set().union(*[sets[s] for s in ROOT_DIRS]) - set(geo_by_rel)
    for x in sorted(extra)[:50]:
        b1_bad.append(f"{x} exists in some tree but NOT in latents — datasets.py never "
                      f"enumerates it at all (REF_mixed_length Gap 3)")
    r.check("B1_per_shape_five_tree_set_equality", not b1_bad,
            "five-tree set equality holds INSIDE every shape class: "
            + ", ".join(f"{expected_classes.get(g, {}).get('name', g)}={len(v)}"
                        for g, v in sorted(classes.items()))
            if not b1_bad else f"{len(b1_bad)} per-shape set-equality violations", b1_bad)

    # ---- B3 exactly the expected shape classes ----------------------------------------
    unexpected = sorted(g for g in classes if g not in expected_classes)
    strat_multi = []
    by_stratum: dict[str, set] = {}
    for rel, g in geo_by_rel.items():
        by_stratum.setdefault(stratum_of(rel), set()).add(g)
    for s, gs in sorted(by_stratum.items()):
        if len(gs) != 1:
            strat_multi.append(f"stratum {s} carries {len(gs)} geometries {sorted(gs)}")
        else:
            g = next(iter(gs))
            allowed = expected_classes.get(g, {}).get("strata")
            if allowed and s not in allowed:
                strat_multi.append(f"stratum {s} has geometry {g} whose shape class allows "
                                   f"only {allowed}")
    r.check("B3_shape_classes_expected", not unexpected and not strat_multi,
            f"{len(classes)} shape class(es), all expected; every stratum maps to exactly one"
            if not unexpected and not strat_multi else
            f"unexpected geometries {unexpected}; {len(strat_multi)} stratum problems",
            [f"unexpected geometry {g} ({len(classes[g])} samples)" for g in unexpected]
            + strat_multi)

    # ---- B4 counts, S4's 6,000 included -----------------------------------------------
    per_stratum = {s: sum(1 for rel in geo_by_rel if stratum_of(rel) == s)
                   for s in sorted(by_stratum)}
    per_class = {expected_classes.get(g, {}).get("name", str(g)): len(v)
                 for g, v in sorted(classes.items())}
    b4_bad, b4_notes = [], []
    man_p = root / "ROOT_MANIFEST.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        mult = (man.get("weights") or {}).get("multipliers") or {}
        for s, base in EXPECTED_BASE_PAIRS.items():
            if s not in per_stratum:
                continue
            m = int(mult.get(s, 0)) or None
            if m is None:
                b4_notes.append(f"{s}: no replica multiplier in ROOT_MANIFEST.json")
                continue
            want = base * m
            if per_stratum[s] != want:
                b4_bad.append(f"{s}: counted {per_stratum[s]} != base {base} x {m} replicas "
                              f"= {want}")
            else:
                b4_notes.append(f"{s}: {base} base pairs x {m} replicas = {want} counted")
    else:
        b4_notes.append("no ROOT_MANIFEST.json — replica arithmetic not checkable "
                        "(fixture mode); base-pair expectations still reported")
        for s, base in EXPECTED_BASE_PAIRS.items():
            if s in per_stratum and per_stratum[s] % base:
                b4_notes.append(f"{s}: counted {per_stratum[s]} is not a whole multiple of "
                                f"the {base} expected base pairs")
    r.check("B4_counts_include_S4_6000", not b4_bad,
            f"per-stratum {per_stratum} | per-shape {per_class}"
            + (" | " + "; ".join(b4_notes) if b4_notes else ""), b4_bad)

    # ---- B6 realized shift per class ---------------------------------------------------
    b6 = []
    shifts = {}
    for g in sorted(classes):
        tok = g[0] * g[1] * g[2]
        sh = shift_for(tok)
        shifts[expected_classes.get(g, {}).get("name", str(g))] = {"tokens": tok, "shift": sh}
        exp_tok = expected_classes.get(g, {}).get("tokens")
        if exp_tok is not None and tok != exp_tok:
            b6.append(f"shape {g}: {tok} tokens != the {exp_tok} the sigma archive assumes")
    r.check("B6_shift_matches_sigma_archive", not b6,
            "; ".join(f"{k}: {v['tokens']} tok -> shift {v['shift']:.6f}"
                      for k, v in shifts.items()), b6)

    # ---- B7 token-count collisions (tolerated; B2 owns the real defect) ------------------
    # Historically B7 forbade two shape classes sharing a token count. EffectData (S6) legitimately
    # breaks that: its native clips come in transpose pairs (1248x704 & 704x1248 -> 9438 tokens;
    # 1056x704 & 704x1056 -> 7986) with DISTINCT masks. A token collision is only dangerous via a
    # WRONG-SHAPE mask, and B2_per_sample_geometry_agreement already asserts, per sample, that the
    # mask's exact (f,h,w) equals the latent's -- so a cross-shape mask fails LOUDLY at B2, never
    # silently. B7 is therefore INFORMATIONAL: it reports collisions but defers the guarantee to B2.
    by_tok: dict[int, list] = {}
    for g in classes:
        by_tok.setdefault(g[0] * g[1] * g[2], []).append(g)
    tok_coll = {t: sorted(v) for t, v in by_tok.items() if len(v) > 1}
    r.check("B7_no_token_count_collision", True,
            (f"token collisions {tok_coll} are tolerated -- each shape keeps its exact mask "
             f"(B2 per-sample), so a cross-shape mask fails LOUDLY at B2" if tok_coll else
             f"token counts {sorted(by_tok)} each map to exactly one geometry"), [])

    # ---- B5 Fast index gate -------------------------------------------------------------
    total = len(geo_by_rel)
    if train_log is None:
        r.check("B5_fast_index_N_of_N", True,
                f"NOT APPLICABLE (no --train-log). The gate to run at launch is: "
                f"'Fast index: {total} valid samples from {total} total', 0 skipped")
    else:
        text = strip_ansi(Path(train_log).read_text(errors="replace")
                          if Path(train_log).exists() else "")
        idx = RE_INDEX.findall(text)
        ok = bool(idx) and all(int(v) == total and int(t) == total and int(s or 0) == 0
                               for v, t, s in idx)
        r.check("B5_fast_index_N_of_N", ok,
                f"{len(idx)} index line(s), every one reads {total} of {total}, 0 skipped "
                f"(both shapes counted)" if ok else
                f"expected {total} of {total} across both shapes; log shows "
                f"{[(v, t, s) for v, t, s in idx] or 'NO index line at all'}")

    return _finish(r, root, report_path, t0, extra={"per_stratum": per_stratum,
                                                    "per_shape": per_class,
                                                    "shifts": shifts})


def _finish(r: Report, root: Path, report_path, t0: float, extra: dict | None = None) -> int:
    rec = {"schema": "ctt_v2_two_shape_assert/1",
           "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "root": str(root),
           "elapsed_s": round(time.time() - t0, 2),
           "expected_shape_classes": {str(k): v for k, v in EXPECTED_SHAPE_CLASSES.items()},
           "n_checks": len(r.results), "failed": r.failed, "results": r.results}
    if extra:
        rec.update(extra)
    out = Path(report_path) if report_path else root / "SHAPE_ASSERT_REPORT.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    except OSError as exc:
        print(f"[shape-assert] WARNING: could not write {out}: {exc}")
    n_fail = len(r.failed)
    print(f"\n[shape-assert] {len(r.results) - n_fail}/{len(r.results)} checks passed in "
          f"{rec['elapsed_s']}s -> {out}")
    if n_fail:
        print(f"[shape-assert] HARD FAILURES: {r.failed}")
        return 1
    print("[shape-assert] TWO-SHAPE ASSERTS PASSED")
    return 0


# --------------------------------------------------------------------------------------
# the deliberately broken fixture — proof that each check FIRES
# --------------------------------------------------------------------------------------
def _mk_latent(f: int, h: int, w: int, fps: float):
    import torch

    return {"latents": torch.zeros(128, f, h, w, dtype=torch.bfloat16),
            "num_frames": f, "height": h, "width": w, "fps": fps}


def _mk_mask(f: int, h: int, w: int, sided: str = "one"):
    import torch

    m = torch.zeros(f, h, w)
    m[:expected_prefix(f, h, w)] = 1.0
    if sided == "two":
        m[-1] = 1.0
    return {"mask": m}


def build_fixture(base: Path, defect: str) -> Path:
    """A minimal two-shape root carrying exactly one named defect (or none)."""
    import shutil

    import torch

    root = base / defect
    if root.exists():
        shutil.rmtree(root)
    G121, G33 = (16, 20, 15), (5, 14, 26)
    plan = [("S1_r00/spec_x", "a__ref_b.pt", G121, 24.0, "one"),
            ("S1_r00/spec_x", "b__ref_a.pt", G121, 24.0, "one"),
            ("S4_r00/eff_y", "c__ref_d.pt", G33, 16.0, "one"),
            ("S4_r00/eff_y", "d__ref_c.pt", G33, 16.0, "one")]
    cond = {"video_prompt_embeds": torch.zeros(1024, 8, dtype=torch.bfloat16),
            "prompt_attention_mask": torch.ones(1024, dtype=torch.long),
            "audio_prompt_embeds": torch.zeros(1024, 8, dtype=torch.bfloat16)}

    for sub in ROOT_DIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    for grp, name, g, fps, sided in plan:
        rel = f"{grp}/{name}"
        f, h, w = g
        tgt = _mk_latent(f, h, w, fps)
        ref = _mk_latent(f, h, w, fps)
        cc = _mk_latent(f, h, w, fps)
        mask = _mk_mask(f, h, w, sided)
        is_s4 = grp.startswith("S4")

        if defect == "mask_wrong_shape" and is_s4 and name.startswith("c"):
            mask = _mk_mask(*G121)                       # 121f mask under a 33f latent
        if defect == "reference_wrong_shape" and is_s4 and name.startswith("c"):
            ref = _mk_latent(*G121, 24.0)                # cross-span reference
        if defect == "cond_clean_wrong_shape" and is_s4 and name.startswith("c"):
            cc = _mk_latent(*G121, 16.0)
        if defect == "wrong_fps" and is_s4 and name.startswith("c"):
            tgt = _mk_latent(f, h, w, 24.0)              # 33f payload claiming 24 fps
        if defect == "third_geometry" and is_s4 and name.startswith("c"):
            tgt, ref, cc = (_mk_latent(5, 20, 15, 16.0) for _ in range(3))
            mask = _mk_mask(5, 20, 15)                   # A9's impossible grid
        if defect == "token_collision" and is_s4 and name.startswith("c"):
            tgt, ref, cc = (_mk_latent(5, 26, 14, 16.0) for _ in range(3))
            mask = _mk_mask(5, 26, 14)                   # 1,820 tokens, DIFFERENT geometry

        for sub, payload in (("latents", tgt), ("reference_latents", ref),
                             ("cond_clean_latents", cc), ("masks", mask),
                             ("conditions", cond)):
            if defect == "missing_mask" and sub == "masks" and is_s4 and name.startswith("c"):
                continue
            if defect == "orphan_in_masks" and sub == "latents" and is_s4 and name.startswith("c"):
                continue                                  # present in masks, absent from latents
            p = root / sub / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, p)

    if defect == "count_short":
        (root / "ROOT_MANIFEST.json").write_text(json.dumps(
            {"weights": {"multipliers": {"S4": 1, "S1": 1}}}) + "\n")
    return root


SELF_TESTS = {
    "clean": [],
    #: B7 is deliberately NOT expected here: it partitions the LATENT shape classes, which a
    #: wrong mask does not touch.  B2 is the check that owns this defect, and it is the one
    #: that must fire.  (First pass over-specified this and the self-test caught it.)
    "mask_wrong_shape": ["B2_per_sample_geometry_agreement"],
    "reference_wrong_shape": ["B2_per_sample_geometry_agreement"],
    "cond_clean_wrong_shape": ["B2_per_sample_geometry_agreement"],
    "wrong_fps": ["B2_per_sample_geometry_agreement"],
    "third_geometry": ["B3_shape_classes_expected"],
    #: token collisions are TOLERATED now (B7 informational; B2 owns wrong-mask). The fixture's
    #: transpose geometry (5,26,14) is unruled, so B3 catches it as an unexpected shape instead.
    "token_collision": ["B3_shape_classes_expected"],
    "missing_mask": ["B1_per_shape_five_tree_set_equality",
                     "B2_per_sample_geometry_agreement"],
    "orphan_in_masks": ["B1_per_shape_five_tree_set_equality"],
    "count_short": ["B4_counts_include_S4_6000"],
}


def self_test(base: Path) -> int:
    """Prove the extension FIRES: each fixture must fail AT LEAST its named checks."""
    base.mkdir(parents=True, exist_ok=True)
    rows, bad = [], []
    for defect, must_fail in SELF_TESTS.items():
        root = build_fixture(base, defect)
        print(f"\n{'=' * 78}\n[self-test] fixture {defect!r}\n{'=' * 78}")
        rc = assert_two_shapes(root, report_path=base / f"report_{defect}.json")
        rep = json.loads((base / f"report_{defect}.json").read_text())
        failed = set(rep["failed"])
        if defect == "clean":
            ok = rc == 0 and not failed
            if not ok:
                bad.append(f"clean fixture must PASS but failed {sorted(failed)}")
        else:
            missing = [c for c in must_fail if c not in failed]
            ok = rc == 1 and not missing
            if rc != 1:
                bad.append(f"{defect}: battery returned {rc}, expected 1")
            if missing:
                bad.append(f"{defect}: expected these checks to FIRE but they passed: {missing}")
        rows.append({"defect": defect, "exit": rc, "failed": sorted(failed),
                     "expected_to_fail": must_fail, "ok": ok})
        print(f"[self-test] {defect}: exit={rc} failed={sorted(failed)} -> "
              f"{'OK' if ok else 'UNPROVEN'}")

    rec = {"schema": "ctt_v2_two_shape_selftest/1",
           "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "fixtures": rows, "problems": bad, "ok": not bad}
    (base / "SELFTEST.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"\n{'=' * 78}")
    for row in rows:
        print(f"  {'OK ' if row['ok'] else 'BAD'} {row['defect']:24s} exit={row['exit']} "
              f"fired={row['failed']}")
    print(f"[self-test] artefact -> {base/'SELFTEST.json'}")
    if bad:
        print(f"[self-test] {len(bad)} PROBLEMS:")
        for b in bad:
            print(f"   - {b}")
        return 1
    print("[self-test] every deliberately broken two-shape fixture was CAUGHT, and the "
          "clean one passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--train-log", help="a trainer stdout log, for the B5 Fast-index gate")
    ap.add_argument("--report")
    ap.add_argument("--max-samples", type=int,
                    help="read only the first N samples (smoke use; NOT for a launch gate)")
    ap.add_argument("--self-test", action="store_true",
                    help="build deliberately broken two-shape fixtures and prove each "
                         "check fires")
    ap.add_argument("--fixture-dir",
                    default=str(LAB / "misc/ctt_v2_final/artefacts/two_shape_fixtures"))
    a = ap.parse_args()

    if a.self_test:
        return self_test(Path(a.fixture_dir))
    if not a.root:
        ap.error("--root is required unless --self-test")
    return assert_two_shapes(Path(a.root),
                             train_log=Path(a.train_log) if a.train_log else None,
                             report_path=Path(a.report) if a.report else None,
                             max_samples=a.max_samples)


if __name__ == "__main__":
    sys.exit(main())
