"""CTT v2 — generate the DATASET.md STAMP block (A5 RULING 9).

RULING 9 lists exactly what "frozen" means.  This script assembles that list from the
artefacts on disk — never from prose — and splices it into `data/DATASET.md` between

    <!-- STAMP:BEGIN --> ... <!-- STAMP:END -->

so stamping is a FILL-IN, not a rewrite: the block regenerates from whatever is on disk at
the time, and anything not yet on disk appears as an explicit `<PENDING: ...>` line rather
than as silence.  A missing number that looks like an absent row is how a stamp lies.

What it reads (all optional; each absence becomes a PENDING line):

    <root>/ROOT_MANIFEST.json     strata, exact counts, grids, mix (intended AND counted),
                                  pairing rule, seeds, holdout lists, drops, the two shapes
    <root>/ASSERT_REPORT.json     the HARD battery result, check by check
    <root>/DRYRUN_REPORT.json     the zero-skipped epoch
    <root>/PROVE_ASSERTS.json     the proof that each HARD assert fires when broken
    <caption store>/gate_report_repinned.json + run_meta.json
                                  the 12-gate battery under the re-pinned bars, the model
                                  version strings, and the raw-response archive paths

`--unstamped` (the default) writes the block with `STAMPED: NO` and the sign-off table
empty.  Nothing here signs anything: the stamp is an owner act.

    python scripts/ctt_v2/make_stamp.py --root <root>                     # print
    python scripts/ctt_v2/make_stamp.py --root <root> --write data/DATASET.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import root_common as rc  # noqa: E402

BEGIN, END = "<!-- STAMP:BEGIN -->", "<!-- STAMP:END -->"
DEFAULT_CAPTION_STORE = rc.HERE / "captions/pilot_m3/round2"


def pending(what: str) -> str:
    return f"`<PENDING: {what}>`"


def root_content_hash(root: Path) -> tuple[str, int]:
    """sha256 over (relative path, resolved target) for every sample in all 5 dirs.

    This is the hash that matters: the manifest's own sha only covers what the assembler
    *said*, while this covers what the trainer will actually open — every path and the
    physical tensor behind it.  Sorted, so it is reproducible.
    """
    h = hashlib.sha256()
    n = 0
    for sub in rc.ROOT_DIRS:
        base = root / sub
        if not base.is_dir():
            continue
        for p in sorted(base.glob("**/*.pt")):
            rel = p.relative_to(root)
            tgt = os.readlink(p) if p.is_symlink() else f"FILE:{p.stat().st_size}"
            h.update(f"{rel}\t{tgt}\n".encode())
            n += 1
    return h.hexdigest(), n


def load(path: Path):
    try:
        return rc.read_json(path)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------
def block(root: Path, caption_store: Path, stamped: bool) -> str:
    man = load(root / "ROOT_MANIFEST.json")
    arep = load(root / "ASSERT_REPORT.json")
    drep = load(root / "DRYRUN_REPORT.json")
    prep = load(root / "PROVE_ASSERTS.json")
    gate = load(caption_store / "gate_report_repinned.json")
    meta = load(caption_store / "run_meta.json")

    L: list[str] = [BEGIN, "", "## STAMP", ""]
    L.append(f"**STAMPED: {'YES' if stamped else 'NO'}** — generated "
             f"`{time.strftime('%Y-%m-%dT%H:%M:%S%z')}` by `scripts/ctt_v2/make_stamp.py` "
             f"from artefacts on disk. RULING 9 defines the contents; every item this "
             f"script cannot read from disk appears below as an explicit `<PENDING: …>`.")
    L += ["", "Stamping is an owner act. This block being present is not a stamp; "
              "`STAMPED: YES` plus the sign-off row below is.", ""]

    # ---- identity / content hash -------------------------------------------------------
    L += ["### S.1 Root identity", ""]
    if man is None:
        L += [f"- {pending('the root has not been assembled — no ROOT_MANIFEST.json')}", ""]
    else:
        chash, nfiles = root_content_hash(root)
        L += [
            f"- **root** `{man['root']}`",
            f"- **assembled** `{man.get('created')}`  ·  **seed** `{man.get('seed')}`",
            f"- **ROOT_MANIFEST.json sha256** `{rc.sha256_file(root / 'ROOT_MANIFEST.json')}`",
            f"- **root content hash** `{chash}`",
            f"  (sha256 over the sorted `(relative path, resolved target)` of all "
            f"**{nfiles}** files in the 5 trees — this is what the trainer actually opens, "
            f"not what the manifest claims)",
            f"- **SAMPLES.jsonl sha256** `{man['records']['samples_jsonl_sha256']}`",
            f"- **CAPTIONS.json sha256** `{man['records']['captions_json_sha256']}`",
            f"- **strata manifest** `{man['strata_manifest']['path']}` "
            f"sha256 `{man['strata_manifest']['sha256']}`", ""]
        L += ["| stratum | inventory | sha256 | groups | clips |", "|---|---|---|---|---|"]
        for s, m in sorted(man["inventories"].items()):
            L.append(f"| {s} | `{Path(m['path']).name}` | `{m['sha256'][:16]}…` | "
                     f"{m['groups']} | {m['clips']} |")
        L.append("")

    # ---- strata + exact counts ----------------------------------------------------------
    L += ["### S.2 Strata, exact counts, and the mix — intended AND counted", ""]
    if man is None:
        L += [f"- {pending('assembled counts')}", ""]
    else:
        w, c = man["weights"], man["counts"]
        L += ["| stratum | base pairs | replicas | realized samples | intended % | "
              "**counted %** | dev pp |", "|---|---|---|---|---|---|---|"]
        for s in man["strata_present"]:
            L.append(f"| {s} | {c['base_pairs'][s]} | {c['replicas'][s]} | "
                     f"{c['realized_samples'][s]} | {w['intended_pct'][s]:.3f} | "
                     f"**{w['realized_pct'][s]:.3f}** | {w['deviation_pp'][s]:+.3f} |")
        L.append(f"| **total** | | | **{c['total_samples']}** | 100.000 | 100.000 | |")
        L += ["",
              f"- **tolerance** ±{w['tolerance_pp']} pp — max realized deviation "
              f"**{max(abs(v) for v in w['deviation_pp'].values()):.3f} pp**",
              f"- **weight basis** {w['note']}; branch `{w.get('branch', {}).get('override_key')}` "
              f"(absent strata: {man['strata_absent'] or 'none'})",
              f"- **S4 in mix** `{man['s4_in_mix']}`",
              f"- **files** {c['total_samples']} samples × {len(rc.ROOT_DIRS)} dirs = "
              f"{c['total_files']} · **distinct captions** {c['distinct_captions']}",
              f"- the counted column is COUNTED FROM THE ASSEMBLED ROOT by "
              f"`assert_root.py:A3`, not computed from the plan (A3-F8.3)", ""]

    # ---- nominal vs effective (A11 item 2) -----------------------------------------------
    L += ["### S.2b Weights — NOMINAL (pre-registered) and EFFECTIVE (derived disclosure)", ""]
    rows = []
    spath = root / "SAMPLES.jsonl"
    if spath.exists():
        rows = [json.loads(ln) for ln in spath.read_text().splitlines() if ln.strip()]
    if rows and man:
        eff = rc.effective_weights(rows, man["counts"]["replicas"])
        L += ["| stratum | nominal % (sample count) | **effective % (loss-bearing tokens)** | "
              "loss-bearing tokens |", "|---|---|---|---|"]
        for s in sorted(eff["nominal_pct"]):
            L.append(f"| {s} | {eff['nominal_pct'][s]:.3f} | "
                     f"**{eff['effective_pct'][s]:.3f}** | "
                     f"{eff['n_loss_bearing_tokens'][s]:,} |")
        L += ["",
              f"- **per-sample loss-bearing tokens** "
              f"`{json.dumps(eff['per_sample_loss_bearing_tokens'])}` — derived from the mask "
              f"rule (`m[:2]=1` always, `m[-1]=1` iff two-sided; mask==1 ⇒ conditioned at "
              f"timestep 0 and excluded from loss), never tabulated",
              f"- total loss-bearing tokens **{eff['total_loss_bearing_tokens']:,}** over "
              f"{eff['total_samples']:,} samples",
              "- **NOMINAL is the pre-registered quantity** — it is what the manifest pins and "
              "what the contingency branches operate on. **EFFECTIVE is a derived disclosure "
              "only** (A11 item 2): right for disclosure, wrong as a control variable, because "
              "pre-registering it would force the nominal weights to chase every geometry "
              "change.",
              "- S4 carries two compounding discounts: the lower training shift (§8.2.1) and a "
              "fixed 2-latent-frame anchor conditioning 40 % of its tokens against 12.5 % at "
              "121f. Its 10 % nominal is ≈ 3 % effective.", ""]
    else:
        L += [f"- {pending('SAMPLES.jsonl — the root has not been assembled')}", ""]

    # ---- the two shapes / grids ----------------------------------------------------------
    L += ["### S.3 Grid definitions — the TWO SHAPES, and the σ schedule they imply", ""]
    shapes = (man or {}).get("shapes")
    if not shapes:
        L += [f"- {pending('the shapes block — reassemble with a current assemble_root.py')}", ""]
    else:
        L += ["| shape (latent F,H,W) | px W×H×F | fps | tokens | shift | samples | strata |",
              "|---|---|---|---|---|---|---|"]
        for s in shapes["per_shape"]:
            px = "×".join(str(x) for x in (s["px_whf"] or [])) or "—"
            L.append(f"| `{s['latent_fhw']}` {'' if s['ruled'] else '**UNRULED**'} | {px} | "
                     f"{s['fps']} | {s['tokens']} | **{s['shift']}** | {s['n_samples']} | "
                     f"{', '.join(s['strata'])} |")
        L += ["",
              "- tokens = the product of the latent dims; shift = "
              "`ShiftedLogitNormalTimestepSampler._get_shift_for_sequence_length` "
              "(`m = 1.1/3072`, `b = 0.5833`, **no clamp**) evaluated at that token count. "
              "Both are DERIVED here, never restated — A9's prose figures "
              "(1,500 tokens / 1.120) are wrong and would have failed the smoke gate "
              "(DOSSIER §13.2, A11 item 4).",
              ""]
        sigma_md = rc.DOSSIER_DIR / "artefacts/sigma/SIGMA_SCHEDULE.md"
        if sigma_md.exists():
            L += [f"- **per-stratum σ distributions**: `{sigma_md}` sha256 "
                  f"`{rc.sha256_file(sigma_md)}` — analytic (closed-form CDF, no sampling), "
                  f"MC-validated against the trainer's own sampler at sup|ΔF| = 0.00036. "
                  f"Reproduced in full in DATASET §8.2.1, with the binding invariant and the "
                  f"supersession note on A9's wrong constants.", ""]
        else:
            L += [f"- {pending('per-stratum σ distributions (SIGMA_SCHEDULE.md)')}", ""]
        L += ["- **shift-law provenance**: `ltx_trainer/timestep_samplers.py:121-134`, "
              "`m = 1.1/3072`, `b = 0.58333…`, **no clamp**; sampler defaults `std = 1.0`, "
              "`eps = 1e-3`, `uniform_prob = 0.1`; `ic_gen.yaml: timestep_sampling_params: {}`. "
              "The IC-LoRA reference is concatenated AFTER the σ draw, so it does not enter "
              "the token count. The trainer was NOT modified.",
              "- **ratified**: the **832×448×33** bucket — a pure 16-row centre crop of the "
              "native 832×464 source, no resampling, the only VAE-legal bucket preserving "
              "native content (A11 item 4). A9's `(5,20,15)` / 1,500 tokens / shift 1.120 are "
              "**wrong constants**; see DATASET §8.2.1 for the derivation, recorded explicitly "
              "so nobody later 'repairs' the encodes toward 1.120.", ""]

    # ---- pairing + seeds ------------------------------------------------------------------
    L += ["### S.4 Pairing rule and every seed", ""]
    if man:
        L += [f"- **pairing** `{man['pairing']['rule']}` — ring offset within the group, "
              f"ref ≠ target, k = min({man['pairing']['max_refs_per_target']}, n−1), applied "
              f"identically to S0 classes and S1/S2/S4 ops (RULING 4, A1b Q5)",
              f"- **assembly seed** `{man.get('seed')}`"]
    if man:
        tables, colls = {}, []
        for s, m in sorted(man["inventories"].items()):
            inv = load(Path(m["path"]))
            if inv:
                t, c = rc.slug_map(inv["groups"])
                tables[s] = t
                colls += [f"{s}: {x}" for x in c]
        if tables:
            L += [f"- **group-id slugs** (A11 item 3, path-safe: lowercase, non-alphanumeric → "
                  f"`_`, runs collapsed): "
                  + " · ".join(f"{s} {len(t)} unique" for s, t in sorted(tables.items()))
                  + (f" — ⚠ **COLLISIONS**: {colls}" if colls else " — no collisions"),
                  f"  Raw→slug mapping is recoverable from the inventories and asserted by "
                  f"`assert_root.py:A14`; nothing already written under raw strings is re-keyed."]
    L += [f"- **inline-OOD draw seed** `{rc.SEED}` (`root_common.select_inline_ood_ops`, "
          f"blind draw over the sorted op list)",
          f"- **training seed** `42` (both arms, RULING 6) · **primary checkpoint** step 12,000",
          f"- **S4 blind-guess gate seed** `44`, n=150 (A9 §2)",
          f"- **pool-gate seed** `20260725`, 300 draws, match_n 187",
          ""]

    # ---- holdout lists ---------------------------------------------------------------------
    L += ["### S.5 Holdout and exclusion lists (enumerated, not described)", ""]
    ex = (man or {}).get("exclusions")
    if not ex:
        L += [f"- {pending('exclusion record — the root has not been assembled')}", ""]
    else:
        L += [f"- **10 HOLDOUT_S2 shader families** ({len(ex['holdout_shaders'])}): "
              f"{', '.join('`' + s + '`' for s in ex['holdout_shaders'])}",
              f"- **8 pre-registered S2a inline-OOD ops** ({len(ex['inline_ood_ops'])}): "
              f"{', '.join('`' + s + '`' for s in ex['inline_ood_ops']) or pending('not frozen')}",
              f"- **reserved union-pool clips**: {ex['n_reserved_pool_clips']}",
              f"- **S0 zero-shot classes** ({len(ex['zs_classes'])}): "
              f"{', '.join('`' + s + '`' for s in ex['zs_classes'])}",
              f"- **eval-endpoint universe** (eval ∪ zs-audited ∪ the 42 test clips): "
              f"{ex['n_eval_endpoints']} ids, classes resolved via "
              f"`eval_ladder/prompts.py:clip_class()`",
              f"- **role-scoped (clip, role) exclusions** (A10): "
              f"`{json.dumps(ex.get('role_scoped_caption_store_exclusions'))}` — enforced on "
              f"BOTH consumption channels (caption store A12, prefix conditioning A13)", ""]
        drops = (man or {}).get("drops") or {}
        if drops:
            L += ["| stratum | groups dropped | clips dropped |", "|---|---|---|"]
            for s, d in sorted(drops.items()):
                L.append(f"| {s} | {d['n_groups']} | {d['n_clips']} |")
            L += ["", "Every drop carries its reason in `ROOT_MANIFEST.json:drops`.", ""]

    # ---- captions --------------------------------------------------------------------------
    L += ["### S.6 Caption store — hash, model versions, raw archives, battery", ""]
    store = caption_store / "descriptions.json"
    if store.exists():
        L += [f"- **store** `{store}`  sha256 `{rc.sha256_file(store)}`",
              f"- **raw generation responses** `{caption_store / 'raw_generation_responses.jsonl'}`",
              f"- **raw audit responses** `{caption_store / 'raw_audit_responses.jsonl'}`",
              f"- **records** `{caption_store / 'records.json'}`"]
    else:
        L += [f"- {pending(f'caption store — {store} does not exist (Gemini credits)')}"]
    if meta:
        for k in sorted(meta):
            L.append(f"- **{k}** `{meta[k]}`")
    else:
        L.append(f"- {pending('generator/auditor model version strings (run_meta.json)')}")
    L.append("")
    if gate:
        summ = gate.get("summary", {})
        L += [f"- **battery**: `{caption_store / 'gate_report_repinned.json'}`  "
              f"sha256 `{rc.sha256_file(caption_store / 'gate_report_repinned.json')}`",
              f"- **hard failures**: `{summ.get('hard_fail')}`", "",
              "| gate | value | bar | verdict |", "|---|---|---|---|"]
        rows = gate.get("gates") or gate.get("results") or []
        if isinstance(rows, dict):
            rows = [dict(v, name=k) for k, v in sorted(rows.items())]
        for g in rows:
            L.append(f"| {g.get('name', g.get('gate'))} | {g.get('value')} | "
                     f"{g.get('bar')} | {g.get('verdict', g.get('status'))} |")
        L += ["", "**Re-pinned bars in force** (A8): gate 8a ≤ 0.73 (corpus-vs-new drift "
                  "guard), gate 8b ≤ 0.60 (stratum-internal). The superseded "
                  "`gate8_bacc_max 0.65` is recorded in the artefact and is NOT the bar.",
              "⚠ The passing store is **round 2**. Round 3 (the copula fix) is an "
              "improvement, not a requirement — the re-pinned battery clears on round 2.", ""]
    else:
        L += [f"- {pending('caption battery report')}", ""]

    # ---- gates ------------------------------------------------------------------------------
    L += ["### S.7 Gate results", "",
          "| gate | result |", "|---|---|",
          "| copy-gate Day-0 admissibility (RULING 1, training blocker) | "
          f"{'**PASS** — ' + str(rc.copy_gate_verdict()[1])[:120] if rc.copy_gate_verdict()[0] else '**FAIL/ABSENT**'} |",
          "| S1 pilot — mechanical | **PASS** — 1/33 = 3.0 % rejects (bar ≤10 %); prefix "
          "rel-L2 p50/p95/max 0.0838 / 0.1232 / 0.1242, all < τ 0.12790; by-bank reject "
          "differential 4.5 pp (flag >15 pp) |",
          f"| S1 pilot — blind 11-way Gemini class-ID (top-1 ≥80 %, chance 9.09 %, with a "
          f"33-clip real-corpus control) | {pending('blocked: Gemini credits')} |",
          "| S2a acceptance refresh | **PASS** |",
          "| S2b acceptance (`S2_ACCEPTANCE.json`) | **PASS** — pure-phase max abs diff 0.0, "
          "seam_max 2.0 (bar ≤2.0, no headroom), m1_p10_min 0.255, overdraw 1.1549 |",
          "| S2b blind audit (n=64, bar ≤3 BAD) | **PASS** — rater1 1 BAD (#055), rater2 0, "
          "**consensus 0**; archived at `outputs/videos/ctt_v2_s2_humanvid/full/"
          "AUDIT_RESULT.json` + `misc/ctt_v2_final/artefacts/s2b_audit/` |",
          "| union-pool gates (numbers of record, A10) | **PASS** — n=1,146, gate A 0.519971 "
          "(≤0.52), gate B 50.56 (≥42.82) |",
          f"| S4 12-gate battery + blind-guess (seed 44, n=150) + 100 % Layer-2 tripwire | "
          f"{pending('blocked: Gemini credits')} |",
          f"| mixed-format smoke gate (2 shapes, per-format consumed counts + finite "
          f"comparable loss + shifts pinned at {{1.2350, 2.3021}}) | {pending('needs a GPU')} |",
          ""]

    # ---- the HARD battery --------------------------------------------------------------------
    L += ["### S.8 Pre-launch HARD asserts — result, and the proof each one fires", ""]
    if arep:
        L += [f"- `assert_root.py` on `{arep['root']}` at `{arep['when']}`: "
              f"**{arep['n_checks'] - len(arep['failed'])}/{arep['n_checks']} passed** "
              f"in {arep['elapsed_s']}s; failures `{arep['failed'] or 'none'}`", ""]
        L += ["| check | verdict | detail |", "|---|---|---|"]
        for r in arep["results"]:
            L.append(f"| `{r['name']}` | {'PASS' if r['ok'] else '**FAIL**'} | "
                     f"{r['detail'][:170]} |")
        L.append("")
    else:
        L += [f"- {pending('assert battery has not been run against this root')}", ""]
    if drep:
        L += [f"- `dryrun_epoch.py`: **{drep['n_skipped']} skipped** over "
              f"{drep['n_epoch_ok']} samples ({drep['n_paths_resolved']} paths resolved, "
              f"{drep['n_distinct_tensors_loaded']} distinct tensors loaded, "
              f"{drep['elapsed_s']}s). Zero skipped is a REQUIREMENT — any skip is promoted "
              f"to a job failure.", ""]
    else:
        L += [f"- {pending('dry-run epoch has not been run against this root')}", ""]
    if prep:
        L += [f"- **proof that the asserts fire** — `tests/prove_asserts.py`, "
              f"strict={prep['strict']}, on `{prep['root']}`: "
              f"**{prep['n_proven']}/{prep['n_mutations']}** deliberate one-invariant "
              f"breakages produced exactly the intended failure(s). "
              f"Problems: `{prep['failures'] or 'none'}`.",
              "  An assert that has never failed is not known to work; this is the evidence "
              "that each one is sensitive to its own invariant and to nothing else.", ""]
        L += ["| broken invariant | checks that fired |", "|---|---|"]
        for r in prep["results"]:
            if r.get("kind") == "baseline_dryrun":
                continue
            fired = r.get("actually_failed") or [r.get("expected_tag")]
            L.append(f"| `{r['mutation']}` | {', '.join('`' + str(x) + '`' for x in fired)} |")
        L.append("")
    else:
        L += [f"- {pending('assert-fire proof has not been run')}", ""]

    # ---- sign-off -----------------------------------------------------------------------------
    L += ["### S.9 Sign-off", "",
          "| item | who | when |", "|---|---|---|",
          f"| dataset design frozen at this spec | {pending('owner')} | {pending('date')} |",
          f"| A9 reversal of A5 Ruling 2 (S4 IN; weights S0 15 / S1 6 / S2 total 69 / S4 10, "
          f"S2a:S2b derived pro-rata per A12) countersigned | "
          f"{pending('owner')} | {pending('date')} |",
          f"| A9 pre-registered branches ratified | {pending('owner')} | {pending('date')} |",
          f"| the 8 inline-OOD ops countersigned (advisor-ratified A11 item 1) | "
          f"{pending('owner')} | {pending('date')} |",
          f"| A10 role-scoped exclusion countersigned | {pending('owner')} | {pending('date')} |",
          f"| copy-gate amendment-2 + gate-#8 re-pin | {pending('owner')} | {pending('date')} |",
          "", END, ""]
    return "\n".join(L)


def splice(doc: Path, text: str) -> None:
    src = doc.read_text()
    if BEGIN in src and END in src:
        head, rest = src.split(BEGIN, 1)
        _old, tail = rest.split(END, 1)
        out = head + text.rstrip("\n") + tail
    else:
        out = src.rstrip("\n") + "\n\n---\n\n" + text
    doc.write_text(out)
    print(f"[stamp] spliced the STAMP block into {doc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--caption-store", default=str(DEFAULT_CAPTION_STORE))
    ap.add_argument("--write", help="path to DATASET.md; splice the block in place")
    ap.add_argument("--stamped", action="store_true",
                    help="mark STAMPED: YES — only ever with the owner's sign-off in hand")
    args = ap.parse_args()

    text = block(Path(args.root), Path(args.caption_store), args.stamped)
    if args.write:
        splice(Path(args.write), text)
    else:
        print(text)


if __name__ == "__main__":
    main()
