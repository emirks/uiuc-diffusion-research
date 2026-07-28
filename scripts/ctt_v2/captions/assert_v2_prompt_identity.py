#!/usr/bin/env python
"""Byte-identity + config-match assert for the pinned ROUND-2 (v2) production prompt.

Discharges two A14 obligations (`advisors/A14_RECONCILIATION_VERBATIM.md`):

  Q1  "Byte-identity assert on the pinned round-2 prompt at mass-run time."
      Production is v2.  If the v2 prompt text drifts by one byte between the run that
      produced the REUSED round-2 descriptions and the run that produces the remainder,
      the store silently becomes a two-prompt store -- which is exactly the bug class
      gate 8a's bar text names ("mixed prompts") and which A13_unpacked_and_round2's
      "never mix v2 and v3 text in one store" rule forbids.  A gate that can only detect
      the mixture *statistically, after the fact* is not a substitute for refusing to
      create it.

  "WHAT WOULD OVERTURN THIS"
      "Evidence the 199 were generated under a different pipeline config than the mass
      run will use (model version, temperature, length sampler) => drop reuse, regenerate
      all 1,348 under v2.  Verify the config match from `run_meta.json` before step 4."

Both checks are structural, not eyeballed:

  * PROMPT.  The v2 system prompt is RENDERED for both roles across the whole reachable
    length domain (the A4 draw is clamped to [15, 45]) and hashed.  Rendering rather than
    hashing the source is deliberate: the source legitimately changed after round 2 (the
    v3 branch was added, then the auditor pin, then the archive-N fix), and none of those
    edits may alter a single byte of what v2 sends.  Only the rendered text is the
    contract.  The reference hash is recomputed from the round-2-era git blob when git is
    available, so the pin cannot be "verified" against itself.

  * CONFIG.  The reused store's `run_meta.json` is compared field-by-field against the
    mass-run config on the GENERATION-side keys A14 names.  The auditor keys are
    deliberately NOT compared: the auditor differing is the known, ruled-on fact that
    A14 step 2's re-audit exists to cure, and folding it in here would make this assert
    fire on the very condition the campaign has already dispositioned.

Exit 0 = identical.  Exit 3 = drift (that is the overturning condition; report the diff).

    PY=$LAB/envs/diffusion/bin/python
    $PY assert_v2_prompt_identity.py \
        --reused outputs/ctt_v2/captions/pilot_m3_round2/run_meta.json \
        --out outputs/ctt_v2/captions/V2_PROMPT_IDENTITY.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

#: The commit that produced the round-2 store (`ctt_v2(captions): M3 caption pilot
#: dry-run -- pipeline + 2x400 descriptions`).  Used to RE-DERIVE the reference hash so
#: this assert is a comparison against history, not against a hard-coded number.
ROUND2_COMMIT = "407adbd"

#: Fallback reference, used only when git is unavailable (e.g. an exported tree).  It was
#: derived from ROUND2_COMMIT, and the git path re-derives and cross-checks it.
ROUND2_V2_PROMPT_SHA256 = (
    "7ac28612a27f81cfad9d53d959300602876eae1f9de237b7280a0bd11ae2ffef"
)

#: A4 Q1 clamps the length draw to [15, 45]; every reachable prompt is enumerated.
N_DOMAIN = range(15, 46)
ROLES = ("A", "B")

#: GENERATION-side config that must match for the reuse to stand (A14's overturning
#: condition names model version, temperature, and the length sampler).  Round-2's
#: run_meta spells the generator thinking level `thinking_level`; later runs spell it
#: `gen_thinking_level`; both names are accepted for the same key so a pure RENAME is
#: never mistaken for a config change.
GEN_KEYS = {
    "generator_model": ("generator_model",),
    "prompt_variant": ("prompt_variant",),
    "gen_temperature": ("gen_temperature",),
    "gen_max_output_tokens": ("gen_max_output_tokens",),
    "gen_thinking_level": ("gen_thinking_level", "thinking_level"),
    "seed": ("seed",),
}


def _get(meta: dict, names: tuple) -> object:
    for n in names:
        if n in meta:
            return meta[n]
    return "<<ABSENT>>"


def render_hash(build_system_prompt) -> tuple[str, int]:
    """Hash every reachable v2 prompt: both roles x the whole clamped length domain."""
    blob = "\n\x00\n".join(build_system_prompt(role, n, "v2")
                           for role in ROLES for n in N_DOMAIN)
    return hashlib.sha256(blob.encode()).hexdigest(), len(blob)


def _v2_namespace_from_source(src: str) -> dict:
    """Exec ONLY the v2-relevant definitions out of a `generate_descriptions.py` source.

    Importing the historical module is not an option (its imports and pins have moved),
    and re-typing the prompt here would defeat the purpose.  Extracting the definitions
    verbatim keeps the reference genuinely derived from the archived source.
    """
    pats = [
        r"_PROMPT_A_TEMPLATE = \(.*?\n\)\n", r"_PROMPT_B_TEMPLATE = \(.*?\n\)\n",
        r"CORPUS_EXEMPLAR_A = \(.*?\n\)\n", r"CORPUS_EXEMPLAR_B = \(.*?\n\)\n",
        r"_V2_STYLE_A = \(.*?\n\)\n", r"_V2_STYLE_B = \(.*?\n\)\n",
        r"_LEN_FIT_A, _LEN_FIT_B = .*?\n",
        r"_A4_VERB_CLAUSE = \(.*?\n\)\n", r"_V3_VERB_CLAUSE = \(.*?\n\)\n",
        r"def calibrate_ask.*?\n\n", r"def build_system_prompt.*?\n    return tmpl\.format.*?\n",
    ]
    code = "".join(m.group(0) for p in pats for m in [re.search(p, src, re.S)] if m)
    ns: dict = {}
    exec(compile(code, "<round2-era generate_descriptions>", "exec"), ns)  # noqa: S102
    return ns


def reference_hash() -> dict:
    """Re-derive the round-2 prompt hash from the round-2-era git blob."""
    try:
        src = subprocess.run(
            ["git", "show", f"{ROUND2_COMMIT}:scripts/ctt_v2/captions/generate_descriptions.py"],
            cwd=HERE, capture_output=True, text=True, check=True, timeout=60).stdout
    except Exception as e:                                   # pragma: no cover
        return {"source": "PINNED_CONSTANT_fallback", "commit": ROUND2_COMMIT,
                "sha256": ROUND2_V2_PROMPT_SHA256, "git_error": f"{type(e).__name__}: {e}"}
    ns = _v2_namespace_from_source(src)
    bsp = ns["build_system_prompt"]
    try:
        sha, n = render_hash(bsp)
    except TypeError:                                        # pre-variant signature
        blob = "\n\x00\n".join(bsp(role, n) for role in ROLES for n in N_DOMAIN)
        sha, n = hashlib.sha256(blob.encode()).hexdigest(), len(blob)
    return {"source": f"git blob at {ROUND2_COMMIT}", "commit": ROUND2_COMMIT,
            "sha256": sha, "bytes": n,
            "matches_pinned_constant": sha == ROUND2_V2_PROMPT_SHA256}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reused", default="outputs/ctt_v2/captions/pilot_m3_round2/run_meta.json",
                    help="run_meta.json of the store whose descriptions are REUSED")
    ap.add_argument("--against", action="append", default=[],
                    help="run_meta.json of a mass-run chunk to config-match (repeatable)")
    ap.add_argument("--out", default="outputs/ctt_v2/captions/V2_PROMPT_IDENTITY.json")
    a = ap.parse_args()

    import generate_descriptions as gd  # noqa: E402  (after sys.path fix-up)

    live_sha, live_bytes = render_hash(gd.build_system_prompt)
    ref = reference_hash()
    prompt_ok = live_sha == ref["sha256"]

    reused_meta = json.loads(Path(a.reused).read_text())
    config = {}
    config_ok = True
    for key, names in GEN_KEYS.items():
        reused_v = _get(reused_meta, names)
        others = {}
        for p in a.against:
            others[p] = _get(json.loads(Path(p).read_text()), names)
        same = all(v == reused_v for v in others.values())
        config[key] = {"reused": reused_v, "mass_run": others, "match": same}
        config_ok &= same

    res = {
        "assert": "v2 production prompt byte-identity + reuse config match",
        "authority": ("advisors/A14_RECONCILIATION_VERBATIM.md -- Q1 byte-identity assert; "
                      "'WHAT WOULD OVERTURN THIS' config-match check on run_meta.json"),
        "prompt": {
            "domain": f"roles {list(ROLES)} x N in [{N_DOMAIN.start}, {N_DOMAIN.stop - 1}] "
                      f"({len(ROLES) * len(N_DOMAIN)} rendered prompts, {live_bytes} bytes)",
            "reference": ref, "live_sha256": live_sha,
            "BYTE_IDENTICAL": prompt_ok,
        },
        "config_match": {
            "reused_store": a.reused, "compared_against": a.against,
            "generation_keys_only": ("auditor keys are deliberately excluded -- the auditor "
                                     "difference is the ruled-on fact that A14 step 2's "
                                     "re-audit cures, not an undiscovered drift"),
            "fields": config, "ALL_MATCH": config_ok,
        },
        "VERDICT": "PASS" if (prompt_ok and config_ok) else "DRIFT",
        "on_drift": ("A14: any generation-config diff => DROP the reuse and regenerate all "
                     "1,348 under v2. Bring the diff to the owner; do not paper over it."),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if res["VERDICT"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
