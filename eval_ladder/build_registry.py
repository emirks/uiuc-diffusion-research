"""ladder2 — build `registry.jsonl`: THE single source of truth for the eval ladder.

One row = one generation item. `run_gen.py` consumes rows keyed by `arm` (which model makes
them); `run_eval.py` consumes the same rows keyed by `item_id`. Nothing downstream re-derives
a fact: the cell label, the GT pool, the % type, the priority and the base twin are all
computed HERE, once, from three frozen inputs — `split_v1.2.json`, the caption corpus, and
`arms.yaml`.

That is the whole point. Every defect the old ladder shipped (the prompt leak, the C4/C8
join flip, the suffix bleed, the copy_max mix-up, the id collision) came from the same root
cause: the same fact written down twice, by hand, with no conformance check.

Ontology (frozen, owner-approved):
  reference-novelty  seen (exact demo trained) -> unseen (new demo of a trained class)
                     -> zero-shot (held-out class);  specialists have no reference axis
  content            same / cross / foreign(DAVIS), always sidedness-matched
  endpoints          always untrained content (test band / held-out / DAVIS), except the two
                     fit anchors which deliberately use train endpoints

Run:  python eval_ladder/build_registry.py [--stats]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402

STD = REPO_ROOT / "data/processed/transitions_std121"
SPLIT_PATH = STD / "split_v1.2.json"
SPLIT_SHA = "c694659d6d2e264528ccb546b43b9974bfecd2770ab674ba7b514981d026e6ce"
REGISTRY = HERE / "registry.jsonl"
ARMS = HERE / "arms.yaml"
INVENTORY = HERE / "train/inventory.json"

#: how many cross recipients each donor gets (sidedness-matched, deterministic round-robin)
CROSS_PER_DONOR = 2
#: DAVIS endpoints per donor. Owner call 2026-07-23: the foreign lane is only generation, so run it
#: at meaningful n instead of a token gate — ALL donors, 2 DAVIS endpoints each. The %-typing
#: discipline is unchanged: foreign stays %_proxy and its claim is the margin vs base, never the level.
FOREIGN_PER_DONOR = 2

PRIORITY = {
    "SP-fit": "P0", "SP-same": "P0", "SP-cross": "P0",
    "G-fit": "P0", "G-unseen-same": "P0", "G-unseen-cross": "P0",
    "G-memo-probe": "P1", "G-zs-cross": "P1",
    "G-zs-same": "P2",
    "G-ref-control": "P1",
    "SP-foreign": "P2", "G-unseen-foreign": "P2", "G-zs-foreign": "P2",
    "text_floor": "P0",
}


# --------------------------------------------------------------------------- frozen inputs
def load() -> tuple[dict, dict, dict]:
    split = json.loads(SPLIT_PATH.read_text())
    assert split["split"] == "v1.2" and split["sha256"] == SPLIT_SHA, "split is not the frozen v1.2"
    arms = yaml.safe_load(ARMS.read_text())
    inv = json.loads(INVENTORY.read_text())
    return split, arms, inv


class Corpus:
    """Every clip fact the builder needs, derived once from the frozen split."""

    def __init__(self, split: dict):
        self.split = split
        self.quarantined = set(split["quarantined"])
        self.held_out = set(split["generalist_holdout"])
        self.roster = list(split["specialist_roster"])
        self.sided = prompts.sidedness()
        self.audited = prompts.audited_clips()
        self.train, self.test = {}, {}
        for cls, entry in split["classes"].items():
            self.train[cls] = [c for c in sorted(entry["train"]) if c not in self.quarantined]
            self.test[cls] = [c for c in sorted(entry["test"]) if c not in self.quarantined]
        self.held_in = [c for c in sorted(split["classes"]) if c not in self.held_out]
        #: the generalist comparison pool: held-in classes with >=2 test clips (so the same
        #: class can supply both an unseen reference and a distinct unseen endpoint)
        self.g_pool = [c for c in self.held_in if len(self.test[c]) >= 2]

    def clips_of(self, cls: str, band: str) -> list[str]:
        return {"train": self.train, "test": self.test}[band][cls]

    def band(self, clip: str) -> str:
        cls = prompts.clip_class(clip)
        if clip in self.test[cls]:
            return "test"
        if clip in self.train[cls]:
            return "train"
        raise KeyError(clip)

    def endpoint_source(self, clip: str) -> str:
        cls = prompts.clip_class(clip)
        return "heldout" if cls in self.held_out else f"heldin_{self.band(clip)}"

    def eval_endpoints(self, sided: str, exclude_class: str | None = None) -> list[str]:
        """Untrained-content endpoints usable at eval: held-in TEST + all held-out clips."""
        out = []
        for cls in sorted(self.split["classes"]):
            if self.sided[cls] != sided or cls == exclude_class:
                continue
            band = self.train[cls] + self.test[cls] if cls in self.held_out else self.test[cls]
            out += [c for c in band if c in self.audited]
        return sorted(out)


# --------------------------------------------------------------------------- row assembly
def novelty_of(reference: str | None, corpus: Corpus) -> str:
    """Reference novelty is a pure function of the reference's class + band. Never the endpoint."""
    if reference is None:
        return "none"                                   # specialist: transition is in the weights
    cls = prompts.clip_class(reference)
    if cls in corpus.held_out:
        return "zero_shot"
    return "seen" if corpus.band(reference) == "train" else "unseen"


def content_of(endpoint_class: str, donor_class: str, foreign: bool) -> str:
    if foreign:
        return "foreign"
    return "same" if endpoint_class == donor_class else "cross"


def make_row(cell: str, arm: str, endpoint: str, donor_class: str, corpus: Corpus,
             token: str, reference: str | None = None,
             mismatched_reference: bool = False) -> dict:
    ep_class = prompts.clip_class(endpoint)
    foreign = prompts.is_davis(endpoint)
    sided = corpus.sided[donor_class]
    assert prompts.clip_sidedness(endpoint) == sided, (
        f"{cell}: sidedness mismatch endpoint={endpoint}({prompts.clip_sidedness(endpoint)}) "
        f"donor={donor_class}({sided})")
    content = content_of(ep_class, donor_class, foreign=foreign)
    prompt = prompts.render_prompt(endpoint, sided, token)
    ec.cond_paths(endpoint, sided)                       # seatbelt 6: windows exist, one rule
    row = {
        "item_id": f"{cell}__{arm}__{endpoint}" + (f"__ref_{reference}" if reference else ""),
        "mismatched_reference": mismatched_reference,
        "cell": cell,
        "priority": PRIORITY[cell],
        "arm": arm,
        "ref_novelty": novelty_of(reference, corpus),
        "content": content,
        "donor_class": donor_class,
        "endpoint": endpoint,
        "endpoint_class": ep_class,
        "endpoint_split": "davis" if foreign else corpus.band(endpoint),
        "endpoint_source": "davis" if foreign else corpus.endpoint_source(endpoint),
        "sided": sided,
        "reference": reference,
        "reference_split": corpus.band(reference) if reference else None,
        "prompt": prompt,
        "pct_type": "same" if content == "same" else "proxy",
        "gt_pool_class": donor_class,                    # GT pool is ALWAYS the donor class
    }
    row["input_key"] = input_key(row)
    return row


def input_key(row: dict) -> str:
    """Identity of the generator INPUT (arm-independent) — the base arm dedups on this."""
    payload = json.dumps([row["endpoint"], row["prompt"], row["sided"], row["reference"]])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def rotate(pool: list[str], i: int, k: int) -> list[str]:
    """Deterministic spread: donor i takes k items starting at offset i (no RNG anywhere)."""
    return [pool[(i + j) % len(pool)] for j in range(min(k, len(pool)))] if pool else []


# --------------------------------------------------------------------------- the cells
def build_rows(corpus: Corpus, token: str) -> list[dict]:
    rows: list[dict] = []

    # ---------------- specialists: the transition is baked into the weights, no reference
    for i, cls in enumerate(corpus.roster):
        arm = f"spec_{cls}"
        sided = corpus.sided[cls]
        rows.append(make_row("SP-fit", arm, corpus.train[cls][0], cls, corpus, token))
        for clip in corpus.test[cls]:
            rows.append(make_row("SP-same", arm, clip, cls, corpus, token))
        for clip in rotate(corpus.eval_endpoints(sided, exclude_class=cls), i * 3, CROSS_PER_DONOR):
            rows.append(make_row("SP-cross", arm, clip, cls, corpus, token))

    # ---------------- generalist: the reference carries the transition
    arm = "ic_gen"
    for i, cls in enumerate(corpus.g_pool):
        train, test = corpus.train[cls], corpus.test[cls]
        sided = corpus.sided[cls]
        # seen demo + train endpoint  -> fit anchor / memorisation ceiling
        rows.append(make_row("G-fit", arm, train[1], cls, corpus, token, reference=train[0]))
        # seen demo + TEST endpoint   -> holds endpoint-novelty fixed vs G-unseen-same
        rows.append(make_row("G-memo-probe", arm, test[0], cls, corpus, token, reference=train[0]))
        # unseen demo + TEST endpoint -> the in-distribution capability claim
        rows.append(make_row("G-unseen-same", arm, test[1], cls, corpus, token, reference=test[0]))
        # unseen demo + foreign-class endpoint -> reference dominance
        for clip in rotate(corpus.eval_endpoints(sided, exclude_class=cls), i * 3, CROSS_PER_DONOR):
            rows.append(make_row("G-unseen-cross", arm, clip, cls, corpus, token,
                                 reference=test[0]))
        # reference-USE control: byte-identical input to G-unseen-same except the demo, which
        # comes from a DIFFERENT class. Without it, G-unseen-same and G-memo-probe cannot show
        # the model actually uses the demo rather than the endpoint + token alone. Scored
        # against the SAME pool as its twin, so the pair isolates the demo's contribution.
        wrong = [d for d in corpus.g_pool if d != cls and corpus.sided[d] == sided]
        if wrong:
            rows.append(make_row("G-ref-control", arm, test[1], cls, corpus, token,
                                 reference=corpus.test[wrong[i % len(wrong)]][0],
                                 mismatched_reference=True))

    # ---------------- zero-shot: reference from a class the generalist never trained on
    for i, cls in enumerate(sorted(corpus.held_out)):
        pool = corpus.train[cls] + corpus.test[cls]
        if len(pool) < 2:
            continue
        # reference = a train-band demo of the held-out class (references are never prompted,
        # so they need no audit); same-content endpoint = an AUDITED clip of that class.
        ref = pool[0]
        audited_same = [c for c in corpus.test[cls] + corpus.train[cls]
                        if c != ref and c in corpus.audited]
        same_endpoint = audited_same[0] if audited_same else None
        sided = corpus.sided[cls]
        # zero-shot demo + held-in test endpoint (known content, never-trained transition)
        heldin = [c for c in corpus.eval_endpoints(sided, exclude_class=cls)
                  if prompts.clip_class(c) not in corpus.held_out]
        for clip in rotate(heldin, i * 3, CROSS_PER_DONOR):
            rows.append(make_row("G-zs-cross", arm, clip, cls, corpus, token, reference=ref))
        if same_endpoint is not None:
            rows.append(make_row("G-zs-same", arm, same_endpoint, cls, corpus, token, reference=ref))

    # ---------------- foreign (DAVIS): does a corpus-learned transition apply to arbitrary
    # real footage? Gated lane: %-suppressed to ranking-only, claim = margin vs base + 2AFC.
    davis_by_side = {"one": [], "two": []}
    for name, entry in prompts.davis().items():
        davis_by_side[entry["sided"]].append(name)
    for side in davis_by_side:
        davis_by_side[side].sort()

    def davis_pick(sided: str, i: int) -> str:
        pool = davis_by_side[sided]
        return pool[i % len(pool)]

    # specialists: EVERY donor x FOREIGN_PER_DONOR DAVIS endpoints
    for i, cls in enumerate(corpus.roster):
        for j in range(FOREIGN_PER_DONOR):
            rows.append(make_row("SP-foreign", f"spec_{cls}",
                                 davis_pick(corpus.sided[cls], i * FOREIGN_PER_DONOR + j),
                                 cls, corpus, token))
    # generalist, unseen demo: every g_pool donor
    for i, cls in enumerate(corpus.g_pool):
        for j in range(FOREIGN_PER_DONOR):
            rows.append(make_row("G-unseen-foreign", "ic_gen",
                                 davis_pick(corpus.sided[cls], i * FOREIGN_PER_DONOR + j),
                                 cls, corpus, token, reference=corpus.test[cls][0]))
    # generalist, zero-shot demo: every held-out donor
    for i, cls in enumerate(sorted(corpus.held_out)):
        pool = corpus.train[cls] + corpus.test[cls]
        if not pool:
            continue
        for j in range(FOREIGN_PER_DONOR):
            rows.append(make_row("G-zs-foreign", "ic_gen",
                                 davis_pick(corpus.sided[cls], i * FOREIGN_PER_DONOR + j),
                                 cls, corpus, token, reference=pool[0]))

    # ---------------- base twins: identical input, no adapter (margin denominator)
    seen: set[str] = set()
    base_rows = []
    for r in rows:
        if r["input_key"] in seen:
            continue
        seen.add(r["input_key"])
        b = dict(r)
        b["arm"] = "base"
        b["cell"] = f"base:{r['cell']}"
        b["item_id"] = f"base__{r['input_key']}"
        base_rows.append(b)

    # ---------------- text floor: prompt only, nothing else. If it scores near the pool
    # floor, the prompt genuinely carries no transition (the leak-proof).
    floor_rows = []
    # 12 distinct classes (the specialist roster and the generalist pool overlap)
    floor_classes = list(dict.fromkeys(corpus.roster + corpus.g_pool))[:12]
    for cls in floor_classes:
        clip = corpus.test[cls][0]
        floor_rows.append({
            "item_id": f"text_floor__{clip}", "cell": "text_floor", "priority": "P0",
            "arm": "text_floor", "ref_novelty": "none", "content": "same",
            "donor_class": cls, "endpoint": None, "endpoint_class": cls,
            "endpoint_split": "test", "endpoint_source": "heldin_test",
            "sided": corpus.sided[cls], "reference": None, "reference_split": None,
            "prompt": prompts.render_prompt(clip, corpus.sided[cls], token),
            "pct_type": "same", "gt_pool_class": cls,
            "input_key": f"floor_{clip}", "conditioning": "none",
        })
    return rows + base_rows + floor_rows


# --------------------------------------------------------------------------- seatbelts
def seatbelts(rows: list[dict], corpus: Corpus, inv: dict, token: str) -> None:
    ids = [r["item_id"] for r in rows]
    assert len(ids) == len(set(ids)), f"item_id collision: {len(ids) - len(set(ids))} duplicates"

    trained_classes = set(inv["models"]["ic_gen"]["classes"])
    trained_clips = {c for v in inv["clips"].values() for c in v}

    for r in rows:
        # 2 — prompt is RENDERED, never authored
        if r["endpoint"] is not None:
            assert r["prompt"] == prompts.render_prompt(r["endpoint"], r["sided"], token), \
                f"{r['item_id']}: prompt is not the rendered form"
        assert prompts.MARKER not in r["prompt"], f"{r['item_id']}: outcome text leaked"
        assert f" {token}." in r["prompt"], f"{r['item_id']}: token missing"

        # 3 — contamination: no held-out class in the generalist's training set; quarantine held
        assert not (trained_classes & corpus.held_out), "a held-out class is in the train roster"
        for clip in (r["endpoint"], r["reference"]):
            assert clip not in corpus.quarantined, f"{r['item_id']}: quarantined clip {clip}"

        # 7 — cell-derivation asserts
        assert r["ref_novelty"] == novelty_of(r["reference"], corpus), f"{r['item_id']}: novelty"
        if r["reference"] is not None:
            assert r["reference"] != r["endpoint"], f"{r['item_id']}: reference == endpoint"
            if r["content"] != "same":
                assert prompts.clip_class(r["reference"]) != r["endpoint_class"], \
                    f"{r['item_id']}: cross row whose reference shares the endpoint class"
            elif r.get("mismatched_reference"):
                # the control INVERTS the invariant on purpose: same content, wrong demo
                assert prompts.clip_class(r["reference"]) != r["donor_class"], \
                    f"{r['item_id']}: reference-use control whose demo is the donor class"
            else:
                assert prompts.clip_class(r["reference"]) == r["donor_class"], \
                    f"{r['item_id']}: same-content row whose demo is not the donor class"
            if r["ref_novelty"] == "zero_shot":
                assert r["reference"] not in trained_clips, f"{r['item_id']}: zs ref was trained"
        # Only the fit anchors may use content the ARM ITSELF was trained on. "train band" is
        # not the test: a held-out class's train clips were never trained by anyone, and are
        # legitimate untrained content (advisor: endpoint split = content-novelty only).
        base_cell = r["cell"].removeprefix("base:")
        if r["endpoint"] is not None and r["arm"] in inv["clips"]:
            if r["endpoint"] in inv["clips"][r["arm"]]:
                assert base_cell in ("SP-fit", "G-fit"), \
                    f"{r['item_id']}: endpoint was in this arm's training set, outside a fit anchor"
        # eval endpoints must be audited (leak-checked against their own frames)
        if r["endpoint"] is not None and base_cell not in ("SP-fit", "G-fit"):
            assert r["endpoint"] in corpus.audited, f"{r['item_id']}: unaudited eval endpoint"

        # %-typing firewall: %_same iff the endpoint really is the donor class
        expect = "same" if r["endpoint_class"] == r["donor_class"] else "proxy"
        assert r["pct_type"] == expect, f"{r['item_id']}: %-type {r['pct_type']} != {expect}"
        assert r["gt_pool_class"] == r["donor_class"], f"{r['item_id']}: GT pool is not the donor"

        # 1/5 — strict sidedness matching (mask is derived from this at eval time)
        assert corpus.sided[r["donor_class"]] == r["sided"], f"{r['item_id']}: sidedness"

    # base twins: exactly one per distinct input, and every treatment row has one
    base_keys = {r["input_key"] for r in rows if r["arm"] == "base"}
    treat = [r for r in rows if r["arm"] not in ("base", "text_floor")]
    missing = {r["input_key"] for r in treat} - base_keys
    assert not missing, f"{len(missing)} treatment rows have no base twin"
    print(f"[seatbelts] all green on {len(rows)} rows")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    split, arms, inv = load()
    token = arms["token"]
    corpus = Corpus(split)
    rows = build_rows(corpus, token)
    seatbelts(rows, corpus, inv, token)

    REGISTRY.write_text("".join(json.dumps(r) + "\n" for r in rows))

    n_seeds = len(arms["seeds"])
    by_cell: dict[str, int] = {}
    for r in rows:
        by_cell[r["cell"]] = by_cell.get(r["cell"], 0) + 1
    by_prio: dict[str, int] = {}
    for r in rows:
        by_prio[r["priority"]] = by_prio.get(r["priority"], 0) + 1

    print(f"[registry] {len(rows)} items x {n_seeds} seeds = {len(rows) * n_seeds} generations")
    print(f"[registry] g_pool ({len(corpus.g_pool)}): {corpus.g_pool}")
    width = max(len(c) for c in by_cell)
    for cell in sorted(by_cell, key=lambda c: (c.startswith("base:"), c)):
        print(f"  {cell:{width}s} {by_cell[cell]:4d} items  x{n_seeds} = {by_cell[cell] * n_seeds:4d} gens")
    print(f"[registry] by priority: " + ", ".join(
        f"{p}={by_prio[p] * n_seeds} gens" for p in sorted(by_prio)))
    print(f"[registry] wrote {REGISTRY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
