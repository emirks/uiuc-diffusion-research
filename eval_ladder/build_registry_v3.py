"""ladder2 — build `registry_v3.jsonl` (design 3.0.0): a SUPERSET of registry.jsonl's generalist rows.

Same contract as build_registry.py — one row = one generation item, every fact derived once from
frozen inputs, seatbelts asserted, nothing downstream re-derives anything. This builder IMPORTS the
v2 machinery (Corpus, make_row, rotate, input_key, seatbelts) instead of re-implementing it, so a v3
row is schema-identical to a v2 row by construction and stamps into store/prompts with stamp_rows.py.

Frozen inputs: split_v1.2.json (sha-pinned by build_registry), the caption corpus, arms.yaml,
davis.yaml, inventory.json, PLUS grid_v3.yaml (the only new hand-written file: sources, never rows).

Two outputs, one schema:
  registry_v3.jsonl          rows whose inputs exist TODAY (old ic_gen rows byte-identical + new rows on
                             audited corpus endpoints + base twins). Pluggable now.
  registry_v3_pending.jsonl  rows derived by the same rules whose endpoint/reference is not yet a corpus
                             member (std121 processing / caption / audit / conditioning windows / split entry).
                             Same fields, prompt = null, input_key = null, plus `pending` = the blockers.
                             Re-running the builder after the data lands moves rows into registry_v3.jsonl.
Plus the health report (misc/2026-09-07_eval_grid_v2/HEALTH.md + health.json) and a staged prompts-shelf
family (misc/2026-09-07_eval_grid_v2/prompts_staging/) in the exact seed_prompts.py format.

Ontology is unchanged (reference novelty x content, sidedness-matched). New endpoint sources:
  humanvid   the never-trained reserve (CONTENT_POOL_union.json:reserved), content = foreign
  effectdata EffectData clips, one-sided; content = same iff the endpoint clip's effect == donor
Run:  python eval_ladder/build_registry_v3.py [--stats] [--no-probe]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import build_registry as v2  # noqa: E402  (Corpus, make_row, rotate, input_key, seatbelts, PRIORITY)
import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402

GRID = HERE / "grid_v3.yaml"
OUT = HERE / "registry_v3.jsonl"
OUT_PENDING = HERE / "registry_v3_pending.jsonl"
HEALTH_DIR = REPO_ROOT / "misc/2026-09-07_eval_grid_v2"
STAGING = HEALTH_DIR / "prompts_staging"
NPZ_V4 = (REPO_ROOT / ".claude/worktrees/eval-v4-cert/outputs/eval/certification"
          "/4.0.0-draft.1/analysis/distance_matrices.npz")

# measured 2026-09-07 on evals 012/024 (dual-force control, seeds 42 vs 43): per-generation seed SD of pool-%
SEED_SD_PP = 11.7

ROW_FIELDS = ["item_id", "mismatched_reference", "cell", "priority", "arm", "ref_novelty", "content",
              "donor_class", "endpoint", "endpoint_class", "endpoint_split", "endpoint_source", "sided",
              "reference", "reference_split", "prompt", "pct_type", "gt_pool_class", "input_key"]


# --------------------------------------------------------------------------- helpers
def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def video_geo(path: Path):
    import av
    c = av.open(str(path))
    s = c.streams.video[0]
    g = (s.codec_context.width, s.codec_context.height, s.frames, float(s.average_rate or 0))
    c.close()
    return g


def pending_row(cell: str, endpoint: str, donor: str, sided: str, reference: str | None,
                endpoint_class: str, endpoint_source: str, endpoint_split: str,
                reference_split: str | None, ref_novelty: str, content: str,
                pending: list[str]) -> dict:
    """A v3 row whose prompt cannot be rendered yet. Same fields as make_row(); prompt/input_key null."""
    return {
        "item_id": f"{cell}__ic_gen__{endpoint}" + (f"__ref_{reference}" if reference else ""),
        "mismatched_reference": False,
        "cell": cell,
        "priority": v2.PRIORITY[cell],
        "arm": "ic_gen",
        "ref_novelty": ref_novelty,
        "content": content,
        "donor_class": donor,
        "endpoint": endpoint,
        "endpoint_class": endpoint_class,
        "endpoint_split": endpoint_split,
        "endpoint_source": endpoint_source,
        "sided": sided,
        "reference": reference,
        "reference_split": reference_split,
        "prompt": None,
        "pct_type": "same" if content == "same" else "proxy",
        "gt_pool_class": donor,
        "input_key": None,
        "pending": sorted(set(pending)),
    }


class Reserve:
    """The never-trained foreign roster (CONTENT_POOL_union.json:reserved). Deterministic rotation."""

    def __init__(self, cfg: dict):
        pool = json.loads((REPO_ROOT / cfg["pool"]).read_text())
        self.entries = sorted((r for r in pool["reserved"] if r["bank"] == cfg["bank"]),
                              key=lambda r: r["clip_id"])
        self.ids = [r["clip_id"] for r in self.entries]
        self.mp4 = {r["clip_id"]: r["mp4"].replace("/projects/illinois", "/taiga/illinois") for r in self.entries}
        self.offset = int(cfg["pair_offset"])
        self.i_one = 0
        self.i_two = 0
        assert len(self.ids) > self.offset, "reserve too small for pairing"

    def one(self) -> str:
        c = self.ids[self.i_one % len(self.ids)]
        self.i_one += 1
        return c

    def two(self) -> str:
        n = len(self.ids) - self.offset
        a = self.ids[self.i_two % n]
        b = self.ids[self.i_two % n + self.offset]
        self.i_two += 1
        return f"hvpair.{a}.{b}"

    def pick(self, sided: str) -> str:
        return self.one() if sided == "one" else self.two()


class EffectData:
    """EffectData facts + on-demand orientation probing (cached). Everything sorted, no RNG."""

    def __init__(self, cfg: dict, probe: bool = True):
        self.cfg = cfg
        ann = json.loads((REPO_ROOT / cfg["annotations"]).read_text())
        recs = list(ann.values()) if isinstance(ann, dict) else ann
        self.cat: dict[str, set[str]] = collections.defaultdict(set)
        self.clips: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)  # effect -> (stem, subject, tag)
        self.subj: dict[str, set[str]] = collections.defaultdict(set)                    # subject -> effects (ALL)
        self.stem_path: dict[str, str] = {}
        for r in recs:
            fn = r["video_path"].rsplit("/", 1)[-1][:-4]
            parts = fn.split(",")
            if len(parts) != 3:
                continue                                   # untagged uuid clips: excluded (no subject key)
            effect, subject, tag = parts
            self.clips[effect].append((fn, subject, tag))
            self.subj[subject].add(effect)
            self.cat[effect].add(r["vfx_en"])
            self.stem_path[fn] = r["video_path"]
        for e in self.clips:
            self.clips[e].sort()
        sel = json.loads((REPO_ROOT / cfg["selection"]).read_text())
        self.selected_subjects = {c["subject"] for c in sel["clips"]}
        self.cache_path = REPO_ROOT / cfg["shape_cache"]
        self.shapes: dict[str, list[int]] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.probe_enabled = probe
        self.ok_shapes = {tuple(s) for s in cfg["portrait_ok_shapes"]}
        self._zips: dict[str, zipfile.ZipFile] = {}

    def category(self, effect: str) -> str:
        return sorted(self.cat[effect])[0]

    def eligible_effect(self, effect: str) -> bool:
        return len(self.clips[effect]) >= self.cfg["min_clips_per_effect"] and len(self.cat[effect]) == 1

    def shape(self, stem: str) -> tuple[int, int] | None:
        if stem in self.shapes:
            w, h = self.shapes[stem]
            return (w, h)
        if not self.probe_enabled:
            return None
        effect = stem.split(",")[0]
        z = self._zips.get(effect)
        if z is None:
            z = zipfile.ZipFile(REPO_ROOT / self.cfg["zips"] / f"{effect}.zip")
            self._zips[effect] = z
        import av
        member = self.stem_path[stem]
        try:
            data = z.read(member)
        except KeyError:
            data = z.read(member.split("/", 1)[-1]) if "/" in member else None
        c = av.open(io.BytesIO(data))
        s = c.streams.video[0]
        wh = (s.codec_context.width, s.codec_context.height)
        c.close()
        self.shapes[stem] = list(wh)
        return wh

    def portrait_ok(self, stem: str) -> bool:
        sh = self.shape(stem)
        return sh is not None and sh in self.ok_shapes

    def subject_portrait_ok(self, subject: str) -> bool:
        """Orientation is per subject (verified on the S6 roster: 2000/2000). Probe one clip."""
        effect = sorted(self.subj[subject])[0]
        stem = next(fn for fn, s, t in self.clips[effect] if s == subject)
        return self.portrait_ok(stem)

    def portrait_clips(self, effect: str) -> list[tuple[str, str, str]]:
        return [c for c in self.clips[effect] if self.portrait_ok(c[0])]

    def save(self):
        self.cache_path.write_text(json.dumps(self.shapes, sort_keys=True))

    @staticmethod
    def std_stem(prefix: str, stem: str) -> str:
        effect, subject, tag = stem.split(",")
        return f"{prefix}.{effect}.{subject}.{tag}"


# --------------------------------------------------------------------------- the v3 cells
def build(cfg: dict, corpus: v2.Corpus, token: str, inv: dict, probe: bool):
    keep = set(cfg["keep_cells"])
    mid_cls = set(cfg["mid_effect_anchor"]["classes"])
    mid_clip = set(cfg["mid_effect_anchor"]["clips"])
    reserve = Reserve(cfg["reserve"])
    trained_clips = {c for v in inv["clips"].values() for c in v}

    # ---------------- 0. old rows: the v2 generalist rows, kept byte-identical
    v2rows = v2.build_rows(corpus, token)
    old = [r for r in v2rows if r["arm"] == "ic_gen" and r["cell"] in keep]
    frozen = {r["item_id"]: r for r in jl(v2.REGISTRY)}
    for r in old:
        assert frozen.get(r["item_id"]) == r, f"v2 row drifted: {r['item_id']}"
    flags: dict[str, list[str]] = collections.defaultdict(list)
    for r in old:
        if r["endpoint_class"] in mid_cls or r["endpoint"] in mid_clip:
            flags[r["item_id"]].append("mid_effect_anchor")

    new_a: list[dict] = []      # renderable now
    pend: list[dict] = []       # blocked on data
    raw_root = REPO_ROOT / cfg["raw_root"]

    def raw_path(stem: str, cls: str) -> Path | None:
        """Raw folder names do not always equal the class (flame_transition_* -> class flame): search by stem."""
        p = raw_root / cls / f"{stem}.mp4"
        if p.exists():
            return p
        hits = sorted(raw_root.glob(f"*/{stem}.mp4"))
        return hits[0] if hits else None

    def raw_blockers(stem: str, cls: str, folder: Path | None = None) -> list[str]:
        p = (folder / f"{stem}.mp4") if folder else raw_path(stem, cls)
        b = ["std121", "cond_windows", "split_entry"]
        if p is None or not p.exists():
            return b + ["raw_missing"]
        w, h, n, r = video_geo(p)
        if min(w, h) < cfg["min_short_side_px"]:
            b.append(f"low_res_{w}x{h}")
        if r and n / r < cfg["min_duration_s"]:
            b.append("short")
        return b

    # ---------------- 1. Tier 1: second reference for the 13 two-test classes (renderable now)
    for i, cls in enumerate(corpus.g_pool):
        test, sided = corpus.test[cls], corpus.sided[cls]
        ref = test[1]
        if cls not in mid_cls and test[0] not in mid_clip:
            new_a.append(v2.make_row("G-unseen-same", "ic_gen", test[0], cls, corpus, token, reference=ref))
        for clip in v2.rotate(corpus.eval_endpoints(sided, exclude_class=cls), i * 3 + v2.CROSS_PER_DONOR, 1):
            new_a.append(v2.make_row("G-unseen-cross", "ic_gen", clip, cls, corpus, token, reference=ref))
        fe = reserve.pick(sided)
        pend.append(pending_row("G-unseen-foreign", fe, cls, sided, ref, "davis", "humanvid", "foreign",
                                "test", "unseen", "foreign", ["caption", "audit", "cond_windows", "roster_entry"]))

    # ---------------- 2. Tier 1: one-test trained classes with a raw top-up (same endpoint = the new clip)
    for cls, stems in cfg["tier1_topups"].items():
        sided = corpus.sided[cls]
        ref = corpus.test[cls][0]
        assert ref not in trained_clips
        i = sorted(cfg["tier1_topups"]).index(cls)
        for stem in stems:
            pend.append(pending_row("G-unseen-same", stem, cls, sided, ref, cls, "heldin_test", "test", "test",
                                    "unseen", "same", raw_blockers(stem, cls) + ["caption", "audit", "owner_review"]))
        for clip in v2.rotate(corpus.eval_endpoints(sided, exclude_class=cls), i * 3, 1):
            new_a.append(v2.make_row("G-unseen-cross", "ic_gen", clip, cls, corpus, token, reference=ref))
        pend.append(pending_row("G-unseen-foreign", reserve.pick(sided), cls, sided, ref, "davis", "humanvid",
                                "foreign", "test", "unseen", "foreign", ["caption", "audit", "cond_windows", "roster_entry"]))

    # ---------------- 3. Tier 1: trained classes with one test clip and NO top-up -> cross + foreign only
    for j, cls in enumerate(sorted(c for c in corpus.held_in if len(corpus.test[c]) == 1
                                   and c not in cfg["tier1_topups"] and c not in cfg["flame_additions"])):
        sided = corpus.sided[cls]
        ref = corpus.test[cls][0]
        for clip in v2.rotate(corpus.eval_endpoints(sided, exclude_class=cls), j * 3, 1):
            new_a.append(v2.make_row("G-unseen-cross", "ic_gen", clip, cls, corpus, token, reference=ref))
        pend.append(pending_row("G-unseen-foreign", reserve.pick(sided), cls, sided, ref, "davis", "humanvid",
                                "foreign", "test", "unseen", "foreign", ["caption", "audit", "cond_windows", "roster_entry"]))

    # ---------------- 4. Tier 1: trained classes whose only untrained clip is a new raw clip (reference role)
    for cls, stems in cfg["tier1_reference_only"].items():
        sided = corpus.sided[cls]
        for stem in stems:
            b = raw_blockers(stem, cls) + ["owner_review"]
            i = sorted(cfg["tier1_reference_only"]).index(cls)
            for clip in v2.rotate(corpus.eval_endpoints(sided, exclude_class=cls), i * 3, 1):
                pend.append(pending_row("G-unseen-cross", clip, cls, sided, stem, prompts.clip_class(clip),
                                        corpus.endpoint_source(clip), corpus.band(clip), "test", "unseen", "cross", b))
            pend.append(pending_row("G-unseen-foreign", reserve.pick(sided), cls, sided, stem, "davis", "humanvid",
                                    "foreign", "test", "unseen", "foreign", b + ["caption", "audit", "cond_windows", "roster_entry"]))

    # ---------------- 5. Tier 1: flame (two-sided) — 3 untrained curated full-res clips
    for cls, stems in cfg["flame_additions"].items():
        sided = corpus.sided[cls]
        folder = None                       # resolved by stem through raw_path()
        refs = stems[:2]
        for k, ref in enumerate(refs):
            same_ep = refs[1 - k]
            b = raw_blockers(ref, cls, folder) + raw_blockers(same_ep, cls, folder) + ["caption", "audit"]
            pend.append(pending_row("G-unseen-same", same_ep, cls, sided, ref, cls, "heldin_test", "test", "test", "unseen", "same", b))
            for clip in v2.rotate(corpus.eval_endpoints(sided, exclude_class=cls), k * 3, 1):
                pend.append(pending_row("G-unseen-cross", clip, cls, sided, ref, prompts.clip_class(clip),
                                        corpus.endpoint_source(clip), corpus.band(clip), "test", "unseen", "cross",
                                        raw_blockers(ref, cls, folder)))
            pend.append(pending_row("G-unseen-foreign", reserve.pick(sided), cls, sided, ref, "davis", "humanvid",
                                    "foreign", "test", "unseen", "foreign",
                                    raw_blockers(ref, cls, folder) + ["caption", "audit", "cond_windows", "roster_entry"]))

    # ---------------- 6. Tier 2: second reference for the 10 held-out classes
    for i, cls in enumerate(sorted(corpus.held_out)):
        pool = corpus.train[cls] + corpus.test[cls]
        if len(pool) < 2:
            continue
        ref = pool[1]
        sided = corpus.sided[cls]
        audited_same = [c for c in corpus.test[cls] + corpus.train[cls]
                        if c != ref and c in corpus.audited and c not in mid_clip]
        if cls not in mid_cls and audited_same:
            new_a.append(v2.make_row("G-zs-same", "ic_gen", audited_same[0], cls, corpus, token, reference=ref))
        heldin = [c for c in corpus.eval_endpoints(sided, exclude_class=cls)
                  if prompts.clip_class(c) not in corpus.held_out]
        for clip in v2.rotate(heldin, i * 3 + v2.CROSS_PER_DONOR, 1):
            new_a.append(v2.make_row("G-zs-cross", "ic_gen", clip, cls, corpus, token, reference=ref))
        pend.append(pending_row("G-zs-foreign", reserve.pick(sided), cls, sided, ref, "davis", "humanvid",
                                "foreign", "train" if ref in corpus.train[cls] else "test", "zero_shot", "foreign",
                                ["caption", "audit", "cond_windows", "roster_entry"]))

    # ---------------- 7. Tier 2: new zero-shot Higgsfield classes (raw full-res, PROVISIONAL list)
    zs_new = cfg["new_zero_shot_classes"]
    for i, (cls, sided) in enumerate(sorted(zs_new.items())):
        folder = raw_root / cls
        stems = sorted(p.stem for p in folder.glob("*.mp4"))
        good = [s for s in stems if not any(x.startswith("low_res") or x == "short" for x in raw_blockers(s, cls))]
        assert len(good) >= 4, f"{cls}: only {len(good)} usable full-res clips"
        refs = good[:2]
        for k, ref in enumerate(refs):
            same_ep = refs[1 - k]
            b = ["std121", "cond_windows", "split_entry", "caption", "audit", "owner_review", "sidedness_assumed"]
            pend.append(pending_row("G-zs-same", same_ep, cls, sided, ref, cls, "heldout", "test", "train", "zero_shot", "same", b))
            for clip in v2.rotate([c for c in corpus.eval_endpoints(sided) if prompts.clip_class(c) not in corpus.held_out],
                                  i * 3 + k, 1):
                pend.append(pending_row("G-zs-cross", clip, cls, sided, ref, prompts.clip_class(clip),
                                        corpus.endpoint_source(clip), corpus.band(clip), "train", "zero_shot", "cross",
                                        ["std121", "cond_windows", "split_entry", "owner_review", "sidedness_assumed"]))
            pend.append(pending_row("G-zs-foreign", reserve.pick(sided), cls, sided, ref, "davis", "humanvid",
                                    "foreign", "train", "zero_shot", "foreign",
                                    ["std121", "split_entry", "owner_review", "sidedness_assumed", "caption", "audit", "cond_windows", "roster_entry"]))

    # ---------------- 8. Tier 3: EffectData blocks
    ed_cfg = cfg["effectdata"]
    ed = EffectData(ed_cfg, probe=probe)
    blocks = []
    eligible = {e for e in ed.clips if ed.eligible_effect(e)}
    outside = sorted(s for s, es in ed.subj.items() if s not in ed.selected_subjects and len(es & eligible) >= 2)
    used_eff: set[str] = set()
    used_subj: set[str] = set()
    used_ref: set[str] = set()
    pref = ed_cfg["clip_prefix"]
    cursor_cross = 0

    def enough_portrait(effect: str) -> bool:
        return len(ed.portrait_clips(effect)) >= ed_cfg["min_portrait_ok_clips"]

    def reference_for(effect: str, avoid: set[str]) -> tuple[str, str] | None:
        for fn, s, t in ed.portrait_clips(effect):
            if s in ed.selected_subjects or s in avoid or fn in used_ref:
                continue
            return fn, s
        return None

    for s in outside:
        if len(blocks) >= ed_cfg["blocks"]:
            break
        if s in used_subj or not ed.subject_portrait_ok(s):
            continue
        cands = sorted(e for e in ed.subj[s] & eligible if e not in used_eff)
        e1 = next((e for e in cands if enough_portrait(e)), None)
        if e1 is None:
            continue
        e2 = next((e for e in cands if e != e1 and ed.category(e) != ed.category(e1) and enough_portrait(e)), None)
        if e2 is None:
            continue
        # cross subject: outside, portrait, under NEITHER effect (checked over ALL its effects)
        s_cross = None
        while cursor_cross < len(outside):
            c = outside[cursor_cross]
            cursor_cross += 1
            if c != s and c not in used_subj and not ({e1, e2} & ed.subj[c]) and ed.subject_portrait_ok(c):
                s_cross = c
                break
        if s_cross is None:
            break
        refs = {}
        for e in (e1, e2):
            r = reference_for(e, {s, s_cross})
            if r is None:
                break
            refs[e] = r
        if len(refs) < 2:
            continue
        used_eff |= {e1, e2}
        used_subj |= {s, s_cross}
        for e in (e1, e2):
            used_ref.add(refs[e][0])
        h = reserve.one()
        blocks.append({"e1": e1, "e2": e2, "cat": [ed.category(e1), ed.category(e2)], "same": s, "cross": s_cross,
                       "foreign": h, "refs": {e: refs[e][0] for e in (e1, e2)}})
        for e in (e1, e2):
            donor = f"{pref}.{e}"
            ref_std = ed.std_stem(pref, refs[e][0])
            gt = next(fn for fn, sub, t in ed.clips[e] if sub == s)
            cross_clip = next(fn for fn, sub, t in ed.clips[sorted(ed.subj[s_cross])[0]] if sub == s_cross)
            pool = [ed.std_stem(pref, fn) for fn, sub, t in ed.portrait_clips(e) if fn not in (refs[e][0], gt)][:ed_cfg["pool_size"]]
            b = ["std121", "cond_windows", "split_entry", "corpus_manifest", "instrument_reference"]
            pend.append(pending_row("G-zs-same", ed.std_stem(pref, gt), donor, "one", ref_std, donor, "effectdata", "test",
                                    "train", "zero_shot", "same", b + ["caption", "audit"]))
            pend.append(pending_row("G-zs-cross", ed.std_stem(pref, cross_clip), donor, "one", ref_std,
                                    f"{pref}.{sorted(ed.subj[s_cross])[0]}", "effectdata", "test", "train", "zero_shot", "cross",
                                    b + ["caption", "audit"]))
            pend.append(pending_row("G-zs-foreign", h, donor, "one", ref_std, "davis", "humanvid", "foreign", "train",
                                    "zero_shot", "foreign", b + ["caption", "audit", "roster_entry"]))
            blocks[-1].setdefault("pools", {})[e] = pool
    ed.save()

    # ---------------- 9. base twins for everything renderable (identical rule to build_registry)
    treat = old + new_a
    seen: set[str] = set()
    base_rows = []
    for r in treat:
        if r["input_key"] in seen:
            continue
        seen.add(r["input_key"])
        b = dict(r)
        b["arm"] = "base"
        b["cell"] = f"base:{r['cell']}"
        b["item_id"] = f"base__{r['input_key']}"
        base_rows.append(b)

    ids = [r["item_id"] for r in treat + pend]
    assert len(ids) == len(set(ids)), "item_id collision across v3 rows"
    return old, new_a, base_rows, pend, blocks, flags, ed, reserve


# --------------------------------------------------------------------------- health
def ceiling_bands(classes: list[str]) -> dict[str, dict]:
    """Bootstrap the per-class ceiling at its own n from the certified v4 matrix (existing corpus only)."""
    import numpy as np
    z = np.load(NPZ_V4, allow_pickle=True)
    S = 1.0 - z["m1a_S3"]
    names = [str(x) for x in z["keys"]]
    idx = collections.defaultdict(list)
    for i, n in enumerate(names):
        idx[n.split("/")[0]].append(i)
    rng = np.random.default_rng(0)
    out = {}
    for c in classes:
        ii = idx.get(c, [])
        n = len(ii)
        if n < 3:
            out[c] = {"n": n, "ceiling": None, "band90_pct": None}
            continue
        full = float(S[np.ix_(ii, ii)][~np.eye(n, dtype=bool)].mean())
        vals = []
        arr = np.array(ii)
        for _ in range(1000):
            s = rng.choice(arr, size=n, replace=True)
            m = S[np.ix_(s, s)]
            distinct = s[:, None] != s[None, :]          # a clip resampled twice must not pair with itself
            if distinct.sum() == 0:
                continue
            vals.append(m[distinct].mean())
        lo, hi = np.percentile(vals, [5, 95])
        out[c] = {"n": n, "ceiling": round(full, 4), "band90_pct": [round((lo / full - 1) * 100, 1), round((hi / full - 1) * 100, 1)]}
    return out


def health(cfg, corpus, old, new_a, base_rows, pend, blocks, flags, ed, split_classes) -> tuple[str, dict]:
    tiers = {"seen": ["G-fit", "G-memo-probe"], "unseen": ["G-unseen-same", "G-unseen-cross", "G-unseen-foreign"],
             "zero-shot": ["G-zs-same", "G-zs-cross", "G-zs-foreign"]}
    cells = ["G-fit", "G-memo-probe", "G-unseen-same", "G-unseen-cross", "G-unseen-foreign", "G-zs-same", "G-zs-cross", "G-zs-foreign"]

    def n(rows, cell, src=None):
        return sum(1 for r in rows if r["cell"] == cell and (src is None or r["endpoint_source"] == src or
                                                              (src == "higgsfield" and r["endpoint_source"] not in ("effectdata", "humanvid"))))
    H = {}
    lines = ["# Grid v3 — HEALTH (generated by eval_ladder/build_registry_v3.py, %s)" % date.today().isoformat(), ""]
    lines += ["## 1. Cells: rows now (registry_v3.jsonl) vs pending (registry_v3_pending.jsonl)", "",
              "| cell | v2 kept | new, renderable now | pending | total | of which EffectData | of which two-sided |", "|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for c in cells:
        a, b, p = n(old, c), n(new_a, c), n(pend, c)
        edn = sum(1 for r in pend if r["cell"] == c and (r["endpoint_source"] == "effectdata" or r["donor_class"].startswith(cfg["effectdata"]["clip_prefix"] + ".")))
        two = sum(1 for r in old + new_a + pend if r["cell"] == c and r["sided"] == "two")
        H[c] = {"v2_kept": a, "new_now": b, "pending": p, "total": a + b + p, "effectdata": edn, "two_sided": two}
        tot.update({"a": a, "b": b, "p": p, "ed": edn, "two": two})
        lines.append(f"| {c} | {a} | {b} | {p} | {a + b + p} | {edn} | {two} |")
    lines.append(f"| **total** | {tot['a']} | {tot['b']} | {tot['p']} | {tot['a'] + tot['b'] + tot['p']} | {tot['ed']} | {tot['two']} |")
    lines += ["", f"base twins now: {len(base_rows)} (one per distinct input of the renderable rows). "
                  f"Generations at 2 seeds: renderable now {(len(old) + len(new_a)) * 2}, full grid {(len(old) + len(new_a) + len(pend)) * 2}."]

    # per-tier resolving power
    lines += ["", "## 2. Resolving power per tier (from the measured per-generation seed SD of %.1f pp)" % SEED_SD_PP, "",
              "| tier | rows (full) | gens at 2 seeds | ±95 % on an arm delta | same-content rows |", "|---|---|---|---|---|"]
    H["tiers"] = {}
    allrows = old + new_a + pend
    for t, cs in tiers.items():
        rows = [r for r in allrows if r["cell"] in cs]
        g = 2 * len(rows)
        ci = 1.96 * SEED_SD_PP * (2 ** 0.5) / (g ** 0.5) if g else float("nan")
        same = sum(1 for r in rows if r["content"] == "same")
        H["tiers"][t] = {"rows": len(rows), "gens": g, "ci95_delta_pp": round(ci, 2), "same_rows": same}
        lines.append(f"| {t} | {len(rows)} | {g} | ±{ci:.1f} pp | {same} |")
    g = 2 * len(allrows)
    lines.append(f"| all | {len(allrows)} | {g} | ±{1.96 * SEED_SD_PP * 2 ** 0.5 / g ** 0.5:.1f} pp | {sum(1 for r in allrows if r['content'] == 'same')} |")

    # per-class yardstick support (Higgsfield)
    man = json.loads((REPO_ROOT / "data/processed/transitions_std121/corpus_manifest.json").read_text())
    corpus_n = collections.Counter(c["class"] for c in man["clips"].values())
    add = collections.Counter()
    for k in ("tier1_topups", "tier1_reference_only", "flame_additions", "zs_pool_topups"):
        for cls, stems in cfg[k].items():
            add[cls] += len(stems)
    for cls in cfg["new_zero_shot_classes"]:
        add[cls] += len(list((REPO_ROOT / cfg["raw_root"] / cls).glob("*.mp4")))   # all passed the res/duration filter at build
    donors = sorted({r["donor_class"] for r in allrows if not r["donor_class"].startswith(cfg["effectdata"]["clip_prefix"] + ".")})
    bands = ceiling_bands([c for c in donors if c in corpus_n])
    lines += ["", "## 3. Higgsfield donor classes: yardstick support", "",
              "| class | tier | corpus clips | additions | after | same rows | pool on same rows (after) | ceiling pairs (after) | ceiling 90 % band now | sided | flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    H["classes"] = {}
    for cls in donors:
        now = corpus_n.get(cls, 0)
        after = now + add.get(cls, 0)
        same = sum(1 for r in allrows if r["donor_class"] == cls and r["content"] == "same")
        tier = "zero-shot" if cls in corpus.held_out or cls in cfg["new_zero_shot_classes"] else "unseen"
        sided = corpus.sided.get(cls, cfg["new_zero_shot_classes"].get(cls, "?"))
        fl = []
        if after < 5:
            fl.append("thin")
        if cls in cfg["mid_effect_anchor"]["classes"]:
            fl.append("mid-effect anchors: reference only")
        if cls in cfg["new_zero_shot_classes"]:
            fl.append("PROVISIONAL class, sidedness assumed")
        if add.get(cls):
            fl.append("owner review of raw clips")
        bd = bands.get(cls, {}).get("band90_pct")
        if bd and max(abs(bd[0]), abs(bd[1])) > 5:
            fl.append("ceiling band > ±5 %")
        H["classes"][cls] = {"tier": tier, "corpus": now, "additions": add.get(cls, 0), "after": after, "same_rows": same,
                             "pool_same": max(min(after - 2, 8), 0), "ceiling_pairs": after * (after - 1) // 2,
                             "ceiling_band90_pct": bd, "sided": sided, "flags": fl}
        lines.append(f"| {cls} | {tier} | {now} | +{add.get(cls, 0)} | {after} | {same} | {max(min(after - 2, 8), 0)} | {after * (after - 1) // 2} | "
                     f"{'' if not bd else f'[{bd[0]:+.1f}, {bd[1]:+.1f}] %'} | {sided} | {'; '.join(fl)} |")

    # EffectData blocks
    lines += ["", "## 4. EffectData blocks (Tier 3)", "",
              "| block | e1 | cat | e2 | cat | same subject | cross subject | foreign | portrait-ok clips e1/e2 | pool | ceiling pairs e1/e2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    H["effectdata_blocks"] = blocks
    for k, b in enumerate(blocks, 1):
        p1, p2 = len(ed.portrait_clips(b["e1"])), len(ed.portrait_clips(b["e2"]))
        lines.append(f"| {k} | {b['e1']} | {b['cat'][0]} | {b['e2']} | {b['cat'][1]} | {b['same']} | {b['cross']} | {b['foreign']} | "
                     f"{p1}/{p2} | {cfg['effectdata']['pool_size']} | {p1 * (p1 - 1) // 2}/{p2 * (p2 - 1) // 2} |")
    lines.append(f"\nprobed EffectData clips (orientation cache): {len(ed.shapes)}; blocks built: {len(blocks)}/{cfg['effectdata']['blocks']}.")

    # blockers
    blk = collections.Counter()
    for r in pend:
        for b in r["pending"]:
            blk[b] += 1
    lines += ["", "## 5. Pending work (rows blocked, by blocker; one row can carry several)", "", "| blocker | rows |", "|---|---|"]
    for b, c in blk.most_common():
        lines.append(f"| {b} | {c} |")
    ep_src = {}
    for r in pend:
        if "caption" in r["pending"]:
            ep_src[r["endpoint"]] = "effectdata" if r["endpoint_source"] == "effectdata" else (
                "reserve" if r["endpoint_source"] == "humanvid" else "higgsfield")
    cnt = collections.Counter(ep_src.values())
    lines.append(f"\nendpoints needing caption + audit: {len(ep_src)} "
                 f"(reserve {cnt['reserve']}, effectdata {cnt['effectdata']}, higgsfield {cnt['higgsfield']}).")
    review = []
    for k in ("tier1_topups", "tier1_reference_only", "flame_additions", "zs_pool_topups"):
        for cls, stems in cfg[k].items():
            review += [f"{cls}: {s}" for s in stems]
    lines += ["", "Raw Higgsfield clips awaiting the owner's watch-through (the corpus curation left them out; reason unrecorded):",
              ""] + [f"- {x}" for x in review] + [
              "", f"Provisional zero-shot classes awaiting the watch-through (sidedness assumed one): "
              f"{', '.join(sorted(cfg['new_zero_shot_classes']))}."]
    H["blockers"] = dict(blk)
    H["flags_on_kept_rows"] = dict(flags)
    lines += ["", "## 6. Flags on kept v2 rows", ""] + [f"- `{k}`: {', '.join(v)}" for k, v in sorted(flags.items())]
    lines += ["", "## 7. Seatbelts", "", "- v2 `seatbelts()` PASSED on registry_v3.jsonl (old + new + base twins).",
              "- old ic_gen rows byte-identical to registry.jsonl (asserted).",
              "- pending rows: unique item_id (asserted); EffectData endpoints/references outside the S6 selection, portrait-only, "
              "effect pairs category-disjoint (by construction).",
              "- NOT yet checkable for pending rows: rendered prompt, audited endpoint, conditioning windows, split membership — "
              "these become asserts when the rows move to registry_v3.jsonl."]
    return "\n".join(lines) + "\n", H


# --------------------------------------------------------------------------- staging shelf (seed_prompts.py format)
ARM_FIELDS = {"arm", "item_id", "video_key", "conditioning", "use_reference", "no_twin", "code_source_reference"}


def stage_family(rows: list[dict], dirname: str, grammar: str, role: str, src: Path) -> str:
    def key(r):
        return (r["cell"], r["endpoint"], r.get("reference") or "", r["sided"])
    canon = [{k: v for k, v in r.items() if k not in ARM_FIELDS} for r in rows]
    uniq = {key(r): r for r in canon}
    sha = hashlib.sha256("".join(r["prompt"] for r in sorted(uniq.values(), key=key)).encode()).hexdigest()[:12]
    d = STAGING / dirname
    d.mkdir(parents=True, exist_ok=True)
    with (d / "grid.jsonl").open("w") as f:
        for r in sorted(canon, key=key):
            f.write(json.dumps(r, sort_keys=True) + "\n")
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    (d / "meta.yaml").write_text(f"""id: prompts/{dirname}
seq: {int(dirname[:3])}
shelf: prompts
created: {date.today().isoformat()}
family: A
grammar: "{grammar}"
role: {role}
rows: {len(canon)}
prompt_corpus_sha: {sha}   # sha256[:12] over prompts concatenated in (cell,endpoint,reference,sided) order, unique items
renderer: eval_ladder/prompts.py::render_prompt (single source; families B/D/refvfx splice clauses from misc/refvfx_baseline/reference_effects.json)
derived_from: {src.relative_to(REPO_ROOT)} (sha256[:12] {src_sha}; design 3.0.0 superset of prompts/001_ctt152_neutral)
source: eval_ladder/build_registry_v3.py
notes: rows are ARM-FREE — arm/item_id/video_key + arm-contract fields stripped; stamp_rows.py re-adds them per arm. STAGED under misc/ until registered (store/README.md §5).
""")
    return sha


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--no-probe", action="store_true", help="do not open EffectData zips (use the shape cache only)")
    args = ap.parse_args()

    cfg = yaml.safe_load(GRID.read_text())
    split, arms, inv = v2.load()
    token = arms["token"]
    corpus = v2.Corpus(split)
    old, new_a, base_rows, pend, blocks, flags, ed, reserve = build(cfg, corpus, token, inv, probe=not args.no_probe)

    rows_now = old + new_a + base_rows
    v2.seatbelts(rows_now, corpus, inv, token)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows_now))
    OUT_PENDING.write_text("".join(json.dumps(r) + "\n" for r in pend))

    md, H = health(cfg, corpus, old, new_a, base_rows, pend, blocks, flags, ed, split["classes"])
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    (HEALTH_DIR / "HEALTH.md").write_text(md)
    (HEALTH_DIR / "health.json").write_text(json.dumps(H, indent=1, sort_keys=True))
    sha = stage_family(old + new_a, "009_ctt_v3_neutral", "{S1}. sksz. [{S2}.]",
                       "neutral for adapter arms (grid v3, renderable subset)", OUT)

    n_seeds = 2   # claim seeds 42/43 (arms.yaml lists 8 for the noise-floor study)
    print(f"[registry_v3] {len(old)} v2 rows kept + {len(new_a)} new renderable + {len(base_rows)} base twins -> {OUT.relative_to(REPO_ROOT)}")
    print(f"[registry_v3] {len(pend)} pending rows -> {OUT_PENDING.relative_to(REPO_ROOT)}  (EffectData blocks {len(blocks)})")
    print(f"[registry_v3] full grid = {len(old) + len(new_a) + len(pend)} treatment rows x {n_seeds} seeds = {(len(old) + len(new_a) + len(pend)) * n_seeds} gens")
    print(f"[registry_v3] staged family 009_ctt_v3_neutral prompt_sha={sha} ({len(old) + len(new_a)} rows)")
    print(f"[registry_v3] health -> {HEALTH_DIR.relative_to(REPO_ROOT)}/HEALTH.md")
    if args.stats:
        by = collections.Counter((r["cell"], "now") for r in old + new_a) + collections.Counter((r["cell"], "pending") for r in pend)
        for k in sorted(by):
            print(f"  {k[0]:18s} {k[1]:8s} {by[k]:4d}")


if __name__ == "__main__":
    main()
