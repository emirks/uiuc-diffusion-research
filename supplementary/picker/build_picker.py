#!/usr/bin/env python3
"""Build the SEGUE supplementary PICKER — an INTERNAL page that shows, for every row in
the owner's four collections, ALL candidate generations (neutral + effect) in a FIXED
per-section column grid, and lets the owner pick the best "ours" clip per row.

Run with the login-node python3.12 (login python3 is 3.6):

    /usr/bin/python3.12 supplementary/picker/build_picker.py            # == --html
    /usr/bin/python3.12 supplementary/picker/build_picker.py --check    # verify every ref resolves

Unlike the public supplementary (supplementary/build_site.py), this page does NOT transcode:
it references the ORIGINAL clips through relative symlinks placed in picker/site/
(store -> ../../../store, outputs, data, eval_ladder), exactly as a viewer mount does, so a
repo-relative path like store/gens/... or outputs/videos/... or eval_ladder/conds/... resolves
next to index.html. collections.json (read-only display source) and picks.json (the writable
picks file, realpath inside eval_ladder/viewer/collections/ so the static server accepts a POST
to it) are symlinked in too.

Data model (verified): eval_ladder/viewer/collections/neutral_effect_collections.json ->
collections[] with ids teg_user_study (24, section A), vfx_transfer_user_study (26, B),
supplementary_teg (54, C), supplementary_vfx_transfer (224, D). Each row's CANDIDATES come from
(a) every present columns[] entry (gens[] seeds 42/43), excluding EXCLUDE_ARMS, plus (b) our
EFFECT-prompt generations pulled straight from the STORE by filename (the new w=1.5/w=3 effect
completions are not in the snapshot or data.js). A column/arm is "ours" (pickable) unless its
category is a prior-work category (external / teg). Every clip is uniquely identified by its
VIDEO PATH — the pick highlight, the row-header label and the saved record all key on the video,
never on the row-shared grid item_id (data-gen).

Layout: every row in a section renders the SAME fixed column grid — one column per (arm, variant)
slot, in a fixed global order (own first, then prior work); the slot list per section is the union
of slots that occur in that section. A slot with no clip for a row renders an explicit "absent"
placeholder so the correspondence lines up. Both seeds sit side by side inside a slot.
"""
import argparse
import datetime
import html
import json
import os
import re
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))        # supplementary/picker
REPO = os.path.dirname(os.path.dirname(HERE))            # repo root
SITE = os.path.join(HERE, "site")
TEMPLATE = os.path.join(HERE, "template.html")
INDEX = os.path.join(SITE, "index.html")

COLLECTIONS = os.path.join(REPO, "eval_ladder/viewer/collections/neutral_effect_collections.json")
PICKS = os.path.join(REPO, "eval_ladder/viewer/collections/supplementary_picks.json")

# prior-work categories (NOT pickable). Everything else is "ours".
PRIOR_CATS = {"external", "teg"}

# owner 2026-09-24 ("i'll never choose base ltx 2, remove from all"): drop these arm ids
# (category "baseline", labels "Base LTX-2 …") from EVERY candidate list in the picker, incl.
# the A/B prior-work context. Keyed on the column's arm id, not its (localizable) label.
EXCLUDE_ARMS = {"base_cond", "base_prompt", "ic_gen"}  # ic_gen = Plain LoRA, dropped per owner

# owner 2026-09-24 ("add effect versions where there is the generation for effect"): our
# EFFECT-prompt generations, pulled from the STORE by filename (os.path.exists per candidate;
# the partial w1.5/w3 entries have no grid.jsonl). (arm, arm_label, store gens dir). For each
# row we try the HF subentry 05_effect_v3__dai (harness_arm <arm>_effect_v3) and the EffectData
# subentry 06_effect_v3ed81__dai (<arm>_effect_v3ed81); the ref clip prefix "ed." selects which.
EFFECT_ARMS = [
    ("dualforce_control",  "SEGUE w/o NRG", "013_dualforce_control"),
    ("dualforce_dcg_w1p5", "SEGUE (w=1.5)", "030_dualforce_dcg_w1p5"),
    ("dualforce_dcg_w3",   "SEGUE (w=3)",   "031_dualforce_dcg_w3"),
    ("dualforce_dcg_w6",   "SEGUE (w=6)",   "032_dualforce_dcg_w6"),
]

# fixed global column order (own slots first); slots not present in a section are skipped, other
# own arms found in the snapshots (ctt_v2 …) follow in data order, then prior-work in data order.
FIXED_SLOT_ORDER = [
    ("dualforce_dcg_w6", "neutral"), ("dualforce_dcg_w3", "neutral"),
    ("dualforce_dcg_w1p5", "neutral"), ("dualforce_control", "neutral"), ("ctt_v2", "neutral"),
    ("dualforce_dcg_w6", "effect"), ("dualforce_dcg_w3", "effect"),
    ("dualforce_dcg_w1p5", "effect"), ("dualforce_control", "effect"), ("ctt_v2", "effect"),
]

# owner 2026-09-24 (bands layout): columns are ARMS; each own-arm column stacks its EFFECT band
# (top) directly over its NEUTRAL band (bottom). Own arms in this fixed order, then prior-work arms.
FIXED_ARM_ORDER = ["dualforce_dcg_w6", "dualforce_dcg_w3", "dualforce_dcg_w1p5",
                   "dualforce_control", "ctt_v2"]

# relative symlinks placed in picker/site/ (target -> is relative so the tree is relocatable
# together with the repo); repo-relative media paths then resolve next to index.html.
SYMLINKS = {
    "store": "../../../store",
    "outputs": "../../../outputs",
    "data": "../../../data",
    "eval_ladder": "../../../eval_ladder",   # start/end endpoint clips live in eval_ladder/conds/
    "collections.json": "../../../eval_ladder/viewer/collections/neutral_effect_collections.json",
    "picks.json": "../../../eval_ladder/viewer/collections/supplementary_picks.json",
}

SECTIONS = [
    ("A", "teg_user_study",
     "A — TEG comparisons (both endpoints given)",
     "All 24 judge win-all rows of collection teg_user_study PLUS the 32 rows the judge filter dropped (tagged), one fixed column grid. Own arms first (neutral "
     "then effect), then the prior-work TEG opponents (refVFX + Wan2.1 baselines, category teg; "
     "greyed, not pickable). Base LTX-2 (base_cond/base_prompt) is removed per owner request. "
     "Effect-prompt clips come straight from the store. 'absent' = no generation for that slot."),
    ("B", "vfx_transfer_user_study",
     "B — VFX-transfer comparisons (start frame given)",
     "All 26 judge win-all rows of collection vfx_transfer_user_study PLUS the 128 rows the judge filter dropped (tagged), one fixed column grid. Own arms first, "
     "then prior-work (VAP / VFXMaster / refVFX; greyed, not pickable)."),
    ("C", "supplementary_teg",
     "C — Supplementary TEG results",
     "All 54 rows of collection supplementary_teg (including the one row with no w=6 generation). "
     "Two endpoints given. One fixed column grid; 'absent' placeholders where a slot has no clip."),
    ("D", "supplementary_vfx_transfer",
     "D — Supplementary VFX-transfer results",
     "All 224 rows of collection supplementary_vfx_transfer. Start frame given (no end endpoint). "
     "One fixed column grid; 'absent' placeholders where a slot has no clip."),
]


# --------------------------------------------------------------------------- utils
def apath(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def esc(s):
    return html.escape("" if s is None else str(s), quote=True)


def now_iso():
    t = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    return t.isoformat().replace("+00:00", "Z")


# 2026-09-25 (owner: "add the rows that didnt win in the vlm judge too. i want to see all"): the comparison
# collections A/B were filtered on 2026-09-24 to the VLM-judge win-all rows; the rows dropped by that filter are
# taken back from the pre-filter snapshot and appended to A/B (tagged, not in the live collections file).
SNAPSHOT_PREFILTER = os.path.join(REPO, "eval_ladder/viewer/collections/snapshots/"
                                  "neutral_effect_collections.2026-09-24T05-12Z_after_split_before_judge_filter.json")
EXTRA_FROM_SNAPSHOT = ("teg_user_study", "vfx_transfer_user_study")


def load_collections():
    d = json.load(open(COLLECTIONS))
    cols = {c["id"]: c for c in d["collections"]}
    if os.path.exists(SNAPSHOT_PREFILTER):
        snap = {c["id"]: c for c in json.load(open(SNAPSHOT_PREFILTER))["collections"]}
        for cid in EXTRA_FROM_SNAPSHOT:
            if cid not in cols or cid not in snap:
                continue
            live_ids = {it["id"] for it in cols[cid]["items"]}
            extra = [dict(it, _extra=True) for it in snap[cid]["items"] if it["id"] not in live_ids]
            cols[cid] = dict(cols[cid], items=list(cols[cid]["items"]) + extra)
            print("[collections] %s: %d live (judge win-all) + %d from the pre-filter snapshot"
                  % (cid, len(live_ids), len(extra)))
    # 2026-09-25 (owner): rows that share a card across collections pool their candidate columns. Each
    # item only carries the columns that were VISIBLE when it was added, so an older comparison row may
    # lack arms (w=1.5 neutral, w/o NRG ...) that its supplementary twin has; every twin now gets the
    # union (deduped by arm/variant/tier; row_slots dedups clips by path). In-memory only.
    from collections import defaultdict
    sib = defaultdict(list)
    for c in cols.values():
        for it in c["items"]:
            sib[it["card_key"]].append(it)
    pooled = 0
    for key, its in sib.items():
        if len(its) < 2:
            continue
        merged, seen = [], set()
        for it in its:
            for col in it.get("columns", []):
                if not col.get("present"):
                    continue
                k = (col.get("arm"), col.get("variant") or "", col.get("tier") or "")
                if k in seen:
                    continue
                seen.add(k); merged.append(col)
        for it in its:
            if len([c for c in it.get("columns", []) if c.get("present")]) != len(merged):
                pooled += 1
            it["columns"] = merged
    print("[collections] %d rows gained candidate columns from a twin row in another collection" % pooled)
    return cols


def ensure_symlinks():
    os.makedirs(SITE, exist_ok=True)
    made = []
    for name, target in SYMLINKS.items():
        link = os.path.join(SITE, name)
        if os.path.islink(link):
            if os.readlink(link) != target:
                os.remove(link)
                os.symlink(target, link)
                made.append("relinked %s -> %s" % (name, target))
        elif os.path.exists(link):
            raise SystemExit("refusing to overwrite non-symlink %s" % link)
        else:
            os.symlink(target, link)
            made.append("linked %s -> %s" % (name, target))
    return made


def seed_picks():
    """Create the writable picks file seeded empty if it does not exist yet."""
    if os.path.exists(PICKS):
        return False
    doc = {"schema": 1, "updated": now_iso(), "picks": {}}
    with open(PICKS, "w") as f:
        f.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return True


# --------------------------------------------------------------------------- row model
def ref_clip_of(it):
    """The row's reference clip: ref= suffix if present, else inputs.refs[0].clip,
    else the __ref_<clip> token in a gen id."""
    ck = it.get("card_key", "")
    if "|ref=" in ck:
        return ck.split("|ref=", 1)[1]
    refs = it.get("inputs", {}).get("refs") or []
    if refs:
        return refs[0].get("clip")
    for col in it.get("columns", []):
        for g in (col.get("gens") or []):
            m = re.search(r"__ref_(.+?)(?:__s\d+)?$", g.get("id") or "")
            if m:
                return m.group(1)
    return None


def ref_of(it):
    """(ref_clip, ref_class, ref_video) for the row."""
    rc = ref_clip_of(it)
    refs = it.get("inputs", {}).get("refs") or []
    match = None
    for r in refs:
        if r.get("clip") == rc:
            match = r
            break
    if match is None and refs:
        match = refs[0]
    if match is None:
        return rc, "", None
    return rc, match.get("cls") or "", match.get("video")


def _cell_of(it):
    """The grid 'cell' token (e.g. G-zs-foreign) — from inputs.tier.cells[0], else a gen id."""
    tier = it.get("inputs", {}).get("tier") or {}
    cs = tier.get("cells") or []
    if cs:
        return cs[0]
    for col in it.get("columns", []):
        for g in (col.get("gens") or []):
            gid = g.get("id") or ""
            if "__" in gid:
                return gid.split("__", 1)[0]
    return None


_SLOT_CACHE = {}
_EFFECT_HITS = {}   # (section arm) hit counters filled during a build, for the report


def row_slots(it):
    """OrderedDict (arm,variant) -> slot dict {arm, variant, arm_label, category, pickable,
    category_label, clips:[{seed, video, gen_id, pct, pickable}], source}. Every clip is deduped
    by basename so the same file never appears twice (snapshot outputs/ path and store/ path share
    a basename), which guarantees the per-row video-path key is unique."""
    k = id(it)
    if k in _SLOT_CACHE:
        return _SLOT_CACHE[k]
    slots = OrderedDict()
    seen_paths = set()          # dedup snapshot clips by full path (ctt_v2 neutral/effect share a
    seen_base_snap = set()      # basename in different dirs, so basename dedup would drop one)

    # (a) snapshot present columns (excluding EXCLUDE_ARMS). When both a non-regen and a *_regen
    # tier are present for the same (arm, variant) (only ctt_v2 does this), prefer the non-regen.
    present_cols = [c for c in it.get("columns", [])
                    if c.get("present") and c.get("arm") not in EXCLUDE_ARMS]
    has_nonregen = set()
    for c in present_cols:
        if "regen" not in (c.get("tier") or ""):
            has_nonregen.add((c.get("arm"), c.get("variant") or ""))
    for col in present_cols:
        if "regen" in (col.get("tier") or "") \
                and (col.get("arm"), col.get("variant") or "") in has_nonregen:
            continue
        arm = col.get("arm")
        var = col.get("variant") or ""
        cat = col.get("category")
        pick = cat not in PRIOR_CATS
        sk = (arm, var)
        slot = slots.get(sk)
        if slot is None:
            slot = {"arm": arm, "variant": var, "arm_label": col.get("arm_label") or arm,
                    "category": cat, "pickable": pick,
                    "category_label": col.get("category_label") or "", "clips": [],
                    "source": "snapshot"}
            slots[sk] = slot
        for g in sorted(col.get("gens") or [], key=lambda g: str(g.get("seed"))):
            v = g.get("video")
            if not v or v in seen_paths:
                continue
            seen_paths.add(v)
            seen_base_snap.add(os.path.basename(v))
            slot["clips"].append({"seed": g.get("seed"), "video": v, "gen_id": g.get("id"),
                                  "pct": g.get("pct"), "pickable": pick})

    # (b) EFFECT-prompt clips straight from the store (dedup by basename against the snapshot)
    cell = _cell_of(it)
    ep = it.get("inputs", {}).get("endpoint")
    rc = ref_clip_of(it)
    if cell and ep and rc:
        ed = str(rc).startswith("ed.")
        suffix = "effect_v3ed81" if ed else "effect_v3"
        sub = "06_effect_v3ed81__dai" if ed else "05_effect_v3__dai"
        for arm, label, sdir in EFFECT_ARMS:
            clips = []
            for seed in ("42", "43"):
                fn = "%s__%s_%s__%s__ref_%s__s%s.mp4" % (cell, arm, suffix, ep, rc, seed)
                if fn in seen_base_snap:          # already carried by the snapshot -> no duplicate
                    continue
                rel = "store/gens/%s/%s/videos/%s" % (sdir, sub, fn)
                if rel in seen_paths or not os.path.exists(apath(rel)):
                    continue
                seen_paths.add(rel)
                seen_base_snap.add(fn)
                clips.append({"seed": int(seed), "video": rel,
                              "gen_id": "%s__%s__%s__ref_%s" % (cell, arm, ep, rc),
                              "pct": None, "pickable": True})
            if not clips:
                continue
            sk = (arm, "effect")
            slot = slots.get(sk)
            if slot is None:
                slot = {"arm": arm, "variant": "effect", "arm_label": label,
                        "category": "df_dcg_store", "pickable": True,
                        "category_label": "effect prompt (store)", "clips": [], "source": "store"}
                slots[sk] = slot
            slot["clips"].extend(clips)

    _SLOT_CACHE[k] = slots
    return slots


def candidates_of(it):
    """Flat list of candidate CLIP cells (no placeholders) — for counts and the uniqueness check."""
    out = []
    for _sk, slot in row_slots(it).items():
        for c in slot["clips"]:
            out.append({"arm": slot["arm"], "arm_label": slot["arm_label"],
                        "variant": slot["variant"], "category": slot["category"],
                        "category_label": slot["category_label"], "pickable": slot["pickable"],
                        "seed": c["seed"], "video": c["video"], "gen_id": c["gen_id"],
                        "pct": c["pct"]})
    return out


def section_slot_order(items):
    """Ordered [(slotkey, meta)] for a section: fixed own order first (only slots that occur),
    then other own arms in first-seen order, then prior-work in first-seen order."""
    meta = OrderedDict()
    for it in items:
        for sk, slot in row_slots(it).items():
            if sk not in meta:
                meta[sk] = {"arm": slot["arm"], "variant": slot["variant"],
                            "arm_label": slot["arm_label"], "pickable": slot["pickable"],
                            "category": slot["category"],
                            "category_label": slot["category_label"]}
    ordered, used = [], set()
    for sk in FIXED_SLOT_ORDER:
        if sk in meta:
            ordered.append(sk); used.add(sk)
    for sk in meta:                                   # other own (first-seen)
        if sk not in used and meta[sk]["pickable"]:
            ordered.append(sk); used.add(sk)
    for sk in meta:                                   # prior work (first-seen)
        if sk not in used and not meta[sk]["pickable"]:
            ordered.append(sk); used.add(sk)
    return [(sk, meta[sk]) for sk in ordered]


def section_columns(items):
    """(own_arms, prior_arms, arm_meta) for a section. Each arm is ONE column (own arms in the
    fixed order then data order; prior-work arms after). arm_meta[arm] = {arm_label, pickable,
    variants:set, prior_variant}."""
    arm_meta = OrderedDict()
    for it in items:
        for (arm, var), slot in row_slots(it).items():
            m = arm_meta.get(arm)
            if m is None:
                m = {"arm_label": slot["arm_label"], "pickable": slot["pickable"],
                     "variants": set(), "prior_variant": None}
                arm_meta[arm] = m
            m["variants"].add(var)
            if not slot["pickable"]:
                m["prior_variant"] = var
    own = [a for a in arm_meta if arm_meta[a]["pickable"]]
    prior = [a for a in arm_meta if not arm_meta[a]["pickable"]]
    own_ordered = [a for a in FIXED_ARM_ORDER if a in own] \
        + [a for a in own if a not in FIXED_ARM_ORDER]
    return own_ordered, prior, arm_meta


# --------------------------------------------------------------------------- html
def _video(src, badge=None, extra=""):
    """A lazy <video> whose <source> defers `src` to `data-src` — the VAP lazy-load contract:
    the IntersectionObserver assigns src only when the cell enters the viewport, so the network
    is not touched for the thousands of off-screen clips."""
    b = '<span class="seg-badge">%s</span>' % html.escape(badge) if badge else ""
    return ('<div class="seg-media %s">%s<video class="lazy-video" preload="none" muted '
            'playsinline loop data-autoplay="true">'
            '<source data-src="%s" type="video/mp4"/></video></div>'
            % (extra, b, esc(src)))


def _noend():
    return ('<div class="seg-media seg-noend"><span class="seg-badge">no end</span>'
            '<span class="seg-xlabel">no end given</span></div>')


def _unit(inner, label, extra=""):
    lab = '<div class="seg-label">%s</div>' % html.escape(label) if label else ""
    return '<div class="seg-unit %s">%s%s</div>' % (extra, inner, lab)


def _verdict_html(it):
    j = (it.get("judge") or {}).get("transition_vs") or {}
    if it.get("_extra"):
        return '<span class="pk-verdict pk-notwin">judge: not a win-all row (dropped by the 2026-09-24 filter; shown from the pre-filter snapshot)</span>'
    if not j:
        return ""
    parts = []
    for opp, v in j.items():
        if v == 1.0:
            tok, cls = "W", "w"
        elif v == 0.5:
            tok, cls = "T", "t"
        else:
            tok, cls = "L", "l"
        parts.append('vs %s <span class="%s">%s</span>' % (html.escape(opp), cls, tok))
    return '<span class="pk-verdict">judge: %s</span>' % " · ".join(parts)


def _cand_html(cid, item_id, card_key, slot, clip):
    """One pickable/context clip cell, uniquely keyed by its video path (data-video)."""
    pickable = slot["pickable"]
    var = slot["variant"]
    pct = ""
    if clip.get("pct") is not None:
        try:
            pct = ' <span class="pk-pct">%.2f</span>' % float(clip["pct"])
        except (TypeError, ValueError):
            pct = ""
    if not pickable:
        tag = '<span class="pk-priortag">prior work</span>'
    elif var == "effect":
        tag = '<span class="pk-efftag">effect prompt</span>'
    else:
        tag = ""
    cap = ('<div class="seg-label pk-caplabel">%s · %s · s%s%s%s</div>'
           % (html.escape(slot["arm_label"]), html.escape(var), html.escape(str(clip["seed"])),
              pct, tag))
    if pickable:
        foot = '<button class="pk-btn" type="button">Pick</button>'
        mark = '<span class="pk-pickedmark">✓ picked</span>'
    else:
        foot = ('<div class="pk-notpick">%s — context only</div>'
                % html.escape(slot.get("category_label") or "prior work"))
        mark = ""
    data = (' data-cid="%s" data-item="%s" data-cardkey="%s" data-gen="%s" data-video="%s"'
            ' data-arm="%s" data-armlabel="%s" data-variant="%s" data-seed="%s" data-pick="%s"'
            % (esc(cid), esc(item_id), esc(card_key or ""), esc(clip.get("gen_id") or ""),
               esc(clip["video"] or ""), esc(slot["arm"] or ""), esc(slot["arm_label"]),
               esc(var), esc(clip["seed"]), "1" if pickable else "0"))
    cls = "pk-cand seg-unit seg-gen" + ("" if pickable else " prior")
    return ('<div class="%s"%s>%s%s%s%s</div>'
            % (cls, data, mark, _video(clip["video"]), cap, foot))


def _band_cell(cid, item_id, card_key, meta, variant, slot, style):
    """One grid cell: the clip(s) for (arm,variant) in this row, or an 'absent' placeholder."""
    if slot and slot["clips"]:
        inner = "".join(_cand_html(cid, item_id, card_key, slot, c) for c in slot["clips"])
        return '<div class="pk-slot" style="%s">%s</div>' % (style, inner)
    return ('<div class="pk-slot pk-slot-absent" style="%s">'
            '<div class="seg-media seg-noend"><span class="seg-xlabel">absent</span></div>'
            '<div class="seg-label pk-caplabel pk-absentlabel">%s · %s · absent</div></div>'
            % (style, html.escape(meta["arm_label"]), html.escape(variant or "")))


def _colhead_html(meta):
    cls = "pk-colhead" + ("" if meta["pickable"] else " prior")
    return ('<div class="%s"><div class="pk-ch-arm">%s</div></div>'
            % (cls, html.escape(meta["arm_label"])))


def _header_html(own_arms, prior_arms, arm_meta, tmpl):
    heads = "".join(_colhead_html(arm_meta[a]) for a in own_arms + prior_arms)
    return ('<div class="pk-header" style="grid-template-columns:%s">'
            '<div class="pk-hb"></div><div class="pk-hb">reference</div>'
            '<div class="pk-hb">endpoints</div>%s</div>' % (tmpl, heads))


def _row_html(sec, cid, it, missing, own_arms, prior_arms, arm_meta, tmpl):
    item_id = it.get("id")
    card_key = it.get("card_key")
    inp = it.get("inputs", {})
    ep = inp.get("endpoint") or ""
    sided = inp.get("sided")
    rc, refclass, refvideo = ref_of(it)
    tier = inp.get("tier") or {}
    if tier:
        nov = (tier.get("novelty") or ["?"])[0]
        con = (tier.get("content") or ["?"])[0]
        tier_words = "%s · %s" % (nov, con)
    else:
        tier_words = ""

    def add(src, kind):
        if src and not os.path.exists(apath(src)):
            missing.append((sec, kind, src))
        return src

    head = ('<span class="pk-rowhead">%s %s&rarr; %s</span>'
            % (html.escape(refclass or rc or "?"),
               ("(%s) " % html.escape(rc) if rc else ""), html.escape(ep)))
    tierspan = '<span class="seg-tier">%s</span>' % html.escape(tier_words) if tier_words else ""
    verdict = _verdict_html(it)
    notes = it.get("notes") or ""
    notes_html = '<div class="pk-notes">note: %s</div>' % html.escape(notes) if notes.strip() else ""
    restart_btn = ('<button class="pk-restart" type="button" '
                   'title="Restart every visible clip in this row from frame 0, in sync">'
                   '&#8635; Restart all (synced)</button>')
    rowlabel = ('<div class="seg-rowlabel">%s%s%s<span class="pk-state">no pick</span>%s%s</div>'
                % (head, tierspan, verdict, restart_btn, notes_html))

    if refvideo:
        ref_unit = _unit(_video(add(refvideo, "reference")),
                         "reference: %s" % (refclass or rc or ""), "seg-ref")
    else:
        ref_unit = _unit('<div class="seg-media seg-noend"><span class="seg-xlabel">no reference'
                         '</span></div>', "reference", "seg-ref")
    prefix = inp.get("prefix_video")
    suffix = inp.get("suffix_video")
    start_div = (_video(add(prefix, "start"), badge="start", extra="seg-endpoint")
                 if prefix else _noend())
    end_div = (_video(add(suffix, "end"), badge="end", extra="seg-endpoint")
               if (sided == "two" and suffix) else _noend())
    stack = '<div class="seg-stack">%s%s</div>' % (start_div, end_div)
    end_unit = _unit(stack, "given endpoint(s)", "seg-endstack")

    rs = row_slots(it)
    parts = [rowlabel]
    # band labels (col1) + reference (col2) + endpoints (col3), both bands tall
    parts.append('<div class="pk-bandlabel" style="grid-column:1;grid-row:2">effect</div>')
    parts.append('<div class="pk-bandlabel" style="grid-column:1;grid-row:3">neutral</div>')
    parts.append('<div class="pk-io" style="grid-column:2;grid-row:2/4">%s</div>' % ref_unit)
    parts.append('<div class="pk-io" style="grid-column:3;grid-row:2/4">%s</div>' % end_unit)
    # own-arm columns: EFFECT on top (row 2), NEUTRAL below (row 3), aligned
    for i, arm in enumerate(own_arms):
        col = 4 + i
        meta = arm_meta[arm]
        parts.append(_band_cell(cid, item_id, card_key, meta, "effect", rs.get((arm, "effect")),
                                "grid-column:%d;grid-row:2" % col))
        parts.append(_band_cell(cid, item_id, card_key, meta, "neutral", rs.get((arm, "neutral")),
                                "grid-column:%d;grid-row:3" % col))
    # prior-work columns: one cell spanning both bands
    for j, arm in enumerate(prior_arms):
        col = 4 + len(own_arms) + j
        meta = arm_meta[arm]
        var = meta.get("prior_variant") or (sorted(meta["variants"])[0] if meta["variants"] else "")
        parts.append(_band_cell(cid, item_id, card_key, meta, var, rs.get((arm, var)),
                                "grid-column:%d;grid-row:2/4" % col))

    return ('<div class="picker-row" data-cid="%s" data-item="%s" data-cardkey="%s" '
            'style="grid-template-columns:%s">%s</div>'
            % (esc(cid), esc(item_id), esc(card_key or ""), tmpl, "".join(parts)))


def _grid_template(n_own, n_prior):
    L = "calc(var(--seg-h) * 0.34)"       # band-label column
    IO = "calc(var(--seg-h) * 0.82)"      # reference, endpoints
    C = "calc(var(--seg-h) * 1.6)"        # arm column (two seeds side by side, closed up)
    return "%s %s %s %s" % (L, IO, IO, " ".join([C] * (n_own + n_prior)))


def do_html():
    made = ensure_symlinks()
    created = seed_picks()
    cols = load_collections()
    tpl = open(TEMPLATE).read()
    assert tpl.count("{{SECTIONS}}") == 1, "template must have exactly one {{SECTIONS}}"

    missing = []
    parts = []
    counts = {}
    for sec, cid, title, desc in SECTIONS:
        items = cols[cid]["items"]
        for it in items:
            vids = [c["video"] for c in candidates_of(it)]
            if len(set(vids)) != len(vids):
                dup = sorted(v for v in set(vids) if vids.count(v) > 1)
                raise SystemExit("FAIL: row %s/%s has candidate cells sharing a video path "
                                 "(cells must be uniquely keyed by video): %s"
                                 % (cid, it.get("id"), dup))
        own_arms, prior_arms, arm_meta = section_columns(items)
        tmpl = _grid_template(len(own_arms), len(prior_arms))
        header = _header_html(own_arms, prior_arms, arm_meta, tmpl)
        rows = "".join(_row_html(sec, cid, it, missing, own_arms, prior_arms, arm_meta, tmpl)
                       for it in items)

        n_cand = sum(len(candidates_of(it)) for it in items)
        n_pick = sum(sum(1 for c in candidates_of(it) if c["pickable"]) for it in items)
        eff_real = neu_real = prior_real = ph = 0
        eff_by_arm = {}
        eff_from_store = {}
        ctt = {"neutral": 0, "effect": 0}
        for it in items:
            rs = row_slots(it)
            for arm in own_arms:
                e = rs.get((arm, "effect"))
                if e and e["clips"]:
                    eff_real += 1
                    eff_by_arm[arm] = eff_by_arm.get(arm, 0) + len(e["clips"])
                    if e.get("source") == "store":
                        eff_from_store[arm] = eff_from_store.get(arm, 0) + len(e["clips"])
                else:
                    ph += 1
                n = rs.get((arm, "neutral"))
                if n and n["clips"]:
                    neu_real += 1
                else:
                    ph += 1
                if arm == "ctt_v2":
                    if e and e["clips"]:
                        ctt["effect"] += 1
                    if n and n["clips"]:
                        ctt["neutral"] += 1
            for arm in prior_arms:
                var = arm_meta[arm].get("prior_variant") or ""
                p = rs.get((arm, var))
                if p and p["clips"]:
                    prior_real += 1
                else:
                    ph += 1
        counts[sec] = {"rows": len(items), "cand": n_cand, "own": n_pick, "prior": n_cand - n_pick,
                       "own_arms": own_arms, "prior_arms": prior_arms, "arm_meta": arm_meta,
                       "eff_real": eff_real, "neu_real": neu_real, "prior_real": prior_real,
                       "ph": ph, "eff_by_arm": eff_by_arm, "eff_from_store": eff_from_store,
                       "ctt": ctt}

        parts.append('<section class="pk-section" id="sec-%s">' % sec)
        parts.append('<div class="section-title">%s</div>' % html.escape(title))
        parts.append('<p>%s</p>' % html.escape(desc))
        parts.append('<div class="pk-legend">Columns = arms (effect band on top, the same arm\'s '
                     'neutral directly below); columns are closed up. <b>absent</b> = no generation '
                     'for that arm/variant in this row. Scroll sideways for all columns.</div>')
        parts.append('<div class="pk-gridwrap">%s%s</div>' % (header, rows))
        parts.append('</section>')

    if missing:
        for s, k, p in missing[:50]:
            print("MISSING SOURCE  [%s/%s]  %s" % (s, k, p), file=sys.stderr)
        raise SystemExit("FAIL: %d source clip(s) missing; refusing to build." % len(missing))

    out = tpl.replace("{{SECTIONS}}", "\n".join(parts))
    os.makedirs(SITE, exist_ok=True)
    open(INDEX, "w").write(out)
    for m in made:
        print("[picker]", m)
    if created:
        print("[picker] seeded", os.path.relpath(PICKS, REPO))
    total_c = total_o = total_ph = 0
    for sec in ("A", "B", "C", "D"):
        v = counts[sec]
        total_c += v["cand"]; total_o += v["own"]; total_ph += v["ph"]
        collist = " · ".join(v["arm_meta"][a]["arm_label"] for a in v["own_arms"]) \
            + (" || prior: " + " · ".join(v["arm_meta"][a]["arm_label"] for a in v["prior_arms"])
               if v["prior_arms"] else "")
        print("[picker] %s: rows=%3d  own-cols=%d prior-cols=%d  candidates=%4d (own %4d/prior %3d)  "
              "band-cells eff=%4d neu=%4d prior=%3d  placeholders=%4d"
              % (sec, v["rows"], len(v["own_arms"]), len(v["prior_arms"]), v["cand"], v["own"],
                 v["prior"], v["eff_real"], v["neu_real"], v["prior_real"], v["ph"]))
        print("[picker]     columns: %s" % collist)
        if v["eff_by_arm"]:
            print("[picker]     effect clips per arm: %s"
                  % " · ".join("%s=%d(store %d)" % (a, v["eff_by_arm"][a], v["eff_from_store"].get(a, 0))
                               for a in sorted(v["eff_by_arm"])))
        if "ctt_v2" in v["own_arms"]:
            print("[picker]     ctt_v2 real rows: neutral=%d effect=%d (of %d)"
                  % (v["ctt"]["neutral"], v["ctt"]["effect"], v["rows"]))
    print("[picker] TOTAL candidates=%d  own=%d  prior-work=%d  placeholders=%d"
          % (total_c, total_o, total_c - total_o, total_ph))
    print("[picker] wrote %s (%.0f KB)" % (INDEX, len(out) / 1024))
    return out


# --------------------------------------------------------------------------- check
def do_check():
    raw = open(INDEX).read()
    # scan real markup: drop the retained VAP <script> bodies (inert JS template literals).
    text = re.sub(r"<script\b.*?</script>", "", raw, flags=re.DOTALL)
    # our cells defer the URL to data-src; VAP demo markup (inert here) may still use src.
    srcs = re.findall(r'<source[^>]+(?:data-src|src)="([^"]+)"', text)
    imgs = re.findall(r'<img[^>]+src="([^"]+)"', text)
    hrefs = re.findall(r'href="([^"]+)"', text)
    bad_abs, missing = [], []
    for s in srcs + imgs:
        if s.startswith("/") or s.startswith("http://") or s.startswith("https://"):
            bad_abs.append(s)
            continue
        if not os.path.exists(os.path.join(SITE, s)):
            missing.append(s)
    ext = [h for h in hrefs if h.startswith("http://") or h.startswith("https://")]
    allowed_ext = {"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"}
    stray_ext = [h for h in ext if h not in allowed_ext]
    print("[check] <source> refs=%d  <img> refs=%d  href=%d" % (len(srcs), len(imgs), len(hrefs)))
    print("[check] external URLs: %s" % (sorted(set(ext)) or "none"))
    print("[check] absolute media paths: %d  missing media: %d  stray external: %d"
          % (len(bad_abs), len(missing), len(stray_ext)))
    ok = True
    for s in bad_abs[:50]:
        print("[check]   ABSOLUTE", s); ok = False
    for s in missing[:50]:
        print("[check]   MISSING", s); ok = False
    for s in stray_ext:
        print("[check]   STRAY-EXTERNAL", s); ok = False
    if ok:
        print("[check] OK: every media ref resolves through the site/ symlinks, no absolute "
              "paths, only the Font Awesome stylesheet is external")
    else:
        raise SystemExit("[check] FAILED")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", action="store_true", help="render picker/site/index.html (default)")
    ap.add_argument("--check", action="store_true", help="verify every media ref resolves")
    a = ap.parse_args()
    if not (a.html or a.check):
        a.html = True
    if a.html:
        do_html()
    if a.check:
        do_check()


if __name__ == "__main__":
    main()
