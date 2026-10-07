#!/usr/bin/env python3
"""Build the SEGUE supplementary website (adapted from the Video-As-Prompt gh-pages page).

Run with the login-node python3.12 (login python3 is 3.6):

    /usr/bin/python3.12 supplementary/build_site.py --all

Stages (composable):
    --select   resolve every row/section from the picks (+ collections/snapshot/pairs) and
               write site_manifest.json (no media touched). A row is on the site IFF it has an
               owner pick (eval_ladder/viewer/collections/supplementary_picks.json).
    --media    re-encode every clip into site/videos/<section>/<kind>/<row_id>.mp4 with the
               human-study recipe (libx264 yuv420p crf26 preset medium -an +faststart -threads 2),
               dedup by source (hardlink) and REUSE any source already encoded under an old dest
               name (dest names are item_id-based; nothing is ever deleted), write media_manifest.json
    --html     render site/index.html from template.html ({{SECTIONS}} placeholder), applying the
               curation site plan (eval_ladder/viewer/collections/site_plan.json): class grouping,
               class/block/row order, hidden blocks/rows/cells, merge_into. Seeds the plan (auto
               order + migrated code overrides) if it is absent.
    --all      select, then media, then html
    --check    verify every <source [data-]src>/<img src>/href in index.html resolves inside site/

Row source / sections (see supplementary/README.md):
    A  teg_user_study            comparisons (both endpoints given), SEGUE vs Base LTX-2 vs refVFX
    B  vfx_transfer_user_study   comparisons (start frame given), SEGUE vs VAP vs VFXMaster vs refVFX
    C  supplementary_teg         SEGUE gallery, transition effect generation
    D  supplementary_vfx_transfer SEGUE gallery, visual effect transfer
Every section shows ALL picked rows. A/B additionally take back the rows the 2026-09-24 judge
win-all filter dropped, merged from the pre-filter snapshot exactly like the picker (live items
first, snapshot extras appended, same ids); a row's judge verdict is kept as data for the overlay.
The SEGUE clip in every section is the owner's picked clip (pick.video). Reference video and the
endpoint clips come from the collection item's inputs; A/B opponents come verbatim from the human
study's pairs.json where the row was in the study, else from the item's prior-work columns (seed 42).
"""
import argparse
import glob
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SITE = os.path.join(HERE, "site")
VIDEOS = os.path.join(SITE, "videos")
TEMPLATE = os.path.join(HERE, "template.html")
INDEX = os.path.join(SITE, "index.html")      # PRODUCTION page: clean, hidden items absent, no overlay
CURATE = os.path.join(SITE, "curate.html")    # internal curation page (localhost overlay, hidden items inert)
PROD = False                                   # set while rendering INDEX
SITE_MANIFEST = os.path.join(HERE, "site_manifest.json")
MEDIA_MANIFEST = os.path.join(SITE, "media_manifest.json")

COLLECTIONS = os.path.join(REPO, "eval_ladder/viewer/collections/neutral_effect_collections.json")
# the pre-filter snapshot: A/B rows the 2026-09-24 judge win-all filter dropped (same ids)
SNAPSHOT_PREFILTER = os.path.join(REPO, "eval_ladder/viewer/collections/snapshots/"
                                  "neutral_effect_collections.2026-09-24T05-12Z_after_split_before_judge_filter.json")
EXTRA_FROM_SNAPSHOT = ("teg_user_study", "vfx_transfer_user_study")
PAIRS = os.path.join(REPO, "misc/2026-09-22_user_study/pairs.json")
# owner's per-row picks (from the internal picker/); the row source for the whole site
PICKS_DEFAULT = os.path.join(REPO, "eval_ladder/viewer/collections/supplementary_picks.json")
# curation site plan (order + hidden); lives in the static server's POST allow-list
SITE_PLAN = os.path.join(REPO, "eval_ladder/viewer/collections/site_plan.json")
SITE_PLAN_LINK = os.path.join(SITE, "site_plan.json")               # symlink -> SITE_PLAN
SITE_PLAN_LINK_TARGET = "../../eval_ladder/viewer/collections/site_plan.json"

NOVELTY_ORDER = ["seen", "unseen", "zero_shot"]
CONTENT_ORDER = ["same", "cross", "foreign"]
NOVELTY_DISP = {"seen": "seen", "unseen": "unseen", "zero_shot": "zero-shot"}

SYS_LABEL = {"segue": "SEGUE", "base_ltx2": "Base LTX-2", "refvfx": "refVFX",
             "vap": "VAP", "vfxmaster": "VFXMaster"}
# section slug -> (display, prior-work arm id in the collection columns); opponents shown per section
A_OPP = [("base_ltx2", "Base LTX-2", "base_cond"), ("refvfx", "refVFX", "refvfx_teg")]
B_OPP = [("vap", "VAP", "vap"), ("vfxmaster", "VFXMaster", "vfxmaster"), ("refvfx", "refVFX", "refvfx")]


# --------------------------------------------------------------------------- utils
def apath(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def relrepo(p):
    """absolute-or-relative source path -> repo-relative."""
    ap = os.path.abspath(apath(p))
    root = os.path.abspath(REPO) + os.sep
    if ap.startswith(root):
        return ap[len(root):]
    return p


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def load_collections():
    """Live collections, with the A/B pre-filter snapshot extras appended (same ids, tagged
    _extra) — mirrors picker/build_picker.py's load_collections snapshot merge."""
    cols = {c["id"]: c for c in json.load(open(COLLECTIONS))["collections"]}
    if os.path.exists(SNAPSHOT_PREFILTER):
        snap = {c["id"]: c for c in json.load(open(SNAPSHOT_PREFILTER))["collections"]}
        for cid in EXTRA_FROM_SNAPSHOT:
            if cid not in cols or cid not in snap:
                continue
            live_ids = {it["id"] for it in cols[cid]["items"]}
            extra = [dict(it, _extra=True) for it in snap[cid]["items"] if it["id"] not in live_ids]
            cols[cid] = dict(cols[cid], items=list(cols[cid]["items"]) + extra)
    return cols


def load_pairs():
    return json.load(open(PAIRS))


def load_picks(path):
    """The internal picker's per-row picks as {collection_id: {item_id: pick}}; {} if absent."""
    if not path:
        return {}
    try:
        doc = json.load(open(path))
    except (FileNotFoundError, ValueError):
        return {}
    return (doc.get("picks") or {}) if isinstance(doc, dict) else {}


def pick_for(picks, cid, item_id):
    return (picks.get(cid) or {}).get(item_id)


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    lab = os.environ.get("LAB", "/taiga/illinois/eng/cs/jrehg/users/emirkisa")
    for pat in (os.path.join(lab, "envs-aarch64/ltx2/lib/python*/site-packages/"
                             "imageio_ffmpeg/binaries/ffmpeg-*"),):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    sys.exit("no ffmpeg found (system PATH or imageio_ffmpeg binary)")


def tier_words(novelty, content):
    return "%s · %s" % (NOVELTY_DISP.get(novelty, novelty), content)


def tier_sort_key(novelty, content, endpoint):
    n = NOVELTY_ORDER.index(novelty) if novelty in NOVELTY_ORDER else 99
    c = CONTENT_ORDER.index(content) if content in CONTENT_ORDER else 99
    return (n, c, endpoint)


def _now():
    import datetime
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def _now_iso_z():
    import datetime
    t = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    return t.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- select
def _ref_of(it):
    return it["inputs"]["refs"][0]["clip"]


def _refcls_of(it):
    return it["inputs"]["refs"][0]["cls"]


def _ep_of(it):
    return it["inputs"]["endpoint"]


def _novelty_of(it):
    t = it["inputs"].get("tier") or {}
    return (t.get("novelty") or ["zero_shot"])[0]


def _content_of(it, pairs_seed=None):
    t = it["inputs"].get("tier") or {}
    c = (t.get("content") or [None])[0]
    if c:
        return c
    if pairs_seed:
        return pairs_seed.get("content") or ""
    return ""


def _col_clip(it, arm, seed="42"):
    """Prior-work opponent clip from the collection item's columns (present, seed 42)."""
    for c in it.get("columns", []):
        if c.get("arm") == arm and c.get("present"):
            for g in (c.get("gens") or []):
                if str(g.get("seed")) == seed and g.get("video"):
                    return g["video"]
    return None


def do_select(picks_path=None, apply_cmp=True):
    picks = load_picks(picks_path)
    cols = load_collections()
    pairs = load_pairs()
    pidx = defaultdict(list)
    for p in pairs:
        pidx[(p["task"], p["endpoint"], p["reference"])].append(p)

    missing_sources = []

    def add_src(section, src, kind):
        fp = apath(src)
        if not src or not os.path.exists(fp):
            missing_sources.append((section, kind, src))
        return relrepo(src) if src else src

    # ---- A / B: comparisons (one row per picked (reference, endpoint)) ----
    def comparisons(cid, task, opp, section, prefix):
        rows = []
        pk_all = picks.get(cid, {})
        for it in cols[cid]["items"]:
            pk = pick_for(picks, cid, it["id"])
            if not pk:                     # a row is on the site iff it has an owner pick
                continue
            inp = it["inputs"]
            ep, ref, refclass = _ep_of(it), _ref_of(it), _refcls_of(it)
            ps = {p["opponent"]: p for p in pidx[(task, ep, ref)]}
            seed_p = next(iter(ps.values()), None)
            rid = "%s__%s" % (prefix, it["id"])
            nov = "zero_shot"
            content = _content_of(it, seed_p)
            media = {
                "reference": {"src": add_src(section, inp["refs"][0].get("video"), "reference"),
                              "dest": "videos/%s/reference/%s.mp4" % (section, rid)},
                "start": {"src": add_src(section, inp.get("prefix_video"), "start"),
                          "dest": "videos/%s/start/%s.mp4" % (section, rid)},
                "segue": {"src": add_src(section, pk["video"], "segue"),
                          "dest": "videos/%s/segue/%s.mp4" % (section, rid)},
            }
            if section == "A":
                media["end"] = {"src": add_src(section, inp.get("suffix_video"), "end"),
                                "dest": "videos/%s/end/%s.mp4" % (section, rid)}
            for slug_name, disp, arm in opp:
                src = ps[disp]["opponent_clip"] if disp in ps else _col_clip(it, arm)
                media[slug_name] = {"src": add_src(section, src, slug_name),
                                    "dest": "videos/%s/%s/%s.mp4" % (section, slug_name, rid)}
            rows.append({"row_id": rid, "endpoint": ep, "reference": ref,
                         "ref_class": refclass, "novelty": nov, "content": content,
                         "tier_words": tier_words(nov, content), "media": media,
                         "src_collection": cid, "item_id": it["id"], "card_key": it.get("card_key"),
                         "judge": it.get("judge"), "is_extra": bool(it.get("_extra")),
                         "pick": pk, "pick_applied": bool(apply_cmp)})
        return rows

    a_rows = comparisons("teg_user_study", "two", A_OPP, "A", "teg_cmp")
    b_rows = comparisons("vfx_transfer_user_study", "one", B_OPP, "B", "vfx_cmp")

    # ---- C / D: SEGUE galleries, grouped by reference (all picked cells; no "already in a
    # comparison" drop) ----
    def galleries(cid, section, prefix):
        pk_all = picks.get(cid, {})
        items = [it for it in cols[cid]["items"] if pick_for(picks, cid, it["id"])]
        by_ref = defaultdict(list)
        for it in items:
            by_ref[_ref_of(it)].append(it)
        blocks = []
        for ref in sorted(by_ref):                # stable; final order comes from the plan
            entries = by_ref[ref]
            refclass = _refcls_of(entries[0])
            block_id = "%s_ref__%s" % (prefix, slug(ref))
            entries = sorted(entries, key=lambda it: tier_sort_key(_novelty_of(it),
                                                                   _content_of(it), _ep_of(it)))
            ref_src = add_src(section, entries[0]["inputs"]["refs"][0].get("video"), "reference")
            cells = []
            for it in entries:
                ep = _ep_of(it)
                nov, con = _novelty_of(it), _content_of(it)
                pk = pick_for(picks, cid, it["id"])
                rid = "%s_res__%s" % (prefix, it["id"])
                media = {
                    "start": {"src": add_src(section, it["inputs"].get("prefix_video"), "start"),
                              "dest": "videos/%s/start/%s.mp4" % (section, rid)},
                    "segue": {"src": add_src(section, pk["video"], "segue"),
                              "dest": "videos/%s/segue/%s.mp4" % (section, rid)},
                }
                if section == "C":
                    media["end"] = {"src": add_src(section, it["inputs"].get("suffix_video"), "end"),
                                    "dest": "videos/%s/end/%s.mp4" % (section, rid)}
                cells.append({"row_id": rid, "endpoint": ep, "novelty": nov, "content": con,
                              "tier_words": tier_words(nov, con), "media": media,
                              "src_collection": cid, "item_id": it["id"],
                              "card_key": it.get("card_key"), "pick": pk})
            blocks.append({"block_id": block_id, "reference": ref, "ref_class": refclass,
                           "reference_media": {"src": ref_src,
                                               "dest": "videos/%s/reference/%s.mp4"
                                                       % (section, block_id)},
                           "cells": cells})
        return blocks

    c_blocks = galleries("supplementary_teg", "C", "teg")
    d_blocks = galleries("supplementary_vfx_transfer", "D", "vfx")

    # ---- stills (endpoint frames): not shown on the page, generated for the registry link ----
    stills = []

    def add_still(section, src, frame, kind, rid):
        if not src:
            return
        stills.append({"src": src, "frame": frame, "kind": kind,
                       "dest": "stills/%s/%s/%s.jpg" % (section, kind, rid)})

    for r in a_rows:
        add_still("A", r["media"]["start"]["src"], 0, "start", r["row_id"])
        add_still("A", r["media"]["end"]["src"], 8, "end", r["row_id"])
    for r in b_rows:
        add_still("B", r["media"]["start"]["src"], 0, "start", r["row_id"])
    for blk in c_blocks:
        for cell in blk["cells"]:
            add_still("C", cell["media"]["start"]["src"], 0, "start", cell["row_id"])
            add_still("C", cell["media"]["end"]["src"], 8, "end", cell["row_id"])
    for blk in d_blocks:
        for cell in blk["cells"]:
            add_still("D", cell["media"]["start"]["src"], 0, "start", cell["row_id"])

    if missing_sources:
        for s, k, p in missing_sources:
            print("MISSING SOURCE  [%s/%s]  %s" % (s, k, p), file=sys.stderr)
        sys.exit("FAIL: %d source clip(s) missing; refusing to build." % len(missing_sources))

    g_rows = []
    for g in GUIDANCE_ROWS:
        gid = g["id"]
        media = {"reference": {"src": add_src("G", g["reference"], "reference"),
                               "dest": "videos/G/reference/%s.mp4" % gid},
                 "start": {"src": add_src("G", g["start"], "start"),
                           "dest": "videos/G/start/%s.mp4" % gid}}
        gens = []
        for k, (label, src) in enumerate(g["gens"]):
            key = "gen%d" % k
            media[key] = {"src": add_src("G", src, label), "dest": "videos/G/%s/%s.mp4" % (key, gid)}
            gens.append({"key": key, "label": label})
        g_rows.append({"part": g["part"], "id": gid, "ref_class": g["ref_class"], "gens": gens, "media": media})

    l_rows = []
    for g in LIMITATION_ROWS:
        rid = g["id"]
        media = {"reference": {"src": add_src("L", g["reference"], "reference"), "dest": "videos/L/reference/%s.mp4" % rid},
                 "start": {"src": add_src("L", g["start"], "start"), "dest": "videos/L/start/%s.mp4" % rid},
                 "end": {"src": add_src("L", g["end"], "end"), "dest": "videos/L/end/%s.mp4" % rid},
                 "gen": {"src": add_src("L", g["gen"][1], g["gen"][0]), "dest": "videos/L/gen/%s.mp4" % rid}}
        l_rows.append({"part": g["part"], "id": rid, "ref_class": g["ref_class"], "label": g["gen"][0], "media": media})

    manifest = {
        "generated": _now(),
        "row_source": "owner picks (%s)" % relrepo(picks_path or PICKS_DEFAULT),
        "recipe": "libx264 yuv420p crf26 preset medium -an +movflags faststart -threads 2 (native fps/frames, no audio)",
        "sources": {"collections": relrepo(COLLECTIONS), "snapshot": relrepo(SNAPSHOT_PREFILTER),
                    "pairs": relrepo(PAIRS), "picks": relrepo(picks_path or PICKS_DEFAULT)},
        "sections": {
            "A": {"title": "Comparisons — Transition Effect Generation (both endpoints given)",
                  "systems": ["segue", "base_ltx2", "refvfx"], "rows": a_rows},
            "B": {"title": "Comparisons — Visual Effect Transfer (start frame given)",
                  "systems": ["segue", "vap", "vfxmaster", "refvfx"], "rows": b_rows},
            "C": {"title": "SEGUE results — Transition Effect Generation", "blocks": c_blocks},
            "D": {"title": "SEGUE results — Visual Effect Transfer", "blocks": d_blocks},
        },
        "guidance": g_rows,
        "limitations": l_rows,
        "stills": stills,
        "counts": {
            "A_rows": len(a_rows), "B_rows": len(b_rows),
            "C_blocks": len(c_blocks), "C_cells": sum(len(b["cells"]) for b in c_blocks),
            "D_blocks": len(d_blocks), "D_cells": sum(len(b["cells"]) for b in d_blocks),
            "stills": len(stills),
        },
    }
    json.dump(manifest, open(SITE_MANIFEST, "w"), indent=1)
    print("[select] A=%d rows  B=%d rows  C=%d blocks/%d cells  D=%d blocks/%d cells"
          % (len(a_rows), len(b_rows), len(c_blocks), manifest["counts"]["C_cells"],
             len(d_blocks), manifest["counts"]["D_cells"]))
    print("[select] wrote", SITE_MANIFEST)
    return manifest


# --------------------------------------------------------------------------- media
def _iter_media(manifest):
    S = manifest["sections"]
    for r in S["A"]["rows"] + S["B"]["rows"]:
        for m in r["media"].values():
            yield m["dest"], m["src"]
    for sec in ("C", "D"):
        for blk in S[sec]["blocks"]:
            m = blk["reference_media"]
            yield m["dest"], m["src"]
            for cell in blk["cells"]:
                for m in cell["media"].values():
                    yield m["dest"], m["src"]
    for r in manifest.get("guidance", []):
        for m in r["media"].values():
            yield m["dest"], m["src"]
    for r in manifest.get("limitations", []):
        for m in r["media"].values():
            yield m["dest"], m["src"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _encode(ffmpeg, src, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-map", "0:v:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-preset", "medium", "-an",
                    "-movflags", "+faststart", "-threads", "2", out], check=True)


def _extract_still(ffmpeg, src, frame, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-vf", "select=eq(n\\,%d)" % frame, "-frames:v", "1",
                    "-q:v", "3", "-threads", "2", out], check=True)


def _probe(ffmpeg, path):
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0",
                        "-c", "copy", "-f", "null", "-"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    se = r.stderr
    mw = re.search(r"Video:.*?,\s*(\d+)x(\d+)", se)
    w, h = (int(mw.group(1)), int(mw.group(2))) if mw else (None, None)
    mf = re.search(r"([\d.]+)\s+fps", se)
    fps = float(mf.group(1)) if mf else None
    fr = re.findall(r"frame=\s*(\d+)", se)
    frames = int(fr[-1]) if fr else None
    return frames, fps, w, h


def _link_or_copy(src_file, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    try:
        os.link(src_file, out)
    except OSError:
        shutil.copyfile(src_file, out)


def do_media(manifest=None):
    if manifest is None:
        manifest = json.load(open(SITE_MANIFEST))
    ffmpeg = find_ffmpeg()
    old = json.load(open(MEDIA_MANIFEST)) if os.path.exists(MEDIA_MANIFEST) else {}

    # source -> an existing (still-on-disk) encoded clip dest, so a source already encoded under an
    # old (item_id-based dest names changed between builds) name is reused, never re-encoded.
    old_by_source = {}
    for dest, e in old.items():
        if "frame" in e:                       # a still, not a clip
            continue
        s = e.get("source")
        if s and s not in old_by_source and os.path.exists(os.path.join(SITE, dest)):
            old_by_source[s] = dest
    old_still_by_key = {}
    for dest, e in old.items():
        if "frame" not in e:
            continue
        k = (e.get("source"), e.get("frame"))
        if k[0] is not None and k not in old_still_by_key and os.path.exists(os.path.join(SITE, dest)):
            old_still_by_key[k] = dest

    by_src = defaultdict(list)
    for dest, src in _iter_media(manifest):
        by_src[src].append(dest)
    for s in by_src:
        by_src[s] = sorted(set(by_src[s]))

    total_dests = sum(len(v) for v in by_src.values())
    print("[media] %d unique sources -> %d dest files" % (len(by_src), total_dests))

    enc = skip = reuse = link = 0

    def encode_canonical(src):
        """encode/reuse/probe the canonical dest for a source; return (dest0, entry, status)."""
        dest0 = by_src[src][0]
        out0 = os.path.join(SITE, dest0)
        src_abs = apath(src)
        if os.path.exists(out0) and dest0 in old and old[dest0].get("source") == src:
            return dest0, old[dest0], "skip"
        prev = old_by_source.get(src)
        if prev and os.path.exists(os.path.join(SITE, prev)):
            _link_or_copy(os.path.join(SITE, prev), out0)
            entry = dict(old[prev]); entry["source"] = src; entry["bytes"] = os.path.getsize(out0)
            return dest0, entry, "reuse"
        _encode(ffmpeg, src_abs, out0)
        frames, fps, w, h = _probe(ffmpeg, out0)
        entry = {"source": src, "sha256": _sha256(src_abs), "frames": frames,
                 "fps": fps, "w": w, "h": h, "bytes": os.path.getsize(out0)}
        return dest0, entry, "enc"

    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(encode_canonical, src): src for src in by_src}
        done = 0
        for fut in list(futs):
            dest0, entry, status = fut.result()
            src = futs[fut]
            results[src] = (dest0, entry, status)
            enc += status == "enc"; skip += status == "skip"; reuse += status == "reuse"
            done += 1
            if done % 100 == 0 or done == len(by_src):
                print("[media] sources %d/%d (encoded=%d reused=%d skipped=%d)"
                      % (done, len(by_src), enc, reuse, skip))

    manifest_out = {}
    for src, (dest0, entry, status) in results.items():
        manifest_out[dest0] = entry
        for dest in by_src[src][1:]:
            out = os.path.join(SITE, dest)
            if os.path.exists(out) and dest in old and old[dest].get("source") == src:
                manifest_out[dest] = old[dest]; skip += 1; continue
            _link_or_copy(os.path.join(SITE, dest0), out)
            e2 = dict(entry); e2["bytes"] = os.path.getsize(out)
            manifest_out[dest] = e2; link += 1

    # ---- stills (deduped by (source, frame); reuse old on-disk extractions) ----
    still_by_key = defaultdict(list)
    for s in manifest.get("stills", []):
        still_by_key[(s["src"], s["frame"])].append((s["dest"], s["kind"]))

    def extract_canonical(key):
        src, frame = key
        dest0, kind0 = still_by_key[key][0]
        out0 = os.path.join(SITE, dest0)
        src_abs = apath(src)
        if os.path.exists(out0) and dest0 in old and old[dest0].get("source") == src \
                and old[dest0].get("frame") == frame:
            return key, dest0, old[dest0], "skip"
        prev = old_still_by_key.get(key)
        if prev and os.path.exists(os.path.join(SITE, prev)):
            _link_or_copy(os.path.join(SITE, prev), out0)
            entry = dict(old[prev]); entry["kind"] = kind0; entry["bytes"] = os.path.getsize(out0)
            return key, dest0, entry, "reuse"
        _extract_still(ffmpeg, src_abs, frame, out0)
        _, _, w, h = _probe(ffmpeg, out0)
        entry = {"source": src, "sha256": _sha256(src_abs), "kind": kind0,
                 "frame": frame, "w": w, "h": h, "bytes": os.path.getsize(out0)}
        return key, dest0, entry, "enc"

    senc = sskip = sreuse = slink = 0
    sresults = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(extract_canonical, k): k for k in still_by_key}
        for fut in list(futs):
            key, dest0, entry, status = fut.result()
            sresults[key] = (dest0, entry, status)
            senc += status == "enc"; sskip += status == "skip"; sreuse += status == "reuse"
    for key, (dest0, entry, status) in sresults.items():
        manifest_out[dest0] = entry
        for dest, _kind in still_by_key[key][1:]:
            out = os.path.join(SITE, dest)
            if os.path.exists(out) and dest in old and old[dest].get("source") == key[0] \
                    and old[dest].get("frame") == key[1]:
                manifest_out[dest] = old[dest]; sskip += 1; continue
            _link_or_copy(os.path.join(SITE, dest0), out)
            e2 = dict(entry); e2["bytes"] = os.path.getsize(out)
            manifest_out[dest] = e2; slink += 1
    print("[media] stills: unique=%d extracted=%d reused=%d skipped=%d hardlinked=%d"
          % (len(still_by_key), senc, sreuse, sskip, slink))

    json.dump(manifest_out, open(MEDIA_MANIFEST, "w"), indent=1)
    total = sum(e["bytes"] for e in manifest_out.values())
    print("[media] files=%d encoded=%d reused=%d skipped=%d hardlinked=%d  total=%.1f MB"
          % (len(manifest_out), enc, reuse, skip, link, total / 1e6))
    print("[media] wrote", MEDIA_MANIFEST)
    return manifest_out


# --------------------------------------------------------------------------- html
def _media_div(dest, extra_class="", badge=None):
    """lazy <video> with <source data-src> (never fetched until it scrolls into view; hidden
    clips are display:none, never intersect, so nothing loads)."""
    b = '<span class="seg-badge">%s</span>' % html.escape(badge) if badge else ""
    return ('<div class="seg-media %s">%s<video class="lazy-video" preload="none" '
            'muted playsinline loop data-autoplay="true">'
            '<source data-src="%s" type="video/mp4"/></video></div>'
            % (extra_class, b, dest))


def _still_div(dest, extra_class="", badge=None):
    """A single frame instead of a clip (2026-09-26, owner: for visual effect transfer the start
    endpoint shows only its first frame — the effect starts too early in the clip)."""
    b = '<span class="seg-badge">%s</span>' % html.escape(badge) if badge else ""
    return ('<div class="seg-media seg-stillbox %s">%s<img class="seg-still" src="%s" alt="start frame" loading="lazy"/></div>'
            % (extra_class, b, dest))


def _noend_div():
    return ('<div class="seg-media seg-noend"><span class="seg-badge">no end</span>'
            '<span class="seg-xlabel">no end given</span></div>')


def _unit(inner, label, extra_class=""):
    lab = '<div class="seg-label">%s</div>' % html.escape(label) if label else ""
    return '<div class="seg-unit %s">%s%s</div>' % (extra_class, inner, lab)


CLASS_NAMES = os.path.join(HERE, "class_names.json")

# public section titles (order top->bottom: C, A, D, B); no internal codenames
SECTION_ORDER = ["C", "A", "D", "B"]
SECTION_TITLE = {
    "C": "Transition effect generation",
    "A": "Transition effect generation — comparison with previous work",
    "D": "Visual effect transfer",
    "B": "Visual effect transfer — comparison with previous work",
}
SECTION_DESC = {
    "A": "Both endpoint frames and a reference effect video are given. Left: the reference "
         "effect video, then the given start and end endpoints (stacked). Right: the generated "
         "transition from SEGUE and the two prior-work systems it was compared against.",
    "B": "One endpoint frame and a reference effect video are given. Left: the reference effect "
         "video and the given start endpoint. Right: the generated result from SEGUE and the "
         "three prior-work systems it was compared against.",
    "C": "Transition-effect-generation results from SEGUE, grouped by transition class and "
         "reference effect video. Each cell shows the given start and end endpoints (stacked) "
         "and the generated transition.",
    "D": "Visual-effect-transfer results from SEGUE, grouped by transition class and reference "
         "effect video. Each cell shows the given start endpoint (no end endpoint is given) and "
         "the generated result.",
}
SECTION_FOOT = {
    "A": "SEGUE, seed 42. Rows shown are the owner-picked comparisons (including cases the blind "
         "study did not win against every prior-work system); the clip shown for SEGUE is the "
         "owner-picked clip.",
    "B": "SEGUE, seed 42. Rows shown are the owner-picked comparisons (including cases the blind "
         "study did not win against every prior-work system); the clip shown for SEGUE is the "
         "owner-picked clip.",
    "C": "SEGUE, seed 42. The clip shown for SEGUE is the owner-picked clip.",
    "D": "SEGUE, seed 42. The clip shown for SEGUE is the owner-picked clip.",
}
# 2026-09-26 (owner): first guidance row — NRG sweep on one seen/same one-sided cell (color_rain_1 reference,
# color_rain_3 start clip, grid cell G-fit, neutral prompt, seed 42): "the rain intensifies with guidance".
_G3 = "outputs/videos/grid_v3/%s_neutral_v3/G-fit__%s_neutral_v3__color_rain_3__ref_color_rain_1__s42.mp4"
GUIDANCE_ROWS = [
    {"part": "nrg", "id": "nrg_color_rain_3_gfit", "ref_class": "color_rain",
     "reference": "data/processed/transitions_std121/color_rain/color_rain_1.mp4",
     "start": "eval_ladder/conds/color_rain_3_start9.mp4",
     "gens": [("SEGUE w/o NRG", _G3 % ("dualforce_control", "dualforce_control")),
              ("SEGUE (w=1.5)", _G3 % ("dualforce_dcg_w1p5", "dualforce_dcg_w1p5")),
              ("SEGUE (w=3)", _G3 % ("dualforce_dcg_w3", "dualforce_dcg_w3")),
              ("SEGUE (w=6)", _G3 % ("dualforce_dcg_w6", "dualforce_dcg_w6"))]},
]
# 2026-09-26 (owner): Limitations — failure cases, all two-sided, neutral prompt, seed 42, grid v3.
_L3 = "outputs/videos/grid_v3/%s_neutral_v3/%s__%s_neutral_v3__%s__ref_%s__s42.mp4"   # arm, cell, arm, endpoint, ref
def _lim(part, rid, arm, label, cell, endpoint, ref, ref_class):
    return {"part": part, "id": rid, "ref_class": ref_class,
            "reference": "data/processed/transitions_std121/%s/%s.mp4" % (ref_class, ref),
            "start": "eval_ladder/conds/%s_start9.mp4" % endpoint,
            "end": "eval_ladder/conds/%s_end9.mp4" % endpoint,
            "gen": (label, _L3 % (arm, cell, arm, endpoint, ref))}
LIMITATION_ROWS = [
    _lim("leak", "leak_raven_airbending1", "dualforce_dcg_w6", "SEGUE (w=6)", "G-zs-cross",
         "air_bending_1", "raven_transition_0", "raven_transition"),
    _lim("leak", "leak_display_shadowsmoke0", "dualforce_dcg_w3", "SEGUE (w=3)", "G-zs-cross",
         "shadow_smoke_0", "display_transition_2", "display_transition"),
    _lim("endpoint", "endpoint_shadowsmoke_tennis", "dualforce_dcg_w3", "SEGUE (w=3)", "G-unseen-foreign",
         "davis_tennis_snowboard", "shadow_smoke_0", "shadow_smoke"),
]
LIMITATION_PARTS = [
    ("leak", "Reference scene leaks into the transition",
     "Content of the reference clip, not only its transition effect, appears in the generated middle."),
    ("endpoint", "Endpoint quality not preserved",
     "The quality of the given endpoints is not preserved in the generated transition."),
]
GUIDANCE_PARTS = [
    ("nrg", "Null-reference guidance (NRG)", "The same reference and start clip with the guidance weight increasing left to right; neutral prompt. The transition effect intensifies with the weight."),
    # ("emptynull", "Empty null", ...) — removed from the site by the owner on 2026-09-26
]

# --- migrated code overrides (2026-09-25) baked into the SEEDED site plan (section C).
# Old code-side BLOCK_DROP / BLOCK_MOVE_AFTER / BLOCK_MERGE_INTO / CELL_DROP are gone; their
# curated effect is carried into site_plan.json so nothing already curated is lost. Cell ids are
# resolved by item_id (robust to re-picks) — computed from the pre-picks manifest rendered order.
MIGRATED_C = {
    "hidden_blocks": ["teg_ref__hero_flight_5", "teg_ref__hero_flight_0", "teg_ref__display_transition_2"],
    "merge_into": {"teg_ref__shadow_smoke_1": "teg_ref__shadow_smoke_7"},
    "hidden_cells": {"teg_ref__shadow_smoke_0": ["x40mchkc"],
                     "teg_ref__shadow_smoke_7": ["7u8csnc6", "juo75g38"]},
    # within class shadow_smoke: keep shadow_smoke_7 directly after shadow_smoke_0
    "block_after": [("shadow_smoke", "teg_ref__shadow_smoke_7", "teg_ref__shadow_smoke_0")],
    # NOTE: the old cross-class move (water_bending_3 after flame_transition_3) is NOT representable
    # under per-class grouping (different classes) and is intentionally dropped.
}


def _auto_class_name(cls):
    s = cls[3:] if cls.startswith("ed.") else cls
    s = s.split(".")[0].replace("_", " ").strip()
    return (s[:1].upper() + s[1:].lower()) if s else cls


def load_class_names(all_classes):
    data = {}
    if os.path.exists(CLASS_NAMES):
        try:
            data = json.load(open(CLASS_NAMES))
        except ValueError:
            data = {}
    changed = False
    for c in sorted(all_classes):
        if c not in data:
            data[c] = _auto_class_name(c)
            changed = True
    if changed:
        json.dump(data, open(CLASS_NAMES, "w"), indent=1, ensure_ascii=False, sort_keys=True)
    return data


def human_name(cls, names):
    return names.get(cls) or _auto_class_name(cls)


def _apply_order(auto_list, plan_list):
    """Plan order first (only ids still present), then auto ids missing from the plan appended in
    auto order; plan ids that no longer exist are ignored."""
    aset = set(auto_list)
    out, seen = [], set()
    for x in (plan_list or []):
        if x in aset and x not in seen:
            out.append(x); seen.add(x)
    for x in auto_list:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def ensure_site_plan_symlink():
    os.makedirs(SITE, exist_ok=True)
    if os.path.islink(SITE_PLAN_LINK):
        if os.readlink(SITE_PLAN_LINK) != SITE_PLAN_LINK_TARGET:
            os.remove(SITE_PLAN_LINK); os.symlink(SITE_PLAN_LINK_TARGET, SITE_PLAN_LINK)
    elif os.path.exists(SITE_PLAN_LINK):
        raise SystemExit("refusing to overwrite non-symlink %s" % SITE_PLAN_LINK)
    else:
        os.symlink(SITE_PLAN_LINK_TARGET, SITE_PLAN_LINK)


def _auto_gallery(blocks, names):
    """{class: [block_id...]} auto order + [class...] auto order for a C/D section."""
    by_class = defaultdict(list)
    for b in blocks:
        by_class[b["ref_class"]].append(b)
    ncell = {b["block_id"]: len(b["cells"]) for b in blocks}
    block_order = {c: sorted([b["block_id"] for b in bs],
                             key=lambda bid: (-ncell[bid], bid)) for c, bs in by_class.items()}
    class_order = sorted(by_class,
                         key=lambda c: (-sum(len(b["cells"]) for b in by_class[c]),
                                        human_name(c, names).lower()))
    return class_order, block_order


def _auto_comparison(rows, names):
    """{class: [item_id...]} auto order (references consecutive) + [class...] for an A/B section."""
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["ref_class"]].append(r)
    row_order = {}
    for c, rs in by_class.items():
        refs = defaultdict(list)
        for r in rs:
            refs[r["reference"]].append(r)
        ref_order = sorted(refs, key=lambda rf: (-len(refs[rf]), rf))
        seq = []
        for rf in ref_order:
            for r in sorted(refs[rf], key=lambda r: tier_sort_key(r["novelty"], r["content"], r["endpoint"])):
                seq.append(r["item_id"])
        row_order[c] = seq
    class_order = sorted(by_class, key=lambda c: (-len(by_class[c]), human_name(c, names).lower()))
    return class_order, row_order


def seed_site_plan(manifest, names):
    S = manifest["sections"]
    plan = {"schema": 1, "updated": _now_iso_z(), "sections": {}}
    for sec in ("C", "D"):
        co, bo = _auto_gallery(S[sec]["blocks"], names)
        plan["sections"][sec] = {"class_order": co, "block_order": bo, "cell_order": {},
                                 "hidden_blocks": [], "hidden_cells": {}, "merge_into": {}}
    for sec in ("A", "B"):
        co, ro = _auto_comparison(S[sec]["rows"], names)
        plan["sections"][sec] = {"class_order": co, "row_order": ro, "hidden_rows": []}
    # migrate the old code-side C overrides into the seeded plan
    pc = plan["sections"]["C"]
    present_blocks = {b["block_id"] for b in S["C"]["blocks"]}
    pc["hidden_blocks"] = [b for b in MIGRATED_C["hidden_blocks"]]         # kept even if absent (ignored later)
    pc["merge_into"] = dict(MIGRATED_C["merge_into"])
    pc["hidden_cells"] = {k: list(v) for k, v in MIGRATED_C["hidden_cells"].items()}
    # remove merged-away source blocks from block_order
    for src, dst in pc["merge_into"].items():
        for cls, lst in pc["block_order"].items():
            if src in lst:
                lst.remove(src)
    # within-class "keep B right after A" moves
    for cls, mover, anchor in MIGRATED_C["block_after"]:
        lst = pc["block_order"].get(cls)
        if lst and mover in lst and anchor in lst:
            lst.remove(mover)
            lst.insert(lst.index(anchor) + 1, mover)
    json.dump(plan, open(SITE_PLAN, "w"), indent=1)
    print("[html] seeded site plan %s" % SITE_PLAN)
    return plan


def load_or_seed_plan(manifest, names):
    if os.path.exists(SITE_PLAN):
        try:
            plan = json.load(open(SITE_PLAN))
            if isinstance(plan, dict) and plan.get("sections"):
                print("[html] read site plan %s (updated %s)" % (SITE_PLAN, plan.get("updated")))
                return plan
        except ValueError:
            pass
    return seed_site_plan(manifest, names)


def _plan_sec(plan, sec):
    return (plan.get("sections") or {}).get(sec) or {}


def _refidx_label(idx, total):
    return "ref %d/%d" % (idx, total) if total > 1 else ""


# ---- cell / row / block renderers ----------------------------------------------
def _cell_html(cell, section, hidden):
    m = cell["media"]
    if section == "C":
        stack = ('<div class="seg-stack">%s%s</div>'
                 % (_media_div(m["start"]["dest"], "seg-endpoint", badge="start"),
                    _media_div(m["end"]["dest"], "seg-endpoint", badge="end")))
    else:   # D: first frame of the start clip only
        stack = ('<div class="seg-stack">%s%s</div>'
                 % (_still_div("stills/D/start/%s.jpg" % cell["row_id"], "seg-endpoint", badge="start"), _noend_div()))
    inner = ('<div class="seg-cell">%s%s</div>'
             % (_unit(stack, "endpoint", "seg-endstack"),
                _unit(_media_div(m["segue"]["dest"]), SYS_LABEL["segue"], "seg-gen")))
    if PROD and hidden:
        return ''
    dh = ' data-hidden="1"' if hidden else ''
    return ('<div class="seg-cellwrap" data-item="%s"%s>%s</div>'
            % (html.escape(cell["item_id"]), dh, inner))


def _block_html(blk, section, hidden, hidden_cells, refidx, cell_seq=None):
    if cell_seq is None:
        cell_seq = blk["cells"]
    ref_unit = _unit(_media_div(blk["reference_media"]["dest"]), "reference", "seg-ref")
    cells = "".join(_cell_html(c, section, c["item_id"] in hidden_cells) for c in cell_seq)
    if PROD and hidden:
        return ''
    dh = ' data-hidden="1"' if hidden else ''
    ridx = ('<span class="seg-refidx">%s</span>' % html.escape(refidx)) if refidx else ''
    return ('<div class="seg-blockwrap" data-section="%s" data-class="%s" data-block="%s"%s>'
            '%s<div class="seg-block">%s<div class="seg-cells">%s</div></div></div>'
            % (section, html.escape(blk["ref_class"]), html.escape(blk["block_id"]), dh,
               ridx, ref_unit, cells))


def _guidance_row_html(r):
    """One guidance row: reference · start (one-sided) · the generations left to right (labels = arms)."""
    m = r["media"]
    cells = [_unit(_media_div(m["reference"]["dest"]), "reference", "seg-ref"),
             _unit(_media_div(m["start"]["dest"], badge="start"), "start", "seg-startfull")]
    for g in r["gens"]:
        cells.append(_unit(_media_div(m[g["key"]]["dest"]), g["label"], "seg-gen"))
    return ('<div class="seg-strip seg-guidance" data-section="G" data-item="%s">'
            '<div class="seg-cellsrow">%s</div></div>' % (html.escape(r["id"]), "".join(cells)))


def _limitation_row_html(r):
    """One failure case: reference · start/end stack · the SEGUE generation."""
    m = r["media"]
    stack = ('<div class="seg-stack">%s%s</div>'
             % (_media_div(m["start"]["dest"], "seg-endpoint", badge="start"),
                _media_div(m["end"]["dest"], "seg-endpoint", badge="end")))
    cells = [_unit(_media_div(m["reference"]["dest"]), "reference", "seg-ref"),
             _unit(stack, "endpoints", "seg-endstack"),
             _unit(_media_div(m["gen"]["dest"]), r["label"], "seg-gen")]
    return ('<div class="seg-strip seg-limitation" data-section="L" data-item="%s">'
            '<div class="seg-cellsrow">%s</div></div>' % (html.escape(r["id"]), "".join(cells)))


def _cmp_row_html(r, section, hidden, refidx):
    m = r["media"]
    if section == "A":
        stack = ('<div class="seg-stack">%s%s</div>'
                 % (_media_div(m["start"]["dest"], "seg-endpoint", badge="start"),
                    _media_div(m["end"]["dest"], "seg-endpoint", badge="end")))
        cells = [
            _unit(_media_div(m["reference"]["dest"]), "reference", "seg-ref"),
            _unit(stack, "endpoints", "seg-endstack"),
            _unit(_media_div(m["segue"]["dest"]), SYS_LABEL["segue"], "seg-gen"),
            _unit(_media_div(m["base_ltx2"]["dest"]), SYS_LABEL["base_ltx2"], "seg-gen"),
            _unit(_media_div(m["refvfx"]["dest"]), SYS_LABEL["refvfx"], "seg-gen"),
        ]
    else:
        cells = [
            _unit(_media_div(m["reference"]["dest"]), "reference", "seg-ref"),
            _unit(_still_div("stills/B/start/%s.jpg" % r["row_id"], badge="start"), "start", "seg-startfull"),
            _unit(_media_div(m["segue"]["dest"]), SYS_LABEL["segue"], "seg-gen"),
            _unit(_media_div(m["vap"]["dest"]), SYS_LABEL["vap"], "seg-gen"),
            _unit(_media_div(m["vfxmaster"]["dest"]), SYS_LABEL["vfxmaster"], "seg-gen"),
            _unit(_media_div(m["refvfx"]["dest"]), SYS_LABEL["refvfx"], "seg-gen"),
        ]
    if PROD and hidden:
        return ''
    dh = ' data-hidden="1"' if hidden else ''
    jv = (r.get("judge") or {}).get("transition_vs") if r.get("judge") else None
    jattr = (" data-judge='%s'" % html.escape(json.dumps(jv), quote=True)) if jv else ''
    ridx = ('<span class="seg-refidx">%s</span>' % html.escape(refidx)) if refidx else ''
    return ('<div class="seg-strip" data-section="%s" data-class="%s" data-item="%s" data-ref="%s"%s%s>'
            '%s<div class="seg-cellsrow">%s</div></div>'
            % (section, html.escape(r["ref_class"]), html.escape(r["item_id"]),
               html.escape(r["reference"]), dh, jattr, ridx, "".join(cells)))


def _classhead(section, cls, names):
    return ('<div class="seg-classhead" id="sec-%s-cls-%s" data-section="%s" data-class="%s">%s</div>'
            % (section, slug(cls), section, html.escape(cls), html.escape(human_name(cls, names))))


def _render_gallery(section, blocks, plan, names):
    ps = _plan_sec(plan, section)
    merge_into = ps.get("merge_into") or {}
    hidden_blocks = set(ps.get("hidden_blocks") or [])
    hidden_cells = ps.get("hidden_cells") or {}
    cell_order = ps.get("cell_order") or {}
    by_id = {b["block_id"]: dict(b) for b in blocks}
    merged_away = set()
    for src, dst in merge_into.items():
        if src in by_id and dst in by_id:
            by_id[dst]["cells"] = list(by_id[dst]["cells"]) + list(by_id[src]["cells"])
            merged_away.add(src)
    blocks2 = [by_id[b["block_id"]] for b in blocks if b["block_id"] not in merged_away]
    auto_co, auto_bo = _auto_gallery(blocks2, names)
    class_order = _apply_order(auto_co, ps.get("class_order"))
    by_class = defaultdict(list)
    for b in blocks2:
        by_class[b["ref_class"]].append(b)
    out = []
    for cls in class_order:
        cblocks = {b["block_id"]: b for b in by_class[cls]}
        bo = _apply_order(auto_bo[cls], (ps.get("block_order") or {}).get(cls))
        n = len(bo)
        group = [_classhead(section, cls, names)]
        for i, bid in enumerate(bo, 1):
            b = cblocks[bid]
            hc = set(hidden_cells.get(bid) or [])
            cmap = {c["item_id"]: c for c in b["cells"]}
            seq = [cmap[iid] for iid in _apply_order([c["item_id"] for c in b["cells"]],
                                                     cell_order.get(bid))]
            group.append(_block_html(b, section, bid in hidden_blocks, hc, _refidx_label(i, n), seq))
        # a class whose blocks are all hidden: hide the heading too (public page shows nothing; the
        # localhost overlay un-hides it like any data-hidden node)
        ghid = ' data-hidden="1"' if bo and all(bid in hidden_blocks for bid in bo) else ""
        if PROD and ghid:
            continue
        out.append('<div class="seg-classgroup" data-section="%s" data-class="%s"%s>%s</div>'
                   % (section, html.escape(cls), ghid, "".join(group)))
    return "".join(out)


def _render_comparison(section, rows, plan, names):
    ps = _plan_sec(plan, section)
    hidden_rows = set(ps.get("hidden_rows") or [])
    auto_co, auto_ro = _auto_comparison(rows, names)
    class_order = _apply_order(auto_co, ps.get("class_order"))
    by_class = defaultdict(dict)
    for r in rows:
        by_class[r["ref_class"]][r["item_id"]] = r
    out = []
    for cls in class_order:
        rmap = by_class[cls]
        ro = _apply_order(auto_ro[cls], (ps.get("row_order") or {}).get(cls))
        # reference index within the class, in final order
        refs = []
        for iid in ro:
            rf = rmap[iid]["reference"]
            if rf not in refs:
                refs.append(rf)
        nref = len(refs)
        group = [_classhead(section, cls, names)]
        for iid in ro:
            r = rmap[iid]
            ridx = _refidx_label(refs.index(r["reference"]) + 1, nref)
            group.append(_cmp_row_html(r, section, iid in hidden_rows, ridx))
        ghid = ' data-hidden="1"' if ro and all(iid in hidden_rows for iid in ro) else ""
        if PROD and ghid:
            continue
        out.append('<div class="seg-classgroup" data-section="%s" data-class="%s"%s>%s</div>'
                   % (section, html.escape(cls), ghid, "".join(group)))
    return "".join(out)


def _render_page(manifest, tpl):
    S = manifest["sections"]

    all_classes = set()
    for r in S["A"]["rows"] + S["B"]["rows"]:
        all_classes.add(r["ref_class"])
    for blk in S["C"]["blocks"] + S["D"]["blocks"]:
        all_classes.add(blk["ref_class"])
    names = load_class_names(all_classes)

    plan = load_or_seed_plan(manifest, names)
    ensure_site_plan_symlink()

    parts = ['<div class="seg-legend">'
             '<span><span class="sw sw-in"></span>given inputs (reference &amp; endpoints)</span>'
             '<span><span class="sw sw-gen"></span>generated clips (SEGUE &amp; prior work)</span>'
             '</div>']
    toc = ['<nav class="seg-toc" id="contents"><div class="seg-toc-title">Contents</div><ul>']
    for key in SECTION_ORDER:
        toc.append('<li><a href="#sec-%s">%s</a></li>' % (key, html.escape(SECTION_TITLE[key])))
    toc.append('<li><a href="#sec-guidance">Guidance</a><ul>')
    for gid, gtitle, _ in GUIDANCE_PARTS:
        toc.append('<li><a href="#sec-guidance-%s">%s</a></li>' % (gid, html.escape(gtitle)))
    toc.append("</ul></li>")
    toc.append('<li><a href="#sec-limitations">Limitations</a><ul>')
    for lid, ltitle, _ in LIMITATION_PARTS:
        toc.append('<li><a href="#sec-limitations-%s">%s</a></li>' % (lid, html.escape(ltitle)))
    toc.append("</ul></li></ul></nav>")
    parts.append("".join(toc))

    for key in SECTION_ORDER:
        parts.append('<div class="section-title" id="sec-%s">%s</div>' % (key, html.escape(SECTION_TITLE[key])))
        parts.append("<p>%s</p>" % html.escape(SECTION_DESC[key]))
        if key in ("C", "D"):
            body = _render_gallery(key, S[key]["blocks"], plan, names)
        else:
            body = _render_comparison(key, S[key]["rows"], plan, names)
        parts.append('<div class="seg-section">%s</div>' % body)
        parts.append('<p class="footnotes">%s</p>' % html.escape(SECTION_FOOT[key]))

    parts.append('<div class="section-title" id="sec-guidance">Guidance</div>')
    for gid, gtitle, gdesc in GUIDANCE_PARTS:
        parts.append('<div class="subsection-title seg-subtitle" id="sec-guidance-%s">%s</div>' % (gid, html.escape(gtitle)))
        parts.append("<p>%s</p>" % html.escape(gdesc))
        grows = [r for r in manifest.get("guidance", []) if r["part"] == gid]
        if grows:
            parts.append('<div class="seg-section">%s</div>' % "".join(_guidance_row_html(r) for r in grows))
        else:
            parts.append('<div class="seg-placeholder">rows to come</div>')
    parts.append('<div class="section-title" id="sec-limitations">Limitations</div>')
    for lid, ltitle, ldesc in LIMITATION_PARTS:
        parts.append('<div class="subsection-title seg-subtitle" id="sec-limitations-%s">%s</div>' % (lid, html.escape(ltitle)))
        parts.append("<p>%s</p>" % html.escape(ldesc))
        lrows = [r for r in manifest.get("limitations", []) if r["part"] == lid]
        parts.append('<div class="seg-section">%s</div>' % "".join(_limitation_row_html(r) for r in lrows))

    return tpl.replace("{{SECTIONS}}", "\n".join(parts))


_OV_CSS = re.compile(r"/\*OVERLAY-CSS-BEGIN\*/.*?/\*OVERLAY-CSS-END\*/", re.DOTALL)
_OV_JS = re.compile(r"<!--OVERLAY-JS-BEGIN-->.*?<!--OVERLAY-JS-END-->", re.DOTALL)


def do_html(manifest=None):
    """Writes TWO pages from one template + plan: CURATE (site/curate.html, internal: every picked
    item present, hidden ones inert data-hidden, localhost overlay for reorder/hide) and INDEX
    (site/index.html, PRODUCTION: hidden items and empty classes absent, overlay CSS/JS stripped)."""
    global PROD
    if manifest is None:
        manifest = json.load(open(SITE_MANIFEST))
    tpl = open(TEMPLATE).read()
    assert tpl.count("{{SECTIONS}}") == 1, "template must have exactly one {{SECTIONS}}"
    assert len(_OV_CSS.findall(tpl)) == 1 and len(_OV_JS.findall(tpl)) == 1, "overlay markers missing in template"
    os.makedirs(SITE, exist_ok=True)
    PROD = False
    cur = _render_page(manifest, tpl)
    open(CURATE, "w").write(cur)
    print("[html] wrote %s (%.0f KB)  [curation: overlay + inert hidden items]" % (CURATE, len(cur) / 1024))
    PROD = True
    try:
        prod = _render_page(manifest, _OV_JS.sub("", _OV_CSS.sub("", tpl)))
    finally:
        PROD = False
    assert 'data-hidden="1"' not in prod and "seg-ov-" not in prod and "applyPlan" not in prod, "prod page not clean"
    open(INDEX, "w").write(prod)
    print("[html] wrote %s (%.0f KB)  [PRODUCTION: clean]" % (INDEX, len(prod) / 1024))
    return prod


# --------------------------------------------------------------------------- check
def do_check():
    for page in (INDEX, CURATE):
        if os.path.exists(page):
            print("[check] %s" % os.path.relpath(page, REPO))
            _check_page(page)


def _check_page(page):
    raw = open(page).read()
    text = re.sub(r"<script\b.*?</script>", "", raw, flags=re.DOTALL)
    srcs = re.findall(r'<source[^>]+(?:data-src|src)="([^"]+)"', text)
    imgs = re.findall(r'<img[^>]+src="([^"]+)"', text)
    hrefs = re.findall(r'href="([^"]+)"', text)
    bad_abs = []
    missing = []
    for s in srcs + imgs:
        if s.startswith("/") or s.startswith("http://") or s.startswith("https://"):
            bad_abs.append(s); continue
        if not os.path.exists(os.path.join(SITE, s)):
            missing.append(s)
    ext = [h for h in hrefs if h.startswith("http://") or h.startswith("https://")]
    allowed_ext = {"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"}
    stray_ext = [h for h in ext if h not in allowed_ext]
    print("[check] <source> refs=%d  <img> refs=%d  href=%d" % (len(srcs), len(imgs), len(hrefs)))
    print("[check] external URLs: %s" % (ext or "none"))
    print("[check] absolute media paths: %d  missing media: %d  stray external: %d"
          % (len(bad_abs), len(missing), len(stray_ext)))
    ok = True
    for s in bad_abs:
        print("[check]   ABSOLUTE", s); ok = False
    for s in missing[:50]:
        print("[check]   MISSING", s); ok = False
    for s in stray_ext:
        print("[check]   STRAY-EXTERNAL", s); ok = False
    if ok:
        print("[check] OK: every media ref resolves inside site/, no absolute paths, "
              "only the Font Awesome stylesheet is external")
    else:
        sys.exit("[check] FAILED")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--media", action="store_true")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--picks", nargs="?", const=PICKS_DEFAULT, default=PICKS_DEFAULT,
                    help="owner per-row picks (default %s); the row source for the whole site "
                         "and the SEGUE clip in every section." % os.path.relpath(PICKS_DEFAULT, REPO))
    ap.add_argument("--apply-comparison-picks", dest="apply_cmp", action="store_true", default=True,
                    help="(default on) the picked clip replaces the SEGUE clip in A/B too")
    ap.add_argument("--no-apply-comparison-picks", dest="apply_cmp", action="store_false",
                    help="label A/B SEGUE as the picked clip but keep semantics off")
    a = ap.parse_args()
    if not any([a.select, a.media, a.html, a.all, a.check]):
        ap.error("choose at least one of --select --media --html --all --check")
    man = None
    if a.all or a.select:
        man = do_select(a.picks, a.apply_cmp)
    if a.all or a.media:
        do_media(man)
    if a.all or a.html:
        do_html(man)
    if a.check:
        do_check()


if __name__ == "__main__":
    main()
