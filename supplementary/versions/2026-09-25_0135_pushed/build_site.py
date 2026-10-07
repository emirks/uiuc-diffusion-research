#!/usr/bin/env python3
"""Build the SEGUE supplementary website (adapted from the Video-As-Prompt gh-pages page).

Run with the login-node python3.12 (login python3 is 3.6):

    /usr/bin/python3.12 supplementary/build_site.py --all

Stages (composable):
    --select   resolve every row/section from the collections JSON + pairs.json + data.js
               and write site_manifest.json (no media touched)
    --media    re-encode every clip into site/videos/<section>/<system>/<row_id>.mp4 with the
               human-study recipe (libx264 yuv420p crf26 preset medium -an +faststart -threads 2),
               dedup by source (hardlink), write site/media_manifest.json
    --html     render site/index.html from template.html ({{SECTIONS}} placeholder)
    --all      select, then media, then html
    --check    verify every <source src>/<img src>/href in index.html resolves inside site/,
               that no path is absolute, and that the only external URL is the Font Awesome CSS

Sources / rules (see supplementary/README.md):
    A  teg_user_study            (24)  SEGUE vs Base LTX-2 vs refVFX, both endpoints given
    B  vfx_transfer_user_study   (26)  SEGUE vs VAP vs VFXMaster vs refVFX, start given
    C  supplementary_teg not in A (33 candidates -> 32 shown; 1 has no v3-neutral gen)
    D  supplementary_vfx_transfer not in B (199)
SEGUE = arm dualforce_dcg_w6, NEUTRAL prompt, seed 42 (grid 03_neutral_v3 / 04_neutral_v3ed81).
A/B clips come verbatim from the human study's pairs.json (exactly what the VLM judge saw).
C/D SEGUE clips are resolved through data.js (g.videos["42"]) and taken from the store
(outputs/videos/grid_v3/... is a hardlink of the store file; the store path is recorded).
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
INDEX = os.path.join(SITE, "index.html")
SITE_MANIFEST = os.path.join(HERE, "site_manifest.json")
MEDIA_MANIFEST = os.path.join(SITE, "media_manifest.json")

COLLECTIONS = os.path.join(REPO, "eval_ladder/viewer/collections/neutral_effect_collections.json")
PAIRS = os.path.join(REPO, "misc/2026-09-22_user_study/pairs.json")
DATAJS = os.path.join(REPO, "outputs/reports/iclora_neutral_effect_v2/data.js")
STORE = "store/gens/032_dualforce_dcg_w6"
# owner's per-row picks (from the internal picker/); default path for --picks
PICKS_DEFAULT = os.path.join(REPO, "eval_ladder/viewer/collections/supplementary_picks.json")

NOVELTY_ORDER = ["seen", "unseen", "zero_shot"]
CONTENT_ORDER = ["same", "cross", "foreign"]
NOVELTY_DISP = {"seen": "seen", "unseen": "unseen", "zero_shot": "zero-shot"}

# opponent label -> the system slug/column used on the page
OPP_SLUG = {"Base LTX-2": "base_ltx2", "refVFX": "refvfx",
            "VAP": "vap", "VFXMaster": "vfxmaster"}
SYS_LABEL = {"segue": "SEGUE", "base_ltx2": "Base LTX-2", "refvfx": "refVFX",
             "vap": "VAP", "vfxmaster": "VFXMaster"}


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
    d = json.load(open(COLLECTIONS))
    return {c["id"]: c for c in d["collections"]}


def load_pairs():
    return json.load(open(PAIRS))


def load_datajs():
    raw = open(DATAJS).read()
    m = re.match(r"\s*window\.__NE_DATA__\s*=\s*", raw)
    body = raw[m.end():].rstrip()
    if body.endswith(";"):
        body = body[:-1]
    return json.loads(body)


def load_picks(path):
    """The internal picker's per-row picks as {collection_id: {item_id: pick}}; {} if absent.

    A pick = {card_key, gen_id, video (repo-rel), arm, arm_label, variant, seed, picked_at}.
    """
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


# --------------------------------------------------------------------------- select
def _ref_of(it):
    return it["inputs"]["refs"][0]["clip"]


def _ep_of(it):
    return it["inputs"]["endpoint"]


def resolve_segue(it, cards):
    """(store_path, ref_video, prefix_video, suffix_video, sided) or (None, reason)."""
    base = it["card_key"].split("|ref=")[0]
    ref = _ref_of(it)
    card = cards.get(base)
    if card is None:
        return None, "no-card:" + base
    for slot, sub in (("dualforce_dcg_w6_neutral_v3", "03_neutral_v3__dai"),
                      ("dualforce_dcg_w6_neutral_v3ed81", "04_neutral_v3ed81__dai")):
        for g in card["slots"].get(slot, []) or []:
            if g.get("ref") == ref and g.get("videos", {}).get("42"):
                bn = os.path.basename(g["videos"]["42"])
                sp = os.path.join(STORE, sub, "videos", bn)
                if not os.path.exists(apath(sp)):
                    return None, "store-missing:" + sp
                return {"segue": sp, "ref_video": g.get("ref_video"),
                        "prefix": card.get("prefix_video"),
                        "suffix": card.get("suffix_video"),
                        "sided": it["inputs"]["sided"]}, None
    return None, "no-gen"


def do_select(picks_path=None, apply_cmp=False):
    picks = load_picks(picks_path)
    cols = load_collections()
    pairs = load_pairs()
    cards = {c["key"]: c for c in load_datajs()["cards"]}
    pidx = defaultdict(list)
    for p in pairs:
        pidx[(p["task"], p["endpoint"], p["reference"])].append(p)

    excluded = []
    missing_sources = []

    def add_src(section, src, kind):
        fp = apath(src)
        if not os.path.exists(fp):
            missing_sources.append((section, kind, src))
        return relrepo(src)

    # ---- A: TEG comparisons ----
    A = cols["teg_user_study"]["items"]
    A_keys = {(_ep_of(it), _ref_of(it)) for it in A}
    a_rows = []
    for i, it in enumerate(A):
        ep, ref = _ep_of(it), _ref_of(it)
        ps = {p["opponent"]: p for p in pidx[("two", ep, ref)]}
        assert set(ps) == {"Base LTX-2", "refVFX"}, (ep, ref, set(ps))
        judge = it.get("judge", {}).get("transition_vs", {})
        assert judge and set(judge.values()) == {1.0}, (ep, ref, judge)
        base_p, ref_p = ps["Base LTX-2"], ps["refVFX"]
        seed_p = base_p
        rid = "teg_cmp_%03d__%s__%s" % (i, slug(ref), slug(ep))
        refclass = seed_p["gt_pool_class"]
        nov = "zero_shot"
        content = seed_p["content"]
        media = {
            "reference": {"src": add_src("A", seed_p["reference_clip"], "reference"),
                          "dest": "videos/A/reference/%s.mp4" % rid},
            "start": {"src": add_src("A", seed_p["start_still"], "start"),
                      "dest": "videos/A/start/%s.mp4" % rid},
            "end": {"src": add_src("A", seed_p["end_still"], "end"),
                    "dest": "videos/A/end/%s.mp4" % rid},
            "segue": {"src": add_src("A", seed_p["segue_clip"], "segue"),
                      "dest": "videos/A/segue/%s.mp4" % rid},
            "base_ltx2": {"src": add_src("A", base_p["opponent_clip"], "base_ltx2"),
                          "dest": "videos/A/base_ltx2/%s.mp4" % rid},
            "refvfx": {"src": add_src("A", ref_p["opponent_clip"], "refvfx"),
                       "dest": "videos/A/refvfx/%s.mp4" % rid},
        }
        pk = pick_for(picks, "teg_user_study", it["id"])
        row = {"row_id": rid, "endpoint": ep, "reference": ref,
               "ref_class": refclass, "novelty": nov, "content": content,
               "tier_words": tier_words(nov, content), "media": media,
               "src_collection": "teg_user_study", "item_id": it["id"],
               "card_key": it["card_key"], "pick": pk, "pick_applied": False}
        if pk and apply_cmp:
            row["media"]["segue"]["src"] = add_src("A", pk["video"], "segue")
            row["pick_applied"] = True
        a_rows.append(row)

    # ---- B: VFX transfer comparisons ----
    B = cols["vfx_transfer_user_study"]["items"]
    B_keys = {(_ep_of(it), _ref_of(it)) for it in B}
    b_rows = []
    for i, it in enumerate(B):
        ep, ref = _ep_of(it), _ref_of(it)
        ps = {p["opponent"]: p for p in pidx[("one", ep, ref)]}
        assert set(ps) == {"VAP", "VFXMaster", "refVFX"}, (ep, ref, set(ps))
        seed_p = ps["VAP"]
        rid = "vfx_cmp_%03d__%s__%s" % (i, slug(ref), slug(ep))
        refclass = seed_p["gt_pool_class"]
        nov, content = "zero_shot", seed_p["content"]
        media = {
            "reference": {"src": add_src("B", seed_p["reference_clip"], "reference"),
                          "dest": "videos/B/reference/%s.mp4" % rid},
            "start": {"src": add_src("B", seed_p["start_still"], "start"),
                      "dest": "videos/B/start/%s.mp4" % rid},
            "segue": {"src": add_src("B", seed_p["segue_clip"], "segue"),
                      "dest": "videos/B/segue/%s.mp4" % rid},
            "vap": {"src": add_src("B", ps["VAP"]["opponent_clip"], "vap"),
                    "dest": "videos/B/vap/%s.mp4" % rid},
            "vfxmaster": {"src": add_src("B", ps["VFXMaster"]["opponent_clip"], "vfxmaster"),
                          "dest": "videos/B/vfxmaster/%s.mp4" % rid},
            "refvfx": {"src": add_src("B", ps["refVFX"]["opponent_clip"], "refvfx"),
                       "dest": "videos/B/refvfx/%s.mp4" % rid},
        }
        pk = pick_for(picks, "vfx_transfer_user_study", it["id"])
        row = {"row_id": rid, "endpoint": ep, "reference": ref,
               "ref_class": refclass, "novelty": nov, "content": content,
               "tier_words": tier_words(nov, content), "media": media,
               "src_collection": "vfx_transfer_user_study", "item_id": it["id"],
               "card_key": it["card_key"], "pick": pk, "pick_applied": False}
        if pk and apply_cmp:
            row["media"]["segue"]["src"] = add_src("B", pk["video"], "segue")
            row["pick_applied"] = True
        b_rows.append(row)

    # ---- C / D: SEGUE results, grouped by reference ----
    def grouped(cid, keys, prefix, section):
        items = [it for it in cols[cid]["items"] if (_ep_of(it), _ref_of(it)) not in keys]
        # resolve & drop rows with no v3-neutral SEGUE gen
        resolved = []
        for it in items:
            r, err = resolve_segue(it, cards)
            if r is None:
                excluded.append({"section": section, "card_key": it["card_key"], "reason": err})
                continue
            resolved.append((it, r))
        # group by reference clip
        by_ref = defaultdict(list)
        for it, r in resolved:
            by_ref[_ref_of(it)].append((it, r))
        # order references by #endpoints desc then name
        ref_order = sorted(by_ref, key=lambda rr: (-len(by_ref[rr]), rr))
        blocks = []
        cell_i = 0
        for ref in ref_order:
            entries = by_ref[ref]
            # tier words / class from any entry (reference class is constant)
            tier = entries[0][0]["inputs"].get("tier") or {}
            refclass = entries[0][0]["inputs"]["refs"][0]["cls"]
            block_id = "%s_ref__%s" % (prefix, slug(ref))
            # order cells by tier words then endpoint
            def ck(e):
                t = e[0]["inputs"].get("tier") or {}
                nov = (t.get("novelty") or ["zero_shot"])[0]
                con = (t.get("content") or ["foreign"])[0]
                return tier_sort_key(nov, con, _ep_of(e[0]))
            entries = sorted(entries, key=ck)
            # reference video (block-addressed, shared)
            ref_src = add_src(section, entries[0][1]["ref_video"], "reference")
            cells = []
            for it, r in entries:
                ep = _ep_of(it)
                t = it["inputs"].get("tier") or {}
                nov = (t.get("novelty") or ["zero_shot"])[0]
                con = (t.get("content") or ["foreign"])[0]
                rid = "%s_res_%03d__%s__%s" % (prefix, cell_i, slug(ref), slug(ep))
                cell_i += 1
                media = {
                    "start": {"src": add_src(section, r["prefix"], "start"),
                              "dest": "videos/%s/start/%s.mp4" % (section, rid)},
                    "segue": {"src": add_src(section, r["segue"], "segue"),
                              "dest": "videos/%s/segue/%s.mp4" % (section, rid)},
                }
                if section == "C":
                    assert r["sided"] == "two" and r["suffix"], it["card_key"]
                    media["end"] = {"src": add_src(section, r["suffix"], "end"),
                                    "dest": "videos/%s/end/%s.mp4" % (section, rid)}
                pk = pick_for(picks, cid, it["id"])
                if pk:
                    media["segue"]["src"] = add_src(section, pk["video"], "segue")
                cells.append({"row_id": rid, "endpoint": ep, "novelty": nov,
                              "content": con, "tier_words": tier_words(nov, con),
                              "media": media, "src_collection": cid,
                              "item_id": it["id"], "card_key": it["card_key"], "pick": pk})
            blocks.append({"block_id": block_id, "reference": ref, "ref_class": refclass,
                           "reference_media": {"src": ref_src,
                                               "dest": "videos/%s/reference/%s.mp4"
                                                       % (section, block_id)},
                           "cells": cells})
        return blocks

    c_blocks = grouped("supplementary_teg", A_keys, "teg", "C")
    d_blocks = grouped("supplementary_vfx_transfer", B_keys, "vfx", "D")

    # ---- stills (endpoint frames): NOT shown on the page (which plays the endpoint
    # clips), but generated cheaply so the registry `stills` link resolves and the
    # owner has frame 0 (start) / frame 8 (end) if they restyle. Mirrors build_media.py.
    stills = []

    def add_still(section, src, frame, kind, rid):
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

    manifest = {
        "generated": _now(),
        "segue_arm": "dualforce_dcg_w6 (neutral prompt, seed 42; grids 03_neutral_v3 / 04_neutral_v3ed81)",
        "recipe": "libx264 yuv420p crf26 preset medium -an +movflags faststart -threads 2 (native fps/frames, no audio)",
        "sources": {"collections": relrepo(COLLECTIONS), "pairs": relrepo(PAIRS),
                    "data_js": relrepo(DATAJS)},
        "sections": {
            "A": {"title": "Comparisons — Transition Effect Generation (both endpoints given)",
                  "systems": ["segue", "base_ltx2", "refvfx"], "rows": a_rows},
            "B": {"title": "Comparisons — Visual Effect Transfer (start frame given)",
                  "systems": ["segue", "vap", "vfxmaster", "refvfx"], "rows": b_rows},
            "C": {"title": "SEGUE results — Transition Effect Generation",
                  "blocks": c_blocks},
            "D": {"title": "SEGUE results — Visual Effect Transfer",
                  "blocks": d_blocks},
        },
        "stills": stills,
        "excluded": excluded,
        "counts": {
            "A_rows": len(a_rows), "B_rows": len(b_rows),
            "C_blocks": len(c_blocks), "C_cells": sum(len(b["cells"]) for b in c_blocks),
            "D_blocks": len(d_blocks), "D_cells": sum(len(b["cells"]) for b in d_blocks),
            "stills": len(stills),
            "excluded": len(excluded),
        },
    }
    json.dump(manifest, open(SITE_MANIFEST, "w"), indent=1)
    print("[select] A=%d rows  B=%d rows  C=%d blocks/%d cells  D=%d blocks/%d cells  excluded=%d"
          % (len(a_rows), len(b_rows), len(c_blocks), manifest["counts"]["C_cells"],
             len(d_blocks), manifest["counts"]["D_cells"], len(excluded)))
    for e in excluded:
        print("[select] EXCLUDED %s/%s -> %s" % (e["section"], e["card_key"], e["reason"]))
    print("[select] wrote", SITE_MANIFEST)
    return manifest


def _now():
    import datetime
    return datetime.datetime.now().replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- media
def _iter_media(manifest):
    """yield (dest_rel, src_rel) for every media element in the manifest."""
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
    # mjpeg -q:v 3 (~JPEG q90) exactly as scripts/user_study/build_media.py extracts the
    # study stills, so these are consistent with the human-study frames.
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


def do_media(manifest=None):
    if manifest is None:
        manifest = json.load(open(SITE_MANIFEST))
    ffmpeg = find_ffmpeg()
    old = json.load(open(MEDIA_MANIFEST)) if os.path.exists(MEDIA_MANIFEST) else {}

    # unique source -> list of dest_rel
    by_src = defaultdict(list)
    for dest, src in _iter_media(manifest):
        by_src[src].append(dest)
    for s in by_src:
        by_src[s] = sorted(set(by_src[s]))

    total_dests = sum(len(v) for v in by_src.values())
    print("[media] %d unique sources -> %d dest files" % (len(by_src), total_dests))

    manifest_out = {}
    enc = skip = link = 0
    lock_note = []

    def encode_canonical(src):
        """encode/probe/sha the canonical dest for a source; return (dest0, entry, status)."""
        nonlocal enc, skip
        dest0 = by_src[src][0]
        out0 = os.path.join(SITE, dest0)
        src_abs = apath(src)
        if os.path.exists(out0) and dest0 in old and old[dest0].get("source") == src:
            return dest0, old[dest0], "skip"
        _encode(ffmpeg, src_abs, out0)
        frames, fps, w, h = _probe(ffmpeg, out0)
        entry = {"source": src, "sha256": _sha256(src_abs), "frames": frames,
                 "fps": fps, "w": w, "h": h, "bytes": os.path.getsize(out0)}
        return dest0, entry, "enc"

    # encode unique sources, <=3 concurrent ffmpeg
    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(encode_canonical, src): src for src in by_src}
        done = 0
        for fut in futs:
            pass
        for fut in list(futs):
            dest0, entry, status = fut.result()
            src = futs[fut]
            results[src] = (dest0, entry, status)
            if status == "enc":
                enc += 1
            else:
                skip += 1
            done += 1
            if done % 100 == 0 or done == len(by_src):
                print("[media] sources %d/%d (encoded=%d skipped=%d)"
                      % (done, len(by_src), enc, skip))

    # write canonical entries, then hardlink the duplicates
    for src, (dest0, entry, status) in results.items():
        manifest_out[dest0] = entry
        for dest in by_src[src][1:]:
            out = os.path.join(SITE, dest)
            out0 = os.path.join(SITE, dest0)
            if os.path.exists(out) and dest in old and old[dest].get("source") == src:
                manifest_out[dest] = old[dest]
                skip += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out):
                os.remove(out)
            try:
                os.link(out0, out)
            except OSError:
                shutil.copyfile(out0, out)
            e2 = dict(entry)
            e2["bytes"] = os.path.getsize(out)
            manifest_out[dest] = e2
            link += 1

    # ---- stills (deduped by (source, frame); <=3 concurrent) ----
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
        _extract_still(ffmpeg, src_abs, frame, out0)
        _, _, w, h = _probe(ffmpeg, out0)
        entry = {"source": src, "sha256": _sha256(src_abs), "kind": kind0,
                 "frame": frame, "w": w, "h": h, "bytes": os.path.getsize(out0)}
        return key, dest0, entry, "enc"

    senc = sskip = slink = 0
    sresults = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(extract_canonical, k): k for k in still_by_key}
        for fut in list(futs):
            key, dest0, entry, status = fut.result()
            sresults[key] = (dest0, entry, status)
            senc += status == "enc"
            sskip += status == "skip"
    for key, (dest0, entry, status) in sresults.items():
        manifest_out[dest0] = entry
        for dest, _kind in still_by_key[key][1:]:
            out = os.path.join(SITE, dest)
            out0 = os.path.join(SITE, dest0)
            if os.path.exists(out) and dest in old and old[dest].get("source") == key[0] \
                    and old[dest].get("frame") == key[1]:
                manifest_out[dest] = old[dest]
                sskip += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out):
                os.remove(out)
            try:
                os.link(out0, out)
            except OSError:
                shutil.copyfile(out0, out)
            e2 = dict(entry)
            e2["bytes"] = os.path.getsize(out)
            manifest_out[dest] = e2
            slink += 1
    print("[media] stills: unique=%d extracted=%d skipped=%d hardlinked=%d"
          % (len(still_by_key), senc, sskip, slink))

    json.dump(manifest_out, open(MEDIA_MANIFEST, "w"), indent=1)
    total = sum(e["bytes"] for e in manifest_out.values())
    print("[media] files=%d encoded=%d skipped=%d hardlinked=%d  total=%.1f MB"
          % (len(manifest_out), enc, skip, link, total / 1e6))
    print("[media] wrote", MEDIA_MANIFEST)
    return manifest_out


# --------------------------------------------------------------------------- html
def _media_div(dest, extra_class="", badge=None):
    b = '<span class="seg-badge">%s</span>' % html.escape(badge) if badge else ""
    return ('<div class="seg-media %s">%s<video class="lazy-video" preload="none" '
            'muted playsinline loop data-autoplay="true">'
            '<source src="%s" type="video/mp4"/></video></div>'
            % (extra_class, b, dest))


def _noend_div():
    return ('<div class="seg-media seg-noend"><span class="seg-badge">no end</span>'
            '<span class="seg-xlabel">no end given</span></div>')


def _unit(inner, label, extra_class=""):
    lab = '<div class="seg-label">%s</div>' % html.escape(label) if label else ""
    return '<div class="seg-unit %s">%s%s</div>' % (extra_class, inner, lab)


def _pick_suffix(pick):
    p = pick or {}
    return "%s · %s · s%s" % (p.get("arm_label", ""), p.get("variant", ""), p.get("seed"))


CLASS_NAMES = os.path.join(HERE, "class_names.json")

# public section titles (order top->bottom: C, A, D, B); no internal codenames
SECTION_ORDER = ["C", "A", "D", "B"]

# owner edits to the gallery block order (render time only; the manifest keeps every block).
# 2026-09-25: drop the first Hero flight block (hero_flight_5; keep the dog one, hero_flight_0)
# and carry Water bending up, directly under the two Flame blocks.
# 2026-09-25 01:00 (owner): also drop the remaining Hero flight and Display transition blocks; second Shadow smoke
# (shadow_smoke_7) directly under the first (shadow_smoke_0); the 5th cell of the first Shadow smoke removed; the
# one-cell Shadow smoke block (shadow_smoke_1) merged into shadow_smoke_7 (its cell appended, its reference dropped).
BLOCK_DROP = {"C": {"teg_ref__hero_flight_5", "teg_ref__hero_flight_0", "teg_ref__display_transition_2"}, "D": set()}
BLOCK_MOVE_AFTER = {"C": [("teg_ref__water_bending_3", "teg_ref__flame_transition_3"),
                          ("teg_ref__shadow_smoke_7", "teg_ref__shadow_smoke_0")], "D": []}
BLOCK_MERGE_INTO = {"C": [("teg_ref__shadow_smoke_1", "teg_ref__shadow_smoke_7")], "D": []}   # (source, target)
CELL_DROP = {"C": {"teg_ref__shadow_smoke_0": [4],                                              # 0-based, rendered order,
                   "teg_ref__shadow_smoke_7": [2, 3]}, "D": {}}                                  # applied AFTER merges


def _arrange_blocks(sec, key):
    import copy
    blocks = [copy.deepcopy(b) for b in sec["blocks"] if b["block_id"] not in BLOCK_DROP.get(key, set())]
    for src, dst in BLOCK_MERGE_INTO.get(key, []):
        by = {b["block_id"]: b for b in blocks}
        if src not in by or dst not in by:
            print("[html] WARNING: block merge %s -> %s: not found in %s" % (src, dst, key), file=sys.stderr)
            continue
        by[dst]["cells"].extend(by[src]["cells"])
        blocks = [b for b in blocks if b["block_id"] != src]
    for bid, idxs in CELL_DROP.get(key, {}).items():
        for b in blocks:
            if b["block_id"] == bid:
                b["cells"] = [c for i, c in enumerate(b["cells"]) if i not in set(idxs)]
    for mover, anchor in BLOCK_MOVE_AFTER.get(key, []):
        ids = [b["block_id"] for b in blocks]
        if mover not in ids or anchor not in ids:
            print("[html] WARNING: block move %s after %s: not found in %s" % (mover, anchor, key), file=sys.stderr)
            continue
        b = blocks.pop(ids.index(mover))
        ids = [x["block_id"] for x in blocks]
        blocks.insert(ids.index(anchor) + 1, b)
    dropped = len(sec["blocks"]) - len(blocks)
    print("[html] section %s: %d blocks rendered, %d dropped (kept in manifest)" % (key, len(blocks), dropped))
    return {"blocks": blocks}
SECTION_TITLE = {
    "C": "Transition effect generation",
    "A": "Transition effect generation \u2014 comparison with previous work",
    "D": "Visual effect transfer",
    "B": "Visual effect transfer \u2014 comparison with previous work",
}
# 2026-09-25 (owner): table of contents on top (as on rave-video.github.io/supp) + a Guidance section with
# placeholder rows (a few rows per guidance: NRG, empty null which leaks the scene); rows to be picked later.
GUIDANCE_PARTS = [
    ("nrg", "Null-reference guidance (NRG)", "A few rows contrasting SEGUE with and without NRG."),
    ("emptynull", "Empty null", "A few rows with the empty null in place of the dissolve; the reference scene leaks in."),
]
# TEG comparisons: keep only these 6 rows (transition class -> endpoint), in this order
A_KEEP = [
    ("firelava", "davis_blackswan_rhino"),
    ("display_transition", "davis_bear_elephant"),
    ("flying_cam_transition", "davis_bear_elephant"),
    ("flying_cam_transition", "davis_blackswan_rhino"),
    ("firelava", "air_bending_1"),
    ("melt_transition", "hero_flight_6"),
]


def _auto_class_name(cls):
    """Auto human name: strip 'ed.' prefix and hash/suffix tokens, underscores->spaces, sentence case."""
    s = cls[3:] if cls.startswith("ed.") else cls
    s = s.split(".")[0].replace("_", " ").strip()
    return (s[:1].upper() + s[1:].lower()) if s else cls


def load_class_names(all_classes):
    """Read supplementary/class_names.json (owner-editable); auto-seed any missing class; persist."""
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


def _render_A(sec, names):
    out = []
    for r in sec["rows"]:
        m = r["media"]
        stack = ('<div class="seg-stack">%s%s</div>'
                 % (_media_div(m["start"]["dest"], "seg-endpoint", badge="start"),
                    _media_div(m["end"]["dest"], "seg-endpoint", badge="end")))
        rowlabel = '<div class="seg-rowlabel">%s</div>' % html.escape(human_name(r["ref_class"], names))
        cells = [
            _unit(_media_div(m["reference"]["dest"]), "reference", "seg-ref"),
            _unit(stack, "endpoints", "seg-endstack"),
            _unit(_media_div(m["segue"]["dest"]), SYS_LABEL["segue"], "seg-gen"),
            _unit(_media_div(m["base_ltx2"]["dest"]), SYS_LABEL["base_ltx2"], "seg-gen"),
            _unit(_media_div(m["refvfx"]["dest"]), SYS_LABEL["refvfx"], "seg-gen"),
        ]
        out.append('<div class="seg-strip">%s<div class="seg-cellsrow">%s</div></div>'
                   % (rowlabel, "".join(cells)))
    return "".join(out)


def _render_B(sec, names):
    out = []
    for r in sec["rows"]:
        m = r["media"]
        rowlabel = '<div class="seg-rowlabel">%s</div>' % html.escape(human_name(r["ref_class"], names))
        cells = [
            _unit(_media_div(m["reference"]["dest"]), "reference", "seg-ref"),
            _unit(_media_div(m["start"]["dest"], badge="start"), "start", "seg-startfull"),
            _unit(_media_div(m["segue"]["dest"]), SYS_LABEL["segue"], "seg-gen"),
            _unit(_media_div(m["vap"]["dest"]), SYS_LABEL["vap"], "seg-gen"),
            _unit(_media_div(m["vfxmaster"]["dest"]), SYS_LABEL["vfxmaster"], "seg-gen"),
            _unit(_media_div(m["refvfx"]["dest"]), SYS_LABEL["refvfx"], "seg-gen"),
        ]
        out.append('<div class="seg-strip">%s<div class="seg-cellsrow">%s</div></div>'
                   % (rowlabel, "".join(cells)))
    return "".join(out)


def _render_grouped(sec, section, names):
    out = []
    for blk in sec["blocks"]:
        block_label = ('<div class="seg-rowlabel">%s</div>'
                       % html.escape(human_name(blk["ref_class"], names)))
        ref_unit = _unit(_media_div(blk["reference_media"]["dest"]), "reference", "seg-ref")
        cell_html = []
        for cell in blk["cells"]:
            m = cell["media"]
            if section == "C":
                stack = ('<div class="seg-stack">%s%s</div>'
                         % (_media_div(m["start"]["dest"], "seg-endpoint", badge="start"),
                            _media_div(m["end"]["dest"], "seg-endpoint", badge="end")))
            else:
                stack = ('<div class="seg-stack">%s%s</div>'
                         % (_media_div(m["start"]["dest"], "seg-endpoint", badge="start"),
                            _noend_div()))
            inner = ('<div class="seg-cell">%s%s</div>'
                     % (_unit(stack, "endpoint", "seg-endstack"),
                        _unit(_media_div(m["segue"]["dest"]), SYS_LABEL["segue"], "seg-gen")))
            cell_html.append('<div class="seg-cellwrap">%s</div>' % inner)
        out.append('<div class="seg-blockwrap">%s<div class="seg-block">%s'
                   '<div class="seg-cells">%s</div></div></div>'
                   % (block_label, ref_unit, "".join(cell_html)))
    return "".join(out)


SECTION_DESC = {
    "A": "Both endpoint frames and a reference effect video are given. Left: the reference "
         "effect video, then the given start and end endpoints (stacked). Right: the generated "
         "transition from SEGUE and the two prior-work systems it was compared against.",
    "B": "One endpoint frame and a reference effect video are given. Left: the reference effect "
         "video and the given start endpoint. Right: the generated result from SEGUE and the "
         "three prior-work systems it was compared against.",
    "C": "Transition-effect-generation results from SEGUE, grouped by reference effect video. "
         "Each cell shows the given start and end endpoints (stacked) and the generated transition.",
    "D": "Visual-effect-transfer results from SEGUE, grouped by reference effect video. Each cell "
         "shows the given start endpoint (no end endpoint is given) and the generated result.",
}
SECTION_FOOT = {
    "A": "SEGUE, neutral prompt, seed 42. The transition-effect cases where SEGUE won the blind "
         "study against every prior-work system; the clips shown are exactly those the study presented.",
    "B": "SEGUE, neutral prompt, seed 42. The visual-effect-transfer cases where SEGUE won the blind "
         "study against every prior-work system; the clips shown are exactly those the study presented.",
    "C": "SEGUE, neutral prompt, seed 42.",
    "D": "SEGUE, neutral prompt, seed 42.",
}


def _filter_A(rows):
    """Keep only the 6 A_KEEP rows, in that order. Returns (kept, missing_pairs, dropped_rows)."""
    idx = {}
    for r in rows:
        idx[(r["ref_class"], r["endpoint"])] = r
    kept, missing = [], []
    for key in A_KEEP:
        r = idx.get(key)
        if r:
            kept.append(r)
        else:
            missing.append(key)
    keep_ids = {id(r) for r in kept}
    dropped = [r for r in rows if id(r) not in keep_ids]
    return kept, missing, dropped


def do_html(manifest=None):
    if manifest is None:
        manifest = json.load(open(SITE_MANIFEST))
    tpl = open(TEMPLATE).read()
    assert tpl.count("{{SECTIONS}}") == 1, "template must have exactly one {{SECTIONS}}"
    S = manifest["sections"]

    all_classes = set()
    for r in S["A"]["rows"] + S["B"]["rows"]:
        all_classes.add(r["ref_class"])
    for blk in S["C"]["blocks"] + S["D"]["blocks"]:
        all_classes.add(blk["ref_class"])
    names = load_class_names(all_classes)

    a_kept, a_missing, a_dropped = _filter_A(S["A"]["rows"])
    if a_missing:
        print("[html] WARNING: A_KEEP rows not found in manifest: %s" % a_missing, file=sys.stderr)
        near = sorted({(r["ref_class"], r["endpoint"]) for r in S["A"]["rows"]})
        print("[html] nearest A candidates: %s" % near, file=sys.stderr)
    a_sec = {"rows": a_kept}
    print("[html] section A: kept %d of %d rows; dropped %d (kept in manifest)"
          % (len(a_kept), len(S["A"]["rows"]), len(a_dropped)))

    renderers = {"A": (lambda s: _render_A(s, names), a_sec),
                 "B": (lambda s: _render_B(s, names), S["B"]),
                 "C": (lambda s: _render_grouped(s, "C", names), _arrange_blocks(S["C"], "C")),
                 "D": (lambda s: _render_grouped(s, "D", names), _arrange_blocks(S["D"], "D"))}
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
    toc.append("</ul></li></ul></nav>")
    parts.append("".join(toc))
    for key in SECTION_ORDER:
        render, sec = renderers[key]
        parts.append('<div class="section-title" id="sec-%s">%s</div>' % (key, html.escape(SECTION_TITLE[key])))
        parts.append("<p>%s</p>" % html.escape(SECTION_DESC[key]))
        parts.append('<div class="seg-section">%s</div>' % render(sec))
        parts.append('<p class="footnotes">%s</p>' % html.escape(SECTION_FOOT[key]))
    parts.append('<div class="section-title" id="sec-guidance">Guidance</div>')
    for gid, gtitle, gdesc in GUIDANCE_PARTS:
        parts.append('<div class="subsection-title seg-subtitle" id="sec-guidance-%s">%s</div>' % (gid, html.escape(gtitle)))
        parts.append("<p>%s</p>" % html.escape(gdesc))
        parts.append('<div class="seg-placeholder">rows to come</div>')
    body = "\n".join(parts)
    out = tpl.replace("{{SECTIONS}}", body)
    os.makedirs(SITE, exist_ok=True)
    open(INDEX, "w").write(out)
    print("[html] wrote %s (%.0f KB)" % (INDEX, len(out) / 1024))
    return out


# --------------------------------------------------------------------------- check
def do_check():
    raw = open(INDEX).read()
    # scan only real markup: drop the retained VAP <script> bodies (they carry inert
    # JS template literals like ${oursSrc} that are never executed on this page).
    text = re.sub(r"<script\b.*?</script>", "", raw, flags=re.DOTALL)
    srcs = re.findall(r'<source[^>]+src="([^"]+)"', text)
    imgs = re.findall(r'<img[^>]+src="([^"]+)"', text)
    hrefs = re.findall(r'href="([^"]+)"', text)
    bad_abs = []
    missing = []
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
    ap.add_argument("--picks", nargs="?", const=PICKS_DEFAULT, default=None,
                    help="apply the internal picker's per-row picks: in C/D the picked gen "
                         "becomes the SEGUE clip (relabeled); bare flag uses the default path "
                         "(%s). Picks change the transcode SOURCE, so they take effect through "
                         "--select (+ --media to re-encode); --html alone renders the current "
                         "manifest." % os.path.relpath(PICKS_DEFAULT, REPO))
    ap.add_argument("--apply-comparison-picks", action="store_true",
                    help="also apply picks to A/B (otherwise the judged clip stays; the cell "
                         "then says 'owner pick, not the judged clip')")
    a = ap.parse_args()
    if not any([a.select, a.media, a.html, a.all, a.check]):
        ap.error("choose at least one of --select --media --html --all --check")
    if a.picks and not (a.all or a.select):
        print("[warn] --picks takes effect at --select time (it changes the transcode source); "
              "with --html alone the existing manifest is rendered unchanged. Use --all --picks "
              "(or --select --media --html --picks) to apply picks end to end.", file=sys.stderr)
    man = None
    if a.all or a.select:
        man = do_select(a.picks, a.apply_comparison_picks)
    if a.all or a.media:
        do_media(man)
    if a.all or a.html:
        do_html(man)
    if a.check:
        do_check()


if __name__ == "__main__":
    main()
