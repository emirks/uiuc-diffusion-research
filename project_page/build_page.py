#!/usr/bin/env python3
"""Build the SEGUE project page — a simple, static, local-only landing page.

Run with the login-node python3.12 (login python3 is 3.6):

    /usr/bin/python3.12 project_page/build_page.py            # --html then --check
    /usr/bin/python3.12 project_page/build_page.py --html
    /usr/bin/python3.12 project_page/build_page.py --check
    /usr/bin/python3.12 project_page/build_page.py --candidates

What appears on the page is driven by `picks.json` (owner-editable). The media and the
block/cell/row metadata are read from the supplementary build's `site_manifest.json`; the
`site_plan.json` curation plan decides which blocks/cells/rows are VISIBLE (not hidden). This
script never re-encodes media and never writes anything under `supplementary/` or `eval_ladder/`
— it only reads them, reuses the supplementary renderers by import, and symlinks the clips/stills.

    --html        render the page(s) from the template(s) (fills {{HEADER}} + {{SECTIONS}})
    --check       verify every <source data-src>/<img src> on the page resolves inside site/
                  (through the symlinks); print counts and the total MB of referenced media
    --candidates  (re)write CANDIDATES.md — the menu of every VISIBLE block/cell/row the owner
                  can put in picks.json, in plan order
    --theme       dark|light|both (default both). dark -> template.html -> site/index.html;
                  light -> template_light.html -> site/light.html (same dir, identical media).
"""
import argparse
import html
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))        # project_page/
REPO = os.path.dirname(HERE)
SUPP = os.path.join(REPO, "supplementary")

# reuse the supplementary renderers verbatim (no re-implementation of the box/strip markup)
sys.path.insert(0, SUPP)
import build_site as bs                                   # noqa: E402
bs.PROD = True                                            # clean markup (hidden items would be dropped)

SITE = os.path.join(HERE, "site")
TEMPLATE = os.path.join(HERE, "template.html")            # dark variant
TEMPLATE_LIGHT = os.path.join(HERE, "template_light.html")  # light variant (PixelDiT-style)
INDEX = os.path.join(SITE, "index.html")                  # dark -> site/index.html
LIGHT_INDEX = os.path.join(SITE, "light.html")            # light -> site/light.html (same dir)
DEFAULT_ACCENT = "#4f46e5"       # deep indigo
DEFAULT_ACCENT_DARK = "#3730a3"  # darker indigo
CONFIG = os.path.join(HERE, "config.json")
PICKS = os.path.join(HERE, "picks.json")
CANDIDATES = os.path.join(HERE, "CANDIDATES.md")
MANIFEST = os.path.join(SUPP, "site_manifest.json")
SITE_PLAN = os.path.join(REPO, "eval_ladder/viewer/collections/site_plan.json")
LOCAL_CLASS_NAMES = os.path.join(HERE, "class_names.json")

# redirect build_site's class-name cache to a LOCAL copy so load_class_names never writes under
# supplementary/ (it rewrites the file if a class is missing).
if os.path.exists(bs.CLASS_NAMES):
    shutil.copyfile(bs.CLASS_NAMES, LOCAL_CLASS_NAMES)
bs.CLASS_NAMES = LOCAL_CLASS_NAMES

TLDR = ("One-shot transition effect generation: given a reference clip that shows a transition "
        "effect and two endpoint clips, generate the transition that applies that effect between "
        "them. SEGUE reads the effect strictly from the visual reference: visual reference "
        "isolation in training and null-reference guidance at inference, with no effect "
        "description in the prompt. One model covers two-sided transitions and single-sided "
        "visual effect transfer, where only the start clip is given.")

METHOD_P1 = ("Training pairs the inputs with effect-free neutral prompts and applies a one-way "
             "attention mask, so the reference stays independent of the target and the effect can "
             "only be read from the visual demonstration.")
METHOD_P2 = ("At inference the unconditional path is a dissolve of the endpoints that adds no "
             "motion or material of its own, so guidance isolates and amplifies the visual "
             "transition effect.")

LEGEND = ('<div class="seg-legend">'
          '<span><span class="sw sw-in"></span>given inputs (reference &amp; endpoints)</span>'
          '<span><span class="sw sw-gen"></span>generated clips (SEGUE &amp; prior work)</span>'
          '</div>')

# header buttons: (key, label, font-awesome icon, link-key); BibTeX scrolls to the bibtex block
BUTTONS = [
    ("paper", "arXiv", "ai ai-arxiv", "paper"),
    ("code", "Code", "fa-brands fa-github", "code"),
    ("model", "Model", "hf", "model"),
    ("dataset", "Dataset", "hf", "dataset"),
    ("supplementary", "Supplementary", "fa-solid fa-film", "supplementary"),
    ("bibtex", "BibTeX", "fa-solid fa-quote-right", None),
]
# Hugging Face has no Font Awesome glyph; inline the official logo mark (yellow face) as SVG.
HF_SVG = '<svg class="icon hf" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 95 88" fill="none" aria-hidden="true"> <path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z" /> <path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z" /> <path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z" /> <path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z" /> <path fill="#3A3B45" fill-rule="evenodd" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z" clip-rule="evenodd" /> <path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z" /> <path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z" /> <path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z" /> <path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z" /> </svg>'


# --------------------------------------------------------------------------- loaders / indexes
def _load(p):
    with open(p) as f:
        return json.load(f)


def _plan_sec(plan, sec):
    return (plan.get("sections") or {}).get(sec) or {}


def _block_hidden(plan, sec, bid):
    ps = _plan_sec(plan, sec)
    return bid in set(ps.get("hidden_blocks") or []) or bid in (ps.get("merge_into") or {})


def _hidden_cells(plan, sec, bid):
    return set((_plan_sec(plan, sec).get("hidden_cells") or {}).get(bid) or [])


def _row_hidden(plan, sec, iid):
    return iid in set(_plan_sec(plan, sec).get("hidden_rows") or [])


def _block_cells(blocks_idx, plan, sec, bid):
    """All cells belonging to a block, INCLUDING cells of any block merged into it."""
    cells = list(blocks_idx[bid]["cells"])
    for src, dst in (_plan_sec(plan, sec).get("merge_into") or {}).items():
        if dst == bid and src in blocks_idx:
            cells += list(blocks_idx[src]["cells"])
    return cells


def _visible_cells(blocks_idx, plan, sec, bid):
    """Visible cells of a block in plan order (cell_order first, then the rest; hidden removed)."""
    ps = _plan_sec(plan, sec)
    cells = _block_cells(blocks_idx, plan, sec, bid)
    cmap = {c["item_id"]: c for c in cells}
    order = bs._apply_order([c["item_id"] for c in cells], (ps.get("cell_order") or {}).get(bid))
    hc = _hidden_cells(plan, sec, bid)
    return [cmap[i] for i in order if i not in hc]


# --------------------------------------------------------------------------- render: header
def _title_html(cfg):
    """Title with a leading 'SEGUE' token wrapped in <span class="pp-accent"> — emitted in BOTH
    themes. The dark template leaves .pp-accent unstyled (inherits white, so no visual change);
    the light template colours it with the accent."""
    esc = html.escape(cfg["title"])
    lead = "SEGUE"
    if esc.startswith(lead):
        return '<span class="pp-accent">%s</span>%s' % (lead, esc[len(lead):])
    return esc


def render_header(cfg, theme="dark"):
    links = cfg.get("links") or {}
    authors = cfg.get("authors") or []
    affil = cfg.get("affiliations") or {}
    supp = links.get("supplementary", "") or "#"
    title_html = _title_html(cfg)

    if theme == "light":
        return _render_header_light(links, authors, affil, supp, title_html)

    # ---- dark: markup unchanged; only the title now carries the accent span ----
    au = []
    for a in authors:
        sups = "".join("<sup>%s</sup>" % html.escape(str(n)) for n in (a.get("aff") or []))
        au.append('<span class="author">%s%s</span>' % (html.escape(a["name"]), sups))
    authors_html = "".join(au)

    aff = []
    for k in sorted(affil, key=lambda x: int(x) if str(x).isdigit() else x):
        aff.append('<span class="a"><sup>%s</sup>%s</span>'
                   % (html.escape(str(k)), html.escape(affil[k])))
    affil_html = "".join(aff)

    btns = []
    for key, label, icon, lk in BUTTONS:
        ic = HF_SVG if icon == "hf" else '<i class="icon %s" aria-hidden="true"></i>' % icon
        if key == "bibtex":
            btns.append('<a class="btn" href="#bibtex">%s%s</a>' % (ic, html.escape(label)))
            continue
        url = links.get(lk, "") or ""
        title = "Hugging Face" if key == "model" else label
        if url:
            btns.append('<a class="btn" href="%s" title="%s">%s%s</a>'
                        % (html.escape(url, quote=True), html.escape(title, quote=True),
                           ic, html.escape(label)))
        else:
            btns.append('<span class="btn btn-disabled" title="coming soon">%s%s'
                        '<span class="pp-soon">soon</span></span>' % (ic, html.escape(label)))
    buttons_html = "".join(btns)

    return ('<div class="pp-header">'
            '<h1 class="pp-title">%s</h1>'
            '<div class="pp-authors">%s</div>'
            '<div class="pp-affil">%s</div>'
            '<div class="cta-buttons">%s</div>'
            '<p class="pp-supp-line"><a href="%s">Many more generations on the full '
            'supplementary page &rarr;</a></p>'
            '</div>'
            % (title_html, authors_html, affil_html, buttons_html,
               html.escape(supp, quote=True)))


def _render_header_light(links, authors, affil, supp, title_html):
    """PixelDiT-style light hero: publication-title-fancy + authors-fancy + affiliations-fancy +
    btn-fancy buttons (icons in a <span class="icon">, incl. the inlined HF SVG; muted 'soon'
    buttons as a grey btn-fancy variant)."""
    n = len(authors)
    au = []
    for i, a in enumerate(authors):
        sup = ",".join(html.escape(str(x)) for x in (a.get("aff") or []))
        suph = '<sup class="author-sup">%s</sup>' % sup if sup else ''
        comma = "," if i < n - 1 else ""
        au.append('<span class="author-block">%s%s%s</span>'
                  % (html.escape(a["name"]), suph, comma))
    authors_html = "".join(au)

    aff = []
    for k in sorted(affil, key=lambda x: int(x) if str(x).isdigit() else x):
        aff.append('<span class="aff-item"><span class="aff-marker">%s</span>%s</span>'
                   % (html.escape(str(k)), html.escape(affil[k])))
    affil_html = "".join(aff)

    btns = []
    for key, label, icon, lk in BUTTONS:
        ic = HF_SVG if icon == "hf" else '<i class="%s" aria-hidden="true"></i>' % icon
        icon_span = '<span class="icon">%s</span>' % ic
        if key == "bibtex":
            btns.append('<a class="btn-fancy" href="#bibtex">%s<span>%s</span></a>'
                        % (icon_span, html.escape(label)))
            continue
        url = links.get(lk, "") or ""
        title = "Hugging Face" if key == "model" else label
        if url:
            btns.append('<a class="btn-fancy" href="%s" title="%s">%s<span>%s</span></a>'
                        % (html.escape(url, quote=True), html.escape(title, quote=True),
                           icon_span, html.escape(label)))
        else:
            btns.append('<span class="btn-fancy btn-fancy-disabled" title="coming soon">'
                        '%s<span>%s</span><span class="pp-soon">soon</span></span>'
                        % (icon_span, html.escape(label)))
    buttons_html = "".join(btns)

    return ('<section class="hero hero-fancy"><div class="hero-body">'
            '<div class="container is-max-desktop"><div class="columns is-centered">'
            '<div class="column has-text-centered">'
            '<h1 class="title publication-title-fancy">%s</h1>'
            '<div class="authors-fancy">%s</div>'
            '<div class="affiliations-fancy">%s</div>'
            '<div class="btn-fancy-container">%s</div>'
            '<p class="pp-supp-line"><a href="%s">Many more generations on the full '
            'supplementary page &rarr;</a></p>'
            '</div></div></div></div></section>'
            % (title_html, authors_html, affil_html, buttons_html,
               html.escape(supp, quote=True)))


# --------------------------------------------------------------------------- render: sections
def render_hero(picks, cblocks, dblocks, plan, warn):
    rows = []
    for p in picks.get("hero", []):
        sec, bid, cid = p.get("section"), p.get("block"), p.get("cell")
        idx = cblocks if sec == "C" else dblocks
        if bid not in idx:
            warn.append("hero: block %r not in section %s" % (bid, sec)); continue
        blk = idx[bid]
        cmap = {c["item_id"]: c for c in _block_cells(idx, plan, sec, bid)}
        if cid not in cmap:
            warn.append("hero: cell %r not in block %s" % (cid, bid)); continue
        if _block_hidden(plan, sec, bid):
            warn.append("hero: block %s is HIDDEN in site_plan (rendering anyway)" % bid)
        if cid in _hidden_cells(plan, sec, bid):
            warn.append("hero: cell %s is HIDDEN in block %s (rendering anyway)" % (cid, bid))
        rows.append(bs._block_html(blk, sec, False, set(), "", [cmap[cid]]))
    if not rows:
        return ""
    return '<div class="seg-section pp-hero">%s</div>' % "".join(rows)



_GEN_UNIT = '<div class="seg-unit seg-gen">'
_ARROW = '<div class="seg-arrow" aria-hidden="true">&#8594;</div>'

def _mark_ours(frag):
    """Add class seg-ours to every generation unit whose label starts with SEGUE (ours), so the
    accent frame marks our output and prior-work outputs keep the neutral output frame."""
    def repl(m):
        unit = m.group(0)
        lab = re.search(r'<div class="seg-label">([^<]*)</div>', unit)
        if lab and lab.group(1).strip().startswith("SEGUE"):
            return unit.replace(_GEN_UNIT, '<div class="seg-unit seg-gen seg-ours">', 1)
        return unit
    return re.sub(r'<div class="seg-unit seg-gen">.*?<div class="seg-label">[^<]*</div></div>', repl, frag, flags=re.S)

def decorate(frag):
    """Owner 2026-10-07 (input/output legibility): one arrow between the given clips and the first
    generated clip of every cell and strip, SEGUE outputs tagged seg-ours. Pure markup; the templates
    carry the CSS."""
    frag = _mark_ours(frag)
    def first_arrow(chunk):
        return chunk.replace('<div class="seg-unit seg-gen', _ARROW + '<div class="seg-unit seg-gen', 1)
    for opener in ('<div class="seg-cell">', '<div class="seg-strip'):
        parts = frag.split(opener)
        frag = parts[0] + "".join(opener + first_arrow(c) for c in parts[1:])
    return frag

def render_gallery(key, sec, blocks_idx, picks, plan, names, warn):
    out = []
    for p in picks.get(key, []):
        bid = p.get("block")
        if bid not in blocks_idx:
            warn.append("%s: block %r not found" % (key, bid)); continue
        blk = blocks_idx[bid]
        if _block_hidden(plan, sec, bid):
            warn.append("%s: block %s is HIDDEN in site_plan (rendering anyway)" % (key, bid))
        if p.get("cells") is None:
            seq = _visible_cells(blocks_idx, plan, sec, bid)
        else:
            cmap = {c["item_id"]: c for c in _block_cells(blocks_idx, plan, sec, bid)}
            hc = _hidden_cells(plan, sec, bid)
            seq = []
            for cid in p["cells"]:
                if cid not in cmap:
                    warn.append("%s: cell %r not in block %s" % (key, cid, bid)); continue
                if cid in hc:
                    warn.append("%s: cell %s is HIDDEN in block %s (rendering anyway)"
                                % (key, cid, bid))
                seq.append(cmap[cid])
        if not seq:
            warn.append("%s: block %s has no cells to show" % (key, bid)); continue
        head = '<div class="pp-blockhead">%s</div>' % html.escape(bs.human_name(blk["ref_class"], names))
        out.append(head + bs._block_html(blk, sec, False, set(), "", seq))
    return '<div class="seg-section">%s</div>' % "".join(out)


def render_cmp(key, sec, rows_idx, picks, plan, warn):
    out = []
    for iid in picks.get(key, []):
        if iid not in rows_idx:
            warn.append("%s: row %r not found" % (key, iid)); continue
        if _row_hidden(plan, sec, iid):
            warn.append("%s: row %s is HIDDEN in site_plan (rendering anyway)" % (key, iid))
        r = dict(rows_idx[iid])
        r["judge"] = None        # drop the blind-study verdict so no data-judge attribute leaks
        out.append(bs._cmp_row_html(r, sec, False, ""))
    return out


def render_sections(cfg, picks, man, plan, names, warn):
    S = man["sections"]
    cblocks = {b["block_id"]: b for b in S["C"]["blocks"]}
    dblocks = {b["block_id"]: b for b in S["D"]["blocks"]}
    arows = {r["item_id"]: r for r in S["A"]["rows"]}
    brows = {r["item_id"]: r for r in S["B"]["rows"]}
    supp = (cfg.get("links") or {}).get("supplementary", "") or "#"

    parts = []

    # B. Hero rows removed (owner 2026-10-07: start directly with the galleries); picks["hero"] is ignored.

    # C. TL;DR
    parts.append('<div class="section-title">TL;DR</div>')
    parts.append('<p class="pp-lead">%s</p>' % html.escape(TLDR))

    # D. Transition effect generation gallery
    parts.append('<div class="section-title" id="teg">Transition effect generation</div>')
    parts.append('<p class="pp-cap">Each reference is applied to two different endpoint pairs. Grey frames are the given clips; the coloured frame is generated by SEGUE.</p>')
    parts.append(render_gallery("teg", "C", cblocks, picks, plan, names, warn))

    # E. Visual effect transfer gallery
    parts.append('<div class="section-title" id="vet">Visual effect transfer</div>')
    parts.append('<p class="pp-cap">Only the start clip is given; the effect is transferred onto it. Grey frames are the given clips; the coloured frame is generated by SEGUE.</p>')
    parts.append(render_gallery("vet", "D", dblocks, picks, plan, names, warn))

    # F. Comparison with previous work (TEG rows then VFX-transfer rows)
    parts.append('<div class="section-title" id="compare">Comparison with previous work</div>')
    teg_rows = render_cmp("teg_cmp", "A", arows, picks, plan, warn)
    vet_rows = render_cmp("vet_cmp", "B", brows, picks, plan, warn)
    body = ""
    if teg_rows:
        body += '<div class="pp-blockhead">Transition effect generation</div>' + "".join(teg_rows)
    if vet_rows:
        body += '<div class="pp-blockhead">Visual effect transfer</div>' + "".join(vet_rows)
    parts.append('<div class="seg-section">%s</div>' % body)
    parts.append('<p class="pp-note">Prior methods receive a text description of the effect in '
                 'the prompt; SEGUE does not. SEGUE is the coloured frame.</p>')

    # G. How SEGUE works
    parts.append('<div class="section-title" id="method">How SEGUE works</div>')
    parts.append('<div class="figure pp-method"><img src="img/method.png" '
                 'alt="SEGUE method overview"/></div>')
    parts.append('<p><span class="emph">Visual reference isolation.</span> %s</p>' % html.escape(METHOD_P1))
    parts.append('<p><span class="emph">Null-reference guidance.</span> %s</p>' % html.escape(METHOD_P2))
    nrg_id = picks.get("nrg")
    grow = next((g for g in man.get("guidance", []) if g["id"] == nrg_id), None)
    if grow is None:
        warn.append("nrg: guidance row %r not found in manifest" % nrg_id)
    else:
        parts.append('<div class="seg-section">%s</div>' % bs._guidance_row_html(grow))
        parts.append('<p class="pp-note">Same reference and start clip; the guidance weight '
                     'increases left to right and the effect intensifies.</p>')

    # H. Footer: supplementary box + BibTeX
    parts.append('<div class="pp-footer-supp"><a href="%s">Full results: many more generations, '
                 'all comparison rows, guidance and limitations &rarr;</a></div>'
                 % html.escape(supp, quote=True))
    parts.append('<div class="bibtex-block" id="bibtex">'
                 '<button class="copy-btn" id="copyBib">Copy</button>'
                 '<pre>%s</pre></div>' % html.escape(cfg.get("bibtex", "")))

    return "\n".join(parts)


def render_sections_light(cfg, picks, man, plan, names, warn):
    """Light variant of render_sections: the SAME content (galleries, comparison strips, method,
    guidance, footer, BibTeX — identical text) wrapped in PixelDiT-style Bulma `section`/`container`
    blocks with `title is-3 has-text-centered` headings. The .seg-* markup is reused verbatim and
    recoloured by template_light.html's CSS."""
    S = man["sections"]
    cblocks = {b["block_id"]: b for b in S["C"]["blocks"]}
    dblocks = {b["block_id"]: b for b in S["D"]["blocks"]}
    arows = {r["item_id"]: r for r in S["A"]["rows"]}
    brows = {r["item_id"]: r for r in S["B"]["rows"]}
    supp = (cfg.get("links") or {}).get("supplementary", "") or "#"

    def section(body, title=None, anchor=None, container="is-max-widescreen"):
        aid = ' id="%s"' % anchor if anchor else ''
        h = ('<h2 class="title is-3 has-text-centered"%s>%s</h2>' % (aid, html.escape(title))
             if title else '')
        return ('<section class="section"><div class="container %s">%s%s</div></section>'
                % (container, h, body))

    out = []

    # TL;DR
    out.append(section('<p class="pp-lead">%s</p>' % html.escape(TLDR),
                       title="TL;DR", container="is-max-desktop"))

    # Transition effect generation gallery
    teg_body = ('<p class="pp-cap">Each reference is applied to two different endpoint pairs. Grey frames are the given clips; the coloured frame is generated by SEGUE.</p>'
                + render_gallery("teg", "C", cblocks, picks, plan, names, warn))
    out.append(section(teg_body, title="Transition effect generation", anchor="teg"))

    # Visual effect transfer gallery
    vet_body = ('<p class="pp-cap">Only the start clip is given; the effect is transferred onto it. Grey frames are the given clips; the coloured frame is generated by SEGUE.</p>'
                + render_gallery("vet", "D", dblocks, picks, plan, names, warn))
    out.append(section(vet_body, title="Visual effect transfer", anchor="vet"))

    # Comparison with previous work (TEG rows then VFX-transfer rows)
    teg_rows = render_cmp("teg_cmp", "A", arows, picks, plan, warn)
    vet_rows = render_cmp("vet_cmp", "B", brows, picks, plan, warn)
    cbody = ""
    if teg_rows:
        cbody += '<div class="pp-blockhead">Transition effect generation</div>' + "".join(teg_rows)
    if vet_rows:
        cbody += '<div class="pp-blockhead">Visual effect transfer</div>' + "".join(vet_rows)
    body = ('<div class="seg-section">%s</div>' % cbody
            + '<p class="pp-note">Prior methods receive a text description of the effect in '
              'the prompt; SEGUE does not. SEGUE is the coloured frame.</p>')
    out.append(section(body, title="Comparison with previous work", anchor="compare"))

    # How SEGUE works
    mbody = ('<div class="pp-method"><img src="img/method.png" alt="SEGUE method overview"/></div>'
             '<p class="pp-method-p"><span class="emph">Visual reference isolation.</span> %s</p>'
             '<p class="pp-method-p"><span class="emph">Null-reference guidance.</span> %s</p>'
             % (html.escape(METHOD_P1), html.escape(METHOD_P2)))
    nrg_id = picks.get("nrg")
    grow = next((g for g in man.get("guidance", []) if g["id"] == nrg_id), None)
    if grow is None:
        warn.append("nrg: guidance row %r not found in manifest" % nrg_id)
    else:
        mbody += ('<div class="seg-section">%s</div>' % bs._guidance_row_html(grow)
                  + '<p class="pp-note">Same reference and start clip; the guidance weight '
                    'increases left to right and the effect intensifies.</p>')
    out.append(section(mbody, title="How SEGUE works", anchor="method"))

    # Footer: supplementary box (no heading) + BibTeX
    out.append(section('<div class="pp-footer-supp"><a href="%s">Full results: many more '
                       'generations, all comparison rows, guidance and limitations &rarr;</a></div>'
                       % html.escape(supp, quote=True)))
    out.append(section('<div class="bibtex-block" id="bibtex">'
                       '<button class="copy-btn" id="copyBib">Copy</button>'
                       '<pre>%s</pre></div>' % html.escape(cfg.get("bibtex", "")),
                       title="BibTeX", container="is-max-desktop"))

    return "\n".join(out)


# --------------------------------------------------------------------------- html
def do_html(theme="dark"):
    cfg = _load(CONFIG)
    picks = _load(PICKS)
    man = _load(MANIFEST)
    plan = _load(SITE_PLAN)

    all_classes = set()
    for r in man["sections"]["A"]["rows"] + man["sections"]["B"]["rows"]:
        all_classes.add(r["ref_class"])
    for b in man["sections"]["C"]["blocks"] + man["sections"]["D"]["blocks"]:
        all_classes.add(b["ref_class"])
    names = bs.load_class_names(all_classes)

    warn = []
    header = render_header(cfg, theme)
    if theme == "light":
        sections = render_sections_light(cfg, picks, man, plan, names, warn)
        tpl_path, out_path = TEMPLATE_LIGHT, LIGHT_INDEX
    else:
        sections = render_sections(cfg, picks, man, plan, names, warn)
        tpl_path, out_path = TEMPLATE, INDEX

    tpl = open(tpl_path).read()
    assert tpl.count("{{HEADER}}") == 1 and tpl.count("{{SECTIONS}}") == 1, \
        "template must have exactly one {{HEADER}} and one {{SECTIONS}}"
    sections = decorate(sections)
    out = tpl.replace("{{HEADER}}", header).replace("{{SECTIONS}}", sections)
    if theme == "light":
        accent = cfg.get("accent") or DEFAULT_ACCENT
        accent_dark = cfg.get("accent_dark") or DEFAULT_ACCENT_DARK
        out = out.replace("__ACCENT_DARK__", accent_dark).replace("__ACCENT__", accent)

    # the public page must be clean of any curation/overlay leftovers
    for bad in ('data-hidden', 'seg-ov-', 'applyPlan', 'anonymous'):
        assert bad not in out, "project page not clean: found %r" % bad
    assert 'owner' not in out.lower(), "project page not clean: found 'owner'"

    os.makedirs(SITE, exist_ok=True)
    open(out_path, "w").write(out)
    print("[html] wrote %s (%.0f KB) [%s]"
          % (os.path.relpath(out_path, REPO), len(out) / 1024, theme))
    for w in warn:
        print("[html] WARNING:", w)
    if not warn:
        print("[html] no warnings")
    return warn


# --------------------------------------------------------------------------- check
def do_check(theme="dark"):
    page = INDEX if theme == "dark" else LIGHT_INDEX
    if not os.path.exists(page):
        sys.exit("[check] %s does not exist — run --html first" % os.path.relpath(page, REPO))
    raw = open(page).read()
    text = re.sub(r"<script\b.*?</script>", "", raw, flags=re.DOTALL)
    srcs = re.findall(r'<source[^>]+(?:data-src|src)="([^"]+)"', text)
    imgs = re.findall(r'<img[^>]+src="([^"]+)"', text)
    hrefs = re.findall(r'href="([^"]+)"', text)

    bad_abs, missing = [], []
    for s in srcs + imgs:
        if s.startswith("/") or s.startswith("http://") or s.startswith("https://"):
            bad_abs.append(s); continue
        if not os.path.exists(os.path.join(SITE, s)):
            missing.append(s)

    allowed_ext = {"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
                   "https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css",
                   # Google Fonts for the light variant (Google Sans / Noto Sans / Castoro)
                   "https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"}
    ext = [h for h in hrefs if h.startswith("http://") or h.startswith("https://")]
    stray_ext = [h for h in ext if h not in allowed_ext]

    # total MB of referenced media (unique files; getsize follows the symlinks)
    uniq = sorted(set(s for s in srcs + imgs if s not in bad_abs and s not in missing))
    total = sum(os.path.getsize(os.path.join(SITE, s)) for s in uniq)
    clips = sorted(set(s for s in srcs if s not in bad_abs and s not in missing))
    stills = sorted(set(s for s in imgs if s not in bad_abs and s not in missing
                        and s.lower().endswith((".jpg", ".jpeg", ".png")) and "/videos/" not in s))

    print("[check] page: %s [%s]" % (os.path.relpath(page, REPO), theme))
    print("[check] <source> refs=%d  <img> refs=%d  href=%d" % (len(srcs), len(imgs), len(hrefs)))
    print("[check] external stylesheet(s): %s" % (ext or "none"))
    print("[check] unique media files=%d (clips=%d, stills=%d)  total=%.1f MB"
          % (len(uniq), len(clips), len(stills), total / 1e6))
    print("[check] absolute media paths=%d  missing media=%d  stray external=%d"
          % (len(bad_abs), len(missing), len(stray_ext)))

    ok = True
    for s in bad_abs:
        print("[check]   ABSOLUTE", s); ok = False
    for s in missing:
        print("[check]   MISSING", s); ok = False
    for s in stray_ext:
        print("[check]   STRAY-EXTERNAL", s); ok = False

    # cleanliness tokens (public page must carry none of these)
    for bad in ('data-hidden', 'seg-ov-', 'applyPlan', 'anonymous'):
        if bad in raw:
            print("[check]   NOT-CLEAN token present:", bad); ok = False
    if 'owner' in raw.lower():
        print("[check]   NOT-CLEAN token present: owner"); ok = False

    if ok:
        print("[check] OK: every media ref resolves inside site/, page is clean, all external "
              "stylesheets are allow-listed")
    else:
        sys.exit("[check] FAILED")


# --------------------------------------------------------------------------- candidates
def _arm_label(item):
    pk = item.get("pick") or {}
    return pk.get("arm_label") or bs.SYS_LABEL.get("segue", "SEGUE")


def _ordered_gallery(sec, blocks_idx, plan, names):
    ps = _plan_sec(plan, sec)
    mi = ps.get("merge_into") or {}
    hb = set(ps.get("hidden_blocks") or [])
    merged_away = set(s for s, d in mi.items() if s in blocks_idx and d in blocks_idx)
    blocks = [b for b in blocks_idx.values() if b["block_id"] not in merged_away]
    auto_co, auto_bo = bs._auto_gallery(blocks, names)
    class_order = bs._apply_order(auto_co, ps.get("class_order"))
    by_class = {}
    for b in blocks:
        by_class.setdefault(b["ref_class"], []).append(b)
    out = []
    for cls in class_order:
        if cls not in by_class:
            continue
        cmap = {b["block_id"]: b for b in by_class[cls]}
        bo = bs._apply_order(auto_bo[cls], (ps.get("block_order") or {}).get(cls))
        vis = [cmap[bid] for bid in bo if bid not in hb]
        if vis:
            out.append((cls, vis))
    return out


def _ordered_comparison(sec, rows_idx, plan, names):
    ps = _plan_sec(plan, sec)
    hr = set(ps.get("hidden_rows") or [])
    rows = list(rows_idx.values())
    auto_co, auto_ro = bs._auto_comparison(rows, names)
    class_order = bs._apply_order(auto_co, ps.get("class_order"))
    by_class = {}
    for r in rows:
        by_class.setdefault(r["ref_class"], []).append(r)
    out = []
    for cls in class_order:
        if cls not in by_class:
            continue
        rmap = {r["item_id"]: r for r in by_class[cls]}
        ro = bs._apply_order(auto_ro[cls], (ps.get("row_order") or {}).get(cls))
        vis = [rmap[iid] for iid in ro if iid not in hr]
        if vis:
            out.append((cls, vis))
    return out


def do_candidates():
    man = _load(MANIFEST)
    plan = _load(SITE_PLAN)
    S = man["sections"]
    cblocks = {b["block_id"]: b for b in S["C"]["blocks"]}
    dblocks = {b["block_id"]: b for b in S["D"]["blocks"]}
    arows = {r["item_id"]: r for r in S["A"]["rows"]}
    brows = {r["item_id"]: r for r in S["B"]["rows"]}

    all_classes = set()
    for r in S["A"]["rows"] + S["B"]["rows"]:
        all_classes.add(r["ref_class"])
    for b in S["C"]["blocks"] + S["D"]["blocks"]:
        all_classes.add(b["ref_class"])
    names = bs.load_class_names(all_classes)

    L = []
    L.append("# SEGUE project page — candidates")
    L.append("")
    L.append("Every VISIBLE block / cell / row from the supplementary site plan, in plan order. "
             "This is the menu for `picks.json` (edit that file, then rebuild). `item_id`s are the "
             "ids you paste into `picks.json`. Hidden-in-plan items are intentionally omitted; a "
             "pick that names a hidden item still renders but prints a WARNING at build.")
    L.append("")
    L.append("_Generated from `supplementary/site_manifest.json` + "
             "`eval_ladder/viewer/collections/site_plan.json` (read-only)._")
    L.append("")

    def gallery_md(title, sec, blocks_idx, blockpfx):
        L.append("## %s gallery  (picks: `%s`, block id `%s<reference>`)"
                 % (title, "teg" if sec == "C" else "vet", blockpfx))
        L.append("")
        for cls, blocks in _ordered_gallery(sec, blocks_idx, plan, names):
            L.append("### %s" % bs.human_name(cls, names))
            for b in blocks:
                cells = _visible_cells(blocks_idx, plan, sec, b["block_id"])
                L.append("")
                L.append("- **block** `%s` — reference `%s` — %d visible cell(s)"
                         % (b["block_id"], b["reference"], len(cells)))
                for c in cells:
                    L.append("    - cell `%s` | endpoint `%s` | %s | %s"
                             % (c["item_id"], c["endpoint"], c["tier_words"], _arm_label(c)))
            L.append("")

    def comparison_md(title, sec, rows_idx, pickkey):
        L.append("## %s — comparison with previous work  (picks: `%s`, list of item_id)"
                 % (title, pickkey))
        L.append("")
        for cls, rows in _ordered_comparison(sec, rows_idx, plan, names):
            L.append("### %s" % bs.human_name(cls, names))
            for r in rows:
                L.append("- row `%s` | reference `%s` | endpoint `%s` | %s | %s"
                         % (r["item_id"], r["reference"], r["endpoint"], r["tier_words"], _arm_label(r)))
            L.append("")

    gallery_md("Transition effect generation", "C", cblocks, "teg_ref__")
    comparison_md("Transition effect generation", "A", arows, "teg_cmp")
    gallery_md("Visual effect transfer", "D", dblocks, "vfx_ref__")
    comparison_md("Visual effect transfer", "B", brows, "vet_cmp")

    open(CANDIDATES, "w").write("\n".join(L) + "\n")
    nb_c = sum(len(v) for _, v in _ordered_gallery("C", cblocks, plan, names))
    nb_d = sum(len(v) for _, v in _ordered_gallery("D", dblocks, plan, names))
    nr_a = sum(len(v) for _, v in _ordered_comparison("A", arows, plan, names))
    nr_b = sum(len(v) for _, v in _ordered_comparison("B", brows, plan, names))
    print("[candidates] wrote %s  (visible: TEG %d blocks / VET %d blocks / TEG-cmp %d rows / "
          "VET-cmp %d rows)" % (os.path.relpath(CANDIDATES, REPO), nb_c, nb_d, nr_a, nr_b))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--theme", choices=["dark", "light", "both"], default="both",
                    help="which variant(s) to build/check: dark -> site/index.html, "
                         "light -> site/light.html (default: both)")
    a = ap.parse_args()
    if not (a.html or a.check or a.candidates):   # default = html + check
        a.html = a.check = True
    themes = ["dark", "light"] if a.theme == "both" else [a.theme]
    if a.candidates:
        do_candidates()
    if a.html:
        for t in themes:
            do_html(t)
    if a.check:
        for t in themes:
            do_check(t)


if __name__ == "__main__":
    main()
