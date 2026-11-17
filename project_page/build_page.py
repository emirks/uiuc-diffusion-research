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
    --theme       dark|light|both (default both). BOTH themes render from the SAME template
                  (template_light.html) with the SAME light-style header/section renderers; the
                  two pages differ ONLY in the <html> theme class (theme-dark / theme-light) and
                  the injected accent values. dark -> site/index.html; light -> site/light.html
                  (same dir, identical media). template_dark_v1.html is the retired original dark
                  shell, kept as a fallback and no longer built.
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
# ONE template drives both themes; template_dark_v1.html is the retired dark shell (not built).
TEMPLATE_LIGHT = os.path.join(HERE, "template_light.html")
INDEX = os.path.join(SITE, "index.html")                  # dark -> site/index.html
LIGHT_INDEX = os.path.join(SITE, "light.html")            # light -> site/light.html (same dir)
# --supp -> site/supp/index.html: the FULL results gallery, rendered through the SAME template in
# the dark theme (so it matches the project page's dark showcase band exactly). It carries ALL of
# the supplementary's content (not the picks subset); media resolve through symlinks under supp/.
SUPP_SITE = os.path.join(SITE, "supp")
SUPP_INDEX = os.path.join(SUPP_SITE, "index.html")
# light-theme accent (config "accent"/"accent_dark"); dark-theme accent ("accent_dark_theme"/...)
DEFAULT_ACCENT = "#0F766E"                   # light accent
DEFAULT_ACCENT_DARK = "#115E59"              # light accent (hover/darker)
DEFAULT_ACCENT_DARK_THEME = "#2DD4BF"        # dark-theme accent
DEFAULT_ACCENT_DARK_THEME_HOVER = "#14B8A6"  # dark-theme accent (hover/darker)
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

ABSTRACT_HTML = 'One-Shot Transition Effect Generation (TEG) aims to bridge two video clips using a transition effect provided in a reference clip. However, existing methods learn a text shortcut: they rely heavily on prompts with detailed effect descriptions, and the visual transition signal loses its weight. We present SEGUE, a framework designed to read the transition strictly from the visual demonstration via two key mechanisms: i) <b>Visual reference isolation</b> during training, pairing inputs with effect-free neutral prompts and applying a one-way attention mask that keeps the reference independent of the target; and ii) <b>Null-reference guidance</b> at inference, which defines the unconditional path as a dissolve of the reference endpoints—adding no motion or material of its own—to isolate and amplify the visual transition effect. SEGUE flexibly supports endpoints that are video clips, static images, or absent, unifying two-sided TEG and single-sided visual effect transfer within a single model. We also introduce the SEGUE-Benchmark, a cross-paired dataset of 1,692 transition classes and 56,368 training pairs spanning creative presets, parametric procedural transitions, and visual effects. On held-out effects, SEGUE outperforms baselines adapted to TEG and previous visual effect transfer methods even without a detailed effect description in the prompt, serving as the first open-source framework for one-shot TEG.'
TLDR_HTML = 'Show SEGUE a transition effect in one reference clip, and it generates that effect between any two video clips. It reads the effect strictly from the visual reference, with no effect description in the prompt, and one model covers both two-sided transitions and single-sided visual effect transfer.'
METHOD_INTRO_HTML = 'A transition effect brings in motion and material that belong to neither clip: smoke, a portal, a wipe. Prior reference-conditioned methods pair the reference clip with a prompt that describes the effect in words. Text is the easier signal, so the model learns to read the effect from the prompt and under-uses the video. Remove the description and the effect vanishes; and text has no words for the exact motion and timing of a new effect anyway. SEGUE is built so that the only place the effect can come from is the reference video.'
TLDR = ("One-shot transition effect generation: given a reference clip that shows a transition "
        "effect and two endpoint clips, generate the transition that applies that effect between "
        "them. SEGUE reads the effect strictly from the visual reference: visual reference "
        "isolation in training and null-reference guidance at inference, with no effect "
        "description in the prompt. One model covers two-sided transitions and single-sided "
        "visual effect transfer, where only the start clip is given.")

METHOD_P1_HTML = 'Prior methods pair the reference clip with a prompt that describes the effect in words. But text cannot capture the dynamics of an effect, its exact motion and timing, and once the model is given the description it leans on it and the visual reference loses its weight: remove the description and the effect vanishes. SEGUE therefore isolates the visual reference, so that the effect is read from the reference video and from nothing else, with two rules. <b>Attention is one-way:</b> the target reads the reference, but the reference never reads the target. The representation of the reference therefore depends on the reference video alone, in every layer, and the target cannot change it. <b>The prompt is neutral:</b> it captions the two scenes and says nothing about the effect, so there is no shortcut to take.'
METHOD_P2_HTML = 'Guidance should steer away from a reference that keeps the scenes of the reference and removes only its transition effect, so that the contrast amplifies the effect and nothing else. A dissolve is exactly that: the <b>null reference</b> is the reference clip with its transition replaced by a dissolve between its own endpoints. At each step the model runs twice, once with the reference and once with the null reference, and the sample is pushed away from the dissolve toward the effect. The weight <i>w</i> sets how strongly; <i>w</i> = 1 is no guidance.'

LEGEND = ('<div class="seg-legend">'
          '<span><span class="sw sw-in"></span>given inputs (reference &amp; endpoints)</span>'
          '<span><span class="sw sw-gen"></span>generated clips (SEGUE &amp; prior work)</span>'
          '</div>')

# header buttons: (key, label, font-awesome icon, link-key); BibTeX scrolls to the bibtex block
BUTTONS = [
    ("paper_pdf", "Paper", "fa-solid fa-file-pdf", "paper_pdf"),   # the current PDF build, hosted on the site
    ("paper", "arXiv", "ai ai-arxiv", "paper"),
    ("code", "Code", "fa-brands fa-github", "code"),
    ("model", "Model", "hf", "model"),
    ("dataset", "Dataset", "hf", "dataset"),
    ("supplementary", "Results gallery", "fa-solid fa-film", "supplementary"),
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
    """Title with a leading 'SEGUE' token wrapped in <span class="pp-accent">; the template
    colours that span with the accent (gradient) in both themes."""
    esc = html.escape(cfg["title"])
    lead = "SEGUE"
    if esc.startswith(lead):
        return '<span class="pp-accent">%s</span>%s' % (lead, esc[len(lead):])
    return esc


def render_header(cfg):
    """Build the single (light-style) hero header used by both themes."""
    links = cfg.get("links") or {}
    authors = cfg.get("authors") or []
    affil = cfg.get("affiliations") or {}
    supp = links.get("supplementary", "") or "#"
    title_html = _title_html(cfg)
    return _render_header_light(links, authors, affil, supp, title_html, cfg.get("author_note", ""))


def _render_header_light(links, authors, affil, supp, title_html, cfg_author_note=""):
    """Light-style hero: publication-title-fancy + authors-fancy + affiliations-fancy +
    btn-fancy buttons (icons in a <span class="icon">, incl. the inlined HF SVG; muted 'soon'
    buttons as a grey btn-fancy variant)."""
    n = len(authors)
    au = []
    for i, a in enumerate(authors):
        sup = ",".join(html.escape(str(x)) for x in (a.get("aff") or []))
        sup += html.escape(a.get("mark", ""))          # e.g. the dagger of the joint last authors
        suph = '<sup class="author-sup">%s</sup>' % sup if sup else ''
        comma = "," if i < n - 1 else ""
        if a.get("break"):
            au.append('<br class="author-break"/>')     # owner's line break (joint last authors on their own line)
        name = html.escape(a["name"])
        if a.get("url"):
            name = '<a class="author-link" href="%s" target="_blank" rel="noopener">%s</a>' % (html.escape(a["url"], quote=True), name)
        au.append('<span class="author-block">%s%s%s</span>' % (name, suph, comma))
    authors_html = "".join(au)

    aff = []
    for k in sorted(affil, key=lambda x: int(x) if str(x).isdigit() else x):
        aff.append('<span class="aff-item"><span class="aff-marker">%s</span>%s</span>'
                   % (html.escape(str(k)), html.escape(affil[k])))
    affil_html = "".join(aff)
    if cfg_author_note:
        affil_html += '<div class="aff-note">%s</div>' % html.escape(cfg_author_note)

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
            '<p class="pp-supp-line"><a href="%s">Many more generations in the full '
            'results gallery &rarr;</a></p>'
            '</div></div></div></div></section>'
            % (title_html, authors_html, affil_html, buttons_html,
               html.escape(supp, quote=True)))


# --------------------------------------------------------------------------- render: sections
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


def _one_sided_stack(still_dest):
    """Start frame stacked above the 'no end given' box: the same endpoint format the effect-transfer
    gallery uses, now also for its comparison rows and the guidance row (2026-10-07)."""
    return ('<div class="seg-stack">%s%s</div>'
            % (bs._still_div(still_dest, "seg-endpoint", badge="start"), bs._noend_div()))


def cmp_row_html(r, section):
    """bs._cmp_row_html with one change: one-sided rows (section B) show the start frame in the
    endpoint stack instead of a full-height still."""
    if section != "B":
        return bs._cmp_row_html(r, section, False, "")
    m = r["media"]
    cells = [
        bs._unit(bs._media_div(m["reference"]["dest"]), "reference", "seg-ref"),
        bs._unit(_one_sided_stack("stills/B/start/%s.jpg" % r["row_id"]), "endpoint", "seg-endstack"),
        bs._unit(bs._media_div(m["segue"]["dest"]), bs.SYS_LABEL["segue"], "seg-gen"),
        bs._unit(bs._media_div(m["vap"]["dest"]), bs.SYS_LABEL["vap"], "seg-gen"),
        bs._unit(bs._media_div(m["vfxmaster"]["dest"]), bs.SYS_LABEL["vfxmaster"], "seg-gen"),
        bs._unit(bs._media_div(m["refvfx"]["dest"]), bs.SYS_LABEL["refvfx"], "seg-gen"),
    ]
    return ('<div class="seg-strip" data-section="%s" data-class="%s" data-item="%s" data-ref="%s">'
            '<div class="seg-cellsrow">%s</div></div>'
            % (section, html.escape(r["ref_class"]), html.escape(r["item_id"]),
               html.escape(r["reference"]), "".join(cells)))


def guidance_row_html(r):
    """bs._guidance_row_html with the start clip's first frame in the endpoint stack (one-sided)."""
    m = r["media"]
    still = "img/nrg_start.jpg"   # first frame of the start clip (extracted once with ffmpeg)
    cells = [bs._unit(bs._media_div(m["reference"]["dest"]), "reference", "seg-ref"),
             bs._unit(bs._still_div(still, badge="start"), "start", "seg-startfull")]   # start frame only (no 'no end' box here)
    for g in r["gens"]:
        cells.append(bs._unit(bs._media_div(m[g["key"]]["dest"]), g["label"], "seg-gen"))
    return ('<div class="seg-strip seg-guidance" data-section="G" data-item="%s">'
            '<div class="seg-cellsrow">%s</div></div>' % (html.escape(r["id"]), "".join(cells)))

def _png_size(path):
    """(width, height) from a PNG's IHDR chunk (no image library needed)."""
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "%s is not a PNG" % path
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def ensure_webp(name, quality=90):
    """site/img/<name>.webp from site/img/<name>.png (2026-11-16: the method figures were 1.1 MB and
    0.6 MB PNGs; WebP q90 is ~0.3 MB / ~0.1 MB and looks identical). Re-encoded only when the PNG is
    newer. The PNG stays: it is the full-size link target and the og:image."""
    png = os.path.join(SITE, "img", name + ".png")
    webp = os.path.join(SITE, "img", name + ".webp")
    if os.path.exists(webp) and os.path.getmtime(webp) >= os.path.getmtime(png):
        return webp
    ff = shutil.which("ffmpeg") or os.path.join(os.path.dirname(REPO), "bin", "ffmpeg")
    import subprocess
    subprocess.run([ff, "-v", "error", "-y", "-i", png, "-c:v", "libwebp", "-quality", str(quality),
                    "-compression_level", "6", webp], check=True)
    print("[img] wrote %s (%.0f KB, from %.0f KB PNG)"
          % (os.path.relpath(webp, REPO), os.path.getsize(webp) / 1024, os.path.getsize(png) / 1024))
    return webp


def figure_html(name, alt):
    """A method figure: the light WebP, loaded lazily with its real size (no layout jump), wrapped in
    a link to the full-size PNG so it can be opened and zoomed on a phone."""
    ensure_webp(name)
    w, h = _png_size(os.path.join(SITE, "img", name + ".png"))
    return ('<a class="pp-fig-link" href="img/%s.png" target="_blank" rel="noopener">'
            '<img src="img/%s.webp" width="%d" height="%d" loading="lazy" decoding="async" alt="%s"/></a>'
            % (name, name, w, h, html.escape(alt, quote=True)))


FULL_ROWS = True    # default since 2026-10-07 ("full rows is ok"); --full-rows kept as a no-op switch

# Label under every reference clip on this site (2026-11-17: "reference effect"). The shared supplementary
# builder writes "reference"; the anonymous supplementary site keeps that, only this site is relabelled.
REF_LABEL = ('<div class="seg-label">reference</div>', '<div class="seg-label">reference effect</div>')

GALLERY_SLOTS = 2   # samples per reference row (deployed 2026-10-09 layout: 2); fewer -> empty slot(s)
PAGE_SIZE = 2       # gallery rows per page (2026-10-09: 2); comparison groups use CMP_PAGE_SIZE
CMP_PAGE_SIZE = 2   # comparison rows per page (2026-10-09: same as the galleries)

# Label shown in each pager's top bar (the per-group card heading moved here 2026-10-09).
PAGER_LABELS = {
    "teg": "Results gallery",
    "vet": "Results gallery",
    "teg_cmp": "Transition effect generation",
    "vet_cmp": "Visual effect transfer",
}


def paginate(rows, key, label, size=PAGE_SIZE):
    """Wrap a list of per-row HTML strings into ONE card (``.pp-pager``) with a label bar.

    ``rows`` is a list of HTML strings, one per visual row (a gallery row = blockhead + blockwrap;
    a comparison row = one seg-strip). Each is wrapped in a flat ``.pp-row`` so the card draws the
    chrome. The card is ALWAYS returned (so a single group of <=size rows is still one labelled
    card); the side arrows and the dots are added only when there is more than one page. The template's pager script toggles the pages."""
    n = len(rows)
    pages = (n + size - 1) // size
    wrapped = ['<div class="pp-row">%s</div>' % r for r in rows]
    page_divs = []
    for i in range(pages):
        chunk = "".join(wrapped[i * size:(i + 1) * size])
        if i == 0:
            page_divs.append('<div class="pp-page is-active" data-page="1">%s</div>' % chunk)
        else:
            page_divs.append('<div class="pp-page" data-page="%d" hidden>%s</div>' % (i + 1, chunk))
    lbl = '<span class="pp-pagerlabel">%s</span>' % html.escape(label)
    top = '<div class="pp-pagenav pp-pagenav-top"><div class="pp-pagerleft">%s</div></div>' % lbl
    if pages > 1:
        # 2026-10-09: slideshow look (round arrows, dots + "1 / n"); the label bar keeps only the label.
        # 2026-11-16: the arrows sit in the dots bar under the rows ("< . . o . 2 / 6 >"); on wide
        # screens the template floats them at the card's sides instead (CSS only, same markup).
        prev = '<button type="button" class="pp-ss-arrow pp-prev" aria-label="Previous rows">&#10094;</button>'
        nxt = '<button type="button" class="pp-ss-arrow pp-next" aria-label="Next rows">&#10095;</button>'
        dots = ('<div class="pp-ss-dots">%s%s<span class="pp-ss-count"><span class="pp-pagecur">1</span> / %d</span>%s</div>'
                % (prev,
                   "".join('<span class="pp-ss-dot%s" data-go="%d"></span>' % (" on" if i == 0 else "", i + 1)
                           for i in range(pages)), pages, nxt))
    else:
        dots = ""
    return ('<div class="pp-pager" id="pager-%s" data-size="%d" data-total="%d">'
            '%s<div class="pp-slideshow"><div class="pp-pages">%s</div></div>%s</div>'
            % (key, size, n, top, "".join(page_divs), dots))


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
        body = bs._block_html(blk, sec, False, set(), "", seq)
        if len(seq) < GALLERY_SLOTS:    # keep every row's clip size equal: empty slot(s) on the right
            tail = "</div></div></div>"   # closes seg-cells, seg-block, seg-blockwrap
            assert body.endswith(tail)
            body = (body[:-len(tail)]
                    + '<div class="seg-cellwrap pp-cell-empty" aria-hidden="true"></div>' * (GALLERY_SLOTS - len(seq))
                    + tail)
        out.append(head + body)
    return '<div class="seg-section">' + paginate(out, key, PAGER_LABELS[key]) + '</div>'


def render_cmp(key, sec, rows_idx, picks, plan, warn):
    out = []
    for iid in picks.get(key, []):
        if iid not in rows_idx:
            warn.append("%s: row %r not found" % (key, iid)); continue
        if _row_hidden(plan, sec, iid):
            warn.append("%s: row %s is HIDDEN in site_plan (rendering anyway)" % (key, iid))
        r = dict(rows_idx[iid])
        r["judge"] = None        # drop the blind-study verdict so no data-judge attribute leaks
        out.append(cmp_row_html(r, sec))
    return out


def render_sections_light(cfg, picks, man, plan, names, warn):
    """The page body used by BOTH themes: galleries, comparison strips, method, guidance, footer
    and BibTeX wrapped in Bulma `section`/`container` blocks with `title is-3 has-text-centered`
    headings. The .seg-* markup is reused verbatim; template_light.html's CSS recolours it per
    theme (light by default, dark under html.theme-dark)."""
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
    out.append(section(
        '<div class="pp-abstract-box">'
        '<h3 class="pp-subhead pp-subhead-top">TL;DR</h3>'
        '<p class="pp-tldr">%s</p>'
        '<details class="pp-abs-details"><summary><span class="pp-abs-toggle">Click for the full abstract</span></summary>'
        '<h3 class="pp-subhead">Abstract</h3>'
        '<p class="pp-abstract">%s</p></details>'
        '</div>' % (TLDR_HTML, ABSTRACT_HTML),
        container="is-max-desktop"))

    # Transition effect generation gallery
    teg_body = ('<p class="pp-cap">Each reference effect is transferred onto two different endpoint pairs. Grey boxes are the inputs; the blue box is generated by SEGUE.</p>'
                + render_gallery("teg", "C", cblocks, picks, plan, names, warn))
    out.append('<div class="pp-band">')   # dark showcase band: galleries + comparisons
    out.append(section(teg_body, title="Transition effect generation", anchor="teg"))

    # Visual effect transfer gallery
    vet_body = ('<p class="pp-cap">Only the start is given; the reference effect is transferred onto it. Grey boxes are the inputs; the blue box is generated by SEGUE.</p>'
                + render_gallery("vet", "D", dblocks, picks, plan, names, warn))
    out.append(section(vet_body, title="Visual effect transfer", anchor="vet"))

    # Comparison with previous work (TEG rows then VFX-transfer rows)
    teg_rows = render_cmp("teg_cmp", "A", arows, picks, plan, warn)
    vet_rows = render_cmp("vet_cmp", "B", brows, picks, plan, warn)
    cbody = ""
    if teg_rows:
        cbody += paginate(teg_rows, "teg_cmp", PAGER_LABELS["teg_cmp"], size=CMP_PAGE_SIZE)
    if vet_rows:
        cbody += paginate(vet_rows, "vet_cmp", PAGER_LABELS["vet_cmp"], size=CMP_PAGE_SIZE)
    body = ('<div class="seg-section%s">%s</div>' % (" pp-full-rows" if FULL_ROWS else "", cbody)
            + '<p class="pp-note">Prior methods receive a text description of the effect in '
              'the prompt; SEGUE does not. SEGUE is the blue box.</p>')
    out.append(section(body, title="Comparison with previous work", anchor="compare"))
    out.append('</div>')   # end of the showcase band

    # How SEGUE works: two parts (2026-10-07), each = its paper figure + one paragraph;
    # the null-reference part ends with the guidance-weight row, full width (no slider).
    mbody = ('<h3 class="pp-subhead" id="method-vri"><span class="pp-subhead-kicker">Training:</span> Visual Reference Isolation</h3>'
             '<div class="pp-method">%s</div>'
             '<p class="pp-method-p">%s</p>'
             '<h3 class="pp-subhead" id="method-nrg"><span class="pp-subhead-kicker">Inference:</span> Null-Reference Guidance</h3>'
             '<div class="pp-method pp-method-nrg">%s</div>'
             '<p class="pp-method-p">%s</p>'
             % (figure_html("method", "SEGUE training: in-context reference conditioning, one-way reference attention, neutral text"),
                METHOD_P1_HTML,
                figure_html("nrg", "Null-reference guidance: the null reference replaces the transition with a dissolve between the same endpoints; guidance steers away from it"),
                METHOD_P2_HTML))
    nrg_id = picks.get("nrg")
    grow = next((g for g in man.get("guidance", []) if g["id"] == nrg_id), None)
    if grow is None:
        warn.append("nrg: guidance row %r not found in manifest" % nrg_id)
    else:
        mbody += ('<div class="pp-rowhead">The same reference and start clip generated with increasing guidance strength, from no guidance to w = 6</div>'
                  + '<div class="seg-section pp-nrg-row pp-full-rows">%s</div>' % guidance_row_html(grow)
                  + '<p class="pp-note">Same reference and start clip; the guidance weight '
                    'increases left to right and the effect intensifies.</p>')
    out.append(section(mbody, title="How SEGUE works", anchor="method"))

    # Footer: supplementary box (no heading) + BibTeX
    out.append(section('<div class="pp-footer-supp"><a href="%s">Results gallery: many more '
                       'generations, all comparison rows, guidance and limitations &rarr;</a></div>'
                       % html.escape(supp, quote=True)))
    out.append(section('<div class="bibtex-block" id="bibtex">'
                       '<button class="copy-btn" id="copyBib">Copy</button>'
                       '<pre>%s</pre></div>' % html.escape(cfg.get("bibtex", "")),
                       title="BibTeX", container="is-max-desktop"))

    return "\n".join(out)


# --------------------------------------------------------------------------- supp (results gallery)
# Extra CSS for site/supp/index.html ONLY (prepended to its {{SECTIONS}}; never added to the shared
# template, so the two theme pages stay byte-identical). It restyles the supplementary's contents
# list (.seg-toc) and the gallery subtitle for the project page's dark palette (cards, square
# corners, accent links). Everything else is painted by the template's dark palette already.
SUPP_CSS = (
    '<style id="pp-supp">'
    '.pp-supp-subtitle{font-family:\'Google Sans\',sans-serif;color:#9ca3af;font-size:1.15rem;'
    'margin:-6px 0 2px;letter-spacing:-0.01em}'
    '.seg-toc{max-width:820px;margin:0 auto;background:#121212;border:1px solid #262626;'
    'border-radius:0;box-shadow:0 4px 12px rgba(0,0,0,.45);padding:18px 22px}'
    '.seg-toc-title{color:#9ca3af}'
    '.seg-toc a{color:var(--accent)}'
    '.seg-toc li ul li{color:#9ca3af}'
    '.pp-supp-sub{font-family:\'Google Sans\',sans-serif;font-size:1.15rem;font-weight:600;'
    'color:#f3f4f6;margin:28px 0 8px;text-align:center}'
    '.pp-supp-sub .pp-supp-subdesc{display:block;font-family:\'Noto Sans\',sans-serif;'
    'font-size:.85rem;font-weight:400;color:#9ca3af;margin-top:4px}'
    '</style>'
)


def render_supp_header(cfg):
    """Compact hero for the results gallery: the SEGUE title (accent span), a 'Results gallery'
    subtitle and one row of two btn-fancy buttons (Project page -> ../, Paper -> ../paper.pdf). No
    authors, affiliations, TL;DR or abstract."""
    title_html = _title_html(cfg)
    btns = [("../", "fa-solid fa-arrow-left", "Project page"),
            ("../paper.pdf", "fa-solid fa-file-pdf", "Paper")]
    bb = []
    for url, icon, label in btns:
        icon_span = '<span class="icon"><i class="%s" aria-hidden="true"></i></span>' % icon
        bb.append('<a class="btn-fancy" href="%s">%s<span>%s</span></a>'
                  % (html.escape(url, quote=True), icon_span, html.escape(label)))
    return ('<section class="hero hero-fancy"><div class="hero-body">'
            '<div class="container is-max-desktop"><div class="columns is-centered">'
            '<div class="column has-text-centered">'
            '<h1 class="title publication-title-fancy">%s</h1>'
            '<p class="pp-supp-subtitle">Results gallery</p>'
            '<div class="btn-fancy-container">%s</div>'
            '</div></div></div></div></section>'
            % (title_html, "".join(bb)))



SUPP_PICKED = {"blocks": set(), "rows": set()}   # set by do_supp from picks.json


def _to_end(seq, key, picked):
    """Stable partition: items not in `picked` first, then the picked ones, both in their original order."""
    return [x for x in seq if key(x) not in picked] + [x for x in seq if key(x) in picked]


def render_supp_gallery(section, blocks, plan, names):
    """bs._render_gallery's content (merge_into, hidden blocks/cells, cell_order, class/block order)
    as ONE flat sequence in the plan's order, with the blocks that are on the project page moved to
    the end of the section (2026-10-08). A class heading is written each time the class changes, so a
    class can appear twice (its unseen references first, its seen ones at the end)."""
    ps = bs._plan_sec(plan, section)
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
    auto_co, auto_bo = bs._auto_gallery(blocks2, names)
    class_order = bs._apply_order(auto_co, ps.get("class_order"))
    by_class = {}
    for b in blocks2:
        by_class.setdefault(b["ref_class"], []).append(b)
    flat = []   # (cls, block, refidx_label) in the plan's order, hidden blocks dropped
    for cls in class_order:
        cblocks = {b["block_id"]: b for b in by_class.get(cls, [])}
        bo = [bid for bid in bs._apply_order(auto_bo.get(cls, []), (ps.get("block_order") or {}).get(cls))
              if bid not in hidden_blocks]
        for i, bid in enumerate(bo, 1):
            flat.append((cls, cblocks[bid], bs._refidx_label(i, len(bo))))
    flat = _to_end(flat, lambda t: t[1]["block_id"], SUPP_PICKED["blocks"])
    out, group, cur = [], [], None
    def flush():
        if group:
            out.append('<div class="seg-classgroup" data-section="%s" data-class="%s">%s</div>'
                       % (section, html.escape(cur), "".join(group)))
    for cls, b, lab in flat:
        if cls != cur:
            flush(); group, cur = [bs._classhead(section, cls, names)], cls
        bid = b["block_id"]
        hc = set(hidden_cells.get(bid) or [])
        cmap = {c["item_id"]: c for c in b["cells"]}
        seq = [cmap[iid] for iid in bs._apply_order([c["item_id"] for c in b["cells"]], cell_order.get(bid))]
        group.append(bs._block_html(b, section, False, hc, lab, seq))
    flush()
    return "".join(out)

def render_supp_comparison(section, rows, plan, names):
    """bs._render_comparison's plan iteration (class grouping, class/row order, hidden rows) as ONE
    flat sequence, rows that are on the project page moved to the end (2026-10-08), and the project
    page's cmp_row_html for every row (A: two-endpoint stack; B: start frame + 'no end given')."""
    ps = bs._plan_sec(plan, section)
    hidden_rows = set(ps.get("hidden_rows") or [])
    auto_co, auto_ro = bs._auto_comparison(rows, names)
    class_order = bs._apply_order(auto_co, ps.get("class_order"))
    by_class = {}
    for r in rows:
        by_class.setdefault(r["ref_class"], {})[r["item_id"]] = r
    flat = []
    for cls in class_order:
        rmap = by_class.get(cls, {})
        for iid in bs._apply_order(auto_ro.get(cls, []), (ps.get("row_order") or {}).get(cls)):
            if iid in hidden_rows or iid not in rmap:
                continue
            r = dict(rmap[iid]); r["judge"] = None   # no blind-study verdict attribute on the public page
            flat.append((cls, r))
    flat = _to_end(flat, lambda t: t[1]["item_id"], SUPP_PICKED["rows"])
    out, group, cur = [], [], None
    def flush():
        if group:
            out.append('<div class="seg-classgroup" data-section="%s" data-class="%s">%s</div>'
                       % (section, html.escape(cur), "".join(group)))
    for cls, r in flat:
        if cls != cur:
            flush(); group, cur = [bs._classhead(section, cls, names)], cls
        group.append(cmp_row_html(r, section))
    flush()
    return "".join(out)


SUPP_TITLE = {
    "C": "Transition effect generation",
    "A": "Transition effect generation: comparison with previous work",
    "D": "Visual effect transfer",
    "B": "Visual effect transfer: comparison with previous work",
}


def render_supp_toc(man, plan, names):
    """Contents card (the supplementary's .seg-toc, restyled for the dark palette): the six sections
    in C-A-D-B-Guidance-Limitations order, with each gallery/comparison section's class headings
    (and the guidance/limitation parts) nested beneath, linking to the in-page anchors."""
    S = man["sections"]
    cblocks = {b["block_id"]: b for b in S["C"]["blocks"]}
    dblocks = {b["block_id"]: b for b in S["D"]["blocks"]}
    arows = {r["item_id"]: r for r in S["A"]["rows"]}
    brows = {r["item_id"]: r for r in S["B"]["rows"]}

    def classlinks(sec, pairs):
        li = []
        for cls, _ in pairs:
            li.append('<li><a href="#sec-%s-cls-%s">%s</a></li>'
                      % (sec, bs.slug(cls), html.escape(bs.human_name(cls, names))))
        return "<ul>%s</ul>" % "".join(li) if li else ""

    # section headers only (2026-10-08: no per-class or per-part sub-lists)
    toc = ['<nav class="seg-toc" id="contents"><div class="seg-toc-title">Contents</div><ul>']
    for anchor, title in (("sec-C", SUPP_TITLE["C"]), ("sec-A", SUPP_TITLE["A"]), ("sec-D", SUPP_TITLE["D"]),
                          ("sec-B", SUPP_TITLE["B"]), ("sec-guidance", "Guidance"), ("sec-limitations", "Limitations")):
        toc.append('<li><a href="#%s">%s</a></li>' % (anchor, html.escape(title)))
    toc.append("</ul></nav>")
    return "".join(toc)


def render_supp_sections(cfg, man, plan, names, warn):
    """The results-gallery body: the contents card, then the four galleries/comparisons in
    C-A-D-B order, then Guidance and Limitations — every section as a section() with the
    supplementary's SECTION_TITLE and SECTION_DESC (as a pp-cap line). Galleries via
    bs._render_gallery; comparison strips via render_supp_comparison (project-page rows); the
    guidance NRG row via guidance_row_html; limitations via bs._limitation_row_html with the
    LIMITATION_PARTS subheadings. FULL_ROWS wraps every comparison/guidance/limitation strip."""
    S = man["sections"]
    cblocks = {b["block_id"]: b for b in S["C"]["blocks"]}
    dblocks = {b["block_id"]: b for b in S["D"]["blocks"]}
    arows = {r["item_id"]: r for r in S["A"]["rows"]}
    brows = {r["item_id"]: r for r in S["B"]["rows"]}
    fr = " pp-full-rows" if FULL_ROWS else ""

    def section(body, title=None, anchor=None, container="is-max-widescreen"):
        aid = ' id="%s"' % anchor if anchor else ''
        h = ('<h2 class="title is-3 has-text-centered"%s>%s</h2>' % (aid, html.escape(title))
             if title else '')
        return ('<section class="section"><div class="container %s">%s%s</div></section>'
                % (container, h, body))

    def cap(sec):
        return '<p class="pp-cap">%s</p>' % html.escape(bs.SECTION_DESC[sec])

    out = []
    out.append(section(render_supp_toc(man, plan, names), container="is-max-desktop"))

    # C: Transition effect generation (gallery) ; A: its comparison
    out.append(section(cap("C") + '<div class="seg-section">%s</div>'
                       % render_supp_gallery("C", S["C"]["blocks"], plan, names),
                       title=SUPP_TITLE["C"], anchor="sec-C"))
    out.append(section(cap("A") + '<div class="seg-section%s">%s</div>'
                       % (fr, render_supp_comparison("A", S["A"]["rows"], plan, names)),
                       title=SUPP_TITLE["A"], anchor="sec-A"))

    # D: Visual effect transfer (gallery) ; B: its comparison
    out.append(section(cap("D") + '<div class="seg-section">%s</div>'
                       % render_supp_gallery("D", S["D"]["blocks"], plan, names),
                       title=SUPP_TITLE["D"], anchor="sec-D"))
    out.append(section(cap("B") + '<div class="seg-section%s">%s</div>'
                       % (fr, render_supp_comparison("B", S["B"]["rows"], plan, names)),
                       title=SUPP_TITLE["B"], anchor="sec-B"))

    # Guidance: the NRG part(s), each = subhead + one-line description + the guidance row(s)
    gbody = []
    for gid, gtitle, gdesc in bs.GUIDANCE_PARTS:
        gbody.append('<h3 class="pp-supp-sub" id="sec-guidance-%s">%s<span class="pp-supp-subdesc">%s</span></h3>'
                     % (gid, html.escape(gtitle), html.escape(gdesc)))
        grows = [r for r in man.get("guidance", []) if r["part"] == gid]
        if grows:
            gbody.append('<div class="seg-section pp-nrg-row pp-full-rows">%s</div>'
                         % "".join(guidance_row_html(r) for r in grows))
        else:
            warn.append("guidance: no rows for part %r" % gid)
    out.append(section("".join(gbody), title="Guidance", anchor="sec-guidance"))

    # Limitations: each part = subhead + one-line description + its failure-case row(s)
    lbody = []
    for lid, ltitle, ldesc in bs.LIMITATION_PARTS:
        lbody.append('<h3 class="pp-supp-sub" id="sec-limitations-%s">%s<span class="pp-supp-subdesc">%s</span></h3>'
                     % (lid, html.escape(ltitle), html.escape(ldesc)))
        lrows = [r for r in man.get("limitations", []) if r["part"] == lid]
        if lrows:
            # limitation rows stay small (fixed clip height, no full-width stretch), as before
            lbody.append('<div class="seg-section pp-small-rows">%s</div>'
                         % "".join(bs._limitation_row_html(r) for r in lrows))
        else:
            warn.append("limitations: no rows for part %r" % lid)
    out.append(section("".join(lbody), title="Limitations", anchor="sec-limitations"))

    return "\n".join(out)


def _supp_assets():
    """Create the media symlinks (videos/stills -> the supplementary build; assets -> ../assets so
    the vendored Bulma/index.css load unchanged) and copy the NRG start still into supp/img/. Never
    re-encodes or writes under supplementary/ — only reads/links it."""
    os.makedirs(SUPP_SITE, exist_ok=True)
    links = {"videos": "../../../supplementary/site/videos",
             "stills": "../../../supplementary/site/stills",
             "assets": "../assets"}
    for name, target in links.items():
        link = os.path.join(SUPP_SITE, name)
        if os.path.islink(link):
            if os.readlink(link) != target:
                os.remove(link); os.symlink(target, link)
        elif os.path.exists(link):
            raise SystemExit("refusing to overwrite non-symlink %s" % link)
        else:
            os.symlink(target, link)
        if not os.path.exists(link):       # the symlink target must resolve
            raise SystemExit("symlink %s -> %s does not resolve" % (link, target))
    img_dir = os.path.join(SUPP_SITE, "img")
    os.makedirs(img_dir, exist_ok=True)
    nrg = os.path.join(SITE, "img", "nrg_start.jpg")
    if not os.path.exists(nrg):
        raise SystemExit("missing %s (needed for the guidance row)" % nrg)
    shutil.copyfile(nrg, os.path.join(img_dir, "nrg_start.jpg"))



def do_supp():
    """Render site/supp/index.html: the full results gallery through template_light.html in the
    DARK theme, with the config's band accent so it matches the project page's showcase band."""
    cfg = _load(CONFIG)
    man = _load(MANIFEST)
    plan = _load(SITE_PLAN)
    picks = _load(PICKS)

    all_classes = set()
    for r in man["sections"]["A"]["rows"] + man["sections"]["B"]["rows"]:
        all_classes.add(r["ref_class"])
    for b in man["sections"]["C"]["blocks"] + man["sections"]["D"]["blocks"]:
        all_classes.add(b["ref_class"])
    names = bs.load_class_names(all_classes)

    SUPP_PICKED["blocks"] = {e["block"] for e in (picks.get("teg") or []) + (picks.get("vet") or [])}
    SUPP_PICKED["rows"] = set((picks.get("teg_cmp") or []) + (picks.get("vet_cmp") or []))
    _supp_assets()

    warn = []
    header = render_supp_header(cfg)
    sections = decorate(render_supp_sections(cfg, man, plan, names, warn))

    tpl = open(TEMPLATE_LIGHT).read()
    assert tpl.count("{{HEADER}}") == 1 and tpl.count("{{SECTIONS}}") == 1, \
        "template must have exactly one {{HEADER}} and one {{SECTIONS}}"
    assert tpl.count("__THEME__") == 1, "template must have exactly one __THEME__ sentinel"
    out = tpl.replace("{{HEADER}}", header).replace("{{SECTIONS}}", SUPP_CSS + sections)
    out = out.replace("__THEME__", "theme-dark")
    # accent = the config's BAND accent (what paints the project page's dark showcase band); the
    # whole supp page is theme-dark, so :root's --accent is the band accent throughout.
    accent = cfg.get("accent_band") or cfg.get("accent_dark_theme") or DEFAULT_ACCENT_DARK_THEME
    accent_dark = cfg.get("accent_band_dark") or cfg.get("accent_dark_theme_hover") or DEFAULT_ACCENT_DARK_THEME_HOVER
    out = out.replace("__BAND_DARK__", accent_dark).replace("__BAND__", accent)
    out = out.replace("__ACCENT_DARK__", accent_dark).replace("__ACCENT__", accent)
    out = _apply_seo(out, cfg, "supp")

    out = out.replace(*REF_LABEL)
    # the public gallery must carry no curation/overlay/internal leftovers
    for bad in ('data-hidden', 'seg-ov-', 'applyPlan', 'anonymous', '__ACCENT', '__BAND', '__THEME'):
        assert bad not in out, "supp gallery not clean: found %r" % bad
    assert 'owner' not in out.lower(), "supp gallery not clean: found 'owner'"

    os.makedirs(SUPP_SITE, exist_ok=True)
    open(SUPP_INDEX, "w").write(out)
    print("[supp] wrote %s (%.0f KB) [dark, band accent]"
          % (os.path.relpath(SUPP_INDEX, REPO), len(out) / 1024))
    for w in warn:
        print("[supp] WARNING:", w)
    if not warn:
        print("[supp] no warnings")
    return warn


def do_check_supp():
    """--check for site/supp/index.html: every <source data-src>/<img src> resolves under site/supp/
    (through the symlinks), stylesheets allow-listed, and the cleanliness greps pass."""
    page = SUPP_INDEX
    if not os.path.exists(page):
        sys.exit("[check] %s does not exist — run --supp first" % os.path.relpath(page, REPO))
    raw = open(page).read()
    text = re.sub(r"<script\b.*?</script>", "", raw, flags=re.DOTALL)
    srcs = re.findall(r'<source[^>]+(?:data-src|src)="([^"]+)"', text)
    imgs = re.findall(r'<img[^>]+src="([^"]+)"', text)
    hrefs = re.findall(r'href="([^"]+)"', text)

    bad_abs, missing = [], []
    for s in srcs + imgs:
        if s.startswith("/") or s.startswith("http://") or s.startswith("https://"):
            bad_abs.append(s); continue
        if not os.path.exists(os.path.join(SUPP_SITE, s)):
            missing.append(s)

    allowed_ext = {"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
                   "https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css",
                   "https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"}
    sheets = set(re.findall(r'<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"', text))
    sheets |= set(re.findall(r'<link[^>]*href="([^"]+)"[^>]*rel="stylesheet"', text))
    ext = [h for h in hrefs if (h.startswith("http://") or h.startswith("https://")) and h in sheets]
    outbound = sorted({h for h in hrefs if (h.startswith("http://") or h.startswith("https://")) and h not in sheets})
    print("[check] outbound links: %s" % (outbound or "none"))
    stray_ext = [h for h in ext if h not in allowed_ext]

    uniq = sorted(set(s for s in srcs + imgs if s not in bad_abs and s not in missing))
    total = sum(os.path.getsize(os.path.join(SUPP_SITE, s)) for s in uniq)
    clips = sorted(set(s for s in srcs if s not in bad_abs and s not in missing))
    stills = sorted(set(s for s in imgs if s not in bad_abs and s not in missing
                        and s.lower().endswith((".jpg", ".jpeg", ".png")) and "/videos/" not in s))

    print("[check] page: %s [supp]" % os.path.relpath(page, REPO))
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

    for bad in ('data-hidden', 'seg-ov-', 'applyPlan', 'anonymous', 'ICLR', 'PixelDiT', 'NVIDIA'):
        if bad in raw:
            print("[check]   NOT-CLEAN token present:", bad); ok = False
    if 'owner' in raw.lower():
        print("[check]   NOT-CLEAN token present: owner"); ok = False

    if ok:
        print("[check] OK: every media ref resolves inside site/supp/, page is clean, all external "
              "stylesheets are allow-listed")
    else:
        sys.exit("[check] FAILED")


# --------------------------------------------------------------------------- html

def seo_head(cfg, page):
    """Search-engine metadata (2026-10-09): full paper title in <title>, description, canonical,
    Open Graph / Twitter cards and a ScholarlyArticle JSON-LD block, so the page is indexable and
    shows a proper card when linked. page = "main" | "supp"."""
    import re as _re
    site = (cfg.get("site_url") or "https://segue-video.github.io/").rstrip("/") + "/"
    title = cfg.get("title") or "SEGUE"
    desc = _re.sub(r"<[^>]+>", "", TLDR_HTML)
    if page == "supp":
        url, ptitle = site + "supp/", title + " (results gallery)"
        desc = "Full results gallery for SEGUE: transition effect generation and visual effect transfer, comparisons with previous work, guidance and limitations. " + desc
    else:
        url, ptitle = site, title
    img = site + "img/method.png"
    authors = [{"@type": "Person", "name": a["name"], **({"url": a["url"]} if a.get("url") else {})}
               for a in cfg.get("authors", [])]
    ld = {"@context": "https://schema.org", "@type": "ScholarlyArticle", "headline": title,
          "name": title, "author": authors, "url": site, "image": img,
          "description": _re.sub(r"<[^>]+>", "", TLDR_HTML)}
    if cfg.get("links", {}).get("paper_pdf"):
        ld["encoding"] = {"@type": "MediaObject", "contentUrl": site + cfg["links"]["paper_pdf"], "encodingFormat": "application/pdf"}
    if cfg.get("links", {}).get("code"):
        ld["codeRepository"] = cfg["links"]["code"]
    e = lambda t: html.escape(t, quote=True)
    return ("".join([
        '<title>%s</title>\n' % e(ptitle),
        '<meta name="description" content="%s"/>\n' % e(desc),
        '<meta name="robots" content="index,follow,max-image-preview:large"/>\n',
        '<link rel="canonical" href="%s"/>\n' % e(url),
        '<meta property="og:type" content="website"/>\n',
        '<meta property="og:site_name" content="SEGUE"/>\n',
        '<meta property="og:title" content="%s"/>\n' % e(ptitle),
        '<meta property="og:description" content="%s"/>\n' % e(desc),
        '<meta property="og:url" content="%s"/>\n' % e(url),
        '<meta property="og:image" content="%s"/>\n' % e(img),
        '<meta name="twitter:card" content="summary_large_image"/>\n',
        '<meta name="twitter:title" content="%s"/>\n' % e(ptitle),
        '<meta name="twitter:description" content="%s"/>\n' % e(desc),
        '<meta name="twitter:image" content="%s"/>\n' % e(img),
        '<script type="application/ld+json">%s</script>\n' % json.dumps(ld, ensure_ascii=False),
    ]))


def _apply_seo(out, cfg, page):
    import re as _re
    out = _re.sub(r"<title>.*?</title>\s*", "", out, count=1, flags=_re.S)
    return out.replace("</head>", seo_head(cfg, page) + "</head>", 1)


def write_robots_and_sitemap(cfg):
    site = (cfg.get("site_url") or "https://segue-video.github.io/").rstrip("/") + "/"
    open(os.path.join(SITE, "robots.txt"), "w").write("User-agent: *\nAllow: /\nSitemap: %ssitemap.xml\n" % site)
    open(os.path.join(SITE, "sitemap.xml"), "w").write(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        '  <url><loc>%s</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n'
        '  <url><loc>%ssupp/</loc><changefreq>monthly</changefreq><priority>0.8</priority></url>\n'
        '</urlset>\n' % (site, site))

def do_html(theme="dark"):
    """Render one theme. BOTH themes use TEMPLATE_LIGHT and the SAME light-style renderers; the
    two outputs differ ONLY in the <html> theme class (theme-dark / theme-light) and the injected
    accent values (dark: accent_dark_theme / accent_dark_theme_hover; light: accent / accent_dark)."""
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
    header = render_header(cfg)
    sections = decorate(render_sections_light(cfg, picks, man, plan, names, warn))

    if theme == "dark":
        theme_class = "theme-dark"
        accent = cfg.get("accent_dark_theme") or DEFAULT_ACCENT_DARK_THEME
        accent_dark = cfg.get("accent_dark_theme_hover") or DEFAULT_ACCENT_DARK_THEME_HOVER
        out_path = INDEX
    else:
        theme_class = "theme-light"
        accent = cfg.get("accent") or DEFAULT_ACCENT
        accent_dark = cfg.get("accent_dark") or DEFAULT_ACCENT_DARK
        out_path = LIGHT_INDEX

    tpl = open(TEMPLATE_LIGHT).read()
    assert tpl.count("{{HEADER}}") == 1 and tpl.count("{{SECTIONS}}") == 1, \
        "template must have exactly one {{HEADER}} and one {{SECTIONS}}"
    assert tpl.count("__THEME__") == 1, "template must have exactly one __THEME__ sentinel"
    out = tpl.replace("{{HEADER}}", header).replace("{{SECTIONS}}", sections)
    out = out.replace("__THEME__", theme_class)
    # __ACCENT_DARK__ before __ACCENT__ (the latter is a substring of the former)
    band = cfg.get("accent_band") or accent
    band_dark = cfg.get("accent_band_dark") or accent_dark
    out = out.replace("__BAND_DARK__", band_dark).replace("__BAND__", band)
    out = out.replace("__ACCENT_DARK__", accent_dark).replace("__ACCENT__", accent)
    out = _apply_seo(out, cfg, "main")
    write_robots_and_sitemap(cfg)

    out = out.replace(*REF_LABEL)
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
    # only <link rel="stylesheet"> targets are checked against the allow-list; the header buttons
    # (arXiv, GitHub, Hugging Face, …) are ordinary outbound links and are listed, not flagged
    sheets = set(re.findall(r'<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"', text))
    sheets |= set(re.findall(r'<link[^>]*href="([^"]+)"[^>]*rel="stylesheet"', text))
    ext = [h for h in hrefs if (h.startswith("http://") or h.startswith("https://")) and h in sheets]
    outbound = sorted({h for h in hrefs if (h.startswith("http://") or h.startswith("https://")) and h not in sheets})
    print("[check] outbound links: %s" % (outbound or "none"))
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
    ap.add_argument("--supp", action="store_true",
                    help="render the full results gallery site/supp/index.html (dark theme, band "
                         "accent; also built by the default no-flag invocation after the two themes)")
    ap.add_argument("--full-rows", action="store_true", help="comparison rows at full width (no slider) on desktop")
    ap.add_argument("--theme", choices=["dark", "light", "both"], default="both",
                    help="which variant(s) to build/check: dark -> site/index.html, "
                         "light -> site/light.html (default: both)")
    a = ap.parse_args()
    global FULL_ROWS
    FULL_ROWS = FULL_ROWS or bool(a.full_rows)
    if not (a.html or a.check or a.candidates or a.supp):   # default = html + supp + check
        a.html = a.check = a.supp = True
    themes = ["dark", "light"] if a.theme == "both" else [a.theme]
    if a.candidates:
        do_candidates()
    if a.html:
        for t in themes:
            do_html(t)
    if a.supp:
        do_supp()
    if a.check:
        for t in themes:
            do_check(t)
        if os.path.exists(SUPP_INDEX):
            do_check_supp()


if __name__ == "__main__":
    main()
