#!/usr/bin/env python3
"""viewerctl — the viewer system for diffusion-research.

One tracked registry (registry.json) describes every viewer worth keeping.
`outputs/` is gitignored and disposable, so nothing durable may live only there:
the registry is the source of truth and this script reconstitutes the rest.

Subcommands
-----------
  mount [--all|SLUG]   build outputs/viewers/<slug>/ mount dirs from the registry
  check [SLUG]         health-check every viewer (does the page exist? does its media resolve?)
  hub                  write outputs/viewers/index.html — the dashboard
  serve [--port 8017]  free the port, mount, build the hub, serve the repo root
  new SLUG             scaffold a new viewer directory

The whole system in one sentence: a viewer is a directory under
outputs/viewers/<slug>/ holding index.html plus symlinks to its media, and every
path inside the page is relative to that directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                      # .../diffusion-research
REGISTRY = HERE / "registry.json"
VIEWERS_DIR = REPO / "outputs" / "viewers"
HUB = VIEWERS_DIR / "index.html"
DEFAULT_PORT = 8017

MEDIA_RE = re.compile(r"""["'(]([^"'()\s>]+\.(?:mp4|webm|png|jpg|jpeg|webp|gif|json|csv))["')]""", re.I)
SAMPLE_N = 12

# ─────────────────────────────────────────────────────────── registry ────


def load_registry() -> dict:
    with open(REGISTRY, encoding="utf-8") as fh:
        return json.load(fh)


def viewers(reg: dict) -> list[dict]:
    return reg["viewers"]


def servers(reg: dict) -> list[dict]:
    """Viewers that are an application, not a file: they run their own HTTP
    server (streaming out of tar shards, querying an index) and cannot be served
    by the static file server."""
    return reg.get("servers", [])


def probe(port: int, path: str = "/") -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2)
        return True
    except Exception:
        return False


def by_slug(reg: dict, slug: str) -> dict:
    for v in viewers(reg):
        if v["slug"] == slug:
            return v
    sys.exit(f"unknown viewer slug: {slug}")


# ───────────────────────────────────────────────────────────── mount ─────


def _link(link_path: Path, target_repo_rel: str) -> str:
    """Create link_path -> target, as a path relative to the link's own dir."""
    target = REPO / target_repo_rel
    if not target.exists():
        return f"MISSING TARGET {target_repo_rel}"
    rel = os.path.relpath(target, link_path.parent)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and os.readlink(link_path) == rel:
            return "ok (unchanged)"
        if link_path.is_symlink():
            link_path.unlink()
        else:
            return f"SKIP (real file in the way): {link_path}"
    link_path.symlink_to(rel)
    return "ok"


def mount_one(v: dict, verbose: bool = True) -> None:
    """Build outputs/viewers/<slug>/ for a viewer that has a `mount` spec.

    A mount dir turns a page that lives elsewhere (an eval record, an experiment
    run) into a normal viewer: the page is symlinked in, and every directory the
    page's relative paths expect is symlinked beside it.
    """
    spec = v.get("mount")
    if not spec:
        return
    d = REPO / spec["dir"]
    d.mkdir(parents=True, exist_ok=True)
    log = []

    for page in spec.get("pages", []):
        dst = d / page["name"]
        src = REPO / page["target"]
        if not src.exists():
            log.append(f"  {page['name']}: MISSING SOURCE {page['target']}")
            continue
        rewrite = page.get("rewrite_abs")
        if rewrite:
            # The original page is frozen (a published record, a blind study):
            # never edit it. Write a derived copy with its absolute server paths
            # made relative to the mount dir.
            text = src.read_text(encoding="utf-8", errors="replace")
            for old, new in rewrite.items():
                text = text.replace(old, new)
            dst.write_text(text, encoding="utf-8")
            log.append(f"  {page['name']}: derived copy (rewrote {len(rewrite)} prefix)")
        else:
            log.append(f"  {page['name']}: {_link(dst, page['target'])}")

    for link in spec.get("links", []):
        log.append(f"  {link['name']}/: {_link(d / link['name'], link['target'])}")

    bulk = spec.get("links_from")
    if bulk:
        base = REPO / bulk
        made = err = 0
        for child in sorted(base.iterdir()):
            r = _link(d / child.name, f"{bulk}/{child.name}")
            made += r.startswith("ok")
            err += not r.startswith("ok")
        log.append(f"  links_from {bulk}: {made} ok, {err} problem")

    if verbose:
        print(f"[mount] {v['slug']} -> {spec['dir']}")
        print("\n".join(log))


def cmd_mount(args) -> None:
    reg = load_registry()
    targets = viewers(reg) if args.all or not args.slug else [by_slug(reg, args.slug)]
    n = 0
    for v in targets:
        if v.get("mount"):
            mount_one(v)
            n += 1
    print(f"\n{n} mount dir(s) built.")


# ───────────────────────────────────────────────────────────── check ─────


def page_path(v: dict) -> Path:
    return REPO / v["path"]


def health(v: dict, sample_n: int = SAMPLE_N) -> dict:
    """Resolve a sample of a page's media references the way a browser would.

    Relative refs resolve against the page's directory; refs starting with "/"
    resolve against the server root (the repo root). Anything http(s) is external
    and is reported separately rather than counted as broken.
    """
    p = page_path(v)
    out = {"status": "missing", "ok": 0, "sampled": 0, "refs": 0, "external": 0, "examples": []}
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="replace")
    seen, refs, external = set(), [], 0
    for m in MEDIA_RE.finditer(text):
        r = m.group(1)
        if r.startswith(("http://", "https://", "//", "data:")):
            external += 1
            continue
        if r in seen:
            continue
        seen.add(r)
        refs.append(r)
    out["refs"], out["external"] = len(refs), external

    if not refs:
        out["status"] = "standalone" if external else "standalone"
        return out

    step = max(1, len(refs) // sample_n)
    sample = refs[::step][:sample_n]
    bad = []
    subdirs = [d for d in p.parent.iterdir() if d.is_dir()][:60] if p.parent.is_dir() else []
    for r in sample:
        target = (REPO / r.lstrip("/")) if r.startswith("/") else (p.parent / r)
        if target.exists():
            out["ok"] += 1
        # A page may compose paths in JS ("<row>/" + file), so a bare filename can
        # still be reachable one directory down. Count that as resolved.
        elif not r.startswith("/") and any((d / r).exists() for d in subdirs):
            out["ok"] += 1
            out["composed"] = True
        else:
            bad.append(r)
    out["sampled"] = len(sample)
    out["examples"] = bad[:3]
    if out["ok"] == out["sampled"]:
        out["status"] = "live"
    elif out["ok"]:
        out["status"] = "partial"
    else:
        out["status"] = "broken"
    return out


def cmd_check(args) -> None:
    reg = load_registry()
    targets = [by_slug(reg, args.slug)] if args.slug else viewers(reg)
    worst = 0
    for v in targets:
        h = health(v)
        badge = {"live": "LIVE", "partial": "PARTIAL", "broken": "BROKEN",
                 "standalone": "SELF-CONTAINED", "missing": "MISSING"}[h["status"]]
        line = f"{badge:15s} {h['ok']}/{h['sampled']:<3d} refs={h['refs']:<5d}"
        if h["external"]:
            line += f" ext={h['external']:<4d}"
        tag = "  [archived]" if v.get("archived") else ""
        print(f"{line}  {v['slug']}{tag}")
        if h["examples"]:
            print(f"{'':15s} e.g. {h['examples'][0]}")
        # Archived viewers are expected to be broken — that is what archived means.
        if not v.get("archived"):
            worst = max(worst, {"live": 0, "standalone": 0, "partial": 1,
                                "broken": 2, "missing": 2}[h["status"]])
    if args.strict:
        sys.exit(1 if worst >= 2 else 0)


# ─────────────────────────────────────────────────────────────── hub ─────

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#f7f7f5; --panel:#fff; --ink:#16150f; --muted:#6b6960; --line:#e3e1da;
  --accent:#8c5a2b; --accent-soft:#f0e6db;
  --live:#3f7a4a; --live-bg:#e6f0e6; --partial:#8a6d1f; --partial-bg:#f6eed6;
  --dead:#8f4139; --dead-bg:#f6e2df; --self:#3f6480; --self-bg:#e2ecf3;
  --radius:10px; --shadow:0 1px 2px rgba(20,18,12,.06),0 4px 14px rgba(20,18,12,.05);
}
@media (prefers-color-scheme:dark){
  :root{--bg:#131311;--panel:#1c1c19;--ink:#eceae2;--muted:#9a978c;--line:#2e2e29;
    --accent:#d3a06a;--accent-soft:#2a2219;
    --live:#8fc79a;--live-bg:#1d2a1f;--partial:#d9bd6d;--partial-bg:#2b2718;
    --dead:#e09b92;--dead-bg:#2e1f1d;--self:#9dc3dc;--self-bg:#1c262d;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 14px rgba(0,0,0,.3);}
}
:root[data-theme="dark"]{--bg:#131311;--panel:#1c1c19;--ink:#eceae2;--muted:#9a978c;--line:#2e2e29;
  --accent:#d3a06a;--accent-soft:#2a2219;--live:#8fc79a;--live-bg:#1d2a1f;--partial:#d9bd6d;
  --partial-bg:#2b2718;--dead:#e09b92;--dead-bg:#2e1f1d;--self:#9dc3dc;--self-bg:#1c262d;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 14px rgba(0,0,0,.3)}
:root[data-theme="light"]{--bg:#f7f7f5;--panel:#fff;--ink:#16150f;--muted:#6b6960;--line:#e3e1da;
  --accent:#8c5a2b;--accent-soft:#f0e6db;--live:#3f7a4a;--live-bg:#e6f0e6;--partial:#8a6d1f;
  --partial-bg:#f6eed6;--dead:#8f4139;--dead-bg:#f6e2df;--self:#3f6480;--self-bg:#e2ecf3;
  --shadow:0 1px 2px rgba(20,18,12,.06),0 4px 14px rgba(20,18,12,.05)}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 90px}
header.top{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;justify-content:space-between;
  padding-bottom:18px;border-bottom:1px solid var(--line);margin-bottom:26px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em;font-weight:640}
.sub{color:var(--muted);font-size:13.5px;margin:0}
.tools{display:flex;gap:8px;align-items:center}
input[type=search]{background:var(--panel);border:1px solid var(--line);color:var(--ink);
  border-radius:var(--radius);padding:8px 12px;font-size:13.5px;min-width:210px;font-family:inherit}
input[type=search]:focus{outline:2px solid var(--accent);outline-offset:1px}
button.ghost{background:var(--panel);border:1px solid var(--line);color:var(--muted);
  border-radius:var(--radius);padding:8px 11px;font-size:13px;cursor:pointer;font-family:inherit}
button.ghost:hover{color:var(--ink);border-color:var(--accent)}
h2.group{font-size:12.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
  margin:34px 0 4px;font-weight:620}
h2.group:first-of-type{margin-top:8px}
.group-note{color:var(--muted);font-size:13px;margin:0 0 14px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(310px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  padding:15px 16px 13px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:9px;
  position:relative;transition:border-color .12s ease,transform .12s ease}
.card:hover{border-color:var(--accent);transform:translateY(-1px)}
.card.feature{border-left:3px solid var(--accent)}
.card h3{margin:0;font-size:15.5px;font-weight:620;letter-spacing:-.01em;line-height:1.32}
.card h3 a{color:inherit;text-decoration:none}
.card h3 a::after{content:"";position:absolute;inset:0}
.card h3 a:hover{color:var(--accent)}
.blurb{margin:0;color:var(--muted);font-size:13.2px;line-height:1.5}
.meta{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:auto;padding-top:4px}
.pill{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;font-weight:640;
  padding:3px 7px;border-radius:5px;white-space:nowrap}
.pill.live{background:var(--live-bg);color:var(--live)}
.pill.partial{background:var(--partial-bg);color:var(--partial)}
.pill.broken,.pill.missing{background:var(--dead-bg);color:var(--dead)}
.pill.standalone{background:var(--self-bg);color:var(--self)}
.pill.date{background:transparent;color:var(--muted);border:1px solid var(--line);
  text-transform:none;letter-spacing:.02em;font-weight:520}
.pill.latest{background:var(--accent-soft);color:var(--accent)}
.extra{display:flex;flex-wrap:wrap;gap:8px;font-size:12.5px;position:relative;z-index:1}
.extra a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent}
.extra a:hover{border-bottom-color:var(--accent)}
details.versions{position:relative;z-index:1;font-size:12.3px;color:var(--muted)}
details.versions summary{cursor:pointer;list-style:none;color:var(--muted);
  border-bottom:1px dotted var(--line);display:inline-block;padding-bottom:1px}
details.versions summary::-webkit-details-marker{display:none}
details.versions summary:hover{color:var(--ink)}
details.versions ul{margin:7px 0 0;padding-left:16px;display:flex;flex-direction:column;gap:3px}
details.versions a{color:var(--muted);text-decoration:none;border-bottom:1px solid var(--line)}
details.versions a:hover{color:var(--accent)}
.archive{margin-top:40px;border-top:1px solid var(--line);padding-top:8px}
.archive summary{cursor:pointer;color:var(--muted);font-size:12.5px;text-transform:uppercase;
  letter-spacing:.09em;font-weight:620;padding:14px 0;list-style:none}
.archive summary::-webkit-details-marker{display:none}
.archive summary:hover{color:var(--ink)}
.archive table{width:100%;border-collapse:collapse;font-size:13px}
.archive td{padding:7px 10px 7px 0;border-bottom:1px solid var(--line);vertical-align:top}
.archive td a{color:var(--accent);text-decoration:none}
.archive td.why{color:var(--muted)}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px;display:flex;flex-wrap:wrap;gap:14px;justify-content:space-between}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em;
  background:var(--accent-soft);color:var(--accent);padding:1px 5px;border-radius:4px}
.empty{color:var(--muted);font-size:13px;padding:18px 0}
@media (max-width:640px){.wrap{padding:22px 15px 60px}.grid{grid-template-columns:1fr}
  header.top{align-items:flex-start}input[type=search]{min-width:0;width:100%}}
"""

JS = """
const q=document.getElementById('q');
const cards=[...document.querySelectorAll('.card')];
q.addEventListener('input',()=>{
  const t=q.value.trim().toLowerCase();
  cards.forEach(c=>{c.style.display=!t||c.dataset.hay.includes(t)?'':'none'});
  document.querySelectorAll('section.group-sec').forEach(s=>{
    const any=[...s.querySelectorAll('.card')].some(c=>c.style.display!=='none');
    s.style.display=any?'':'none';
  });
});
const root=document.documentElement;
const saved=localStorage.getItem('viewerhub-theme');
if(saved)root.dataset.theme=saved;
document.getElementById('theme').addEventListener('click',()=>{
  const dark=root.dataset.theme?root.dataset.theme==='dark'
    :matchMedia('(prefers-color-scheme:dark)').matches;
  root.dataset.theme=dark?'light':'dark';
  localStorage.setItem('viewerhub-theme',root.dataset.theme);
});
"""


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def url_for(repo_rel: str) -> str:
    return "/" + repo_rel.lstrip("/")


def card_html(v: dict, h: dict) -> str:
    status = h["status"]
    label = {"live": "live", "partial": "partial media", "broken": "media missing",
             "standalone": "self-contained", "missing": "page missing"}[status]
    pills = [f'<span class="pill {status}">{label}</span>']
    if v.get("latest_of"):
        pills.append(f'<span class="pill latest">latest · {esc(v["latest_of"])}</span>')
    if v.get("date"):
        pills.append(f'<span class="pill date">{esc(v["date"])}</span>')
    if h["refs"]:
        pills.append(f'<span class="pill date">{h["refs"]} media</span>')
    elif h["external"]:
        pills.append(f'<span class="pill date">{h["external"]} remote</span>')

    extra = ""
    if v.get("pages"):
        links = " ".join(
            f'<a href="{esc(url_for(p["path"]))}">{esc(p["label"])} →</a>'
            for p in v["pages"])
        extra = f'<div class="extra">{links}</div>'

    versions = ""
    if v.get("supersedes"):
        items = "".join(
            f'<li><a href="{esc(url_for(o["path"]))}">{esc(o["label"])}</a></li>'
            for o in v["supersedes"])
        n = len(v["supersedes"])
        versions = (f'<details class="versions"><summary>{n} earlier '
                    f'version{"s" if n > 1 else ""}</summary><ul>{items}</ul></details>')

    hay = esc(" ".join([v["slug"], v["title"], v.get("blurb", ""),
                        v.get("group", ""), v.get("latest_of", "")]).lower())
    feature = " feature" if v.get("featured") else ""
    return f"""      <article class="card{feature}" data-hay="{hay}">
        <h3><a href="{esc(url_for(v['path']))}">{esc(v['title'])}</a></h3>
        <p class="blurb">{esc(v.get('blurb', ''))}</p>
        {extra}
        {versions}
        <div class="meta">{''.join(pills)}</div>
      </article>"""


def server_card_html(s: dict) -> str:
    up = probe(s["port"])
    pills = [f'<span class="pill {"live" if up else "partial"}">'
             f'{"running" if up else "not running"} · port {s["port"]}</span>']
    if s.get("latest_of"):
        pills.append(f'<span class="pill latest">latest · {esc(s["latest_of"])}</span>')
    if s.get("date"):
        pills.append(f'<span class="pill date">{esc(s["date"])}</span>')
    pills.append('<span class="pill date">own server</span>')
    hay = esc(" ".join([s["slug"], s["title"], s.get("blurb", "")]).lower())
    start = esc(s.get("start_hint", ""))
    note = (f'<div class="extra"><code>{start}</code></div>' if start else "")
    return f"""      <article class="card{' feature' if s.get('featured') else ''}" data-hay="{hay}">
        <h3><a href="http://localhost:{s['port']}/">{esc(s['title'])}</a></h3>
        <p class="blurb">{esc(s.get('blurb', ''))}</p>
        {note}
        <div class="meta">{''.join(pills)}</div>
      </article>"""


def build_hub(reg: dict) -> str:
    checks = {v["slug"]: health(v) for v in viewers(reg)}
    live = sum(1 for v in viewers(reg)
               if not v.get("archived") and checks[v["slug"]]["status"] in ("live", "standalone"))
    groups = {g["id"]: g for g in reg["groups"]}
    sections = []
    for gid, g in groups.items():
        vs = [v for v in viewers(reg) if v.get("group") == gid and not v.get("archived")]
        srv = [s for s in servers(reg) if s.get("group") == gid]
        if not vs and not srv:
            continue
        vs.sort(key=lambda v: (not v.get("featured"), -_datekey(v.get("date", "")), v["title"]))
        cards = "\n".join([card_html(v, checks[v["slug"]]) for v in vs]
                          + [server_card_html(s) for s in srv])
        note = f'<p class="group-note">{esc(g["blurb"])}</p>' if g.get("blurb") else ""
        sections.append(f"""    <section class="group-sec">
      <h2 class="group">{esc(g['title'])}</h2>
      {note}
      <div class="grid">
{cards}
      </div>
    </section>""")

    archived = [v for v in viewers(reg) if v.get("archived")]
    arch_html = ""
    if archived:
        rows = "".join(
            f'<tr><td><a href="{esc(url_for(v["path"]))}">{esc(v["title"])}</a></td>'
            f'<td class="why">{esc(v.get("archived", ""))}</td>'
            f'<td class="why">{esc(v.get("date", ""))}</td></tr>'
            for v in sorted(archived, key=lambda v: -_datekey(v.get("date", ""))))
        arch_html = f"""    <details class="archive">
      <summary>Archive — {len(archived)} superseded or stale viewer(s)</summary>
      <table>{rows}</table>
    </details>"""

    built = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(viewers(reg)) - len(archived) + len(servers(reg))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>diffusion-research — viewers</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1>diffusion-research · viewers</h1>
      <p class="sub">{total} current viewers · {live} with all media resolving · {len(archived)} archived · built {built}</p>
    </div>
    <div class="tools">
      <input type="search" id="q" placeholder="filter viewers…" aria-label="filter viewers">
      <button class="ghost" id="theme" title="toggle light / dark">◐</button>
    </div>
  </header>
{chr(10).join(sections)}
{arch_html}
  <footer>
    <span>Rebuild: <code>python3 scripts/viewers/viewerctl.py hub</code> · serve: <code>viewerctl.py serve</code></span>
    <span>Registry: <code>scripts/viewers/registry.json</code></span>
  </footer>
</div>
<script>{JS}</script>
</body>
</html>
"""


def _datekey(d: str) -> int:
    try:
        return int(d.replace("-", ""))
    except ValueError:
        return 0


def cmd_hub(args) -> None:
    reg = load_registry()
    VIEWERS_DIR.mkdir(parents=True, exist_ok=True)
    HUB.write_text(build_hub(reg), encoding="utf-8")
    print(f"[hub] wrote {HUB.relative_to(REPO)}")


# ───────────────────────────────────────────────────────────── serve ─────


def free_port(port: int) -> None:
    """Kill whatever this user has listening on `port` (ours to reclaim)."""
    killed = []
    try:
        out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        out = ""
    for pid in set(re.findall(r"pid=(\d+)", out)):
        try:
            os.kill(int(pid), signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            sys.exit(f"port {port} is held by pid {pid}, owned by another user — pick another port")
    if killed:
        time.sleep(0.6)
        print(f"[serve] freed port {port} (killed pid {', '.join(killed)})")


def cmd_serve(args) -> None:
    port = args.port
    reg = load_registry()
    free_port(port)
    if not args.no_build:
        for v in viewers(reg):
            if v.get("mount"):
                mount_one(v, verbose=False)
        HUB.write_text(build_hub(reg), encoding="utf-8")

    log = Path(args.log or (REPO / "outputs" / "viewers" / ".server.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
         "--directory", str(REPO)],
        stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(0.8)
    if proc.poll() is not None:
        sys.exit(f"server died immediately — see {log}")

    # Server-backed viewers are applications, not files — start each on its own port.
    if not args.static_only:
        for s in servers(reg):
            if probe(s["port"]):
                print(f"[serve] {s['slug']} already up on {s['port']}")
                continue
            free_port(s["port"])
            slog = log.parent / f".{s['slug']}.log"
            cwd = REPO / s.get("cwd", ".")
            cmd = list(s["cmd"])
            if cmd[0] in ("python", "python3"):
                cmd[0] = sys.executable
            script = cwd / cmd[1] if len(cmd) > 1 else None
            if script and not script.exists():
                print(f"[serve] {s['slug']}: SKIP — {cmd[1]} not found under {cwd}")
                continue
            subprocess.Popen(cmd, cwd=cwd, stdout=open(slog, "w"),
                             stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(20):
                time.sleep(0.5)
                if probe(s["port"]):
                    break
            print(f"[serve] {s['slug']} on {s['port']}: "
                  f"{'up' if probe(s['port']) else f'FAILED — see {slog}'}")

    print(f"[serve] repo root on http://localhost:{port}  (pid {proc.pid}, log {log})")
    print(f"\n  ➜  DASHBOARD   http://localhost:{port}/outputs/viewers/index.html\n")
    for g in reg["groups"]:
        vs = [v for v in viewers(reg) if v.get("group") == g["id"] and not v.get("archived")]
        srv = [s for s in servers(reg) if s.get("group") == g["id"]]
        if not vs and not srv:
            continue
        print(f"  {g['title']}")
        for v in sorted(vs, key=lambda v: (not v.get("featured"), v["title"])):
            h = health(v)
            mark = {"live": " ", "standalone": " ", "partial": "~", "broken": "!", "missing": "!"}[h["status"]]
            print(f"   {mark} http://localhost:{port}{url_for(v['path'])}")
            print(f"       {v['title']}")
        for s in srv:
            mark = " " if probe(s["port"]) else "!"
            print(f"   {mark} http://localhost:{s['port']}/")
            print(f"       {s['title']}  (own server)")
        print()


# ─────────────────────────────────────────────────────────────── new ─────

SCAFFOLD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{slug}</title>
<style>
 body{{margin:0;background:#131311;color:#eceae2;font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}
 .wrap{{max-width:1100px;margin:0 auto;padding:28px 20px}}
 h1{{font-size:22px;margin:0 0 4px}} p.sub{{color:#9a978c;margin:0 0 22px;font-size:13px}}
 .grid{{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(240px,1fr))}}
 video{{width:100%;border-radius:8px;background:#000}}
</style>
</head>
<body>
<div class="wrap">
  <h1>{slug}</h1>
  <p class="sub">what this shows · how it was made · what to look for</p>
  <!-- every path below must be RELATIVE to this directory, through ./media -->
  <div class="grid" id="grid"></div>
</div>
<script>
const CLIPS = [];  // e.g. ["media/clip_000.mp4", ...]
document.getElementById('grid').innerHTML = CLIPS.map(
  s => `<div><video src="${{s}}" muted loop playsinline controls></video></div>`).join('');
</script>
</body>
</html>
"""


def cmd_new(args) -> None:
    d = VIEWERS_DIR / args.slug
    d.mkdir(parents=True, exist_ok=True)
    page = d / "index.html"
    if page.exists():
        sys.exit(f"{page} already exists")
    page.write_text(SCAFFOLD.format(slug=args.slug), encoding="utf-8")
    if args.media:
        print(f"  media/: {_link(d / 'media', args.media)}")
    print(f"[new] {page.relative_to(REPO)}")
    print("\nNext:\n  1. point media/ at the clips  "
          "(ln -s ../../videos/<run> outputs/viewers/%s/media)" % args.slug)
    print("  2. add an entry to scripts/viewers/registry.json")
    print("  3. python3 scripts/viewers/viewerctl.py hub")


# ────────────────────────────────────────────────────────────── main ─────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mount", help="build mount dirs from the registry")
    m.add_argument("slug", nargs="?")
    m.add_argument("--all", action="store_true")
    m.set_defaults(func=cmd_mount)

    c = sub.add_parser("check", help="health-check viewers")
    c.add_argument("slug", nargs="?")
    c.add_argument("--strict", action="store_true", help="exit 1 if anything is broken")
    c.set_defaults(func=cmd_check)

    h = sub.add_parser("hub", help="write the dashboard")
    h.set_defaults(func=cmd_hub)

    s = sub.add_parser("serve", help="free the port, rebuild, serve the repo root")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--no-build", action="store_true")
    s.add_argument("--static-only", action="store_true",
                   help="do not start server-backed viewers on their own ports")
    s.add_argument("--log")
    s.set_defaults(func=cmd_serve)

    n = sub.add_parser("new", help="scaffold a viewer directory")
    n.add_argument("slug")
    n.add_argument("--media", help="repo-relative dir to symlink as ./media")
    n.set_defaults(func=cmd_new)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
