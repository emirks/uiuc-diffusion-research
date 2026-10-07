"""Render docs/PROPOSAL_v2.md as a static viewer page (outputs/viewers/proposal_v2).

Markdown + TeX render client-side (marked + MathJax, CDN). Math spans are
placeholder-protected before marked runs so underscores inside $...$ survive.
Run from any checkout; output goes to the main checkout's outputs/ so the
8017 server picks it up.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
main_root = REPO
if "/.claude/worktrees/" in str(REPO):
    main_root = Path(str(REPO).split("/.claude/worktrees/")[0])

src = (REPO / "docs" / "PROPOSAL_v2.md").read_text()
assert "</script" not in src.lower()
out = main_root / "outputs" / "viewers" / "proposal_v2"
out.mkdir(parents=True, exist_ok=True)

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CTT - Project Proposal v2</title>
<style>
  body { margin: 0; background: #fafaf8; color: #1c1c1c;
         font: 17px/1.65 Georgia, 'Times New Roman', serif; }
  main { max-width: 46rem; margin: 0 auto; padding: 3rem 1.5rem 6rem; }
  h1, h2, h3 { font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.25; }
  h1 { font-size: 1.9rem; margin-top: 0; }
  h2 { font-size: 1.35rem; margin-top: 2.4em; border-bottom: 1px solid #ddd; padding-bottom: .25em; }
  h3 { font-size: 1.08rem; margin-top: 1.8em; }
  table { border-collapse: collapse; margin: 1.2em 0;
          font: 14px/1.5 'Helvetica Neue', Arial, sans-serif; display: block; overflow-x: auto; }
  th, td { border: 1px solid #ccc; padding: .4em .65em; text-align: left; vertical-align: top; }
  th { background: #f0efe9; }
  code { font: .88em/1.4 ui-monospace, Menlo, monospace; background: #eee;
         padding: .1em .3em; border-radius: 3px; }
  blockquote { margin: 1em 0; padding: .2em 1em; border-left: 3px solid #ccc; color: #444; }
  hr { border: none; border-top: 1px solid #ddd; margin: 2.5em 0; }
  #status { color: #999; font-family: sans-serif; }
</style>
<script>
window.MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] },
                   options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] } };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<main><div id="status">rendering&hellip;</div><div id="doc"></div></main>
<script type="text/plain" id="src">__MD__</script>
<script>
window.addEventListener('DOMContentLoaded', () => {
  const raw = document.getElementById('src').textContent;
  const store = [];
  const protectedTxt = raw.replace(/\\$\\$[\\s\\S]+?\\$\\$|\\$[^$\\n]+\\$/g,
    m => { store.push(m); return '@@M' + (store.length - 1) + '@@'; });
  let html = marked.parse(protectedTxt);
  html = html.replace(/@@M(\\d+)@@/g,
    (_, i) => store[+i].replace(/&/g, '&amp;').replace(/</g, '&lt;'));
  document.getElementById('doc').innerHTML = html;
  document.getElementById('status').remove();
  if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise();
});
</script>
</body>
</html>
"""

(out / "index.html").write_text(PAGE.replace("__MD__", src))
(out / "viewer.json").write_text(json.dumps({
    "title": "CTT - Project Proposal v2",
    "blurb": ("The August 2026 project proposal, rendered: task formalized as "
              "operator transfer T(V_S, V_E), test-tier grid, CTT v2 strata, v4 "
              "instrument, unified arm results, bottleneck branch next. Source: "
              "docs/PROPOSAL_v2.md (branch worktree-enchanted-whistling-riddle)."),
    "group": "reports",
    "featured": True,
}, indent=2))
print(f"wrote {out}/index.html and viewer.json")
