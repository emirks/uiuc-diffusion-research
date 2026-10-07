#!/usr/bin/env python3
"""Assemble a self-contained hosting bundle for the SEGUE human study.

Produces outputs/presentation/user_study_bundle/ with everything needed to run
the study on any host (Python >= 3.10 stdlib only, behind an HTTPS reverse
proxy). Media are copied as REAL files (not symlinks) so the folder is portable.

    /usr/bin/python3.12 scripts/user_study/make_bundle.py

Layout produced:
    user_study_bundle/
      site/            index.html, status.html, media/ (real .mp4/.jpg)
      serve.py         the study server (stdlib only, runs standalone with --site)
      pairs.json       frozen selection (server reads it next to itself)
      attention.json   attention-check pool
  examples.json    worked examples (optional) (if present)
      examples.json    worked examples shown before the first set (if present)
      README.md        how to run / export / analyse
      user_study_bundle.zip   a zip of this folder, inside it (excludes *.zip)
"""
import json
import os
import shutil
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIEWER_DIR = os.path.join(REPO, "outputs", "viewers", "user_study")
STUDY = os.path.join(REPO, "misc", "2026-09-22_user_study")
MEDIA_SRC = os.path.join(STUDY, "media")
MANIFEST = os.path.join(STUDY, "media_manifest.json")
BUNDLE = os.path.join(REPO, "outputs", "presentation", "user_study_bundle")
ZIP_NAME = "user_study_bundle.zip"

DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /app
COPY serve.py pairs.json ./
COPY attention.json examples.json ./
COPY site ./site
EXPOSE 8021
# state on the mounted volume (/data); STUDY_ADMIN_TOKEN and PROLIFIC_COMPLETION_CODE come from the environment
CMD ["python3", "-u", "serve.py", "--port", "8021", "--bind", "0.0.0.0", "--site", "site", "--state-dir", "/data"]
"""

FLY_TOML = """\
# Fly.io app for the SEGUE human study (created 2026-09-25). Deploy from this folder:
#   flyctl deploy --remote-only
app = "segue-study"
primary_region = "ord"

[build]

[http_service]
  internal_port = 8021
  force_https = true
  auto_stop_machines = "off"
  auto_start_machines = true
  min_machines_running = 1

[[vm]]
  size = "shared-cpu-1x"
  memory = "512mb"

[mounts]
  source = "study_data"
  destination = "/data"
"""

README = """\
# SEGUE human study — hosting bundle

A self-contained copy of the blind A/B forced-choice study. It serves the rater
page plus a small JSON API and records ratings to a local state directory.
No external dependencies: **Python >= 3.10, standard library only.**

## Layout

```
user_study_bundle/
  site/            index.html, status.html, media/ (real .mp4 / .jpg files)
  serve.py         the study server
  pairs.json       frozen pair selection (the server reads it beside itself)
  attention.json   attention-check pool
  README.md        this file
```

## Run

```bash
cd user_study_bundle
STUDY_ADMIN_TOKEN=choose-a-long-secret \\
  python3 serve.py --port 8080 --site site --state-dir state
```

- `--site site` serves `site/index.html`, `site/status.html`, `site/media/…`.
- `--state-dir state` is where `ratings.jsonl` and `sessions.json` are written
  (created on first rating). Back this directory up; it is the collected data.
- `pairs.json` / `attention.json` / `examples.json` are read from beside `serve.py` automatically.
- When `STUDY_ADMIN_TOKEN` is set, the organizer endpoints require it:
  `GET /api/status`, `GET /api/export`, and `GET /status.html` return **403**
  unless the request carries `?token=<value>` (or header `X-Admin-Token`).
  Open the status page as `…/status.html?token=<value>` — it passes the token
  through to its own API calls. Leave the variable unset to keep them open.

The server binds `127.0.0.1` by default. **Put it behind HTTPS** with any
reverse proxy (nginx, Caddy, Cloudflare Tunnel, …) forwarding to that port;
pass `--bind 0.0.0.0` only if the proxy runs on another host. The rater page
(`/`) needs no token; only the organizer views do.

## Raters

Send raters the site root, e.g. `https://your-host/`. Each session is 15 comparisons (attention checks are off by default; set
`ATTENTION_PER_SESSION` in `serve.py` to re-enable them). Progress is kept in the
rater's own browser, so they can stop and resume.

## Export the ratings

```bash
# live, over HTTP (organizer token required if set):
curl -H "X-Admin-Token: $STUDY_ADMIN_TOKEN" https://your-host/api/export > ratings.jsonl
# or just copy the state file off the host:
cp state/ratings.jsonl ratings.jsonl
```

Each line is one rating. Attention-check lines carry `"attention": true` and are
never counted in win rates.

## Analyse

Use `analyze.py` from the diffusion-research repo against the exported file and
this bundle's `pairs.json`:

```bash
python3 scripts/user_study/analyze.py --ratings ratings.jsonl --pairs pairs.json
# add --csv out.csv for the per-pair majority table (attention excluded)
# add --drop-failed-sessions to also see win rates with failed-check sessions removed
```
"""


def copy_manifest_media(src, dst):
    """Copy ONLY the media files referenced by the current manifest into dst.

    The media dir can hold stale files from an earlier selection; the manifest
    is the authority for what this study actually uses. Real bytes, no symlinks.
    """
    os.makedirs(dst, exist_ok=True)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    n = 0
    for h, e in manifest.items():
        ext = "jpg" if e["kind"] == "still" else "mp4"
        name = f"{h}.{ext}"
        sp = os.path.join(src, name)
        if not os.path.isfile(sp):
            raise SystemExit(f"manifest media missing on disk: {sp}")
        shutil.copyfile(sp, os.path.join(dst, name))  # follows symlink -> real bytes
        n += 1
    return n


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


def main():
    # fresh rebuild so no stale files (and no stale zip) survive
    if os.path.exists(BUNDLE):
        shutil.rmtree(BUNDLE)
    site = os.path.join(BUNDLE, "site")
    os.makedirs(site, exist_ok=True)

    shutil.copyfile(os.path.join(VIEWER_DIR, "index.html"), os.path.join(site, "index.html"))
    shutil.copyfile(os.path.join(VIEWER_DIR, "status.html"), os.path.join(site, "status.html"))
    if os.path.exists(os.path.join(VIEWER_DIR, "admin.html")):   # organizer browser (admin-gated; store arms unavailable off-cluster)
        shutil.copyfile(os.path.join(VIEWER_DIR, "admin.html"), os.path.join(site, "admin.html"))
    # container + Fly.io deployment files (fly deploy --remote-only from the bundle dir)
    with open(os.path.join(BUNDLE, "Dockerfile"), "w") as f:
        f.write(DOCKERFILE)
    with open(os.path.join(BUNDLE, "fly.toml"), "w") as f:
        f.write(FLY_TOML)
    with open(os.path.join(BUNDLE, ".dockerignore"), "w") as f:
        f.write("*.zip\nREADME.md\n")
    n_media = copy_manifest_media(MEDIA_SRC, os.path.join(site, "media"))

    shutil.copyfile(os.path.join(REPO, "scripts", "user_study", "serve.py"),
                    os.path.join(BUNDLE, "serve.py"))
    shutil.copyfile(os.path.join(STUDY, "pairs.json"), os.path.join(BUNDLE, "pairs.json"))
    attn = os.path.join(STUDY, "attention.json")
    if os.path.exists(attn):
        shutil.copyfile(attn, os.path.join(BUNDLE, "attention.json"))
    exm = os.path.join(STUDY, "examples.json")
    if os.path.exists(exm):
        shutil.copyfile(exm, os.path.join(BUNDLE, "examples.json"))
    with open(os.path.join(BUNDLE, "README.md"), "w") as f:
        f.write(README)

    # zip the folder INSIDE the folder, excluding any *.zip
    zip_path = os.path.join(BUNDLE, ZIP_NAME)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(BUNDLE):
            for fn in files:
                if fn.endswith(".zip"):
                    continue
                full = os.path.join(root, fn)
                arc = os.path.join("user_study_bundle", os.path.relpath(full, BUNDLE))
                z.write(full, arc)

    total = dir_size(BUNDLE)
    zbytes = os.path.getsize(zip_path)
    print(f"[bundle] site media files copied: {n_media}")
    print(f"[bundle] {BUNDLE}")
    print(f"[bundle] folder size = {total/1e6:.1f} MB (zip = {zbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
