"""ctt_v2 dataset viewer — zero-dependency server.

Streams videos straight out of the refVFX WebDataset tars (byte-range seeks into the
shard files — nothing is ever extracted) and the VFXMaster extracted tree, behind a
single-page UI with axis-based browsing:

    code  (refVFX code_based_edits, 136,800): effect / spatial family / temporal family /
                                              content (same base video under many effects) / browse
    lora  (refVFX I2V_LoRA, 6,995):           effect / content / browse
    vfx   (VFXMaster, ~9.9k):                 class / browse

Run (login node is fine — pure I/O):
    nice -n 19 python scripts/ctt_v2/dataset_viewer/serve.py --port 8799
Then from your machine:
    ssh -L 8799:cc-login3:8799 cc          # or whichever login node it runs on
    open http://localhost:8799
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
RAW = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/raw")
IDX = RAW / "refvfx/_viewer_index"
SHARD_DIR = {"code": RAW / "refvfx/data/code_based_edits", "lora": RAW / "refvfx/data/I2V_LoRA"}
VFX_ROOT = RAW / "vfxmaster/extracted/data"
# VFXMaster ships MPEG-4 Part 2, which browsers cannot decode; a transcode job writes an
# H.264 twin tree. Serve it when present, fall back to the original otherwise.
VFX_H264 = RAW / "vfxmaster/extracted_h264/data"
MIME = {"mp4": "video/mp4", "png": "image/png"}

DB: dict[str, list[dict]] = {}
GROUPS: dict[str, dict[str, dict[str, list[int]]]] = {}  # subset -> axis -> label -> row ids

AXES = {"code": ["effect", "spatial", "temporal", "content", "browse"],
        "lora": ["effect", "content", "browse"],
        "vfx": ["class", "browse"]}
AXIS_KEY = {"effect": "et", "spatial": "sfam", "temporal": "tfam", "content": "fp", "class": "cls"}


def load() -> None:
    for subset in ("code", "lora", "vfx"):
        p = IDX / f"{subset}.jsonl.gz"
        if not p.exists():
            print(f"[serve] WARNING: {p} missing — run build_index.py; subset disabled")
            DB[subset] = []
            GROUPS[subset] = {}
            continue
        rows = [json.loads(line) for line in gzip.open(p, "rt")]
        DB[subset] = rows
        g: dict[str, dict[str, list[int]]] = {}
        for axis in AXES[subset]:
            if axis == "browse":
                continue
            key = AXIS_KEY[axis]
            d: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(rows):
                v = r.get(key)
                if v:
                    d[v].append(i)
            if axis == "content":  # only groups that realise the counterfactual (>1 sample)
                d = {k: v for k, v in d.items() if len(v) > 1}
            g[axis] = dict(d)
        GROUPS[subset] = g
        print(f"[serve] {subset}: {len(rows)} samples, "
              + ", ".join(f"{a}={len(g.get(a, {}))}" for a in AXES[subset] if a != "browse"))


def row_payload(subset: str, i: int) -> dict:
    r = DB[subset][i]
    media = {}
    if subset == "vfx":
        media["out"] = f"/media/vfx/{i}/out"
    else:
        for role, spec in r["m"].items():
            media[role] = f"/media/{subset}/{i}/{role}"
    out = {"id": i, "media": media,
           "mime": {role: MIME.get(spec[2], "video/mp4") for role, spec in r.get("m", {}).items()}
           if subset != "vfx" else {"out": "video/mp4"}}
    if subset == "vfx":
        out |= {"cls": r["cls"], "pr": r["cap"], "et": r["cls"],
                "links": {"class": len(GROUPS["vfx"]["class"].get(r["cls"], []))}}
    else:
        out |= {"et": r.get("et"), "pr": r.get("pr"), "mt": r.get("mt"), "ori": r.get("ori"),
                "fp": r.get("fp"), "sfam": r.get("sfam"), "tfam": r.get("tfam"),
                "links": {"effect": len(GROUPS[subset].get("effect", {}).get(r.get("et"), [])),
                          "content": len(GROUPS[subset].get("content", {}).get(r.get("fp"), []))}}
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj) -> None:
        raw = json.dumps(obj).encode()
        gz = "gzip" in (self.headers.get("Accept-Encoding") or "")
        body = gzip.compress(raw, 5) if gz else raw
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                body = (HERE / "viewer.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/meta":
                self._meta()
            elif u.path == "/api/samples":
                self._samples(q)
            elif u.path.startswith("/media/"):
                self._media(u.path)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            try:
                self.send_error(500, str(e)[:200])
            except Exception:  # noqa: BLE001
                pass

    def _meta(self) -> None:
        meta = {}
        for subset in ("code", "lora", "vfx"):
            axes = {}
            for axis in AXES[subset]:
                if axis == "browse":
                    continue
                groups = sorted(((k, len(v)) for k, v in GROUPS[subset].get(axis, {}).items()),
                                key=lambda kv: (-kv[1], kv[0]))
                axes[axis] = groups[:4000]
            meta[subset] = {"n": len(DB[subset]), "axes": axes, "axis_order": AXES[subset]}
        self._json(meta)

    def _samples(self, q: dict) -> None:
        subset = q.get("subset", "code")
        axis = q.get("axis", "browse")
        page, per = int(q.get("page", 0)), min(int(q.get("per", 12)), 48)
        if axis == "browse":
            ids = list(range(len(DB[subset])))
            if q.get("shuffle") == "1":
                random.Random(int(q.get("seed", 0))).shuffle(ids)
        else:
            ids = GROUPS[subset].get(axis, {}).get(q.get("group", ""), [])
        total = len(ids)
        ids = ids[page * per:(page + 1) * per]
        self._json({"total": total, "rows": [row_payload(subset, i) for i in ids]})

    def _media(self, path: str) -> None:
        m = re.fullmatch(r"/media/(code|lora|vfx)/(\d+)/(in|out|mask)", path)
        if not m:
            self.send_error(404)
            return
        subset, i, role = m.group(1), int(m.group(2)), m.group(3)
        if i >= len(DB[subset]):
            self.send_error(404)
            return
        r = DB[subset][i]
        if subset == "vfx":
            p = VFX_H264 / r["path"]
            if not p.exists():
                p = VFX_ROOT / r["path"]
            f = open(p, "rb")
            f.seek(0, io.SEEK_END)
            size, base, mime = f.tell(), 0, "video/mp4"
        else:
            spec = r["m"].get(role)
            if not spec:
                self.send_error(404)
                return
            base, size, ext = spec[0], spec[1], spec[2]
            mime = MIME.get(ext, "video/mp4")
            f = open(sorted(SHARD_DIR[subset].glob("shard-*.tar"))[r["sh"]], "rb")

        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and (mm := re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())):
            if mm.group(1):
                start = int(mm.group(1))
                if mm.group(2):
                    end = min(int(mm.group(2)), size - 1)
            elif mm.group(2):
                start = max(0, size - int(mm.group(2)))
        if start > end or start >= size:
            self.send_error(416)
            f.close()
            return
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        f.seek(base + start)
        left = length
        while left > 0:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                break
            self.wfile.write(chunk)
            left -= len(chunk)
        f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    load()
    srv = ThreadingHTTPServer((args.bind, args.port), H)
    print(f"[serve] http://{args.bind}:{args.port}  (tunnel: ssh -L {args.port}:<this-host>:{args.port} cc)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
