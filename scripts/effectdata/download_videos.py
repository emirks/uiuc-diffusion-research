#!/usr/bin/env python3
"""Robust parallel downloader for EffectData Videos/*.zip.

Plain HTTPS resolve URLs (no xet -> no Lustre lock hang). Per file:
  - skip if final file already present with correct size
  - stream to <name>.part with HTTP Range resume; retry on failure
  - verify size, then sha256 (streamed free on fresh downloads; re-hashed on resume)
  - atomic os.replace into place
Idempotent: safe to re-run / resume after interruption. Progress -> download.log
(kept beside this script, not in the data dir). Downloads into data/raw/effectdata/Videos/.

    python scripts/effectdata/download_videos.py          # full run
    SMOKE=3 python scripts/effectdata/download_videos.py  # smallest 3 files only
"""
import json, os, sys, time, hashlib, threading, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO_ID = "ysy31415926/EffectData"
BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
HERE = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "effectdata"
OUTDIR = str(DATA / "Videos")
MANIFEST = DATA / "videos_manifest.json"
LOG = str(HERE / "download.log")
WORKERS = 8
CHUNK = 1 << 20          # 1 MiB
MAX_TRIES = 5

os.makedirs(OUTDIR, exist_ok=True)
man = json.load(open(MANIFEST))
FILES = man["files"]
_smoke = int(os.environ.get("SMOKE", "0"))
if _smoke:
    FILES = sorted(man["files"], key=lambda x: x["size"])[:_smoke]  # smallest N
TOTAL_BYTES = sum(x["size"] for x in FILES)

lock = threading.Lock()        # guards `state`
_loglock = threading.Lock()    # guards log writes (separate -> no re-entrancy deadlock)
state = {"done": 0, "ok": 0, "fail": 0, "skip": 0, "bytes": 0, "failed": []}
t0 = time.time()

def logline(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _loglock:
        with open(LOG, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

def session():
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1.5,
              status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r, pool_connections=WORKERS,
                                    pool_maxsize=WORKERS))
    return s

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()

def fetch_one(rec, s):
    path = rec["path"]                     # "Videos/<name>.zip"
    name = os.path.basename(path)
    final = os.path.join(OUTDIR, name)
    part = final + ".part"
    exp = rec["size"]; sha = rec["sha256"]
    url = BASE + urllib.parse.quote(path, safe="/")

    if os.path.exists(final) and os.path.getsize(final) == exp:
        return ("skip", name, exp)

    for attempt in range(1, MAX_TRIES + 1):
        try:
            start = os.path.getsize(part) if os.path.exists(part) else 0
            if start > exp:
                os.remove(part); start = 0
            headers = {"Range": f"bytes={start}-"} if start else {}
            h = hashlib.sha256() if start == 0 else None
            with s.get(url, headers=headers, stream=True, timeout=(30, 120)) as resp:
                if start and resp.status_code == 200:   # server ignored Range
                    start = 0; h = hashlib.sha256()
                else:
                    resp.raise_for_status()
                mode = "ab" if start else "wb"
                with open(part, mode) as f:
                    for chunk in resp.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                            if h is not None:
                                h.update(chunk)
            got = os.path.getsize(part)
            if got != exp:
                raise IOError(f"size {got} != expected {exp}")
            digest = h.hexdigest() if h is not None else sha256_file(part)
            if sha and digest != sha:
                os.remove(part)
                raise IOError("sha256 mismatch")
            os.replace(part, final)
            return ("ok", name, exp)
        except Exception as e:
            if attempt == MAX_TRIES:
                return ("fail", f"{name}: {e}", 0)
            time.sleep(2 * attempt)
    return ("fail", name, 0)

tl = threading.local()
def init_thread():
    tl.session = session()

def worker(rec):
    status, info, nbytes = fetch_one(rec, tl.session)
    with lock:
        state["done"] += 1
        state[status] += 1
        if status in ("ok", "skip"):
            state["bytes"] += nbytes
        if status == "fail":
            state["failed"].append(info)
        d = state["done"]; n = len(FILES)
        if status == "fail" or d % 25 == 0 or d == n:
            el = time.time() - t0
            gb = state["bytes"] / 1e9
            rate = gb / el * 60 if el else 0            # GB/min
            eta = (TOTAL_BYTES/1e9 - gb) / rate if rate else 0  # minutes
            logline(f"{d}/{n}  ok={state['ok']} skip={state['skip']} "
                    f"fail={state['fail']}  {gb:.0f}/{TOTAL_BYTES/1e9:.0f} GB  "
                    f"{rate:.1f} GB/min  ETA {eta:.0f} min")
    return status

def main():
    logline(f"START {len(FILES)} files, {TOTAL_BYTES/1e9:.0f} GB, {WORKERS} workers -> {OUTDIR}")
    with ThreadPoolExecutor(max_workers=WORKERS, initializer=init_thread) as ex:
        list(ex.map(worker, FILES))
    el = (time.time() - t0) / 60
    logline(f"DONE ok={state['ok']} skip={state['skip']} fail={state['fail']} "
            f"in {el:.1f} min ({state['bytes']/1e9:.0f} GB)")
    if state["failed"]:
        logline("FAILED: " + "; ".join(state["failed"][:20]))
        json.dump(state["failed"], open(HERE / "download_failed.json", "w"), indent=1)
        sys.exit(1)

if __name__ == "__main__":
    main()
