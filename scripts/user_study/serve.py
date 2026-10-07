#!/usr/bin/env python3
"""Study server for the SEGUE human study (stdlib only, python3.12).

Serves the blind rater page (outputs/viewers/user_study/) plus a small JSON API
that hands out balanced sessions, records ratings, and reports organizer status.
The SEGUE side of each pair is decided server-side and never sent to the client,
so the page stays blind.

Protocol v5 (2026-09-25, DESIGN_v5.md "Human page mirror"): per pair two A/B
answers (transition, overall) plus three flags per video (no_effect, endpoint,
leak). The no-effect gate is applied per rating from that rater's own flags and
stored as segue_score; segue_wins keeps the raw A/B answers. v4 ratings (four A/B
questions) live in misc/2026-09-22_user_study/state/ and are never loaded here. Names/paths/state live under misc/2026-09-22_user_study/
and are never served (only opaque media/<hash>.{mp4,jpg} is).

    /usr/bin/python3.12 scripts/user_study/serve.py --port 8021
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- config ------------------------------------------------------------------
SESSION_SIZE = 15          # real pairs per session (30 until 2026-09-25; owner asked for 15)
TARGET_RATINGS = 3
ASSIGN_TTL_MIN = 30   # was 90; a set takes ~6 min, abandoned reservations must lapse within a Prolific batch (2026-09-25 load test)
TTL = ASSIGN_TTL_MIN * 60
ATTENTION_PER_SESSION = 0  # hidden attention items per session; 0 = attention checks OFF (owner, 2026-09-25 pm; was 2)
# attention items are scored on two flags: the COPY must carry it, the REAL must not
ATTENTION_SCORED = ("endpoint", "leak")
ATTENTION_PASS = 3         # >= this many of the 4 scored answers correct = pass
PROTOCOL = "v5"
QUESTIONS = ("transition", "overall")          # A/B, forced
FLAGS = ("no_effect", "endpoint", "leak")      # per video, unchecked by default

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths below are the cluster defaults; main() may reassign them from --site /
# --state-dir (and locate pairs.json / attention.json next to serve.py when the
# bundle runs standalone). They are module globals so the handlers see them.
VIEWER_DIR = os.path.join(REPO, "outputs", "viewers", "user_study")
MEDIA_DIR = os.path.join(VIEWER_DIR, "media")
STUDY = os.path.join(REPO, "misc", "2026-09-22_user_study")
PAIRS_FILE = os.path.join(STUDY, "pairs.json")
ATTENTION_FILE = os.path.join(STUDY, "attention.json")
EXAMPLES_FILE = os.path.join(STUDY, "examples.json")  # worked examples shown before the first set (optional)
STATE_DIR = os.path.join(STUDY, "state_v5")  # v5 ratings only; v4 test ratings stay in state/
RATINGS_FILE = os.path.join(STATE_DIR, "ratings.jsonl")
SESSIONS_FILE = os.path.join(STATE_DIR, "sessions.json")
ADMIN_TOKEN = None  # set from STUDY_ADMIN_TOKEN in main(); None = no gating
PROLIFIC_CODE = None  # set from PROLIFIC_COMPLETION_CODE in main(); returned by /api/complete after a full set

MEDIA_RE = re.compile(r"^[0-9a-f]{12}\.(mp4|jpg)$")
RNG = random.Random()  # unseeded: fair coins + random tie-breaks

# --- state (guarded by LOCK) -------------------------------------------------
LOCK = threading.Lock()
PAIRS = {}        # pair_id -> pair dict
STUDY_ID = ""     # sha1 of pairs.json + protocol (set in load_pairs)
ALL_PIDS = []     # list of pair_ids
ROW_OF = {}       # pair_id -> (task, endpoint, reference)
ATTN_POOL = []    # list of attention-pool entries (from attention.json)
EXAMPLES = []     # examples.json entries (served by /api/examples, never rated)
SESSIONS = {}     # session_id -> {..., items, attn, order, rated, attn_rated, pairset}
RATED_BY = {}     # pair_id -> set(rater_id) who rated it (real pairs only)
N_RATINGS = 0     # real ratings only (attention excluded)
ATTN_RECS = []    # list of attention rating records (for status/quality signal)
SKIPPED_OTHER_PROTOCOL = 0  # rating lines of another protocol found in the state dir (ignored)


def mh(path):
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def media_url(path, ext):
    return "media/%s.%s" % (mh(path), ext)


def now():
    return time.time()


def load_pairs():
    global PAIRS, ALL_PIDS, ROW_OF, ATTN_POOL
    with open(PAIRS_FILE) as f:
        rows = json.load(f)
    PAIRS = {r["pair_id"]: r for r in rows}
    ALL_PIDS = [r["pair_id"] for r in rows]
    global STUDY_ID  # identifies pair set + protocol; the page drops stored sessions from another one
    with open(PAIRS_FILE, "rb") as f:
        STUDY_ID = hashlib.sha1(f.read() + b"\nprotocol=" + PROTOCOL.encode()).hexdigest()[:12]
    ROW_OF = {r["pair_id"]: (r["task"], r["endpoint"], r["reference"]) for r in rows}
    ATTN_POOL = []
    if ATTENTION_PER_SESSION and os.path.exists(ATTENTION_FILE):
        with open(ATTENTION_FILE) as f:
            ATTN_POOL = json.load(f)
    global EXAMPLES
    EXAMPLES = []
    if os.path.exists(EXAMPLES_FILE):
        with open(EXAMPLES_FILE) as f:
            EXAMPLES = json.load(f)


def arm_of(path):
    m = re.search(r"store/gens/([^/]+/[^/]+)", path)
    return m.group(1) if m else os.path.basename(os.path.dirname(os.path.dirname(path)))


def pairs_payload():
    """Every pair, unblinded, for the organizer's admin page (never served to raters)."""
    out = []
    for pid in ALL_PIDS:
        pr = PAIRS[pid]
        out.append({
            "pair_id": pid, "task": pr["task"], "opponent": pr["opponent"], "cell": pr.get("cell"),
            "content": pr.get("content"), "endpoint": pr["endpoint"], "reference": pr["reference"],
            "segue_arm": arm_of(pr["segue_clip"]), "opponent_arm": arm_of(pr["opponent_clip"]),
            "ref": media_url(pr["reference_clip"], "mp4"),
            "start": media_url(pr["start_still"], "jpg"),
            "end": media_url(pr["end_still"], "jpg") if pr["end_still"] else None,
            "segue": media_url(pr["segue_clip"], "mp4"), "opp": media_url(pr["opponent_clip"], "mp4"),
        })
    return out


STORE_GENS = os.path.join(REPO, "store", "gens")
ARM_RE = re.compile(r"^\d{3}_[A-Za-z0-9_]+/\d{2}_[A-Za-z0-9_]+$")


def list_arms():
    """store/gens/NNN_<arm>/KK_<variant>__<machine> dirs that hold videos (organizer page)."""
    out = []
    if not os.path.isdir(STORE_GENS):
        return out
    for arm in sorted(os.listdir(STORE_GENS)):
        ad = os.path.join(STORE_GENS, arm)
        if not os.path.isdir(ad) or arm.startswith("_"):
            continue
        for var in sorted(os.listdir(ad)):
            vd = os.path.join(ad, var, "videos")
            if os.path.isdir(vd):
                n = sum(1 for f in os.listdir(vd) if f.endswith(".mp4"))
                if n:
                    out.append({"arm": arm + "/" + var, "clips": n})
    return out


def arm_clips(arm):
    """pair_id -> /store/<arm>/videos/<file> for every study row this variant generated (seed 42).
    File name contract: <cell>__<tag>__<endpoint>__ref_<reference>__s<seed>.mp4"""
    if not ARM_RE.match(arm):
        return None
    vd = os.path.join(STORE_GENS, arm, "videos")
    if not os.path.isdir(vd):
        return None
    idx = {}
    for f in os.listdir(vd):
        if not f.endswith(".mp4"):
            continue
        parts = f[:-4].split("__")
        if len(parts) == 5 and parts[3].startswith("ref_"):
            idx[(parts[0], parts[2], parts[3][4:], parts[4])] = f
    out = {}
    for pid in ALL_PIDS:
        pr = PAIRS[pid]
        f = idx.get((pr.get("cell"), pr["endpoint"], pr["reference"], "s%s" % pr.get("seed", 42)))
        if f:
            out[pid] = "/store/%s/videos/%s" % (arm, f)
    return out


def examples_payload():
    """examples.json -> what the page shows before the first set. Media names are
    hashed like everything else; `expected` (answers + flags) is optional and
    drives the highlighted answers / pre-ticked boxes on the example screen."""
    out = []
    for e in EXAMPLES:
        out.append({
            "id": e["id"], "task": e["task"],
            "title": e.get("title", ""), "note": e.get("note", ""),
            "ref": media_url(e["reference_clip"], "mp4"),
            "start": media_url(e["start_still"], "jpg"),
            "end": media_url(e["end_still"], "jpg") if e.get("end_still") else None,
            "a": media_url(e["a_clip"], "mp4"), "b": media_url(e["b_clip"], "mp4"),
            "expected": e.get("expected"),
            "focus": e.get("focus", []),
        })
    return out


def load_state():
    global SESSIONS, RATED_BY, N_RATINGS, ATTN_RECS, SKIPPED_OTHER_PROTOCOL
    SESSIONS, RATED_BY, N_RATINGS, ATTN_RECS = {}, {}, 0, []
    SKIPPED_OTHER_PROTOCOL = 0
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE) as f:
            for s in json.load(f).get("sessions", []):
                s["rated"] = set()
                s["attn_rated"] = set()
                s.setdefault("attn", [])
                s.setdefault("order", [])
                s["pairset"] = {it["pair_id"] for it in s["items"]}
                SESSIONS[s["session_id"]] = s
    if os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("protocol") != PROTOCOL:
                    SKIPPED_OTHER_PROTOCOL += 1
                    continue
                sid = r.get("session_id")
                if r.get("attention"):
                    ATTN_RECS.append(r)
                    if sid in SESSIONS:
                        SESSIONS[sid]["attn_rated"].add(r["pair_id"])
                    continue
                pid, rid = r["pair_id"], r["rater_id"]
                RATED_BY.setdefault(pid, set()).add(rid)
                if not r.get("revision"):
                    N_RATINGS += 1
                if sid in SESSIONS:
                    SESSIONS[sid]["rated"].add(pid)


def persist_sessions():
    os.makedirs(STATE_DIR, exist_ok=True)
    out = {"sessions": [
        {"session_id": s["session_id"], "rater_id": s["rater_id"],
         "nickname": s["nickname"], "created": s["created"], "items": s["items"],
         "attn": s.get("attn", []), "order": s.get("order", [])}
        for s in SESSIONS.values()]}
    tmp = SESSIONS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, SESSIONS_FILE)


def append_rating(rec):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(RATINGS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())   # every rating is on disk before the page hears "ok"


# --- assignment --------------------------------------------------------------
def active_for_pair(s, pid):
    return (now() - s["created"] < TTL) and (pid not in s["rated"])


def coverage(pid, requester):
    """distinct raters who rated it + other raters' active assignments of it."""
    c = len(RATED_BY.get(pid, ()))
    for s in SESSIONS.values():
        if s["rater_id"] == requester:
            continue
        if pid in s["pairset"] and active_for_pair(s, pid):
            c += 1
    return c


def parse_prolific(body):
    """{pid, study_id, session_id} from the page (Prolific URL params) or None."""
    pr = body.get("prolific")
    if not isinstance(pr, dict) or not pr.get("pid"):
        return None
    return {k: str(pr.get(k) or "")[:64] for k in ("pid", "study_id", "session_id")}


def assign_session(rater_id, nickname, prolific=None):
    rated_by_r = {pid for pid, rs in RATED_BY.items() if rater_id in rs}
    rows_assigned_r, pids_active_r = set(), set()
    for s in SESSIONS.values():
        if s["rater_id"] != rater_id:
            continue
        for it in s["items"]:
            pid = it["pair_id"]
            rows_assigned_r.add(ROW_OF[pid])
            if active_for_pair(s, pid):
                pids_active_r.add(pid)

    eligible = [p for p in ALL_PIDS if p not in rated_by_r and p not in pids_active_r]
    tier1 = [p for p in eligible if ROW_OF[p] not in rows_assigned_r]
    tier2 = [p for p in eligible if ROW_OF[p] in rows_assigned_r]

    picked, used_rows = [], set()

    def take(tier):
        for p in sorted(tier, key=lambda p: (coverage(p, rater_id), RNG.random())):
            if len(picked) >= SESSION_SIZE:
                break
            row = ROW_OF[p]
            if row in used_rows:
                continue
            picked.append(p)
            used_rows.add(row)

    take(tier1)
    if len(picked) < SESSION_SIZE:
        take(tier2)
    if not picked:
        return {"items": []}  # nothing left -> the page thanks the rater (no attention added)

    # real items: per-item SEGUE side (blinding), shuffled as before
    items = [{"pair_id": p, "segue_side": RNG.choice(["A", "B"])} for p in picked]
    RNG.shuffle(items)

    # attention items: draw ATTENTION_PER_SESSION at random from the pool, no
    # repeat within the session; REAL/COPY side flipped by a coin, stored only
    # server-side. A rater may see the same pool entry in a later session.
    k = min(ATTENTION_PER_SESSION, len(ATTN_POOL))
    attn = [{"a": e["a"], "real_side": RNG.choice(["A", "B"]), "entry": e}
            for e in RNG.sample(ATTN_POOL, k)] if k else []

    # interleave: real items keep their (already random) order; attention items
    # are inserted at random positions -> a (SESSION_SIZE + ATTENTION_PER_SESSION)-slot serving order.
    order = [{"t": "r", "id": it["pair_id"]} for it in items]
    for at in attn:
        order.insert(RNG.randint(0, len(order)), {"t": "a", "id": at["a"]})

    sid = uuid.uuid4().hex
    s = {"session_id": sid, "rater_id": rater_id, "nickname": nickname, "prolific": prolific,
         "created": now(), "items": items, "attn": attn, "order": order,
         "rated": set(), "attn_rated": set(),
         "pairset": {it["pair_id"] for it in items}}
    SESSIONS[sid] = s
    persist_sessions()

    real_by_id = {it["pair_id"]: it for it in items}
    attn_by_id = {at["a"]: at for at in attn}
    out = []
    for i, slot in enumerate(order):
        if slot["t"] == "r":
            it = real_by_id[slot["id"]]
            pr = PAIRS[it["pair_id"]]
            seg = media_url(pr["segue_clip"], "mp4")
            opp = media_url(pr["opponent_clip"], "mp4")
            a, b = (seg, opp) if it["segue_side"] == "A" else (opp, seg)
            out.append({
                "i": i, "pair_id": it["pair_id"], "task": pr["task"],
                "ref": media_url(pr["reference_clip"], "mp4"),
                "start": media_url(pr["start_still"], "jpg"),
                "end": media_url(pr["end_still"], "jpg") if pr["end_still"] else None,
                "a": a, "b": b,
            })
        else:  # attention item -- served identically to a real one
            at = attn_by_id[slot["id"]]
            e = at["entry"]
            real_u = media_url(e["real_clip"], "mp4")
            copy_u = media_url(e["copy_clip"], "mp4")
            a, b = (real_u, copy_u) if at["real_side"] == "A" else (copy_u, real_u)
            out.append({
                "i": i, "pair_id": at["a"], "task": e["task"],
                "ref": media_url(e["reference_clip"], "mp4"),
                "start": media_url(e["start_still"], "jpg"),
                "end": media_url(e["end_still"], "jpg") if e["end_still"] else None,
                "a": a, "b": b,
            })
    return {"session_id": sid, "study_id": STUDY_ID, "items": out}


def parse_flags(body):
    """{"A": {flag: bool}, "B": {flag: bool}} or None when malformed."""
    fl = body.get("flags")
    if not isinstance(fl, dict):
        return None
    out = {}
    for side in ("A", "B"):
        d = fl.get(side)
        if not isinstance(d, dict):
            return None
        out[side] = {f: bool(d.get(f, False)) for f in FLAGS}
    return out


def gate(ans, side, flags_segue, flags_opp):
    """DESIGN_v5 gate from this rater's own no-effect flags -> (gate, segue_score)."""
    se, oe = not flags_segue["no_effect"], not flags_opp["no_effect"]
    if se and oe:
        return "both_effect", {q: (1.0 if ans[q] == side else 0.0) for q in QUESTIONS}
    if se:
        return "segue_only_effect", {q: 1.0 for q in QUESTIONS}
    if oe:
        return "opponent_only_effect", {q: 0.0 for q in QUESTIONS}
    return "neither_effect", {q: 0.5 for q in QUESTIONS}


def record_rating(body):
    sid = body.get("session_id")
    pid = body.get("pair_id")
    ans = body.get("answers") or {}
    s = SESSIONS.get(sid)
    if not s:
        return 400, {"error": "unknown session"}
    flags = parse_flags(body)
    if flags is None:
        return 400, {"error": "flags required"}

    # attention item? (its pair_id lives in the a... space, in s["attn"])
    at = next((a for a in s.get("attn", []) if a["a"] == pid), None)
    if at is not None:
        if any(ans.get(q) not in ("A", "B") for q in QUESTIONS):
            return 400, {"error": "both answers required"}
        if pid in s["attn_rated"]:
            return 200, {"duplicate": True}
        real_side = at["real_side"]
        copy_side = "B" if real_side == "A" else "A"
        e = at["entry"]
        rec = {
            "ts": now(), "protocol": PROTOCOL, "rater_id": s["rater_id"], "nickname": s["nickname"], "prolific": s.get("prolific"),
            "session_id": sid, "pair_id": pid, "task": e["task"],
            "attention": True, "real_side": real_side,
            "answers": {q: ans[q] for q in QUESTIONS}, "flags": flags,
            # the COPY (= the reference clip) neither starts on the given frames nor
            # hides the reference's content: its endpoint and leak boxes should be
            # ticked and the REAL's left empty.
            "correct": {f: (flags[copy_side][f] and not flags[real_side][f])
                        for f in ATTENTION_SCORED},
            "seconds": body.get("seconds"), "replays": body.get("replays"),
        }
        append_rating(rec)
        s["attn_rated"].add(pid)
        ATTN_RECS.append(rec)
        return 200, {"ok": True}

    if pid not in s["pairset"]:
        return 400, {"error": "pair not in session"}
    if any(ans.get(q) not in ("A", "B") for q in QUESTIONS):
        return 400, {"error": "both answers required"}
    revision = pid in s["rated"]   # the rater went back with "Previous" and changed an answer: append, keep the last

    side = next(it["segue_side"] for it in s["items"] if it["pair_id"] == pid)
    opp_side = "B" if side == "A" else "A"
    pr = PAIRS[pid]
    g, score = gate(ans, side, flags[side], flags[opp_side])
    rec = {
        "ts": now(), "protocol": PROTOCOL, "rater_id": s["rater_id"], "nickname": s["nickname"], "prolific": s.get("prolific"),
        "session_id": sid, "pair_id": pid, "task": pr["task"],
        "opponent": pr["opponent"], "segue_side": side,
        "answers": {q: ans[q] for q in QUESTIONS}, "flags": flags,
        "flags_segue": flags[side], "flags_opponent": flags[opp_side],
        "segue_wins": {q: (ans[q] == side) for q in QUESTIONS},   # raw A/B, before the gate
        "gate": g, "segue_score": score,                          # after the gate (1 / 0.5 / 0)
        "seconds": body.get("seconds"), "replays": body.get("replays"),
        "revision": revision,
    }
    append_rating(rec)
    global N_RATINGS
    if revision:
        return 200, {"ok": True, "revised": True}
    RATED_BY.setdefault(pid, set()).add(s["rater_id"])
    s["rated"].add(pid)
    N_RATINGS += 1
    return 200, {"ok": True}


def status():
    def bucket(n):
        return "3+" if n >= 3 else str(n)
    cov = {"0": 0, "1": 0, "2": 0, "3+": 0}
    per_opp, per_task = {}, {}
    for pid in ALL_PIDS:
        n = len(RATED_BY.get(pid, ()))
        cov[bucket(n)] += 1
        pr = PAIRS[pid]
        for grp, key in ((per_opp, pr["opponent"]), (per_task, pr["task"])):
            d = grp.setdefault(key, {"pairs": 0, "ratings": 0, "pairs_at_target": 0})
            d["pairs"] += 1
            d["ratings"] += n
            if n >= TARGET_RATINGS:
                d["pairs_at_target"] += 1
    raters = set()
    for rs in RATED_BY.values():
        raters |= rs
    active = sum(1 for s in SESSIONS.values() if now() - s["created"] < TTL)
    return {
        "pairs": len(ALL_PIDS), "ratings": N_RATINGS, "raters": len(raters),
        "sessions_active": active, "coverage_hist": cov,
        "per_opponent": per_opp, "per_task": per_task,
        "attention": attention_status(), "protocol": PROTOCOL, "study_id": STUDY_ID,
        "session_size": SESSION_SIZE, "attention_per_session": ATTENTION_PER_SESSION,
    }


def attention_status():
    """Attention-check summary (never mixed into win rates).

    A session is FULLY CHECKED when it has answered all its assigned attention
    items (normally 2 -> 4 scored answers); it FAILS if fewer than
    ATTENTION_PASS of those scored answers are correct.
    """
    n = len(ATTN_RECS)
    ce = sum(1 for r in ATTN_RECS if r["correct"].get("endpoint"))
    cl = sum(1 for r in ATTN_RECS if r["correct"].get("leak"))
    by_sess = {}
    for r in ATTN_RECS:
        by_sess.setdefault(r["session_id"], []).append(r)
    failed = 0
    for sid, recs in by_sess.items():
        assigned = len(SESSIONS[sid]["attn"]) if sid in SESSIONS else len(recs)
        fully_checked = assigned >= ATTENTION_PER_SESSION and len(recs) >= assigned
        n_correct = sum(1 for r in recs for q in ATTENTION_SCORED if r["correct"].get(q))
        if fully_checked and n_correct < ATTENTION_PASS:
            failed += 1
    return {
        "ratings": n,
        "correct_rate_endpoint": (ce / n) if n else None,
        "correct_rate_leak": (cl / n) if n else None,
        "sessions_with_check": len(by_sess),
        "sessions_failed": failed,
    }


# --- HTTP --------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "study/1.5"

    def log_message(self, *a):
        pass  # quiet

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _admin_ok(self):
        """True unless STUDY_ADMIN_TOKEN is set and the request lacks it.

        Token accepted as ?token=<value> or header X-Admin-Token. When
        ADMIN_TOKEN is None (cluster use) organizer endpoints are open.
        """
        if ADMIN_TOKEN is None:
            return True
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        supplied = self.headers.get("X-Admin-Token") or (q.get("token") or [None])[0]
        return supplied == ADMIN_TOKEN

    def _send_file(self, path, ctype, cache=False):
        if not os.path.isfile(path):
            self._json(404, {"error": "not found"})
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.end_headers()
                    return
                partial = True
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(os.path.join(VIEWER_DIR, "index.html"), "text/html; charset=utf-8")
            return
        if path == "/status.html":
            if not self._admin_ok():
                self._json(403, {"error": "forbidden"})
                return
            self._send_file(os.path.join(VIEWER_DIR, "status.html"), "text/html; charset=utf-8")
            return
        if path.startswith("/media/"):
            name = path[len("/media/"):]
            if not MEDIA_RE.match(name):
                self._json(404, {"error": "not found"})
                return
            ctype = "video/mp4" if name.endswith(".mp4") else "image/jpeg"
            self._send_file(os.path.join(MEDIA_DIR, name), ctype, cache=True)
            return
        if path == "/admin.html":   # organizer browser: every pair, unblinded (admin-gated like status)
            if not self._admin_ok():
                self._json(403, {"error": "forbidden"})
                return
            self._send_file(os.path.join(VIEWER_DIR, "admin.html"), "text/html; charset=utf-8")
            return
        if path == "/api/arms":
            if not self._admin_ok():
                self._json(403, {"error": "forbidden"})
                return
            self._json(200, {"arms": list_arms()})
            return
        if path == "/api/arm_clips":
            if not self._admin_ok():
                self._json(403, {"error": "forbidden"})
                return
            qs = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            arm = (qs.get("arm") or [""])[0]
            clips = arm_clips(arm)
            if clips is None:
                self._json(404, {"error": "unknown arm"})
                return
            self._json(200, {"arm": arm, "clips": clips})
            return
        if path.startswith("/store/"):   # store clips for the organizer page (admin-gated, mp4 under store/gens only)
            if not self._admin_ok():
                self._json(403, {"error": "forbidden"})
                return
            rel = urllib.parse.unquote(path[len("/store/"):])
            full = os.path.realpath(os.path.join(STORE_GENS, rel))
            if ".." in rel.split("/") or not full.startswith(os.path.realpath(STORE_GENS) + os.sep) \
                    or not full.endswith(".mp4") or not os.path.isfile(full):
                self._json(404, {"error": "not found"})
                return
            self._send_file(full, "video/mp4", cache=True)
            return
        if path == "/api/pairs":
            if not self._admin_ok():
                self._json(403, {"error": "forbidden"})
                return
            self._json(200, {"study_id": STUDY_ID, "pairs": pairs_payload()})
            return
        if path == "/api/ping":
            self._json(200, {"study_id": STUDY_ID, "protocol": PROTOCOL, "pairs": len(PAIRS),
                             "session_size": SESSION_SIZE, "attention_per_session": ATTENTION_PER_SESSION,
                             "examples": len(EXAMPLES)})
            return
        if path == "/api/examples":
            self._json(200, {"study_id": STUDY_ID, "examples": examples_payload()})
            return
        if path == "/api/status":
            if not self._admin_ok():
                self._json(403, {"error": "forbidden"})
                return
            with LOCK:
                self._json(200, status())
            return
        if path == "/api/export":
            if not self._admin_ok():
                self._json(403, {"error": "forbidden"})
                return
            data = ""
            if os.path.exists(RATINGS_FILE):
                with open(RATINGS_FILE) as f:
                    data = f.read()
            body = data.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._json(404, {"error": "not found"})

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_body()
        except Exception:
            self._json(400, {"error": "bad json"})
            return
        if path == "/api/session":
            rater = body.get("rater_id")
            if not rater:
                self._json(400, {"error": "rater_id required"})
                return
            nick = (body.get("nickname") or "").strip()[:64]
            with LOCK:
                self._json(200, assign_session(rater, nick, parse_prolific(body)))
            return
        if path == "/api/complete":   # after the last item: confirms the set is fully recorded, hands out the code
            sid = body.get("session_id")
            with LOCK:
                s = SESSIONS.get(sid)
                if not s:
                    self._json(404, {"error": "unknown session"})
                    return
                done = len(s["rated"]) >= len(s["items"]) and len(s["attn_rated"]) >= len(s["attn"])
                self._json(200, {"complete": done, "rated": len(s["rated"]), "items": len(s["items"]),
                                 "code": PROLIFIC_CODE if done else None})
            return
        if path == "/api/rating":
            with LOCK:
                code, obj = record_rating(body)
            self._json(code, obj)
            return
        self._json(404, {"error": "not found"})


def main():
    global VIEWER_DIR, MEDIA_DIR, STATE_DIR, RATINGS_FILE, SESSIONS_FILE
    global PAIRS_FILE, ATTENTION_FILE, EXAMPLES_FILE, ADMIN_TOKEN, PROLIFIC_CODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--site", default=VIEWER_DIR,
                    help="viewer dir to serve (index.html, status.html, media/); "
                         "default = the repo viewer dir. Lets the hosting bundle run standalone.")
    ap.add_argument("--state-dir", default=STATE_DIR,
                    help="where ratings.jsonl / sessions.json live; default = the repo state dir.")
    ap.add_argument("--pairs", default=None,
                    help="pairs.json; default = next to serve.py if present (bundle), else the repo copy.")
    ap.add_argument("--attention", default=None,
                    help="attention.json; default = next to serve.py if present (bundle), else the repo copy.")
    ap.add_argument("--examples", default=None,
                    help="examples.json (worked examples before the first set); default = next to serve.py "
                         "if present (bundle), else the repo copy; missing file = no examples screen.")
    args = ap.parse_args()

    VIEWER_DIR = os.path.abspath(args.site)
    MEDIA_DIR = os.path.join(VIEWER_DIR, "media")
    STATE_DIR = os.path.abspath(args.state_dir)
    RATINGS_FILE = os.path.join(STATE_DIR, "ratings.jsonl")
    SESSIONS_FILE = os.path.join(STATE_DIR, "sessions.json")
    # pairs.json / attention.json: explicit flag > next to serve.py (bundle) > repo default
    PAIRS_FILE = (args.pairs or (os.path.join(SCRIPT_DIR, "pairs.json")
                                 if os.path.exists(os.path.join(SCRIPT_DIR, "pairs.json"))
                                 else PAIRS_FILE))
    ATTENTION_FILE = (args.attention or (os.path.join(SCRIPT_DIR, "attention.json")
                                         if os.path.exists(os.path.join(SCRIPT_DIR, "attention.json"))
                                         else ATTENTION_FILE))
    EXAMPLES_FILE = (args.examples or (os.path.join(SCRIPT_DIR, "examples.json")
                                       if os.path.exists(os.path.join(SCRIPT_DIR, "examples.json"))
                                       else EXAMPLES_FILE))
    ADMIN_TOKEN = os.environ.get("STUDY_ADMIN_TOKEN") or None
    PROLIFIC_CODE = os.environ.get("PROLIFIC_COMPLETION_CODE") or None

    load_pairs()
    load_state()
    print("[serve] protocol %s, study_id %s, state %s" % (PROTOCOL, STUDY_ID, STATE_DIR))
    print("[serve] %d pairs, %d attention items, %d examples, session = %d + %d, %d ratings loaded%s"
          % (len(ALL_PIDS), len(ATTN_POOL), len(EXAMPLES), SESSION_SIZE, ATTENTION_PER_SESSION, N_RATINGS,
             ("  [admin token ON]" if ADMIN_TOKEN else "") + ("  [prolific code ON]" if PROLIFIC_CODE else "")))
    if SKIPPED_OTHER_PROTOCOL:
        print("[serve] WARNING: ignored %d rating line(s) of another protocol in %s"
              % (SKIPPED_OTHER_PROTOCOL, RATINGS_FILE))
    httpd = ThreadingHTTPServer((args.bind, args.port), Handler)
    print("[serve] http://%s:%d/  (Ctrl-C to stop)" % (args.bind, args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
