"""SEGUE VLM judge v2 — A/B pairwise + absolute scoring (Gemini 2.5 Pro).

Two judges over the same client / cache / retry / probe layer:

  JudgeAB.judge_pair(task, start, end|None, reference, out_a, out_b, meta) -> dict
      one forced A/B comparison of OUTPUT A vs OUTPUT B on four questions
      (transition / given_frames / reference_content / defects), evidence
      BEFORE choice, plus a margin (none|slight|clear). No tie option. The
      runner calls this twice per unit with SEGUE as A then as B; the decision
      rule (decide_question) combines the two orders.

  JudgeScore.judge(task, start, end|None, reference, output, meta) -> dict
      one absolute 0-4 score of a single OUTPUT on five items
      (occurrence / transition / given_frames / reference_content / defects),
      evidence BEFORE score. transition_adj = 0 if occurrence == 0.

Contract (misc/2026-09-22_vlm_judge_ab/SPEC.md; unchanged from the graded judge
judge_ctt.py, whose helpers this module IMPORTS and never modifies): pinned
model string, temperature 0, seed 0, thinking_budget 2048, sampling fps PINNED
to 8 via VideoMetadata, endpoints sent as JPEG q95 stills, response constrained
by a JSON schema with property_ordering so evidence precedes the choice/score,
max_output_tokens 3072 (thinking counts against it; 2048 truncates the JSON),
and every call's raw response + full provenance cached to disk before parsing
so reruns are free and auditable. No torch. The REFERENCE is the only
specification of the effect: nothing in any prompt names a system, an arm, an
effect class, or the prompt a system received.

The Gemini client is constructed LAZILY, only on the first real (non-cached)
call. Offline tests inject a mock via the `client=` constructor argument.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import time

# --- shared helpers from the certified graded judge (do NOT modify it) --------
# Import by name if it is already importable; otherwise load the sibling file by
# path (the runner and tests load this module by path, so the package __init__
# — which pulls torch on a GPU-less login node — is never touched). Importing
# judge_ctt runs cv2.setNumThreads(1) at its top, which is exactly what we want.
try:
    from judge_ctt import (  # type: ignore
        sha256, probe, still_jpeg_bytes, _usage_dict, JudgeCTT as _JudgeCTT)
except Exception:  # pragma: no cover - exercised only when not on sys.path
    _JC_PATH = pathlib.Path(__file__).resolve().parent / "judge_ctt.py"
    _spec = importlib.util.spec_from_file_location("judge_ctt", _JC_PATH)
    _jc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_jc)
    sha256, probe, still_jpeg_bytes = _jc.sha256, _jc.probe, _jc.still_jpeg_bytes
    _usage_dict, _JudgeCTT = _jc._usage_dict, _jc.JudgeCTT

# _fs_safe is a staticmethod on JudgeCTT (path-piece sanitiser: '/' '\\' -> '-')
_fs_safe = _JudgeCTT._fs_safe

MODEL = "gemini-2.5-pro"   # pinned; do not float to previews silently
DEFAULT_FPS = 8.0          # sampling rate requested for every video (VideoMetadata)

_RETRYABLE = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")

AB_QUESTIONS = ("transition", "given_frames", "reference_content", "defects")
# Rubric version, recorded in every cached call. v1 = SPEC §3.1 as approved
# 2026-09-23 (pilot 1: A-preference 58/73/63 % on the three real questions).
# v2 adds ONE position-bias control paragraph (owner, 2026-09-23); criteria unchanged.
# FROZEN = v1 (2026-09-23 01:xx). v2 (+bias paragraph) and v3 (placeholder format
# example) were each tried once on the same 45 pairs under pre-registered gates on the
# A-share and both failed (given_frames A-share 73/73/67 %, defects 63/68/60 %); their
# outputs are kept as results_ab_v{2,3}_pilot.jsonl. No further rubric edits.
# v4 (owner, 2026-09-23 ~02:00): criterion DEFINITIONS revised after reading the v1
# evidence — endpoint = the given frames ONLY (as Table 4's endpoint metrics); quality =
# visual failures only, task-aware, a null output (still/cut/crossfade) cannot win.
# Applied to the human page helpers at the same time (no human rating existed yet).
AB_RUBRIC_VERSION = "v4"
SCORE_DIMS = ("occurrence", "transition", "given_frames", "reference_content", "defects")

_TASK_LINE = {
    "two": "Task: TWO ENDPOINTS GIVEN",
    "one": "Task: START ENDPOINT GIVEN (open end)",
}

# --- task-variant clause substitutions (SPEC §3.1 `{two-sided: ...}` clauses) -
# Resolved by str.replace so the JSON braces in the rubric need no escaping.
_AB_SUBS = {
    "two": {
        "{p_end_still}": "the GIVEN END frame as a still image, ",
        "{p_close_end}": " and close on the GIVEN END,",
        "{p_before_end}": " and right before the end,",
        "{p_last_end}": " and its last frame(s) with the GIVEN END",
    },
    "one": {"{p_end_still}": "", "{p_close_end}": "", "{p_before_end}": "", "{p_last_end}": ""},
}
_SCORE_SUBS = {
    "two": {
        "{p_end_still}": "the GIVEN END frame as a still image, ",
        "{p_close_end}": " and close on the GIVEN END,",
        "{p_before_end}": " and right before the end,",
        "{p_end_ref}": " and closes on the END",
        "{p_last_end}": " and its last frame(s) with the GIVEN END",
    },
    "one": {"{p_end_still}": "", "{p_close_end}": "", "{p_before_end}": "",
            "{p_end_ref}": "", "{p_last_end}": ""},
}

# --- A/B rubric (system instruction) — SPEC §3.1, verbatim -------------------
_AB_RUBRIC = """You are an expert judge of video transition effects. You receive the GIVEN START frame as a still image,
{p_end_still}a REFERENCE video, and two candidate videos, OUTPUT A and OUTPUT B. The REFERENCE shows a transition effect between its own two scenes. Both outputs were asked to reproduce that same effect on the given frame(s), which show different scenes from the reference's. No text describes the effect: the REFERENCE is the only specification.

The outputs may come from different systems. All videos are time-normalized to a similar duration and sampled at 8 frames per second. Native frame rates and lengths differ, so held or repeated frames, choppier motion or a shorter clip are expected and must never decide a question.

Answer four questions. Answer each on its own: a video may win one question and lose another. For each, first write one sentence of evidence citing timestamps (for example 0:01-0:02), then choose A or B, the better of the two even when the difference is small, then state the margin: "none" if you see no real difference, "slight", or "clear". Judge what is visible, not what is plausible.

1. TRANSITION. Which output better reproduces the REFERENCE's effect? First check that an effect occurs at all: a plain cut, crossfade, dissolve, fade through black or white, blur or blend between the frames, or a static image is not an effect, and an output without an effect loses this question. Then compare the effect itself: the same mechanism and material (smoke, fire, liquid, particles, a portal, a morph, a shatter...), entering, spreading and clearing the same way, with similar timing. The scenes are supposed to differ: different people, objects or backgrounds are not errors.

2. GIVEN FRAMES. Which output better matches the given frame(s)? Compare ONLY the output's first frame(s) with the GIVEN START{p_last_end}: same scene, subject, layout and identity. Ignore everything in between, including changes the effect causes.

3. REFERENCE CONTENT. Which output shows less content that belongs to the REFERENCE's own scenes: its people, objects, backgrounds, the look of its scenes? Material that is part of the effect (the smoke, the fire, the particles) is not reference content. If neither output shows any, choose either and set the margin to "none".

4. DEFECTS. The task is transition effect transfer: a successful output changes a great deal, because it performs the REFERENCE's effect on the given scenes. Nothing that belongs to a successful transfer is a defect: the scene changing, the effect's own material, a subject transformed by the effect. Which output has fewer VISUAL FAILURES a viewer would notice on one viewing: distortion, smearing, broken or duplicated anatomy, tiling, corruption, flicker or strobing, frozen frames the effect does not explain? An output that stays still, cuts, or merely crossfades has not attempted the task and cannot win this question. Ignore low frame rate, held frames, clip length and codec softness.

Return only a JSON object: {"transition": {"evidence": "...", "choice": "A", "margin": "clear"}, "given_frames": {...}, "reference_content": {...}, "defects": {...}}."""

# --- Scoring rubric (system instruction) — SPEC §4 --------------------------
# The SAME four criteria as §3.1 in single-output phrasing, on the certified
# judge's 0-4 graded scale (evidence first), plus the OCCURRENCE item whose
# scale SPEC §4 states verbatim. The four graded scales mirror the certified
# judge (judge_ctt §3): transition<-fidelity, given_frames<-endpoint,
# reference_content<-disentanglement, defects<-quality.
_SCORE_RUBRIC = """You are an expert judge of video transition effects. You receive the GIVEN START frame as a still image,
{p_end_still}a REFERENCE video, and one OUTPUT video. The REFERENCE shows a transition effect between its own two scenes. The OUTPUT was asked to reproduce that same effect on the given frame(s), which show different scenes from the reference's. No text describes the effect: the REFERENCE is the only specification.

All videos are time-normalized to a similar duration and sampled at 8 frames per second. Native frame rates and lengths differ, so held or repeated frames, choppier motion or a shorter clip are expected and must never be penalized.

Score five items, each an integer from 0 to 4. For each, first write one sentence of evidence citing timestamps (for example 0:01-0:02), then give the score. Judge what is visible, not what is plausible.

1. OCCURRENCE. Does the OUTPUT contain a transition effect at all, beyond a plain cut, crossfade, dissolve, fade through black or white, blur or blend between the frames, or a static image? 0 = no effect, or only a cut, dissolve, fade, blur or blend. 1 = faint traces of an effect. 2 = an effect that is weak or partial. 3 = a clear effect. 4 = a clear, complete effect that runs its course: onset, development, resolution.

2. TRANSITION. Given that an effect occurs, is it the REFERENCE's effect? The same mechanism and material (smoke, fire, liquid, particles, a portal, a morph, a shatter...), entering, spreading and clearing the same way, with similar timing. Judge the effect's properties only; the scenes are supposed to differ, so different people, objects or backgrounds are not errors. 0 = a different effect, or no effect. 1 = only a loose resemblance. 2 = the same family but a different mechanism, material or motion. 3 = the same mechanism with minor differences in look or timing. 4 = the same effect, convincingly. If OCCURRENCE is 0, TRANSITION is 0.

3. GIVEN FRAMES. Does the OUTPUT match the given frame(s)? Compare ONLY the output's first frame(s) with the GIVEN START{p_last_end}: same scene, subject, layout and identity. Ignore everything in between, including changes the effect causes. 0 = the given frame(s) are not shown. 1 = a different scene or subject. 2 = the same scene with a wrong layout or identity. 3 = a close match with small differences. 4 = the same frame(s).

4. REFERENCE CONTENT. Does content that belongs to the REFERENCE's own scenes (its people, objects, backgrounds, the look of its scenes) appear in the OUTPUT? Material that is part of the effect (the smoke, the fire, the particles) is not reference content. 4 = none of the reference's scene content appears. 3 = a faint trace, such as a color cast or a vague shape. 2 = a recognizable element of a reference scene appears briefly. 1 = reference scene content appears prominently. 0 = the OUTPUT shows the reference's scenes instead of the given frame(s).

5. DEFECTS. The task is transition effect transfer: a successful output changes a great deal. Nothing that belongs to a successful transfer is a defect: the scene changing, the effect's own material, a subject transformed by the effect. Visual FAILURES a viewer would notice on one viewing: distortion, smearing, broken or duplicated anatomy, tiling, corruption, flicker or strobing, frozen frames the effect does not explain. 4 = clean. 3 = minor. 2 = noticeable. 1 = distracting. 0 = severe, or the output stays still, cuts, or merely crossfades (it has not attempted the task). Ignore low frame rate, held frames, clip length and codec softness.

Return only a JSON object of the form
{"occurrence": {"evidence": "...", "score": 0}, "transition": {...}, "given_frames": {...}, "reference_content": {...}, "defects": {...}}."""


def _apply(text: str, subs: dict) -> str:
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def build_ab_rubric(task: str) -> str:
    """System-instruction text for one A/B task variant ('two' | 'one')."""
    return _apply(_AB_RUBRIC, _AB_SUBS[task])


def build_score_rubric(task: str) -> str:
    """System-instruction text for one scoring task variant ('two' | 'one')."""
    return _apply(_SCORE_RUBRIC, _SCORE_SUBS[task])


# --- Response schemas — SPEC §3.2 / §4 --------------------------------------
# Built lazily from the SDK's types module (passed in) so importing this module
# never requires google-genai. property_ordering IS a real field on
# google.genai.types.Schema in google-genai 2.14, so evidence precedes the
# choice/score both in the rubric text and in the schema itself.
def ab_schema(types_mod):
    S = types_mod.Schema
    dim = S(type="OBJECT",
            required=["evidence", "choice", "margin"],
            property_ordering=["evidence", "choice", "margin"],
            properties={
                "evidence": S(type="STRING"),
                "choice": S(type="STRING", enum=["A", "B"]),
                "margin": S(type="STRING", enum=["none", "slight", "clear"]),
            })
    return S(type="OBJECT",
             required=list(AB_QUESTIONS),
             property_ordering=list(AB_QUESTIONS),
             properties={q: dim for q in AB_QUESTIONS})


def score_schema(types_mod):
    S = types_mod.Schema
    dim = S(type="OBJECT",
            required=["evidence", "score"],
            property_ordering=["evidence", "score"],
            properties={"evidence": S(type="STRING"), "score": S(type="INTEGER")})
    return S(type="OBJECT",
             required=list(SCORE_DIMS),
             property_ordering=list(SCORE_DIMS),
             properties={d: dim for d in SCORE_DIMS})


# --- Parsers ----------------------------------------------------------------
def parse_ab(raw: str) -> dict:
    """Parse + validate an A/B response. Each question needs a choice in
    {A,B} and a margin in {none,slight,clear}; anything else -> parse_error
    (never coerced, never retried). Returns {answers, parse_error, error}."""
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {"parse_error": True, "error": f"json: {e}", "answers": {}}
    if not isinstance(obj, dict):
        return {"parse_error": True, "error": "top-level not an object", "answers": {}}
    answers, errs = {}, []
    for q in AB_QUESTIONS:
        node = obj.get(q)
        if not isinstance(node, dict):
            errs.append(f"{q}: missing")
            continue
        choice, margin = node.get("choice"), node.get("margin")
        if choice not in ("A", "B"):
            errs.append(f"{q}: bad choice {choice!r}")
            continue
        if margin not in ("none", "slight", "clear"):
            errs.append(f"{q}: bad margin {margin!r}")
            continue
        answers[q] = {"choice": choice, "margin": margin,
                      "evidence": node.get("evidence", "")}
    if errs:
        return {"parse_error": True, "error": "; ".join(errs), "answers": answers}
    return {"parse_error": False, "error": None, "answers": answers}


def parse_score(raw: str) -> dict:
    """Parse + validate a scoring response. Every score must be an int in 0..4,
    else parse_error; never coerce. Returns {scores, evidence, parse_error}."""
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {"parse_error": True, "error": f"json: {e}", "scores": {}, "evidence": {}}
    if not isinstance(obj, dict):
        return {"parse_error": True, "error": "top-level not an object",
                "scores": {}, "evidence": {}}
    scores, evidence, errs = {}, {}, []
    for d in SCORE_DIMS:
        node = obj.get(d)
        if not isinstance(node, dict) or "score" not in node:
            errs.append(f"{d}: missing")
            continue
        s = node.get("score")
        if isinstance(s, bool) or not isinstance(s, int) or not (0 <= s <= 4):
            errs.append(f"{d}: bad score {s!r}")
            continue
        scores[d] = s
        evidence[d] = node.get("evidence", "")
    if errs:
        return {"parse_error": True, "error": "; ".join(errs),
                "scores": scores, "evidence": evidence}
    return {"parse_error": False, "error": None, "scores": scores, "evidence": evidence}


def score_postprocess(scores: dict) -> dict:
    """transition_adj = 0 if occurrence == 0 else transition (SPEC §4).
    Record both; report per item, never as a composite."""
    occ = scores.get("occurrence")
    tr = scores.get("transition")
    return {"transition_adj": (0 if occ == 0 else tr)}


# --- A/B decision rule (pure; SPEC §3) --------------------------------------
def segue_won(choice: str, segue_side: str) -> bool:
    """Did the SEGUE side win this call? SEGUE was on `segue_side` (A or B)."""
    return choice == segue_side


def decide_question(o1_choice, o1_segue_side, o2_choice, o2_segue_side):
    """Combine the two presentation orders for one question.

    both orders choose SEGUE -> (1.0, False)  [win]
    both choose the opponent -> (0.0, False)  [loss]
    the orders disagree      -> (0.5, True)   [tie, inconsistent]
    """
    w1 = segue_won(o1_choice, o1_segue_side)
    w2 = segue_won(o2_choice, o2_segue_side)
    if w1 and w2:
        return 1.0, False
    if (not w1) and (not w2):
        return 0.0, False
    return 0.5, True


def margin_none_both(o1_margin, o2_margin) -> bool:
    """Secondary margin view: a unit whose margin is 'none' in BOTH orders is a
    tie regardless of the choices (SPEC §3)."""
    return o1_margin == "none" and o2_margin == "none"


def is_decisive(decision, o1_margin, o2_margin) -> bool:
    """A decisive SEGUE win: decision == 1.0 and both orders margin 'clear'."""
    return decision == 1.0 and o1_margin == "clear" and o2_margin == "clear"


# --- shared client / cache / retry / probe layer ----------------------------
class _JudgeBase:
    """Client, config, retry, media parts and cache — shared by both judges.

    The client is built lazily on the first real (non-cached) call; a mock can
    be injected via `client=`. `_generate` can be overridden in a subclass or
    monkeypatched for offline tests.
    """

    def __init__(self, api_key: str | None = None, model: str = MODEL,
                 fps: float = DEFAULT_FPS, media_resolution: str = "default",
                 thinking_budget: int = 2048, max_output_tokens: int = 3072,
                 temperature: float = 0.0, seed: int = 0, max_retries: int = 5,
                 client=None):
        self._api_key = api_key
        self._client = client        # a real or mock client, or None (lazy)
        self._types = None           # google.genai.types, imported lazily
        self.model = model
        self.fps = fps
        self.media_resolution = media_resolution
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries

    # -- lazy SDK handles -----------------------------------------------------
    @property
    def types(self):
        if self._types is None:
            self._types = __import__("google.genai.types", fromlist=["types"])
        return self._types

    def _ensure_client(self):
        if self._client is None:
            from google import genai  # deferred: only a real call needs the SDK client
            self._client = (genai.Client(api_key=self._api_key)
                            if self._api_key else genai.Client())
        return self._client

    # -- request pieces -------------------------------------------------------
    def _video_part(self, path):
        t = self.types
        return t.Part(
            inline_data=t.Blob(mime_type="video/mp4",
                               data=pathlib.Path(path).read_bytes()),
            video_metadata=t.VideoMetadata(fps=self.fps),
        )

    def _image_part(self, path):
        t = self.types
        return t.Part(inline_data=t.Blob(mime_type="image/jpeg",
                                         data=still_jpeg_bytes(path)))

    def _config(self, system_instruction, response_schema):
        t = self.types
        kw = dict(
            temperature=self.temperature,
            seed=self.seed,
            response_mime_type="application/json",
            response_schema=response_schema,
            max_output_tokens=self.max_output_tokens,
            thinking_config=t.ThinkingConfig(thinking_budget=self.thinking_budget,
                                             include_thoughts=False),
            system_instruction=system_instruction,
        )
        if self.media_resolution == "low":
            kw["media_resolution"] = t.MediaResolution.MEDIA_RESOLUTION_LOW
        return t.GenerateContentConfig(**kw)

    def _generate(self, contents, config):
        """One request with the graded judge's retry policy: retry only on
        429/500/503/RESOURCE_EXHAUSTED/UNAVAILABLE, honoring the server's
        'retry in Ns' hint (+10 s), max 5; any other error is raised."""
        client = self._ensure_client()
        last = None
        for attempt in range(self.max_retries):
            try:
                return client.models.generate_content(
                    model=self.model, contents=contents, config=config)
            except Exception as e:
                last = e
                msg = str(e)
                if not any(k in msg for k in _RETRYABLE):
                    raise
                m = re.search(r"retry in ([0-9.]+)s", msg)
                time.sleep(float(m.group(1)) + 10.0 if m
                           else min(60.0, 2.0 ** attempt * 5.0))
        raise last

    # -- provenance -----------------------------------------------------------
    def _settings(self):
        return {
            "model": self.model, "fps": self.fps,
            "media_resolution": self.media_resolution,
            "thinking_budget": self.thinking_budget,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature, "seed": self.seed,
            "ab_rubric_version": AB_RUBRIC_VERSION,
        }

    @staticmethod
    def _video_provenance(label, path):
        return {f"{label}": str(path),
                f"{label}_sha256": sha256(path),
                f"{label}_probe": probe(path)}

    def _base_record(self, task, meta, raw, resp, wall, videos, stills):
        """Common provenance block (SPEC §6). `videos` = {label: path};
        `stills` = {label: path|None}."""
        um = getattr(resp, "usage_metadata", None)
        rec = dict(self._settings())
        rec.update({
            "model_version": getattr(resp, "model_version", self.model),
            "task": task,
            "row": meta.get("row", {}),
            "unit": meta.get("unit", {}),
            "usage": _usage_dict(um),
            "wall_s": wall,
            "raw": raw,
            "_cached": False,
        })
        files = {}
        for label, p in stills.items():
            files[label] = (str(p) if p else None)
        for label, p in videos.items():
            files.update(self._video_provenance(label, p))
        rec["files"] = files
        return rec

    # -- cache ----------------------------------------------------------------
    @staticmethod
    def _read_cache(cache_file):
        if cache_file and pathlib.Path(cache_file).exists():
            rec = json.loads(pathlib.Path(cache_file).read_text())
            rec["_cached"] = True
            return rec
        return None

    @staticmethod
    def _write_cache(cache_file, rec):
        if not cache_file:
            return
        cache_file = pathlib.Path(cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
        tmp.write_text(json.dumps(rec, indent=2))
        tmp.replace(cache_file)


class JudgeAB(_JudgeBase):
    """Forced A/B pairwise judge. One call = one presentation order."""

    def judge_pair(self, task: str, start, end, reference, out_a, out_b,
                   meta: dict) -> dict:
        if task not in ("two", "one"):
            raise ValueError(f"task must be 'two' or 'one', got {task!r}")
        cache_file = meta.get("cache_file")
        cached = self._read_cache(cache_file)
        if cached is not None:
            return cached
        if task == "two" and not end:
            raise ValueError("two-sided task requires an end still")

        contents = [self.types.Part(text=_TASK_LINE[task]),
                    self.types.Part(text="GIVEN START (still image):"),
                    self._image_part(start)]
        if task == "two":
            contents += [self.types.Part(text="GIVEN END (still image):"),
                         self._image_part(end)]
        contents += [
            self.types.Part(text="REFERENCE video (shows the effect on its own scenes):"),
            self._video_part(reference),
            self.types.Part(text="OUTPUT A:"), self._video_part(out_a),
            self.types.Part(text="OUTPUT B:"), self._video_part(out_b),
            self.types.Part(text="Compare A and B now. Return only the JSON object."),
        ]
        config = self._config(build_ab_rubric(task), ab_schema(self.types))

        t0 = time.time()
        resp = self._generate(contents, config)
        wall = round(time.time() - t0, 2)
        raw = getattr(resp, "text", "") or ""
        parsed = parse_ab(raw)

        rec = self._base_record(
            task, meta, raw, resp, wall,
            videos={"reference_mp4": reference, "out_a_mp4": out_a, "out_b_mp4": out_b},
            stills={"start_png": start, "end_png": end})
        rec.update({
            "mode": "ab",
            "order": meta.get("order"),          # 'order1' | 'order2'
            "segue_side": meta.get("segue_side"),  # 'A' | 'B'
            "system_a": meta.get("system_a"),
            "system_b": meta.get("system_b"),
            "answers": parsed["answers"],
            "parse_error": parsed["parse_error"],
            "parse_detail": parsed["error"],
        })
        self._write_cache(cache_file, rec)
        return rec


class JudgeScore(_JudgeBase):
    """Absolute 0-4 scoring judge. One call = one OUTPUT clip."""

    def judge(self, task: str, start, end, reference, output, meta: dict) -> dict:
        if task not in ("two", "one"):
            raise ValueError(f"task must be 'two' or 'one', got {task!r}")
        cache_file = meta.get("cache_file")
        cached = self._read_cache(cache_file)
        if cached is not None:
            return cached
        if task == "two" and not end:
            raise ValueError("two-sided task requires an end still")

        contents = [self.types.Part(text=_TASK_LINE[task]),
                    self.types.Part(text="GIVEN START (still image):"),
                    self._image_part(start)]
        if task == "two":
            contents += [self.types.Part(text="GIVEN END (still image):"),
                         self._image_part(end)]
        contents += [
            self.types.Part(text="REFERENCE video (shows the effect on its own scenes):"),
            self._video_part(reference),
            self.types.Part(text="OUTPUT video to judge:"), self._video_part(output),
            self.types.Part(text="Score the OUTPUT now. Return only the JSON object."),
        ]
        config = self._config(build_score_rubric(task), score_schema(self.types))

        t0 = time.time()
        resp = self._generate(contents, config)
        wall = round(time.time() - t0, 2)
        raw = getattr(resp, "text", "") or ""
        parsed = parse_score(raw)
        post = score_postprocess(parsed["scores"])

        rec = self._base_record(
            task, meta, raw, resp, wall,
            videos={"reference_mp4": reference, "output_mp4": output},
            stills={"start_png": start, "end_png": end})
        rec.update({
            "mode": "score",
            "system": meta.get("system"),
            "scores": parsed["scores"],
            "evidence": parsed["evidence"],
            "transition_adj": post["transition_adj"],
            "parse_error": parsed["parse_error"],
            "parse_detail": parsed["error"],
        })
        self._write_cache(cache_file, rec)
        return rec
