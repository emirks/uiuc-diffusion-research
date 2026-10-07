"""CTT VLM judge (SEGUE) — Gemini 2.5 Pro, native video + endpoint stills.

Judges one OUTPUT transition against a REFERENCE transition of the same effect,
given the endpoint still(s). The REFERENCE is the ONLY specification of the
effect: no class name, no effect text ever reaches the model. Five dimensions
(occurrence / fidelity / disentanglement / endpoint / quality), each 0-4 with
one sentence of timestamp-citing evidence; reported per dimension, never as a
composite (VFXMaster's occurrence/fidelity/leakage decomposition, extended with
an endpoint dimension and graded 0-4).

Contract (misc/2026-09-21_vlm_judge/SPEC.md): pinned model string, temperature
0, seed 0, thinking_budget 2048, sampling fps PINNED to 8 via VideoMetadata
(the default 1 fps starves motion judgments — a 2.2 s clip becomes 3 frames),
endpoints sent as JPEG q95 stills, response constrained by a JSON schema, and
EVERY call's raw response + full provenance cached to disk before parsing so
reruns are free and auditable. No torch dependency; runs on a login node.

Login-node gotchas baked in: cv2.setNumThreads(1) at import (thread spawns are
refused); video FRAME/FPS probing uses cv2 container metadata only (pixel
decode via cv2's bundled swscaler returns black frames on this node — callers
that need pixels must use imageio_ffmpeg's binary, see run_examples.py).

STATUS: EXPERIMENTAL until validated against the fixed anchor set.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import time

import cv2

cv2.setNumThreads(1)  # login node refuses thread spawns; do this before any cv2 call

MODEL = "gemini-2.5-pro"   # pinned; do not float to previews silently
DEFAULT_FPS = 8.0          # sampling rate requested for BOTH videos (VideoMetadata)

DIMENSIONS = ("occurrence", "fidelity", "disentanglement", "endpoint", "quality")

_RETRYABLE = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")

# --- Rubric (system instruction) — verbatim from SPEC.md §3 -------------------
RUBRIC_TEMPLATE = """You are an expert judge of video transition effects. You receive: the GIVEN START endpoint as a still image,
{task_clause} a REFERENCE video, and an OUTPUT video. The REFERENCE shows a transition effect between its own two
scenes. The OUTPUT is supposed to reproduce that same transition effect between the given endpoint(s), which are
different scenes from the reference's. No text describes the effect: the REFERENCE is the only specification.

All videos are time-normalized to the same duration and sampled at {fps} frames per second. Their native frame
rates differ, so held or repeated frames are expected and must never be penalized.

Score five dimensions, each an integer from 0 to 4, and give one sentence of evidence per dimension that cites
timestamps (for example 0:01-0:02). Be strict and literal: score what is visible, not what is plausible.

1. OCCURRENCE. Does the OUTPUT contain a transition effect at all, beyond a plain cut, a crossfade or dissolve,
   a fade through black or white, a blur or blend between the endpoints, or a static image?
   0 = no transition, or only a cut, dissolve, fade, blur or blend. 1 = faint traces of an effect.
   2 = an effect is present but weak or partial. 3 = a clear effect. 4 = a clear, complete effect that runs its
   course: onset, development, resolution.

2. FIDELITY. Given that an effect occurs, is it the REFERENCE's effect? Same mechanism and material (what appears
   and how: smoke, fire, liquid, particles, a portal, a morph, a shatter...), the same way it enters, spreads and
   clears, and a similar timing. Judge the effect's properties only. The scenes are supposed to differ, so
   differences in people, objects or backgrounds are NOT fidelity errors. 0 = a different effect, or no effect.
   1 = only a loose resemblance. 2 = same family but a different mechanism, material or motion. 3 = the same
   mechanism with minor differences in look or timing. 4 = the same effect, convincingly. If OCCURRENCE is 0,
   FIDELITY is 0.

3. DISENTANGLEMENT. Does content that belongs to the REFERENCE's own scenes (its people, objects, backgrounds,
   the look of its scenes) appear in the OUTPUT? The OUTPUT should show only the given endpoints' content plus the
   effect's own material. Material that is part of the effect itself (the smoke, the fire, the particles) is NOT
   leakage. 4 = nothing from the reference's scenes appears. 3 = a faint trace, such as a color cast or a vague
   shape. 2 = a recognizable element of a reference scene appears briefly. 1 = reference scene content appears
   prominently. 0 = the OUTPUT shows the reference's scenes instead of the given endpoints.

4. ENDPOINT ADHERENCE. Does the OUTPUT open on the GIVEN START (same scene, subject, layout and identity)
   {endpoint_clause} Is the given content entered and exited without a jump, a swap or identity drift?
   0 = the given endpoint(s) are not shown. 1 = shown, but with the wrong identity or layout, or an abrupt swap.
   2 = shown, with visible drift or a hard jump. 3 = shown, with minor drift. 4 = the OUTPUT clearly opens on
   START{end_ref} with a seamless hand-off.

5. QUALITY. Visual defects a viewer would notice on one viewing: flicker, strobing, smearing, broken anatomy,
   duplicated limbs, tiling, corruption, frozen frames that the effect does not explain. 4 = clean. 3 = minor.
   2 = noticeable. 1 = distracting. 0 = severe. Ignore low native frame rate, held frames, codec softness, and
   blending that is part of the effect.

Return only a JSON object of the form
{{"occurrence": {{"score": 0, "evidence": "..."}}, "fidelity": {{...}}, "disentanglement": {{...}},
 "endpoint": {{...}}, "quality": {{...}}}}."""

# Task-variant substitutions — SPEC.md §3.
_TASK_LINE = {
    "two": "Task: TWO ENDPOINTS GIVEN",
    "one": "Task: START ENDPOINT GIVEN (open end)",
}
_TASK_CLAUSE = {"two": "the GIVEN END endpoint as a still image,", "one": ""}
_ENDPOINT_CLAUSE = {"two": "and close on the GIVEN END?",
                    "one": "(the end is open: judge the start only)?"}
_END_REF = {"two": " and closes on END", "one": ""}


def build_rubric(task: str, fps: float = DEFAULT_FPS) -> str:
    """The system-instruction text for one task variant ('two' | 'one')."""
    return RUBRIC_TEMPLATE.format(
        fps=f"{fps:g}",
        task_clause=_TASK_CLAUSE[task],
        endpoint_clause=_ENDPOINT_CLAUSE[task],
        end_ref=_END_REF[task],
    )


# --- Response schema — SPEC.md §4 --------------------------------------------
_DIM = {"type": "OBJECT", "required": ["score", "evidence"],
        "properties": {"score": {"type": "INTEGER"}, "evidence": {"type": "STRING"}}}
RESPONSE_SCHEMA = {"type": "OBJECT", "required": list(DIMENSIONS),
                   "properties": {d: _DIM for d in DIMENSIONS}}


# --- Small utilities ---------------------------------------------------------
def sha256(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: str | pathlib.Path) -> dict:
    """(frames, fps, duration) from cv2 CONTAINER metadata — no pixel decode
    (decode is broken on the login node). Verified exact against ffmpeg for the
    std121 / VAP / refVFX / FLF2V / VACE / ED-81 clip families."""
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    dur = (frames / fps) if fps else 0.0
    return {"frames": frames, "fps": round(fps, 3), "duration_s": round(dur, 3)}


def still_jpeg_bytes(path: str | pathlib.Path, quality: int = 95) -> bytes:
    """Load a still (PNG/JPEG) and re-encode to JPEG q95 for the API.
    cv2.imread/imencode use libpng/libjpeg (no swscaler) and work on the node."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read still: {path}")
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"JPEG encode failed: {path}")
    return buf.tobytes()


def _usage_dict(um) -> dict:
    """Flatten usage_metadata; prompt token modalities as {modality: count}."""
    def modality_map(details):
        out = {}
        for m in (details or []):
            key = getattr(m.modality, "name", None) or str(m.modality)
            out[key] = m.token_count
        return out
    return {
        "prompt_token_count": getattr(um, "prompt_token_count", None),
        "prompt_tokens_details": modality_map(getattr(um, "prompt_tokens_details", None)),
        "cached_content_token_count": getattr(um, "cached_content_token_count", None),
        "candidates_token_count": getattr(um, "candidates_token_count", None),
        "thoughts_token_count": getattr(um, "thoughts_token_count", None),
        "total_token_count": getattr(um, "total_token_count", None),
    }


def parse_scores(raw: str) -> dict:
    """Parse + validate. Every score must be an int in 0..4, else parse_error;
    never coerce. Returns {scores, evidence, parse_error, error}."""
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {"parse_error": True, "error": f"json: {e}", "scores": {}, "evidence": {}}
    scores, evidence, errs = {}, {}, []
    for d in DIMENSIONS:
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
        return {"parse_error": True, "error": "; ".join(errs), "scores": scores, "evidence": evidence}
    return {"parse_error": False, "error": None, "scores": scores, "evidence": evidence}


def postprocess(scores: dict) -> dict:
    """fidelity_adj + convenience pass flag (SPEC.md §3). Reported alongside the
    per-dimension scores, never in place of them."""
    occ = scores.get("occurrence")
    fid = scores.get("fidelity")
    fid_adj = 0 if occ == 0 else fid
    passed = None
    if all(scores.get(d) is not None for d in DIMENSIONS):
        passed = bool(occ >= 3 and fid_adj >= 3
                      and scores["disentanglement"] >= 3 and scores["endpoint"] >= 3)
    return {"fidelity_adj": fid_adj, "pass": passed}


class JudgeCTT:
    """One Gemini call per judged item.

    judge(task, start_png, end_png_or_None, reference_mp4, output_mp4, meta) -> dict

      task            : 'two' (two endpoints given) | 'one' (start only, open end)
      start_png       : path to the GIVEN START still (any image; re-encoded JPEG q95)
      end_png_or_None : path to the GIVEN END still, or None for one-sided
      reference_mp4   : the canonical reference transition (effect specification)
      output_mp4      : the clip being judged
      meta            : provenance dict; MUST carry block, row_key, system, and a
                        `row` sub-dict (endpoint, reference, seed, cell,
                        gt_pool_class, sided). Anything else is recorded verbatim.

    Cache: <cache_dir>/<block>/<row_key>__<system>.json (SPEC.md §5). A cached
    item is returned without an API call.
    """

    def __init__(self, api_key: str | None = None, model: str = MODEL,
                 fps: float = DEFAULT_FPS, cache_dir: str | pathlib.Path | None = None,
                 media_resolution: str = "default", thinking_budget: int = 2048,
                 max_output_tokens: int = 3072, temperature: float = 0.0,
                 seed: int = 0, max_retries: int = 5):
        # NOTE on max_output_tokens: gemini-2.5-pro counts THINKING tokens
        # against max_output_tokens. SPEC.md §2 pins both max_output_tokens and
        # thinking_budget at 2048, but measured on 2026-09-21 that starves the
        # visible JSON (thoughts ran 1.8k-2.05k, leaving <300 tokens for the
        # answer -> 10/13 truncated, unparseable). The pinned thinking_budget is
        # kept at 2048; the total cap is raised to 3072 so the answer always has
        # >=1024 tokens of room. (Recorded as the single spec deviation.)
        from google import genai  # deferred: only this backend needs the SDK

        self._types = __import__("google.genai.types", fromlist=["types"])
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.fps = fps
        self.media_resolution = media_resolution
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None

    # -- request pieces -------------------------------------------------------
    def _video_part(self, path):
        t = self._types
        return t.Part(
            inline_data=t.Blob(mime_type="video/mp4", data=pathlib.Path(path).read_bytes()),
            video_metadata=t.VideoMetadata(fps=self.fps),
        )

    def _image_part(self, path):
        t = self._types
        return t.Part(inline_data=t.Blob(mime_type="image/jpeg", data=still_jpeg_bytes(path)))

    def _config(self, task):
        t = self._types
        kw = dict(
            temperature=self.temperature,
            seed=self.seed,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=self.max_output_tokens,
            thinking_config=t.ThinkingConfig(thinking_budget=self.thinking_budget,
                                             include_thoughts=False),
            system_instruction=build_rubric(task, self.fps),
        )
        if self.media_resolution == "low":
            kw["media_resolution"] = t.MediaResolution.MEDIA_RESOLUTION_LOW
        # 'default' -> leave media_resolution unset
        return t.GenerateContentConfig(**kw)

    def _generate(self, contents, config):
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config)
                return resp
            except Exception as e:
                last = e
                msg = str(e)
                if not any(k in msg for k in _RETRYABLE):
                    raise
                m = re.search(r"retry in ([0-9.]+)s", msg)
                time.sleep(float(m.group(1)) + 10.0 if m else min(60.0, 2.0 ** attempt * 5.0))
        raise last

    # -- cache path -----------------------------------------------------------
    @staticmethod
    def _fs_safe(s: str) -> str:
        # system names can contain '/', e.g. "SEGUE w/o NRG" -> keep it in one
        # path segment instead of spawning a spurious sub-directory.
        return re.sub(r"[/\\]", "-", str(s))

    def _cache_file(self, meta):
        if not self.cache_dir:
            return None
        block = self._fs_safe(meta["block"])
        key = f'{self._fs_safe(meta["row_key"])}__{self._fs_safe(meta["system"])}.json'
        return self.cache_dir / block / key

    # -- public API -----------------------------------------------------------
    def judge(self, task: str, start_png, end_png_or_None, reference_mp4,
              output_mp4, meta: dict) -> dict:
        if task not in ("two", "one"):
            raise ValueError(f"task must be 'two' or 'one', got {task!r}")

        cache_file = self._cache_file(meta)
        if cache_file and cache_file.exists():
            rec = json.loads(cache_file.read_text())
            rec["_cached"] = True
            return rec

        t = self._types
        contents = [t.Part(text=_TASK_LINE[task]),
                    t.Part(text="GIVEN START (still image):"), self._image_part(start_png)]
        if task == "two":
            if not end_png_or_None:
                raise ValueError("two-sided task requires end_png_or_None")
            contents += [t.Part(text="GIVEN END (still image):"),
                         self._image_part(end_png_or_None)]
        contents += [
            t.Part(text="REFERENCE video (shows the effect on its own scenes):"),
            self._video_part(reference_mp4),
            t.Part(text="OUTPUT video to judge:"),
            self._video_part(output_mp4),
            t.Part(text="Score the OUTPUT now. Return only the JSON object."),
        ]

        config = self._config(task)
        t0 = time.time()
        resp = self._generate(contents, config)
        wall = round(time.time() - t0, 2)

        raw = resp.text or ""
        model_version = getattr(resp, "model_version", self.model)
        usage = _usage_dict(getattr(resp, "usage_metadata", None))
        parsed = parse_scores(raw)
        post = postprocess(parsed["scores"])

        rec = {
            "model": self.model,
            "model_version": model_version,
            "fps": self.fps,
            "media_resolution": self.media_resolution,
            "thinking_budget": self.thinking_budget,
            "temperature": self.temperature,
            "seed": self.seed,
            "task": task,
            "row": meta.get("row", {}),
            "block": meta.get("block"),
            "row_key": meta.get("row_key"),
            "system": meta.get("system"),
            "files": {
                "start_png": str(start_png),
                "end_png": (str(end_png_or_None) if end_png_or_None else None),
                "reference_mp4": str(reference_mp4),
                "output_mp4": str(output_mp4),
                "output_sha256": sha256(output_mp4),
                "reference_sha256": sha256(reference_mp4),
                "output_probe": probe(output_mp4),
                "reference_probe": probe(reference_mp4),
            },
            "usage": usage,
            "wall_s": wall,
            "raw": raw,
            "scores": parsed["scores"],
            "evidence": parsed["evidence"],
            "parse_error": parsed["parse_error"],
            "parse_detail": parsed["error"],
            "fidelity_adj": post["fidelity_adj"],
            "pass": post["pass"],
            "_cached": False,
        }
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec, indent=2))
            tmp.replace(cache_file)
        return rec
