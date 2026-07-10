"""M4 backend B — Gemini API judge with NATIVE video input.

Motivation (exp_052 → exp_053): the local Gemma backend sees 8 stills per
video, and its two motion-dependent questions (q2 dynamics, q5 artifacts)
answered false for every item of every arm — sparse sampling cannot ground a
timing judgment. This backend sends both full videos with an explicit
sampling fps, so the judge actually sees the motion it is asked to grade.

Determinism/reproducibility contract: pinned model string, temperature 0,
JSON response mime type, sampling fps recorded per call, and EVERY raw
response cached to disk (item-keyed) before parsing — reruns are free and
auditable. No torch dependency; runs on a login node.

STATUS: EXPERIMENTAL until validated against human labels (same contract as
the local backend).
"""

from __future__ import annotations

import json
import pathlib
import re
import time

from .rubric import RUBRIC_QUESTIONS, parse_judge_json

MODEL = "gemini-3.5-flash"   # GA/stable; do not float to previews silently
DEFAULT_FPS = 8.0            # native sampling rate requested for both videos

VIDEO_PROMPT_TEMPLATE = """You are grading a generated video transition against a reference transition of the same style. You are shown the REFERENCE video first, then the GENERATED video. Both are sampled at {fps:g} frames per second.

Answer the checklist strictly. For every question give: "answer" (true/false), and "evidence" (one sentence citing the specific timestamps, e.g. "0:02-0:03", that support the answer). Do not deduct without concrete timestamp evidence.

Checklist:
{questions}

Respond with ONLY a JSON object of the form:
{{"q1_same_type": {{"answer": true, "evidence": "..."}}, "q2_dynamics": {{...}}, "q3_endpoints": {{...}}, "q4_leakage": {{...}}, "q5_artifacts": {{...}}}}"""

_RETRYABLE = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE")

# Structured-output schema: response_mime_type alone does NOT pin the shape —
# one exp_053 item came back as a score-array instead of the checklist.
_ANSWER = {"type": "OBJECT", "required": ["answer", "evidence"],
           "properties": {"answer": {"type": "BOOLEAN"}, "evidence": {"type": "STRING"}}}
RESPONSE_SCHEMA = {"type": "OBJECT",
                   "required": list(RUBRIC_QUESTIONS),
                   "properties": {q: _ANSWER for q in RUBRIC_QUESTIONS}}


class GeminiJudge:
    """One API call per item: [text, ref video, text, gen video, prompt]."""

    def __init__(self, api_key: str | None = None, model: str = MODEL,
                 fps: float = DEFAULT_FPS, cache_dir: str | pathlib.Path | None = None,
                 max_retries: int = 5):
        from google import genai  # deferred: only this backend needs the SDK

        self._types = __import__("google.genai.types", fromlist=["types"])
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.fps = fps
        self.max_retries = max_retries
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _video_part(self, path: pathlib.Path):
        t = self._types
        return t.Part(
            inline_data=t.Blob(mime_type="video/mp4", data=pathlib.Path(path).read_bytes()),
            video_metadata=t.VideoMetadata(fps=self.fps),
        )

    def _generate(self, contents) -> tuple[str, str]:
        t = self._types
        cfg = t.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=2048,
            thinking_config=t.ThinkingConfig(thinking_level="low"),
        )
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=cfg)
                return resp.text or "", getattr(resp, "model_version", self.model)
            except Exception as e:  # SDK raises typed errors; match on message
                last = e
                msg = str(e)
                if not any(k in msg for k in _RETRYABLE):
                    raise
                # free-tier 429s say "retry in Ns" — honor it (+margin) so a
                # windowed quota drains instead of burning attempts
                m = re.search(r"retry in ([0-9.]+)s", msg)
                time.sleep(float(m.group(1)) + 10.0 if m else min(60.0, 2.0 ** attempt * 5.0))
        raise last

    def judge(self, ref_video: pathlib.Path, gen_video: pathlib.Path,
              item_id: str | None = None) -> dict:
        cache_file = (self.cache_dir / f"{item_id}.json") if (self.cache_dir and item_id) else None
        if cache_file and cache_file.exists():
            cached = json.loads(cache_file.read_text())
            res = parse_judge_json(cached["raw"])
            res["_model_version"] = cached.get("model_version")
            res["_cached"] = True
            return res

        questions = "\n".join(f"- {k}: {v}" for k, v in RUBRIC_QUESTIONS.items())
        prompt = VIDEO_PROMPT_TEMPLATE.format(fps=self.fps, questions=questions)
        t = self._types
        contents = [
            t.Part(text="REFERENCE video:"), self._video_part(ref_video),
            t.Part(text="GENERATED video:"), self._video_part(gen_video),
            t.Part(text=prompt),
        ]
        raw, model_version = self._generate(contents)
        if cache_file:
            cache_file.write_text(json.dumps({
                "item_id": item_id, "model": self.model, "model_version": model_version,
                "fps": self.fps, "ref_video": str(ref_video), "gen_video": str(gen_video),
                "raw": raw}, indent=2))
        res = parse_judge_json(raw)
        res["_model_version"] = model_version
        return res
