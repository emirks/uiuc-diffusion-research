"""M4 — Rubric VLM judge: checklist-based, not vibes-based.

Fixed judge model (local Gemma 3 12B-it, vision tower confirmed), greedy
decoding, per-question yes/no + evidence, JSON output. Directly prompting VLMs
for holistic scores yields under-justified false positives; each deduction
here must cite concrete frames.

STATUS: EXPERIMENTAL until validated against human labels on ~50-100 outputs
(target Spearman >= 0.8). Do not use as a headline number before that.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .rubric import (  # noqa: F401 — re-exported for existing callers
    RUBRIC_QUESTIONS, judge_pass_rate, parse_judge_json,
)

PROMPT_TEMPLATE = """You are grading a generated video transition against a reference transition of the same style. You are shown {n_ref} frames of the REFERENCE video, then {n_gen} frames of the GENERATED video, in temporal order.

Answer the checklist strictly. For every question give: "answer" (true/false), and "evidence" (one sentence naming the specific frame numbers that support the answer). Do not deduct without concrete frame evidence.

Checklist:
{questions}

Respond with ONLY a JSON object of the form:
{{"q1_same_type": {{"answer": true, "evidence": "..."}}, "q2_dynamics": {{...}}, "q3_endpoints": {{...}}, "q4_leakage": {{...}}, "q5_artifacts": {{...}}}}"""


def _sample_frames(frames: np.ndarray, n: int) -> list[Image.Image]:
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [Image.fromarray(frames[i]) for i in idx]


class RubricJudge:
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def judge(self, ref_frames: np.ndarray, gen_frames: np.ndarray, n_frames: int = 8) -> dict:
        questions = "\n".join(f"- {k}: {v}" for k, v in RUBRIC_QUESTIONS.items())
        prompt = PROMPT_TEMPLATE.format(n_ref=n_frames, n_gen=n_frames, questions=questions)
        content = [{"type": "text", "text": "REFERENCE frames (temporal order):"}]
        content += [{"type": "image", "image": im} for im in _sample_frames(ref_frames, n_frames)]
        content += [{"type": "text", "text": "GENERATED frames (temporal order):"}]
        content += [{"type": "image", "image": im} for im in _sample_frames(gen_frames, n_frames)]
        content += [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=600, do_sample=False)
        text = self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_judge_json(text)

    def free(self) -> None:
        del self.model
        torch.cuda.empty_cache()
