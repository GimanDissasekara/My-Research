"""LMM detector using local LLaVA for multi-view hallucination checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import json
import re
import sys

import numpy as np
from PIL import Image


@dataclass
class LmmDetection:
    global_bias: float
    view_scores: Dict[str, float]
    notes: List[str]


class LLaVADetector:
    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        llava_root = Path(__file__).resolve().parents[1] / "LLaVA"
        if llava_root.exists():
            sys.path.insert(0, str(llava_root))

        import torch
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.image_dtype = torch.float16 if device == "cuda" else torch.float32

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            device=device,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.model_name = model_name

    def score_view(self, view_name: str, image_path: Path) -> tuple[float, str]:
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_TOKEN_INDEX,
            IMAGE_PLACEHOLDER,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        image = Image.open(image_path).convert("RGB")
        query = (
            "You are checking a 3D render for geometric hallucinations. "
            f"The view name is '{view_name}'. "
            "Return JSON with keys: severity (0-1 float), notes (string)."
        )

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in query:
            if self.model.config.mm_use_im_start_end:
                query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
            else:
                query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
        else:
            if self.model.config.mm_use_im_start_end:
                query = image_token_se + "\n" + query
            else:
                query = DEFAULT_IMAGE_TOKEN + "\n" + query

        conv_mode = _resolve_conv_mode(self.model_name)
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(
            [image],
            self.image_processor,
            self.model.config,
        ).to(self.model.device, dtype=self.image_dtype)

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        severity, notes = _parse_detection(outputs)
        return severity, notes


def detect_issues(
    view_paths: Dict[str, Path],
    detector: Optional[LLaVADetector] = None,
) -> LmmDetection:
    if not view_paths:
        return LmmDetection(global_bias=0.0, view_scores={}, notes=["no_views"])
    if detector is None:
        raise RuntimeError("LLaVA detector not configured")

    view_scores: Dict[str, float] = {}
    notes: List[str] = []
    for name, path in view_paths.items():
        severity, note = detector.score_view(name, path)
        view_scores[name] = severity
        if note:
            notes.append(f"{name}:{note}")

    global_bias = float(np.mean(list(view_scores.values()))) if view_scores else 0.0
    return LmmDetection(global_bias=global_bias, view_scores=view_scores, notes=notes)


def _parse_detection(text: str) -> tuple[float, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return 0.0, "parse_failed"
    try:
        data = json.loads(match.group(0))
        severity = float(data.get("severity", 0.0))
        notes = str(data.get("notes", ""))
        return float(max(0.0, min(1.0, severity))), notes
    except Exception:
        return 0.0, "parse_failed"


def _resolve_conv_mode(model_name: str) -> str:
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "mistral" in name:
        return "mistral_instruct"
    if "v1.6-34b" in name:
        return "chatml_direct"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"
