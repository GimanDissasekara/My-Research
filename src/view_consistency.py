"""View-consistency checks using lightweight image features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


DEFAULT_PAIRS = (
    ("front", "back"),
    ("left", "right"),
    ("top", "bottom"),
)


@dataclass
class ConsistencyResult:
    view_scores: Dict[str, float]
    pair_scores: Dict[Tuple[str, str], float]


def _embed_image(path: Path, size: int = 32) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((size, size))
    arr = np.asarray(image, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    return arr / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_view_consistency(
    view_paths: Dict[str, Path],
    similarity_threshold: float = 0.85,
) -> ConsistencyResult:
    embeddings: Dict[str, np.ndarray] = {}
    for name, path in view_paths.items():
        embeddings[name] = _embed_image(path)

    view_scores: Dict[str, float] = {name: 0.0 for name in embeddings}
    pair_scores: Dict[Tuple[str, str], float] = {}

    for a, b in DEFAULT_PAIRS:
        if a not in embeddings or b not in embeddings:
            continue
        sim = _cosine(embeddings[a], embeddings[b])
        pair_scores[(a, b)] = sim
        if sim < similarity_threshold:
            severity = min(1.0, max(0.0, 1.0 - sim))
            view_scores[a] = max(view_scores[a], severity)
            view_scores[b] = max(view_scores[b], severity)

    return ConsistencyResult(view_scores=view_scores, pair_scores=pair_scores)
