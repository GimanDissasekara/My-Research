"""View-consistency checks using lightweight image features.

Improvements over v1
--------------------
* All-pairs comparison (itertools.combinations) instead of only 3 fixed pairs.
* Angular distance weighting — nearly-opposite views carry more signal.
* Depth-edge anomaly score via Canny edge detection on the depth image.
  High depth-edge density → floating fragments or geometric tears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


# Canonical view directions used for angular weighting between pairs
_VIEW_DIRECTIONS: Dict[str, np.ndarray] = {
    "front":  np.array([0.0,  0.0,  1.0]),
    "back":   np.array([0.0,  0.0, -1.0]),
    "left":   np.array([-1.0, 0.0,  0.0]),
    "right":  np.array([1.0,  0.0,  0.0]),
    "top":    np.array([0.0,  1.0,  0.0]),
    "bottom": np.array([0.0, -1.0,  0.0]),
}


@dataclass
class ConsistencyResult:
    view_scores: Dict[str, float]
    pair_scores: Dict[Tuple[str, str], float]
    depth_scores: Dict[str, float] = field(default_factory=dict)


def _embed_image(path: Path, size: int = 32) -> np.ndarray:
    """Embed an image as a normalised L2 flat vector for cosine comparison."""
    image = Image.open(path).convert("L")
    image = image.resize((size, size))
    arr = np.asarray(image, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    return arr / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _view_angle_weight(name_a: str, name_b: str) -> float:
    """Return a weight in [0, 1] based on angular separation of two view directions.

    Opposite views (front↔back, cos=-1) → weight 1.0 (most informative).
    Adjacent views (cos≈0)              → weight 0.5.
    Same direction (cos≈1)              → weight 0.0 (not useful to compare).
    """
    da = _VIEW_DIRECTIONS.get(name_a)
    db = _VIEW_DIRECTIONS.get(name_b)
    if da is None or db is None:
        return 0.5  # unknown view name — neutral weight
    cos_ab = float(np.dot(da, db))
    # Map [-1, 1] → [1, 0] linearly
    return float(np.clip((1.0 - cos_ab) / 2.0, 0.0, 1.0))


def depth_edge_score(depth_path: Path, canny_lo: int = 30, canny_hi: int = 100) -> float:
    """Return a normalised anomaly score [0, 1] from depth-image edge density.

    High edge density in the depth map indicates floating fragments or sharp
    geometric tears — a strong indicator of hallucinated geometry.

    Returns 0.0 if OpenCV is unavailable (soft dependency).
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return 0.0  # cv2 unavailable — skip silently

    try:
        depth = np.array(Image.open(depth_path).convert("L"))
        edges = cv2.Canny(depth, canny_lo, canny_hi)
        # Normalised edge density: fraction of pixels that are edges
        density = float(edges.sum()) / (depth.size * 255)
        # Saturate at ~20 % edge density — anything above is very anomalous
        return float(np.clip(density / 0.20, 0.0, 1.0))
    except Exception:
        return 0.0


def check_view_consistency(
    view_paths: Dict[str, Path],
    depth_paths: Optional[Dict[str, Path]] = None,
    similarity_threshold: float = 0.85,
) -> ConsistencyResult:
    """Check cross-view consistency using all view pairs, weighted by angular distance.

    Parameters
    ----------
    view_paths:
        Mapping of view name → RGB image path.
    depth_paths:
        Optional mapping of view name → depth image path.
        When provided, depth-edge scores are computed and blended into
        ``view_scores`` (weight 0.6 so they don't dominate).
    similarity_threshold:
        Pairs with cosine similarity below this threshold trigger a severity
        signal. Default 0.85.
    """
    # Build image embeddings
    embeddings: Dict[str, np.ndarray] = {}
    for name, path in view_paths.items():
        try:
            embeddings[name] = _embed_image(path)
        except Exception:
            pass  # skip unreadable images

    view_scores: Dict[str, float] = {name: 0.0 for name in embeddings}
    pair_scores: Dict[Tuple[str, str], float] = {}

    # All pairs — not just the 3 canonical opposites
    for (a, ea), (b, eb) in combinations(embeddings.items(), 2):
        sim = _cosine(ea, eb)
        pair_scores[(a, b)] = sim
        if sim < similarity_threshold:
            severity = float(np.clip(1.0 - sim, 0.0, 1.0))
            # Weight by angular distance: opposite views carry the most signal
            weight = _view_angle_weight(a, b)
            view_scores[a] = max(view_scores[a], severity * weight)
            view_scores[b] = max(view_scores[b], severity * weight)

    # Depth-edge scores — blended into view_scores at 60 % weight
    depth_scores: Dict[str, float] = {}
    if depth_paths:
        for name, dpath in depth_paths.items():
            if dpath.exists():
                dscore = depth_edge_score(dpath)
                depth_scores[name] = dscore
                if dscore > 0.0:
                    view_scores[name] = max(
                        view_scores.get(name, 0.0),
                        dscore * 0.6,
                    )

    return ConsistencyResult(
        view_scores=view_scores,
        pair_scores=pair_scores,
        depth_scores=depth_scores,
    )
