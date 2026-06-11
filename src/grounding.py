"""Grounding module for mapping critic scores to spatial weights."""

from __future__ import annotations

import numpy as np
import trimesh


def build_region_weights(
    mesh: trimesh.Trimesh,
    scores: np.ndarray,
    top_fraction: float = 0.1,
    min_weight: float = 0.1,
    max_weight: float = 1.0,
    expand_hops: int = 1,
    global_bias: float = 0.0,
    bias_weights: np.ndarray | None = None,
) -> np.ndarray:
    num_vertices = mesh.vertices.shape[0]
    if num_vertices == 0:
        return np.zeros((0,), dtype=float)

    scores = np.clip(scores, 0.0, 1.0)
    k = max(1, int(top_fraction * num_vertices))
    threshold = np.partition(scores, -k)[-k]
    mask = scores >= threshold

    if expand_hops > 0:
        neighbors = mesh.vertex_neighbors
        mask = _expand_mask(mask, neighbors, expand_hops)

    weights = np.full((num_vertices,), min_weight, dtype=float)
    if mask.any():
        scaled = (scores - scores.min())
        if scaled.max() > 0:
            scaled /= scaled.max()
        weights[mask] = min_weight + scaled[mask] * (max_weight - min_weight)
    if global_bias > 0.0:
        weights = np.clip(weights + global_bias * (max_weight - min_weight), min_weight, max_weight)
    if bias_weights is not None and bias_weights.size:
        weights = np.maximum(weights, min_weight + bias_weights * (max_weight - min_weight))
    return weights


def map_views_to_vertex_bias(
    mesh: trimesh.Trimesh,
    view_scores: dict[str, float],
    view_extent: float = 0.3,
    normal_threshold: float = 0.2,
) -> np.ndarray:
    num_vertices = mesh.vertices.shape[0]
    if num_vertices == 0:
        return np.zeros((0,), dtype=float)

    directions = {
        "front": np.array([0.0, 0.0, 1.0]),
        "back": np.array([0.0, 0.0, -1.0]),
        "left": np.array([-1.0, 0.0, 0.0]),
        "right": np.array([1.0, 0.0, 0.0]),
        "top": np.array([0.0, 1.0, 0.0]),
        "bottom": np.array([0.0, -1.0, 0.0]),
    }

    center = mesh.bounds.mean(axis=0)
    normals = mesh.vertex_normals
    bias = np.zeros((num_vertices,), dtype=float)

    for view, score in view_scores.items():
        if score <= 0.0:
            continue
        if view not in directions:
            continue
        direction = directions[view]
        projection = (mesh.vertices - center) @ direction
        threshold = np.quantile(projection, 1.0 - view_extent)
        facing = (normals @ direction) >= normal_threshold
        mask = (projection >= threshold) & facing
        bias[mask] = np.maximum(bias[mask], score)

    return np.clip(bias, 0.0, 1.0)


def _expand_mask(mask: np.ndarray, neighbors: list[list[int]], hops: int) -> np.ndarray:
    current = mask.copy()
    for _ in range(hops):
        expanded = current.copy()
        idxs = np.where(current)[0]
        for idx in idxs:
            for n in neighbors[idx]:
                expanded[n] = True
        current = expanded
    return current
