"""Refinement optimizers.

This module defines a small, pluggable optimizer interface plus a default
implementation used by the refinement loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import trimesh


class MeshOptimizer(Protocol):
    """Pluggable optimizer interface for mesh refinement.

    The loop calls `reset()` once per new mesh, then `step()` each outer
    iteration with per-vertex weights.
    """

    def reset(self, mesh: trimesh.Trimesh) -> None: ...

    def step(self, mesh: trimesh.Trimesh, weights: np.ndarray, outer_iter: int) -> trimesh.Trimesh: ...


@dataclass
class AnchoredLaplacianOptimizer:
    """Uniform Laplacian smoothing anchored to the initial vertices.

    Update rule (per inner step):
        v <- v - lr * (lambda_geo * Lap(v) + lambda_distill * (v - v_anchor)) * w

    Notes
    -----
    * This mirrors the existing `optimizer.refine_mesh` behaviour but keeps a
      persistent anchor across outer iterations (important for iterative loops).
    * `weights` is expected to be shape (V,) in [0, 1].
    """

    inner_steps: int = 5
    lr: float = 0.15
    lambda_geo: float = 0.7
    lambda_distill: float = 0.3

    _anchor: np.ndarray | None = None

    def reset(self, mesh: trimesh.Trimesh) -> None:
        self._anchor = mesh.vertices.astype(np.float64).copy()

    def step(self, mesh: trimesh.Trimesh, weights: np.ndarray, outer_iter: int) -> trimesh.Trimesh:
        if mesh.vertices.shape[0] == 0:
            return mesh
        if self._anchor is None or self._anchor.shape != mesh.vertices.shape:
            self.reset(mesh)

        vertices = mesh.vertices.astype(np.float64)
        neighbors = mesh.vertex_neighbors

        w = weights.reshape(-1, 1).astype(np.float64)

        for _ in range(int(max(1, self.inner_steps))):
            lap = _uniform_laplacian(vertices, neighbors)
            distill = vertices - self._anchor
            step = (self.lambda_geo * lap) + (self.lambda_distill * distill)
            vertices = vertices - self.lr * step * w

        mesh.vertices = vertices
        return mesh


def _uniform_laplacian(vertices: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    from scipy.sparse import csr_matrix

    n = vertices.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            rows.append(i)
            cols.append(j)

    if not rows:
        return np.zeros_like(vertices)

    data = np.ones(len(rows), dtype=np.float64)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0

    mean_nbr = A.dot(vertices) / deg[:, None]
    return vertices - mean_nbr
