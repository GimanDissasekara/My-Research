"""Optimization module for AGD mesh refinement."""

from __future__ import annotations

import numpy as np
import trimesh


def refine_mesh(
    mesh: trimesh.Trimesh,
    weights: np.ndarray,
    iterations: int = 10,
    lr: float = 0.15,
    lambda_geo: float = 0.7,
    lambda_distill: float = 0.3,
) -> trimesh.Trimesh:
    if mesh.vertices.shape[0] == 0:
        return mesh

    vertices = mesh.vertices.astype(np.float64)
    anchor = vertices.copy()
    neighbors = mesh.vertex_neighbors

    w = weights.reshape(-1, 1).astype(np.float64)

    for _ in range(iterations):
        lap = _uniform_laplacian(vertices, neighbors)
        distill = vertices - anchor
        step = (lambda_geo * lap) + (lambda_distill * distill)
        vertices = vertices - lr * step * w

    mesh.vertices = vertices
    return mesh


def _uniform_laplacian(vertices: np.ndarray, neighbors: list) -> np.ndarray:
    """Vectorised Laplacian via sparse matrix multiply — fast on large meshes."""
    from scipy.sparse import csr_matrix
    n = vertices.shape[0]
    rows, cols = [], []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            rows.append(i)
            cols.append(j)
    if not rows:
        return np.zeros_like(vertices)
    data = np.ones(len(rows), dtype=np.float64)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    # degree vector
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    # mean of neighbours for each vertex
    mean_nbr = A.dot(vertices) / deg[:, None]
    return vertices - mean_nbr
