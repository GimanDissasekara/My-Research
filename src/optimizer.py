"""Optimization module for AGD mesh refinement.

Improvements over v1
--------------------
* ``remove_small_components``: drops disconnected floater fragments before
  optimisation — directly fixes meshes stuck at score=0.850 (mba1, Mr_Bean).
* Cotangent Laplacian: area-aware smoothing, more stable on irregular meshes.
* Learning-rate decay: ``lr_t = lr × decay^t`` — prevents oscillation in
  later iterations.
* Convergence early-stop: halts when max vertex displacement < ``lr_tol``.
* Hard displacement clamp: no vertex moves more than ``clamp_fraction × scale``
  per iteration — prevents large jumps on poorly-conditioned meshes.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def remove_small_components(
    mesh: trimesh.Trimesh,
    min_face_fraction: float = 0.001,
) -> trimesh.Trimesh:
    """Remove disconnected mesh components smaller than *min_face_fraction* of total faces.

    This is the most impactful pre-step for meshes with many floating
    fragments (hallucinated geometry).  Components smaller than the threshold
    are dropped entirely before optimisation.

    Parameters
    ----------
    min_face_fraction:
        Minimum fraction of total faces a component must have to be kept.
        Default 0.001 = 0.1 % — tiny floaters are removed while large
        legitimate sub-meshes are preserved.

        Example — mba1.obj has 84 021 faces and 11 141 components:
          threshold = 0.001 × 84 021 ≈ 84 faces
          Only components with ≥ 84 faces are kept.
    """
    total_faces = len(mesh.faces)
    if total_faces == 0:
        return mesh

    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh  # already a single component — nothing to do

    min_faces = max(1, int(min_face_fraction * total_faces))
    kept = [c for c in components if len(c.faces) >= min_faces]

    if not kept:
        # Safety: never return an empty mesh — keep the single largest piece
        kept = [max(components, key=lambda c: len(c.faces))]

    removed = len(components) - len(kept)
    if removed == 0:
        return mesh  # nothing to remove

    logger.info(
        "Floater removal: kept %d / %d components, removed %d small fragments "
        "(threshold = %d faces).",
        len(kept), len(components), removed, min_faces,
    )

    result = trimesh.util.concatenate(kept)
    return result


# ---------------------------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------------------------

def refine_mesh(
    mesh: trimesh.Trimesh,
    weights: np.ndarray,
    iterations: int = 10,
    lr: float = 0.15,
    lambda_geo: float = 0.7,
    lambda_distill: float = 0.3,
    use_cotangent: bool = False,
    lr_decay: float = 0.85,
    lr_tol: float = 1e-6,
    clamp_fraction: float = 0.01,
) -> trimesh.Trimesh:
    """Iteratively refine mesh vertex positions guided by per-vertex weights.

    Each iteration computes:

        step = lambda_geo  × Laplacian(V)     [smooth anomalous vertices]
             + lambda_distill × (V - anchor)  [prevent drift from original]
        delta = clip(current_lr × step × w, -max_disp, max_disp)
        V    = V - delta

    Parameters
    ----------
    weights:
        Per-vertex importance weights in [min_weight, 1.0].
        High weight → vertex is strongly smoothed.
    use_cotangent:
        If True, use the cotangent-weighted Laplacian (better quality on
        irregular meshes, ~3× slower).  Default False (uniform Laplacian).
    lr_decay:
        Multiplicative learning-rate decay per iteration (0 < decay ≤ 1).
        Set to 1.0 to disable decay.  Default 0.85 (15 % decay per step).
    lr_tol:
        Early-stop tolerance.  Stop when max vertex displacement < lr_tol.
        Default 1e-6.
    clamp_fraction:
        Maximum per-vertex displacement per iteration as a fraction of the
        mesh bounding-box diagonal (``mesh.scale``).  Default 0.01 = 1 %.
        Prevents any single vertex from jumping unreasonably far.
    """
    if mesh.vertices.shape[0] == 0:
        return mesh

    vertices  = mesh.vertices.astype(np.float64)
    anchor    = vertices.copy()
    neighbors = mesh.vertex_neighbors

    w        = weights.reshape(-1, 1).astype(np.float64)
    max_disp = clamp_fraction * float(mesh.scale) if mesh.scale > 0 else 1e-3

    current_lr = float(lr)

    for _ in range(iterations):
        if use_cotangent:
            lap = _cotangent_laplacian(vertices, mesh)
        else:
            lap = _uniform_laplacian(vertices, neighbors)

        distill = vertices - anchor
        step    = (lambda_geo * lap) + (lambda_distill * distill)
        delta   = current_lr * step * w

        # Hard displacement clamp — no vertex jumps more than max_disp per step
        delta = np.clip(delta, -max_disp, max_disp)

        vertices   = vertices - delta
        current_lr = current_lr * lr_decay

        # Convergence early-stop
        if float(np.abs(delta).max()) < lr_tol:
            logger.debug("refine_mesh: converged early (max |Δ| < %.2e).", lr_tol)
            break

    mesh.vertices = vertices
    return mesh


# ---------------------------------------------------------------------------
# Laplacian implementations
# ---------------------------------------------------------------------------

def _uniform_laplacian(vertices: np.ndarray, neighbors: list) -> np.ndarray:
    """Vectorised uniform Laplacian via sparse matrix multiply — fast on large meshes."""
    from scipy.sparse import csr_matrix

    n = vertices.shape[0]
    rows: list = []
    cols: list = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            rows.append(i)
            cols.append(j)
    if not rows:
        return np.zeros_like(vertices)

    data = np.ones(len(rows), dtype=np.float64)
    A    = csr_matrix((data, (rows, cols)), shape=(n, n))
    deg  = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    mean_nbr = A.dot(vertices) / deg[:, None]
    return vertices - mean_nbr


def _cotangent_laplacian(vertices: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Cotangent-weighted Laplacian — area-aware, more stable on irregular meshes.

    For each face (i, j, k) the weight on edge (i, j) is cot(angle at k).
    This makes smoothing proportional to local mesh density, preventing
    over-smoothing on dense regions and under-smoothing on coarse ones.

    Cotangent values are clipped to [-10, 10] for numerical safety on
    degenerate or near-degenerate triangles.
    """
    from scipy.sparse import csr_matrix

    V = vertices
    F = mesh.faces
    n = V.shape[0]

    rows: list = []
    cols: list = []
    data: list = []

    # Each row is a cyclic permutation of face vertices: angle at vertex k
    # contributes cotangent weight to edge (i, j)
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        vi = V[F[:, i]]
        vj = V[F[:, j]]
        vk = V[F[:, k]]

        # Edges from vk (the vertex at which we measure the angle)
        e1 = vi - vk
        e2 = vj - vk

        cos_angle = (e1 * e2).sum(axis=1)
        cross_mag = np.linalg.norm(np.cross(e1, e2), axis=1)
        cot = cos_angle / np.maximum(cross_mag, 1e-10)
        cot = np.clip(cot, -10.0, 10.0)  # numerical safety

        # Cotangent weight for edge (i↔j) is cot(angle at k)
        rows.extend(F[:, i].tolist())
        cols.extend(F[:, j].tolist())
        data.extend(cot.tolist())

        rows.extend(F[:, j].tolist())
        cols.extend(F[:, i].tolist())
        data.extend(cot.tolist())

    if not rows:
        return np.zeros_like(V)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    D = np.asarray(W.sum(axis=1)).ravel()
    D[D == 0] = 1.0

    return V - (W @ V) / D[:, None]
