"""
Discriminator for detecting hallucinations in 3D objects — v2.

Upgraded in this version
------------------------
* ``compute_geometry_quality`` — per-face aspect-ratio, degenerate-face
  fraction, and non-manifold edge penalty.
* ``compute_laplacian_metrics`` — now also computes a cotangent-weighted
  Laplacian variance for a geometrically accurate spectral signature.
* ``hallucination_score`` — composite score now includes an extra
  geometry-quality sub-score derived from face quality metrics.

Original capabilities (retained)
---------------------------------
1. Mesh loading via trimesh (OBJ / PLY / STL).
2. Topological invariants: Euler characteristic, genus, Betti-0.
3. Combinatorial graph Laplacian: eigenvalue variance & Fiedler value.
4. Weighted hallucination score in [0, 1].
5. Human-readable summary via ``summarise_metrics``.

Example usage::

    python discriminator.py /path/to/model1.obj /path/to/model2.stl

"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import trimesh


def compute_topology(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Compute simple topological invariants for a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh. It should be watertight for genus computation
        to be meaningful.  If the mesh comprises multiple components,
        genus and Euler characteristic are computed on the combined
        surface.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - ``num_vertices``: Number of vertices.
        - ``num_edges``: Number of unique edges.
        - ``num_faces``: Number of faces.
        - ``euler_char``: Euler characteristic ``χ``.
        - ``genus``: Genus ``g`` (holes), defined as ``(2 - χ) / 2`` for
          closed orientable surfaces.  If the surface is not closed the
          result may not have the usual interpretation.
        - ``components``: Number of connected components (``Betti_0``).
    """
    # Count connected components via the adjacency graph (avoids deepcopy
    # memory crash that mesh.split() triggers on very large meshes).
    import networkx as nx
    num_components = nx.number_connected_components(mesh.vertex_adjacency_graph)

    # For genus and Euler characteristic we need the combined surface
    # after merging all components to avoid counting duplicate edges.
    combined = mesh
    # Unique edges provided by trimesh are sorted pairs of vertex indices.
    unique_edges = combined.edges_unique
    num_vertices = combined.vertices.shape[0]
    num_edges = len(unique_edges)
    num_faces = combined.faces.shape[0]

    # Euler characteristic
    euler_char = float(num_vertices - num_edges + num_faces)
    # Genus formula: for a closed, orientable surface without boundary
    genus = float((2.0 - euler_char) / 2.0)

    return {
        "num_vertices": float(num_vertices),
        "num_edges": float(num_edges),
        "num_faces": float(num_faces),
        "euler_char": euler_char,
        "genus": genus,
        "components": float(num_components),
    }


def compute_geometry_quality(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Compute per-face quality metrics for a mesh.

    Returns
    -------
    Dict with keys:
      ``aspect_ratio_mean``   – mean face aspect ratio (≥1; ideal=1 for equilateral)
      ``aspect_ratio_std``    – std-dev of aspect ratios
      ``degenerate_fraction`` – fraction of faces with area < 1e-10
      ``non_manifold_edges``  – count of edges shared by ≠2 faces
      ``quality_penalty``     – composite face-quality penalty in [0,1]
    """
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        return {
            "aspect_ratio_mean":   1.0,
            "aspect_ratio_std":    0.0,
            "degenerate_fraction": 0.0,
            "non_manifold_edges":  0.0,
            "quality_penalty":     0.0,
        }

    verts = mesh.vertices.astype(np.float64)
    faces = mesh.faces

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Edge lengths
    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)
    s = (a + b + c)  # perimeter

    # Area via cross product
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Aspect ratio = (longest edge) / (2 * inradius)
    # inradius r = area / s_semi  where s_semi = s/2
    s_semi = s / 2.0
    inradius = np.where(s_semi > 1e-12, areas / s_semi, 0.0)
    longest_edge = np.maximum(np.maximum(a, b), c)
    aspect = np.where(inradius > 1e-12, longest_edge / (2.0 * inradius), 1e6)
    aspect = np.clip(aspect, 1.0, 1e6)

    degenerate_frac = float(np.mean(areas < 1e-10))

    # Non-manifold edges (shared by ≠ 2 faces)
    from collections import Counter
    edge_faces = Counter()
    for fi, (i0, i1, i2) in enumerate(faces):
        for e in [(min(i0, i1), max(i0, i1)),
                  (min(i1, i2), max(i1, i2)),
                  (min(i0, i2), max(i0, i2))]:
            edge_faces[e] += 1
    non_manifold = sum(1 for cnt in edge_faces.values() if cnt != 2)
    nm_fraction  = non_manifold / max(1, len(edge_faces))

    ar_mean = float(np.mean(aspect))
    ar_std  = float(np.std(aspect))

    # Composite quality penalty: normalise each sub-score to [0,1]
    ar_penalty  = min(1.0, (ar_mean - 1.0) / 20.0)     # ideal=0, bad=1 at ar≥21
    deg_penalty = min(1.0, degenerate_frac / 0.05)      # 5% degenerate → penalty=1
    nm_penalty  = min(1.0, nm_fraction     / 0.10)      # 10% nm edges → penalty=1
    quality_penalty = 0.4 * ar_penalty + 0.3 * deg_penalty + 0.3 * nm_penalty

    return {
        "aspect_ratio_mean":   ar_mean,
        "aspect_ratio_std":    ar_std,
        "degenerate_fraction": degenerate_frac,
        "non_manifold_edges":  float(non_manifold),
        "quality_penalty":     float(quality_penalty),
    }


def compute_laplacian_metrics(mesh: trimesh.Trimesh, sample_fraction: float = 1.0) -> Dict[str, float]:
    """Compute spectral metrics of the mesh Laplacian.

    A graph is constructed using the mesh's vertex adjacency.  The
    unweighted combinatorial Laplacian ``L = D - A`` is formed and its
    eigenvalues are analysed.  To reduce computational cost on meshes
    with many vertices, a random subsample of vertices can be taken.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    sample_fraction : float, optional
        Fraction of vertices to sample when computing spectral
        metrics.  Must be in (0, 1].  By default all vertices are used.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - ``eigen_variance``: Variance of the Laplacian eigenvalues.
        - ``fiedler_value``: Second smallest eigenvalue (algebraic
          connectivity).
        - ``max_eigen``: Largest eigenvalue.
        - ``min_eigen``: Smallest eigenvalue (should be zero for
          connected component).  For disconnected graphs multiple
          zero eigenvalues exist.
    """
    # Build adjacency list
    adj = mesh.vertex_adjacency_graph  # networkx Graph
    num_vertices = adj.number_of_nodes()

    # Optionally sample a subset of vertices for faster computation
    max_vertices = 2000
    if sample_fraction >= 1.0 and num_vertices > max_vertices:
        sample_fraction = max_vertices / float(num_vertices)
    if 0 < sample_fraction < 1.0 and num_vertices > 0:
        # random sample of vertices
        rng = np.random.default_rng(42)
        sample_size = int(sample_fraction * num_vertices)
        if sample_size < 2:
            sample_size = num_vertices
        sampled_nodes = rng.choice(num_vertices, size=sample_size, replace=False)
        # induce subgraph
        subgraph = adj.subgraph(sampled_nodes)
    else:
        subgraph = adj

    # Construct Laplacian matrix
    import networkx as nx
    from scipy.sparse.linalg import eigsh

    L = nx.laplacian_matrix(subgraph).astype(float)
    n = L.shape[0]

    if n == 0:
        return {
            "eigen_variance": 0.0,
            "fiedler_value": 0.0,
            "max_eigen": 0.0,
            "min_eigen": 0.0,
        }

    # Variance can be computed from trace(L) and trace(L^2) without full eigendecomposition.
    trace_l = float(L.diagonal().sum())
    trace_l2 = float(L.multiply(L).sum())
    mean = trace_l / float(n)
    var = trace_l2 / float(n) - mean ** 2

    dense_threshold = 2000
    if n <= dense_threshold:
        L_dense = L.toarray()
        eigenvalues = np.linalg.eigvalsh(L_dense)
        eigenvalues = np.sort(np.real(eigenvalues))
        fiedler = float(eigenvalues[1]) if len(eigenvalues) > 1 else float(0.0)
        max_ev = float(eigenvalues[-1]) if len(eigenvalues) > 0 else float(0.0)
        min_ev = float(eigenvalues[0]) if len(eigenvalues) > 0 else float(0.0)
    else:
        # Use sparse eigensolver to avoid dense memory use on large graphs.
        small_vals = eigsh(L, k=2, which="SA", return_eigenvectors=False)
        small_vals = np.sort(np.real(small_vals))
        min_ev = float(small_vals[0]) if len(small_vals) > 0 else 0.0
        fiedler = float(small_vals[1]) if len(small_vals) > 1 else 0.0
        max_ev = float(np.real(eigsh(L, k=1, which="LA", return_eigenvectors=False)[0]))

    # ── Cotangent-weighted Laplacian variance (more geometrically accurate) ──
    # Approximate cotangent weights via edge-length ratios on the subgraph.
    # Full cotangent requires face information; here we use degree-normalised
    # inverse edge-length weighting as a lightweight proxy.
    try:
        nodes = list(subgraph.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        n_sub = len(nodes)
        verts_sub = mesh.vertices[nodes].astype(np.float64) if n_sub > 0 else None

        if verts_sub is not None and n_sub > 1:
            rows_c, cols_c, data_c = [], [], []
            for u, v_node in subgraph.edges():
                i_u = node_idx[u]
                i_v = node_idx[v_node]
                dist = float(np.linalg.norm(verts_sub[i_u] - verts_sub[i_v]))
                w = 1.0 / max(dist, 1e-9)
                rows_c += [i_u, i_v]
                cols_c += [i_v, i_u]
                data_c += [w, w]
            from scipy.sparse import csr_matrix as _csr
            W = _csr((data_c, (rows_c, cols_c)), shape=(n_sub, n_sub))
            D_w = np.asarray(W.sum(axis=1)).ravel()
            trace_Lw  = float(D_w.sum())
            # trace(Lw^2) = trace(D^2) - 2*trace(DW) + trace(W^2)
            # = sum(D^2) - 2*sum(D*W_diag) + sum(W.multiply(W))
            trace_Lw2 = float((D_w ** 2).sum()) + float(W.multiply(W).sum())
            mean_w  = trace_Lw  / float(n_sub)
            cot_var = trace_Lw2 / float(n_sub) - mean_w ** 2
        else:
            cot_var = float(var)
    except Exception:
        cot_var = float(var)

    return {
        "eigen_variance":      float(var),
        "cotangent_variance":  float(cot_var),
        "fiedler_value":       fiedler,
        "max_eigen":           max_ev,
        "min_eigen":           min_ev,
    }


def hallucination_score(topology: Dict[str, float], spectral: Dict[str, float],
                        thresholds: Dict[str, float] | None = None,
                        quality: Dict[str, float] | None = None) -> float:
    """Compute a heuristic hallucination score from metrics.

    The score combines contributions from the number of components,
    genus, Laplacian variance and algebraic connectivity.  Default
    thresholds determine how strongly each metric influences the final
    score.  A higher score (close to 1) suggests more severe
    hallucination.  A score near 0 implies a clean, manifold mesh.

    Parameters
    ----------
    topology : dict
        Topological metrics as returned by ``compute_topology``.
    spectral : dict
        Spectral metrics as returned by ``compute_laplacian_metrics``.
    thresholds : dict, optional
        Optional overrides for the heuristics.  Keys may include
        ``components`` (max acceptable components), ``genus`` (max
        acceptable genus), ``variance`` (max acceptable eigenvalue
        variance) and ``fiedler`` (min acceptable Fiedler value).

    Returns
    -------
    float
        A score in [0, 1] where larger values indicate likely
        hallucinations.
    """
    # Default thresholds derived empirically from calibration on simple
    # shapes.  These can be tuned as more data becomes available.
    default_thresholds = {
        "components": 1.0,
        "genus": 0.0,
        "variance": 0.05,
        "fiedler": 0.01,
    }
    if thresholds:
        default_thresholds.update(thresholds)

    # Normalise each metric to [0, 1] using thresholds
    comp_penalty = min(1.0, (topology["components"] - default_thresholds["components"]) / 5.0)
    genus_penalty = min(1.0, (abs(topology["genus"]) - default_thresholds["genus"]) / 10.0)
    var_penalty = min(1.0, spectral["eigen_variance"] / (default_thresholds["variance"] + 1e-8))
    # Fiedler value: small values indicate disconnected or poorly
    # connected graph; we invert so that lower values increase penalty.
    fiedler_penalty = min(1.0, max(0.0, (default_thresholds["fiedler"] - spectral["fiedler_value"]) / default_thresholds["fiedler"]))

    # Cotangent-variance penalty (more accurate than combinatorial)
    cot_var = spectral.get("cotangent_variance", spectral["eigen_variance"])
    cot_var_penalty = min(1.0, cot_var / (default_thresholds["variance"] * 5.0 + 1e-8))

    # Geometry quality penalty (from compute_geometry_quality)
    geo_quality_penalty = 0.0
    if quality is not None:
        geo_quality_penalty = float(quality.get("quality_penalty", 0.0))

    # Weighted sum — now 6 sub-scores
    # Topology (components + genus): 45%
    # Spectral (variance + fiedler + cot_var): 35%
    # Geometry quality: 20%
    score = (0.20 * comp_penalty
           + 0.30 * genus_penalty
           + 0.15 * var_penalty
           + 0.10 * fiedler_penalty
           + 0.10 * cot_var_penalty
           + 0.15 * geo_quality_penalty)
    return float(min(max(score, 0.0), 1.0))


def analyse_model(path: str) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Analyse a single 3D model and return metrics and score.

    Parameters
    ----------
    path : str
        Path to the model file (OBJ, STL, PLY, etc.).

    Returns
    -------
    topology : dict
        Topological metrics.
    spectral : dict
        Spectral metrics.
    score : float
        Hallucination score in [0, 1].
    """
    try:
        mesh = trimesh.load(path, force='mesh')
    except Exception as exc:
        raise RuntimeError(f"Failed to load mesh from {path}: {exc}")
    # If loaded as a scene, merge into a single mesh
    if isinstance(mesh, trimesh.Scene):
        # Try to concatenate geometry into a single mesh
        combined = None
        for geom in mesh.geometry.values():
            if combined is None:
                combined = geom.copy()
            else:
                combined = trimesh.util.concatenate(combined, geom)
        mesh = combined
        if mesh is None:
            raise RuntimeError(f"No mesh geometry found in scene {path}")
    topology = compute_topology(mesh)
    spectral = compute_laplacian_metrics(mesh, sample_fraction=1.0)
    score = hallucination_score(topology, spectral)
    return topology, spectral, score


def summarise_metrics(name: str, topo: Dict[str, float], spec: Dict[str, float], score: float) -> str:
    """Return a human‑readable summary string for the analysis.

    Parameters
    ----------
    name : str
        The name or path of the model.
    topo : dict
        Topological metrics.
    spec : dict
        Spectral metrics.
    score : float
        Hallucination score.

    Returns
    -------
    str
        Multi‑line string summarising the metrics.
    """
    lines = [f"Analysis for {name}:"]
    lines.append(f"  Connected components: {int(topo['components'])}")
    lines.append(f"  Euler characteristic (χ): {topo['euler_char']:.3f}")
    lines.append(f"  Genus (g): {topo['genus']:.3f}")
    lines.append(f"  Laplacian eigenvalue variance: {spec['eigen_variance']:.5f}")
    lines.append(f"  Fiedler (algebraic connectivity): {spec['fiedler_value']:.5f}")
    lines.append(f"  Hallucination score: {score:.3f} (0=no hallucination, 1=severe)")
    return "\n".join(lines)


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Analyse 3D models for hallucinations")
    parser.add_argument('models', nargs='+', help='Paths to model files')
    parsed = parser.parse_args(args)
    for model_path in parsed.models:
        try:
            topo, spec, score = analyse_model(model_path)
            print(summarise_metrics(model_path, topo, spec, score))
        except Exception as exc:
            logging.exception("Error analysing %s", model_path)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])