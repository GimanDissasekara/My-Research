"""Geometry Error Metrics for AGD Pipeline (v2).

Metrics
-------
1.  Chamfer Distance (CD)              — bidirectional mean NN distance
2.  Hausdorff Distance (HD / HD95)     — max worst-case deviation
3.  Normal Consistency Score (NCS)     — cosine similarity at matched points
4.  Silhouette IoU                     — per-view mask overlap
5.  Depth RMSE                         — per-view depth-buffer error
6.  Mean Curvature Error (W1)          — Wasserstein-1 curvature distribution
7.  SSIM per view                 [NEW]— structural similarity of RGB renders
8.  Angular Normal Error          [NEW]— mean angle (degrees) between matched normals
9.  Local Surface Roughness       [NEW]— RMS displacement from local tangent plane
10. Multi-view Aggregate Scan     [NEW]— aggregate error across all N views
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------

def _sample_surface(mesh: trimesh.Trimesh, n_points: int = 10_000) -> Tuple[np.ndarray, np.ndarray]:
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        dummy = np.zeros((1, 3), dtype=np.float64)
        return dummy, dummy
    pts, face_ids = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_ids]
    return pts.astype(np.float64), normals.astype(np.float64)


def _nearest_distances(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(tgt)
    dists, _ = tree.query(src, k=1)
    return dists.astype(np.float64)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def chamfer_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                     n_points: int = 8_000) -> Dict[str, float]:
    """Bidirectional Chamfer Distance: CD = mean(A→B) + mean(B→A)."""
    pts_a, _ = _sample_surface(mesh_a, n_points)
    pts_b, _ = _sample_surface(mesh_b, n_points)
    d_ab = _nearest_distances(pts_a, pts_b)
    d_ba = _nearest_distances(pts_b, pts_a)
    cd_fwd, cd_bwd = float(np.mean(d_ab)), float(np.mean(d_ba))
    return {"cd": cd_fwd + cd_bwd, "cd_forward": cd_fwd, "cd_backward": cd_bwd}


def hausdorff_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                       n_points: int = 8_000) -> Dict[str, float]:
    """Hausdorff and 95th-percentile Hausdorff distances."""
    pts_a, _ = _sample_surface(mesh_a, n_points)
    pts_b, _ = _sample_surface(mesh_b, n_points)
    d_ab = _nearest_distances(pts_a, pts_b)
    d_ba = _nearest_distances(pts_b, pts_a)
    hd_fwd, hd_bwd = float(np.max(d_ab)), float(np.max(d_ba))
    hd95 = float(np.percentile(np.concatenate([d_ab, d_ba]), 95))
    return {"hd": max(hd_fwd, hd_bwd), "hd_forward": hd_fwd,
            "hd_backward": hd_bwd, "hd95": hd95}


def normal_consistency_score(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                              n_points: int = 8_000) -> Dict[str, float]:
    """Normal consistency (cosine similarity) at matched surface points."""
    from scipy.spatial import cKDTree
    pts_a, normals_a = _sample_surface(mesh_a, n_points)
    pts_b, normals_b = _sample_surface(mesh_b, n_points)
    tree = cKDTree(pts_b)
    _, idxs = tree.query(pts_a, k=1)
    matched_nb = normals_b[idxs]
    na = normals_a / (np.linalg.norm(normals_a, axis=1, keepdims=True) + 1e-9)
    nb = matched_nb  / (np.linalg.norm(matched_nb,  axis=1, keepdims=True) + 1e-9)
    cosines = np.sum(na * nb, axis=1)
    return {"ncs": float(np.mean(cosines)),
            "ncs_abs": float(np.mean(np.abs(cosines))),
            "ncs_std": float(np.std(cosines))}


def angular_normal_error(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                          n_points: int = 8_000) -> Dict[str, float]:
    """Mean and max angular error (degrees) between matched surface normals.

    Unlike cosine-based NCS, this gives an interpretable degree value
    directly comparable to sensor tolerance specs.
    """
    from scipy.spatial import cKDTree
    pts_a, normals_a = _sample_surface(mesh_a, n_points)
    pts_b, normals_b = _sample_surface(mesh_b, n_points)
    tree = cKDTree(pts_b)
    _, idxs = tree.query(pts_a, k=1)
    matched_nb = normals_b[idxs]
    na = normals_a / (np.linalg.norm(normals_a, axis=1, keepdims=True) + 1e-9)
    nb = matched_nb  / (np.linalg.norm(matched_nb,  axis=1, keepdims=True) + 1e-9)
    cos_clamped = np.clip(np.sum(na * nb, axis=1), -1.0, 1.0)
    angles_deg  = np.degrees(np.arccos(cos_clamped))
    return {
        "angular_mean_deg": float(np.mean(angles_deg)),
        "angular_max_deg":  float(np.max(angles_deg)),
        "angular_p95_deg":  float(np.percentile(angles_deg, 95)),
    }


def local_surface_roughness(mesh: trimesh.Trimesh,
                             n_points: int = 6_000,
                             k_neighbours: int = 8) -> Dict[str, float]:
    """RMS displacement of surface points from their local tangent plane.

    A higher value means the surface has more high-frequency noise /
    jagged geometry — a key hallucination indicator.

    Algorithm
    ---------
    1. Sample n_points on the surface.
    2. For each point, find k nearest neighbours.
    3. Fit a least-squares plane to the neighbourhood.
    4. Measure signed distance of the central point to that plane.
    5. Report RMS, mean-abs, and 95th-percentile of those distances.
    """
    from scipy.spatial import cKDTree
    pts, normals = _sample_surface(mesh, n_points)
    tree = cKDTree(pts)
    _, idxs = tree.query(pts, k=k_neighbours + 1)  # include self

    displacements = np.zeros(n_points, dtype=np.float64)
    for i in range(n_points):
        nbr_pts = pts[idxs[i]]          # (k+1, 3)
        centroid = nbr_pts.mean(axis=0)
        centered = nbr_pts - centroid
        # Least-squares plane normal via SVD
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        plane_normal = Vt[-1]           # smallest singular vector
        plane_normal /= (np.linalg.norm(plane_normal) + 1e-9)
        displacements[i] = abs(float(np.dot(pts[i] - centroid, plane_normal)))

    return {
        "roughness_rms":  float(np.sqrt(np.mean(displacements ** 2))),
        "roughness_mean": float(np.mean(displacements)),
        "roughness_p95":  float(np.percentile(displacements, 95)),
    }


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-Union of two binary silhouette masks → [0, 1]."""
    a = (mask_a > 0).astype(bool)
    b = (mask_b > 0).astype(bool)
    intersection = np.logical_and(a, b).sum()
    union        = np.logical_or(a,  b).sum()
    return 1.0 if union == 0 else float(intersection / union)


def depth_rmse(depth_a: np.ndarray, depth_b: np.ndarray) -> float:
    """RMSE of depth maps at pixels where both are non-zero."""
    a, b = depth_a.astype(np.float64), depth_b.astype(np.float64)
    valid = (a > 0) & (b > 0)
    if not valid.any():
        return 0.0
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Structural Similarity Index (SSIM) between two uint8 grayscale images.

    Uses the standard formula with k1=0.01, k2=0.03, L=255.
    SSIM ∈ [-1, 1]; higher = more structurally similar.

    Falls back to a simplified version if scikit-image is not installed.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        score = sk_ssim(img_a, img_b, data_range=255)
        return float(score)
    except ImportError:
        pass

    # Pure-numpy fallback (simplified SSIM, 11×11 Gaussian window)
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = a.std(); sig_b = b.std()
    sig_ab = float(np.mean((a - mu_a) * (b - mu_b)))
    ssim = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / \
           ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2))
    return float(ssim)


def mean_curvature_error(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                          n_points: int = 4_000) -> Dict[str, float]:
    """Wasserstein-1 distance between vertex mean-curvature distributions."""
    def _curv(mesh: trimesh.Trimesh) -> np.ndarray:
        from scipy.sparse import csr_matrix
        v = mesh.vertices.astype(np.float64)
        n = v.shape[0]
        edges = mesh.edges_unique
        if len(edges) == 0:
            return np.zeros(n)
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        return np.linalg.norm(v - A.dot(v) / deg[:, None], axis=1)

    c_a, c_b = _curv(mesh_a), _curv(mesh_b)
    rng = np.random.default_rng(0)
    c_a_s = np.sort(rng.choice(c_a, min(n_points, len(c_a)), replace=False))
    c_b_s = np.sort(rng.choice(c_b, min(n_points, len(c_b)), replace=False))
    t  = np.linspace(0, 1, 500)
    w1 = float(np.mean(np.abs(np.quantile(c_a_s, t) - np.quantile(c_b_s, t))))
    return {"curvature_w1": w1,
            "curvature_mean_a": float(np.mean(c_a)), "curvature_mean_b": float(np.mean(c_b)),
            "curvature_std_a":  float(np.std(c_a)),  "curvature_std_b":  float(np.std(c_b))}


# ---------------------------------------------------------------------------
# Per-view metric computation (v2 — supports SSIM + edge_map)
# ---------------------------------------------------------------------------

def _load_gray(path: Path) -> Optional[np.ndarray]:
    try:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception as exc:
        logger.warning("Failed loading %s: %s", path, exc)
        return None


def compute_view_metrics(views_before: Dict[str, Path],
                         views_after:  Dict[str, Path]) -> Dict[str, Dict[str, float]]:
    """Compare rendered views before/after.  Supports silhouette, depth, SSIM, edge."""
    view_names: set = set()
    for key in list(views_before.keys()) + list(views_after.keys()):
        for suffix in ("_silhouette", "_depth", "_rgb", "_edge_map"):
            if key.endswith(suffix):
                view_names.add(key[: -len(suffix)])

    results: Dict[str, Dict[str, float]] = {}
    for view in sorted(view_names):
        m: Dict[str, float] = {}

        def _pair(suffix: str):
            pa = views_before.get(f"{view}{suffix}")
            pb = views_after.get(f"{view}{suffix}")
            if pa and pb:
                a, b = _load_gray(pa), _load_gray(pb)
                if a is not None and b is not None and a.shape == b.shape:
                    return a, b
            return None, None

        sil_a, sil_b = _pair("_silhouette")
        if sil_a is not None:
            m["silhouette_iou"] = silhouette_iou(sil_a, sil_b)

        dep_a, dep_b = _pair("_depth")
        if dep_a is not None:
            m["depth_rmse"] = depth_rmse(dep_a, dep_b)

        rgb_a, rgb_b = _pair("_rgb")
        if rgb_a is not None:
            m["ssim"] = compute_ssim(rgb_a, rgb_b)

        edge_a, edge_b = _pair("_edge_map")
        if edge_a is not None:
            # Edge overlap: IoU of edge pixels (measures structural boundary change)
            m["edge_iou"] = silhouette_iou(edge_a, edge_b)

        if m:
            results[view] = m

    return results


def aggregate_scan_metrics(view_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-view metrics across all N views into summary statistics.

    For each metric key present in any view, computes:
      mean, std, min, max, p5 (worst-case lower bound), p95 (worst-case upper bound).

    This is especially useful when n_views=100 to get a holistic quality score.
    """
    if not view_metrics:
        return {}

    # Collect all metric keys
    all_keys: set = set()
    for vm in view_metrics.values():
        all_keys.update(vm.keys())

    aggregated: Dict[str, float] = {}
    for key in sorted(all_keys):
        vals = [vm[key] for vm in view_metrics.values() if key in vm]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        aggregated[f"{key}_mean"] = float(np.mean(arr))
        aggregated[f"{key}_std"]  = float(np.std(arr))
        aggregated[f"{key}_min"]  = float(np.min(arr))
        aggregated[f"{key}_max"]  = float(np.max(arr))
        aggregated[f"{key}_p5"]   = float(np.percentile(arr, 5))
        aggregated[f"{key}_p95"]  = float(np.percentile(arr, 95))

    aggregated["n_views_evaluated"] = float(len(view_metrics))
    return aggregated


# ---------------------------------------------------------------------------
# Composite error report
# ---------------------------------------------------------------------------

@dataclass
class GeometryErrorReport:
    """Complete geometry error report comparing two mesh states (v2)."""
    # Chamfer & Hausdorff
    cd: float = 0.0; cd_forward: float = 0.0; cd_backward: float = 0.0
    hd: float = 0.0; hd95: float = 0.0

    # Normal consistency
    ncs: float = 0.0; ncs_abs: float = 0.0

    # Angular error (degrees)
    angular_mean_deg: float = 0.0
    angular_max_deg:  float = 0.0
    angular_p95_deg:  float = 0.0

    # Local surface roughness (before mesh)
    roughness_rms: float = 0.0

    # Curvature
    curvature_w1: float = 0.0
    curvature_mean_before: float = 0.0
    curvature_mean_after:  float = 0.0

    # Per-view aggregates (mean across views)
    mean_silhouette_iou: float = 0.0
    mean_depth_rmse:     float = 0.0
    mean_ssim:           float = 0.0
    mean_edge_iou:       float = 0.0

    # Raw per-view + scan aggregate
    view_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scan_aggregate: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        d = {
            "chamfer_distance":     self.cd,
            "chamfer_forward":      self.cd_forward,
            "chamfer_backward":     self.cd_backward,
            "hausdorff_distance":   self.hd,
            "hausdorff_95pct":      self.hd95,
            "normal_consistency":   self.ncs,
            "normal_consistency_abs": self.ncs_abs,
            "angular_mean_deg":     self.angular_mean_deg,
            "angular_p95_deg":      self.angular_p95_deg,
            "roughness_rms":        self.roughness_rms,
            "curvature_wasserstein": self.curvature_w1,
            "curvature_mean_before": self.curvature_mean_before,
            "curvature_mean_after":  self.curvature_mean_after,
            "mean_silhouette_iou":   self.mean_silhouette_iou,
            "mean_depth_rmse":       self.mean_depth_rmse,
            "mean_ssim":             self.mean_ssim,
            "mean_edge_iou":         self.mean_edge_iou,
        }
        d.update(self.scan_aggregate)
        return d

    def summary_str(self) -> str:
        n_views = int(self.scan_aggregate.get("n_views_evaluated", len(self.view_metrics)))
        lines = [
            "  ── Geometry Error Metrics (before vs. after) ──",
            f"  Chamfer Distance    (↓ better): {self.cd:.6f}  [fwd {self.cd_forward:.6f}, bwd {self.cd_backward:.6f}]",
            f"  Hausdorff Distance  (↓ better): {self.hd:.6f}  [HD95 {self.hd95:.6f}]",
            f"  Normal Consistency  (↑ better): {self.ncs:.4f}  [|cos| {self.ncs_abs:.4f}]",
            f"  Angular Normal Err  (↓ better): mean={self.angular_mean_deg:.2f}°  p95={self.angular_p95_deg:.2f}°  max={self.angular_max_deg:.2f}°",
            f"  Surface Roughness   (↓ better): RMS={self.roughness_rms:.6f}",
            f"  Curvature W1        (↓ better): {self.curvature_w1:.6f}",
            f"    curvature before={self.curvature_mean_before:.6f}  after={self.curvature_mean_after:.6f}",
        ]
        if self.view_metrics:
            lines += [
                f"  ── Multi-view scan ({n_views} views) ──",
                f"  Mean Silhouette IoU (↑ better): {self.mean_silhouette_iou:.4f}",
                f"  Mean Depth RMSE     (↓ better): {self.mean_depth_rmse:.4f}",
                f"  Mean SSIM           (↑ better): {self.mean_ssim:.4f}",
                f"  Mean Edge IoU       (↑ better): {self.mean_edge_iou:.4f}",
            ]
            if self.scan_aggregate:
                lines.append("  Scan aggregate (worst-case views):")
                for key in ("silhouette_iou_min", "depth_rmse_max", "ssim_min", "edge_iou_min"):
                    if key in self.scan_aggregate:
                        lines.append(f"    {key}: {self.scan_aggregate[key]:.4f}")
        return "\n".join(lines)


def compute_geometry_error_report(
    mesh_before: trimesh.Trimesh,
    mesh_after:  trimesh.Trimesh,
    views_before: Optional[Dict[str, Path]] = None,
    views_after:  Optional[Dict[str, Path]] = None,
    n_surface_points: int = 6_000,
    compute_roughness: bool = True,
) -> GeometryErrorReport:
    """Compute full geometry error report (v2) with all metrics.

    New vs v1
    ----------
    * Angular normal error in degrees.
    * Local surface roughness of the before-mesh.
    * SSIM and edge-IoU in per-view metrics.
    * scan_aggregate: min/max/std/p5/p95 across all views for dense scans.
    """
    report = GeometryErrorReport()

    # Chamfer
    try:
        r = chamfer_distance(mesh_before, mesh_after, n_points=n_surface_points)
        report.cd = r["cd"]; report.cd_forward = r["cd_forward"]; report.cd_backward = r["cd_backward"]
    except Exception as e:
        logger.warning("Chamfer failed: %s", e)

    # Hausdorff
    try:
        r = hausdorff_distance(mesh_before, mesh_after, n_points=n_surface_points)
        report.hd = r["hd"]; report.hd95 = r["hd95"]
    except Exception as e:
        logger.warning("Hausdorff failed: %s", e)

    # Normal consistency
    try:
        r = normal_consistency_score(mesh_before, mesh_after, n_points=n_surface_points)
        report.ncs = r["ncs"]; report.ncs_abs = r["ncs_abs"]
    except Exception as e:
        logger.warning("NCS failed: %s", e)

    # Angular normal error
    try:
        r = angular_normal_error(mesh_before, mesh_after, n_points=n_surface_points)
        report.angular_mean_deg = r["angular_mean_deg"]
        report.angular_max_deg  = r["angular_max_deg"]
        report.angular_p95_deg  = r["angular_p95_deg"]
    except Exception as e:
        logger.warning("Angular error failed: %s", e)

    # Surface roughness
    if compute_roughness:
        try:
            r = local_surface_roughness(mesh_before, n_points=min(n_surface_points, 4_000))
            report.roughness_rms = r["roughness_rms"]
        except Exception as e:
            logger.warning("Roughness failed: %s", e)

    # Curvature
    try:
        r = mean_curvature_error(mesh_before, mesh_after)
        report.curvature_w1           = r["curvature_w1"]
        report.curvature_mean_before  = r["curvature_mean_a"]
        report.curvature_mean_after   = r["curvature_mean_b"]
    except Exception as e:
        logger.warning("Curvature failed: %s", e)

    # Per-view metrics
    if views_before is not None and views_after is not None:
        try:
            vm = compute_view_metrics(views_before, views_after)
            report.view_metrics   = vm
            report.scan_aggregate = aggregate_scan_metrics(vm)
            if vm:
                def _mean(key):
                    vals = [v[key] for v in vm.values() if key in v]
                    return float(np.mean(vals)) if vals else 0.0
                report.mean_silhouette_iou = _mean("silhouette_iou")
                report.mean_depth_rmse     = _mean("depth_rmse")
                report.mean_ssim           = _mean("ssim")
                report.mean_edge_iou       = _mean("edge_iou")
        except Exception as e:
            logger.warning("View metrics failed: %s", e)

    return report
