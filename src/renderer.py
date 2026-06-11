"""Renderer utilities for multi-view 3D mesh sampling.

Produces per-view PNG images using a pure-numpy software rasteriser that
runs without a GPU or display server.

Upgrade v2
----------
* **Arbitrary view count** — pass ``n_views`` to sample camera positions
  mathematically via Fibonacci-sphere (uniform) or azimuth-elevation grid.
* **Phong shading** — ambient + diffuse + specular highlight per face.
* **Per-pixel interpolated normals** — smoother normal maps using
  barycentric-weighted vertex normals.
* **Curvature / edge buffer** — detect sharp-angle discontinuities across
  rendered triangles for error localisation.
* **Perspective-correct depth** — linear-in-camera-space depth interpolation
  using the standard 1/z trick.
* Backward-compatible: `render_views` still accepts the classic 6-view
  ``CANONICAL_VIEWS`` dict; pass ``n_views`` for dense scanning.

For every camera position the renderer saves:
  - ``<idx>_rgb.png``         – Phong-shaded RGB render
  - ``<idx>_depth.png``       – perspective-correct normalised depth (grayscale)
  - ``<idx>_normal.png``      – interpolated normal map (RGB)
  - ``<idx>_silhouette.png``  – binary silhouette mask (BW)
  - ``<idx>_edge.png``        – curvature / edge detection map (BW)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical 6-view dictionary (kept for backward compatibility)
# ---------------------------------------------------------------------------
CANONICAL_VIEWS: Dict[str, Tuple[float, float]] = {
    "front":  (  0.0,   0.0),
    "back":   (180.0,   0.0),
    "left":   ( 90.0,   0.0),
    "right":  (270.0,   0.0),
    "top":    (  0.0,  90.0),
    "bottom": (  0.0, -90.0),
}


# ---------------------------------------------------------------------------
# Mathematically-defined camera sampling
# ---------------------------------------------------------------------------

def fibonacci_sphere_views(n: int) -> List[Tuple[float, float]]:
    """Generate *n* near-uniformly distributed (azimuth_deg, elevation_deg)
    pairs using the Fibonacci / golden-angle spiral on the unit sphere.

    The golden-angle method gives the best-known equal-area distribution on
    a sphere, with no clustering at the poles unlike a regular lat-lon grid.

    Math
    ----
    For sample index i in [0, n):
        phi   = arccos(1 - 2*(i+0.5)/n)        # inclination from north pole
        theta = 2*pi * i * golden_ratio_conjugate

    Returns list of (azimuth_deg, elevation_deg) where elevation ∈ [-90, 90].
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0  # ~1.618...
    views: List[Tuple[float, float]] = []
    for i in range(n):
        phi   = math.acos(1.0 - 2.0 * (i + 0.5) / n)   # [0, pi]
        theta = 2.0 * math.pi * i / golden               # azimuth (radians)
        az_deg = math.degrees(theta) % 360.0
        el_deg = 90.0 - math.degrees(phi)                # elevation [-90, 90]
        views.append((az_deg, el_deg))
    return views


def azimuth_elevation_grid(n_az: int = 12, n_el: int = 7) -> List[Tuple[float, float]]:
    """Regular azimuth-elevation grid of n_az × n_el views.

    Elevation range: [-75°, +75°] to avoid degenerate pole views.
    Total views = n_az * n_el (default 84).

    Parameters
    ----------
    n_az : Number of azimuth steps (longitude slices).
    n_el : Number of elevation steps (latitude bands).
    """
    azimuths   = np.linspace(0.0, 360.0, n_az, endpoint=False)
    elevations = np.linspace(-75.0, 75.0, n_el)
    views: List[Tuple[float, float]] = []
    for el in elevations:
        for az in azimuths:
            views.append((float(az), float(el)))
    return views


def build_view_schedule(
    n_views: int = 100,
    method: str = "fibonacci",
) -> List[Tuple[str, float, float]]:
    """Return a list of (name, azimuth_deg, elevation_deg) for rendering.

    Parameters
    ----------
    n_views : Target number of views.
    method  : ``"fibonacci"`` (recommended – uniform distribution) or
              ``"grid"`` (regular azimuth-elevation grid).

    Returns
    -------
    List of (name, az_deg, el_deg).  Names are zero-padded integers, e.g.
    ``"view_000"``, ``"view_001"``, …
    """
    if method == "fibonacci":
        raw = fibonacci_sphere_views(n_views)
    elif method == "grid":
        # Fit az/el counts so total is approximately n_views
        n_az = max(4, round(math.sqrt(n_views * 2.0)))
        n_el = max(2, round(n_views / n_az))
        raw = azimuth_elevation_grid(n_az, n_el)
    else:
        raise ValueError(f"Unknown view schedule method '{method}'")

    pad = len(str(len(raw)))
    return [(f"view_{str(i).zfill(pad)}", az, el) for i, (az, el) in enumerate(raw)]


# ---------------------------------------------------------------------------
# Camera / projection math
# ---------------------------------------------------------------------------

def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return 3×3 camera rotation matrix for given azimuth and elevation (°).

    Convention:
      * azimuth  — rotate around world-Y axis (yaw)
      * elevation — rotate around camera-X axis after yaw (pitch)
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    Ry = np.array([
        [ np.cos(az), 0, np.sin(az)],
        [          0, 1,           0],
        [-np.sin(az), 0, np.cos(az)],
    ], dtype=np.float64)
    Rx = np.array([
        [1,          0,           0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el),  np.cos(el)],
    ], dtype=np.float64)
    return Rx @ Ry


def _project(
    vertices: np.ndarray,   # (V, 3) world-space
    R: np.ndarray,           # (3, 3) rotation
    img_size: int,
    fov_deg: float = 45.0,
    camera_dist: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perspective projection with perspective-correct depth.

    The mesh is translated to sit at ``-camera_dist`` along camera-Z before
    projection, ensuring consistent field-of-view at all angles.

    Returns
    -------
    px, py  : (V,) float pixel coordinates
    z_cam   : (V,) camera-space Z (positive = in front of camera)
    """
    cam = (R @ vertices.T).T              # (V, 3) — rotate into camera space
    cam[:, 2] -= camera_dist             # translate to sit in front of camera

    z_cam = -cam[:, 2]                   # flip: positive means in front
    f     = (img_size / 2.0) / np.tan(np.radians(fov_deg / 2.0))

    depth = np.where(np.abs(cam[:, 2]) < 1e-8, 1e-8, cam[:, 2])
    px = ( f * cam[:, 0] / (-depth) + img_size / 2.0)
    py = (-f * cam[:, 1] / (-depth) + img_size / 2.0)
    return px, py, z_cam


# ---------------------------------------------------------------------------
# Phong shading helpers
# ---------------------------------------------------------------------------

def _phong_shade(
    fn_cam: np.ndarray,          # (3,) face normal in camera space
    light_dir: np.ndarray,       # (3,) normalised light direction
    view_dir:  np.ndarray,       # (3,) normalised view direction (towards cam)
    base_color: np.ndarray,      # (3,) RGB in [0,1]
    ambient: float  = 0.18,
    diffuse: float  = 0.72,
    specular: float = 0.25,
    shininess: float = 32.0,
) -> np.ndarray:
    """Phong reflection model for a single face.

    Returns shaded RGB (3,) in [0, 1].
    """
    n = fn_cam / (np.linalg.norm(fn_cam) + 1e-9)
    # Diffuse
    diff = max(0.0, float(np.dot(n, light_dir)))
    # Specular (Blinn-Phong halfway vector)
    half = (light_dir + view_dir)
    half_norm = np.linalg.norm(half)
    spec = 0.0
    if half_norm > 1e-9:
        h = half / half_norm
        spec = max(0.0, float(np.dot(n, h))) ** shininess

    shade = ambient + diffuse * diff + specular * spec
    return np.clip(base_color * shade + specular * spec * 0.4, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Vectorised software rasteriser (v2)
# ---------------------------------------------------------------------------

def _rasterise(
    vertices:     np.ndarray,       # (V, 3) normalised world-space
    faces:        np.ndarray,       # (F, 3) index array
    face_normals: np.ndarray,       # (F, 3) unit face normals
    vertex_normals: np.ndarray,     # (V, 3) unit vertex normals for interpolation
    R:            np.ndarray,       # (3, 3) camera rotation
    img_size:     int,
    fov_deg:      float = 45.0,
    camera_dist:  float = 3.0,
    light_dir:    Optional[np.ndarray] = None,
    edge_threshold: float = 0.35,   # cos(angle) threshold for edge detection
) -> Dict[str, np.ndarray]:
    """Software rasterise mesh into RGB/depth/normal/silhouette/edge buffers.

    Improvements over v1
    --------------------
    * Phong shading (ambient + diffuse + specular).
    * Perspective-correct depth (interpolate 1/z, not z).
    * Per-pixel interpolated normals using barycentric vertex normals.
    * Edge / curvature detection buffer based on adjacent face normal angles.

    Returns
    -------
    Dict with uint8 arrays:
      rgb        : (H, W, 3)
      depth      : (H, W)     – 0..255 perspective-correct normalised depth
      normal_map : (H, W, 3)  – interpolated normals
      silhouette : (H, W)     – 0 or 255
      edge_map   : (H, W)     – 0 or 255  (sharp discontinuity detector)
    """
    if light_dir is None:
        light_dir = np.array([0.4, 0.7, 1.0], dtype=np.float64)
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-9)

    # View direction (towards camera, in camera space = +Z)
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    H = W = img_size
    rgb_buf   = np.zeros((H, W, 3), dtype=np.float32)
    # Store 1/z for perspective-correct depth
    inv_z_buf = np.zeros((H, W), dtype=np.float64)
    norm_buf  = np.zeros((H, W, 3), dtype=np.float32)
    face_id_buf = np.full((H, W), -1, dtype=np.int32)  # for edge detection

    px, py, z_cam = _project(vertices, R, img_size, fov_deg, camera_dist)

    # Rotate vertex normals into camera space
    vn_cam = (R @ vertex_normals.T).T   # (V, 3)

    for fi in range(len(faces)):
        i0, i1, i2 = faces[fi]

        x0, y0 = px[i0], py[i0]
        x1, y1 = px[i1], py[i1]
        x2, y2 = px[i2], py[i2]
        z0, z1, z2 = z_cam[i0], z_cam[i1], z_cam[i2]

        # Skip back-facing (in camera space, face normal Z < 0) and behind camera
        if z0 <= 0 or z1 <= 0 or z2 <= 0:
            continue

        # Back-face culling via face normal camera-Z
        fn_cam = R @ face_normals[fi]
        if fn_cam[2] >= 0:   # normal points away from camera
            continue

        # Bounding box
        xmin = max(0, int(min(x0, x1, x2)))
        xmax = min(W - 1, int(max(x0, x1, x2)) + 1)
        ymin = max(0, int(min(y0, y1, y2)))
        ymax = min(H - 1, int(max(y0, y1, y2)) + 1)
        if xmin > xmax or ymin > ymax:
            continue

        # Shading: deterministic per-face color
        hue = (fi * 2654435761) % (1 << 24)
        base_color = np.array([(hue >> 16) & 0xFF,
                               (hue >>  8) & 0xFF,
                                hue        & 0xFF], dtype=np.float32) / 255.0
        face_color = _phong_shade(fn_cam, light_dir, view_dir, base_color)

        # Area denominator for barycentric coords
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8:
            continue

        xs = np.arange(xmin, xmax + 1, dtype=np.float64)
        ys = np.arange(ymin, ymax + 1, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)

        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        iy = gy[inside].astype(int)
        ix = gx[inside].astype(int)

        # Perspective-correct interpolation using 1/z
        inv_z0, inv_z1, inv_z2 = 1.0 / z0, 1.0 / z1, 1.0 / z2
        w0_p = w0[inside]; w1_p = w1[inside]; w2_p = w2[inside]
        inv_z_interp = w0_p * inv_z0 + w1_p * inv_z1 + w2_p * inv_z2

        # Depth test (larger inv_z = closer to camera)
        mask = inv_z_interp > inv_z_buf[iy, ix]
        iy, ix = iy[mask], ix[mask]
        if len(iy) == 0:
            continue
        w0_f = w0_p[mask]; w1_f = w1_p[mask]; w2_f = w2_p[mask]
        inv_z_f = inv_z_interp[mask]

        inv_z_buf[iy, ix] = inv_z_f
        rgb_buf[iy, ix]   = face_color.astype(np.float32)
        face_id_buf[iy, ix] = fi

        # Perspective-correct normal interpolation
        # n_interp = (w0*n0/z0 + w1*n1/z1 + w2*n2/z2) / inv_z
        n0 = vn_cam[i0]; n1 = vn_cam[i1]; n2 = vn_cam[i2]
        ni = (w0_f[:, None] * n0 + w1_f[:, None] * n1 + w2_f[:, None] * n2)
        ni_norms = np.linalg.norm(ni, axis=1, keepdims=True)
        ni = ni / np.where(ni_norms < 1e-9, 1.0, ni_norms)
        # Map [-1,1] → [0,1] for storage
        norm_buf[iy, ix] = ((ni + 1.0) * 0.5).astype(np.float32)

    # ── Post-process buffers ─────────────────────────────────────────────────

    silhouette = (inv_z_buf > 0).astype(np.uint8) * 255

    # Depth: convert inv_z to normalised 0..255 (closer = brighter)
    valid_mask = inv_z_buf > 0
    if valid_mask.any():
        iz_valid = inv_z_buf[valid_mask]
        iz_min, iz_max = iz_valid.min(), iz_valid.max()
        depth_norm = np.where(
            valid_mask,
            255.0 * (inv_z_buf - iz_min) / max(iz_max - iz_min, 1e-9),
            0.0,
        ).astype(np.uint8)
    else:
        depth_norm = np.zeros((H, W), dtype=np.uint8)

    # Edge detection: mark pixels where neighbouring face IDs have
    # large normal angle difference (curvature / silhouette edges)
    edge_map = np.zeros((H, W), dtype=np.uint8)
    fi_pad = np.pad(face_id_buf, 1, constant_values=-1)
    neighbours = [
        fi_pad[1:-1, 2:],   # right
        fi_pad[1:-1, :-2],  # left
        fi_pad[2:,  1:-1],  # down
        fi_pad[:-2, 1:-1],  # up
    ]
    fn_cam_all = (R @ face_normals.T).T   # (F, 3)
    for nb in neighbours:
        diff_face = (face_id_buf >= 0) & (nb >= 0) & (nb != face_id_buf)
        if not diff_face.any():
            continue
        fi_a = face_id_buf[diff_face]
        fi_b = nb[diff_face]
        na = fn_cam_all[fi_a]
        nb_v = fn_cam_all[fi_b]
        cos_ang = np.sum(na * nb_v, axis=1)   # dot product of unit normals
        is_edge = cos_ang < edge_threshold
        rows, cols = np.where(diff_face)
        edge_map[rows[is_edge], cols[is_edge]] = 255

    rgb_uint8  = (np.clip(rgb_buf,  0, 1) * 255).astype(np.uint8)
    norm_uint8 = (np.clip(norm_buf, 0, 1) * 255).astype(np.uint8)

    return {
        "rgb":        rgb_uint8,
        "depth":      depth_norm,
        "normal_map": norm_uint8,
        "silhouette": silhouette,
        "edge_map":   edge_map,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_views(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    view_names: Optional[Iterable[str]] = None,
    resolution: Tuple[int, int] = (512, 512),
    n_views: int = 0,
    view_method: str = "fibonacci",
    fov_deg: float = 45.0,
    camera_dist: float = 3.0,
    edge_threshold: float = 0.35,
) -> Dict[str, Path]:
    """Render multi-view images of a mesh and save PNG images.

    Parameters
    ----------
    mesh          : Trimesh to render.
    output_dir    : Directory to write images into.
    view_names    : Subset of CANONICAL_VIEWS keys. None = all 6 canonical views.
                    Ignored when ``n_views > 0``.
    resolution    : (width, height) of output images.
    n_views       : If > 0, generate exactly this many views using the
                    ``view_method`` schedule (overrides ``view_names``).
                    Typical: 6 (fast) → 36 (good) → 100 (dense).
    view_method   : ``"fibonacci"`` or ``"grid"`` (only used when n_views > 0).
    fov_deg       : Camera field-of-view in degrees.
    camera_dist   : Distance of camera from mesh origin.
    edge_threshold: cos(θ) threshold for edge detection (lower = more edges).

    Returns
    -------
    Dict mapping ``"<view>_<type>"`` → ``Path`` for every saved image.
    """
    MAX_RENDER_VERTICES = 500_000
    if mesh.vertices.shape[0] > MAX_RENDER_VERTICES:
        logger.warning(
            "Skipping render — mesh has %d vertices (limit %d).",
            mesh.vertices.shape[0], MAX_RENDER_VERTICES,
        )
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    img_size = resolution[0]  # assume square

    # ── Normalise mesh to unit sphere centred at origin ───────────────────
    verts = mesh.vertices.astype(np.float64)
    centre = verts.mean(axis=0)
    verts -= centre
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 1e-9:
        verts /= scale

    faces = mesh.faces

    # Recompute face normals on normalised geometry
    v0f = verts[faces[:, 0]]
    v1f = verts[faces[:, 1]]
    v2f = verts[faces[:, 2]]
    fn  = np.cross(v1f - v0f, v2f - v0f)
    fn_norms = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.where(fn_norms < 1e-12, 1.0, fn_norms)

    # Compute per-vertex normals (area-weighted average of adjacent face normals)
    vn = np.zeros_like(verts)
    np.add.at(vn, faces[:, 0], fn * fn_norms)
    np.add.at(vn, faces[:, 1], fn * fn_norms)
    np.add.at(vn, faces[:, 2], fn * fn_norms)
    vn_norms = np.linalg.norm(vn, axis=1, keepdims=True)
    vertex_normals = vn / np.where(vn_norms < 1e-12, 1.0, vn_norms)

    # ── Build view list ───────────────────────────────────────────────────
    if n_views > 0:
        view_schedule = build_view_schedule(n_views, method=view_method)
        logger.info(
            "Rendering %d views via '%s' schedule.", len(view_schedule), view_method
        )
    else:
        # Use canonical 6-view set (or user-specified subset)
        names_to_render = (
            list(view_names) if view_names is not None else list(CANONICAL_VIEWS.keys())
        )
        view_schedule = [
            (name, *CANONICAL_VIEWS[name])
            for name in names_to_render
            if name in CANONICAL_VIEWS
        ]

    # ── Render each view ──────────────────────────────────────────────────
    outputs: Dict[str, Path] = {}
    for name, az, el in view_schedule:
        R = _rotation_matrix(az, el)
        try:
            buffers = _rasterise(
                verts, faces, fn, vertex_normals, R, img_size,
                fov_deg=fov_deg,
                camera_dist=camera_dist,
                edge_threshold=edge_threshold,
            )
        except Exception as exc:
            logger.warning("Rasterise failed for view '%s': %s", name, exc)
            continue

        for buf_name, arr in buffers.items():
            out_path = output_dir / f"{name}_{buf_name}.png"
            try:
                if arr.ndim == 2:
                    img = Image.fromarray(arr, mode="L")
                else:
                    img = Image.fromarray(arr, mode="RGB")
                img.save(out_path)
                outputs[f"{name}_{buf_name}"] = out_path
            except Exception as exc:
                logger.warning("Failed writing %s: %s", out_path, exc)

    logger.info("Rendered %d views → %d image files.", len(view_schedule), len(outputs))
    return outputs


def render_views_simple(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    view_names: Optional[Iterable[str]] = None,
    resolution: Tuple[int, int] = (512, 512),
    n_views: int = 0,
    view_method: str = "fibonacci",
) -> Dict[str, Path]:
    """Backward-compatible wrapper returning only ``rgb`` images.

    Keys are view names without the ``_rgb`` suffix (e.g. ``"front"``).
    Pass ``n_views > 0`` for dense scanning.
    """
    all_views = render_views(
        mesh, output_dir,
        view_names=view_names,
        resolution=resolution,
        n_views=n_views,
        view_method=view_method,
    )
    rgb_views: Dict[str, Path] = {}
    for key, path in all_views.items():
        if key.endswith("_rgb"):
            rgb_views[key[: -len("_rgb")]] = path
    return rgb_views


# ---------------------------------------------------------------------------
# Convenience: generate azimuth metadata JSON for downstream analysis
# ---------------------------------------------------------------------------

def export_view_metadata(
    view_schedule: List[Tuple[str, float, float]],
    output_dir: Path,
) -> Path:
    """Write ``view_metadata.json`` listing name/az/el for all rendered views."""
    import json
    meta = [
        {"name": name, "azimuth_deg": az, "elevation_deg": el}
        for name, az, el in view_schedule
    ]
    out = output_dir / "view_metadata.json"
    out.write_text(json.dumps(meta, indent=2))
    return out
