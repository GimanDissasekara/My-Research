"""AGD pipeline — high-density multi-view scanning + accurate 3D error metrics.

New in v3
---------
* Dense camera scanning: --num-views 6..100+ via Fibonacci-sphere or grid
* Phong shading, perspective-correct depth, interpolated normals, edge maps
* Additional metrics: Angular Normal Error (°), Surface Roughness RMS, SSIM,
  Edge-IoU, and full multi-view aggregate (min/max/std/p5/p95 across all views)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh

from critic import build_critic, CriticConfig
from grounding import build_region_weights, map_views_to_vertex_bias
from lmm_detector import detect_issues, LLaVADetector
from optimizer import refine_mesh
from renderer import render_views, render_views_simple
from view_consistency import check_view_consistency
from discriminator import compute_laplacian_metrics, compute_topology, hallucination_score
from geometry_metrics import compute_geometry_error_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".obj", ".ply", ".stl"}
MAX_VERTICES   = 500_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_mesh_paths(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def _load_mesh(path: Path) -> Optional[trimesh.Trimesh]:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if not geoms:
            return None
        mesh = trimesh.util.concatenate(geoms)
    return mesh


def _analyse(mesh: trimesh.Trimesh):
    topo  = compute_topology(mesh)
    spec  = compute_laplacian_metrics(mesh, sample_fraction=1.0)
    score = hallucination_score(topo, spec)
    return topo, spec, score


def _header(text: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


def _print_topo_row(label: str, topo, spec, score):
    print(f"  {label}")
    print(f"    score={score:.3f}  variance={spec['eigen_variance']:.5f}  "
          f"fiedler={spec['fiedler_value']:.5f}")
    print(f"    components={int(topo['components'])}  genus={topo['genus']:.1f}  "
          f"euler_char={topo['euler_char']:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run AGD refinement with accurate 3D error metrics")
    parser.add_argument("--input-dir",  default="3d_samples",  help="Folder with mesh files")
    parser.add_argument("--input-file", default=None,
                        help="Path to a single mesh file (overrides --input-dir)")
    parser.add_argument("--output-dir", default="agd_outputs", help="Folder to save refined meshes")
    parser.add_argument("--iterations",  type=int,   default=10,   help="Optimization iterations")
    parser.add_argument("--lr",          type=float, default=0.15, help="Optimization step size")
    parser.add_argument("--lambda-geo",      type=float, default=0.7, help="Geometry loss weight")
    parser.add_argument("--lambda-distill",  type=float, default=0.3, help="Distillation loss weight")
    parser.add_argument("--top-fraction",    type=float, default=0.1, help="Top-fraction of vertices")
    parser.add_argument("--expand-hops",     type=int,   default=1,   help="Neighbor expansion hops")
    parser.add_argument("--use-gnn",         action="store_true")
    parser.add_argument("--gnn-checkpoint",  default=None)
    parser.add_argument("--render-views",    action="store_true", help="Render multi-view images")
    parser.add_argument("--view-size",       type=int,   default=256, help="View resolution")
    # ── Dense view scanning ──────────────────────────────────────────────
    parser.add_argument("--num-views",  type=int, default=0,
                        help="Number of camera views to render (0=use 6 canonical). "
                             "Typical: 6 (fast) / 36 (good) / 100 (dense scan)")
    parser.add_argument("--view-method", default="fibonacci",
                        choices=["fibonacci", "grid"],
                        help="Camera placement strategy for --num-views")
    parser.add_argument("--fov-deg",    type=float, default=45.0, help="Camera FOV degrees")
    parser.add_argument("--edge-threshold", type=float, default=0.35,
                        help="cos(θ) threshold for edge/curvature detection")
    # ── VLM / consistency ───────────────────────────────────────────────
    parser.add_argument("--use-vlm",         action="store_true")
    parser.add_argument("--llava-model-path",  default=None)
    parser.add_argument("--llava-model-base",  default=None)
    parser.add_argument("--llava-max-new-tokens", type=int, default=256)
    parser.add_argument("--consistency-threshold", type=float, default=0.85)
    parser.add_argument("--view-extent",    type=float, default=0.3)
    parser.add_argument("--normal-threshold", type=float, default=0.2)
    parser.add_argument("--n-sample-pts",  type=int, default=6000,
                        help="Surface points for Chamfer/Hausdorff/NCS metrics")
    parser.add_argument("--skip-geometry-metrics", action="store_true",
                        help="Skip Chamfer/Hausdorff (faster, less accurate)")
    args = parser.parse_args()

    if args.use_vlm and not args.render_views:
        raise SystemExit("--use-vlm requires --render-views")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single-file mode
    if args.input_file:
        single = Path(args.input_file)
        if not single.exists():
            raise SystemExit(f"File not found: {single}")
        if single.suffix.lower() not in SUPPORTED_EXTS:
            raise SystemExit(f"Unsupported format '{single.suffix}'. Supported: {SUPPORTED_EXTS}")
        paths = [single]
    else:
        input_dir = Path(args.input_dir)
        paths = list(_iter_mesh_paths(input_dir))
        if not paths:
            raise SystemExit(f"No meshes found in {input_dir}")

    critic = build_critic(CriticConfig(
        use_gnn=args.use_gnn,
        gnn_checkpoint=args.gnn_checkpoint,
    ))

    detector = None
    if args.use_vlm:
        if not args.llava_model_path:
            raise SystemExit("--llava-model-path required with --use-vlm")
        detector = LLaVADetector(
            model_path=args.llava_model_path,
            model_base=args.llava_model_base,
            max_new_tokens=args.llava_max_new_tokens,
        )

    print(f"\n{'═'*62}")
    print(f"  AGD Pipeline — {len(paths)} meshes  │  {args.iterations} iters  │  lr={args.lr}")
    print(f"{'═'*62}")

    summary_rows = []  # for final table

    for path in paths:
        _header(path.name)
        t0 = time.perf_counter()

        mesh = _load_mesh(path)
        if mesh is None:
            print("  [SKIP] No geometry found.")
            continue
        if mesh.vertices.shape[0] > MAX_VERTICES:
            print(f"  [SKIP] {mesh.vertices.shape[0]:,} vertices > limit {MAX_VERTICES:,}")
            continue

        print(f"  Loaded: {mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces")

        # ── Before analysis ──────────────────────────────────────────────
        before_topo, before_spec, before_score = _analyse(mesh)
        _print_topo_row("BEFORE", before_topo, before_spec, before_score)

        # Keep a copy of original vertices for geometry metric comparison
        original_vertices = mesh.vertices.copy()

        # ── Render before views ───────────────────────────────────────────
        views_before: dict = {}
        if args.render_views:
            view_dir_before = output_dir / "views" / path.stem / "before"
            n_views_arg = getattr(args, "num_views", 0)
            view_method_arg = getattr(args, "view_method", "fibonacci")
            logger.info("Rendering %s views (n=%d, method=%s) …",
                        "dense" if n_views_arg > 0 else "canonical-6",
                        n_views_arg or 6, view_method_arg)
            views_before = render_views(
                mesh, view_dir_before,
                resolution=(args.view_size, args.view_size),
                n_views=n_views_arg,
                view_method=view_method_arg,
                fov_deg=getattr(args, "fov_deg", 45.0),
                edge_threshold=getattr(args, "edge_threshold", 0.35),
            )
            # Also get simple RGB dict for view-consistency check
            rgb_views = {k.replace("_rgb", ""): v
                         for k, v in views_before.items() if k.endswith("_rgb")}
            if rgb_views:
                consistency = check_view_consistency(
                    rgb_views, similarity_threshold=args.consistency_threshold)
                bias_weights = map_views_to_vertex_bias(
                    mesh, consistency.view_scores,
                    view_extent=args.view_extent,
                    normal_threshold=args.normal_threshold,
                )
            else:
                bias_weights = None

            global_bias = 0.0
            if args.use_vlm and detector is not None and rgb_views:
                detection   = detect_issues(rgb_views, detector=detector)
                global_bias = detection.global_bias
                lmm_bias    = map_views_to_vertex_bias(
                    mesh, detection.view_scores,
                    view_extent=args.view_extent,
                    normal_threshold=args.normal_threshold,
                )
                bias_weights = (
                    lmm_bias if bias_weights is None
                    else np.maximum(bias_weights, lmm_bias)
                )
        else:
            bias_weights = None
            global_bias  = 0.0

        # ── Critic & weights ─────────────────────────────────────────────
        scores       = critic.score(mesh)
        weights      = build_region_weights(
            mesh, scores,
            top_fraction=args.top_fraction,
            expand_hops=args.expand_hops,
            global_bias=global_bias,
            bias_weights=bias_weights,
        )

        # ── Refine ───────────────────────────────────────────────────────
        # Work on a copy so we can compare with original
        mesh_copy = mesh.copy()
        refined   = refine_mesh(
            mesh_copy, weights=weights,
            iterations=args.iterations,
            lr=args.lr,
            lambda_geo=args.lambda_geo,
            lambda_distill=args.lambda_distill,
        )

        # Restore original mesh (refine_mesh modifies in-place) for metric comparison
        mesh_original = trimesh.Trimesh(
            vertices=original_vertices,
            faces=mesh.faces.copy(),
            process=False,
        )

        # ── After analysis ───────────────────────────────────────────────
        after_topo, after_spec, after_score = _analyse(refined)
        _print_topo_row("AFTER ", after_topo, after_spec, after_score)

        delta_score    = after_score - before_score
        delta_variance = after_spec["eigen_variance"] - before_spec["eigen_variance"]
        sign_score = "↓" if delta_score < 0 else ("↑" if delta_score > 0 else "→")
        print(f"  Δ score={delta_score:+.3f}{sign_score}  "
              f"Δ variance={delta_variance:+.5f}")

        # ── Render after views & compute geometry error metrics ───────────
        geo_report = None
        views_after: dict = {}

        if not args.skip_geometry_metrics:
            if args.render_views:
                view_dir_after = output_dir / "views" / path.stem / "after"
                views_after = render_views(
                    refined, view_dir_after,
                    resolution=(args.view_size, args.view_size),
                    n_views=getattr(args, "num_views", 0),
                    view_method=getattr(args, "view_method", "fibonacci"),
                    fov_deg=getattr(args, "fov_deg", 45.0),
                    edge_threshold=getattr(args, "edge_threshold", 0.35),
                )

            print("  Computing geometry error metrics …")
            geo_report = compute_geometry_error_report(
                mesh_original, refined,
                views_before=views_before if args.render_views else None,
                views_after=views_after  if args.render_views else None,
                n_surface_points=args.n_sample_pts,
            )
            print(geo_report.summary_str())

        # ── Export ───────────────────────────────────────────────────────
        out_path = output_dir / f"{path.stem}_agd{path.suffix}"
        refined.export(out_path)

        elapsed = time.perf_counter() - t0
        print(f"  Saved → {out_path.name}  [{elapsed:.1f}s]")

        # Accumulate summary row
        row = {
            "mesh":           path.name,
            "score_before":   before_score,
            "score_after":    after_score,
            "var_before":     before_spec["eigen_variance"],
            "var_after":      after_spec["eigen_variance"],
        }
        if geo_report:
            row.update(geo_report.to_dict())
        summary_rows.append(row)

    # ── Final summary table ───────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print("  FINAL SUMMARY")
    print(f"{'═'*62}")
    fmt_hdr = (f"  {'Mesh':<28} {'Score B→A':>10} {'CD':>9} {'HD95':>8} "
               f"{'NCS':>6} {'Ang°':>7} {'Rough':>7} {'SSIM':>6}")
    print(fmt_hdr)
    print("  " + "─" * 82)
    for row in summary_rows:
        cd   = row.get("chamfer_distance",   float("nan"))
        hd95 = row.get("hausdorff_95pct",    float("nan"))
        ncs  = row.get("normal_consistency", float("nan"))
        ang  = row.get("angular_mean_deg",   float("nan"))
        rgh  = row.get("roughness_rms",      float("nan"))
        ssim = row.get("mean_ssim",          float("nan"))
        print(
            f"  {row['mesh']:<28} "
            f"{row['score_before']:.3f}→{row['score_after']:.3f}  "
            f"{cd:>9.5f}  {hd95:>8.5f}  {ncs:>6.3f}  "
            f"{ang:>7.2f}  {rgh:>7.5f}  {ssim:>6.4f}"
        )
    print()


if __name__ == "__main__":
    main()
