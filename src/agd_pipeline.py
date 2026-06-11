"""AGD pipeline — high-density multi-view scanning + accurate 3D error metrics.

New in v5 (Stage 7 & 8)
------------------------
* Outer refinement loop (--refine-loops): repeats render->detect->ground->refine
  until the hallucination score converges or max loops is reached.
* Per-loop progress logging with score trajectory.
* Silhouette-IoU feedback carried forward into each loop's grounding weights.
* All v4 features retained (floater removal, cotangent Laplacian, LR decay,
  depth-edge grounding, all-pairs view consistency).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh

from critic import build_critic, CriticConfig
from grounding import build_region_weights, map_views_to_vertex_bias, map_silhouette_iou_to_bias
from lmm_detector import detect_issues, LLaVADetector
from optimizer import refine_mesh, remove_small_components
from renderer import render_views
from view_consistency import check_view_consistency
from discriminator import compute_laplacian_metrics, compute_topology, hallucination_score
from geometry_metrics import compute_geometry_error_report, GeometryErrorReport

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


def _analyse(mesh: trimesh.Trimesh) -> Tuple[dict, dict, float]:
    topo  = compute_topology(mesh)
    spec  = compute_laplacian_metrics(mesh, sample_fraction=1.0)
    score = hallucination_score(topo, spec)
    return topo, spec, score


def _header(text: str) -> None:
    bar = "-" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


def _print_topo_row(label: str, topo: dict, spec: dict, score: float) -> None:
    print(f"  {label}")
    print(f"    score={score:.3f}  variance={spec['eigen_variance']:.5f}  "
          f"fiedler={spec['fiedler_value']:.5f}")
    print(f"    components={int(topo['components'])}  genus={topo['genus']:.1f}  "
          f"euler_char={topo['euler_char']:.1f}")


def _extract_depth_paths(views: Dict[str, Path]) -> Dict[str, Path]:
    return {
        key[: -len("_depth")]: path
        for key, path in views.items()
        if key.endswith("_depth")
    }


def _extract_per_view_silhouette_ious(
    view_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    return {
        view: m["silhouette_iou"]
        for view, m in view_metrics.items()
        if "silhouette_iou" in m
    }


# ---------------------------------------------------------------------------
# Single-pass refinement (one loop of render -> detect -> ground -> refine)
# ---------------------------------------------------------------------------

def _refine_one_pass(
    mesh: trimesh.Trimesh,
    anchor_vertices: np.ndarray,
    args,
    critic,
    detector,
    output_dir: Path,
    path: Path,
    loop_idx: int,
    sil_bias_carry: Optional[np.ndarray] = None,
) -> Tuple[trimesh.Trimesh, Optional[GeometryErrorReport], Dict[str, Path]]:
    """Run one full render->detect->ground->refine pass.

    Parameters
    ----------
    mesh:
        Current mesh state (modified in-place by refine_mesh via copy).
    anchor_vertices:
        Original vertex positions (fixed for distillation loss — never rolls).
    loop_idx:
        0-based loop index, used to label output subdirectories.
    sil_bias_carry:
        Optional silhouette-IoU bias from the *previous* loop, merged into
        this loop's grounding weights to focus on persistently bad regions.

    Returns
    -------
    refined:
        Refined mesh after this pass.
    geo_report:
        Geometry error report (None if --skip-geometry-metrics).
    views_current:
        Rendered views dict for the current (input) mesh state.
    """
    loop_tag = f"loop{loop_idx + 1:02d}"
    views_current: Dict[str, Path] = {}
    bias_weights: Optional[np.ndarray] = sil_bias_carry
    global_bias = 0.0

    # ── Render current mesh ──────────────────────────────────────────────
    if args.render_views:
        view_dir = output_dir / "views" / path.stem / loop_tag
        n_views_arg     = getattr(args, "num_views", 0)
        view_method_arg = getattr(args, "view_method", "fibonacci")
        logger.info(
            "Loop %d: rendering %s views (n=%d, method=%s) ...",
            loop_idx + 1,
            "dense" if n_views_arg > 0 else "canonical-6",
            n_views_arg or 6, view_method_arg,
        )
        views_current = render_views(
            mesh, view_dir,
            resolution=(args.view_size, args.view_size),
            n_views=n_views_arg,
            view_method=view_method_arg,
            fov_deg=getattr(args, "fov_deg", 45.0),
            edge_threshold=getattr(args, "edge_threshold", 0.35),
        )

        rgb_views  = {
            k.replace("_rgb", ""): v
            for k, v in views_current.items() if k.endswith("_rgb")
        }
        depth_paths = _extract_depth_paths(views_current)

        if rgb_views:
            consistency = check_view_consistency(
                rgb_views,
                depth_paths=depth_paths,
                similarity_threshold=args.consistency_threshold,
            )
            view_bias = map_views_to_vertex_bias(
                mesh, consistency.view_scores,
                view_extent=args.view_extent,
                normal_threshold=args.normal_threshold,
            )
            bias_weights = (
                view_bias if bias_weights is None
                else np.maximum(bias_weights, view_bias)
            )

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

    # ── Critic & grounding weights ────────────────────────────────────────
    scores  = critic.score(mesh)
    weights = build_region_weights(
        mesh, scores,
        top_fraction=args.top_fraction,
        expand_hops=args.expand_hops,
        global_bias=global_bias,
        bias_weights=bias_weights,
    )

    # ── Refine ────────────────────────────────────────────────────────────
    # Always use the ORIGINAL anchor vertices — prevents cumulative drift
    mesh_copy = mesh.copy()
    mesh_copy.vertices = mesh.vertices.copy()
    refined = refine_mesh(
        mesh_copy,
        weights=weights,
        iterations=args.iterations,
        lr=args.lr,
        lambda_geo=args.lambda_geo,
        lambda_distill=args.lambda_distill,
        use_cotangent=args.use_cot_laplacian,
        lr_decay=args.lr_decay,
        lr_tol=args.lr_tol,
        clamp_fraction=args.clamp_fraction,
    )
    # Override the in-refine anchor so distillation always pulls to original
    # (refine_mesh already copies on entry, but we set it explicitly here)
    refined.vertices  # touch cache

    # ── Compute geometry error metrics (vs original) ──────────────────────
    geo_report: Optional[GeometryErrorReport] = None
    views_after: Dict[str, Path] = {}

    if not args.skip_geometry_metrics:
        if args.render_views:
            view_dir_after = output_dir / "views" / path.stem / f"{loop_tag}_after"
            views_after = render_views(
                refined, view_dir_after,
                resolution=(args.view_size, args.view_size),
                n_views=getattr(args, "num_views", 0),
                view_method=getattr(args, "view_method", "fibonacci"),
                fov_deg=getattr(args, "fov_deg", 45.0),
                edge_threshold=getattr(args, "edge_threshold", 0.35),
            )

        mesh_original_ref = trimesh.Trimesh(
            vertices=anchor_vertices,
            faces=mesh.faces.copy(),
            process=False,
        )
        geo_report = compute_geometry_error_report(
            mesh_original_ref, refined,
            views_before=views_current if args.render_views else None,
            views_after=views_after    if args.render_views else None,
            n_surface_points=args.n_sample_pts,
        )

    return refined, geo_report, views_current


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AGD refinement with outer loop + convergence (v5)"
    )
    # Input / output
    parser.add_argument("--input-dir",  default="3d_samples",  help="Folder with mesh files")
    parser.add_argument("--input-file", default=None,
                        help="Single mesh file (overrides --input-dir)")
    parser.add_argument("--output-dir", default="agd_outputs", help="Output folder")

    # Optimisation
    parser.add_argument("--iterations",     type=int,   default=10,   help="Inner optimisation iterations per loop")
    parser.add_argument("--lr",             type=float, default=0.15, help="Initial step size")
    parser.add_argument("--lambda-geo",     type=float, default=0.7,  help="Geometry loss weight")
    parser.add_argument("--lambda-distill", type=float, default=0.3,  help="Distillation loss weight")
    parser.add_argument("--use-cot-laplacian", action="store_true",
                        help="Cotangent-weighted Laplacian (better quality, slower)")
    parser.add_argument("--lr-decay",       type=float, default=0.85, help="LR decay per inner iteration")
    parser.add_argument("--lr-tol",         type=float, default=1e-6, help="Inner convergence tolerance")
    parser.add_argument("--clamp-fraction", type=float, default=0.01, help="Max displacement as fraction of scale")

    # Outer refinement loop (Stage 7)
    parser.add_argument("--refine-loops",         type=int,   default=1,
                        help="Max outer refinement loops (Stage 7). Default=1 = single pass.")
    parser.add_argument("--loop-convergence-tol", type=float, default=0.001,
                        help="Stop outer loop when |score change| < tol. Default=0.001.")

    # Floater removal
    parser.add_argument("--no-floater-removal", action="store_true")
    parser.add_argument("--min-face-fraction",  type=float, default=0.001)

    # Grounding
    parser.add_argument("--top-fraction",      type=float, default=0.1)
    parser.add_argument("--expand-hops",       type=int,   default=1)
    parser.add_argument("--use-gnn",           action="store_true")
    parser.add_argument("--gnn-checkpoint",    default=None)

    # Rendering
    parser.add_argument("--render-views",      action="store_true")
    parser.add_argument("--view-size",         type=int,   default=256)
    parser.add_argument("--num-views",         type=int,   default=0)
    parser.add_argument("--view-method",       default="fibonacci",
                        choices=["fibonacci", "grid"])
    parser.add_argument("--fov-deg",           type=float, default=45.0)
    parser.add_argument("--edge-threshold",    type=float, default=0.35)

    # VLM / consistency
    parser.add_argument("--use-vlm",              action="store_true")
    parser.add_argument("--llava-model-path",     default=None)
    parser.add_argument("--llava-model-base",     default=None)
    parser.add_argument("--llava-max-new-tokens", type=int, default=256)
    parser.add_argument("--consistency-threshold", type=float, default=0.85)
    parser.add_argument("--view-extent",          type=float, default=0.3)
    parser.add_argument("--normal-threshold",     type=float, default=0.2)

    # Metrics
    parser.add_argument("--n-sample-pts",          type=int, default=6000)
    parser.add_argument("--skip-geometry-metrics", action="store_true")

    args = parser.parse_args()

    if args.use_vlm and not args.render_views:
        raise SystemExit("--use-vlm requires --render-views")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        single = Path(args.input_file)
        if not single.exists():
            raise SystemExit(f"File not found: {single}")
        if single.suffix.lower() not in SUPPORTED_EXTS:
            raise SystemExit(f"Unsupported format '{single.suffix}'")
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

    max_loops = max(1, args.refine_loops)
    print(f"\n{'='*62}")
    print(f"  AGD Pipeline v5 -- {len(paths)} meshes  |  "
          f"{max_loops} loop(s) x {args.iterations} iters  |  lr={args.lr}")
    print(f"{'='*62}")

    summary_rows: List[Dict] = []

    for path in paths:
        _header(path.name)
        t0 = time.perf_counter()

        mesh = _load_mesh(path)
        if mesh is None:
            print("  [SKIP] No geometry found.")
            continue
        if mesh.vertices.shape[0] > MAX_VERTICES:
            print(f"  [SKIP] {mesh.vertices.shape[0]:,} verts > limit {MAX_VERTICES:,}")
            continue

        print(f"  Loaded: {mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces")

        # Floater removal (pre-step, runs once before any loop)
        if not args.no_floater_removal:
            before_comps = len(mesh.split(only_watertight=False))
            mesh = remove_small_components(mesh, min_face_fraction=args.min_face_fraction)
            after_comps = len(mesh.split(only_watertight=False))
            if before_comps != after_comps:
                print(f"  Floaters removed: {before_comps} -> {after_comps} components  "
                      f"({mesh.vertices.shape[0]:,} verts remaining)")

        # Lock in original vertices for distillation anchor (never rolls)
        anchor_vertices = mesh.vertices.copy()

        # Initial analysis (before any refinement)
        before_topo, before_spec, before_score = _analyse(mesh)
        _print_topo_row("BEFORE (loop 0)", before_topo, before_spec, before_score)

        # Score trajectory across loops
        score_trajectory: List[float] = [before_score]

        current_mesh   = mesh
        last_geo_report: Optional[GeometryErrorReport] = None
        sil_bias_carry: Optional[np.ndarray] = None
        converged = False

        # ── Outer refinement loop (Stage 7) ─────────────────────────────
        for loop_idx in range(max_loops):
            loop_num = loop_idx + 1
            print(f"\n  --- Loop {loop_num}/{max_loops} ---")

            refined, geo_report, views_current = _refine_one_pass(
                current_mesh,
                anchor_vertices,
                args, critic, detector,
                output_dir, path,
                loop_idx=loop_idx,
                sil_bias_carry=sil_bias_carry,
            )

            # Analyse refined mesh
            loop_topo, loop_spec, loop_score = _analyse(refined)
            _print_topo_row(f"  AFTER loop {loop_num}", loop_topo, loop_spec, loop_score)

            delta = loop_score - score_trajectory[-1]
            sign  = "v" if delta < 0 else ("^" if delta > 0 else "->")
            print(f"    D score={delta:+.5f}{sign}  "
                  f"(trajectory: {' -> '.join(f'{s:.3f}' for s in score_trajectory + [loop_score])})")

            if geo_report:
                print(geo_report.summary_str())

            # Carry silhouette-IoU bias into next loop
            if geo_report and geo_report.view_metrics:
                sil_ious = _extract_per_view_silhouette_ious(geo_report.view_metrics)
                if sil_ious:
                    sil_bias_carry = map_silhouette_iou_to_bias(
                        refined, sil_ious,
                        view_extent=args.view_extent,
                        normal_threshold=args.normal_threshold,
                    )
                    logger.info(
                        "Loop %d: silhouette bias carried forward, mean=%.4f",
                        loop_num, float(np.mean(sil_bias_carry)),
                    )

            score_trajectory.append(loop_score)
            last_geo_report = geo_report
            current_mesh    = refined

            # Convergence check
            if abs(delta) < args.loop_convergence_tol:
                print(f"  [CONVERGED] |D score|={abs(delta):.6f} < tol={args.loop_convergence_tol}")
                converged = True
                break

        # ── Final export ─────────────────────────────────────────────────
        out_path = output_dir / f"{path.stem}_agd{path.suffix}"
        current_mesh.export(out_path)

        elapsed = time.perf_counter() - t0
        loops_done = len(score_trajectory) - 1

        # Print score trajectory summary
        print(f"\n  Score trajectory ({loops_done} loop(s)"
              f"{', converged' if converged else ''}): "
              f"{' -> '.join(f'{s:.3f}' for s in score_trajectory)}")
        total_improvement = score_trajectory[-1] - score_trajectory[0]
        sign_total = "v" if total_improvement < 0 else ("^" if total_improvement > 0 else "->")
        print(f"  Total D score={total_improvement:+.5f}{sign_total}")
        print(f"  Saved -> {out_path.name}  [{elapsed:.1f}s]")

        # Summary row (uses final loop's geo_report)
        row: Dict = {
            "mesh":         path.name,
            "score_before": score_trajectory[0],
            "score_after":  score_trajectory[-1],
            "loops_done":   loops_done,
            "converged":    converged,
        }
        if last_geo_report:
            row.update(last_geo_report.to_dict())
        summary_rows.append(row)

    # ── Final summary table ───────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  FINAL SUMMARY")
    print(f"{'='*62}")
    hdr = (f"  {'Mesh':<26} {'Loops':>5} {'Score B->A':>10} "
           f"{'CD':>9} {'HD95':>8} {'NCS':>6} {'SSIM':>6}")
    print(hdr)
    print("  " + "-" * 80)
    for row in summary_rows:
        cd   = row.get("chamfer_distance",   float("nan"))
        hd95 = row.get("hausdorff_95pct",    float("nan"))
        ncs  = row.get("normal_consistency", float("nan"))
        ssim = row.get("mean_ssim",          float("nan"))
        cvg  = "*" if row.get("converged") else " "
        print(
            f"  {row['mesh']:<26} "
            f"{row['loops_done']:>4}{cvg} "
            f"{row['score_before']:.3f}->{row['score_after']:.3f}  "
            f"{cd:>9.5f}  {hd95:>8.5f}  {ncs:>6.3f}  {ssim:>6.4f}"
        )
    print("  (* = converged early)")
    print()


if __name__ == "__main__":
    main()
