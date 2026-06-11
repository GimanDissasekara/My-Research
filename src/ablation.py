"""Stage 8 — Evaluation and Ablation Study.

Runs five pipeline configurations on the same set of meshes and produces:
  1. ablation_results.csv  — full metrics table (one row per mesh x condition)
  2. Printed ASCII summary table
  3. ablation_figures/     — matplotlib bar charts:
       score_comparison.png   hallucination score before vs after per condition
       cd_comparison.png      Chamfer distance per condition
       ssim_comparison.png    mean SSIM per condition

Ablation conditions
-------------------
  baseline        No refinement; original mesh scored only.
  heuristic_6v    Heuristic critic, 6-view rendering, uniform Laplacian.
  heuristic_36v   Heuristic critic, 36-view rendering, uniform Laplacian.
  gnn_6v          GNN critic, 6-view (skipped if checkpoint not found).
  cot_laplacian   Heuristic + cotangent Laplacian, 6-view.

Usage
-----
  python src/ablation.py --input-dir 3d_samples --output-dir src/ablation_out
  python src/ablation.py --input-dir 3d_samples --meshes bunny.ply janus.obj
  python src/ablation.py --input-dir 3d_samples --gnn-checkpoint gnn_critic.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

# Import pipeline building blocks directly (no subprocess overhead)
from critic import build_critic, CriticConfig
from discriminator import compute_laplacian_metrics, compute_topology, hallucination_score
from geometry_metrics import compute_geometry_error_report, GeometryErrorReport
from grounding import build_region_weights, map_views_to_vertex_bias
from optimizer import refine_mesh, remove_small_components
from renderer import render_views
from view_consistency import check_view_consistency
from agd_pipeline import (
    _load_mesh, _analyse, _extract_depth_paths,
    _extract_per_view_silhouette_ious,
    SUPPORTED_EXTS, MAX_VERTICES,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

@dataclass
class AblationCondition:
    """A single ablation configuration."""
    id: str
    description: str
    num_views: int = 6
    use_gnn: bool = False
    gnn_checkpoint: Optional[str] = None
    use_cotangent: bool = False
    iterations: int = 10
    render_views_flag: bool = True
    skip_refinement: bool = False  # baseline condition


def _build_conditions(args) -> List[AblationCondition]:
    conditions = [
        AblationCondition(
            id="baseline",
            description="No refinement (original mesh)",
            skip_refinement=True,
            render_views_flag=False,
        ),
        AblationCondition(
            id="heuristic_6v",
            description="Heuristic critic, 6-view, uniform Laplacian",
            num_views=6,
            use_gnn=False,
            use_cotangent=False,
        ),
        AblationCondition(
            id="heuristic_36v",
            description="Heuristic critic, 36-view, uniform Laplacian",
            num_views=36,
            use_gnn=False,
            use_cotangent=False,
        ),
        AblationCondition(
            id="cot_laplacian",
            description="Heuristic critic, 6-view, cotangent Laplacian",
            num_views=6,
            use_gnn=False,
            use_cotangent=True,
        ),
    ]

    # GNN condition — only added if checkpoint exists
    ckpt = getattr(args, "gnn_checkpoint", None)
    if ckpt and Path(ckpt).exists():
        conditions.append(AblationCondition(
            id="gnn_6v",
            description=f"GNN critic ({Path(ckpt).name}), 6-view, uniform Laplacian",
            num_views=6,
            use_gnn=True,
            gnn_checkpoint=ckpt,
            use_cotangent=False,
        ))
    else:
        print(f"  [NOTE] GNN condition skipped — checkpoint not found: {ckpt}")

    return conditions


# ---------------------------------------------------------------------------
# Single mesh + single condition runner
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    condition_id: str
    mesh_name: str
    score_before: float = 0.0
    score_after: float = 0.0
    elapsed_s: float = 0.0
    geo: Optional[GeometryErrorReport] = None
    error: Optional[str] = None

    def to_row(self) -> Dict:
        row = {
            "condition":    self.condition_id,
            "mesh":         self.mesh_name,
            "score_before": self.score_before,
            "score_after":  self.score_after,
            "score_delta":  self.score_after - self.score_before,
            "elapsed_s":    round(self.elapsed_s, 2),
            "error":        self.error or "",
        }
        if self.geo:
            row.update({
                "chamfer_distance":   self.geo.cd,
                "hausdorff_95pct":    self.geo.hd95,
                "normal_consistency": self.geo.ncs,
                "angular_mean_deg":   self.geo.angular_mean_deg,
                "roughness_rms":      self.geo.roughness_rms,
                "mean_silhouette_iou": self.geo.mean_silhouette_iou,
                "mean_ssim":          self.geo.mean_ssim,
                "mean_edge_iou":      self.geo.mean_edge_iou,
            })
        return row


def _run_condition(
    mesh_path: Path,
    cond: AblationCondition,
    output_dir: Path,
    args,
) -> ConditionResult:
    """Run one ablation condition on one mesh."""
    result = ConditionResult(
        condition_id=cond.id,
        mesh_name=mesh_path.name,
    )
    t0 = time.perf_counter()

    try:
        mesh = _load_mesh(mesh_path)
        if mesh is None:
            result.error = "no geometry"
            return result
        if mesh.vertices.shape[0] > MAX_VERTICES:
            result.error = f"too large ({mesh.vertices.shape[0]:,} verts)"
            return result

        # Floater removal
        mesh = remove_small_components(mesh, min_face_fraction=args.min_face_fraction)

        _, _, before_score = _analyse(mesh)
        result.score_before = before_score

        if cond.skip_refinement:
            # Baseline — no refinement
            result.score_after = before_score
            result.elapsed_s = time.perf_counter() - t0
            return result

        anchor_vertices = mesh.vertices.copy()

        # Render
        views: Dict[str, Path] = {}
        bias_weights = None
        global_bias = 0.0

        if cond.render_views_flag:
            view_dir = output_dir / "views" / mesh_path.stem / cond.id
            views = render_views(
                mesh, view_dir,
                resolution=(args.view_size, args.view_size),
                n_views=cond.num_views,
                view_method="fibonacci",
                fov_deg=45.0,
                edge_threshold=0.35,
            )
            rgb_views   = {k.replace("_rgb", ""): v for k, v in views.items() if k.endswith("_rgb")}
            depth_paths = _extract_depth_paths(views)

            if rgb_views:
                consistency = check_view_consistency(
                    rgb_views,
                    depth_paths=depth_paths,
                    similarity_threshold=0.85,
                )
                bias_weights = map_views_to_vertex_bias(
                    mesh, consistency.view_scores,
                    view_extent=0.3,
                    normal_threshold=0.2,
                )

        # Critic
        critic = build_critic(CriticConfig(
            use_gnn=cond.use_gnn,
            gnn_checkpoint=cond.gnn_checkpoint,
        ))
        scores  = critic.score(mesh)
        weights = build_region_weights(
            mesh, scores,
            top_fraction=0.1,
            expand_hops=1,
            global_bias=global_bias,
            bias_weights=bias_weights,
        )

        # Refine
        mesh_copy = mesh.copy()
        refined = refine_mesh(
            mesh_copy,
            weights=weights,
            iterations=cond.iterations,
            lr=0.15,
            lambda_geo=0.7,
            lambda_distill=0.3,
            use_cotangent=cond.use_cotangent,
            lr_decay=0.85,
            lr_tol=1e-6,
            clamp_fraction=0.01,
        )

        _, _, after_score = _analyse(refined)
        result.score_after = after_score

        # Geometry error metrics
        views_after: Dict[str, Path] = {}
        if views:
            view_dir_after = output_dir / "views" / mesh_path.stem / f"{cond.id}_after"
            views_after = render_views(
                refined, view_dir_after,
                resolution=(args.view_size, args.view_size),
                n_views=cond.num_views,
                view_method="fibonacci",
                fov_deg=45.0,
                edge_threshold=0.35,
            )

        mesh_ref = trimesh.Trimesh(vertices=anchor_vertices, faces=mesh.faces.copy(), process=False)
        result.geo = compute_geometry_error_report(
            mesh_ref, refined,
            views_before=views or None,
            views_after=views_after or None,
            n_surface_points=args.n_sample_pts,
        )

    except Exception as exc:
        result.error = str(exc)
        logger.exception("Error in condition %s on %s", cond.id, mesh_path.name)

    result.elapsed_s = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# CSV + figures
# ---------------------------------------------------------------------------

def _write_csv(results: List[ConditionResult], out_path: Path) -> None:
    rows = [r.to_row() for r in results]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    # Ensure all optional keys present
    all_keys: set = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 999)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  CSV saved -> {out_path}")


def _plot_figures(results: List[ConditionResult], fig_dir: Path) -> None:
    """Generate comparison bar charts for score, CD, and SSIM."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [NOTE] matplotlib not available — skipping figures.")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    condition_ids = list(dict.fromkeys(r.condition_id for r in results))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(condition_ids)}

    # ── Score: before vs after ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(condition_ids))
    w = 0.35

    # Average score_before and score_after per condition
    def _mean_by_cond(attr: str):
        vals = {}
        for cid in condition_ids:
            group = [getattr(r, attr) for r in results if r.condition_id == cid and not r.error]
            vals[cid] = float(np.mean(group)) if group else float("nan")
        return [vals[c] for c in condition_ids]

    before_scores = _mean_by_cond("score_before")
    after_scores  = _mean_by_cond("score_after")

    bars_b = ax.bar(x - w/2, before_scores, w, label="Before", color="#aaaaaa", alpha=0.8)
    bars_a = ax.bar(x + w/2, after_scores,  w, label="After",
                    color=[color_map[c] for c in condition_ids], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_ids, rotation=15, ha="right")
    ax.set_ylabel("Hallucination Score (lower = better)")
    ax.set_title("AGD Ablation — Hallucination Score Before vs After")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = fig_dir / "score_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figure saved -> {out}")

    # ── Chamfer Distance ─────────────────────────────────────────────────
    def _mean_geo_by_cond(attr: str):
        vals = {}
        for cid in condition_ids:
            group = [
                getattr(r.geo, attr)
                for r in results
                if r.condition_id == cid and r.geo and not r.error
            ]
            vals[cid] = float(np.mean(group)) if group else float("nan")
        return [vals[c] for c in condition_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    cd_vals = _mean_geo_by_cond("cd")
    ax.bar(condition_ids, cd_vals, color=[color_map[c] for c in condition_ids], alpha=0.9)
    ax.set_ylabel("Chamfer Distance (lower = better)")
    ax.set_title("AGD Ablation — Mean Chamfer Distance per Condition")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = fig_dir / "cd_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figure saved -> {out}")

    # ── Mean SSIM ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ssim_vals = _mean_geo_by_cond("mean_ssim")
    ax.bar(condition_ids, ssim_vals, color=[color_map[c] for c in condition_ids], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean SSIM (higher = better)")
    ax.set_title("AGD Ablation — Mean SSIM per Condition")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = fig_dir / "ssim_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figure saved -> {out}")


def _print_summary_table(results: List[ConditionResult], conditions: List[AblationCondition]) -> None:
    """Print an ASCII summary table averaged across all meshes per condition."""
    print(f"\n{'='*80}")
    print("  ABLATION SUMMARY  (averaged across all meshes)")
    print(f"{'='*80}")
    hdr = (f"  {'Condition':<18} {'Score B->A':>10} {'D Score':>8} "
           f"{'CD':>10} {'SSIM':>7} {'NCS':>6} {'Time(s)':>8}")
    print(hdr)
    print("  " + "-" * 76)

    for cond in conditions:
        group = [r for r in results if r.condition_id == cond.id and not r.error]
        if not group:
            print(f"  {cond.id:<18}  [no results]")
            continue

        sb  = float(np.mean([r.score_before for r in group]))
        sa  = float(np.mean([r.score_after  for r in group]))
        cd  = float(np.mean([r.geo.cd            for r in group if r.geo]))  if any(r.geo for r in group) else float("nan")
        ssim= float(np.mean([r.geo.mean_ssim      for r in group if r.geo]))  if any(r.geo for r in group) else float("nan")
        ncs = float(np.mean([r.geo.ncs            for r in group if r.geo]))  if any(r.geo for r in group) else float("nan")
        t   = float(np.mean([r.elapsed_s          for r in group]))

        sign = "v" if (sa - sb) < 0 else ("^" if (sa - sb) > 0 else "->")
        print(
            f"  {cond.id:<18} "
            f"{sb:.3f}->{sa:.3f} "
            f"{sa-sb:>+8.4f}{sign} "
            f"{cd:>10.5f}  {ssim:>6.4f}  {ncs:>6.3f}  {t:>8.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AGD Ablation Study — Stage 8"
    )
    parser.add_argument("--input-dir",        default="3d_samples",    help="Folder with mesh files")
    parser.add_argument("--meshes",            nargs="+", default=None,
                        help="Specific mesh filenames to test (from --input-dir)")
    parser.add_argument("--output-dir",        default="src/ablation_out", help="Output root directory")
    parser.add_argument("--gnn-checkpoint",    default="gnn_critic.pt",
                        help="GNN critic checkpoint path (skipped if not found)")
    parser.add_argument("--n-sample-pts",      type=int, default=4000,
                        help="Surface sample points for geometry metrics")
    parser.add_argument("--view-size",         type=int, default=128,
                        help="Render resolution (smaller = faster ablation)")
    parser.add_argument("--min-face-fraction", type=float, default=0.001)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve mesh paths
    if args.meshes:
        paths = [input_dir / m for m in args.meshes]
        paths = [p for p in paths if p.exists() and p.suffix.lower() in SUPPORTED_EXTS]
    else:
        paths = [
            p for p in sorted(input_dir.rglob("*"))
            if p.suffix.lower() in SUPPORTED_EXTS
        ]

    if not paths:
        raise SystemExit(f"No meshes found in {input_dir}")

    conditions = _build_conditions(args)

    print(f"\n{'='*62}")
    print(f"  AGD Ablation Study — {len(paths)} meshes x {len(conditions)} conditions")
    print(f"{'='*62}")
    for cond in conditions:
        print(f"  [{cond.id}]  {cond.description}")
    print()

    all_results: List[ConditionResult] = []

    for mesh_path in paths:
        print(f"\n  Mesh: {mesh_path.name}")
        print("  " + "-" * 40)
        for cond in conditions:
            sys.stdout.write(f"    [{cond.id}] ... ")
            sys.stdout.flush()
            result = _run_condition(mesh_path, cond, output_dir, args)
            all_results.append(result)
            if result.error:
                print(f"ERROR: {result.error}")
            else:
                delta = result.score_after - result.score_before
                sign  = "v" if delta < 0 else ("^" if delta > 0 else "->")
                geo_info = ""
                if result.geo:
                    geo_info = f"  CD={result.geo.cd:.5f}  SSIM={result.geo.mean_ssim:.4f}"
                print(f"{result.score_before:.3f}->{result.score_after:.3f} ({delta:+.4f}{sign})"
                      f"{geo_info}  [{result.elapsed_s:.1f}s]")

    # Output CSV
    csv_path = output_dir / "ablation_results.csv"
    _write_csv(all_results, csv_path)

    # Output figures
    fig_dir = output_dir / "ablation_figures"
    _plot_figures(all_results, fig_dir)

    # Print summary table
    _print_summary_table(all_results, conditions)

    print(f"  Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
