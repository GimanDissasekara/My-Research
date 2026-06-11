"""Iterative refinement loop: render → detect → ground → refine.

This is the missing orchestration layer described in your architecture diagram.
It reuses the existing project modules (renderer, critic, grounding, optimizer,
VLM detector, and view-consistency checks).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from critic import CriticConfig, build_critic
from grounding import build_region_weights, map_views_to_vertex_bias
from lmm_detector import LLaVADetector, detect_issues
from renderer import render_views
from view_consistency import check_view_consistency

from .optimizers import AnchoredLaplacianOptimizer, MeshOptimizer


@dataclass
class RefinementLoopConfig:
    outer_iters: int = 5

    # Rendering / view sampling
    render_views: bool = True
    view_size: int = 256
    num_views: int = 0
    view_method: str = "fibonacci"
    fov_deg: float = 45.0
    edge_threshold: float = 0.35

    # View → vertex bias mapping
    consistency_threshold: float = 0.85
    view_extent: float = 0.3
    normal_threshold: float = 0.2

    # Grounding
    top_fraction: float = 0.1
    expand_hops: int = 1
    min_weight: float = 0.1
    max_weight: float = 1.0


@dataclass
class RefinementLoopResult:
    mesh: trimesh.Trimesh
    weights_last: np.ndarray


def run_refinement_loop(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    *,
    config: RefinementLoopConfig,
    optimizer: MeshOptimizer,
    critic=None,
    detector: Optional[LLaVADetector] = None,
) -> RefinementLoopResult:
    """Run iterative render→detect→refine and return the final refined mesh."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if critic is None:
        critic = build_critic(CriticConfig())

    optimizer.reset(mesh)

    weights_last = np.ones((mesh.vertices.shape[0],), dtype=float)

    for outer_iter in range(int(max(1, config.outer_iters))):
        bias_weights = None
        global_bias = 0.0

        if config.render_views:
            iter_dir = output_dir / "views" / f"iter_{outer_iter:03d}"
            views = render_views(
                mesh,
                iter_dir,
                resolution=(config.view_size, config.view_size),
                n_views=config.num_views,
                view_method=config.view_method,
                fov_deg=config.fov_deg,
                edge_threshold=config.edge_threshold,
            )

            rgb_views = {
                k[: -len("_rgb")]: v
                for k, v in views.items()
                if k.endswith("_rgb")
            }

            if rgb_views:
                # Consistency bias (canonical view pairs only)
                consistency = check_view_consistency(
                    rgb_views,
                    similarity_threshold=config.consistency_threshold,
                )
                bias_weights = map_views_to_vertex_bias(
                    mesh,
                    consistency.view_scores,
                    view_extent=config.view_extent,
                    normal_threshold=config.normal_threshold,
                )

                # VLM bias (optional)
                if detector is not None:
                    detection = detect_issues(rgb_views, detector=detector)
                    global_bias = detection.global_bias
                    lmm_bias = map_views_to_vertex_bias(
                        mesh,
                        detection.view_scores,
                        view_extent=config.view_extent,
                        normal_threshold=config.normal_threshold,
                    )
                    bias_weights = (
                        lmm_bias if bias_weights is None else np.maximum(bias_weights, lmm_bias)
                    )

        scores = critic.score(mesh)
        weights_last = build_region_weights(
            mesh,
            scores,
            top_fraction=config.top_fraction,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
            expand_hops=config.expand_hops,
            global_bias=global_bias,
            bias_weights=bias_weights,
        )

        mesh = optimizer.step(mesh, weights_last, outer_iter=outer_iter)

    return RefinementLoopResult(mesh=mesh, weights_last=weights_last)


def default_optimizer() -> AnchoredLaplacianOptimizer:
    return AnchoredLaplacianOptimizer()
