"""CLI for the iterative refinement loop.

Example
-------
python refinement/cli.py --input-file 3d_samples/balls.tri --output-dir agd_outputs/loop_demo

Notes
-----
* `.tri` is supported by `trimesh` in some setups; if it fails, use `.obj/.ply/.stl`.
* `--use-vlm` requires a local LLaVA model and will be much slower.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import sys

import trimesh

# Allow running as `python refinement/cli.py` (script mode) as well as
# `python -m refinement.cli` (module mode).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from critic import CriticConfig, build_critic
from lmm_detector import LLaVADetector

from refinement.loop import RefinementLoopConfig, run_refinement_loop
from refinement.optimizers import AnchoredLaplacianOptimizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".obj", ".ply", ".stl", ".tri"}


def _load_mesh(path: Path) -> Optional[trimesh.Trimesh]:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if not geoms:
            return None
        mesh = trimesh.util.concatenate(geoms)
    return mesh


def main() -> None:
    p = argparse.ArgumentParser(description="Iterative AGD refinement loop (render→detect→refine)")
    p.add_argument("--input-file", required=True, help="Mesh file (.obj/.ply/.stl)")
    p.add_argument("--output-dir", default="agd_outputs/refinement_loop", help="Output folder")

    # Loop
    p.add_argument("--outer-iters", type=int, default=5, help="Number of render→detect→refine rounds")

    # Rendering
    p.add_argument("--render-views", action="store_true", help="Render multi-view PNGs each outer iter")
    p.add_argument("--view-size", type=int, default=256)
    p.add_argument("--num-views", type=int, default=0, help="0=canonical 6, else dense scan")
    p.add_argument("--view-method", choices=["fibonacci", "grid"], default="fibonacci")
    p.add_argument("--fov-deg", type=float, default=45.0)
    p.add_argument("--edge-threshold", type=float, default=0.35)

    # Bias mapping
    p.add_argument("--consistency-threshold", type=float, default=0.85)
    p.add_argument("--view-extent", type=float, default=0.3)
    p.add_argument("--normal-threshold", type=float, default=0.2)

    # Grounding
    p.add_argument("--top-fraction", type=float, default=0.1)
    p.add_argument("--expand-hops", type=int, default=1)

    # Optimizer (default anchored Laplacian)
    p.add_argument("--inner-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--lambda-geo", type=float, default=0.7)
    p.add_argument("--lambda-distill", type=float, default=0.3)

    # Critic
    p.add_argument("--use-gnn", action="store_true")
    p.add_argument("--gnn-checkpoint", default=None)

    # VLM
    p.add_argument("--use-vlm", action="store_true")
    p.add_argument("--llava-model-path", default=None)
    p.add_argument("--llava-model-base", default=None)
    p.add_argument("--llava-max-new-tokens", type=int, default=256)

    args = p.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")
    if in_path.suffix.lower() not in SUPPORTED_EXTS:
        raise SystemExit(f"Unsupported extension {in_path.suffix}. Supported: {sorted(SUPPORTED_EXTS)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = _load_mesh(in_path)
    if mesh is None:
        raise SystemExit("No geometry found in mesh")

    critic = build_critic(CriticConfig(use_gnn=args.use_gnn, gnn_checkpoint=args.gnn_checkpoint))

    detector = None
    if args.use_vlm:
        if not args.llava_model_path:
            raise SystemExit("--llava-model-path required when --use-vlm")
        detector = LLaVADetector(
            model_path=args.llava_model_path,
            model_base=args.llava_model_base,
            max_new_tokens=args.llava_max_new_tokens,
        )

    cfg = RefinementLoopConfig(
        outer_iters=args.outer_iters,
        render_views=bool(args.render_views),
        view_size=args.view_size,
        num_views=args.num_views,
        view_method=args.view_method,
        fov_deg=args.fov_deg,
        edge_threshold=args.edge_threshold,
        consistency_threshold=args.consistency_threshold,
        view_extent=args.view_extent,
        normal_threshold=args.normal_threshold,
        top_fraction=args.top_fraction,
        expand_hops=args.expand_hops,
    )

    opt = AnchoredLaplacianOptimizer(
        inner_steps=args.inner_steps,
        lr=args.lr,
        lambda_geo=args.lambda_geo,
        lambda_distill=args.lambda_distill,
    )

    result = run_refinement_loop(
        mesh.copy(),
        out_dir,
        config=cfg,
        optimizer=opt,
        critic=critic,
        detector=detector,
    )

    out_mesh = out_dir / f"{in_path.stem}_agd{in_path.suffix}"
    result.mesh.export(out_mesh)

    logger.info("Saved refined mesh → %s", out_mesh)


if __name__ == "__main__":
    main()
