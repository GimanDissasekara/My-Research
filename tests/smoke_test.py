import sys
from pathlib import Path

# Add src/ to import path so modules are found from any working directory
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import trimesh
import numpy as np
from discriminator import compute_topology, compute_laplacian_metrics, compute_geometry_quality, hallucination_score
from renderer import render_views
from geometry_metrics import compute_geometry_error_report
from pathlib import Path

print("=== Discriminator smoke test ===")
mesh = trimesh.load('3d_samples/bunny.ply', force='mesh')
print(f"Loaded: {mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces")

topo  = compute_topology(mesh)
spec  = compute_laplacian_metrics(mesh)
qual  = compute_geometry_quality(mesh)
score = hallucination_score(topo, spec, quality=qual)

print(f"Topo:  components={int(topo['components'])}, genus={topo['genus']:.1f}")
print(f"Spec:  eigen_var={spec['eigen_variance']:.5f}, cot_var={spec['cotangent_variance']:.5f}, fiedler={spec['fiedler_value']:.5f}")
print(f"Qual:  ar_mean={qual['aspect_ratio_mean']:.3f}, degen={qual['degenerate_fraction']:.5f}, nm_edges={int(qual['non_manifold_edges'])}")
print(f"Score: {score:.4f}")

print("\n=== Renderer smoke test (bunny, 128px) ===")
out = render_views(mesh, Path("agd_outputs/smoke_render"), resolution=(128, 128))
print(f"Rendered {len(out)} image(s): {list(out.keys())[:6]}")

print("\n=== Geometry metrics smoke test ===")
# Slightly perturb mesh to simulate refinement
mesh2 = mesh.copy()
mesh2.vertices = mesh2.vertices + np.random.default_rng(0).normal(0, 0.001, mesh2.vertices.shape)
report = compute_geometry_error_report(mesh, mesh2, n_surface_points=2000)
print(report.summary_str())

print("\nAll tests passed.")
