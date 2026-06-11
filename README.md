# EXP\_03 — Multi-View Renderer & Geometry Hallucination Detector

> **Research:** Adversarial Geometric Distillation (AGD) Framework  
> Detects geometry hallucinations in 3D-generated assets and generates differentiable loss functions for feed-forward refinement.

---

## Architecture Overview

```
Input Mesh (.ply / .obj / .tri)
        │
        ▼
┌──────────────────────────────┐
│   Multi-View Renderer        │  pyrender  → 24 RGB+Depth views
│   (renderer/)                │  (8 azimuth × 3 elevation)
└──────────────┬───────────────┘
               │  [renders: color + depth]
               ▼
┌──────────────────────────────┐
│  Geometry Hallucination      │  5 detectors → per-view scores
│  Detector (detector/)        │  + HallucinationReport
└──────────────┬───────────────┘
               │  [report]
               ▼
┌──────────────────────────────┐
│  Geometric Loss Computer     │  4 differentiable losses
│  (loss/)                     │  → LossOutput + anomaly masks
└──────────────┬───────────────┘
               │  L_total = L_SDS + λ · L_geom
               ▼
┌──────────────────────────────┐
│  Refinement Loop             │  Iterative render→detect→refine
│  (refinement/)               │  Pluggable optimiser interface
└──────────────────────────────┘
```

---

## Project Structure

```
EXP_04/
├── src/                    # Core AGD pipeline source code
│   ├── agd_pipeline.py     # Main pipeline entry point
│   ├── critic.py           # Heuristic & GNN critic
│   ├── discriminator.py    # Hallucination discriminator
│   ├── geometry_metrics.py # Geometry error metrics
│   ├── grounding.py        # Spatial grounding
│   ├── lmm_detector.py     # LLaVA-based VLM detector
│   ├── optimizer.py        # Mesh refinement optimizer
│   ├── renderer.py         # Multi-view software rasteriser
│   ├── view_consistency.py # View consistency checks
│   ├── gnn_train.py        # GNN training script
│   └── refinement/         # Iterative refinement loop
├── tests/                  # Tests & notebooks
├── docs/                   # Thesis documentation
│   ├── chapters/           # Markdown chapter content
│   ├── docx/               # Word documents
│   ├── pdf/                # PDF exports
│   └── reports/            # Research reports & notes
├── diagrams/               # All diagrams & figures
│   ├── figures/            # PNG/SVG exported figures
│   ├── drawio/             # Editable draw.io files
│   └── puml/               # PlantUML source files
├── scripts/                # Doc/visual generation scripts
├── presentations/          # Presentation files
├── data/                   # Loose 3D mesh data
├── 3d_samples/             # 3D test sample assets (gitignored)
├── agd_outputs/            # Pipeline output (gitignored)
└── archive/                # Old renders & temp outputs
```

---

## Modules

| Module | File | Description |
|---|---|---|
| **Renderer** | `src/renderer.py` | Software multi-view renderer (RGB/depth/normal/silhouette/edge) |
| **Critic** | `src/critic.py` | Heuristic or optional GNN vertex anomaly scoring |
| **Grounding** | `src/grounding.py` | Maps scores + view bias → per-vertex weights |
| **Optimizer** | `src/optimizer.py` | Mesh refinement (Laplacian + anchored distillation) |
| **Refinement** | `src/refinement/loop.py` | Iterative render→detect→ground→refine orchestration |
| **Entry** | `src/agd_pipeline.py` | End-to-end pipeline CLI |

---

## Hallucination Detectors

| # | Detector | Measures | Catches |
|---|---|---|---|
| 1 | **CLIP Semantic Consistency** | Cosine deviation between CLIP view embeddings | Janus faces, semantic drift |
| 2 | **Depth Discontinuity** | Squared Laplacian of depth map | Floaters, disconnected patches |
| 3 | **Normal Consistency** | Fraction of back-facing surface normals | Inside-out surfaces, flipped normals |
| 4 | **Silhouette Asymmetry** | Area ratio of opposite-view silhouettes | Asymmetric Janus geometry |
| 5 | **Edge Density Variance** | Canny edge density variance across views | Structural view inconsistency |

---

## Loss Functions

| Loss | Formula | Drives |
|---|---|---|
| `L_semantic` | `mean((1 - CLIP_cosine_sim)²)` | Semantic view consistency |
| `L_depth` | `mean(|∇²d|²)` — squared Laplacian | Surface smoothness, anti-floater |
| `L_normal` | `fraction of inverted normals` | Manifold consistency |
| `L_silhouette` | `mean(silhouette_asymmetry²)` | Symmetric geometry coverage |
| **`L_total`** | `Σ wᵢ · Lᵢ / Σwᵢ` | Combined adversarial geometric loss |

The total loss is designed to be added on top of SDS loss:
```
L_optimise = L_SDS + λ · L_total
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python src/agd_pipeline.py --input-dir 3d_samples --render-views
```

### 3. Run Refinement Loop
```bash
python src/refinement/cli.py --input-file 3d_samples/bunny.ply --render-views
```

### 4. Run Smoke Test
```bash
python tests/smoke_test.py
```

---

## Output Structure

```
output/
└── bunny/
    ├── renders/              # detect-only renders
    │   ├── view_000_az000_el-20_color.png
    │   ├── view_000_az000_el-20_depth.npy
    │   └── ...
    └── iter_01/              # refinement iteration renders
        ├── view_000_az000_el-20_color.png
        └── ...
```

---

## Plugging In Your Own Optimiser

Implement the small optimizer interface in `refinement/optimizers.py`:

```python
import numpy as np
import trimesh

class MyOptimizer:
    def reset(self, mesh: trimesh.Trimesh) -> None:
        ...

    def step(self, mesh: trimesh.Trimesh, weights: np.ndarray, outer_iter: int) -> trimesh.Trimesh:
        # update mesh.vertices here (in-place is fine)
        return mesh
```

Then run the loop:

```python
from pathlib import Path
from refinement.loop import RefinementLoopConfig, run_refinement_loop

result = run_refinement_loop(
    mesh,
    Path("agd_outputs/refinement_loop"),
    config=RefinementLoopConfig(outer_iters=5, render_views=True),
    optimizer=MyOptimizer(),
)
refined = result.mesh
```

---

## Dependencies

### Core Rendering
| Package | Version | Purpose |
|---|---|---|
| `pyrender` | 0.1.45 | OpenGL offscreen multi-view rendering |
| `trimesh` | 4.4.1 | Mesh loading (.ply, .obj, .tri) & processing |
| `pyglet` | 1.5.29 | pyrender backend |
| `PyOpenGL` | 3.1.7 | OpenGL Python bindings |

### Numerical / Image
| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.26.4 | Array math |
| `scipy` | 1.13.0 | Laplacian filter, spatial ops |
| `scikit-image` | 0.23.2 | SSIM fallback metric |
| `opencv-python` | 4.9.0 | Canny edges, colour conversion |
| `Pillow` | 10.3.0 | Image I/O |

### Deep Learning
| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.3.0 | Tensor ops, loss feed-forward |
| `torchvision` | 0.18.0 | Image transforms |
| `CLIP` | latest | Semantic view consistency |

### Visualisation
| Package | Version | Purpose |
|---|---|---|
| `matplotlib` | 3.9.0 | Loss curves, heatmaps |
| `rich` | 13.7.1 | Pretty CLI output |

### 3D Geometry (Optional)
| Package | Purpose |
|---|---|
| `open3d` | Normal estimation, ICP, curvature |
| `pyvista` | VTK mesh analysis, genus computation |
| `pytorch3d` | Differentiable rendering (Linux/GPU) |

---

## Sample Meshes Included

| File | Format | Vertices (approx) |
|---|---|---|
| `bunny.ply` | PLY | ~70K |
| `dragon.ply` | PLY | ~430K |
| `happy.ply` | PLY | ~540K |
| `hand.ply` | PLY | ~330K |
| `blade.ply` | PLY | ~1.1M |
| `horse.ply` | PLY | ~50K |
| `elepham.obj` | OBJ | ~90K |
| `ateneam.obj` | OBJ | ~30K |
| `venusm.obj` | OBJ | ~100K |
| `mba1.obj` | OBJ | medium |
| `mba2.obj` | OBJ | medium |
| `balls.tri` | TRI | small |
| `pots.tri` | TRI | small |
| `fullcsie.tri` | TRI | medium |

---

## Cite

If this pipeline helps your research:
```bibtex
@misc{agd2026,
  title  = {Adversarial Geometric Distillation: Hallucination Detection and Refinement for 3D Generative Models},
  author = {Your Name},
  year   = {2026},
}
```
"# Hallucination-Tracker" 
