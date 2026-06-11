# EXP_06 — AGD Mesh Refinement Prototype

Working prototype of Adversarial Geometric Distillation (AGD) focused on refining existing meshes. The pipeline iterates through render, detect, ground, and optimize steps for .obj/.ply/.stl assets.

## Workflow (Current)
1) Load mesh and compute baseline metrics
2) Score anomalies with the critic (heuristic or GNN stub)
3) Render multi-view images
4) Detect issues (view-consistency, optional LLaVA)
5) Ground view signals into per-vertex weights
6) Optimize vertex positions for N iterations
7) Save refined mesh and report metric deltas

## Setup
```powershell
# Create and activate a local venv (PowerShell)
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Optional: install LLaVA (for VLM view scoring)
pip install -e LLaVA
```

```cmd
:: Create and activate a local venv (CMD)
python -m venv .venv
.venv\Scripts\activate.bat

:: Install dependencies
pip install -r requirements.txt

:: Optional: install LLaVA (for VLM view scoring)
pip install -e LLaVA
```

```bash
# Create and activate a local venv (Git Bash / WSL)
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Optional: install LLaVA (for VLM view scoring)
pip install -e LLaVA
```

## Run (Heuristic + View Consistency)
```powershell
python src/agd_pipeline.py --input-dir 3d_samples --output-dir src/agd_outputs --render-views
```

## Run (With Local LLaVA)
```powershell
python src/agd_pipeline.py --input-dir 3d_samples --output-dir src/agd_outputs --render-views --use-vlm --llava-model-path liuhaotian/llava-v1.5-7b
```

## Run (With GNN Critic)
```powershell
# Train the toy GNN critic from heuristic pseudo-labels
python src/gnn_train.py --input-dir 3d_samples --output gnn_critic.pt --epochs 20

# Use it in the pipeline
python src/agd_pipeline.py --input-dir 3d_samples --output-dir src/agd_outputs --render-views --use-gnn --gnn-checkpoint gnn_critic.pt
```

## Smoke Test
```powershell
python tests/smoke_test.py
```

## Stage 7: Outer Refinement Loop
```powershell
# Single mesh, up to 3 refinement loops (stops early if converged)
python src/agd_pipeline.py --input-file 3d_samples/bunny.ply --output-dir src/agd_outputs --render-views --refine-loops 3

# All meshes, 2 loops with cotangent Laplacian
python src/agd_pipeline.py --input-dir 3d_samples --output-dir src/agd_outputs --render-views --refine-loops 2 --use-cot-laplacian
```

## Stage 8: Ablation Study
```powershell
# Full ablation on all meshes (5 conditions)
python src/ablation.py --input-dir 3d_samples --output-dir src/ablation_out

# Fast ablation on specific meshes only
python src/ablation.py --input-dir 3d_samples --meshes bunny.ply janus.obj --output-dir src/ablation_out

# With GNN critic checkpoint included
python src/ablation.py --input-dir 3d_samples --gnn-checkpoint gnn_critic.pt --output-dir src/ablation_out
```

## Outputs
- Refined meshes saved to src/agd_outputs/ (suffix *_agd.*)
- Rendered views saved under src/agd_outputs/views/<mesh_name>/
- Metrics printed before -> after (score, variance, fiedler)

## Notes
- Supported mesh formats: .obj, .ply, .stl (trimesh in this environment does not support .tri).
- LLaVA weights are large; first run will download and cache.
- CPU inference is slow; CUDA is recommended.
- The current optimizer smooths geometry; it does not change topology.

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
