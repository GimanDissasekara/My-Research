# AGD Prototype Documentation (Mesh-Only)

This workspace implements a **working prototype** of the *Adversarial Geometric Distillation (AGD)* refinement layer described in your proposal — scoped to **existing mesh files** (`.obj/.ply/.stl`) rather than full SDS text-to-3D generation.

It provides:
- A **closed-loop refinement loop** over mesh geometry (iterative vertex updates).
- A **geometric critic** (heuristic + optional GNN stub) that scores per-vertex anomalies.
- A **multi-view renderer** that produces views for detection.
- A **view-consistency detector** (feature similarity) to find cross-view inconsistencies.
- A **local VLM (LLaVA)** detector that inspects each view and outputs a severity score.
- A **geometric grounding module** that maps view-level detections into **coarse vertex regions**.

> Important: This is a prototype to validate the loop structure and “grounding into 3D weights”. It is not yet a full research-grade SDS/NeRF/SDF optimizer.

---

## 1) How This Maps to Your Research Proposal

Your proposal components vs what is currently in code:

### A) Generative pipeline (SDS-based synthesis)
- **Proposal:** “SDS-based synthesis of initial SDF/mesh from text/image input.”
- **This repo:** **Not implemented.**
  - Current entry point assumes you already have a mesh on disk (in `3d_samples/`).

### B) Multi-view renderer
- **Proposal:** Render diverse projections for spatial analysis.
- **This repo:** **Implemented** via trimesh scene rendering.
  - Code: [renderer.py](renderer.py)
  - Triggered by: `--render-views`
  - Output: PNGs saved under `agd_outputs/views/<mesh_name>/`

### C) LMM/VLM hallucination detector (Janus, duplicates, inconsistencies)
- **Proposal:** LMM scans renderings and detects structural issues.
- **This repo:** **Implemented (local LLaVA)** with a simple prompt and a JSON parser.
  - Code: [lmm_detector.py](lmm_detector.py)
  - Triggered by: `--use-vlm --llava-model-path <HF_ID>`
  - Output: per-view severities and notes (used internally to bias refinement).

What it does today:
- Runs LLaVA **per view**.
- Expects the model to return JSON like:
  ```json
  {"severity": 0.3, "notes": "duplicated face"}
  ```

What it does **not** do yet:
- No structured localization like bounding boxes or masks from VLM.

### D) Adversarial geometric critic (GNN + metrics)
- **Proposal:** GNN-based critic that detects hallucinations using topological metrics.
- **This repo:** **Partially implemented**.
  - Heuristic critic (works now): [critic.py](critic.py)
  - GNN stub (works structurally, but needs meaningful supervision): [critic.py](critic.py)
  - Training loop (pseudo-labeling from heuristic): [gnn_train.py](gnn_train.py)

Current critic behavior:
- Produces a **per-vertex anomaly score** based on degree/edge-length outliers (heuristic), or a toy GCN if enabled.

### E) Geometric grounding module (map detections to 3D regions)
- **Proposal:** Convert multi-modal observations into spatial constraints.
- **This repo:** **Implemented (coarse mapping)**.
  - Code: [grounding.py](grounding.py)

Grounding method used:
- Each view name corresponds to a direction (front/back/left/right/top/bottom).
- View-level severity becomes a **vertex bias** for vertices:
  - near the extreme extent in that direction (quantile-based), and
  - with normals facing the view direction.

This yields a *spatially-aware weight vector* `w_i` over vertices.

### F) Grounded refinement loop (optimize geometry with energy)
- **Proposal:** Optimize with: 
  - total loss $L_{total} = L_{SDS} + E(x)$
  - energy $E(x) = \lambda L_{geo} + (1-\lambda)D$.
- **This repo:** **Implemented (mesh-only)**.
  - Code: [optimizer.py](optimizer.py)

Current optimization is:
- Geometry smoothing term (uniform Laplacian):
  $$L_{geo} \propto \sum_i \|v_i - \text{mean}(\mathcal{N}(i))\|$$
- Distillation/anchor term (keep close to original):
  $$D \propto \sum_i \|v_i - v_i^{(0)}\|$$
- Weighted update only in grounded regions:
  $$v \leftarrow v - \eta\,(\lambda\,\nabla L_{geo} + (1-\lambda)\,\nabla D)\odot w$$

The “critic + grounding” provides the vertex weights $w$.

### G) Evaluation metrics
- **Proposal:** Laplacian variance, Betti numbers, manifold quality, etc.
- **This repo:** **Partially implemented**.
  - Laplacian spectral metrics + topology metrics in [discriminator.py](discriminator.py)
  - The pipeline prints before/after score changes.

---

## 2) Files and Responsibilities

- Entry point pipeline: [agd_pipeline.py](agd_pipeline.py)
- Spectral/topological metrics + baseline “discriminator”: [discriminator.py](discriminator.py)
- Critic (heuristic + optional GNN): [critic.py](critic.py)
- GNN training (pseudo-labels): [gnn_train.py](gnn_train.py)
- Multi-view rendering: [renderer.py](renderer.py)
- View-consistency detector (feature similarity): [view_consistency.py](view_consistency.py)
- Local LLaVA view detector: [lmm_detector.py](lmm_detector.py)
- Grounding (weights + coarse mapping): [grounding.py](grounding.py)
- Mesh refinement optimizer: [optimizer.py](optimizer.py)

---

## 3) How to Run

### A) Install dependencies
Your `requirements.txt` is in: [requirements.txt](requirements.txt)

If you already have `.venv` activated:
```powershell
pip install -r requirements.txt
```

You also installed LLaVA with:
```powershell
pip install -e LLaVA
```

### B) Run refinement using heuristic critic + view consistency
This runs without LLaVA:
```powershell
python agd_pipeline.py --input-dir 3d_samples --output-dir agd_outputs --render-views
```

### C) Run refinement using local LLaVA (Hugging Face model ID)
This will download the weights on first run and cache them:
```powershell
python agd_pipeline.py --input-dir 3d_samples --output-dir agd_outputs --render-views --use-vlm --llava-model-path liuhaotian/llava-v1.5-7b
```

If you want the GNN critic as well:
```powershell
python agd_pipeline.py --input-dir 3d_samples --output-dir agd_outputs --render-views --use-vlm --llava-model-path liuhaotian/llava-v1.5-7b --use-gnn
```

> Note: the GNN stub isn’t meaningfully trained by default; the heuristic critic is the “working baseline”.

---

## 4) Training the GNN Critic (Prototype)

This creates pseudo-labels from the heuristic critic and trains the toy GCN to reproduce them.

```powershell
python gnn_train.py --input-dir 3d_samples --output gnn_critic.pt --epochs 20
```

Then run:
```powershell
python agd_pipeline.py --input-dir 3d_samples --output-dir agd_outputs --render-views --use-gnn --gnn-checkpoint gnn_critic.pt
```

---

## 5) Minimal Code Walkthrough

### A) Main loop (pipeline)
The loop in [agd_pipeline.py](agd_pipeline.py) is:
1) Load mesh
2) Compute baseline metrics (topology + spectrum)
3) Score anomalies with critic
4) Render views
5) Compute view-consistency severities
6) (Optional) Run LLaVA per view
7) Ground signals into vertex weights
8) Optimize vertices for N iterations
9) Save refined mesh + report metric changes

### B) Grounding: view severity → vertex bias
Core idea in [grounding.py](grounding.py):
```python
bias = map_views_to_vertex_bias(mesh, view_scores)
weights = build_region_weights(mesh, critic_scores, bias_weights=bias)
```
This is the “semantic bottleneck removal” step in your proposal: view-level signals become a **spatial weight map** in 3D.

### C) Energy-based refinement update
In [optimizer.py](optimizer.py), each iteration does:
```python
lap = uniform_laplacian(vertices, neighbors)
distill = vertices - anchor
step = lambda_geo * lap + lambda_distill * distill
vertices = vertices - lr * step * weights
```

---

## 6) How Much This Covers Your Proposal (Honest Assessment)

### What is covered well (prototype-level)
- **Closed-loop refinement** on geometry (implemented and runnable).
- **Geometric grounding** from multi-view signals into 3D weights (implemented).
- **Two detectors** for hallucinations:
  - view-consistency (implemented)
  - local VLM (LLaVA) view inspection (implemented)
- **Adversarial critic abstraction** (heuristic now; GNN stub + training script included).

### What is partially covered
- “Topological metrics” are computed, but they are **not yet used as differentiable losses** beyond the Laplacian smoothing term.
- The GNN critic exists, but the training is currently **pseudo-supervised** (heuristic labels), not real annotations.

### What is not covered yet (research work remaining)
- **SDS-based generation** (text/image → 3D) and the full $L_{SDS}$ term.
- **True localization** from VLM (bounding boxes/masks) and precise 3D projection.
- A principled differentiable energy that explicitly encodes:
  - genus/Betti changes
  - self-intersection penalties
  - manifoldness constraints
- Quantitative evaluation on **Objaverse/GSO** with benchmark scripts.

### A practical “coverage estimate”
If we treat your proposal as 4 major blocks:
1) Detection (LMM + multi-view)
2) Critic (GNN + metrics)
3) Grounding (map to 3D constraints)
4) Refinement loop (optimize 3D)

This prototype covers:
- (1) **Yes** (view-consistency + LLaVA view scoring)
- (2) **Partial** (heuristic works; GNN is scaffold + toy training)
- (3) **Yes (coarse mapping)**
- (4) **Yes (mesh vertex optimization)**

The missing major piece is integrating **SDS** and making the energy/losses more aligned with topology/manifold constraints.

---

## 7) Recommended Next Upgrades (If You Want Research-Grade)

1) Replace coarse mapping with **projection-based grounding**
   - render depth, cast rays, map 2D regions to 3D vertices.

2) Use LLaVA vision tower embeddings for view-consistency features
   - stronger invariances than grayscale downsample.

3) Make the critic actually adversarial
   - train critic to separate clean vs hallucinated meshes,
   - train refinement to fool the critic.

4) Add mesh validity constraints
   - self-intersection penalties,
   - edge-flip/decimation,
   - manifoldness checks.

---

## 8) Common Pitfalls

- LLaVA model weights are large; first run may take time.
- CPU inference will be slow; CUDA is strongly recommended.
- Trimesh rendering uses pyglet; if rendering fails, run without `--render-views` or install graphics dependencies.

---

If you want, I can also generate a short “Methods” section you can paste into your proposal write-up (1–2 pages) describing exactly what this prototype implements and what remains.
