# CHAPTER 6: IMPLEMENTATION

## 6.1 Introduction

This chapter details the translation of the architectural design proposed in Chapter 5 into a functional, programmatic prototype. The Adversarial Geometric Distillation (AGD) pipeline was implemented as a modular Python-based toolset. This chapter outlines the development environment, hardware dependencies, module-specific coding strategies, and algorithmic workflows that facilitate the closed-loop geometric grounding process.

## 6.2 Overall Development Environment

The AGD framework was developed within a standard Python ecosystem to ensure broad compatibility with existing 3D generative toolkits (e.g., Threestudio). The environment is strictly modular, avoiding monolithic classes in favour of pure functions and independent pipeline stages. Execution is primarily CLI-driven via the main orchestrator script, allowing researchers to automate batch processing of mesh directories.

Key design decisions during development included the use of pure-NumPy software rasterisation to eliminate dependency on display servers or GPU rendering engines (such as OpenGL or EGL) during early prototyping, and the strict isolation of the LMM inference code to ensure the main pipeline remains unblocked during visual analysis.

## 6.3 Hardware and Software Platforms

The implementation relies heavily on robust third-party libraries for 3D mathematics, graph processing, and machine learning.

**Software Stack:**
- **Language:** Python 3.10+
- **3D Processing:** `trimesh` (mesh loading, normal computation, scene management)
- **Scientific Computing:** `numpy` (rasterisation, matrix math), `scipy` (sparse KD-Trees, sparse Laplacian eigensolvers)
- **Graph Analysis:** `networkx` (topological invariants, connected components)
- **Machine Learning:** `torch`, `torch_geometric` (GNN critic)
- **Vision-Language:** Hugging Face `transformers` (LLaVA v1.5/1.6 integration)
- **Metrics/Image:** `Pillow` (image I/O), `scikit-image` (SSIM)

**Hardware Profile:**
While the geometric pipeline (discriminator, critic, optimizer) executes efficiently on standard multi-core CPUs, the dense multi-view semantic inspection requires significant VRAM. The pipeline was tested and profiled on an NVIDIA RTX 3090/4090 class GPU (≥ 24 GB VRAM) to support the local loading of the 7B parameter LLaVA vision-language model.

## 6.4 Module-wise Implementation

The implementation is broken down into independent Python modules that correspond directly to the architecture proposed in Section 5.3.

### 6.4.1 Preprocessing and Orchestration (`agd_pipeline.py`)

The main entry point manages the system's I/O and loop execution. It uses `argparse` to handle hyperparameters (learning rate $\eta$, regularisation $\lambda$, view count $N$). It leverages `trimesh.load(force="mesh")` to ensure that multi-geometry scenes are cleanly concatenated into single verifiable surfaces.

### 6.4.2 ML Engine: Discriminator (`discriminator.py`)

This module implements the mathematical baseline for structural integrity. The most computationally sensitive component—eigenvalue extraction from the combinatorial Laplacian—is implemented using sparse matrix operations to support large meshes (up to 500,000 vertices).

```python
# Code Snippet 6.1: Efficient Eigenvalue Variance Extraction
import networkx as nx

def compute_spectral_variance(subgraph):
    L = nx.laplacian_matrix(subgraph).astype(float)
    n = L.shape[0]
    # Trace identity trick for variance without full eigendecomposition
    trace_l = float(L.diagonal().sum())
    trace_l2 = float(L.multiply(L).sum())
    mean = trace_l / float(n)
    variance = trace_l2 / float(n) - mean ** 2
    return variance
```

### 6.4.3 ML Engine: Adversarial Critic (`critic.py`)

The Critic evaluates per-vertex anomalies. The heuristic implementation computes robust statistical outliers by converting vertex degrees and mean edge lengths into z-scores, which are then passed through a sigmoid activation to output a normalised anomaly score $\in [0,1]$.

### 6.4.4 Extended Module: Grounding and Optimization (`optimizer.py`)

The geometric grounding translates the LMM severity scores and critic anomalies into a unified vertex weight vector $w$. The `refine_mesh` function then implements the energy minimisation loop.

```python
# Code Snippet 6.2: Grounded Energy Optimization Loop
def refine_mesh(mesh, weights, iterations=10, lr=0.15, lambda_geo=0.7, lambda_distill=0.3):
    vertices = mesh.vertices.copy()
    anchor = vertices.copy()
    
    for _ in range(iterations):
        # L_geo: Uniform Laplacian smoothing
        lap = uniform_laplacian(vertices, neighbors)
        
        # D: Distillation anchor term
        distill = vertices - anchor
        
        # Combined step gradient
        step = lambda_geo * lap + lambda_distill * distill
        
        # Weighted vertex update
        vertices = vertices - lr * step * weights
        
    mesh.vertices = vertices
    return mesh
```

## 6.5 Algorithms and Pseudocode

The fundamental workflow of the AGD system can be expressed mathematically via the following algorithmic structure. It highlights the closed-loop nature of the grounding process.

**Algorithm 6.1: AGD Grounding and Refinement**
```text
INPUT: Initial Mesh M0 = (V, E, F), Text Prompt Y
OUTPUT: Refined Mesh M1
1. Compute Baseline Topology (χ, β_0, g) and Spectrum (Var(λ), Fiedler)
2. C_scores ← Critic(V, E) // Per-vertex heuristic anomaly scoring
3. R_views ← Renderer(M0, N) // Dense Fibonacci-sphere camera sampling
4. V_scores ← LLaVADetector(R_views, Y) // JSON severity extraction
5. Initialize Weight Map W ← C_scores
6. FOREACH view v IN R_views:
7.     P ← Project(V, d_v) // Vertex projection on view direction
8.     W[TopQuantile(P) & NormalAligned] ← max(W, V_scores[v])
9. ENDFOR
10. Anchor V0 ← V
11. FOR i = 1 TO iterations:
12.     ∇L_geo ← ComputeLaplacian(V)
13.     ∇D ← (V - V0)
14.     V ← V - η * (λ * ∇L_geo + (1-λ) * ∇D) ⊙ W
15. ENDFOR
16. RETURN M1 ← (V, E, F)
```

## 6.6 Workflow Diagrams / Flowcharts

The execution flow from raw mesh to refined output is visualised below, highlighting the iterative loop between the LMM detection and geometric grounding.

**[IMAGE PLACEHOLDER FOR FIGURE 6.1]**<br>
*Suggested Visual: Insert `Figure_6_1_Workflow.png` demonstrating the parallel execution of the ML Engine sub-modules and the subsequent convergence at the Geometric Grounding and Optimizer blocks.*

## 6.7 Integration of Components

A critical implementation challenge was integrating text-based LMM outputs with 3D spatial data. This is resolved via the `grounding.py` module, which acts as the integration bus. By strictly enforcing a numerical interface (`float` severity scores) between the LMM parsing logic and the 3D projection logic, the semantic bottleneck is bypassed. The components integrate via "max-fusion"—ensuring that a severe hallucination detected by either the graph critic or the vision model exerts the necessary pull on the final weight vector $w$.

## 6.8 Testing During Development

Development was heavily driven by iterative validation using a curated set of 21 test meshes, located in the `3d_samples/` directory. This dataset includes verified "clean" geometry (e.g., the Stanford Bunny) alongside explicitly hallucinated models exhibiting Janus faces, duplicated limbs, and geometric spikes.

Continuous integration and functional validation were managed via a `smoke_test.py` script, which rapidly evaluated the entire pipeline utilizing the deterministic heuristic critic and bypassing the heavy LLaVA instantiation to ensure the math and logic paths remained stable after every refactoring phase.

## 6.9 Summary

This chapter documented the programmatic implementation of the AGD framework. The system was realized as a modular, CLI-driven Python pipeline, bridging 3D geometric processing libraries (Trimesh, NetworkX) with heavy machine learning infrastructure (PyTorch, Hugging Face). Detailed explanations and code snippets of the eigenvalue extraction and energy optimizer illustrated the translation of theory into code, while the core pseudocode algorithm laid out the step-by-step data transformations. The use of structured test datasets and dedicated smoke testing ensured the mathematical stability of the closed-loop refinement architecture.
