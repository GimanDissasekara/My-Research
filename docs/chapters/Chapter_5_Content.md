# CHAPTER 5: ANALYSIS AND DESIGN

## 5.1 Introduction

This chapter presents the formal analysis and architectural design of the Adversarial Geometric Distillation (AGD) framework. Drawing from the theoretical foundations established in Chapter 3 and the approach specified in Chapter 4, this chapter translates the research hypothesis into a concrete system architecture. The chapter identifies the rationale for key design decisions (Section 5.2), presents the top-level three-module architecture (Section 5.3), details the data flow and interaction design between modules (Section 5.4), and positions the extension within the broader AI framework landscape (Section 5.5).

The AGD framework is architecturally composed of three cooperating modules: a **Preprocessing Module** that ingests and normalises 3D mesh data; an **ML Engine** that performs dual-signal hallucination detection using both graph-theoretic spectral analysis and a graph-convolutional adversarial critic; and an **Extended Module** that grounds detections into spatially precise 3D vertex weights and executes the closed-loop geometric refinement. The overarching design principle is modularity — each module exposes a well-defined interface and can be independently upgraded without modifying the others.

---

**[DIAGRAM PLACEHOLDER — Figure 5.1: Top-Level AGD System Architecture]**

*Attach the Mermaid diagram: `diagram_5_1_top_level.mmd`*

---

## 5.2 Rationale for the Design of the Extension

The design of the AGD framework is motivated by five explicit design rationale points, each directly traceable to a gap identified in the literature review and formalised in the theoretical analysis.

**Rationale 1 — Eliminate the Semantic Bottleneck by Replacing Language with Numbers.**
Hallo3D [6] routes LMM detections through natural-language negative prompts, which discard spatial coordinates. The AGD design eliminates the text channel entirely: LMM outputs are consumed as floating-point severity scores $s_v \in [0,1]$ and immediately projected into 3D vertex space by the Grounding Module. This architectural decision is the primary novel contribution of the design.

**Rationale 2 — Operate on Graph Structure, Not Rendered Pixels.**
Image-based detection methods can only identify hallucinations visible in rendered views. The AGD discriminator operates on the mesh's adjacency graph using the combinatorial Laplacian $L = D - A$, detecting connectivity-based hallucinations (disconnected components, excessive genus, spectral irregularities) that may be entirely invisible in any 2D projection. This design choice achieves geometry-completeness as defined in Section 3.6.

**Rationale 3 — Achieve Vertex-Level Spatial Precision.**
All prior correction approaches (score-space debiasing, negative prompting) apply corrections globally or diffusely. The AGD design produces a per-vertex weight map $w_i \in [w_{\min}, w_{\max}]$ that concentrates refinement energy on the specific vertices identified as anomalous by the critic and grounding modules, leaving clean vertices unaffected. This spatial selectivity is the direct architectural expression of the theoretical spatial precision property (Section 3.6).

**Rationale 4 — Ensure Theoretical Boundedness via the Distillation Anchor.**
The refinement optimizer is designed with a mandatory anchor term $D = \sum \|v_i - v_i^{(0)}\|^2$ that prevents the Laplacian smoothing from converging to degenerate geometry. This design ensures that the refined mesh cannot drift further from the original than controlled by the $\lambda$ hyperparameter, providing a formal quality guarantee absent from pure smoothing approaches.

**Rationale 5 — Modular, Backend-Agnostic Architecture.**
Each module (renderer, LMM detector, critic, grounder, optimizer) exposes a clean Python interface and can be replaced independently. This design allows the framework to scale: the software rasteriser can be upgraded to `nvdiffrast`, LLaVA can be replaced with GPT-4V, the heuristic critic can be replaced with a fully trained GNN, and the optimizer can be extended with topology-editing operations, all without modifying the orchestration logic in `agd_pipeline.py`.

## 5.3 Top-Level Architecture

The AGD system is decomposed into three primary modules that communicate through defined data contracts. The top-level decomposition is shown in Figure 5.1 above.

```
INPUT LAYER          ML ENGINE               EXTENDED MODULE
─────────────        ──────────────────────  ──────────────────────
Mesh Ingestion   →   Discriminator       →   Geometric Grounding
Normalisation        (topology + spectral)    (vertex weight map)
                     Adversarial Critic   →   Refinement Optimizer
                     (per-vertex scores)      (Laplacian + anchor)
                     LMM Detector        →   Multi-View Renderer
                     (per-view severity)      (5 buffers × N views)
```

### 5.3.1 Preprocessing Module

The Preprocessing Module is responsible for ingesting raw 3D mesh data, validating it, normalising it to a canonical form, and preparing the data structures required by the ML Engine. It is implemented in the ingestion and helper functions of `agd_pipeline.py`.

**Responsibilities:**

- **Format Parsing:** Loads `.obj`, `.ply`, and `.stl` files via `trimesh.load()`. Handles `trimesh.Scene` inputs (multi-geometry files) by concatenating all geometries into a single `trimesh.Trimesh` object.
- **Validity Checking:** Skips meshes with zero geometry or exceeding the vertex limit (500,000 vertices by default) to prevent memory exhaustion.
- **Normalisation:** Centres the mesh at its centroid and scales to a unit bounding sphere to ensure rendering distances and metric thresholds are scale-invariant.
- **Adjacency Extraction:** Extracts the vertex adjacency graph as a NetworkX `Graph` object (used by the discriminator) and a sparse edge-index tensor (used by the GNN critic).
- **Original Vertex Snapshot:** Takes a copy of `mesh.vertices` before any refinement, used as the anchor $v_i^{(0)}$ for the distillation term and for before/after metric comparison.

**Data Contract — Module Output:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `mesh` | `trimesh.Trimesh` | Validated, normalised mesh object |
| `adj_graph` | `nx.Graph` | Vertex adjacency graph for spectral analysis |
| `original_vertices` | `np.ndarray [V×3]` | Snapshot of pre-refinement vertex positions |
| `edge_index` | `torch.LongTensor [2×E]` | Edge index for GNN (if enabled) |

---

**[DIAGRAM PLACEHOLDER — Figure 5.2: Preprocessing Module Data Flow]**

*Attach the Mermaid diagram: `diagram_5_2_preprocessing.mmd`*

---

### 5.3.2 ML Engine

The ML Engine is the analytical core of the AGD framework. It combines three distinct detection mechanisms — global topological/spectral analysis (Discriminator), local per-vertex anomaly scoring (Adversarial Critic), and per-view semantic inspection (LMM Detector) — to produce a comprehensive hallucination signal. It is implemented across `discriminator.py`, `critic.py`, `lmm_detector.py`, `view_consistency.py`, and `renderer.py`.

#### 5.3.2.1 Discriminator Sub-module (`discriminator.py`)

The Discriminator computes global mesh quality metrics from the adjacency graph and combines them into a composite hallucination score $S_{\text{hall}} \in [0,1]$.

**Topological Analysis** (`compute_topology`):

- Connected components $\beta_0$ via `nx.number_connected_components`
- Euler characteristic: $\chi = V - E + F$
- Genus: $g = (2 - \chi)/2$

**Spectral Analysis** (`compute_laplacian_metrics`):

- Constructs $L = D - A$ from the adjacency graph
- Computes eigenvalue variance via trace identity: $\text{Var}(\lambda) = \text{trace}(L^2)/n - (\text{trace}(L)/n)^2$
- Computes Fiedler value $\lambda_2$ (second smallest eigenvalue) via dense solver (meshes ≤ 2000 vertices) or sparse `eigsh` (larger meshes)
- Computes cotangent-weighted Laplacian variance as a geometrically accurate spectral signature

**Geometry Quality Analysis** (`compute_geometry_quality`):

- Per-face aspect ratio (longest edge / 2× inradius)
- Degenerate face fraction (area < 1e-10)
- Non-manifold edge fraction (edges shared by ≠ 2 faces)
- Composite quality penalty: $0.4 \cdot P_{\text{ar}} + 0.3 \cdot P_{\text{deg}} + 0.3 \cdot P_{\text{nm}}$

**Composite Hallucination Score** (`hallucination_score`):

$$S_{\text{hall}} = 0.20\,P_{\text{cc}} + 0.30\,P_{\text{genus}} + 0.15\,P_{\text{var}} + 0.10\,P_{\text{fiedler}} + 0.10\,P_{\text{cot}} + 0.15\,P_{\text{quality}}$$

#### 5.3.2.2 Adversarial Critic Sub-module (`critic.py`)

The Adversarial Critic produces a per-vertex anomaly score vector $\mathbf{s} \in [0,1]^V$ that localises hallucinated regions at the vertex level.

**Heuristic Critic** (always active):

$$\text{anomaly}(v_i) = 0.5 \cdot \sigma(|z_{\text{deg}}(v_i)|) + 0.5 \cdot \sigma(|z_{\text{len}}(v_i)|)$$

where $z_{\text{deg}}$ and $z_{\text{len}}$ are z-scores of vertex degree and mean edge length respectively, and $\sigma$ is the sigmoid function.

**GNN Critic** (optional, requires `--use-gnn`):
A two-layer Graph Convolutional Network with the propagation rule $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$. Input features per vertex: $[\deg(v_i),\; \text{mean\_edge\_len}(v_i)]$. Output: scalar anomaly score per vertex. Trained via `gnn_train.py` using pseudo-labels from the heuristic critic (BCE loss, 20 epochs).

#### 5.3.2.3 Multi-View Renderer Sub-module (`renderer.py`)

Renders $N$ views of the mesh from camera positions distributed by the Fibonacci-sphere algorithm:

$$\phi_i = \arccos\!\left(1 - \frac{2(i+0.5)}{N}\right), \quad \theta_i = \frac{2\pi i}{\varphi_{\text{gold}}}$$

Each view produces **5 image buffers**: RGB (Phong shading), depth ($1/z$ perspective-correct), normal map (barycentric interpolation), silhouette (binary), edge/curvature map (depth-gradient threshold). Outputs are stored as PNG files and returned as a dictionary keyed by `{view_id}_{buffer_type}`.

#### 5.3.2.4 LMM Detector Sub-module (`lmm_detector.py`)

Local LLaVA v1.5/1.6 inspects each rendered RGB view and returns a structured severity score:

```json
{"severity": 0.7, "notes": "duplicated face detected on back view"}
```

The global LMM bias $b_{\text{LMM}} = \frac{1}{N}\sum_v s_v$ is computed and propagated to the Grounding Module alongside the per-view severity dictionary.

#### 5.3.2.5 View Consistency Sub-module (`view_consistency.py`)

Computes pairwise cosine similarity between opposite rendered views (front↔back, left↔right, top↔bottom) on 32×32 grayscale embeddings. Views with similarity below the threshold $\tau = 0.85$ are assigned severity $= 1 - \text{similarity}$, flagging cross-view structural inconsistencies (e.g., Janus faces).

**Data Contract — ML Engine Output:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `before_score` | `float [0,1]` | Global hallucination score before refinement |
| `critic_scores` | `np.ndarray [V]` | Per-vertex anomaly scores from critic |
| `view_scores` | `dict[str, float]` | Per-view severity (from consistency + LMM) |
| `global_bias` | `float` | Mean LMM severity across all views |
| `views_before` | `dict[str, Path]` | Paths to all rendered buffer images |

---

**[DIAGRAM PLACEHOLDER — Figure 5.3: ML Engine Internal Architecture]**

*Attach the Mermaid diagram: `diagram_5_3_ml_engine.mmd`*

---

### 5.3.3 Extended Module

The Extended Module receives the ML Engine's outputs and executes the two-phase closed-loop correction: first, geometric grounding (converting detection signals to a 3D vertex weight map), then the grounded refinement loop (iterative vertex optimization). It is implemented in `grounding.py` and `optimizer.py`.

#### 5.3.3.1 Geometric Grounding Sub-module (`grounding.py`)

The Grounding Module solves the core research problem: converting 2D per-view severity scores into 3D per-vertex weights without passing through a natural-language channel.

**Phase 1 — View-to-Vertex Projection** (`map_views_to_vertex_bias`):

For each view direction $d_v$ with severity $s_v$:
1. Project all vertices: $p_i = (v_i - \bar{v}) \cdot d_v$
2. Select vertices in the top $(1-\text{view\_extent})$ quantile of $p_i$ (closest to camera in view direction)
3. Filter by normal alignment: $n_i \cdot d_v \ge \text{normal\_threshold}$
4. Accumulate: $\text{bias}[i] = \max(\text{bias}[i],\; s_v)$ for selected vertices

**Phase 2 — Critic Score Integration** (`build_region_weights`):
1. Identify the top $k = \lfloor f \cdot V \rfloor$ vertices by critic anomaly score
2. Expand selection by 1-hop neighbourhood (graph BFS)
3. Blend with LMM bias: $w_i = \max(\text{bias}[i],\; \text{critic\_weight}[i])$
4. Apply global bias scaling: $w_i = w_{\min} + (w_{\max} - w_{\min}) \cdot w_i$

#### 5.3.3.2 Refinement Optimizer Sub-module (`optimizer.py`)

The Refinement Optimizer iteratively updates vertex positions using the weighted energy:

$$v_i \leftarrow v_i - \eta \cdot \left(\lambda \nabla L_{\text{geo}}(v_i) + (1-\lambda)\nabla D(v_i)\right) \cdot w_i$$

where:
- $L_{\text{geo}}(v_i) = \|v_i - \text{mean}(\mathcal{N}(i))\|$ — uniform Laplacian smoothing term
- $D(v_i) = \|v_i - v_i^{(0)}\|$ — distillation anchor term (prevents over-smoothing)
- $w_i$ — per-vertex weight from grounding module
- $\eta$ — learning rate (`--lr`, default 0.15)
- $\lambda$ — geometry/distillation trade-off (`--lambda-geo`, default 0.7)

The optimizer runs for `--iterations` steps (default 10). After refinement, the updated vertices are reapplied to the mesh and the mesh is re-analysed by the Discriminator to compute $S_{\text{hall}}^{\text{after}}$.

**Data Contract — Extended Module Output:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `refined_mesh` | `trimesh.Trimesh` | Geometrically corrected mesh |
| `after_score` | `float [0,1]` | Global hallucination score after refinement |
| `weights` | `np.ndarray [V]` | Final per-vertex weight map |
| `geo_report` | `GeometryErrorReport` | 10-metric before/after error report |
| `output_path` | `Path` | Saved `*_agd.{ext}` refined mesh file |

---

**[DIAGRAM PLACEHOLDER — Figure 5.4: Extended Module — Grounding and Refinement]**

*Attach the Mermaid diagram: `diagram_5_4_extended_module.mmd`*

---

## 5.4 Data Flow and Interaction Design

The complete data flow through the three modules is illustrated in Figure 5.5. The pipeline is orchestrated by `agd_pipeline.py`, which acts as the conductor — it instantiates each module, passes outputs between them, and manages the I/O lifecycle.

**Interaction Pattern 1 — Sequential with Optional Branches:**
The core flow (Preprocessing → Discriminator → Critic → Grounder → Optimizer → Discriminator) is strictly sequential. The rendering, view-consistency, and LMM branches are optional and activated by CLI flags (`--render-views`, `--use-vlm`). This design ensures the pipeline degrades gracefully: the full closed-loop refinement runs even without a GPU or LLaVA installation.

**Interaction Pattern 2 — Signal Fusion at the Grounding Module:**
The Grounding Module is the single convergence point for all detection signals. It receives three independent inputs — critic per-vertex scores, view-consistency severity dict, and LMM severity dict — and fuses them into a single $w_i$ vector using element-wise maximum. This max-fusion design means the most anomalous signal for each vertex wins, regardless of its source.

**Interaction Pattern 3 — Before/After Symmetry:**
The Discriminator is called twice — once on the original mesh and once on the refined mesh — using identical code paths. This symmetry is intentional: it ensures that the before/after scores are directly comparable and that any change in $S_{\text{hall}}$ is attributable entirely to the refinement step.

**Interaction Pattern 4 — Geometry Metric Comparison:**
After refinement, `compute_geometry_error_report()` from `geometry_metrics.py` compares the original and refined meshes on 10 metrics (Chamfer Distance, Hausdorff HD95, NCS, Angular Normal Error, Surface Roughness RMS, Curvature Wasserstein-1, Silhouette IoU, Depth RMSE, SSIM, Edge-IoU). These metrics are computed on sampled surface points (default: 6,000) to provide quantitative refinement quality assessment.

---

**[DIAGRAM PLACEHOLDER — Figure 5.5: Complete AGD Data Flow and Module Interaction]**

*Attach the Mermaid diagram: `diagram_5_5_full_dataflow.mmd`*

---

## 5.5 Extension Integration into the AI Framework

The AGD framework is designed for integration at three levels of the AI development ecosystem:

**Level 1 — Post-Processing Integration (Current Implementation):**
In its current form, AGD operates as a standalone post-processing refinement tool. Any 3D mesh file (`.obj`, `.ply`, `.stl`) produced by any generative system (TRELLIS, Hunyuan3D-2, BiDiff, DreamFusion, Magic3D) can be passed to the AGD pipeline via CLI. This integration level requires no modification to the upstream generative system and provides immediate value as a mesh quality assurance tool.

```bash
python agd_pipeline.py --input-file generated_mesh.obj \
    --render-views --num-views 36 --view-method fibonacci \
    --use-vlm --llava-model-path liuhaotian/llava-v1.5-7b
```

**Level 2 — In-Loop Integration (Planned — Threestudio):**
At this level, the AGD refinement loop replaces or augments the final optimization steps of a Threestudio/DreamFusion SDS pipeline. The geometric energy $E(x) = \lambda L_{\text{geo}} + (1-\lambda)D$ is added to the SDS loss: $L_{\text{total}} = L_{\text{SDS}} + E(x)$. This requires differentiable rendering (nvdiffrast or PyTorch3D) so that the vertex position gradients from the geometric energy can flow back into the SDS optimization.

**Level 3 — Benchmark Integration (Planned — Objaverse/GSO):**
At this level, the AGD pipeline is connected to standard benchmark datasets (Objaverse, Google Scanned Objects) to produce quantitative comparisons against baseline methods. This integration level produces the evaluation tables and FID-3D / Chamfer Distance comparisons required for academic publication.

**AI Framework Compatibility:**

| AI Framework | Integration Level | Method |
| :--- | :--- | :--- |
| TRELLIS | Level 1 (current) | Pass output `.glb` → AGD `--input-file` |
| Hunyuan3D-2 | Level 1 (current) | Pass output `.obj` → AGD `--input-file` |
| BiDiff | Level 1 (current) | Pass output `.ply` → AGD `--input-file` |
| DreamFusion (Threestudio) | Level 2 (planned) | Add $E(x)$ to SDS loss loop |
| Magic3D | Level 2 (planned) | Add $E(x)$ to coarse+fine stages |
| Objaverse evaluation | Level 3 (planned) | Batch evaluation + benchmark scripts |

## 5.6 Summary

This chapter has presented the complete analysis and design of the AGD framework. Five explicit design rationale points — eliminating the semantic bottleneck, operating on graph structure, achieving vertex-level precision, ensuring theoretical boundedness, and maintaining modular architecture — were traced directly to the theoretical gaps identified in Chapter 3.

The three-module architecture was specified in detail. The **Preprocessing Module** (`agd_pipeline.py`) handles mesh ingestion, validation, normalisation, and adjacency extraction. The **ML Engine** (`discriminator.py`, `critic.py`, `renderer.py`, `lmm_detector.py`, `view_consistency.py`) executes global topological/spectral analysis, per-vertex adversarial critic scoring, dense multi-view rendering, LMM semantic inspection, and view-consistency checking. The **Extended Module** (`grounding.py`, `optimizer.py`) fuses all detection signals into a spatially precise vertex weight map and executes the bounded, weighted geometric refinement loop.

The data flow and interaction design were characterised by four interaction patterns: sequential execution with optional branches, signal fusion at the Grounding Module via max-fusion, before/after score symmetry, and 10-metric geometry comparison. Three levels of AI framework integration were identified — post-processing (current), in-loop SDS integration (planned with Threestudio), and benchmark evaluation (planned with Objaverse/GSO).

Chapter 6 will present the implementation and experimental evaluation of this design, reporting quantitative results on the 13-mesh test dataset and identifying directions for future development.
