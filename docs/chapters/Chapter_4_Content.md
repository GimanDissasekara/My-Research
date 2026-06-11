# CHAPTER 4: APPROACH

## 4.1 Introduction

This chapter presents the complete system design of the Adversarial Geometric Distillation (AGD) framework — a closed-loop geometric grounding pipeline designed to mitigate hallucinations in text-to-3D generation. The theoretical foundations established in Chapter 3 are here translated into a concrete, implementable architecture. The chapter describes the research hypothesis and its inspirations (Section 4.2), the formal inputs and outputs of the extension (Section 4.3), the process workflow (Section 4.4), the technology stack (Section 4.5), the key features of the extension (Section 4.6), the target users and use scenarios (Section 4.7), and the positioning of the AGD framework within the broader AI body of knowledge (Section 4.8).

The AGD framework is designed as a post-processing refinement layer that can be attached to any existing text-to-3D generation pipeline. Rather than modifying the generative backbone, it intercepts the generated 3D mesh, analyses it for geometric hallucinations using an adversarial critic, grounds the detections into precise 3D spatial constraints, and iteratively refines the geometry. This design is deliberately modular: each component (renderer, detector, critic, grounder, optimizer) is independently replaceable, allowing the framework to evolve with advances in its constituent technologies.

## 4.2 Hypothesis and Its Inspiration

**Core Hypothesis:** A spatially-aware geometric feedback mechanism — grounded in spectral graph theory, algebraic topology, and adversarial learning — is both necessary and sufficient to eliminate the primary 2D-prior hallucinations and secondary LMM reasoning errors that afflict current text-to-3D generation systems, without requiring any modification to the generative backbone or re-prompting of a diffusion model.

This hypothesis is inspired by four converging observations from the literature and from the identified research gap:

**Inspiration 1 — The Semantic Bottleneck of Hallo3D [6].** Hallo3D demonstrated that LMMs can detect geometric hallucinations, but showed that routing corrections through natural language introduces a lossy compression of spatial information. The AGD hypothesis inverts this: LMM detections are consumed as numerical severity scores and immediately projected into 3D vertex space, bypassing the language channel entirely.

**Inspiration 2 — Spectral Completeness of Laplacian Analysis.** The foundational result of Chung (1997) that the Laplacian spectrum encodes all connectivity properties of a graph motivates the hypothesis that spectral and topological metrics are sufficient to characterise all connectivity-based hallucinations — including those not visible in any rendered view.

**Inspiration 3 — Laplacian Mesh Editing (Sorkine et al., 2004).** The result that quadratic Laplacian energies support bounded, targeted vertex deformation with a distillation anchor motivates the hypothesis that geometric refinement can be both selective (confined to hallucinated vertices) and safe (bounded by the anchor term).

**Inspiration 4 — GCN Equivariance (Kipf and Welling, 2017).** The equivariance of GCN propagation to graph isomorphisms motivates the hypothesis that a graph-convolutional critic trained on one set of meshes will generalise to meshes of arbitrary topology — making the approach scalable without per-mesh retraining.

## 4.3 Inputs and Outputs of the Extension

### 4.3.1 Inputs

| Input | Format | Source | Description |
| :--- | :--- | :--- | :--- |
| 3D Mesh | `.obj`, `.ply`, `.stl` | 3D generator or dataset | The initial mesh to be analysed and refined. May contain geometric hallucinations. |
| Text Prompt | String (optional) | User | Natural language description of the desired object, used by the upstream generator. |
| `--num-views` $N$ | Integer (default: 36) | CLI | Number of camera positions for multi-view rendering (Fibonacci-sphere or grid). |
| `--view-method` | `fibonacci` or `grid` | CLI | Camera distribution strategy: Fibonacci-sphere for uniform coverage, grid for azimuth-elevation sweep. |
| `--lr` | Float (default: 0.001) | CLI | Learning rate for the vertex refinement optimizer. |
| `--iterations` | Integer (default: 30) | CLI | Number of refinement iterations per mesh. |
| `--lambda-geo` $\lambda$ | Float $\in [0,1]$ (default: 0.7) | CLI | Trade-off between Laplacian smoothness loss and distillation anchor term. |
| `--edge-threshold` | Float (default: 0.1) | CLI | Threshold for the edge curvature detection buffer. |
| `--fov-deg` | Float (default: 60°) | CLI | Camera field of view for the software rasteriser. |

### 4.3.2 Outputs

| Output | Format | Description |
| :--- | :--- | :--- |
| Refined 3D Mesh | `*_agd.obj/.ply/.stl` | Structurally corrected mesh saved alongside the input. |
| Hallucination Score (Before) | Float $\in [0,1]$ | Composite topological + spectral score before refinement. |
| Hallucination Score (After) | Float $\in [0,1]$ | Composite score after refinement — lower is better. |
| RGB Renders | PNG per view | Phong-shaded renders from all $N$ camera positions. |
| Depth Maps | PNG per view | Perspective-correct $1/z$ depth buffer. |
| Normal Maps | PNG per view | Per-pixel barycentric-interpolated vertex normals. |
| Silhouette Masks | PNG per view | Binary object boundary mask. |
| Edge/Curvature Maps | PNG per view | Sharp geometric discontinuity detection buffer. |
| Geometry Error Report | Console table | 10-metric report: Chamfer Distance, Hausdorff (HD95), Normal Consistency Score (NCS), Angular Normal Error (°), Surface Roughness RMS, Curvature Wasserstein-1, Silhouette IoU, Depth RMSE, SSIM, Edge-IoU. |
| View Metadata | JSON | Azimuth / elevation angles for all $N$ rendered views. |

## 4.4 Process Workflow for the Extension

The AGD pipeline operates in six sequential stages, forming a closed-loop refinement cycle.

### Stage 1 — Mesh Ingestion and Pre-processing

The pipeline loads the input mesh using `trimesh`. Meshes exceeding a configurable vertex limit (default: 500,000) are skipped with a warning to prevent memory exhaustion. The mesh is centred and normalised to a unit bounding sphere. Vertex normals are computed or loaded, and the adjacency graph is extracted as a sparse matrix for spectral analysis.

### Stage 2 — Dense Multi-View Rendering

The custom software rasteriser distributes $N$ camera positions uniformly on the viewing sphere using the **Fibonacci-sphere algorithm**:

$$ \phi_i = \arccos\!\left(1 - \frac{2(i+0.5)}{N}\right), \quad \theta_i = \frac{2\pi i}{\varphi_{\text{gold}}}, \quad \varphi_{\text{gold}} = \frac{1+\sqrt{5}}{2} $$

Each camera position generates five image buffers via pure-NumPy rasterisation:

1. **RGB** — Phong shading (ambient $k_a$, diffuse $k_d$, specular $k_s$ with Blinn–Phong highlights)
2. **Depth** — Perspective-correct $1/z$ interpolation
3. **Normal Map** — Per-pixel barycentric vertex-normal interpolation
4. **Silhouette** — Binary object mask
5. **Edge/Curvature Map** — Discontinuity detection via depth gradient thresholding

### Stage 3 — LMM Hallucination Detection

A locally deployed **LLaVA v1.5/v1.6** vision-language model inspects each rendered RGB view. The prompt requests a JSON-structured response of the form `{"severity": <float>, "notes": "<string>"}`. The severity score $s_v \in [0,1]$ encodes the detector's confidence that view $v$ shows a structural hallucination. A global **LMM bias** is computed as the mean severity across all views:

$$ b_{\text{LMM}} = \frac{1}{N} \sum_{v=1}^{N} s_v $$

View-level severity scores are retained for the grounding stage.

### Stage 4 — Adversarial Geometric Critic

The critic operates directly on the mesh graph, producing a per-vertex anomaly score without any 2D rendering.

**Heuristic critic (always active):**

$$ \text{anomaly}(v_i) = \max\!\left(\left|z_{\text{deg}}(v_i)\right|,\; \left|z_{\text{len}}(v_i)\right|\right) $$

where $z_{\text{deg}}$ and $z_{\text{len}}$ are the degree and mean-edge-length z-scores of vertex $v_i$.

**GNN critic (optional):** Two `GCNConv` layers map vertex features $[\deg(v_i),\; \text{mean\_edge\_len}(v_i)]$ to a scalar anomaly score per vertex via a trained GCN with ReLU activations.

**Discriminator (global):** Computes the composite hallucination score from four topological and spectral penalty terms:

$$ S_{\text{hall}} = 0.25\,P_{\text{cc}} + 0.35\,P_{\text{genus}} + 0.25\,P_{\text{var}} + 0.15\,P_{\text{fiedler}} $$

### Stage 5 — Geometric Grounding

The grounding module fuses the critic's per-vertex anomaly scores and the LMM's per-view severity scores into a unified **vertex weight map** $w_i \in [w_{\text{min}}, w_{\text{max}}]$.

For each view $v$ with severity $s_v > 0$, the grounding module:
1. Projects all vertex positions onto the view direction $d_v$
2. Selects vertices in the top $(1 - \text{view\_extent})$ quantile of the projection (facing the camera)
3. Filters by normal alignment: $n_i \cdot d_v \ge \text{normal\_threshold}$
4. Accumulates $\max(w_i, s_v)$ for each selected vertex

The critic scores are normalised and blended with the LMM-grounded weights:

$$ w_i = \alpha \cdot w_i^{\text{LMM}} + (1-\alpha) \cdot w_i^{\text{critic}} $$

### Stage 6 — Grounded Refinement Loop

The refinement loop iterates for `--iterations` steps. At each step, the combined energy gradient is computed:

$$ \nabla E(v_i) = \lambda \nabla L_{\text{geo}}(v_i) + (1-\lambda) \nabla D(v_i) $$

where $L_{\text{geo}} = \|v_i - \text{mean}(N(i))\|$ (uniform Laplacian) and $D = \|v_i - v_i^{(0)}\|$ (distillation anchor). Vertex positions are updated as:

$$ v_i \leftarrow v_i - \eta \cdot \nabla E(v_i) \cdot w_i $$

Vertices with low $w_i$ (clean geometry) receive negligible updates; vertices with high $w_i$ (hallucinated regions) receive the full gradient step.

## 4.5 Technologies Identified

| Category | Technology | Version | Role in AGD |
| :--- | :--- | :--- | :--- |
| **3D Geometry** | Trimesh | ≥ 3.20 | Mesh I/O, topology extraction, vertex normals, surface sampling |
| **Rendering** | NumPy (custom rasteriser) | ≥ 1.24 | Software rasterisation — no GPU/display server dependency |
| **Spectral Analysis** | SciPy (sparse, cKDTree) | ≥ 1.10 | Laplacian construction, KD-tree nearest-neighbour for CD/HD |
| **Graph ML** | PyTorch + PyTorch Geometric | ≥ 2.0 / ≥ 2.3 | GCN critic (GCNConv layers) for vertex anomaly scoring |
| **VLM Detection** | LLaVA v1.5/1.6 (local) | — | Per-view hallucination severity detection |
| **Image Processing** | Pillow | ≥ 9.0 | PNG I/O for rendered views |
| **Structural Similarity** | scikit-image | ≥ 0.21 | SSIM computation between before/after rendered views |
| **Graph Analysis** | NetworkX | ≥ 3.0 | Connected component counting ($\beta_0$) |
| **Scientific Computing** | NumPy | ≥ 1.24 | All matrix operations in renderer and optimizer |
| **3D Generation (planned)** | Threestudio + PyTorch | — | SDS-based text-to-3D synthesis backbone |
| **Differentiable Rendering (planned)** | nvdiffrast / PyTorch3D | — | GPU-accelerated differentiable rendering for gradient flow |
| **Datasets (planned)** | Objaverse, GSO | — | Benchmark evaluation and GNN training ground truth |
| **Hardware Target** | NVIDIA RTX 3090/4090 (≥ 24 GB VRAM) | — | LMM inference + differentiable 3D optimization |

## 4.6 Features of the Technological Extension

**Feature 1 — Geometry-Complete Hallucination Detection**
The discriminator operates directly on the mesh graph's adjacency structure, detecting hallucinations that are invisible in any rendered view (occluded components, interior voids, non-manifold patches). This property — geometry-completeness — is absent from all image-based detection methods including Hallo3D.

**Feature 2 — Dense Multi-View Scanning (6 to 100+ Views)**
The Fibonacci-sphere camera scheduler provides provably uniform coverage of the viewing sphere for any $N$. Increasing $N$ from 6 to 100 increases hallucination detection probability monotonically without modifying the detection algorithms — a scalability guarantee absent from fixed-view pipelines.

**Feature 3 — Spatially Precise Vertex-Level Grounding**
The grounding module produces a per-vertex weight map $w_i$ that concentrates refinement energy on hallucinated vertices with spatial precision at the resolution of individual vertices. All prior correction approaches apply corrections globally or through diffuse 2D re-conditioning.

**Feature 4 — Theoretical Boundedness via Self-Distillation**
The anchor term $D = \sum \|v_i - v_i^{(0)}\|$ prevents the refinement loop from converging to degenerate geometry. This provides a formal guarantee — absent from pure Laplacian smoothing — that the refined mesh cannot drift further from the original than controlled by $\lambda$.

**Feature 5 — Dual-Signal Complementary Detection**
The pipeline fuses global topological/spectral signals (discriminator) with local per-vertex signals (critic) into a single weight map. Each signal type covers the other's blind spots: global signals detect multi-component hallucinations that local analysis misses; local signals detect individual vertex outliers invisible to global metrics.

**Feature 6 — Modular, Backend-Agnostic Architecture**
Each pipeline stage is independently replaceable. The renderer can be upgraded to `nvdiffrast`; the VLM can be replaced with GPT-4V; the GNN critic can be retrained on new datasets; the optimizer can be extended with topology-editing operations (hole filling, remeshing). No stage is coupled to any specific implementation of another.

**Feature 7 — Single-File and Batch Processing Modes**
The `--input-file` flag processes a single target mesh; `--input-dir` processes an entire dataset directory. Both modes produce identical outputs and are suitable for individual inspection or large-scale benchmark evaluation respectively.

**Feature 8 — Advanced Rendering Pipeline (5 Buffers per View)**
Each rendered view produces RGB (Phong shading), perspective-correct depth, interpolated normal map, binary silhouette, and edge/curvature map — enabling metric computation across all five geometric error dimensions simultaneously.

## 4.7 Target Users and Use Scenarios

**Target User Group 1 — 3D Generative AI Researchers**
Researchers evaluating or improving text-to-3D generation systems (DreamFusion, Magic3D, Fantasia3D, TRELLIS, Hunyuan3D) can use the AGD pipeline as a quantitative benchmark tool. The 10-metric error report and composite hallucination score provide a standardised, reproducible measure of geometric quality that supplements subjective visual evaluation.

*Scenario:* A researcher generates 50 meshes from DreamFusion and 50 from Magic3D, runs both sets through the AGD pipeline, and compares the distributions of Chamfer Distance, hallucination scores, and genus counts to quantify which system produces geometrically superior outputs.

**Target User Group 2 — 3D Asset Creators (VR/AR/Gaming)**
Professionals who use AI tools to generate 3D assets for VR, AR, and gaming environments require structurally sound meshes with low genus, no floating components, and manifold topology. The AGD refinement loop can be integrated as an automatic post-processing step in asset creation pipelines.

*Scenario:* A game studio's pipeline generates character meshes using a generative model. The AGD module automatically detects and refines Janus-face artefacts and floating geometry fragments before the mesh is handed off to the rigging and texturing teams, reducing manual correction workload.

**Target User Group 3 — Medical and Engineering 3D Reconstruction**
In medical imaging and industrial inspection, 3D reconstructions must be topologically valid (genus 0, single component). The AGD discriminator's topological metrics (Euler characteristic, genus, Betti numbers) provide automated quality certificates for reconstructed meshes.

*Scenario:* A medical imaging lab uses the AGD discriminator to screen reconstructed bone meshes for topological defects before surgical simulation — flagging meshes with $\beta_0 > 1$ or $g > 0$ for manual review.

**Target User Group 4 — Academic Researchers in Geometric Deep Learning**
The GNN critic and its training infrastructure (`gnn_train.py`) provide a research platform for developing and evaluating graph-convolutional anomaly detection methods on 3D mesh data — applicable beyond hallucination detection to any mesh quality assessment task.

## 4.8 Positioning the Extension within the AI Body of Knowledge

The AGD framework occupies a specific and novel position at the intersection of four established sub-fields of Artificial Intelligence and Computer Graphics:

**4.8.1 Relation to Generative AI (Text-to-3D)**
AGD is positioned as a post-hoc corrective extension to the text-to-3D generation paradigm initiated by DreamFusion [2] and extended by Magic3D [5], Fantasia3D [4], and ProlificDreamer [3]. It does not replace the generative backbone but augments it with a geometric quality assurance layer. In the taxonomy of generative AI systems, AGD contributes to the sub-field of *output refinement and quality control* for 3D generative models.

**4.8.2 Relation to Multi-Modal AI**
AGD extends the multi-modal paradigm of Hallo3D [6] by fundamentally restructuring the information flow: rather than using LMM outputs as text prompts for a diffusion model, AGD uses LMM outputs as spatial signals for a geometric optimizer. This positions AGD within the sub-field of *grounded multi-modal reasoning* — where multi-modal model outputs are grounded into structured, non-linguistic representations.

**4.8.3 Relation to Geometric Deep Learning**
The GCN-based critic is a direct contribution to the application of geometric deep learning to 3D mesh quality assessment. The use of GCNConv layers for per-vertex anomaly scoring on mesh graphs aligns with the broader research direction of learning-based mesh analysis initiated by MeshCNN, PointNet, and related works. AGD demonstrates that adversarial geometric critics trained with topology-derived pseudo-labels can provide useful spatial signals for mesh refinement without requiring explicit ground-truth anomaly annotations.

**4.8.4 Relation to Computational Geometry and Topology**
The discriminator's use of Euler characteristic, genus, and Betti numbers as hallucination metrics connects the AGD framework to the classical body of computational topology, specifically to the field of *topological data analysis* (TDA). The spectral metrics (Laplacian eigenvalue variance, Fiedler value) connect to *spectral graph theory* and its applications in mesh processing. AGD is the first framework, to the author's knowledge, to use these topological and spectral quantities directly as components of a machine learning feedback signal in a 3D generation pipeline.

**4.8.5 Novel Contribution**
AGD's primary novel contribution to the AI body of knowledge is the closed-loop geometric grounding mechanism: the architectural pattern of (a) detecting hallucinations using a combination of LMM inspection and graph-theoretic analysis, (b) grounding detections to individual vertices via projection-based spatial assignment, and (c) feeding vertex-level weights directly into a geometric energy minimisation loop — without any intermediate conversion to natural language. This pattern had not been instantiated in any prior work in the text-to-3D literature as of the time of writing.

## 4.9 Summary

This chapter has presented the complete approach of the Adversarial Geometric Distillation (AGD) framework. The research hypothesis — that spatially-aware geometric feedback grounded in spectral and topological 3D metrics is sufficient to eliminate hallucinations without re-prompting — was motivated by four distinct theoretical inspirations from the literature. The formal inputs and outputs were specified, covering nine CLI-configurable input parameters and eleven categories of output artefacts.

The six-stage process workflow — Ingestion, Dense Multi-View Rendering, LMM Detection, Adversarial Critic Scoring, Geometric Grounding, and Grounded Refinement — was described in precise mathematical and algorithmic terms, with each stage's role in closing the theoretical research gap made explicit. The technology stack of 13 components was catalogued, and eight key features of the extension — including geometry-completeness, dense multi-view scalability, vertex-level spatial precision, theoretical boundedness, and dual-signal complementarity — were defined and distinguished from prior approaches.

Target user groups spanning 3D generative AI research, VR/AR/gaming asset creation, medical imaging, and geometric deep learning were identified with concrete use scenarios. The AGD framework was positioned within the AI body of knowledge at the intersection of generative AI, multi-modal reasoning, geometric deep learning, and computational topology, with its primary novel contribution identified as the closed-loop geometric grounding pattern.

Chapter 5 will present the implementation details, experimental evaluation, and quantitative results of the AGD framework on the collected mesh dataset.
