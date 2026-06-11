# CHAPTER 2: LITERATURE REVIEW

## 2.1 Introduction

Text-to-3D generation has emerged as a pivotal research domain at the intersection of generative modelling, differentiable rendering, and 3D computer vision. This chapter reviews the key methodological developments that preceded and motivated the Adversarial Geometric Distillation (AGD) framework. The review is structured chronologically to trace the progression from point-cloud synthesis to score-distillation-based pipelines, followed by a comparative analysis of existing methods and a formal identification of the research gap that AGD addresses.

## 2.2 Chronological Review of Prior Work

### 2.2.1 Early Development of Text-to-3D Generation

The earliest data-driven approaches to 3D object generation operated directly on volumetric and point-cloud representations. Fan et al. [10] proposed a **Point Set Generation Network** that reconstructed 3D point clouds from single-view images using a multi-layer perceptron decoder, establishing the first end-to-end learned pipeline for 3D shape synthesis. While geometrically expressive, point clouds lacked surface topology and were incompatible with downstream rendering engines.

**Voxel and Implicit Representations** subsequently emerged as alternatives. Schwarz et al. [11] introduced VoxGRAF, which used sparse voxel grids for efficient 3D-aware image synthesis within a GAN framework. However, the cubic memory complexity of voxel representations limited resolution to low-detail outputs. Simultaneously, Neural Radiance Fields (NeRF) demonstrated that continuous volumetric scenes could be learned purely from posed 2D images via volume rendering, enabling high-fidelity view synthesis but requiring per-scene optimisation.

**Zero-Shot Text-Guided Generation** marked the first integration of language models with 3D synthesis. Dream Fields [12] guided NeRF optimisation using CLIP similarity between rendered views and text prompts, demonstrating that large vision-language models could serve as effective 3D supervision signals. Concurrently, Latent-NeRF [1] introduced shape-guided generation by conditioning NeRF optimisation on latent shape codes, improving geometric consistency over unconstrained CLIP-based approaches.

**Point-E** [13] and **Shap-E** [14] (OpenAI) subsequently presented diffusion-model-based pipelines for direct 3D generation, producing point clouds and implicit functions from text prompts in seconds rather than minutes. However, both systems produced geometrically coarse outputs with limited surface detail.

**[DIAGRAM PLACEHOLDER — Figure 2.1]**
*Suggested Visual: A timeline showing the progression from Point Set Generation Networks (2017) through NeRF (2020), Dream Fields (2022), DreamFusion (2022), and Hallo3D (2023).*

### 2.2.2 Recent Developments and Future Trends

**Score Distillation Sampling (SDS)**, introduced by DreamFusion [2], established the dominant paradigm for high-quality text-to-3D generation. SDS frames the problem as distilling a pre-trained 2D diffusion prior into a 3D NeRF representation by computing:

$$ \nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t,\epsilon}\left[w(t)\left(\hat{\epsilon}_\phi(x_t; y, t) - \epsilon\right) \frac{\partial x}{\partial \theta}\right] $$

where $\hat{\epsilon}_\phi$ is the noise prediction from a text-conditioned diffusion model, and $x = g(\theta)$ is a differentiable rendering of the 3D representation parameterised by $\theta$. DreamFusion achieved 3D generation with multi-view consistency far exceeding prior approaches but suffered from over-saturation and the **Janus problem** — multi-faced artefacts caused by the 2D prior generating front-facing content from all viewpoints.

**Magic3D** [5] addressed fidelity by adopting a coarse-to-fine optimisation strategy, first optimising a NeRF at low resolution then refining a DMTet mesh at high resolution. This enabled texture and geometry quality beyond what volumetric NeRF representations could achieve. **Fantasia3D** [4] further disentangled geometry and appearance into separate optimisation stages, using DMTET for geometry and physically-based rendering (PBR) for appearance:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SDS}}^{\text{geo}} + \mathcal{L}_{\text{SDS}}^{\text{app}}$$

**ProlificDreamer** [3] extended SDS to **Variational Score Distillation (VSD)**, treating the 3D scene as a distribution rather than a point estimate:

$$ \nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}_{t,\epsilon}\left[w(t)\left(\hat{\epsilon}_\phi(x_t; y, t) - \hat{\epsilon}_{\phi_{\text{particle}}}(x_t; t)\right) \frac{\partial x}{\partial \theta}\right] $$

By introducing a LoRA-fine-tuned "particle" model $\hat{\epsilon}_{\phi_{\text{particle}}}$ trained on the current 3D output distribution, VSD substantially improved visual diversity and reduced mode collapse.

**TextMesh** [17] specifically targeted mesh-quality outputs by integrating SDS with mesh regularisation losses, producing assets compatible with standard rendering pipelines. **MeshGPT** [15] proposed a fundamentally different paradigm — autoregressive token-by-token triangle face generation using a decoder-only transformer — achieving topologically clean outputs but at the cost of geometric diversity.

**3D Gaussian Splatting** [9] introduced an explicit point-cloud-based radiance representation using anisotropic Gaussians, enabling real-time rendering with quality comparable to NeRF. Recent 3DGS-based text-to-3D pipelines have rapidly adopted this representation for its efficiency.

**Hallo3D** [6] introduced the first multi-modal hallucination detection framework for 3D generation, employing a Large Multi-modal Model (LMM) to inspect rendered views and generate natural-language negative prompts fed back to the diffusion backbone. While Hallo3D demonstrated that LMMs could identify semantic hallucinations (duplicated faces, structural inconsistencies), the reliance on language as the feedback medium introduced a spatial information bottleneck.

**Visual Instruction Tuning** [18] provided the foundational technique for aligning language models with visual understanding through supervised fine-tuning, enabling the LLaVA family of models that underpins the AGD detection subsystem.

### 2.2.3 Issues and Research Challenges

Despite rapid progress, several persistent issues characterise the current state of text-to-3D generation:

1. **The Janus Problem:** SDS-based methods consistently generate multi-faced artefacts because the 2D diffusion prior scores each rendered view independently, without enforcing view-consistency constraints in 3D space.

2. **Geometric Hallucinations:** Generated meshes frequently contain structural artefacts — floating components ($\beta_0 > 1$), unintended topological handles (genus $g > 0$), degenerate faces, and non-manifold edges — that cannot be detected from any single 2D view.

3. **Over-saturation and Over-smoothing:** SDS gradients tend to produce over-saturated textures (DreamFusion) and geometrically over-smooth surfaces (Magic3D) due to the averaging effect of the expected gradient over the noise schedule.

4. **Mode Collapse:** Without appropriate score conditioning, SDS optimisation converges to a single 3D configuration ignoring multi-modal outputs, as addressed by ProlificDreamer's VSD formulation [3] and score-space techniques [8].

5. **Semantic Bottleneck in Correction Loops:** Existing closed-loop approaches (Hallo3D [6]) convert spatial 3D detections to natural-language prompts, discarding precise vertex-level spatial information in the process.

## 2.3 Comparative Analysis of Existing Methods

| Method | Representation | 3D Supervision | Hallucination Mitigation | Output Quality |
| :--- | :--- | :--- | :--- | :--- |
| Point-E [13] | Point Cloud | Diffusion | None | Low |
| Shap-E [14] | Implicit Function | Diffusion | None | Low–Medium |
| DreamFusion [2] | NeRF | SDS | None | Medium |
| Magic3D [5] | NeRF → DMTet | SDS (coarse-to-fine) | None | Medium–High |
| Fantasia3D [4] | DMTet + PBR | SDS (disentangled) | None | High |
| ProlificDreamer [3] | NeRF | VSD | Partial (diversity) | High |
| Perp-Neg [7] | NeRF | SDS + negative | Janus (partial) | Medium–High |
| TextMesh [17] | Mesh | SDS + mesh reg. | None | Medium |
| MeshGPT [15] | Mesh (autoregressive) | Supervised | None | High (clean topo) |
| Hallo3D [6] | NeRF/Mesh | LMM feedback | Semantic (language) | High |
| **AGD (proposed)** | **Mesh** | **SDS + geometric energy** | **Spatial (3D grounded)** | **High** |

**[DIAGRAM PLACEHOLDER — Figure 2.2]**
*Suggested Visual: A radar / spider chart comparing the above methods across five dimensions: output fidelity, view consistency, topological correctness, hallucination detection coverage, and correction spatial precision.*

## 2.4 Strengths and Limitations of Current Approaches

**Score Distillation Sampling (DreamFusion, Magic3D, ProlificDreamer)** — These methods achieve high visual quality leveraging the richness of pre-trained 2D diffusion priors. Their critical limitation is that the optimisation objective is defined entirely in pixel space; no geometric constraint prevents the optimiser from converging to topologically invalid or hallucinated 3D configurations.

**Score-Space Debiasing (Perp-Neg [7], Entropic SDS [8])** — These methods modify the SDS gradient direction to discourage multi-view mode collapse. Perp-Neg [7] decomposes the score gradient into perpendicular components, with negative prompts applied to non-front views:

$$ \nabla\mathcal{L}_{\text{perp-neg}} = \nabla\mathcal{L}_{\text{SDS}} - \sum_{\text{neg}} \text{proj}_{\hat{\epsilon}_{\text{neg}}} \nabla\mathcal{L}_{\text{SDS}} $$

While effective at reducing Janus artefacts, these approaches operate upstream of the 3D representation and cannot detect or correct hallucinations that have already emerged in the geometry.

**Hallo3D [6]** — The first framework to apply a post-hoc detection loop to 3D hallucinations. Its strength lies in semantic hallucination recognition via LMM inspection. Its critical limitation is the information loss inherent in converting spatial 3D geometry (precise vertex coordinates) into natural language (token sequences), and then re-injecting this as a diffusion conditioning signal — a many-to-one lossy mapping that degrades correction spatial precision.

**Laplacian Mesh Editing [19]** — Sorkine's seminal formulation of geometric deformation as minimisation of a quadratic Laplacian energy provides the theoretical foundation for the AGD refinement loop. The energy:

$$ E = \|Lv\|^2 \quad \text{subject to position constraints} $$

enables targeted, bounded mesh correction but requires explicit spatial constraint specification — which is precisely what the AGD grounding module provides.

## 2.5 Summary of Issues

The reviewed methods reveal three convergent structural limitations in the text-to-3D generation literature:

1. **Geometry-blindness of the optimisation objective** — SDS and its variants define loss entirely in 2D pixel space.
2. **Absence of vertex-level correction signals** — No prior method produces a spatially precise per-vertex correction gradient derived from 3D geometric analysis.
3. **Semantic bottleneck in detection-correction loops** — Language-mediated feedback (Hallo3D) discards the spatial precision required for targeted geometric correction.

## 2.6 Discussion on Research Gaps

### 2.6.1 Definition of the Research Gap / Problem

A systematic examination of the literature reveals the following formally stated research gap:

> **No existing text-to-3D generation or refinement framework implements a closed-loop feedback mechanism that (a) detects geometric hallucinations using spatially complete 3D metrics (spectral graph theory, algebraic topology), (b) grounds detections directly to individual mesh vertices without intermediate natural-language conversion, and (c) applies a spatially selective, theoretically bounded geometric correction using the grounded vertex weights.**

This gap is not a limitation of any specific implementation but a structural consequence of: (a) the 2D-pixel-space definition of SDS objectives, and (b) the architectural decision in Hallo3D to use natural language as the spatial feedback channel.

The AGD framework is proposed as the principled response to this gap, replacing the language channel with a direct numerical grounding mechanism and augmenting the 2D SDS objective with a geometry-aware energy term $E(x) = \lambda L_{\text{geo}} + (1-\lambda)D$ applied selectively at the vertex level through the weight map $w_i$.

## 2.7 Summary

This chapter reviewed the evolution of text-to-3D generation from early point-cloud methods [10] through NeRF-based SDS pipelines [2][3][4][5] to multi-modal detection approaches [6]. A comparative analysis across eleven systems demonstrated that existing methods either lack geometric grounding entirely or introduce a semantic bottleneck that degrades correction precision. The formally stated research gap — the absence of a closed-loop, spatially grounded geometric correction mechanism — directly motivates the design of the AGD framework presented in the subsequent chapters.

---

## References

[1] G. Metzer et al., "Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures," arXiv:2211.07600, 2022.
[2] B. Poole et al., "DreamFusion: Text-to-3D Using 2D Diffusion," ICLR, 2023.
[3] Z. Wang et al., "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation," arXiv:2305.16213, 2023.
[4] R. Chen et al., "Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation," arXiv:2303.13873, 2023.
[5] C.-H. Lin et al., "Magic3D: High-Resolution Text-to-3D Content Creation," arXiv:2211.10440, 2023.
[6] J. Cui et al., "Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Video Diffusion Transformer."
[7] M. Armandpour et al., "Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond," arXiv:2304.04968, 2023.
[8] P. Wang et al., "Taming Mode Collapse in Score Distillation for Text-to-3D Generation," CVPR 2024.
[9] T. Wu et al., "Recent advances in 3D Gaussian splatting," Comp. Visual. Med., vol. 10, no. 4, 2024.
[10] H. Fan et al., "A Point Set Generation Network for 3D Object Reconstruction from a Single Image," CVPR 2017.
[11] K. Schwarz et al., "VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids."
[12] A. Jain et al., "Zero-Shot Text-Guided Object Generation with Dream Fields," CVPR 2022.
[13] A. Nichol et al., "Point-E: A System for Generating 3D Point Clouds from Complex Prompts," arXiv:2212.08751, 2022.
[14] H. Jun and A. Nichol, "Shap-E: Generating Conditional 3D Implicit Functions," arXiv:2305.02463, 2023.
[15] Y. Siddiqui et al., "MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers," CVPR 2024.
[16] J. Yang et al., "SYM3D: Learning Symmetric Triplanes for Better 3D-Awareness of GANs."
[17] C. Tsalicoglou et al., "TextMesh: Generation of Realistic 3D Meshes From Text Prompts," 3DV 2024.
[18] H. Liu et al., "Visual Instruction Tuning."
[19] O. Sorkine, "Laplacian Mesh Processing."
