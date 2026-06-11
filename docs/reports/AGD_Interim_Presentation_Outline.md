# AGD Interim Presentation Outline

This file gives a simple and presentation-friendly explanation of the exact sections you asked for. It is written in a clear academic style so it can be turned into slides later.

## 1. Introduction

This research is about improving **text-to-3D generation**. Current text-to-3D systems can create impressive 3D objects from a text prompt, but they often make **geometry hallucinations** because they depend heavily on **2D image priors**. In simple terms, the model may produce something that looks correct from one view, but the actual 3D shape is structurally wrong.

Common errors include:
- duplicated faces or body parts (Janus problem)
- floating or disconnected mesh parts
- unwanted holes, handles, or broken topology
- irregular spikes or noisy geometry

The main idea of this research is to introduce a framework called **Adversarial Geometric Distillation (AGD)** that can detect these structural problems and refine the 3D mesh using **direct geometric feedback**, instead of depending only on text-based correction.

## 2. Objectives

In simple terms, this research tries to understand **why 3D hallucinations happen**, **how to detect them properly**, and **how to correct them in a more spatially accurate way**.

Main objectives:
- review the literature on geometric hallucinations in text-to-3D generation
- study adversarial learning, self-distillation, and geometry-based refinement methods
- design and develop the **AGD framework** for detecting and correcting 3D structural errors
- evaluate the solution using geometric quality measures and experiment results

## 3. Literature Review

The literature shows that text-to-3D generation has improved a lot, but the geometry problem is still not fully solved.

Important findings from previous work:
- **DreamFusion, Magic3D, and Fantasia3D** improved text-to-3D quality, but they still rely on 2D diffusion guidance, so the generated 3D structure can be inconsistent.
- **Perp-Neg** and **Entropic Score Distillation** tried to reduce Janus-type problems by changing the score or gradient behavior, but they are still mostly **geometry-blind**.
- **Hallo3D** introduced a better idea by using a large multimodal model to detect hallucinations from rendered views.
- However, Hallo3D sends the correction back through **language prompts**, which causes a **semantic bottleneck**. The model understands that an error exists, but it does not know the exact 3D location to fix.

So, the literature review shows a clear trend:
- generation quality has improved
- detection has improved
- but **direct spatial correction in 3D** is still missing

## 4. Problem Definition (Research Gap)

The main research gap is the absence of a **closed-loop framework** that can detect hallucinations and convert them into **direct 3D geometric corrections**.

In simple terms, current systems have three main problems:
- the optimization objective in SDS-based generation does not include strong geometric constraints
- many methods can detect errors, but they do not map them to exact 3D mesh regions
- language-based feedback loses spatial details, so correction becomes weak or too general

Because of this, current methods may know that something is wrong, but they still cannot **precisely fix the wrong vertices or regions** in the mesh.

This is the gap that AGD tries to solve.

## 5. Technology Adopted (Extended)

The AGD extension combines several technologies so that the system can both **understand the geometry** and **refine it safely**.

Main technologies used in the research:
- **Multi-view rendering**: renders the same mesh from many camera angles to observe structural problems from different views
- **Spectral graph theory**: analyzes the mesh as a graph and measures properties such as Laplacian variance and Fiedler value to detect irregular or disconnected geometry
- **Algebraic topology**: uses concepts like Euler characteristic, connected components, and genus to check whether the mesh structure is topologically sound
- **Adversarial critic**: scores suspicious mesh regions at the vertex level using heuristic analysis and an optional GNN-based critic
- **Vision-language inspection**: uses a model like LLaVA to inspect rendered views and estimate how severe a visible hallucination is
- **Geometric grounding**: converts detected view-level problems into 3D vertex-level weights
- **Self-distillation and Laplacian refinement**: updates the mesh carefully so that problematic regions are corrected while the original shape is still preserved

Implementation technologies:
- Python
- Trimesh
- NumPy and SciPy
- NetworkX
- PyTorch and PyTorch Geometric
- Pillow and scikit-image
- local LLaVA integration

## 6. Novel Approach

The novelty of this research is not just detecting hallucinations, but **turning the detection into direct 3D correction**.

The proposed AGD approach is novel in four main ways:
- it works as a **refinement layer** that can be attached to an existing text-to-3D pipeline
- it combines **global geometric analysis** and **local view-based detection**
- it removes the **semantic bottleneck** by converting detections into **numerical vertex weights**, not text prompts
- it applies correction only where needed through a **grounded refinement loop**

In simple terms, AGD does not say only "there is an error."  
It says:
- where the error is
- how strong the error is
- which vertices should be refined

This is the key research contribution.

## 7. Current Implementation

The current implementation shows that the research has already moved beyond theory and into a working prototype stage.

### 7.1 What has already been implemented

The current AGD prototype includes:
- mesh loading, validation, and normalization
- multi-view rendering with dense camera coverage
- topological and spectral analysis of meshes
- heuristic vertex-level anomaly scoring
- optional GNN critic structure
- local LLaVA-based view inspection
- geometric grounding from view scores to vertex weights
- weighted refinement using Laplacian smoothing and anchor-based self-distillation

### 7.2 Completed experiment: EXP_03 baseline

Before the full AGD pipeline, an earlier experiment called **EXP_03** was completed as a baseline.

What EXP_03 did:
- rendered 24 views of a mesh
- used 5 detectors: CLIP semantic deviation, depth discontinuity, normal inconsistency, silhouette asymmetry, and edge density variance
- produced differentiable geometric loss values for refinement

Main result from EXP_03:
- the Stanford Dragon mesh produced a **low global hallucination score of 0.0856**
- no strong Janus, floater, or normal-flip issue was detected
- this showed that training-free multi-view detection can work reliably on clean geometry

Why EXP_03 matters:
- it proved that direct geometric signals can be useful
- it helped shape the move from simple view-level detection to vertex-level grounding in AGD

### 7.3 Current AGD experiment progress

The newer **EXP_04 / AGD** stage has already implemented the main pipeline structure.

Current progress includes:
- a dataset of test meshes has been processed through the pipeline
- topology-based and spectral metrics are being computed before and after refinement
- the framework can detect severe hallucination patterns such as disconnected components, high genus, and structural irregularity
- the full workflow from detection to refinement is already runnable

### 7.4 Current observations

The present implementation is strong as a prototype, but some parts are still under development.

Current observations:
- the pipeline structure is working
- grounding and refinement are implemented
- the heuristic critic is usable now
- the GNN critic is still partial and needs better training
- the current optimizer mainly smooths geometry, so some topology-driven scores may not improve much yet

This means the project already demonstrates a **working proof of concept**, while still leaving room for stronger refinement and benchmark evaluation.

## 8. Design

The AGD framework is designed as a **modular closed-loop system**. This means each part has a clear role, and the whole system forms one correction cycle from input mesh to refined output mesh.

### 8.1 High-level design

The design contains three main modules:

1. **Preprocessing Module**
- loads the mesh
- checks validity
- normalizes the shape
- builds adjacency and graph information

2. **ML Engine**
- performs topological and spectral analysis
- computes vertex anomaly scores
- renders multiple views
- collects view-based hallucination scores

3. **Extended Module**
- grounds the detected errors into 3D vertex weights
- applies weighted refinement
- evaluates the mesh again after correction

### 8.2 Simple workflow

The design flow can be explained simply as:

**Input mesh -> analyze -> detect -> ground -> refine -> evaluate -> output refined mesh**

This design is important because:
- detection is not separated from correction
- feedback stays inside the 3D pipeline
- correction is selective, not global
- the architecture is modular, so future technologies can replace individual parts without rebuilding the full system

### 8.3 Design strength

The biggest strength of the design is that it connects **what is seen**, **what is measured**, and **what is corrected** in one pipeline.

So the design supports the main research goal:
- detect geometric hallucinations accurately
- map them to the right 3D regions
- refine only those regions in a controlled way

---

This outline is ready to be converted into a clean interim presentation deck.
