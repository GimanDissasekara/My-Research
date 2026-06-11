# EXP_06 Progress Report: Anomaly-Grounded 3D Refinement

## 1. Progress Achieved (All Stages Implemented)
We have successfully implemented and verified the entire 8-stage anomaly-grounded refinement (AGD) pipeline on existing 3D meshes.

* **Stage 1 & 2 (Input & Initial 3D):** The system reliably loads `.obj`, `.ply`, and `.stl` files. Pre-processing is robust, including a new **floater removal** feature that automatically cleans up disconnected hallucinatory fragments (e.g., reducing `mba1.obj` from 73 disjoint pieces to 1 solid mesh before refinement).
* **Stage 3 (Multi-view Rendering):** `renderer.py` renders dense canonical camera views, outputting RGB, Depth, Silhouette, Normal, and Edge maps. Support for Fibonacci sphere dense sampling (e.g., 36 or 100 views) is fully operational.
* **Stage 4 (Hallucination Detection):**
    * **View Consistency:** Implemented an all-pairs angularly-weighted consistency check using image embeddings. Also integrated depth-edge density checks (via Canny edge detection) to heavily penalize floating geometric tears.
    * **Geometric Critic:** Both the Heuristic critic and the PyTorch-based GNN critic (`_SimpleGCN`) are fully integrated and provide per-vertex anomaly scores.
* **Stage 5 (Grounding):** `grounding.py` successfully translates 2D view severities back to 3D. We implemented a highly optimized **sparse matrix expansion** (`_expand_mask_sparse`) that runs ~100x faster than traditional loops, and wired in silhouette-IoU feedback.
* **Stage 6 (Refinement Loop):** `agd_pipeline.py` features a full outer loop. The mesh undergoes render -> detect -> ground -> refine cycles. It uses an adaptive learning rate (LR decay), a **cotangent Laplacian** option for area-aware smoothing, hard displacement clamping to prevent vertex explosions, and an early-stop mechanism based on score convergence.
* **Stage 7 & 8 (Evaluation & Ablation):** We developed an automated ablation runner (`ablation.py`). It dynamically evaluates different configurations (baseline, heuristic 6-view, heuristic 36-view, GNN, cotangent laplacian) across datasets and generates CSV reports and matplotlib bar charts.

## 2. What Needs to be Done Next
While the refinement loop itself is mature, to fully realize the AGD vision, the following next steps remain:

1. **Full Text-to-3D Integration (The SDS Loop):** Currently, we are refining existing meshes. The ultimate goal is to integrate this into a live generation pipeline (like DreamFusion or Magic3D) where the generative model (NeRF/SDF) produces the base geometry, and our AGD pipeline guides the Score Distillation Sampling (SDS) gradients.
2. **VLM (LLaVA) Testing at Scale:** The pipeline has hooks for `--use-vlm` and `LLaVADetector`. We need to run large-scale tests supplying actual prompts and evaluating if the Vision-Language Model provides superior hallucination detection compared to our geometric heuristics.
3. **GNN Critic Pre-training:** We trained a toy GNN (`gnn_train.py`) on heuristic pseudo-labels to prove the pipeline works. We now need to train the GNN on a real dataset of human-annotated 3D hallucinations (or a much larger synthetic dataset) so it learns genuine structural anomalies rather than mimicking the heuristic.
4. **.tri Format Support (Optional):** If legacy formats like `newcsieb12.tri` are needed, we must add a custom loader to `trimesh` supported extensions.

## 3. Explanation of Output Values (Metrics)

The pipeline outputs several highly specific metrics to quantify geometry and structural integrity:

### Anomaly / Structural Scores
* **Hallucination Score (`score`):** Ranging from 0 to 1, this combines spectral analysis (Laplacian eigenvalues) and topology. Lower is better. A dropping score means anomalous high-frequency noise and topological defects are being smoothed out.
* **Laplacian Eigen Variance (`variance`):** Measures the spread of the Laplacian spectrum. Highly spiked variances indicate unnatural, jagged geometry.
* **Euler Characteristic (`euler_char`) & Genus:** Topological invariants. A high genus (many "holes") on an object that should be solid (like an apple) flags a severe hallucination.

### Geometry Error Metrics (Baseline vs Refined)
These measure how much the refinement altered the fundamental shape:
* **Chamfer Distance (`CD`):** The average distance between points on the refined mesh and the closest points on the original mesh. (Lower is better).
* **Hausdorff Distance 95% (`HD95`):** The 95th percentile of distances between the meshes. Sensitive to extreme outliers/spikes. (Lower is better).
* **Normal Consistency Score (`NCS`):** The dot product of surface normals between matched points on the original and refined mesh. Approaches 1.0 if the surface orientation is perfectly preserved. (Higher is better).
* **Angular Normal Error (`Ang`):** The same concept as NCS, but expressed in degrees. (Lower is better).
* **Surface Roughness (`Rough`):** RMS of the mean curvature. A drop in roughness indicates successful smoothing of high-frequency hallucinations. (Lower is better).

### 2D View Metrics (Cross-Render Consistency)
* **Mean SSIM:** Structural Similarity Index of rendered images before vs after. 1.0 means the rendered object looks visually identical to the original. (Higher is better).
* **Silhouette IoU:** Intersection over Union of the rendered 2D masks. Measures if the global bounding shape was destroyed. (Higher is better).

## 4. How Evaluation is Happening
Evaluation is fully automated via `src/ablation.py`.

**The Process:**
1. The script loads a set of evaluation meshes from `3d_samples/`.
2. It evaluates the "baseline" condition (no refinement, just scoring the original topology).
3. It iterates through predefined experimental configurations (e.g., using a 6-view vs 36-view render, using the Heuristic vs GNN critic, using Uniform vs Cotangent smoothing).
4. For each condition, the mesh is fully refined.
5. `geometry_metrics.py` calculates the CD, HD95, NCS, and SSIM between the original baseline and the final refined output.
6. The results are aggregated into `ablation_out/ablation_results.csv` and visualized using Python `matplotlib` to produce bar charts (`cd_comparison.png`, `score_comparison.png`, `ssim_comparison.png`), giving us a quantitative view of which configuration removes hallucinations while best preserving the original intended shape.
