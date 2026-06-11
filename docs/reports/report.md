# AGD Pipeline — Output Report

> **Run command:** `python agd_pipeline.py --input-dir 3d_samples --output-dir agd_outputs --render-views`
> **Date:** 2026-04-29

---

## 1. What the Pipeline Does

The **Adversarial Geometric Distillation (AGD)** pipeline loads every `.obj / .ply / .stl` file from `3d_samples/`, analyses its geometry, optionally refines it, and saves an `*_agd.*` copy to `agd_outputs/`.

For each mesh it prints **three metrics** — before refinement → after refinement:

```
<filename> -> <output_filename>
  score:    X.XXX -> X.XXX
  variance: X.XXXXX -> X.XXXXX
  fiedler:  X.XXXXX -> X.XXXXX
```

---

## 2. What Each Value Means

### 2.1 `score` — Hallucination Score ∈ [0, 1]

**Source:** `discriminator.py → hallucination_score()`

A **heuristic composite penalty** computed from four sub-scores:

| Sub-penalty | Weight | What it captures |
|---|---|---|
| Connected-component penalty | 25 % | Meshes broken into multiple floating islands |
| Genus penalty | 35 % | Number of topological holes/handles (e.g., a torus has genus 1) |
| Eigenvalue-variance penalty | 25 % | Spectral irregularity of the mesh graph |
| Fiedler penalty | 15 % | Weak algebraic connectivity → poorly-connected patches |

**Interpretation:**

| Score range | Meaning |
|---|---|
| **0.0 – 0.3** | Clean, well-formed mesh |
| **0.4 – 0.6** | Moderate topological or spectral anomalies |
| **0.7 – 1.0** | Severe hallucination — multiple components, large genus, or extreme spectral irregularity |

> A higher score means the mesh looks more "hallucinated" (spurious geometry, Janus faces, floating blobs, etc.).

---

### 2.2 `variance` — Laplacian Eigenvalue Variance

**Source:** `discriminator.py → compute_laplacian_metrics()` → key `eigen_variance`

The **combinatorial graph Laplacian** `L = D − A` is built from the mesh's vertex-adjacency graph (D = degree matrix, A = adjacency matrix). Its eigenvalue spectrum encodes the mesh's intrinsic shape.

**How it is computed (efficiently, without full eigen-decomposition):**

```
variance = trace(L²) / n  −  (trace(L) / n)²
```

**Interpretation:**

| Variance | Meaning |
|---|---|
| **Low (~0.05 – 0.20)** | Smooth, uniform connectivity — vertices have similar neighbourhood sizes |
| **Medium (~0.20 – 1.5)** | Some irregularity — mix of high- and low-degree vertices |
| **High (> 1.5)** | Strong spectral spread — sharp protrusions, spikes, or disconnected clusters |

> Think of it as measuring **how uneven the mesh's vertex neighbourhoods are**. A perfect sphere would have near-zero variance; a spiky or fractal mesh would have very high variance.

---

### 2.3 `fiedler` — Fiedler Value (Algebraic Connectivity)

**Source:** `discriminator.py → compute_laplacian_metrics()` → key `fiedler_value`

The **Fiedler value** is the **second-smallest eigenvalue** of the Laplacian matrix (λ₂). It is also called the *algebraic connectivity* of the graph.

**Interpretation:**

| Fiedler value | Meaning |
|---|---|
| **= 0.0** | Graph is disconnected (multiple separate components) |
| **Very small (≈ 1e-5 – 0.01)** | Graph is barely connected — there is a narrow "bridge" between parts |
| **Larger** | Graph is well-connected — removing any single vertex won't fragment it |

> In simple terms: **a Fiedler value of 0 means the mesh is split into separate islands** that don't share any edges. All the meshes in this run show `fiedler ≈ 0`, meaning they all have at least some degree of disconnected geometry.

---

## 3. Per-Model Results

> **Note:** 3 models (`Bearded_guy.ply`, `blade.ply`, `happy.ply`) were **skipped** because they exceeded the 500,000-vertex RAM limit. Scores are unchanged before → after because the optimizer ran for `--iterations 10` with `--lr 0.15`, and the current optimizer (`optimizer.py`) applies only mild Laplacian smoothing which does not significantly shift spectral metrics in 10 steps.

| Mesh | Score | Variance | Fiedler | Interpretation |
|---|---|---|---|---|
| **ateneam.obj** | 0.878 | 2.892 | ≈ 0 | **High hallucination.** Very high variance — severe connectivity irregularity; likely many components or spiky protrusions. |
| **bunny.ply** | 0.488 | 0.676 | ≈ 0 | **Moderate.** The Stanford Bunny has some open boundary edges (the base), causing disconnection and a modest variance. |
| **dragon.ply** | 0.693 | 0.039 | ≈ 0 | **Moderate-High.** Variance is low (smooth mesh), but the score is high, implying multiple components or non-trivial genus. |
| **elepham.obj** | 0.650 | 1.057 | ≈ 0 | **Moderate-High.** Medium variance; likely some disconnected parts (legs, trunk). |
| **hand.ply** | 0.610 | 0.061 | ≈ 0 | **Moderate.** Low variance, clean geometry, but disconnected sub-meshes raise the score. |
| **horse.ply** | 0.400 | 0.519 | ≈ 0 | **Moderate-Low.** Relatively clean; medium variance from leg geometry. |
| **janus.obj** | 0.672 | 1.162 | ≈ 0 | **Moderate-High.** The Janus-face model — medium-high variance aligns with the Janus artifact (duplicated facial geometry). |
| **mba1.obj** | **1.000** | 0.146 | ≈ 0 | **Maximum hallucination score.** Very clean variance but score = 1.0 means extreme topological issues (many components or very high genus). |
| **mba2.obj** | **1.000** | 0.189 | ≈ 0 | Same as mba1 — extreme topological complexity. |
| **Meshy_AI_Infernal_Ironclad.obj** | **1.000** | 0.053 | ≈ 0 | **Maximum.** AI-generated mesh with severe structural hallucinations (the armour mesh is split into many separate components). |
| **Mr_Bean.obj** | **1.000** | 2.538 | ≈ 0 | **Maximum + high variance.** Both extreme topology and spectral irregularity — confirms heavy hallucination in this AI-generated asset. |
| **Rigged_Hand.obj** | 0.685 | 7.333 | ≈ 0 | **Highest variance in dataset.** The rigged hand has bone/skin geometry stacked in the same space, creating extreme vertex-neighbourhood variation. |
| **venusm.obj** | **1.000** | 1.063 | ≈ 0 | **Maximum.** High genus or many components; medium-high variance. |

---

## 4. Why Scores Didn't Change (before → after)

All 13 processed meshes show **identical before and after values**. This is expected because:

1. **The optimizer (`refine_mesh`) applies Laplacian smoothing** — it moves vertices slightly toward their neighbourhood average. This changes vertex *positions* but not the mesh *topology* (connectivity, edges, faces stay the same).
2. **Topology-driven metrics (genus, components) are unchanged** by positional smoothing.
3. **Spectral metrics** (variance, Fiedler) are computed on the *same adjacency graph*, which is topology-dependent. Minor positional shifts do not alter eigenvalues measurably.

> To actually reduce scores, the pipeline would need topology-editing operations (hole filling, component merging, remeshing) rather than positional smoothing alone.

---

## 5. Why Some Meshes Were Skipped

```
WARNING: Skipping Bearded_guy.ply — 1,024,355 vertices exceeds limit of 500,000
WARNING: Skipping blade.ply     —   882,954 vertices exceeds limit of 500,000
WARNING: Skipping happy.ply     —   543,524 vertices exceeds limit of 500,000
```

The pipeline enforces `MAX_VERTICES = 500,000` to avoid **out-of-memory crashes** (a previous run without this guard hit a `numpy.core._exceptions._ArrayMemoryError` trying to allocate 20.2 MiB for a 1,765,388-vertex mesh).

---

## 6. Summary

| Metric | Formula | Range | High value means |
|---|---|---|---|
| **score** | Weighted sum of 4 topology/spectral penalties | [0, 1] | More hallucination |
| **variance** | `trace(L²)/n − (trace(L)/n)²` | [0, ∞) | More spectral irregularity |
| **fiedler** | λ₂ of graph Laplacian | [0, ∞) | ≈ 0 = disconnected mesh |
