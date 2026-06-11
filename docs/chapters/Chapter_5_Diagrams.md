---
title: "AGD Chapter 5 — Architecture Diagrams"
---

## Diagram 5.1 — Top-Level AGD System Architecture

```mermaid
flowchart TD
    subgraph INPUT["PREPROCESSING MODULE"]
        direction TB
        A1["Load Mesh\n(.obj / .ply / .stl)"]
        A2["Validate & Normalise\n(centre + unit sphere)"]
        A3["Extract Adjacency Graph\n(NetworkX + edge_index)"]
        A4["Snapshot Original Vertices\nv_i^(0)"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph ML["ML ENGINE"]
        direction TB
        B1["Discriminator\n(topology + spectral + geometry quality)"]
        B2["Adversarial Critic\n(heuristic z-score OR GNN GCNConv)"]
        B3["Multi-View Renderer\n(Fibonacci-sphere, N views, 5 buffers)"]
        B4["LMM Detector\n(LLaVA — per-view severity JSON)"]
        B5["View Consistency\n(cosine similarity front↔back etc.)"]
    end

    subgraph EXT["EXTENDED MODULE"]
        direction TB
        C1["Geometric Grounding\n(view scores + critic → vertex weights w_i)"]
        C2["Refinement Optimizer\nv_i ← v_i - η·(λ∇L_geo + (1-λ)∇D)·w_i"]
        C3["Post-Analysis\n(Discriminator + 10-Metric Report)"]
        C4["Export Refined Mesh\n(*_agd.obj / .ply / .stl)"]
        C1 --> C2 --> C3 --> C4
    end

    INPUT --> ML
    B1 -->|"S_hall (global score)"| C1
    B2 -->|"critic_scores [V]"| C1
    B3 --> B4
    B3 --> B5
    B4 -->|"per-view severity + global bias"| C1
    B5 -->|"consistency severity dict"| C1
    EXT -->|"refined mesh"| OUTPUT["Clean 3D Mesh\n+ Geometry Error Report"]
```

---

## Diagram 5.2 — Preprocessing Module Data Flow

```mermaid
flowchart LR
    IN["Input File\n(.obj / .ply / .stl)"] --> LOAD["trimesh.load()\nforce='mesh'"]
    LOAD --> CHK{"Scene?"}
    CHK -->|"Yes"| CONCAT["Concatenate\nall geometries"]
    CHK -->|"No"| VALID
    CONCAT --> VALID{"Valid?\n(V > 0, V ≤ 500k)"}
    VALID -->|"No"| SKIP["SKIP — log warning"]
    VALID -->|"Yes"| NORM["Centre + Normalise\nto unit bounding sphere"]
    NORM --> ADJ["Extract nx.Graph\nvertex adjacency"]
    NORM --> SNAP["Snapshot vertices\noriginal_vertices = mesh.vertices.copy()"]
    ADJ --> OUT["→ ML Engine"]
    SNAP --> OUT
```

---

## Diagram 5.3 — ML Engine Internal Architecture

```mermaid
flowchart TD
    MESH["Normalised Mesh\n(trimesh.Trimesh)"] --> DISC & CRIT & REND

    subgraph DISC["Discriminator — discriminator.py"]
        D1["compute_topology()\nV, E, F → χ, genus, β_0"]
        D2["compute_laplacian_metrics()\nL=D-A → Var(λ), Fiedler λ_2\nCotangent variance"]
        D3["compute_geometry_quality()\nAspect ratio, degenerate faces\nNon-manifold edges"]
        D4["hallucination_score()\n0.20·P_cc + 0.30·P_genus\n+ 0.15·P_var + 0.10·P_fiedler\n+ 0.10·P_cot + 0.15·P_quality"]
        D1 & D2 & D3 --> D4
    end

    subgraph CRIT["Adversarial Critic — critic.py"]
        C1["Heuristic Critic\ndeg z-score + edge-len z-score\n→ sigmoid blend per vertex"]
        C2["GNN Critic (optional)\n2-layer GCNConv\ninput: [deg, mean_edge_len]\noutput: anomaly score [V]"]
        C1 -.->|"--use-gnn"| C2
    end

    subgraph REND["Renderer + Detectors"]
        R1["renderer.py\nFibonacci-sphere N views\n5 buffers: RGB, depth, normal\nsilhouette, edge_map"]
        R2["view_consistency.py\n32×32 grayscale cosine sim\nfront↔back, left↔right, top↔bottom"]
        R3["lmm_detector.py\nLLaVA v1.5/1.6 local\nJSON: {severity, notes}"]
        R1 --> R2
        R1 -->|"--use-vlm"| R3
    end

    D4 -->|"S_hall"| GND["→ Grounding Module"]
    C1 -->|"scores [V]"| GND
    C2 -->|"scores [V]"| GND
    R2 -->|"view_scores dict"| GND
    R3 -->|"view_scores + global_bias"| GND
```

---

## Diagram 5.4 — Extended Module: Grounding and Refinement

```mermaid
flowchart TD
    INPUTS["Extension Inputs\ncritic_scores [V]\nview_scores dict\nglobal_bias\noriginal_vertices"]

    subgraph GND["Geometric Grounding"]
        G1["Project multi-view evidence\ninto 3D vertex space"]
        G2["Build vertex weight map\nw_i ∈ [w_min, w_max]"]
        G1 --> G2
    end

    subgraph OPT["Refinement Optimizer"]
        O1["Geometric regularization\nL_geo"]
        O2["Anchor constraint\nD = Σ||v_i - v_i^(0)||^2"]
        O3["Weighted refinement update"]
        O1 & O2 --> O3
    end

    subgraph EVAL["Post-Refinement Evaluation"]
        E1["Re-check hallucination score\nand geometry metrics"]
        E2["Export refined mesh\nand summary report"]
        E1 --> E2
    end

    OUTPUTS["Extension Outputs\nrefined mesh\nS_hall^after\ngeometry report"]

    INPUTS --> G1
    INPUTS -->|"anchor reference"| O2
    GND -->|"w_i [V]"| OPT
    OPT -->|"refined mesh"| EVAL
    EVAL --> OUTPUTS
```

---

## Diagram 5.5 — Complete AGD Data Flow and Module Interaction

```mermaid
flowchart LR
    IN["Input Mesh"] --> PRE["Preprocessing\nload, validate, normalize"]
    PRE --> GEO["Geometry Analysis\nDiscriminator + Critic"]
    PRE --> VIEW["Multi-View Analysis\nRenderer + View Consistency\n+ optional LMM"]
    GEO --> GND["Grounding Module\ncombine analysis signals\n→ weights w_i"]
    VIEW --> GND
    GND --> OPT["Refinement Optimizer\nupdate mesh geometry"]
    OPT --> EVAL["Evaluation and Export\nhallucination score\n+ geometry report"]
    EVAL --> OUT["Outputs\nRefined Mesh\n+ Quality Report"]
```
