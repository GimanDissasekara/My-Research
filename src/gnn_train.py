"""Train a simple GNN critic using heuristic pseudo-labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from critic import HeuristicCritic, _SimpleGCN


SUPPORTED_EXTS = {".obj", ".ply", ".stl"}


def iter_mesh_paths(input_dir: Path) -> List[Path]:
    paths = []
    for path in input_dir.rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTS:
            paths.append(path)
    return paths


def mesh_to_data(mesh: trimesh.Trimesh, labels: np.ndarray) -> Data:
    vertices = mesh.vertices
    num_vertices = vertices.shape[0]

    degrees = mesh.vertex_degree.astype(np.float32)
    edges = mesh.edges_unique
    if len(edges) == 0:
        mean_len = np.zeros((num_vertices,), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).astype(np.float32)
        mean_len = _accumulate_mean(lengths, edges, num_vertices).astype(np.float32)
        edge_index = np.stack([edges[:, 0], edges[:, 1]], axis=0)
        edge_index = np.concatenate([edge_index, edge_index[[1, 0], :]], axis=1)

    x = np.stack([degrees, mean_len], axis=1)
    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(labels.astype(np.float32)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN critic from pseudo-labels")
    parser.add_argument("--input-dir", default="3d_samples", help="Folder with mesh files")
    parser.add_argument("--output", default="gnn_critic.pt", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-threshold", type=float, default=0.7)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = iter_mesh_paths(input_dir)
    if not paths:
        raise SystemExit(f"No meshes found in {input_dir}")

    critic = HeuristicCritic()
    dataset: List[Data] = []

    for path in paths:
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        scores = critic.score(mesh)
        labels = (scores >= args.label_threshold).astype(np.float32)
        dataset.append(mesh_to_data(mesh, labels))

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = _SimpleGCN()
    optimizer = torch.optim.Adam(model_parameters(model), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index).squeeze(-1)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / len(loader):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved checkpoint to {args.output}")


def model_parameters(model: _SimpleGCN):
    params = []
    params.extend(model.conv1.parameters())
    params.extend(model.conv2.parameters())
    return params


def _accumulate_mean(lengths: np.ndarray, edges: np.ndarray, num_vertices: int) -> np.ndarray:
    """Vectorised scatter-add — avoids Python loop and OOM on large meshes."""
    sums = np.zeros((num_vertices,), dtype=float)
    counts = np.zeros((num_vertices,), dtype=float)
    if len(edges) == 0:
        return sums
    np.add.at(sums, edges[:, 0], lengths)
    np.add.at(sums, edges[:, 1], lengths)
    np.add.at(counts, edges[:, 0], 1.0)
    np.add.at(counts, edges[:, 1], 1.0)
    counts[counts == 0.0] = 1.0
    return sums / counts


if __name__ == "__main__":
    main()
