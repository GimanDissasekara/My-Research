"""Critic module for AGD pipeline.

Provides a heuristic critic and an optional GNN stub (torch-geometric).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class CriticConfig:
    use_gnn: bool = False
    gnn_checkpoint: Optional[str] = None
    seed: int = 42


class HeuristicCritic:
    def score(self, mesh: trimesh.Trimesh) -> np.ndarray:
        vertices = mesh.vertices
        num_vertices = vertices.shape[0]
        if num_vertices == 0:
            return np.zeros((0,), dtype=float)

        # Degree-based anomaly score
        degrees = mesh.vertex_degree
        deg_z = _zscore(degrees)

        # Edge-length anomaly score
        edges = mesh.edges_unique
        if len(edges) == 0:
            len_z = np.zeros_like(deg_z)
        else:
            lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
            mean_len = _accumulate_mean(lengths, edges, num_vertices)
            len_z = _zscore(mean_len)

        score = 0.5 * _sigmoid(np.abs(deg_z)) + 0.5 * _sigmoid(np.abs(len_z))
        return np.clip(score, 0.0, 1.0)


class GnnCritic:
    def __init__(self, checkpoint: Optional[str] = None, seed: int = 42) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch and torch_geometric are required for GnnCritic")
        torch.manual_seed(seed)
        self.model = _SimpleGCN()
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state)
        self.model.eval()

    def score(self, mesh: trimesh.Trimesh) -> np.ndarray:
        vertices = mesh.vertices
        num_vertices = vertices.shape[0]
        if num_vertices == 0:
            return np.zeros((0,), dtype=float)

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
        x_t = torch.from_numpy(x)
        edge_t = torch.from_numpy(edge_index)
        with torch.no_grad():
            out = self.model(x_t, edge_t)
            score = torch.sigmoid(out).squeeze(-1).cpu().numpy()
        return np.clip(score, 0.0, 1.0)


class _SimpleGCN(nn.Module):
    """Two-layer GCN that maps per-vertex features to a scalar anomaly logit."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def build_critic(config: CriticConfig):
    if config.use_gnn:
        try:
            return GnnCritic(checkpoint=config.gnn_checkpoint, seed=config.seed)
        except Exception:
            # Fallback to heuristic critic if torch-geometric is unavailable.
            return HeuristicCritic()
    return HeuristicCritic()


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
