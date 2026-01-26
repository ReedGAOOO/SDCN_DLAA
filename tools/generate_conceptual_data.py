#!/usr/bin/env python3
"""
Generate a small labeled graph dataset for comparing SpatialConv variants.

Outputs into `--output_dir`:
- node_features.npy: float32 [N, F]
- labels.npy: int64 [N]
- binary_adj.npz: scipy sparse CSR adjacency (0/1) [N, N]
- edge_attr.npy: float32 [E, edge_dim] (aligned to CSR nonzero order: row-major)
- edge_index.npy: int64 [2, E] (row/col pairs, same order as edge_attr)
- data_info.json: metadata for reproducibility
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class DataInfo:
    seed: int
    n_clusters: int
    points_per_cluster: int
    num_nodes: int
    feature_dim: int
    knn_k: int
    cross_edge_prob: float
    cross_edges_per_node: int
    edge_dim: int


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _expand_edge_features(base_feature: np.ndarray, edge_dim: int) -> np.ndarray:
    base_feature = base_feature.astype(np.float32).reshape(-1)
    num_edges = base_feature.shape[0]

    if edge_dim <= 1:
        return base_feature.reshape(num_edges, 1)

    expanded = np.zeros((num_edges, edge_dim), dtype=np.float32)
    expanded[:, 0] = base_feature

    for i in range(1, edge_dim):
        if i % 3 == 0:
            expanded[:, i] = np.power(base_feature, 2)
        elif i % 3 == 1:
            expanded[:, i] = np.exp(-base_feature * 5.0)
        else:
            expanded[:, i] = 1.0 / (base_feature + 1e-6)

    return expanded


def _make_cluster_coords(rng: np.random.Generator, n_clusters: int, points_per_cluster: int) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.5],
            [2.0, 3.5],
        ],
        dtype=np.float32,
    )
    if n_clusters > centers.shape[0]:
        extra = rng.normal(loc=0.0, scale=1.0, size=(n_clusters - centers.shape[0], 2)).astype(np.float32)
        centers = np.concatenate([centers, extra], axis=0)
    centers = centers[:n_clusters]

    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), points_per_cluster)
    coords = np.zeros((labels.shape[0], 2), dtype=np.float32)

    for cid in range(n_clusters):
        mask = labels == cid
        coords[mask] = centers[cid] + rng.normal(loc=0.0, scale=0.6, size=(mask.sum(), 2)).astype(np.float32)

    # Shuffle node order so clusters are not contiguous in index space.
    perm = rng.permutation(coords.shape[0])
    return coords[perm], labels[perm]


def _make_node_features(rng: np.random.Generator, coords: np.ndarray, labels: np.ndarray, feature_dim: int) -> np.ndarray:
    # Intentionally weakly-informative node features: mostly noise + tiny cluster bias.
    x = rng.normal(loc=0.0, scale=1.0, size=(coords.shape[0], feature_dim)).astype(np.float32)
    bias = (labels.astype(np.float32) / max(labels.max(), 1)) * 0.15
    x[:, 0] = x[:, 0] + bias

    # Add a small coordinate-projection to avoid perfect symmetry while keeping overlap high.
    proj = rng.normal(loc=0.0, scale=0.2, size=(2, feature_dim)).astype(np.float32)
    x = x + coords @ proj
    return x


def _pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=-1, dtype=np.float32)).astype(np.float32)


def _build_edges(
    rng: np.random.Generator,
    labels: np.ndarray,
    dist: np.ndarray,
    knn_k: int,
    cross_edge_prob: float,
    cross_edges_per_node: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = labels.shape[0]

    dist_no_diag = dist.copy()
    np.fill_diagonal(dist_no_diag, np.inf)

    edges: set[tuple[int, int]] = set()

    for i in range(num_nodes):
        nn = np.argpartition(dist_no_diag[i], knn_k)[:knn_k]
        for j in nn.tolist():
            if i != j:
                edges.add((i, j))
                edges.add((j, i))

    # Add a few cross-cluster edges to make the task non-trivial without edge features.
    clusters = {cid: np.where(labels == cid)[0].tolist() for cid in np.unique(labels)}
    for i in range(num_nodes):
        if rng.random() > cross_edge_prob:
            continue
        src_c = int(labels[i])
        other_clusters = [c for c in clusters.keys() if c != src_c]
        if not other_clusters:
            continue
        for _ in range(cross_edges_per_node):
            dst_c = int(rng.choice(other_clusters))
            j = int(rng.choice(clusters[dst_c]))
            if i == j:
                continue
            edges.add((i, j))
            edges.add((j, i))

    rows_cols = sorted(edges)
    rows = np.array([rc[0] for rc in rows_cols], dtype=np.int64)
    cols = np.array([rc[1] for rc in rows_cols], dtype=np.int64)
    return rows, cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/tmp/sdcn_dlaa_concept_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--points_per_cluster", type=int, default=60)
    parser.add_argument("--feature_dim", type=int, default=3)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--cross_edge_prob", type=float, default=0.10)
    parser.add_argument("--cross_edges_per_node", type=int, default=1)
    parser.add_argument("--edge_dim", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    coords, labels = _make_cluster_coords(rng, args.n_clusters, args.points_per_cluster)
    x = _make_node_features(rng, coords, labels, args.feature_dim)
    dist = _pairwise_dist(coords)

    rows, cols = _build_edges(
        rng=rng,
        labels=labels,
        dist=dist,
        knn_k=args.knn_k,
        cross_edge_prob=args.cross_edge_prob,
        cross_edges_per_node=args.cross_edges_per_node,
    )

    num_nodes = x.shape[0]
    num_edges = rows.shape[0]

    adj = sp.csr_matrix((np.ones(num_edges, dtype=np.float32), (rows, cols)), shape=(num_nodes, num_nodes))

    raw_edge_dist = dist[rows, cols].astype(np.float32)
    edge_base = _minmax_normalize(raw_edge_dist)
    edge_attr = _expand_edge_features(edge_base, args.edge_dim)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "node_features.npy"), x.astype(np.float32))
    np.save(os.path.join(args.output_dir, "labels.npy"), labels.astype(np.int64))
    np.save(os.path.join(args.output_dir, "coords.npy"), coords.astype(np.float32))
    sp.save_npz(os.path.join(args.output_dir, "binary_adj.npz"), adj)
    np.save(os.path.join(args.output_dir, "edge_attr.npy"), edge_attr.astype(np.float32))
    np.save(os.path.join(args.output_dir, "edge_index.npy"), np.vstack([rows, cols]).astype(np.int64))

    info = DataInfo(
        seed=args.seed,
        n_clusters=args.n_clusters,
        points_per_cluster=args.points_per_cluster,
        num_nodes=int(num_nodes),
        feature_dim=args.feature_dim,
        knn_k=args.knn_k,
        cross_edge_prob=args.cross_edge_prob,
        cross_edges_per_node=args.cross_edges_per_node,
        edge_dim=args.edge_dim,
    )
    with open(os.path.join(args.output_dir, "data_info.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(info), f, indent=2)

    print(f"Wrote conceptual dataset to: {args.output_dir}")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Edge dim: {args.edge_dim}, Clusters: {args.n_clusters}")


if __name__ == "__main__":
    main()

