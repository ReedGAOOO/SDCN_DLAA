#!/usr/bin/env python3
"""
Run classic baseline clustering methods on a labeled dataset directory.

Baselines (by default):
- kmeans_x: KMeans on node features
- kmeans_edge_mean: KMeans on per-node mean pooled edge_attr
- kmeans_x_edge_mean: KMeans on [x || mean(edge_attr)] concatenated
- spectral_adj_binary: Spectral clustering on binary adjacency
- spectral_edge_distance: Spectral clustering on distance-derived weights (uses edge_attr[:,0])
- spectral_edge_l2: Spectral clustering on weights derived from all edge_attr dims (z-scored L2 norm)
- spectral_node_edge_rbf: Spectral clustering on RBF weights computed from per-node pooled edge_attr

Expected files in --data_dir:
- node_features.npy
- labels.npy
- binary_adj.npz
- edge_attr.npy

Outputs:
- summary_json (default: summary_baselines.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler

# Ensure repo root is importable when running from an arbitrary cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import eva  # noqa: E402


def _minmax(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _cluster_distribution(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def _pool_edge_attr_to_nodes_mean(adj_sp: sp.spmatrix, edge_attr: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Mean-pool edge_attr to nodes by aggregating incident edges to both endpoints.
    Expects edge_attr rows aligned with adj_sp.tocoo() nonzeros.
    """
    adj = adj_sp.tocoo()
    if edge_attr.shape[0] != adj.nnz:
        raise ValueError(f"edge_attr rows ({edge_attr.shape[0]}) != adj.nnz ({adj.nnz})")
    edge_dim = int(edge_attr.shape[1])

    rows = adj.row.astype(np.int64, copy=False)
    cols = adj.col.astype(np.int64, copy=False)

    node_sum = np.zeros((n_nodes, edge_dim), dtype=np.float32)
    node_cnt = np.zeros((n_nodes,), dtype=np.int64)

    np.add.at(node_sum, rows, edge_attr)
    np.add.at(node_cnt, rows, 1)
    np.add.at(node_sum, cols, edge_attr)
    np.add.at(node_cnt, cols, 1)

    denom = np.maximum(node_cnt, 1).astype(np.float32).reshape(-1, 1)
    return (node_sum / denom).astype(np.float32)


def _run_kmeans_x(x: np.ndarray, n_clusters: int, seed: int | None) -> np.ndarray:
    x_std = StandardScaler().fit_transform(x)
    kwargs: dict[str, Any] = {"n_clusters": n_clusters, "n_init": 20}
    if seed is not None:
        kwargs["random_state"] = seed
    return KMeans(**kwargs).fit_predict(x_std)


def _run_kmeans_features(features: np.ndarray, n_clusters: int, seed: int | None) -> np.ndarray:
    feats = StandardScaler().fit_transform(features)
    kwargs: dict[str, Any] = {"n_clusters": n_clusters, "n_init": 20}
    if seed is not None:
        kwargs["random_state"] = seed
    return KMeans(**kwargs).fit_predict(feats)


def _run_spectral(adj: sp.spmatrix, n_clusters: int, seed: int | None) -> np.ndarray:
    # SpectralClustering expects a dense array for precomputed affinity in many versions.
    affinity = adj
    if sp.issparse(affinity):
        affinity = affinity.tocsr()
    affinity = affinity.maximum(affinity.T)
    affinity = affinity.astype(np.float32)
    affinity_dense = affinity.toarray()

    kwargs: dict[str, Any] = {
        "n_clusters": n_clusters,
        "affinity": "precomputed",
        "assign_labels": "kmeans",
        "n_init": 10,
    }
    if seed is not None:
        kwargs["random_state"] = seed
    return SpectralClustering(**kwargs).fit_predict(affinity_dense)


def _weighted_adj_from_edge_distance(adj: sp.csr_matrix, edge_attr: np.ndarray, gamma: float) -> sp.csr_matrix:
    """
    Build a weighted adjacency by transforming edge_attr[:,0] (treated as distance-like).
    Weight = exp(-gamma * dist_norm).
    """
    adj_coo = adj.tocoo()
    rows = adj_coo.row
    cols = adj_coo.col
    if edge_attr.shape[0] != adj_coo.nnz:
        raise ValueError(
            f"edge_attr row count mismatch: edge_attr has {edge_attr.shape[0]}, "
            f"but adjacency has {adj_coo.nnz} nonzeros"
        )
    dist = edge_attr[:, 0].astype(np.float32).reshape(-1)
    dist_norm = _minmax(dist)
    weights = np.exp(-gamma * dist_norm).astype(np.float32)
    return sp.csr_matrix((weights, (rows, cols)), shape=adj.shape)


def _weighted_adj_from_edge_l2(adj: sp.csr_matrix, edge_attr: np.ndarray, gamma: float) -> sp.csr_matrix:
    """
    Build a weighted adjacency using all edge_attr dimensions:
    - z-score edge_attr across edges
    - scalarize each edge by its L2 norm
    - weight = exp(-gamma * minmax(norm))
    """
    adj_coo = adj.tocoo()
    rows = adj_coo.row
    cols = adj_coo.col
    if edge_attr.shape[0] != adj_coo.nnz:
        raise ValueError(
            f"edge_attr row count mismatch: edge_attr has {edge_attr.shape[0]}, "
            f"but adjacency has {adj_coo.nnz} nonzeros"
        )

    edge_std = StandardScaler().fit_transform(edge_attr.astype(np.float32, copy=False))
    norms = np.linalg.norm(edge_std.astype(np.float32, copy=False), axis=1).astype(np.float32)
    norm_scaled = _minmax(norms)
    weights = np.exp(-gamma * norm_scaled).astype(np.float32)
    return sp.csr_matrix((weights, (rows, cols)), shape=adj.shape)


def _weighted_adj_from_node_edge_rbf(adj: sp.csr_matrix, node_edge: np.ndarray, gamma: float) -> sp.csr_matrix:
    """
    Build a weighted adjacency using per-node pooled edge features:
    weight(i,j) = exp(-gamma * minmax(||node_edge[i]-node_edge[j]||)).
    """
    adj_coo = adj.tocoo()
    rows = adj_coo.row.astype(np.int64, copy=False)
    cols = adj_coo.col.astype(np.int64, copy=False)
    diffs = node_edge[rows] - node_edge[cols]
    dist = np.linalg.norm(diffs.astype(np.float32, copy=False), axis=1).astype(np.float32)
    dist_scaled = _minmax(dist)
    weights = np.exp(-gamma * dist_scaled).astype(np.float32)
    return sp.csr_matrix((weights, (rows, cols)), shape=adj.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edge_attr_norm",
        type=str,
        default="none",
        help="Edge feature normalization: none | zscore | zscore_clip | minmax",
    )
    parser.add_argument("--edge_attr_clip", type=float, default=5.0, help="Clip threshold used by zscore_clip.")
    parser.add_argument(
        "--edge_noise_std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise std added to edge_attr after normalization (seeded by --seed).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="kmeans_x,kmeans_edge_mean,kmeans_x_edge_mean,spectral_adj_binary,spectral_edge_distance,spectral_edge_l2,spectral_node_edge_rbf",
    )
    parser.add_argument("--distance_gamma", type=float, default=5.0)
    parser.add_argument("--summary_json", type=str, default="summary_baselines.json")
    args = parser.parse_args()

    seed = int(args.seed) if args.seed is not None else None
    methods = [m.strip() for m in args.methods.split(",") if m.strip() != ""]

    x = np.load(os.path.join(args.data_dir, "node_features.npy")).astype(np.float32)
    y_true = np.load(os.path.join(args.data_dir, "labels.npy")).astype(np.int64)
    adj = sp.load_npz(os.path.join(args.data_dir, "binary_adj.npz")).tocsr()
    edge_attr = np.load(os.path.join(args.data_dir, "edge_attr.npy")).astype(np.float32)

    # Optional edge normalization (for fair comparison with model pipelines).
    norm = (args.edge_attr_norm or "none").strip().lower()
    if norm not in {"none", "zscore", "zscore_clip", "minmax"}:
        raise SystemExit(f"Unknown --edge_attr_norm={norm!r}. Use one of: none, zscore, zscore_clip, minmax.")
    if norm in {"zscore", "zscore_clip"}:
        mean = edge_attr.mean(axis=0, keepdims=True)
        std = edge_attr.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        edge_attr = (edge_attr - mean) / std
        if norm == "zscore_clip" and args.edge_attr_clip is not None and float(args.edge_attr_clip) > 0:
            edge_attr = np.clip(edge_attr, -float(args.edge_attr_clip), float(args.edge_attr_clip))
        edge_attr = edge_attr.astype(np.float32)
    elif norm == "minmax":
        mn = edge_attr.min(axis=0, keepdims=True)
        mx = edge_attr.max(axis=0, keepdims=True)
        denom = np.where((mx - mn) < 1e-6, 1.0, (mx - mn))
        edge_attr = ((edge_attr - mn) / denom).astype(np.float32)

    # Optional Gaussian edge noise (seeded).
    if args.edge_noise_std is not None and float(args.edge_noise_std) > 0:
        rng = np.random.default_rng(int(args.seed) if args.seed is not None else 0)
        edge_attr = (edge_attr + rng.normal(loc=0.0, scale=float(args.edge_noise_std), size=edge_attr.shape)).astype(np.float32)

    n_clusters = int(np.unique(y_true).size)
    node_edge = _pool_edge_attr_to_nodes_mean(adj, edge_attr, n_nodes=int(x.shape[0]))

    results: list[dict[str, Any]] = []

    for method in methods:
        if method == "kmeans_x":
            y_pred = _run_kmeans_x(x, n_clusters=n_clusters, seed=seed)
        elif method == "kmeans_edge_mean":
            y_pred = _run_kmeans_features(node_edge, n_clusters=n_clusters, seed=seed)
        elif method == "kmeans_x_edge_mean":
            feats = np.concatenate([x.astype(np.float32, copy=False), node_edge], axis=1)
            y_pred = _run_kmeans_features(feats, n_clusters=n_clusters, seed=seed)
        elif method == "spectral_adj_binary":
            y_pred = _run_spectral(adj, n_clusters=n_clusters, seed=seed)
        elif method == "spectral_edge_distance":
            w_adj = _weighted_adj_from_edge_distance(adj, edge_attr, gamma=float(args.distance_gamma))
            y_pred = _run_spectral(w_adj, n_clusters=n_clusters, seed=seed)
        elif method == "spectral_edge_l2":
            w_adj = _weighted_adj_from_edge_l2(adj, edge_attr, gamma=float(args.distance_gamma))
            y_pred = _run_spectral(w_adj, n_clusters=n_clusters, seed=seed)
        elif method == "spectral_node_edge_rbf":
            w_adj = _weighted_adj_from_node_edge_rbf(adj, node_edge, gamma=float(args.distance_gamma))
            y_pred = _run_spectral(w_adj, n_clusters=n_clusters, seed=seed)
        else:
            raise SystemExit(f"Unknown baseline method: {method}")

        acc, f1, nmi, ari = eva(y_true, y_pred, epoch=f"baseline:{method}")
        results.append(
            {
                "method": method,
                "metrics": {"acc": float(acc), "f1": float(f1), "nmi": float(nmi), "ari": float(ari)},
                "cluster_distribution": _cluster_distribution(y_pred),
            }
        )

    summary = {
        "data_dir": os.path.abspath(args.data_dir),
        "seed": seed,
        "edge_attr_norm": norm,
        "edge_attr_clip": float(args.edge_attr_clip),
        "edge_noise_std": float(args.edge_noise_std),
        "n_clusters": n_clusters,
        "n_nodes": int(x.shape[0]),
        "baselines": results,
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to: {os.path.abspath(args.summary_json)}")


if __name__ == "__main__":
    main()
