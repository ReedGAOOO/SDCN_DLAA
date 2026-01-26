#!/usr/bin/env python3
"""
Run classic baseline clustering methods on a labeled dataset directory.

Baselines (by default):
- kmeans_x: KMeans on node features
- spectral_adj_binary: Spectral clustering on binary adjacency
- spectral_edge_distance: Spectral clustering on distance-derived weights (uses edge_attr[:,0])

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


def _run_kmeans_x(x: np.ndarray, n_clusters: int, seed: int | None) -> np.ndarray:
    x_std = StandardScaler().fit_transform(x)
    kwargs: dict[str, Any] = {"n_clusters": n_clusters, "n_init": 20}
    if seed is not None:
        kwargs["random_state"] = seed
    return KMeans(**kwargs).fit_predict(x_std)


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
    adj = adj.tocsr()
    rows, cols = adj.nonzero()
    if edge_attr.shape[0] != rows.shape[0]:
        raise ValueError(
            f"edge_attr row count mismatch: edge_attr has {edge_attr.shape[0]}, "
            f"but adjacency has {rows.shape[0]} nonzeros"
        )
    dist = edge_attr[:, 0].astype(np.float32).reshape(-1)
    dist_norm = _minmax(dist)
    weights = np.exp(-gamma * dist_norm).astype(np.float32)
    return sp.csr_matrix((weights, (rows, cols)), shape=adj.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", type=str, default="kmeans_x,spectral_adj_binary,spectral_edge_distance")
    parser.add_argument("--distance_gamma", type=float, default=5.0)
    parser.add_argument("--summary_json", type=str, default="summary_baselines.json")
    args = parser.parse_args()

    seed = int(args.seed) if args.seed is not None else None
    methods = [m.strip() for m in args.methods.split(",") if m.strip() != ""]

    x = np.load(os.path.join(args.data_dir, "node_features.npy")).astype(np.float32)
    y_true = np.load(os.path.join(args.data_dir, "labels.npy")).astype(np.int64)
    adj = sp.load_npz(os.path.join(args.data_dir, "binary_adj.npz")).tocsr()
    edge_attr = np.load(os.path.join(args.data_dir, "edge_attr.npy")).astype(np.float32)

    n_clusters = int(np.unique(y_true).size)

    results: list[dict[str, Any]] = []

    for method in methods:
        if method == "kmeans_x":
            y_pred = _run_kmeans_x(x, n_clusters=n_clusters, seed=seed)
        elif method == "spectral_adj_binary":
            y_pred = _run_spectral(adj, n_clusters=n_clusters, seed=seed)
        elif method == "spectral_edge_distance":
            w_adj = _weighted_adj_from_edge_distance(adj, edge_attr, gamma=float(args.distance_gamma))
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
        "n_clusters": n_clusters,
        "n_nodes": int(x.shape[0]),
        "baselines": results,
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to: {os.path.abspath(args.summary_json)}")


if __name__ == "__main__":
    main()

