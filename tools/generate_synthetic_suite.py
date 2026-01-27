#!/usr/bin/env python3
"""
Generate multiple synthetic graph clustering datasets for benchmarking:

Two categories:
1) Traditional: node clustering + 1D edge distance features.
2) Rich-edge: higher-dimensional edge attributes with practical semantics.

Each dataset directory contains:
- node_features.npy: float32 [N, F]
- labels.npy: int64 [N]
- coords.npy: float32 [N, 2] (for debugging/visualization)
- binary_adj.npz: scipy sparse CSR adjacency (0/1) [N, N]
- edge_attr.npy: float32 [E, edge_dim] aligned to CSR nonzero order (row-major)
- edge_index.npy: int64 [2, E] (row/col pairs, same order as edge_attr)
- data_info.json: metadata

Example:
  python tools/generate_synthetic_suite.py --output_root /tmp/sdcn_suite --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable
import zlib

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    category: str  # "distance_1d" | "rich_edge"
    seed: int
    n_clusters: int
    num_nodes: int
    feature_dim: int
    edge_dim: int
    graph_knn_k: int
    extra: dict


def _minmax(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _make_knn_edges(coords: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nn.fit(coords)
    indices = nn.kneighbors(return_distance=False)

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            edges.add((i, int(j)))
            edges.add((int(j), i))

    rows_cols = sorted(edges)
    rows = np.fromiter((rc[0] for rc in rows_cols), dtype=np.int64, count=len(rows_cols))
    cols = np.fromiter((rc[1] for rc in rows_cols), dtype=np.int64, count=len(rows_cols))
    return rows, cols


def _make_random_k_edges(num_nodes: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an undirected random-k graph by sampling k neighbors per node.
    Returns directed edges (both directions) in a stable sorted order.
    """
    n = int(num_nodes)
    if n <= 1:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    k = int(max(0, min(int(k), n - 1)))
    if k <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    edges: set[tuple[int, int]] = set()
    all_nodes = np.arange(n, dtype=np.int64)
    for i in range(n):
        candidates = all_nodes[all_nodes != i]
        if candidates.size == 0:
            continue
        nbrs = rng.choice(candidates, size=min(k, int(candidates.size)), replace=False)
        for j in nbrs.tolist():
            edges.add((i, int(j)))
            edges.add((int(j), i))

    rows_cols = sorted(edges)
    rows = np.fromiter((rc[0] for rc in rows_cols), dtype=np.int64, count=len(rows_cols))
    cols = np.fromiter((rc[1] for rc in rows_cols), dtype=np.int64, count=len(rows_cols))
    return rows, cols


def _add_random_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    per_node: int,
    p_within: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = labels.shape[0]
    edges: set[tuple[int, int]] = set(zip(rows.tolist(), cols.tolist()))

    clusters = {cid: np.where(labels == cid)[0].tolist() for cid in np.unique(labels)}
    all_nodes = np.arange(n, dtype=np.int64)

    for i in range(n):
        src_c = int(labels[i])
        within_nodes = clusters[src_c]
        for _ in range(per_node):
            if rng.random() < p_within and len(within_nodes) > 1:
                j = int(rng.choice([v for v in within_nodes if v != i]))
            else:
                # sample from other clusters
                candidates = all_nodes[labels != src_c]
                if candidates.size == 0:
                    continue
                j = int(rng.choice(candidates))
            if i == j:
                continue
            edges.add((i, j))
            edges.add((j, i))

    rows_cols = sorted(edges)
    rows = np.fromiter((rc[0] for rc in rows_cols), dtype=np.int64, count=len(rows_cols))
    cols = np.fromiter((rc[1] for rc in rows_cols), dtype=np.int64, count=len(rows_cols))
    return rows, cols


def _make_node_features(
    rng: np.random.Generator,
    labels: np.ndarray,
    coords: np.ndarray,
    feature_dim: int,
    cluster_bias: float,
    coord_proj_scale: float,
    noise_scale: float,
) -> np.ndarray:
    x = rng.normal(loc=0.0, scale=noise_scale, size=(labels.shape[0], feature_dim)).astype(np.float32)

    # Small cluster bias on first channel to avoid fully degenerate node-only setting.
    if feature_dim > 0 and cluster_bias > 0.0:
        bias = (labels.astype(np.float32) / max(int(labels.max()), 1)) * cluster_bias
        x[:, 0] = x[:, 0] + bias

    # Optional coordinate projection (makes node features partially spatial).
    if coord_proj_scale > 0.0:
        proj = rng.normal(loc=0.0, scale=coord_proj_scale, size=(2, feature_dim)).astype(np.float32)
        x = x + coords @ proj

    return x.astype(np.float32)


def _pairwise_dist_for_edges(coords: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    diffs = coords[rows] - coords[cols]
    return np.sqrt(np.sum(diffs * diffs, axis=1, dtype=np.float32)).astype(np.float32)


def _make_edge_attr_distance_1d(coords: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    d = _pairwise_dist_for_edges(coords, rows, cols)
    d_norm = _minmax(d).reshape(-1, 1)
    return d_norm.astype(np.float32)


def _make_edge_attr_rich_profiles(
    rng: np.random.Generator,
    coords: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    edge_dim: int,
    latent_dim: int = 8,
    profile_noise: float = 0.6,
) -> np.ndarray:
    """
    Rich edge features derived from latent node profiles + geometry.
    Interpretable channels:
    - distance (normalized)
    - direction (dx, dy)
    - latent similarity metrics (cosine, l2, dot)
    Remaining channels are nonlinear transforms / noise (for higher dimensionality).
    """
    n = labels.shape[0]
    n_clusters = int(np.unique(labels).size)

    # Cluster-specific profile centers.
    centers = rng.normal(loc=0.0, scale=1.0, size=(n_clusters, latent_dim)).astype(np.float32)
    profiles = centers[labels] + rng.normal(loc=0.0, scale=profile_noise, size=(n, latent_dim)).astype(np.float32)

    d = _pairwise_dist_for_edges(coords, rows, cols)
    d_norm = _minmax(d)

    diffs = coords[cols] - coords[rows]
    dir_norm = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-6)

    p_i = profiles[rows]
    p_j = profiles[cols]
    dot = np.sum(p_i * p_j, axis=1, dtype=np.float32)
    l2 = np.sqrt(np.sum((p_i - p_j) ** 2, axis=1, dtype=np.float32))
    cos = dot / ((np.linalg.norm(p_i, axis=1) * np.linalg.norm(p_j, axis=1)) + 1e-6)

    base = [
        d_norm,
        dir_norm[:, 0].astype(np.float32),
        dir_norm[:, 1].astype(np.float32),
        _minmax(cos.astype(np.float32)),
        _minmax(l2.astype(np.float32)),
        _minmax(dot.astype(np.float32)),
    ]
    base_mat = np.stack(base, axis=1).astype(np.float32)

    if edge_dim <= base_mat.shape[1]:
        return base_mat[:, :edge_dim].astype(np.float32)

    out = np.zeros((rows.shape[0], edge_dim), dtype=np.float32)
    out[:, : base_mat.shape[1]] = base_mat

    # Fill remaining dims with nonlinear transforms + small noise.
    cursor = base_mat.shape[1]
    while cursor < edge_dim:
        src = out[:, 0]  # distance
        if cursor < edge_dim:
            out[:, cursor] = np.exp(-src * 5.0).astype(np.float32)
            cursor += 1
        if cursor < edge_dim:
            out[:, cursor] = (src * src).astype(np.float32)
            cursor += 1
        if cursor < edge_dim:
            out[:, cursor] = (1.0 / (src + 1e-3)).astype(np.float32)
            cursor += 1
        if cursor < edge_dim:
            out[:, cursor] = rng.normal(loc=0.0, scale=0.05, size=rows.shape[0]).astype(np.float32)
            cursor += 1

    return out.astype(np.float32)


def _make_edge_attr_rich_geo_temporal(
    rng: np.random.Generator,
    coords: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    edge_dim: int,
) -> tuple[np.ndarray, dict]:
    """
    Geo/temporal edge features (distance, delta_t, speed, direction, road type).
    Road type distribution differs by cluster (proxy for region/road class).
    """
    n = labels.shape[0]
    n_clusters = int(np.unique(labels).size)

    # Per-node "timestamp" with cluster-dependent offsets.
    t_center = np.linspace(0.0, 10.0, n_clusters, dtype=np.float32)
    t = t_center[labels] + rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float32)

    d = _pairwise_dist_for_edges(coords, rows, cols)
    d_norm = _minmax(d)

    dt = np.abs(t[rows] - t[cols]).astype(np.float32)
    dt_norm = _minmax(dt)
    speed = d / (dt + 0.25)
    speed_norm = _minmax(speed.astype(np.float32))

    diffs = coords[cols] - coords[rows]
    dir_norm = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-6)

    # Road type: 3 classes, cluster-dependent multinomial.
    road_probs = np.full((n_clusters, 3), 1 / 3, dtype=np.float32)
    for c in range(n_clusters):
        road_probs[c] = np.roll(np.array([0.6, 0.3, 0.1], dtype=np.float32), c % 3)

    same_cluster = labels[rows] == labels[cols]
    road_type = np.zeros(rows.shape[0], dtype=np.int64)
    for idx in range(rows.shape[0]):
        if same_cluster[idx]:
            c = int(labels[rows[idx]])
            road_type[idx] = int(rng.choice(3, p=road_probs[c]))
        else:
            road_type[idx] = int(rng.choice(3, p=np.array([0.33, 0.33, 0.34], dtype=np.float32)))

    road_onehot = np.eye(3, dtype=np.float32)[road_type]

    base_mat = np.concatenate(
        [
            d_norm.reshape(-1, 1).astype(np.float32),
            dt_norm.reshape(-1, 1),
            speed_norm.reshape(-1, 1),
            dir_norm.astype(np.float32),
            road_onehot.astype(np.float32),
        ],
        axis=1,
    )

    # Pad/truncate.
    if edge_dim <= base_mat.shape[1]:
        edge_attr = base_mat[:, :edge_dim].astype(np.float32)
    else:
        edge_attr = np.zeros((rows.shape[0], edge_dim), dtype=np.float32)
        edge_attr[:, : base_mat.shape[1]] = base_mat
        edge_attr[:, base_mat.shape[1] :] = rng.normal(loc=0.0, scale=0.05, size=(rows.shape[0], edge_dim - base_mat.shape[1])).astype(np.float32)

    extra = {"road_type_counts": {int(k): int(v) for k, v in zip(*np.unique(road_type, return_counts=True))}}
    return edge_attr.astype(np.float32), extra


def _make_edge_attr_rich_multirelation(
    rng: np.random.Generator,
    coords: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    edge_dim: int,
    n_relations: int = 4,
) -> tuple[np.ndarray, dict]:
    """
    Multi-relation edge attributes, inspired by social/interaction graphs.

    Each edge carries:
    - distance (normalized)
    - interaction strength (lognormal)
    - message count (Poisson, normalized)
    - reciprocity (0/1)
    - relation type one-hot (n_relations)
    Remaining dims are noise.

    Relation type distribution differs by cluster for within-cluster edges.
    """
    n_clusters = int(np.unique(labels).size)

    d = _pairwise_dist_for_edges(coords, rows, cols)
    d_norm = _minmax(d)

    # Cluster-specific relation probabilities (within-cluster edges).
    rel_probs = np.zeros((n_clusters, n_relations), dtype=np.float32)
    base = np.linspace(0.55, 0.15, n_relations, dtype=np.float32)
    base = base / base.sum()
    for c in range(n_clusters):
        rel_probs[c] = np.roll(base, c % n_relations)

    # Cross-cluster edges use a flatter distribution (more "weak ties").
    cross_probs = np.full(n_relations, 1.0 / n_relations, dtype=np.float32)

    same = labels[rows] == labels[cols]
    rel = np.zeros(rows.shape[0], dtype=np.int64)
    for i in range(rows.shape[0]):
        if same[i]:
            c = int(labels[rows[i]])
            rel[i] = int(rng.choice(n_relations, p=rel_probs[c]))
        else:
            rel[i] = int(rng.choice(n_relations, p=cross_probs))

    rel_onehot = np.eye(n_relations, dtype=np.float32)[rel]

    # Relation-dependent interaction strength.
    strength_mu = np.linspace(1.2, 0.3, n_relations, dtype=np.float32)
    strength_sigma = np.linspace(0.35, 0.55, n_relations, dtype=np.float32)
    strength = np.zeros(rows.shape[0], dtype=np.float32)
    for r in range(n_relations):
        mask = rel == r
        if not np.any(mask):
            continue
        strength[mask] = rng.lognormal(mean=float(strength_mu[r]), sigma=float(strength_sigma[r]), size=int(mask.sum())).astype(np.float32)
    strength_norm = _minmax(strength)

    # Message count correlated with strength.
    msg_lambda = 2.0 + 10.0 * strength_norm
    msg_count = rng.poisson(lam=msg_lambda).astype(np.float32)
    msg_norm = _minmax(msg_count)

    # Reciprocity: stronger relations more reciprocal; cross edges lower.
    recip_p = 0.15 + 0.7 * strength_norm
    recip_p = np.where(same, recip_p, 0.10 + 0.30 * strength_norm).astype(np.float32)
    reciprocity = (rng.random(size=rows.shape[0]) < recip_p).astype(np.float32)

    base_mat = np.concatenate(
        [
            d_norm.reshape(-1, 1).astype(np.float32),
            strength_norm.reshape(-1, 1),
            msg_norm.reshape(-1, 1),
            reciprocity.reshape(-1, 1),
            rel_onehot.astype(np.float32),
        ],
        axis=1,
    )

    if edge_dim <= base_mat.shape[1]:
        edge_attr = base_mat[:, :edge_dim].astype(np.float32)
    else:
        edge_attr = np.zeros((rows.shape[0], edge_dim), dtype=np.float32)
        edge_attr[:, : base_mat.shape[1]] = base_mat
        edge_attr[:, base_mat.shape[1] :] = rng.normal(loc=0.0, scale=0.05, size=(rows.shape[0], edge_dim - base_mat.shape[1])).astype(np.float32)

    extra = {
        "relation_type_counts": {int(k): int(v) for k, v in zip(*np.unique(rel, return_counts=True))},
        "n_relations": int(n_relations),
    }
    return edge_attr.astype(np.float32), extra


def _make_edge_attr_rich_semantic_only(
    rng: np.random.Generator,
    coords: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    edge_dim: int,
) -> tuple[np.ndarray, dict]:
    """
    Rich edge features where *distance is not predictive*, but semantic channels are.

    Intended to stress-test edge-aware models against distance-based baselines:
    - Labels are decoupled from geometry.
    - Graph is built from KNN in coordinate space.
    - Edge attributes carry semantic signals correlated with true labels.

    Channels (starting):
    - dist_norm (still present, but not label-predictive by design)
    - same_cluster (0/1)
    - relation type one-hot (n_clusters) for within-cluster edges; random for cross-cluster edges
    - strength (noisy, higher when same_cluster)
    Remaining dims are noise.
    """
    n_clusters = int(np.unique(labels).size)

    d = _pairwise_dist_for_edges(coords, rows, cols)
    d_norm = _minmax(d)

    same = (labels[rows] == labels[cols]).astype(np.float32)

    # For within-cluster edges: relation_type = cluster id; for cross: random type.
    rel = np.zeros(rows.shape[0], dtype=np.int64)
    for i in range(rows.shape[0]):
        if same[i] > 0.5:
            rel[i] = int(labels[rows[i]])
        else:
            rel[i] = int(rng.integers(low=0, high=max(n_clusters, 1)))
    rel_onehot = np.eye(n_clusters, dtype=np.float32)[rel] if n_clusters > 0 else np.zeros((rows.shape[0], 0), dtype=np.float32)

    strength = (0.6 * same + 0.4 * rng.random(size=rows.shape[0]).astype(np.float32)).astype(np.float32)

    base_mat = np.concatenate(
        [
            d_norm.reshape(-1, 1).astype(np.float32),
            same.reshape(-1, 1),
            rel_onehot.astype(np.float32),
            strength.reshape(-1, 1),
        ],
        axis=1,
    )

    if edge_dim <= base_mat.shape[1]:
        edge_attr = base_mat[:, :edge_dim].astype(np.float32)
    else:
        edge_attr = np.zeros((rows.shape[0], edge_dim), dtype=np.float32)
        edge_attr[:, : base_mat.shape[1]] = base_mat
        edge_attr[:, base_mat.shape[1] :] = rng.normal(loc=0.0, scale=0.05, size=(rows.shape[0], edge_dim - base_mat.shape[1])).astype(np.float32)

    extra = {
        "n_clusters": n_clusters,
        "same_edge_frac": float(same.mean()) if same.size else 0.0,
    }
    return edge_attr.astype(np.float32), extra


def _make_edge_attr_rich_semantic_only_nonknn(
    rng: np.random.Generator,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: np.ndarray,
    edge_dim: int,
) -> tuple[np.ndarray, dict]:
    """
    Semantic-only edge attributes for non-KNN graphs.

    NOTE: edge_attr[:,0] is intentionally uninformative random noise so that
    `spectral_edge_distance` (which only uses edge_attr[:,0]) cannot exploit it.
    """
    n_clusters = int(np.unique(labels).size)
    same = (labels[rows] == labels[cols]).astype(np.float32)

    noise0 = rng.random(size=rows.shape[0]).astype(np.float32)

    rel = np.zeros(rows.shape[0], dtype=np.int64)
    for i in range(rows.shape[0]):
        if same[i] > 0.5:
            rel[i] = int(labels[rows[i]])
        else:
            rel[i] = int(rng.integers(low=0, high=max(n_clusters, 1)))
    rel_onehot = np.eye(n_clusters, dtype=np.float32)[rel] if n_clusters > 0 else np.zeros((rows.shape[0], 0), dtype=np.float32)

    strength = (0.75 * same + 0.25 * rng.random(size=rows.shape[0]).astype(np.float32)).astype(np.float32)

    base_mat = np.concatenate(
        [
            noise0.reshape(-1, 1),
            same.reshape(-1, 1),
            rel_onehot.astype(np.float32),
            strength.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    if edge_dim <= base_mat.shape[1]:
        edge_attr = base_mat[:, :edge_dim].astype(np.float32)
    else:
        edge_attr = np.zeros((rows.shape[0], edge_dim), dtype=np.float32)
        edge_attr[:, : base_mat.shape[1]] = base_mat
        edge_attr[:, base_mat.shape[1] :] = rng.normal(loc=0.0, scale=0.05, size=(rows.shape[0], edge_dim - base_mat.shape[1])).astype(np.float32)

    extra = {
        "n_clusters": n_clusters,
        "same_edge_frac": float(same.mean()) if same.size else 0.0,
        "edge_attr": "semantic_only_nonknn",
        "edge_attr_note": "edge_attr[:,0] is random noise; semantics live in later channels",
    }
    return edge_attr.astype(np.float32), extra


def _save_dataset(
    output_dir: str,
    *,
    x: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    edge_attr: np.ndarray,
    info: DatasetInfo,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n = x.shape[0]
    e = rows.shape[0]

    adj = sp.csr_matrix((np.ones(e, dtype=np.float32), (rows, cols)), shape=(n, n))

    np.save(os.path.join(output_dir, "node_features.npy"), x.astype(np.float32))
    np.save(os.path.join(output_dir, "labels.npy"), labels.astype(np.int64))
    np.save(os.path.join(output_dir, "coords.npy"), coords.astype(np.float32))
    sp.save_npz(os.path.join(output_dir, "binary_adj.npz"), adj)
    np.save(os.path.join(output_dir, "edge_attr.npy"), edge_attr.astype(np.float32))
    np.save(os.path.join(output_dir, "edge_index.npy"), np.vstack([rows, cols]).astype(np.int64))

    with open(os.path.join(output_dir, "data_info.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(info), f, indent=2)


def _preset_dist_blobs_easy(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n_clusters = 3
    points_per = 60

    centers = np.array([[0.0, 0.0], [4.0, 0.5], [2.0, 3.8]], dtype=np.float32)
    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), points_per)
    coords = np.vstack([centers[c] + rng.normal(0.0, 0.55, size=(points_per, 2)) for c in range(n_clusters)]).astype(np.float32)

    perm = rng.permutation(coords.shape[0])
    coords = coords[perm]
    labels = labels[perm]

    x = _make_node_features(rng, labels, coords, feature_dim=6, cluster_bias=0.25, coord_proj_scale=0.25, noise_scale=1.0)
    return coords, labels, x, {"points_per_cluster": points_per}


def _preset_dist_blobs_overlap(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n_clusters = 3
    points_per = 60

    centers = np.array([[0.0, 0.0], [2.0, 0.8], [1.0, 2.0]], dtype=np.float32)
    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), points_per)
    coords = np.vstack([centers[c] + rng.normal(0.0, 1.0, size=(points_per, 2)) for c in range(n_clusters)]).astype(np.float32)

    perm = rng.permutation(coords.shape[0])
    coords = coords[perm]
    labels = labels[perm]

    x = _make_node_features(rng, labels, coords, feature_dim=6, cluster_bias=0.12, coord_proj_scale=0.10, noise_scale=1.2)
    return coords, labels, x, {"points_per_cluster": points_per}


def _preset_dist_two_moons(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    coords, labels = make_moons(n_samples=220, noise=0.08, random_state=seed)
    coords = coords.astype(np.float32)
    labels = labels.astype(np.int64)

    perm = rng.permutation(coords.shape[0])
    coords = coords[perm]
    labels = labels[perm]

    x = _make_node_features(rng, labels, coords, feature_dim=6, cluster_bias=0.05, coord_proj_scale=0.20, noise_scale=1.0)
    return coords, labels, x, {"n_samples": int(coords.shape[0])}


def _preset_rich_edge_profiles(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    coords, labels, x, extra = _preset_dist_blobs_overlap(seed)
    extra = {**extra, "note": "weak node features, strong edge profile signal"}
    return coords, labels, x, extra


def _preset_rich_geo_temporal(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n_clusters = 3
    points_per = 60

    centers = np.array([[0.0, 0.0], [3.5, 0.2], [2.0, 3.0]], dtype=np.float32)
    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), points_per)
    coords = np.vstack([centers[c] + rng.normal(0.0, 0.9, size=(points_per, 2)) for c in range(n_clusters)]).astype(np.float32)

    perm = rng.permutation(coords.shape[0])
    coords = coords[perm]
    labels = labels[perm]

    # Node features are intentionally noisy; include a tiny projection.
    x = _make_node_features(rng, labels, coords, feature_dim=4, cluster_bias=0.08, coord_proj_scale=0.10, noise_scale=1.2)
    return coords, labels, x, {"points_per_cluster": points_per, "note": "geo-temporal rich edge attrs"}


def _preset_rich_multirelation(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n_clusters = 4
    points_per = 50

    centers = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), points_per)
    coords = np.vstack([centers[c] + rng.normal(0.0, 0.8, size=(points_per, 2)) for c in range(n_clusters)]).astype(np.float32)

    perm = rng.permutation(coords.shape[0])
    coords = coords[perm]
    labels = labels[perm]

    # Node features: weakly informative.
    x = _make_node_features(rng, labels, coords, feature_dim=6, cluster_bias=0.10, coord_proj_scale=0.05, noise_scale=1.0)
    return coords, labels, x, {"points_per_cluster": points_per, "note": "multi-relation rich edge attrs"}


def _preset_rich_edge_semantic_only(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Labels are decoupled from geometry; edge_attr contains semantic signal.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 3
    points_per = 60

    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), points_per)
    labels = labels[rng.permutation(labels.shape[0])]

    # Single cloud: coords not predictive of labels.
    coords = rng.normal(loc=0.0, scale=1.0, size=(labels.shape[0], 2)).astype(np.float32)

    # Node features are weak/uninformative.
    x = _make_node_features(rng, labels, coords, feature_dim=6, cluster_bias=0.0, coord_proj_scale=0.0, noise_scale=1.0)
    return coords, labels, x, {"points_per_cluster": points_per, "note": "geometry uninformative; edge_attr carries semantic signal"}

PresetFn = Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray, dict]]


PRESETS: dict[str, dict] = {
    # Traditional distance-only edge features.
    "dist_blobs_easy": {"category": "distance_1d", "fn": _preset_dist_blobs_easy, "edge_dim": 1, "knn_k": 10},
    "dist_blobs_overlap": {"category": "distance_1d", "fn": _preset_dist_blobs_overlap, "edge_dim": 1, "knn_k": 10},
    "dist_two_moons": {"category": "distance_1d", "fn": _preset_dist_two_moons, "edge_dim": 1, "knn_k": 12},
    # Rich edge attributes.
    "rich_edge_profiles": {"category": "rich_edge", "fn": _preset_rich_edge_profiles, "edge_dim": 16, "knn_k": 10},
    "rich_geo_temporal": {"category": "rich_edge", "fn": _preset_rich_geo_temporal, "edge_dim": 12, "knn_k": 10},
    "rich_multirelation": {"category": "rich_edge", "fn": _preset_rich_multirelation, "edge_dim": 20, "knn_k": 10},
    "rich_edge_semantic_only": {"category": "rich_edge", "fn": _preset_rich_edge_semantic_only, "edge_dim": 16, "knn_k": 10},
    "rich_edge_semantic_only_nonknn": {
        "category": "rich_edge",
        "fn": _preset_rich_edge_semantic_only,
        "edge_dim": 16,
        "knn_k": 10,  # used as random-k for this preset
        "graph_type": "random_k",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="/tmp/sdcn_dlaa_synth_suite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--presets",
        type=str,
        default=",".join(PRESETS.keys()),
        help="Comma-separated preset names. Use 'all' for all presets.",
    )
    parser.add_argument("--list_presets", action="store_true", help="Print available presets and exit.")
    parser.add_argument("--random_edges_per_node", type=int, default=2, help="Extra random edges per node (rich-edge presets only).")
    parser.add_argument("--random_edges_within_prob", type=float, default=0.5, help="Probability random edge stays within cluster.")
    args = parser.parse_args()

    if args.list_presets:
        for name, spec in PRESETS.items():
            print(f"{name}\tcategory={spec['category']}\tedge_dim={spec['edge_dim']}\tknn_k={spec['knn_k']}")
        return

    if args.presets.strip().lower() == "all":
        preset_names = list(PRESETS.keys())
    else:
        preset_names = [p.strip() for p in args.presets.split(",") if p.strip() != ""]

    unknown = [p for p in preset_names if p not in PRESETS]
    if unknown:
        raise SystemExit(f"Unknown presets: {unknown}. Available: {sorted(PRESETS.keys())}")

    out_root = os.path.abspath(args.output_root)
    os.makedirs(out_root, exist_ok=True)

    for name in preset_names:
        spec = PRESETS[name]
        category = str(spec["category"])
        fn: PresetFn = spec["fn"]
        edge_dim = int(spec["edge_dim"])
        knn_k = int(spec["knn_k"])
        graph_type = str(spec.get("graph_type", "knn")).strip().lower()

        coords, labels, x, extra = fn(args.seed)

        # Stable per-dataset RNG (independent of Python's hash randomization).
        name_hash = int(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
        rng = np.random.default_rng(int(args.seed) * 1_000_003 + name_hash)
        if graph_type in {"random_k", "rand_k", "randomk"}:
            rows, cols = _make_random_k_edges(int(coords.shape[0]), k=knn_k, rng=rng)
            extra = {**extra, "graph": f"random_k(k={knn_k})"}
        else:
            rows, cols = _make_knn_edges(coords, k=knn_k, rng=rng)

        if category == "rich_edge":
            rows, cols = _add_random_edges(
                rows,
                cols,
                labels,
                rng=rng,
                per_node=args.random_edges_per_node,
                p_within=float(args.random_edges_within_prob),
            )

        if category == "distance_1d":
            edge_attr = _make_edge_attr_distance_1d(coords, rows, cols)
            extra = {**extra, "edge_attr": "distance_norm_1d"}
        else:
            if name == "rich_geo_temporal":
                edge_attr, extra_edge = _make_edge_attr_rich_geo_temporal(rng, coords, rows, cols, labels, edge_dim=edge_dim)
                extra = {**extra, **extra_edge, "edge_attr": "geo_temporal"}
            elif name == "rich_multirelation":
                edge_attr, extra_edge = _make_edge_attr_rich_multirelation(rng, coords, rows, cols, labels, edge_dim=edge_dim)
                extra = {**extra, **extra_edge, "edge_attr": "multi_relation"}
            elif name == "rich_edge_semantic_only":
                edge_attr, extra_edge = _make_edge_attr_rich_semantic_only(rng, coords, rows, cols, labels, edge_dim=edge_dim)
                extra = {**extra, **extra_edge, "edge_attr": "semantic_only"}
            elif name == "rich_edge_semantic_only_nonknn":
                edge_attr, extra_edge = _make_edge_attr_rich_semantic_only_nonknn(rng, rows, cols, labels, edge_dim=edge_dim)
                extra = {**extra, **extra_edge}
            else:
                edge_attr = _make_edge_attr_rich_profiles(rng, coords, rows, cols, labels, edge_dim=edge_dim)
                extra = {**extra, "edge_attr": "profiles_plus_noise"}

        info = DatasetInfo(
            name=name,
            category=category,
            seed=int(args.seed),
            n_clusters=int(np.unique(labels).size),
            num_nodes=int(x.shape[0]),
            feature_dim=int(x.shape[1]),
            edge_dim=int(edge_attr.shape[1]),
            graph_knn_k=int(knn_k),
            extra=extra,
        )

        out_dir = os.path.join(out_root, name)
        _save_dataset(
            out_dir,
            x=x,
            labels=labels,
            coords=coords,
            rows=rows,
            cols=cols,
            edge_attr=edge_attr,
            info=info,
        )

        print(f"Wrote dataset: {name} -> {out_dir} (N={x.shape[0]}, E={rows.shape[0]}, edge_dim={edge_attr.shape[1]})")


if __name__ == "__main__":
    main()
