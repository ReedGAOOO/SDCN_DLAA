#!/usr/bin/env python3
"""
Analyze "why baselines are strong" by measuring signal strength in the synthetic suite.

For each dataset directory (layout produced by tools/generate_synthetic_suite.py), compute:
- graph homophily (edge endpoints share label)
- distance separability on edges (AUC using edge_attr[:,0] as distance-like feature)
- silhouette scores for coords and node features (label-based, diagnostic only)

Outputs:
- JSON summary (per-dataset + aggregate)
- Optional Markdown table
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, silhouette_score


@dataclass(frozen=True)
class DatasetSignal:
    dataset: str
    data_dir: str
    category: str
    n_nodes: int
    n_edges: int
    n_clusters: int
    feature_dim: int
    edge_dim: int
    homophily: float
    edge_distance_auc: float | None
    edge_distance_within_mean: float | None
    edge_distance_between_mean: float | None
    edge_distance_effect_size_d: float | None
    coords_silhouette: float | None
    x_silhouette: float | None


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = ((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1)
    if pooled <= 0:
        return None
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def _safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        if np.unique(labels).size < 2:
            return None
        # silhouette_score can error if some cluster has 1 sample.
        counts = np.bincount(labels - labels.min())
        if np.any(counts < 2):
            return None
        return float(silhouette_score(x, labels, metric="euclidean"))
    except Exception:
        return None


def _dataset_dirs(suite_dir: Path, datasets: list[str] | None) -> list[Path]:
    if datasets:
        out: list[Path] = []
        for name in datasets:
            d = (suite_dir / name).resolve()
            if not d.is_dir():
                raise SystemExit(f"Dataset directory not found: {d}")
            out.append(d)
        return out

    needed = {"node_features.npy", "labels.npy", "binary_adj.npz", "edge_attr.npy", "data_info.json"}
    out: list[Path] = []
    for d in sorted(suite_dir.iterdir()):
        if not d.is_dir():
            continue
        names = {p.name for p in d.iterdir()}
        if needed.issubset(names):
            out.append(d)
    return out


def _load_info(data_dir: Path) -> dict:
    info_path = data_dir / "data_info.json"
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def analyze_dataset(data_dir: Path) -> DatasetSignal:
    info = _load_info(data_dir)
    dataset = data_dir.name
    category = str(info.get("category", "n/a"))

    x = np.load(data_dir / "node_features.npy").astype(np.float32)
    y = np.load(data_dir / "labels.npy").astype(np.int64)
    edge_attr = np.load(data_dir / "edge_attr.npy").astype(np.float32)
    adj = sp.load_npz(data_dir / "binary_adj.npz").tocoo()

    n_nodes = int(x.shape[0])
    n_edges = int(adj.nnz)
    n_clusters = int(np.unique(y).size)

    # Graph homophily (directed, consistent with stored adjacency).
    homophily = float(np.mean(y[adj.row] == y[adj.col])) if adj.nnz > 0 else 0.0

    # Edge distance separability (only uses edge_attr[:,0], treating it as distance-like).
    edge_distance_auc: float | None = None
    within_mean: float | None = None
    between_mean: float | None = None
    effect_d: float | None = None

    if edge_attr.shape[0] == adj.nnz and edge_attr.shape[1] >= 1 and adj.nnz > 0:
        same = (y[adj.row] == y[adj.col]).astype(np.int8)
        dist = edge_attr[:, 0].astype(np.float32).reshape(-1)

        within = dist[same == 1]
        between = dist[same == 0]
        within_mean = float(within.mean()) if within.size else None
        between_mean = float(between.mean()) if between.size else None
        effect_d = _cohen_d(within, between)

        # If both classes exist, compute AUC using -distance as score (smaller dist => more likely same).
        if within.size and between.size:
            try:
                auc = float(roc_auc_score(same, -dist))
                edge_distance_auc = auc
            except Exception:
                edge_distance_auc = None

    coords_sil: float | None = None
    coords_path = data_dir / "coords.npy"
    if coords_path.is_file():
        coords = np.load(coords_path).astype(np.float32)
        coords_sil = _safe_silhouette(coords, y)

    x_sil = _safe_silhouette(x, y)

    return DatasetSignal(
        dataset=dataset,
        data_dir=str(data_dir),
        category=category,
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_clusters=n_clusters,
        feature_dim=int(x.shape[1]),
        edge_dim=int(edge_attr.shape[1]),
        homophily=homophily,
        edge_distance_auc=edge_distance_auc,
        edge_distance_within_mean=within_mean,
        edge_distance_between_mean=between_mean,
        edge_distance_effect_size_d=effect_d,
        coords_silhouette=coords_sil,
        x_silhouette=x_sil,
    )


def _to_md(signals: list[DatasetSignal], suite_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# Synthetic Suite Signal Analysis")
    lines.append("")
    lines.append(f"- suite_dir: `{suite_dir}`")
    lines.append(f"- datasets: `{len(signals)}`")
    lines.append("")
    lines.append("该表用于解释“为什么谱聚类 baseline 很强”：若图的同配性（homophily）很高，且 edge_attr[:,0] 的距离特征对同簇/异簇边有很强可分性（AUC 高、effect size 大），谱聚类很容易接近最优。")
    lines.append("")
    lines.append("| dataset | category | N | E | homophily | dist_auc | within_mean | between_mean | effect_d | coords_sil | x_sil |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def f(v: float | None) -> str:
        if v is None or not np.isfinite(v):
            return "n/a"
        return f"{v:.4f}"

    for s in signals:
        lines.append(
            "| "
            + " | ".join(
                [
                    s.dataset,
                    s.category,
                    str(s.n_nodes),
                    str(s.n_edges),
                    f(s.homophily),
                    f(s.edge_distance_auc),
                    f(s.edge_distance_within_mean),
                    f(s.edge_distance_between_mean),
                    f(s.edge_distance_effect_size_d),
                    f(s.coords_silhouette),
                    f(s.x_silhouette),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="", help="Optional comma-separated dataset subdir names.")
    parser.add_argument("--out_json", type=str, default="suite_signal.json")
    parser.add_argument("--out_md", type=str, default="", help="Optional output markdown path.")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    datasets = [p.strip() for p in args.datasets.split(",") if p.strip()] if args.datasets.strip() else None

    dirs = _dataset_dirs(suite_dir, datasets)
    if not dirs:
        raise SystemExit(f"No datasets found under: {suite_dir}")

    signals = [analyze_dataset(d) for d in dirs]

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in signals], f, indent=2)
    print(f"Wrote JSON: {out_json.resolve()}")

    if args.out_md.strip():
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_to_md(signals, suite_dir=suite_dir), encoding="utf-8")
        print(f"Wrote Markdown: {out_md.resolve()}")


if __name__ == "__main__":
    main()

