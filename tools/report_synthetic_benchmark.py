#!/usr/bin/env python3
"""
Create a Markdown report from `tools/benchmark_synthetic_suite.py` aggregate.json.

Example:
  python tools/report_synthetic_benchmark.py \
    --aggregate_json /tmp/sdcn_dlaa_synth_results/aggregate.json \
    --out_md reports/synthetic_benchmark_report_zh.md
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Run:
    dataset: str
    approach: str  # "baseline" | "model"
    name: str
    seed: int
    metrics: dict[str, float]
    cluster_distribution: dict[str, int] | dict[int, int]
    data_dir: str


def _as_int_dict(d: dict[str, Any] | dict[int, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for k, v in d.items():
        out[int(k)] = int(v)
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _collapse_flag(dist: dict[int, int], n_nodes: int, n_clusters: int) -> bool:
    if n_nodes <= 0:
        return False
    counts = np.asarray(list(dist.values()), dtype=np.int64)
    if counts.size == 0:
        return True
    effective_k = int((counts > 0).sum())
    max_frac = float(counts.max() / max(n_nodes, 1))
    # Heuristic: either too few effective clusters, or one cluster dominates.
    return effective_k < n_clusters or max_frac >= 0.90


def _load_dataset_info(data_dir: str) -> dict[str, Any]:
    info_path = Path(data_dir) / "data_info.json"
    if not info_path.is_file():
        return {}
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _fmt(m: float, s: float) -> str:
    if not np.isfinite(m):
        return "n/a"
    if s <= 0:
        return f"{m:.4f}"
    return f"{m:.4f} ± {s:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate_json", type=str, required=True)
    parser.add_argument("--out_md", type=str, required=True)
    parser.add_argument("--title", type=str, default="Synthetic Benchmark Report (SDCN_DLAA)")
    parser.add_argument("--run_config", type=str, default="", help="Optional free-form run configuration string to embed in the report.")
    args = parser.parse_args()

    aggregate_path = Path(args.aggregate_json).resolve()
    with open(aggregate_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    runs: list[Run] = []
    for r in raw:
        runs.append(
            Run(
                dataset=str(r["dataset"]),
                approach=str(r["approach"]),
                name=str(r["name"]),
                seed=int(r["seed"]),
                metrics={k: float(v) for k, v in r["metrics"].items()},
                cluster_distribution=r.get("cluster_distribution", {}),
                data_dir=str(r["data_dir"]),
            )
        )

    # Dataset metadata
    dataset_info: dict[str, dict[str, Any]] = {}
    for run in runs:
        dataset_info.setdefault(run.dataset, _load_dataset_info(run.data_dir))

    # Group runs
    grouped: dict[tuple[str, str, str], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.dataset, run.approach, run.name), []).append(run)

    # Render report
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append(f"- Aggregate: `{aggregate_path}`")
    lines.append(f"- Total runs: `{len(runs)}`")
    if args.run_config.strip():
        lines.append(f"- Run config: {args.run_config.strip()}")
    lines.append("")
    lines.append("说明：这是基于 `tools/generate_synthetic_suite.py` 生成的多组合成数据集，")
    lines.append("对比了 SDCN_DLAA 的 `SpatialConv` 版本（v2/v3）与传统聚类 baseline（KMeans、谱聚类）。")
    lines.append("")

    # Dataset summary
    lines.append("## 数据集一览")
    lines.append("")
    lines.append("| dataset | category | N | edge_dim | knn_k | note |")
    lines.append("|---|---|---:|---:|---:|---|")
    for dataset in sorted({r.dataset for r in runs}):
        info = dataset_info.get(dataset, {}) or {}
        category = str(info.get("category", "n/a"))
        n_nodes = int(info.get("num_nodes", 0) or 0)
        edge_dim = int(info.get("edge_dim", 0) or 0)
        knn_k = int(info.get("graph_knn_k", 0) or 0)
        note = ""
        extra = info.get("extra")
        if isinstance(extra, dict) and "note" in extra:
            note = str(extra.get("note", ""))
        lines.append(f"| {dataset} | {category} | {n_nodes} | {edge_dim} | {knn_k} | {note} |")
    lines.append("")

    # Per-dataset results
    lines.append("## 结果汇总（mean ± std over seeds）")
    lines.append("")

    for dataset in sorted({r.dataset for r in runs}):
        info = dataset_info.get(dataset, {}) or {}
        n_nodes = int(info.get("num_nodes", 0) or 0)
        n_clusters = int(info.get("n_clusters", 0) or 0)
        category = str(info.get("category", "n/a"))

        lines.append(f"### {dataset} ({category})")
        lines.append("")
        lines.append("| approach | name | acc | nmi | ari | f1 | collapse_rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")

        # Stable ordering: baselines first, then models.
        keys = [k for k in grouped.keys() if k[0] == dataset]
        keys.sort(key=lambda k: (0 if k[1] == "baseline" else 1, k[2]))

        for _, approach, name in keys:
            rs = grouped[(dataset, approach, name)]
            acc_m, acc_s = _mean_std([r.metrics["acc"] for r in rs])
            nmi_m, nmi_s = _mean_std([r.metrics["nmi"] for r in rs])
            ari_m, ari_s = _mean_std([r.metrics["ari"] for r in rs])
            f1_m, f1_s = _mean_std([r.metrics["f1"] for r in rs])

            collapses = 0
            for r in rs:
                dist = _as_int_dict(r.cluster_distribution if isinstance(r.cluster_distribution, dict) else {})
                if _collapse_flag(dist, n_nodes=n_nodes, n_clusters=n_clusters):
                    collapses += 1
            collapse_rate = collapses / max(len(rs), 1)

            lines.append(
                f"| {approach} | {name} | {_fmt(acc_m, acc_s)} | {_fmt(nmi_m, nmi_s)} | {_fmt(ari_m, ari_s)} | {_fmt(f1_m, f1_s)} | {collapse_rate:.2f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
