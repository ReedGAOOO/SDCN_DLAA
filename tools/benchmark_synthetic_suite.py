#!/usr/bin/env python3
"""
Benchmark SDCN_DLAA SpatialConv variants (v2/v3) against classic baselines across a dataset suite.

This script expects a suite directory containing subdirectories created by
`tools/generate_synthetic_suite.py` (or any dataset that follows the same file layout).

It runs:
- Model variants in isolated subprocesses (to respect import-time `SPATIALCONV_VARIANT`).
- Baselines via `tools/run_baselines.py`.

Outputs under --out_dir:
- per-dataset/per-run logs and summary JSON files
- aggregate.json: flat list of all runs

Example:
  python tools/generate_synthetic_suite.py --output_root /tmp/sdcn_suite --seed 0
  python tools/benchmark_synthetic_suite.py --suite_dir /tmp/sdcn_suite --out_dir /tmp/sdcn_suite_results --epochs 30 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TEST_SCRIPT = REPO_ROOT / "tools" / "test_conceptual_data.py"
BASELINE_SCRIPT = REPO_ROOT / "tools" / "run_baselines.py"


def _parse_int_list(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _parse_str_list(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip() != ""]


def _dataset_dirs(suite_dir: Path, datasets: list[str] | None) -> list[Path]:
    if datasets:
        dirs: list[Path] = []
        for name in datasets:
            d = (suite_dir / name).resolve()
            if not d.is_dir():
                raise SystemExit(f"Dataset directory not found: {d}")
            dirs.append(d)
        return dirs

    dirs = [p for p in sorted(suite_dir.iterdir()) if p.is_dir()]
    # Filter to those that look like dataset dirs.
    needed = {"node_features.npy", "labels.npy", "binary_adj.npz", "edge_attr.npy"}
    out: list[Path] = []
    for d in dirs:
        if needed.issubset({p.name for p in d.iterdir()}):
            out.append(d)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/tmp/sdcn_dlaa_suite_bench")
    parser.add_argument("--datasets", type=str, default="", help="Optional comma-separated dataset subdir names.")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--max_edges_per_node", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--variants", type=str, default="v2edge_single_layer,v3edge_cross_layers")
    parser.add_argument("--baselines", type=str, default="kmeans_x,spectral_adj_binary,spectral_edge_distance")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_int_list(args.seeds)
    variants = _parse_str_list(args.variants)
    baselines = _parse_str_list(args.baselines)

    datasets = _parse_str_list(args.datasets) if args.datasets.strip() else None
    dataset_dirs = _dataset_dirs(suite_dir, datasets)
    if not dataset_dirs:
        raise SystemExit(f"No datasets found under: {suite_dir}")

    all_runs: list[dict] = []

    for data_dir in dataset_dirs:
        dataset_name = data_dir.name
        dataset_out = out_dir / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)

        # Baselines
        for seed in seeds:
            run_dir = dataset_out / "baselines" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(BASELINE_SCRIPT),
                "--data_dir",
                str(data_dir),
                "--seed",
                str(seed),
                "--methods",
                ",".join(baselines),
                "--summary_json",
                "summary_baselines.json",
            ]

            log_path = run_dir / "run.log"
            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(f"$ {' '.join(cmd)}\n\n")
                log_f.flush()
                subprocess.run(cmd, cwd=str(run_dir), stdout=log_f, stderr=subprocess.STDOUT, check=True)

            summary_path = run_dir / "summary_baselines.json"
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            for b in summary["baselines"]:
                all_runs.append(
                    {
                        "dataset": dataset_name,
                        "approach": "baseline",
                        "name": b["method"],
                        "seed": seed,
                        "metrics": b["metrics"],
                        "cluster_distribution": b["cluster_distribution"],
                        "data_dir": str(data_dir),
                    }
                )

        # Model variants
        for variant in variants:
            for seed in seeds:
                run_dir = dataset_out / variant / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                env = os.environ.copy()
                env["SPATIALCONV_VARIANT"] = variant
                env["SDCN_SEED"] = str(seed)
                env["SDCN_EPOCHS"] = str(args.epochs)

                cmd = [
                    sys.executable,
                    str(MODEL_TEST_SCRIPT),
                    "--data_dir",
                    str(data_dir),
                    "--heads",
                    str(args.heads),
                    "--max_edges_per_node",
                    str(args.max_edges_per_node),
                    "--summary_json",
                    "summary_model.json",
                ]
                if args.cpu:
                    cmd.append("--cpu")

                log_path = run_dir / "run.log"
                with open(log_path, "w", encoding="utf-8") as log_f:
                    log_f.write(f"$ {' '.join(cmd)}\n")
                    log_f.write(f"SPATIALCONV_VARIANT={variant} SDCN_SEED={seed} SDCN_EPOCHS={args.epochs}\n\n")
                    log_f.flush()
                    subprocess.run(cmd, cwd=str(run_dir), env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)

                summary_path = run_dir / "summary_model.json"
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)

                all_runs.append(
                    {
                        "dataset": dataset_name,
                        "approach": "model",
                        "name": variant,
                        "seed": seed,
                        "metrics": summary["metrics"],
                        "cluster_distribution": summary["cluster_distribution"],
                        "cuda": summary.get("cuda"),
                        "torch_device": summary.get("torch_device"),
                        "data_dir": str(data_dir),
                    }
                )

    aggregate_path = out_dir / "aggregate.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    print(f"Saved aggregate results to: {aggregate_path}")
    print("dataset, approach, name, seed, acc, nmi, ari, f1, cluster_distribution")
    for r in all_runs:
        m = r["metrics"]
        print(
            f"{r['dataset']}, {r['approach']}, {r['name']}, {r['seed']}, "
            f"{m['acc']:.4f}, {m['nmi']:.4f}, {m['ari']:.4f}, {m['f1']:.4f}, {r['cluster_distribution']}"
        )


if __name__ == "__main__":
    main()

