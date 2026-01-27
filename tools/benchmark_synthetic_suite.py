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


def _load_dataset_info(data_dir: Path) -> dict:
    info_path = data_dir / "data_info.json"
    if not info_path.is_file():
        return {}
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _auto_edge_message(dataset_name: str, info: dict, policy: str) -> bool | None:
    policy = (policy or "").strip().lower()
    if policy in {"", "none"}:
        return None
    if policy in {"on", "true", "1", "yes"}:
        return True
    if policy in {"off", "false", "0", "no"}:
        return False
    if policy not in {"auto"}:
        raise SystemExit(f"Unknown --edge_message_policy={policy!r}. Use one of: auto, on, off.")

    category = str(info.get("category", "")).strip().lower()
    name = (dataset_name or "").strip().lower()

    # Heuristic:
    # - rich_edge* datasets: edge message helps inject edge semantics.
    # - distance_1d datasets: keep it off by default to avoid unnecessary perturbations.
    if category.startswith("distance") or name.startswith("dist_"):
        return False
    if category.startswith("rich_edge") or name.startswith("rich_") or "rich" in name:
        return True
    return None


def _auto_edge_attr_norm(dataset_name: str, info: dict, profiles_norm: str) -> str | None:
    category = str(info.get("category", "")).strip().lower()
    name = (dataset_name or "").strip().lower()
    if "profiles" in name or "profiles" in category:
        return profiles_norm
    return None


def _is_rich_edge_dataset(dataset_name: str, info: dict) -> bool:
    category = str(info.get("category", "")).strip().lower()
    name = (dataset_name or "").strip().lower()
    return category.startswith("rich_edge") or name.startswith("rich_") or "rich" in name


def _is_distance_dataset(dataset_name: str, info: dict) -> bool:
    category = str(info.get("category", "")).strip().lower()
    name = (dataset_name or "").strip().lower()
    return category.startswith("distance") or name.startswith("dist_")


def _auto_q_source(dataset_name: str, info: dict) -> str:
    # Heuristic:
    # - rich_edge tasks: use graph embedding (h4) to align self-training target with edge-driven signal.
    # - distance_1d tasks: for binary clustering (K=2) z tends to be more stable; otherwise use h4.
    if _is_distance_dataset(dataset_name, info):
        k = int(info.get("n_clusters", 0) or 0)
        if k <= 2:
            return "z"
        return "h4"
    return "h4"


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
    parser.add_argument("--skip_baselines", action="store_true", help="Skip running classic baselines (models only).")
    parser.add_argument(
        "--recommended_h4",
        action="store_true",
        help="Recommended combo: set SDCN_Q_SOURCE=h4 and apply per-dataset edge_message/norm heuristics.",
    )
    parser.add_argument(
        "--recommended_auto",
        action="store_true",
        help="Recommended combo: distance datasets use SDCN_Q_SOURCE=z; rich_edge datasets use SDCN_Q_SOURCE=h4; "
        "also apply per-dataset edge_message/norm heuristics.",
    )
    parser.add_argument(
        "--edge_message_policy",
        type=str,
        default="auto",
        help="When --recommended_h4: edge message policy: auto|on|off (implemented via SDCN_EDGE_MESSAGE).",
    )
    parser.add_argument(
        "--profiles_edge_attr_norm",
        type=str,
        default="zscore_clip",
        help="When --recommended_h4: edge_attr normalization to use for rich_edge_profiles.",
    )
    parser.add_argument(
        "--edge_attr_clip",
        type=float,
        default=5.0,
        help="When passing --edge_attr_norm zscore_clip, use this clip threshold.",
    )
    parser.add_argument(
        "--strategy_rich_only",
        action="store_true",
        help="If set, keep stability strategy env vars only for rich_edge datasets (clears them for distance_1d datasets).",
    )
    args = parser.parse_args()
    if args.recommended_h4 and args.recommended_auto:
        raise SystemExit("Use only one of: --recommended_h4 or --recommended_auto")

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
        info = _load_dataset_info(data_dir)
        is_rich_edge = _is_rich_edge_dataset(dataset_name, info)
        dataset_out = out_dir / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)

        # Baselines
        if not args.skip_baselines:
            for seed in seeds:
                run_dir = dataset_out / "baselines" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    "-B",
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
        recommended = bool(args.recommended_h4 or args.recommended_auto)
        edge_message = _auto_edge_message(dataset_name, info, args.edge_message_policy) if recommended else None
        edge_attr_norm = (
            _auto_edge_attr_norm(dataset_name, info, profiles_norm=args.profiles_edge_attr_norm) if recommended else None
        )
        q_source = None
        if args.recommended_h4:
            q_source = "h4"
        elif args.recommended_auto:
            q_source = _auto_q_source(dataset_name, info)

        for variant in variants:
            for seed in seeds:
                run_dir = dataset_out / variant / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                env = os.environ.copy()
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env["SPATIALCONV_VARIANT"] = variant
                env["SDCN_SEED"] = str(seed)
                env["SDCN_EPOCHS"] = str(args.epochs)
                if q_source:
                    env["SDCN_Q_SOURCE"] = q_source
                edge_message_variant = edge_message
                # For v3, edge_attr already carries learned edge embeddings; injecting an extra edge->node message
                # often over-smooths and increases collapse. Keep it off by default under the auto policy.
                if (
                    recommended
                    and (args.edge_message_policy or "").strip().lower() == "auto"
                    and variant.strip().lower() == "v3edge_cross_layers"
                ):
                    edge_message_variant = False
                if recommended and edge_message_variant is not None:
                    env["SDCN_EDGE_MESSAGE"] = "1" if edge_message_variant else "0"
                if args.strategy_rich_only and not is_rich_edge:
                    for k in ["SDCN_P_SMOOTHING", "SDCN_CE_WARMUP_EPOCHS", "SDCN_PRED_MI_WEIGHT", "SDCN_Q_MI_WEIGHT"]:
                        env.pop(k, None)

                cmd = [
                    sys.executable,
                    "-B",
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
                if edge_attr_norm:
                    cmd.extend(["--edge_attr_norm", str(edge_attr_norm), "--edge_attr_clip", str(args.edge_attr_clip)])
                if args.cpu:
                    cmd.append("--cpu")

                log_path = run_dir / "run.log"
                with open(log_path, "w", encoding="utf-8") as log_f:
                    log_f.write(f"$ {' '.join(cmd)}\n")
                    log_f.write(f"SPATIALCONV_VARIANT={variant} SDCN_SEED={seed} SDCN_EPOCHS={args.epochs}\n")
                    if recommended:
                        log_f.write(
                            f"SDCN_Q_SOURCE={q_source or ''} SDCN_EDGE_MESSAGE={(env.get('SDCN_EDGE_MESSAGE',''))} "
                            f"edge_attr_norm={(edge_attr_norm or '')}\n"
                        )
                    log_f.write("\n")
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
