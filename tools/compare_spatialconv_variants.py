#!/usr/bin/env python3
"""
Benchmark clustering performance of SpatialConv variants on a labeled conceptual dataset.

This script runs each variant in an isolated subprocess (so import-time variant selection works),
stores logs + outputs under `--out_dir`, and prints a compact summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = REPO_ROOT / "tools" / "test_conceptual_data.py"


def _parse_int_list(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/tmp/sdcn_dlaa_variant_compare")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--variants", type=str, default="v1original,v2edge_single_layer,v3edge_cross_layers")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--max_edges_per_node", type=int, default=10)
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    variants = [v.strip() for v in args.variants.split(",") if v.strip() != ""]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs: list[dict] = []

    for variant in variants:
        for seed in seeds:
            run_dir = out_dir / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["SPATIALCONV_VARIANT"] = variant
            env["SDCN_SEED"] = str(seed)
            env["SDCN_EPOCHS"] = str(args.epochs)

            cmd = [
                sys.executable,
                "-B",
                str(TEST_SCRIPT),
                "--data_dir",
                os.path.abspath(args.data_dir),
                "--heads",
                str(args.heads),
                "--max_edges_per_node",
                str(args.max_edges_per_node),
                "--summary_json",
                "summary.json",
            ]

            log_path = run_dir / "run.log"
            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(f"$ {' '.join(cmd)}\n")
                log_f.write(f"SPATIALCONV_VARIANT={variant} SDCN_SEED={seed} SDCN_EPOCHS={args.epochs}\n\n")
                log_f.flush()
                subprocess.run(cmd, cwd=str(run_dir), env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)

            summary_path = run_dir / "summary.json"
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["variant"] = variant
            summary["seed"] = seed
            all_runs.append(summary)

    aggregate_path = out_dir / "aggregate.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    # Print a compact table.
    print(f"Saved aggregate results to: {aggregate_path}")
    print("variant, seed, acc, nmi, ari, f1, cluster_distribution")
    for r in all_runs:
        m = r["metrics"]
        dist = r["cluster_distribution"]
        print(f"{r['variant']}, {r['seed']}, {m['acc']:.4f}, {m['nmi']:.4f}, {m['ari']:.4f}, {m['f1']:.4f}, {dist}")


if __name__ == "__main__":
    main()
