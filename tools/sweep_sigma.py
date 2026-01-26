#!/usr/bin/env python3
"""
Sweep SDCN fusion coefficient (sigma) to diagnose whether AE↔GNN mixing hurts performance.

This runs `tools/test_conceptual_data.py` in isolated subprocesses while setting:
- SDCN_SIGMA
- SDCN_SEED
- SDCN_EPOCHS
- SPATIALCONV_VARIANT

Example:
  python tools/sweep_sigma.py \
    --data_dir /tmp/sdcn_dlaa_suite_seed0/rich_edge_profiles \
    --out_dir /tmp/sigma_sweep_rich_edge_profiles \
    --sigmas 0,0.25,0.5,0.75,1 \
    --seeds 0,1,2 \
    --epochs 30 \
    --variants v2edge_single_layer,v3edge_cross_layers \
    --heads 1
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


def _parse_float_list(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def _parse_str_list(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip() != ""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/tmp/sdcn_dlaa_sigma_sweep")
    parser.add_argument("--sigmas", type=str, default="0,0.5,1")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--variants", type=str, default="v2edge_single_layer,v3edge_cross_layers")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--max_edges_per_node", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = _parse_float_list(args.sigmas)
    seeds = _parse_int_list(args.seeds)
    variants = _parse_str_list(args.variants)

    all_runs: list[dict] = []

    for variant in variants:
        for sigma in sigmas:
            for seed in seeds:
                run_dir = out_dir / variant / f"sigma_{sigma:g}" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                env = os.environ.copy()
                env["SPATIALCONV_VARIANT"] = variant
                env["SDCN_SEED"] = str(seed)
                env["SDCN_EPOCHS"] = str(args.epochs)
                env["SDCN_SIGMA"] = str(sigma)

                cmd = [
                    sys.executable,
                    "-B",
                    str(TEST_SCRIPT),
                    "--data_dir",
                    str(data_dir),
                    "--heads",
                    str(args.heads),
                    "--max_edges_per_node",
                    str(args.max_edges_per_node),
                    "--summary_json",
                    "summary.json",
                ]
                if args.cpu:
                    cmd.append("--cpu")

                log_path = run_dir / "run.log"
                with open(log_path, "w", encoding="utf-8") as log_f:
                    log_f.write(f"$ {' '.join(cmd)}\n")
                    log_f.write(f"SPATIALCONV_VARIANT={variant} SDCN_SEED={seed} SDCN_EPOCHS={args.epochs} SDCN_SIGMA={sigma}\n\n")
                    log_f.flush()
                    subprocess.run(cmd, cwd=str(run_dir), env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)

                summary_path = run_dir / "summary.json"
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)

                summary["variant"] = variant
                summary["seed"] = seed
                summary["sigma"] = sigma
                all_runs.append(summary)

    aggregate_path = out_dir / "aggregate.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    print(f"Saved aggregate results to: {aggregate_path}")
    print("variant, sigma, seed, acc, nmi, ari, f1, cluster_distribution")
    for r in all_runs:
        m = r["metrics"]
        print(f"{r['variant']}, {r['sigma']}, {r['seed']}, {m['acc']:.4f}, {m['nmi']:.4f}, {m['ari']:.4f}, {m['f1']:.4f}, {r['cluster_distribution']}")


if __name__ == "__main__":
    main()
