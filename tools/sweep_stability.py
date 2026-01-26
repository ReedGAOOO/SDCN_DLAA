#!/usr/bin/env python3
"""
Sweep hyperparameters on a single conceptual dataset to diagnose stability/collapse.

This runs `tools/test_conceptual_data.py` in subprocesses and collects:
- final metrics + cluster_distribution
- per-epoch trace-derived collapse epochs/rates (requires --trace_jsonl)

Example:
  python tools/sweep_stability.py \
    --data_dir /tmp/suite_ablation/rich_edge_profiles \
    --out_dir /tmp/sweep_rich_edge_profiles \
    --variants v2edge_single_layer,v3edge_cross_layers \
    --seeds 0,1,2 \
    --epochs 30,60 \
    --lrs 1e-3,5e-4 \
    --dropouts 0.0,0.2 \
    --heads 1,4 \
    --n_z 10,20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np


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


def _parse_enc_dims_list(value: str) -> list[str]:
    """
    Semicolon-separated triples like: "256,256,256;500,500,512".
    Returns a list of raw strings (to pass into SDCN_ENC_DIMS).
    """
    value = (value or "").strip()
    if not value:
        return [""]
    items = [v.strip() for v in value.split(";") if v.strip() != ""]
    for it in items:
        parts = [p.strip() for p in it.split(",") if p.strip() != ""]
        if len(parts) != 3 or any(not p.lstrip("+-").isdigit() for p in parts):
            raise SystemExit(f"Invalid --enc_dims_list item: {it!r}. Expected like '256,256,512'.")
    return items


def _as_int_dict(d: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def _collapse_flag(dist: dict[int, int], n_nodes: int, n_clusters: int) -> bool:
    if n_nodes <= 0:
        return False
    counts = np.asarray(list(dist.values()), dtype=np.int64)
    if counts.size == 0:
        return True
    effective_k = int((counts > 0).sum())
    max_frac = float(counts.max() / max(n_nodes, 1))
    return effective_k < int(n_clusters) or max_frac >= 0.90


def _analyze_trace(trace_path: Path, n_nodes: int, n_clusters: int) -> dict:
    if not trace_path.is_file():
        return {}

    collapse_epoch_q: int | None = None
    collapse_epoch_pred: int | None = None
    collapse_epoch_p: int | None = None

    total = 0
    collapsed_q = 0
    collapsed_pred = 0
    collapsed_p = 0

    ent_q: list[float] = []
    ent_pred: list[float] = []
    ent_p: list[float] = []
    kl_p_q: list[float] = []
    kl_p_pred: list[float] = []

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            cq = bool(rec.get("collapse_q", False))
            cp = bool(rec.get("collapse_pred", False))
            ctp = bool(rec.get("collapse_p", False))

            if cq:
                collapsed_q += 1
                if collapse_epoch_q is None:
                    collapse_epoch_q = int(rec.get("epoch", 0))
            if cp:
                collapsed_pred += 1
                if collapse_epoch_pred is None:
                    collapse_epoch_pred = int(rec.get("epoch", 0))
            if ctp:
                collapsed_p += 1
                if collapse_epoch_p is None:
                    collapse_epoch_p = int(rec.get("epoch", 0))

            q = rec.get("q") or {}
            pred = rec.get("pred") or {}
            p = rec.get("p") or {}
            if "entropy" in q:
                ent_q.append(float(q["entropy"]))
            if "entropy" in pred:
                ent_pred.append(float(pred["entropy"]))
            if "entropy" in p:
                ent_p.append(float(p["entropy"]))

            if "kl_p_q" in rec:
                kl_p_q.append(float(rec["kl_p_q"]))
            if "kl_p_pred" in rec:
                kl_p_pred.append(float(rec["kl_p_pred"]))

    if total <= 0:
        return {}

    # Sanity check: compute final collapse flags from last hard distributions if present.
    # (Trace already stores flags, but this helps detect schema mismatch.)
    final_collapse = {}
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            last = None
            for line in f:
                if line.strip():
                    last = json.loads(line)
            if last:
                dist_pred = _as_int_dict(last.get("hard_pred") or {})
                final_collapse["collapse_final_pred"] = bool(_collapse_flag(dist_pred, n_nodes=n_nodes, n_clusters=n_clusters))
    except Exception:
        pass

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    return {
        "epochs_logged": int(total),
        "collapse_epoch_q": collapse_epoch_q,
        "collapse_epoch_pred": collapse_epoch_pred,
        "collapse_epoch_p": collapse_epoch_p,
        "collapse_rate_q": float(collapsed_q / total),
        "collapse_rate_pred": float(collapsed_pred / total),
        "collapse_rate_p": float(collapsed_p / total),
        "q_entropy_mean": _mean(ent_q),
        "pred_entropy_mean": _mean(ent_pred),
        "p_entropy_mean": _mean(ent_p),
        "kl_p_q_mean": _mean(kl_p_q),
        "kl_p_pred_mean": _mean(kl_p_pred),
        **final_collapse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/tmp/sdcn_dlaa_stability_sweep")
    parser.add_argument("--variants", type=str, default="v2edge_single_layer,v3edge_cross_layers")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=str, default="60", help="Comma-separated epoch counts (sets SDCN_EPOCHS).")
    parser.add_argument("--lrs", type=str, default="1e-3")
    parser.add_argument("--dropouts", type=str, default="0.2")
    parser.add_argument("--heads", type=str, default="1")
    parser.add_argument("--n_z", type=str, default="10")
    parser.add_argument("--sigmas", type=str, default="", help="Optional comma-separated SDCN_SIGMA values (env override).")
    parser.add_argument("--enc_dims_list", type=str, default="", help="Optional AE encoder dims list: '256,256,256;500,500,512'.")
    parser.add_argument("--max_edges_per_node", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _parse_str_list(args.variants)
    seeds = _parse_int_list(args.seeds)
    epochs_list = _parse_int_list(args.epochs)
    lrs = _parse_float_list(args.lrs)
    dropouts = _parse_float_list(args.dropouts)
    heads_list = _parse_int_list(args.heads)
    n_z_list = _parse_int_list(args.n_z)
    sigmas = _parse_float_list(args.sigmas) if args.sigmas.strip() else [None]
    enc_dims_list = _parse_enc_dims_list(args.enc_dims_list)

    all_runs: list[dict] = []

    for variant, seed, epochs, lr, dropout, heads, n_z, sigma, enc_dims in product(
        variants, seeds, epochs_list, lrs, dropouts, heads_list, n_z_list, sigmas, enc_dims_list
    ):
        run_name = f"seed{seed}_ep{epochs}_lr{lr:g}_do{dropout:g}_h{heads}_nz{n_z}"
        if sigma is not None:
            run_name += f"_sigma{sigma:g}"
        if enc_dims:
            run_name += f"_enc{enc_dims.replace(',', '-')}"

        run_dir = out_dir / data_dir.name / variant / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["SPATIALCONV_VARIANT"] = variant
        env["SDCN_SEED"] = str(seed)
        env["SDCN_EPOCHS"] = str(epochs)
        if sigma is not None:
            env["SDCN_SIGMA"] = str(sigma)
        else:
            env.pop("SDCN_SIGMA", None)
        if enc_dims:
            env["SDCN_ENC_DIMS"] = enc_dims
        else:
            env.pop("SDCN_ENC_DIMS", None)

        cmd = [
            sys.executable,
            "-B",
            str(TEST_SCRIPT),
            "--data_dir",
            str(data_dir),
            "--lr",
            str(lr),
            "--dropout",
            str(dropout),
            "--heads",
            str(heads),
            "--n_z",
            str(n_z),
            "--max_edges_per_node",
            str(args.max_edges_per_node),
            "--summary_json",
            "summary.json",
            "--trace_jsonl",
            "trace.jsonl",
        ]
        if args.cpu:
            cmd.append("--cpu")

        log_path = run_dir / "run.log"
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(f"$ {' '.join(cmd)}\n")
            log_f.write(
                f"SPATIALCONV_VARIANT={variant} SDCN_SEED={seed} SDCN_EPOCHS={epochs} "
                f"SDCN_SIGMA={(sigma if sigma is not None else '')} SDCN_ENC_DIMS={enc_dims}\n\n"
            )
            log_f.flush()
            subprocess.run(cmd, cwd=str(run_dir), env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)

        summary_path = run_dir / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        n_nodes = int(summary.get("n_nodes", 0) or 0)
        n_clusters = int(summary.get("n_clusters", 0) or 0)
        cluster_dist = summary.get("cluster_distribution") or {}
        collapse_final = bool(_collapse_flag(_as_int_dict(cluster_dist), n_nodes=n_nodes, n_clusters=n_clusters))

        trace_path = run_dir / "trace.jsonl"
        trace_stats = _analyze_trace(trace_path, n_nodes=n_nodes, n_clusters=n_clusters)

        all_runs.append(
            {
                "dataset": data_dir.name,
                "data_dir": str(data_dir),
                "variant": variant,
                "seed": seed,
                "epochs": epochs,
                "lr": float(lr),
                "dropout": float(dropout),
                "heads": int(heads),
                "n_z": int(n_z),
                "sigma": None if sigma is None else float(sigma),
                "enc_dims": enc_dims,
                "metrics": summary.get("metrics"),
                "cluster_distribution": cluster_dist,
                "collapse_final": collapse_final,
                "trace_jsonl": str(trace_path),
                "trace_stats": trace_stats,
                "run_dir": str(run_dir),
            }
        )

        m = summary.get("metrics") or {}
        acc = m.get("acc")
        nmi = m.get("nmi")
        ari = m.get("ari")
        f1 = m.get("f1")
        print(
            f"{data_dir.name}, {variant}, seed={seed}, ep={epochs}, lr={lr:g}, do={dropout:g}, heads={heads}, nz={n_z}, "
            f"sigma={(sigma if sigma is not None else 'default')}, enc_dims={(enc_dims if enc_dims else 'default')}: "
            f"acc={acc:.4f} nmi={nmi:.4f} ari={ari:.4f} f1={f1:.4f} collapse_final={collapse_final}"
        )

    aggregate_path = out_dir / "aggregate.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
    print(f"Saved aggregate results to: {aggregate_path}")


if __name__ == "__main__":
    main()
