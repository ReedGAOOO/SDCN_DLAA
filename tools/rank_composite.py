#!/usr/bin/env python3
"""
Rank SDCN_DLAA variant runs with a composite score.

This script is designed to consume `aggregate.json` produced by:
- `tools/sweep_stability.py` (preferred; includes run_dir + collapse + trace stats)

It can also read aggregate lists from other scripts as long as they contain:
- dataset name
- variant name
- per-run metrics dict (at least graph metrics, optionally acc/nmi/ari/f1)

Example:
  conda run -n gnn python tools/rank_composite.py \
    --aggregate_json /tmp/sweep_all_variants_edge_edge_denoise_nonknn/aggregate.json \
    --profile unlabeled_city \
    --out_md /tmp/rank_edge_edge_denoise_nonknn.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass(frozen=True)
class MetricSpec:
    higher_is_better: bool
    weight: float


PROFILES: dict[str, dict[str, MetricSpec]] = {
    # For unlabeled city-network clustering: prioritize stability + graph partition quality.
    "unlabeled_city": {
        "collapse_rate": MetricSpec(higher_is_better=False, weight=0.20),
        "consistency_nmi": MetricSpec(higher_is_better=True, weight=0.15),
        # Convergence / branch alignment (trace-derived, no ground truth needed).
        "kl_p_pred_last_mean": MetricSpec(higher_is_better=False, weight=0.07),
        "kl_p_pred_mean": MetricSpec(higher_is_better=False, weight=0.03),
        "align_nmi_q_pred_last_mean": MetricSpec(higher_is_better=True, weight=0.03),
        "align_nmi_q_pred_mean": MetricSpec(higher_is_better=True, weight=0.02),
        "modularity": MetricSpec(higher_is_better=True, weight=0.125),
        "conductance_mean": MetricSpec(higher_is_better=False, weight=0.125),
        "ncut_mean": MetricSpec(higher_is_better=False, weight=0.04),
        "within_edge_ratio": MetricSpec(higher_is_better=True, weight=0.05),
        "largest_cc_ratio_mean": MetricSpec(higher_is_better=True, weight=0.05),
        "cluster_entropy_norm": MetricSpec(higher_is_better=True, weight=0.05),
        # Optional embedding-space internal metrics (only present when computed by the runner).
        "silhouette": MetricSpec(higher_is_better=True, weight=0.03),
        "davies_bouldin": MetricSpec(higher_is_better=False, weight=0.02),
        "calinski_harabasz": MetricSpec(higher_is_better=True, weight=0.01),
    },
    # For labeled dev/benchmark: include acc/nmi/ari/f1 while still penalizing collapse/instability.
    "labeled_dev": {
        "acc": MetricSpec(higher_is_better=True, weight=0.25),
        "nmi": MetricSpec(higher_is_better=True, weight=0.20),
        "ari": MetricSpec(higher_is_better=True, weight=0.15),
        "f1": MetricSpec(higher_is_better=True, weight=0.10),
        "collapse_rate": MetricSpec(higher_is_better=False, weight=0.10),
        "consistency_nmi": MetricSpec(higher_is_better=True, weight=0.03),
        # Convergence / branch alignment (trace-derived, no ground truth needed).
        "kl_p_pred_last_mean": MetricSpec(higher_is_better=False, weight=0.03),
        "kl_p_pred_mean": MetricSpec(higher_is_better=False, weight=0.02),
        "align_nmi_q_pred_last_mean": MetricSpec(higher_is_better=True, weight=0.02),
        "align_nmi_q_pred_mean": MetricSpec(higher_is_better=True, weight=0.01),
        "modularity": MetricSpec(higher_is_better=True, weight=0.03),
        "conductance_mean": MetricSpec(higher_is_better=False, weight=0.03),
        "ncut_mean": MetricSpec(higher_is_better=False, weight=0.02),
        "within_edge_ratio": MetricSpec(higher_is_better=True, weight=0.01),
    },
}


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v == v and math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _minmax_norm(values: dict[str, float], *, higher_is_better: bool) -> dict[str, float]:
    if not values:
        return {}
    xs = list(values.values())
    lo = min(xs)
    hi = max(xs)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return {k: 0.5 for k in values}
    out: dict[str, float] = {}
    for k, v in values.items():
        t = (float(v) - lo) / (hi - lo)
        if not higher_is_better:
            t = 1.0 - t
        out[k] = float(max(0.0, min(1.0, t)))
    return out


def _read_final_clusters(run_dir: Path) -> np.ndarray | None:
    """
    Read `sdcn_dlaa_final_cluster_results.csv` as an integer array of shape [N].
    Returns None if missing/unreadable.
    """
    path = run_dir / "sdcn_dlaa_final_cluster_results.csv"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pairs: list[tuple[int, int]] = []
            for row in reader:
                nid = int(row.get("NodeID", "0"))
                cid = int(row.get("ClusterID", "0"))
                pairs.append((nid, cid))
        if not pairs:
            return None
        pairs.sort(key=lambda t: t[0])
        n = int(pairs[-1][0]) + 1
        arr = np.zeros((n,), dtype=np.int64)
        for nid, cid in pairs:
            if 0 <= nid < n:
                arr[nid] = cid
        return arr
    except Exception:
        return None


def _pairwise_consistency(labelings: list[np.ndarray]) -> dict[str, float]:
    if len(labelings) < 2:
        return {}
    nmi_vals: list[float] = []
    ari_vals: list[float] = []
    for a, b in combinations(labelings, 2):
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            continue
        try:
            nmi_vals.append(float(normalized_mutual_info_score(a, b, average_method="arithmetic")))
        except Exception:
            pass
        try:
            ari_vals.append(float(adjusted_rand_score(a, b)))
        except Exception:
            pass
    out: dict[str, float] = {}
    if nmi_vals:
        out["consistency_nmi"] = float(sum(nmi_vals) / len(nmi_vals))
    if ari_vals:
        out["consistency_ari"] = float(sum(ari_vals) / len(ari_vals))
    return out


def _get_variant_name(run: dict[str, Any]) -> str | None:
    v = run.get("variant")
    if isinstance(v, str) and v.strip():
        return v.strip()
    name = run.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _get_dataset_name(run: dict[str, Any]) -> str:
    d = run.get("dataset")
    if isinstance(d, str) and d.strip():
        return d.strip()
    return "dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate_json", type=str, required=True)
    parser.add_argument("--profile", type=str, default="unlabeled_city", choices=sorted(PROFILES.keys()))
    parser.add_argument("--datasets", type=str, default="", help="Optional comma-separated dataset names to include.")
    parser.add_argument("--gate", action="store_true", help="If set, filter out variants failing the hard gates below.")
    parser.add_argument("--max_collapse_rate", type=float, default=None, help="Hard gate: collapse_rate <= value.")
    parser.add_argument("--min_consistency_nmi", type=float, default=None, help="Hard gate: consistency_nmi >= value.")
    parser.add_argument("--max_max_cluster_frac", type=float, default=None, help="Hard gate: max_cluster_frac <= value.")
    parser.add_argument("--min_effective_k", type=float, default=None, help="Hard gate: effective_k >= value.")
    parser.add_argument("--max_kl_p_pred_mean", type=float, default=None, help="Hard gate: kl_p_pred_mean <= value.")
    parser.add_argument("--max_kl_p_pred_last_mean", type=float, default=None, help="Hard gate: kl_p_pred_last_mean <= value.")
    parser.add_argument("--min_align_nmi_q_pred_mean", type=float, default=None, help="Hard gate: align_nmi_q_pred_mean >= value.")
    parser.add_argument(
        "--min_align_nmi_q_pred_last_mean",
        type=float,
        default=None,
        help="Hard gate: align_nmi_q_pred_last_mean >= value.",
    )
    parser.add_argument("--out_md", type=str, default="", help="Optional output Markdown path.")
    parser.add_argument("--out_json", type=str, default="", help="Optional output JSON path.")
    args = parser.parse_args()

    agg_path = Path(args.aggregate_json).resolve()
    with open(agg_path, "r", encoding="utf-8") as f:
        runs = json.load(f)
    if not isinstance(runs, list):
        raise SystemExit(f"Expected a JSON list in {agg_path}, got: {type(runs).__name__}")

    dataset_filter = {d.strip() for d in (args.datasets or "").split(",") if d.strip()}

    # Group runs by (dataset, variant)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in runs:
        if not isinstance(r, dict):
            continue
        variant = _get_variant_name(r)
        if not variant:
            continue
        dataset = _get_dataset_name(r)
        if dataset_filter and dataset not in dataset_filter:
            continue
        groups.setdefault((dataset, variant), []).append(r)

    if not groups:
        raise SystemExit("No runs matched. Check --datasets filter and aggregate schema.")

    profile = PROFILES[str(args.profile)]
    metric_names = list(profile.keys())

    # Compute raw aggregated metrics for each group.
    rows: list[dict[str, Any]] = []
    for (dataset, variant), rs in sorted(groups.items()):
        collapse_rate = None
        if any("collapse_final" in r for r in rs):
            collapse_rate = float(sum(1 for r in rs if bool(r.get("collapse_final"))) / max(1, len(rs)))

        # Mean of scalar metrics.
        metrics_by_name: dict[str, list[float]] = {}
        for r in rs:
            m = r.get("metrics") or {}
            if isinstance(m, dict):
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        metrics_by_name.setdefault(str(k), []).append(float(v))
            ts = r.get("trace_stats") or {}
            if isinstance(ts, dict):
                for k, v in ts.items():
                    if isinstance(v, (int, float)):
                        metrics_by_name.setdefault(str(k), []).append(float(v))

        agg_metrics: dict[str, float] = {}
        for k, vs in metrics_by_name.items():
            mu = _mean(vs)
            if mu is not None:
                agg_metrics[k] = mu

        if collapse_rate is not None:
            agg_metrics["collapse_rate"] = float(collapse_rate)

        # Consistency across runs (if run_dir is available and cluster csv exists).
        labelings: list[np.ndarray] = []
        for r in rs:
            run_dir = r.get("run_dir")
            if isinstance(run_dir, str) and run_dir.strip():
                arr = _read_final_clusters(Path(run_dir))
                if arr is not None:
                    labelings.append(arr)
        agg_metrics.update(_pairwise_consistency(labelings))

        # Optional hard gates (fail-fast). Missing required metrics counts as fail.
        gate_reasons: list[str] = []
        def _need_le(name: str, threshold: float | None) -> None:
            if threshold is None:
                return
            v = agg_metrics.get(name)
            if v is None or not math.isfinite(float(v)):
                gate_reasons.append(f"missing:{name}")
            elif float(v) > float(threshold):
                gate_reasons.append(f"{name}>{threshold:g}")

        def _need_ge(name: str, threshold: float | None) -> None:
            if threshold is None:
                return
            v = agg_metrics.get(name)
            if v is None or not math.isfinite(float(v)):
                gate_reasons.append(f"missing:{name}")
            elif float(v) < float(threshold):
                gate_reasons.append(f"{name}<{threshold:g}")

        _need_le("collapse_rate", args.max_collapse_rate)
        _need_ge("consistency_nmi", args.min_consistency_nmi)
        _need_le("max_cluster_frac", args.max_max_cluster_frac)
        _need_ge("effective_k", args.min_effective_k)
        _need_le("kl_p_pred_mean", args.max_kl_p_pred_mean)
        _need_le("kl_p_pred_last_mean", args.max_kl_p_pred_last_mean)
        _need_ge("align_nmi_q_pred_mean", args.min_align_nmi_q_pred_mean)
        _need_ge("align_nmi_q_pred_last_mean", args.min_align_nmi_q_pred_last_mean)

        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_runs": int(len(rs)),
                "metrics": agg_metrics,
                "gate_pass": bool(len(gate_reasons) == 0),
                "gate_reasons": gate_reasons,
            }
        )

    # Normalize metrics per dataset for fair aggregation.
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    output_rows: list[dict[str, Any]] = []

    for dataset, ds_rows in sorted(by_dataset.items()):
        # Optional: filter rows by hard-gate status.
        if bool(args.gate):
            ds_rows = [r for r in ds_rows if bool(r.get("gate_pass"))]
            if not ds_rows:
                continue
        # Collect raw values per metric for min-max.
        raw_per_metric: dict[str, dict[str, float]] = {mn: {} for mn in metric_names}
        for row in ds_rows:
            v = str(row["variant"])
            m = row["metrics"]
            for mn in metric_names:
                if mn in m and isinstance(m[mn], (int, float)) and math.isfinite(float(m[mn])):
                    raw_per_metric[mn][v] = float(m[mn])

        norm_per_metric: dict[str, dict[str, float]] = {}
        for mn, vals in raw_per_metric.items():
            if not vals:
                continue
            norm_per_metric[mn] = _minmax_norm(vals, higher_is_better=bool(profile[mn].higher_is_better))

        # Composite scores
        scored: list[dict[str, Any]] = []
        for row in ds_rows:
            v = str(row["variant"])
            num = 0.0
            den = 0.0
            used: dict[str, float] = {}
            for mn in metric_names:
                w = float(profile[mn].weight)
                if mn not in norm_per_metric or v not in norm_per_metric[mn]:
                    continue
                s = float(norm_per_metric[mn][v])
                used[mn] = s
                num += w * s
                den += w
            score = float(num / den) if den > 1e-12 else float("nan")
            scored.append(
                {
                    **row,
                    "composite_score": score,
                    "norm_metrics": used,
                }
            )

        scored.sort(key=lambda r: (float(r["composite_score"]) if r["composite_score"] == r["composite_score"] else -1.0), reverse=True)
        output_rows.extend(scored)

    # Save JSON
    if str(args.out_json).strip():
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "aggregate_json": str(agg_path),
                    "profile": str(args.profile),
                    "rows": output_rows,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    # Print + optional markdown
    lines: list[str] = []
    lines.append(f"# Composite Ranking ({args.profile})")
    lines.append("")
    lines.append(f"- aggregate: `{agg_path}`")
    if dataset_filter:
        lines.append(f"- datasets: `{','.join(sorted(dataset_filter))}`")
    if bool(args.gate):
        lines.append("- hard_gate: enabled")
    lines.append("")
    lines.append("| dataset | rank | variant | composite | n_runs | collapse_rate | consistency_nmi | kl_p_pred_last_mean | align_nmi_q_pred_last_mean | modularity | conductance |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for dataset, ds_rows in sorted(by_dataset.items()):
        ranked = [r for r in output_rows if str(r["dataset"]) == dataset]
        for i, row in enumerate(ranked, 1):
            m = row.get("metrics") or {}
            fallback = {
                "kl_p_pred_last_mean": "kl_p_pred_mean",
                "align_nmi_q_pred_last_mean": "align_nmi_q_pred_mean",
            }
            def _fmt(key: str) -> str:
                v = m.get(key)
                if v is None and key in fallback:
                    v = m.get(fallback[key])
                if v is None:
                    return ""
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return ""

            lines.append(
                "| "
                + " | ".join(
                    [
                        str(dataset),
                        str(i),
                        str(row["variant"]),
                        f"{float(row['composite_score']):.4f}" if row["composite_score"] == row["composite_score"] else "",
                        str(row["n_runs"]),
                        _fmt("collapse_rate"),
                        _fmt("consistency_nmi"),
                        _fmt("kl_p_pred_last_mean"),
                        _fmt("align_nmi_q_pred_last_mean"),
                        _fmt("modularity"),
                        _fmt("conductance_mean"),
                    ]
                )
                + " |"
            )

    text = "\n".join(lines)
    print(text)

    if str(args.out_md).strip():
        out_md = Path(args.out_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
