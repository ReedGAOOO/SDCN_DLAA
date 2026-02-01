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

def _parse_optional_str_list(value: str) -> list[str | None]:
    value = (value or "").strip()
    if not value:
        return [None]
    return [p.strip() for p in value.split(",") if p.strip() != ""]

def _parse_optional_int_list(value: str) -> list[int | None]:
    value = (value or "").strip()
    if not value:
        return [None]
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    out: list[int | None] = []
    for p in parts:
        out.append(int(p))
    return out


def _parse_optional_bool_list(value: str) -> list[bool | None]:
    value = (value or "").strip()
    if not value:
        return [None]
    out: list[bool | None] = []
    for raw in [p.strip().lower() for p in value.split(",") if p.strip() != ""]:
        if raw in {"1", "true", "yes", "y", "on"}:
            out.append(True)
        elif raw in {"0", "false", "no", "n", "off"}:
            out.append(False)
        else:
            raise SystemExit(f"Invalid --edge_messages item: {raw!r}. Use 0/1 or true/false.")
    return out


def _parse_optional_float_list(value: str) -> list[float | None]:
    value = (value or "").strip()
    if not value:
        return [None]
    return [float(p.strip()) for p in value.split(",") if p.strip() != ""]


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
    parser.add_argument(
        "--q_sources",
        type=str,
        default="",
        help="Optional comma-separated SDCN_Q_SOURCE values (z|h4|h4_pool|pool|fused).",
    )
    parser.add_argument(
        "--edge_messages",
        type=str,
        default="",
        help="Optional comma-separated 0/1 (or true/false) to set SDCN_EDGE_MESSAGE.",
    )
    parser.add_argument(
        "--node_att_edges",
        type=str,
        default="",
        help="Optional comma-separated 0/1 (or true/false) to set SDCN_NODE_ATT_EDGE (use raw edge_attr in node attention).",
    )
    parser.add_argument(
        "--edge_ees",
        type=str,
        default="",
        help="Optional comma-separated 0/1 (or true/false) to set SDCN_EDGE_EE (edge-edge update on/off).",
    )
    parser.add_argument(
        "--ee_graphs",
        type=str,
        default="",
        help="Optional comma-separated SDCN_EE_GRAPH values (incidence|incidence_sim|edge_sim|hybrid|none).",
    )
    parser.add_argument(
        "--ee_topks",
        type=str,
        default="",
        help="Optional comma-separated ints to set SDCN_EE_TOPK (used when SDCN_EE_GRAPH=edge_sim).",
    )
    parser.add_argument(
        "--ee_sim_min_sims",
        type=str,
        default="",
        help="Optional comma-separated floats to set SDCN_EE_SIM_MIN_SIM (filters low-sim edges in *_sim ee graphs).",
    )
    parser.add_argument(
        "--ee_sim_mutuals",
        type=str,
        default="",
        help="Optional comma-separated 0/1 to set SDCN_EE_SIM_MUTUAL (keep only mutual sim edges).",
    )
    parser.add_argument(
        "--edge_denoise_alphas",
        type=str,
        default="",
        help="Optional comma-separated floats to set SDCN_EDGE_DENOISE_ALPHA (for v8/v10/v11/v12-style denoisers).",
    )
    parser.add_argument(
        "--edge_sim_gammas",
        type=str,
        default="",
        help="Optional comma-separated floats to set SDCN_EDGE_SIM_GAMMA (for v12 similarity denoiser).",
    )
    parser.add_argument("--enc_dims_list", type=str, default="", help="Optional AE encoder dims list: '256,256,256;500,500,512'.")
    parser.add_argument("--kl_weights", type=str, default="", help="Optional comma-separated SDCN_KL_WEIGHT overrides.")
    parser.add_argument("--ce_weights", type=str, default="", help="Optional comma-separated SDCN_CE_WEIGHT overrides.")
    parser.add_argument("--re_weights", type=str, default="", help="Optional comma-separated SDCN_RE_WEIGHT overrides.")
    parser.add_argument("--edge_re_weights", type=str, default="", help="Optional comma-separated SDCN_EDGE_RE_WEIGHT overrides.")
    parser.add_argument(
        "--edge_re_warmups",
        type=str,
        default="",
        help="Optional comma-separated SDCN_EDGE_RE_WARMUP_EPOCHS overrides.",
    )
    parser.add_argument("--pool_re_weights", type=str, default="", help="Optional comma-separated SDCN_POOL_RE_WEIGHT overrides.")
    parser.add_argument(
        "--pool_re_warmups",
        type=str,
        default="",
        help="Optional comma-separated SDCN_POOL_RE_WARMUP_EPOCHS overrides.",
    )
    parser.add_argument("--edge_aux_weights", type=str, default="", help="Optional comma-separated SDCN_EDGE_AUX_WEIGHT overrides.")
    parser.add_argument(
        "--edge_aux_warmups",
        type=str,
        default="",
        help="Optional comma-separated SDCN_EDGE_AUX_WARMUP_EPOCHS overrides.",
    )
    parser.add_argument(
        "--edge_aux_smooth_weights",
        type=str,
        default="",
        help="Optional comma-separated SDCN_EDGE_AUX_SMOOTH_WEIGHT overrides.",
    )
    parser.add_argument(
        "--gat_input_dropouts",
        type=str,
        default="",
        help="Optional comma-separated SDCN_GAT_INPUT_DROPOUT overrides.",
    )
    parser.add_argument("--ce_warmups", type=str, default="", help="Optional comma-separated SDCN_CE_WARMUP_EPOCHS overrides.")
    parser.add_argument("--p_smoothings", type=str, default="", help="Optional comma-separated SDCN_P_SMOOTHING values.")
    parser.add_argument("--pred_mi_weights", type=str, default="", help="Optional comma-separated SDCN_PRED_MI_WEIGHT values.")
    parser.add_argument("--q_mi_weights", type=str, default="", help="Optional comma-separated SDCN_Q_MI_WEIGHT values.")
    parser.add_argument("--q_balance_weights", type=str, default="", help="Optional comma-separated SDCN_Q_BALANCE_WEIGHT values.")
    parser.add_argument("--pred_balance_weights", type=str, default="", help="Optional comma-separated SDCN_PRED_BALANCE_WEIGHT values.")
    parser.add_argument("--q_entropy_weights", type=str, default="", help="Optional comma-separated SDCN_Q_ENTROPY_WEIGHT values.")
    parser.add_argument("--pred_entropy_weights", type=str, default="", help="Optional comma-separated SDCN_PRED_ENTROPY_WEIGHT values.")
    parser.add_argument(
        "--node_edge_pools",
        type=str,
        default="none",
        help="Comma-separated list for --node_edge_pool (none|mean_concat|mean_replace).",
    )
    parser.add_argument("--edge_ablation", type=str, default="none", help="Edge feature ablation passed to test script.")
    parser.add_argument(
        "--edge_attr_norms",
        type=str,
        default="none",
        help="Comma-separated list for --edge_attr_norm (none|zscore|zscore_clip|minmax).",
    )
    parser.add_argument("--edge_attr_clip", type=float, default=5.0, help="Clip used by zscore_clip.")
    parser.add_argument(
        "--edge_noise_stds",
        type=str,
        default="0.0",
        help="Comma-separated list for --edge_noise_std passed to test script (Gaussian noise added to edge_attr).",
    )
    parser.add_argument("--max_edges_per_node", type=int, default=10)
    parser.add_argument("--final_assign", type=str, default="pred", help="Final clustering source: pred|q|p (sets SDCN_FINAL_ASSIGN).")
    parser.add_argument("--pool_residuals", type=str, default="", help="Optional comma-separated 0/1 to set SDCN_POOL_RESIDUAL.")
    parser.add_argument("--pool_raws", type=str, default="", help="Optional comma-separated 0/1 to set SDCN_POOL_RAW.")
    parser.add_argument("--pool_upds", type=str, default="", help="Optional comma-separated 0/1 to set SDCN_POOL_UPD.")
    parser.add_argument(
        "--pool_gate_modes",
        type=str,
        default="",
        help="Optional comma-separated SDCN_POOL_GATE_MODE values (learned|one|zero).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing run_dir outputs (summary.json/trace.jsonl) and skip re-running.",
    )
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
    q_sources = _parse_optional_str_list(args.q_sources)
    edge_messages = _parse_optional_bool_list(args.edge_messages)
    node_att_edges = _parse_optional_bool_list(args.node_att_edges)
    edge_ees = _parse_optional_bool_list(args.edge_ees)
    ee_graphs = _parse_optional_str_list(args.ee_graphs)
    ee_topks = _parse_optional_int_list(args.ee_topks)
    ee_sim_min_sims = _parse_optional_float_list(args.ee_sim_min_sims)
    ee_sim_mutuals = _parse_optional_bool_list(args.ee_sim_mutuals)
    edge_denoise_alphas = _parse_optional_float_list(args.edge_denoise_alphas)
    edge_sim_gammas = _parse_optional_float_list(args.edge_sim_gammas)
    enc_dims_list = _parse_enc_dims_list(args.enc_dims_list)
    kl_weights = _parse_optional_float_list(args.kl_weights)
    ce_weights = _parse_optional_float_list(args.ce_weights)
    re_weights = _parse_optional_float_list(args.re_weights)
    edge_re_weights = _parse_optional_float_list(args.edge_re_weights)
    edge_re_warmups = _parse_optional_int_list(args.edge_re_warmups)
    pool_re_weights = _parse_optional_float_list(args.pool_re_weights)
    pool_re_warmups = _parse_optional_int_list(args.pool_re_warmups)
    edge_aux_weights = _parse_optional_float_list(args.edge_aux_weights)
    edge_aux_warmups = _parse_optional_int_list(args.edge_aux_warmups)
    edge_aux_smooth_weights = _parse_optional_float_list(args.edge_aux_smooth_weights)
    gat_input_dropouts = _parse_optional_float_list(args.gat_input_dropouts)
    ce_warmups = _parse_optional_int_list(args.ce_warmups)
    p_smoothings = _parse_optional_float_list(args.p_smoothings)
    pred_mi_weights = _parse_optional_float_list(args.pred_mi_weights)
    q_mi_weights = _parse_optional_float_list(args.q_mi_weights)
    q_balance_weights = _parse_optional_float_list(args.q_balance_weights)
    pred_balance_weights = _parse_optional_float_list(args.pred_balance_weights)
    q_entropy_weights = _parse_optional_float_list(args.q_entropy_weights)
    pred_entropy_weights = _parse_optional_float_list(args.pred_entropy_weights)
    node_edge_pools = _parse_str_list(args.node_edge_pools) if args.node_edge_pools.strip() else ["none"]
    edge_attr_norms = _parse_str_list(args.edge_attr_norms) if args.edge_attr_norms.strip() else ["none"]
    edge_noise_stds = _parse_float_list(args.edge_noise_stds) if args.edge_noise_stds.strip() else [0.0]
    edge_ablation = (args.edge_ablation or "none").strip()
    final_assign = (args.final_assign or "pred").strip().lower()
    if final_assign not in {"pred", "q", "p"}:
        raise SystemExit(f"Unknown --final_assign={final_assign!r}. Use one of: pred, q, p.")
    pool_residuals = _parse_optional_bool_list(args.pool_residuals)
    pool_raws = _parse_optional_bool_list(args.pool_raws)
    pool_upds = _parse_optional_bool_list(args.pool_upds)
    pool_gate_modes = _parse_optional_str_list(args.pool_gate_modes)
    for m in pool_gate_modes:
        if m is None:
            continue
        if m.strip().lower() not in {"learned", "one", "zero"}:
            raise SystemExit(f"Invalid --pool_gate_modes item: {m!r}. Use learned|one|zero.")

    all_runs: list[dict] = []

    for (
        variant,
        seed,
        epochs,
        lr,
        dropout,
        heads,
        n_z,
        sigma,
        q_source,
        edge_message,
        node_att_edge,
        edge_ee,
        ee_graph,
        ee_topk,
        ee_sim_min_sim,
        ee_sim_mutual,
        edge_denoise_alpha,
        edge_sim_gamma,
        kl_w,
        ce_w,
        re_w,
        edge_re_w,
        edge_re_warmup,
        pool_re_w,
        pool_re_warmup,
        edge_aux_w,
        edge_aux_warmup,
        edge_aux_smooth_w,
        gat_in_do,
        ce_warmup,
        p_smoothing,
        pred_mi_w,
        q_mi_w,
        q_bal_w,
        pred_bal_w,
        q_ent_w,
        pred_ent_w,
        pool_mode,
        edge_norm,
        edge_noise_std,
        enc_dims,
        pool_residual,
        pool_raw,
        pool_upd,
        pool_gate_mode,
    ) in product(
        variants,
        seeds,
        epochs_list,
        lrs,
        dropouts,
        heads_list,
        n_z_list,
        sigmas,
        q_sources,
        edge_messages,
        node_att_edges,
        edge_ees,
        ee_graphs,
        ee_topks,
        ee_sim_min_sims,
        ee_sim_mutuals,
        edge_denoise_alphas,
        edge_sim_gammas,
        kl_weights,
        ce_weights,
        re_weights,
        edge_re_weights,
        edge_re_warmups,
        pool_re_weights,
        pool_re_warmups,
        edge_aux_weights,
        edge_aux_warmups,
        edge_aux_smooth_weights,
        gat_input_dropouts,
        ce_warmups,
        p_smoothings,
        pred_mi_weights,
        q_mi_weights,
        q_balance_weights,
        pred_balance_weights,
        q_entropy_weights,
        pred_entropy_weights,
        node_edge_pools,
        edge_attr_norms,
        edge_noise_stds,
        enc_dims_list,
        pool_residuals,
        pool_raws,
        pool_upds,
        pool_gate_modes,
    ):
        run_name = f"seed{seed}_ep{epochs}_lr{lr:g}_do{dropout:g}_h{heads}_nz{n_z}"
        if sigma is not None:
            run_name += f"_sigma{sigma:g}"
        if q_source is not None:
            run_name += f"_q{q_source}"
        if edge_message is not None:
            run_name += f"_em{1 if edge_message else 0}"
        if node_att_edge is not None:
            run_name += f"_nae{1 if node_att_edge else 0}"
        if edge_ee is not None:
            run_name += f"_ee{1 if edge_ee else 0}"
        if ee_graph is not None:
            run_name += f"_eeg{str(ee_graph).strip().lower()}"
        if ee_topk is not None:
            run_name += f"_eek{int(ee_topk)}"
        if ee_sim_min_sim is not None:
            run_name += f"_eems{ee_sim_min_sim:g}"
        if ee_sim_mutual is not None:
            run_name += f"_eemm{1 if ee_sim_mutual else 0}"
        if edge_denoise_alpha is not None:
            run_name += f"_eda{edge_denoise_alpha:g}"
        if edge_sim_gamma is not None:
            run_name += f"_esg{edge_sim_gamma:g}"
        if kl_w is not None:
            run_name += f"_kl{kl_w:g}"
        if ce_w is not None:
            run_name += f"_ce{ce_w:g}"
        if re_w is not None:
            run_name += f"_re{re_w:g}"
        if edge_re_w is not None:
            run_name += f"_ere{edge_re_w:g}"
        if edge_re_warmup is not None:
            run_name += f"_erew{int(edge_re_warmup)}"
        if pool_re_w is not None:
            run_name += f"_pre{pool_re_w:g}"
        if pool_re_warmup is not None:
            run_name += f"_prew{int(pool_re_warmup)}"
        if edge_aux_w is not None:
            run_name += f"_eaux{edge_aux_w:g}"
        if edge_aux_warmup is not None:
            run_name += f"_eauxw{int(edge_aux_warmup)}"
        if edge_aux_smooth_w is not None:
            run_name += f"_eauxs{edge_aux_smooth_w:g}"
        if gat_in_do is not None:
            run_name += f"_gindo{gat_in_do:g}"
        if ce_warmup is not None:
            run_name += f"_cw{int(ce_warmup)}"
        if p_smoothing is not None:
            run_name += f"_ps{p_smoothing:g}"
        if pred_mi_w is not None:
            run_name += f"_pmi{pred_mi_w:g}"
        if q_mi_w is not None:
            run_name += f"_qmi{q_mi_w:g}"
        if q_bal_w is not None:
            run_name += f"_qbal{q_bal_w:g}"
        if pred_bal_w is not None:
            run_name += f"_pbal{pred_bal_w:g}"
        if q_ent_w is not None:
            run_name += f"_qent{q_ent_w:g}"
        if pred_ent_w is not None:
            run_name += f"_pent{pred_ent_w:g}"
        if pool_mode and pool_mode != "none":
            run_name += f"_pool{pool_mode}"
        if edge_ablation and edge_ablation != "none":
            run_name += f"_abl{edge_ablation}"
        if edge_norm and edge_norm != "none":
            run_name += f"_norm{edge_norm}"
        if edge_noise_std is not None and float(edge_noise_std) != 0.0:
            run_name += f"_enoise{float(edge_noise_std):g}"
        if enc_dims:
            run_name += f"_enc{enc_dims.replace(',', '-')}"
        if final_assign != "pred":
            run_name += f"_final{final_assign}"
        if pool_residual is not None:
            run_name += f"_pr{1 if pool_residual else 0}"
        if pool_raw is not None:
            run_name += f"_praw{1 if pool_raw else 0}"
        if pool_upd is not None:
            run_name += f"_pupd{1 if pool_upd else 0}"
        if pool_gate_mode is not None:
            run_name += f"_pg{str(pool_gate_mode).strip().lower()}"

        run_dir = out_dir / data_dir.name / variant / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Reduce thread-induced instability across many subprocess runs.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env["SPATIALCONV_VARIANT"] = variant
        env["SDCN_SEED"] = str(seed)
        env["SDCN_EPOCHS"] = str(epochs)
        env["SDCN_FINAL_ASSIGN"] = final_assign
        if sigma is not None:
            env["SDCN_SIGMA"] = str(sigma)
        else:
            env.pop("SDCN_SIGMA", None)
        if q_source is not None:
            env["SDCN_Q_SOURCE"] = str(q_source)
        else:
            env.pop("SDCN_Q_SOURCE", None)
        if edge_message is not None:
            env["SDCN_EDGE_MESSAGE"] = "1" if edge_message else "0"
        else:
            env.pop("SDCN_EDGE_MESSAGE", None)
        if node_att_edge is not None:
            env["SDCN_NODE_ATT_EDGE"] = "1" if node_att_edge else "0"
        else:
            env.pop("SDCN_NODE_ATT_EDGE", None)
        if edge_ee is not None:
            env["SDCN_EDGE_EE"] = "1" if edge_ee else "0"
        else:
            env.pop("SDCN_EDGE_EE", None)
        if ee_graph is not None:
            env["SDCN_EE_GRAPH"] = str(ee_graph)
        else:
            env.pop("SDCN_EE_GRAPH", None)
        if ee_topk is not None:
            env["SDCN_EE_TOPK"] = str(int(ee_topk))
        else:
            env.pop("SDCN_EE_TOPK", None)
        if ee_sim_min_sim is not None:
            env["SDCN_EE_SIM_MIN_SIM"] = str(float(ee_sim_min_sim))
        else:
            env.pop("SDCN_EE_SIM_MIN_SIM", None)
        if ee_sim_mutual is not None:
            env["SDCN_EE_SIM_MUTUAL"] = "1" if ee_sim_mutual else "0"
        else:
            env.pop("SDCN_EE_SIM_MUTUAL", None)
        if edge_denoise_alpha is not None:
            env["SDCN_EDGE_DENOISE_ALPHA"] = str(float(edge_denoise_alpha))
        else:
            env.pop("SDCN_EDGE_DENOISE_ALPHA", None)
        if edge_sim_gamma is not None:
            env["SDCN_EDGE_SIM_GAMMA"] = str(float(edge_sim_gamma))
        else:
            env.pop("SDCN_EDGE_SIM_GAMMA", None)
        if kl_w is not None:
            env["SDCN_KL_WEIGHT"] = str(kl_w)
        else:
            env.pop("SDCN_KL_WEIGHT", None)
        if ce_w is not None:
            env["SDCN_CE_WEIGHT"] = str(ce_w)
        else:
            env.pop("SDCN_CE_WEIGHT", None)
        if re_w is not None:
            env["SDCN_RE_WEIGHT"] = str(re_w)
        else:
            env.pop("SDCN_RE_WEIGHT", None)
        if edge_re_w is not None:
            env["SDCN_EDGE_RE_WEIGHT"] = str(float(edge_re_w))
        else:
            env.pop("SDCN_EDGE_RE_WEIGHT", None)
        if edge_re_warmup is not None:
            env["SDCN_EDGE_RE_WARMUP_EPOCHS"] = str(int(edge_re_warmup))
        else:
            env.pop("SDCN_EDGE_RE_WARMUP_EPOCHS", None)
        if pool_re_w is not None:
            env["SDCN_POOL_RE_WEIGHT"] = str(float(pool_re_w))
        else:
            env.pop("SDCN_POOL_RE_WEIGHT", None)
        if pool_re_warmup is not None:
            env["SDCN_POOL_RE_WARMUP_EPOCHS"] = str(int(pool_re_warmup))
        else:
            env.pop("SDCN_POOL_RE_WARMUP_EPOCHS", None)
        if edge_aux_w is not None:
            env["SDCN_EDGE_AUX_WEIGHT"] = str(float(edge_aux_w))
        else:
            env.pop("SDCN_EDGE_AUX_WEIGHT", None)
        if edge_aux_warmup is not None:
            env["SDCN_EDGE_AUX_WARMUP_EPOCHS"] = str(int(edge_aux_warmup))
        else:
            env.pop("SDCN_EDGE_AUX_WARMUP_EPOCHS", None)
        if edge_aux_smooth_w is not None:
            env["SDCN_EDGE_AUX_SMOOTH_WEIGHT"] = str(float(edge_aux_smooth_w))
        else:
            env.pop("SDCN_EDGE_AUX_SMOOTH_WEIGHT", None)
        if gat_in_do is not None:
            env["SDCN_GAT_INPUT_DROPOUT"] = str(float(gat_in_do))
        else:
            env.pop("SDCN_GAT_INPUT_DROPOUT", None)
        if ce_warmup is not None:
            env["SDCN_CE_WARMUP_EPOCHS"] = str(int(ce_warmup))
        else:
            env.pop("SDCN_CE_WARMUP_EPOCHS", None)
        if p_smoothing is not None:
            env["SDCN_P_SMOOTHING"] = str(float(p_smoothing))
        else:
            env.pop("SDCN_P_SMOOTHING", None)
        if pred_mi_w is not None:
            env["SDCN_PRED_MI_WEIGHT"] = str(float(pred_mi_w))
        else:
            env.pop("SDCN_PRED_MI_WEIGHT", None)
        if q_mi_w is not None:
            env["SDCN_Q_MI_WEIGHT"] = str(float(q_mi_w))
        else:
            env.pop("SDCN_Q_MI_WEIGHT", None)
        if q_bal_w is not None:
            env["SDCN_Q_BALANCE_WEIGHT"] = str(float(q_bal_w))
        else:
            env.pop("SDCN_Q_BALANCE_WEIGHT", None)
        if pred_bal_w is not None:
            env["SDCN_PRED_BALANCE_WEIGHT"] = str(float(pred_bal_w))
        else:
            env.pop("SDCN_PRED_BALANCE_WEIGHT", None)
        if q_ent_w is not None:
            env["SDCN_Q_ENTROPY_WEIGHT"] = str(float(q_ent_w))
        else:
            env.pop("SDCN_Q_ENTROPY_WEIGHT", None)
        if pred_ent_w is not None:
            env["SDCN_PRED_ENTROPY_WEIGHT"] = str(float(pred_ent_w))
        else:
            env.pop("SDCN_PRED_ENTROPY_WEIGHT", None)
        if enc_dims:
            env["SDCN_ENC_DIMS"] = enc_dims
        else:
            env.pop("SDCN_ENC_DIMS", None)
        if pool_residual is not None:
            env["SDCN_POOL_RESIDUAL"] = "1" if pool_residual else "0"
        else:
            env.pop("SDCN_POOL_RESIDUAL", None)
        if pool_raw is not None:
            env["SDCN_POOL_RAW"] = "1" if pool_raw else "0"
        else:
            env.pop("SDCN_POOL_RAW", None)
        if pool_upd is not None:
            env["SDCN_POOL_UPD"] = "1" if pool_upd else "0"
        else:
            env.pop("SDCN_POOL_UPD", None)
        if pool_gate_mode is not None:
            env["SDCN_POOL_GATE_MODE"] = str(pool_gate_mode).strip().lower()
        else:
            env.pop("SDCN_POOL_GATE_MODE", None)

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
            "--node_edge_pool",
            str(pool_mode),
            "--edge_ablation",
            str(edge_ablation),
            "--edge_attr_norm",
            str(edge_norm),
            "--edge_attr_clip",
            str(args.edge_attr_clip),
            "--edge_noise_std",
            str(float(edge_noise_std)),
            "--max_edges_per_node",
            str(args.max_edges_per_node),
            "--summary_json",
            "summary.json",
            "--trace_jsonl",
            "trace.jsonl",
        ]
        if args.cpu:
            cmd.append("--cpu")

        summary_path = run_dir / "summary.json"
        trace_path = run_dir / "trace.jsonl"
        log_path = run_dir / "run.log"

        if not (args.resume and summary_path.exists() and trace_path.exists()):
            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(f"$ {' '.join(cmd)}\n")
                log_f.write(
                    f"SPATIALCONV_VARIANT={variant} SDCN_SEED={seed} SDCN_EPOCHS={epochs} "
                    f"SDCN_SIGMA={(sigma if sigma is not None else '')} "
                    f"SDCN_Q_SOURCE={(q_source if q_source is not None else '')} "
                    f"SDCN_EDGE_MESSAGE={(edge_message if edge_message is not None else '')} "
                    f"SDCN_NODE_ATT_EDGE={(node_att_edge if node_att_edge is not None else '')} "
                    f"SDCN_EDGE_EE={(edge_ee if edge_ee is not None else '')} "
                    f"SDCN_EE_GRAPH={(ee_graph if ee_graph is not None else '')} "
                    f"SDCN_EE_TOPK={(ee_topk if ee_topk is not None else '')} "
                    f"SDCN_EDGE_DENOISE_ALPHA={(edge_denoise_alpha if edge_denoise_alpha is not None else '')} "
                    f"SDCN_EDGE_SIM_GAMMA={(edge_sim_gamma if edge_sim_gamma is not None else '')} "
                    f"SDCN_FINAL_ASSIGN={final_assign} "
                    f"SDCN_KL_WEIGHT={(kl_w if kl_w is not None else '')} "
                    f"SDCN_CE_WEIGHT={(ce_w if ce_w is not None else '')} "
                    f"SDCN_RE_WEIGHT={(re_w if re_w is not None else '')} "
                    f"SDCN_EDGE_RE_WEIGHT={(edge_re_w if edge_re_w is not None else '')} "
                    f"SDCN_EDGE_RE_WARMUP_EPOCHS={(edge_re_warmup if edge_re_warmup is not None else '')} "
                    f"SDCN_POOL_RE_WEIGHT={(pool_re_w if pool_re_w is not None else '')} "
                    f"SDCN_POOL_RE_WARMUP_EPOCHS={(pool_re_warmup if pool_re_warmup is not None else '')} "
                    f"SDCN_CE_WARMUP_EPOCHS={(ce_warmup if ce_warmup is not None else '')} "
                    f"SDCN_P_SMOOTHING={(p_smoothing if p_smoothing is not None else '')} "
                    f"SDCN_PRED_MI_WEIGHT={(pred_mi_w if pred_mi_w is not None else '')} "
                    f"SDCN_Q_MI_WEIGHT={(q_mi_w if q_mi_w is not None else '')} "
                    f"SDCN_Q_BALANCE_WEIGHT={(q_bal_w if q_bal_w is not None else '')} "
                    f"SDCN_PRED_BALANCE_WEIGHT={(pred_bal_w if pred_bal_w is not None else '')} "
                    f"SDCN_Q_ENTROPY_WEIGHT={(q_ent_w if q_ent_w is not None else '')} "
                    f"SDCN_PRED_ENTROPY_WEIGHT={(pred_ent_w if pred_ent_w is not None else '')} "
                    f"SDCN_ENC_DIMS={enc_dims}\n"
                    f"SDCN_POOL_RESIDUAL={(pool_residual if pool_residual is not None else '')} "
                    f"SDCN_POOL_RAW={(pool_raw if pool_raw is not None else '')} "
                    f"SDCN_POOL_UPD={(pool_upd if pool_upd is not None else '')} "
                    f"SDCN_POOL_GATE_MODE={(pool_gate_mode if pool_gate_mode is not None else '')}\n"
                    f"node_edge_pool={pool_mode} edge_ablation={edge_ablation} edge_attr_norm={edge_norm} edge_attr_clip={args.edge_attr_clip} edge_noise_std={edge_noise_std}\n\n"
                )
                log_f.flush()
                subprocess.run(cmd, cwd=str(run_dir), env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        n_nodes = int(summary.get("n_nodes", 0) or 0)
        n_clusters = int(summary.get("n_clusters", 0) or 0)
        cluster_dist = summary.get("cluster_distribution") or {}
        collapse_final = bool(_collapse_flag(_as_int_dict(cluster_dist), n_nodes=n_nodes, n_clusters=n_clusters))

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
                "q_source": q_source,
                "edge_message": None if edge_message is None else bool(edge_message),
                "node_att_edge": None if node_att_edge is None else bool(node_att_edge),
                "edge_ee": None if edge_ee is None else bool(edge_ee),
                "ee_graph": ee_graph,
                "ee_topk": None if ee_topk is None else int(ee_topk),
                "edge_denoise_alpha": None if edge_denoise_alpha is None else float(edge_denoise_alpha),
                "edge_sim_gamma": None if edge_sim_gamma is None else float(edge_sim_gamma),
                "kl_weight": None if kl_w is None else float(kl_w),
                "ce_weight": None if ce_w is None else float(ce_w),
                "re_weight": None if re_w is None else float(re_w),
                "edge_re_weight": None if edge_re_w is None else float(edge_re_w),
                "edge_re_warmup_epochs": None if edge_re_warmup is None else int(edge_re_warmup),
                "pool_re_weight": None if pool_re_w is None else float(pool_re_w),
                "pool_re_warmup_epochs": None if pool_re_warmup is None else int(pool_re_warmup),
                "ce_warmup_epochs": None if ce_warmup is None else int(ce_warmup),
                "p_smoothing": None if p_smoothing is None else float(p_smoothing),
                "pred_mi_weight": None if pred_mi_w is None else float(pred_mi_w),
                "q_mi_weight": None if q_mi_w is None else float(q_mi_w),
                "q_balance_weight": None if q_bal_w is None else float(q_bal_w),
                "pred_balance_weight": None if pred_bal_w is None else float(pred_bal_w),
                "q_entropy_weight": None if q_ent_w is None else float(q_ent_w),
                "pred_entropy_weight": None if pred_ent_w is None else float(pred_ent_w),
                "node_edge_pool": str(pool_mode),
                "edge_ablation": str(edge_ablation),
                "edge_attr_norm": str(edge_norm),
                "edge_attr_clip": float(args.edge_attr_clip),
                "edge_noise_std": float(edge_noise_std),
                "enc_dims": enc_dims,
                "final_assign": str(final_assign),
                "pool_residual": None if pool_residual is None else bool(pool_residual),
                "pool_raw": None if pool_raw is None else bool(pool_raw),
                "pool_upd": None if pool_upd is None else bool(pool_upd),
                "pool_gate_mode": pool_gate_mode,
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
            f"sigma={(sigma if sigma is not None else 'default')}, q={(q_source if q_source is not None else 'default')}, "
            f"edge_msg={(edge_message if edge_message is not None else 'default')}, edge_ee={(edge_ee if edge_ee is not None else 'default')}, "
            f"ee_graph={(ee_graph if ee_graph is not None else 'default')}, "
            f"pool={pool_mode}, norm={edge_norm}, edge_noise_std={float(edge_noise_std):g}, "
            f"enc_dims={(enc_dims if enc_dims else 'default')}: "
            f"acc={acc:.4f} nmi={nmi:.4f} ari={ari:.4f} f1={f1:.4f} collapse_final={collapse_final}"
        )

    aggregate_path = out_dir / "aggregate.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
    print(f"Saved aggregate results to: {aggregate_path}")


if __name__ == "__main__":
    main()
