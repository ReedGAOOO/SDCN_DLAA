# 综合评价参数体系（用于比较 SDCN_DLAA 各结构变体）

目标场景：**真实城市网络聚类（无真值）** + **研发阶段（有真值/合成对照）** 都能用同一套框架做比较，并且能显式惩罚“塌缩/缺簇/不稳定”。

---

## 0) 先说清楚：评估分两层

1) **硬门槛（Fail-fast）**：先把明显不可用的排除（塌缩、极端不平衡、严重不稳定）。
2) **软排名（Composite score）**：对通过门槛的方案，用多指标加权得到排序；权重可按任务调整。

> 原因：很多无监督指标在“塌缩”时也能看起来不错（例如 KL 很小、modularity 偶尔偏高），因此必须先做门槛筛掉失败模式。
>
> 另外：**评估体系本身无法从理论上“保证训练一定收敛”**（那是优化与训练策略问题），但可以通过“硬门槛 + 趋势诊断”把不收敛/不对齐的 run 判为 Fail，从而保证最终用于对比/落地的候选都是“已经收敛”的。

---

## 1) 指标分组（推荐 5 类 + 1 类下游）

### A. 外部指标（有真值才用）

- `acc / nmi / ari / f1(macro)`

### B. 图划分质量（无真值也适用，Graph-space）

本仓库已实现并在 `tools/test_conceptual_data.py` 的 `summary.json.metrics` 输出：

- `modularity`（↑）
- `within_edge_ratio`（↑）
- `conductance_mean`（↓）
- `ncut_mean`（↓）
- `largest_cc_ratio_mean`（↑，簇内部连通性诊断）
- `cluster_entropy_norm`（↑，防止极端不均衡；不是越大越好，但可用于塌缩预警）

### C. 内部指标（Embedding-space）

建议在“用于聚类的表示”上算（例如 `SDCN_Q_SOURCE` 对应的 embedding，而不是原始高维 `x`）。

已通过 `tools/test_conceptual_data.py --internal_metrics` 支持输出：

- `silhouette`（↑，默认 cosine + 采样）
- `davies_bouldin`（↓）
- `calinski_harabasz`（↑）

### D. 收敛 / 分布对齐（双重自监督是否“真的拉近”）

这组指标用于回答你关心的核心问题：**AE 分支产生的目标分布 `P` 与结构分支 `pred` 是否对齐**（以及 `q` 与 `pred` 的一致性）。

本仓库会在训练过程中记录每个 epoch 的诊断（`trace.jsonl`），并在 `tools/sweep_stability.py` 的 `trace_stats` 中汇总：

- `kl_p_pred_last_mean`（↓）：最后一段窗口（约 20% epoch，最多 10 个）的 `KL(P||pred)` 均值；没有该字段时退化用 `kl_p_pred_mean`（全程均值）。
- `align_nmi_q_pred_last_mean`（↑）：最后窗口里 `argmax(q)` 与 `argmax(pred)` 的 NMI 均值；没有该字段时退化用 `align_nmi_q_pred_mean`。

> 说明：`KL(P||Q)`（`kl_p_q_*`）通常较小，因为 `P` 由 `Q` 构造（sharpening），更适合作为“分布是否过度尖峰”的辅助诊断，而不是结构分支对齐的主指标。

### E. 稳定性 / 鲁棒性（无真值比较的关键维度）

建议至少包含：

- `collapse_rate`：多 seed/扰动下的塌缩比例（↓）
- `consistency_nmi`：不同 seed/扰动得到的聚类结果两两 NMI 均值（↑）

> 这些可由 `tools/sweep_stability.py` 产生的 `aggregate.json` + 每次 run 的 `sdcn_dlaa_final_cluster_results.csv` 计算得到。

### F. 下游可用性（最终裁判）

若你有下游任务（预测/检索/解释性分析），以“下游指标提升”作为最终选择依据；
无下游时可先用 A–D 做筛选。

---

## 2) 推荐的“硬门槛”（无真值时）

你可以用下面的默认门槛做第一轮筛选（可按实际调整）：

- `collapse_rate <= 0.0`（3 seeds 全不塌缩；或放宽到 `<= 1/3`）
- `max_cluster_frac < 0.9`（可从 cluster_distribution 计算；防止单簇吃掉 90% 节点）
- `consistency_nmi >= 0.2`（扰动下仍能保持基本一致）
- （建议新增）`kl_p_pred_last_mean <= 0.1`（结构分支与目标分布对齐；更严格可用 `<= 0.05`）
- （可选）`align_nmi_q_pred_last_mean >= 0.2`（两分支输出的硬聚类一致性达到“基本同意”）

---

## 3) 推荐的“软排名”权重（Profile）

### 3.1 无真值城市网络（`unlabeled_city`）

默认权重（已编码在 `tools/rank_composite.py`）：

- `collapse_rate` 0.20（↓）
- `consistency_nmi` 0.15（↑）
- `kl_p_pred_last_mean` 0.07（↓）
- `kl_p_pred_mean` 0.03（↓，无 last_mean 时兜底）
- `align_nmi_q_pred_last_mean` 0.03（↑）
- `align_nmi_q_pred_mean` 0.02（↑，无 last_mean 时兜底）
- `modularity` 0.125（↑）
- `conductance_mean` 0.125（↓）
- `ncut_mean` 0.04（↓）
- `within_edge_ratio` 0.05（↑）
- `largest_cc_ratio_mean` 0.05（↑）
- `cluster_entropy_norm` 0.05（↑）
- `silhouette` 0.03（↑，可选）
- `davies_bouldin` 0.02（↓，可选）
- `calinski_harabasz` 0.01（↑，可选）

### 3.2 有真值研发对照（`labeled_dev`）

- `acc` 0.25（↑）
- `nmi` 0.20（↑）
- `ari` 0.15（↑）
- `f1` 0.10（↑）
- `collapse_rate` 0.10（↓）
- `consistency_nmi` 0.03（↑）
- `kl_p_pred_last_mean` 0.03（↓）
- `kl_p_pred_mean` 0.02（↓）
- `align_nmi_q_pred_last_mean` 0.02（↑）
- `align_nmi_q_pred_mean` 0.01（↑）
- `modularity` 0.03（↑）
- `conductance_mean` 0.03（↓）
- `ncut_mean` 0.02（↓）
- `within_edge_ratio` 0.01（↑）

---

## 4) 推荐落地方式（脚本）

1) 跑多 seed/扰动（并可选开启内部指标）：

```bash
conda run -n gnn python tools/sweep_stability.py ... --internal_metrics
```

2) 生成综合排名：

```bash
conda run -n gnn python tools/rank_composite.py \
  --aggregate_json /path/to/aggregate.json \
  --profile unlabeled_city
```

3) （可选）启用“硬门槛”先筛收敛/稳定性：

```bash
conda run -n gnn python tools/rank_composite.py \
  --aggregate_json /path/to/aggregate.json \
  --profile unlabeled_city \
  --gate \
  --max_collapse_rate 0.0 \
  --min_consistency_nmi 0.2 \
  --max_kl_p_pred_mean 0.1
```
