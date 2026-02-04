# edge_edge_denoise_nonknn：所有 SpatialConv 实验版本效果排名（本次配置）

> 说明：该排名**只对当前诊断集与当前统一超参设置有效**（用于比较“同一训练配方下，各结构的相对好坏”）。
> 如需“跨数据集/跨配方”的更稳健排名，需要再指定 suite 与评价指标（acc/nmi/ari/f1 + 图分割指标等）。
>
> 更新（2026-02-04）：本仓库新增了“综合评价体系”（含收敛/对齐、稳定性、图划分质量等）。本文件保留原先的 `acc_mean` 排名用于对照，并追加新的 **composite 排名**（见文末）。

- 数据集：`/tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn`
- 结果聚合：`/tmp/sweep_all_variants_edge_edge_denoise_nonknn/aggregate.json`

---

## 复现命令

```bash
conda run -n gnn python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/sweep_all_variants_edge_edge_denoise_nonknn \
  --variants v1original,v2edge_single_layer,v3edge_cross_layers,v4edge_pool_fusion,v5edge_pool_residual,v6edge_ee_aux,v7edge_attr_fusion,v8edge_denoise_attr,v9edge_context_denoise,v10edge_base_denoise_plus_context,v11edge_adaptive_denoise_context,v12edge_similarity_denoise,v13edge_context_similarity_denoise,v14edge_pool_concat_fusion,v15edge_ee_aux_context_similarity_denoise,v16edge_ee_residual_aux_fusion,v17edge_attr_gate,v18edge_attr_mlp_fuse,v19edge_attn_pool,v20edge_attr_scalar_gate,v21dual_sgat_edge_attr,v22stable_denoise_scalar_fuse,v23sman_incidence \
  --seeds 0,1,2 \
  --epochs 60 \
  --lrs 1e-3 \
  --dropouts 0.2 \
  --heads 1 \
  --n_z 10 \
  --sigmas 0.2 \
  --q_sources h4 \
  --edge_messages 1 \
  --edge_ees 1 \
  --ee_graphs incidence_sim \
  --ee_topks 4 \
  --ee_sim_min_sims 0.4 \
  --edge_denoise_alphas 0.1 \
  --edge_sim_gammas 1.0 \
  --q_balance_weights 0.1 \
  --pred_balance_weights 0.1 \
  --edge_attr_fuses 1 \
  --edge_attr_fuse_scales 0.1 \
  --edge_attr_fuse_detaches 0 \
  --edge_aux_weights 0.0 \
  --kl_weights 0.1 \
  --ce_weights 1.0 \
  --ce_warmups 20 \
  --final_assign pred
```

---

## 排名（按 `acc_mean` 降序）

| rank | variant | acc_mean±std | nmi_mean | ari_mean | f1_mean | collapse |
|---:|---|---:|---:|---:|---:|---:|
| 1 | v14edge_pool_concat_fusion | 0.5778±0.0605 | 0.3554 | 0.3185 | 0.4959 | 1/3 |
| 2 | v15edge_ee_aux_context_similarity_denoise | 0.5653±0.1233 | 0.3697 | 0.3359 | 0.4745 | 2/3 |
| 3 | v10edge_base_denoise_plus_context | 0.5556±0.1230 | 0.3225 | 0.2768 | 0.5023 | 0/3 |
| 4 | v13edge_context_similarity_denoise | 0.4819±0.0719 | 0.2511 | 0.2062 | 0.4174 | 1/3 |
| 5 | v12edge_similarity_denoise | 0.4764±0.0749 | 0.2528 | 0.2078 | 0.4103 | 1/3 |
| 6 | v11edge_adaptive_denoise_context | 0.4542±0.0657 | 0.1951 | 0.1451 | 0.3673 | 2/3 |
| 7 | v8edge_denoise_attr | 0.4514±0.0264 | 0.2361 | 0.1680 | 0.3626 | 1/3 |
| 8 | v9edge_context_denoise | 0.4222±0.0778 | 0.2241 | 0.1458 | 0.3199 | 1/3 |
| 9 | v5edge_pool_residual | 0.4125±0.0442 | 0.0966 | 0.0874 | 0.3767 | 1/3 |
| 10 | v21dual_sgat_edge_attr | 0.3750±0.0531 | 0.0961 | 0.0615 | 0.3159 | 1/3 |
| 11 | v4edge_pool_fusion | 0.3514±0.0142 | 0.0921 | 0.0292 | 0.2812 | 2/3 |
| 12 | v7edge_attr_fusion | 0.3486±0.0553 | 0.0755 | 0.0359 | 0.3200 | 0/3 |
| 13 | v17edge_attr_gate | 0.3472±0.0559 | 0.0720 | 0.0362 | 0.2775 | 2/3 |
| 14 | v16edge_ee_residual_aux_fusion | 0.3319±0.0580 | 0.0707 | 0.0305 | 0.2305 | 2/3 |
| 15 | v22stable_denoise_scalar_fuse | 0.3194±0.0510 | 0.0653 | 0.0401 | 0.2286 | 1/3 |
| 16 | v2edge_single_layer | 0.3167±0.0358 | 0.0566 | 0.0105 | 0.2352 | 2/3 |
| 17 | v1original | 0.3167±0.0335 | 0.0505 | 0.0126 | 0.2293 | 2/3 |
| 18 | v20edge_attr_scalar_gate | 0.3014±0.0342 | 0.0529 | 0.0075 | 0.2021 | 3/3 |
| 19 | v23sman_incidence | 0.3014±0.0391 | 0.0541 | 0.0197 | 0.1981 | 2/3 |
| 20 | v18edge_attr_mlp_fuse | 0.3014±0.0367 | 0.0380 | 0.0086 | 0.2045 | 3/3 |
| 21 | v19edge_attn_pool | 0.2931±0.0341 | 0.0282 | 0.0137 | 0.1766 | 3/3 |
| 22 | v3edge_cross_layers | 0.2806±0.0227 | 0.0145 | 0.0031 | 0.1810 | 2/3 |
| 23 | v6edge_ee_aux | 0.2653±0.0020 | 0.0271 | 0.0002 | 0.1309 | 3/3 |

---

## 重新评估：综合评价体系（Composite Ranking）

综合评价体系说明见：`reports/composite_evaluation_system_zh.md`。

本次“重新评估”做了两件事：

1) 使用 `tools/sweep_stability.py --resume` 在不重训的情况下，重新汇总 `trace.jsonl`，补齐 `kl_p_pred_last_mean` 等“收敛趋势”统计写回 `aggregate.json`。
2) 用 `tools/rank_composite.py` 在同一套配置下，对 23 个变体做综合评分排序（分别给出 `unlabeled_city` 与 `labeled_dev` 两个 profile）。

> 备注：由于这些 runs 是在更早版本代码下生成的，`trace.jsonl` 中不包含 `align_nmi_q_pred_*`（q↔pred 对齐度）。如果你希望把“两个自监督模块硬聚类一致性”纳入评分，需要用最新代码 **重新跑 sweep（不加 `--resume`）**。

### 复现命令（只重算汇总与排名，不重训）

```bash
# 1) 重算 trace_stats（不重训）
conda run -n gnn python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/sweep_all_variants_edge_edge_denoise_nonknn \
  --variants v1original,v2edge_single_layer,v3edge_cross_layers,v4edge_pool_fusion,v5edge_pool_residual,v6edge_ee_aux,v7edge_attr_fusion,v8edge_denoise_attr,v9edge_context_denoise,v10edge_base_denoise_plus_context,v11edge_adaptive_denoise_context,v12edge_similarity_denoise,v13edge_context_similarity_denoise,v14edge_pool_concat_fusion,v15edge_ee_aux_context_similarity_denoise,v16edge_ee_residual_aux_fusion,v17edge_attr_gate,v18edge_attr_mlp_fuse,v19edge_attn_pool,v20edge_attr_scalar_gate,v21dual_sgat_edge_attr,v22stable_denoise_scalar_fuse,v23sman_incidence \
  --seeds 0,1,2 \
  --epochs 60 \
  --lrs 1e-3 \
  --dropouts 0.2 \
  --heads 1 \
  --n_z 10 \
  --sigmas 0.2 \
  --q_sources h4 \
  --edge_messages 1 \
  --edge_ees 1 \
  --ee_graphs incidence_sim \
  --ee_topks 4 \
  --ee_sim_min_sims 0.4 \
  --edge_denoise_alphas 0.1 \
  --edge_sim_gammas 1.0 \
  --q_balance_weights 0.1 \
  --pred_balance_weights 0.1 \
  --edge_attr_fuses 1 \
  --edge_attr_fuse_scales 0.1 \
  --edge_attr_fuse_detaches 0 \
  --edge_aux_weights 0.0 \
  --kl_weights 0.1 \
  --ce_weights 1.0 \
  --ce_warmups 20 \
  --final_assign pred \
  --resume

# 2) 输出 composite 排名（无真值城市网络侧重点）
conda run -n gnn python tools/rank_composite.py \
  --aggregate_json /tmp/sweep_all_variants_edge_edge_denoise_nonknn/aggregate.json \
  --profile unlabeled_city

# 3) 输出 composite 排名（有真值研发对照侧重点）
conda run -n gnn python tools/rank_composite.py \
  --aggregate_json /tmp/sweep_all_variants_edge_edge_denoise_nonknn/aggregate.json \
  --profile labeled_dev
```

### 排名（`unlabeled_city`：稳定性 + 图指标 + 收敛）

> 原始输出保存在：`/tmp/rank_edge_edge_denoise_nonknn_unlabeled_city.md`

| rank | variant | composite | collapse_rate | consistency_nmi | kl_p_pred_last_mean |
|---:|---|---:|---:|---:|---:|
| 1 | v10edge_base_denoise_plus_context | 0.6808 | 0/3 | 0.2266 | 0.0730 |
| 2 | v15edge_ee_aux_context_similarity_denoise | 0.6659 | 2/3 | 0.3503 | 0.0675 |
| 3 | v5edge_pool_residual | 0.6070 | 1/3 | 0.1478 | 0.0129 |
| 4 | v7edge_attr_fusion | 0.5771 | 0/3 | 0.1596 | 0.0057 |
| 5 | v22stable_denoise_scalar_fuse | 0.5635 | 1/3 | 0.0536 | 0.0229 |

（完整 23 个变体请查看 `/tmp/rank_edge_edge_denoise_nonknn_unlabeled_city.md`）

### 排名（`labeled_dev`：acc/nmi/ari/f1 + 收敛 + 稳定性）

> 原始输出保存在：`/tmp/rank_edge_edge_denoise_nonknn_labeled_dev.md`

| rank | variant | composite | collapse_rate | consistency_nmi | kl_p_pred_last_mean |
|---:|---|---:|---:|---:|---:|
| 1 | v15edge_ee_aux_context_similarity_denoise | 0.8646 | 2/3 | 0.3503 | 0.0675 |
| 2 | v10edge_base_denoise_plus_context | 0.8491 | 0/3 | 0.2266 | 0.0730 |
| 3 | v14edge_pool_concat_fusion | 0.8366 | 1/3 | 0.2225 | 0.2445 |
| 4 | v13edge_context_similarity_denoise | 0.6489 | 1/3 | 0.1372 | 0.0512 |
| 5 | v12edge_similarity_denoise | 0.6468 | 1/3 | 0.1406 | 0.0513 |

（完整 23 个变体请查看 `/tmp/rank_edge_edge_denoise_nonknn_labeled_dev.md`）

---

## 结构原因分析：为什么“新排名前列”会是这些变体？

下面按结构把前列模型拆成几条“信息通路”，解释它们为何在当前综合体系下占优（尤其是 **不塌缩/更稳定** + **图划分质量更好** + **`P`↔`pred` 更对齐（KL 更小）**）。

> 重要备注：本次是在旧 run 上 `--resume` 重算 `trace_stats`，因此 `trace.jsonl` 里**没有**写入 `align_nmi_q_pred_*`（q↔pred 硬聚类对齐度），所以 composite 排名实际主要受 `collapse_rate / consistency_nmi / KL(P||pred) / 图指标` 驱动。若你希望把“两个自监督模块硬聚类一致性”也纳入评估，需要用最新代码重新跑 sweep。

### 1) v10：`v10edge_base_denoise_plus_context`（无真值 profile 排名 #1）

结构关键点：

- **edge↔edge 角色很“克制”**：先在 *raw edge_attr* 上做 denoise（`edge_base=(1-α)raw+α*ee(raw)`），再加 node-pair context。`ee` 更像“正则化/去噪器”，而不是“重写边语义的强表征器”。（`DLAA_NEW.py:2180`）
- **context 是门控残差**：`node_ctx_scale` 初值为 0，通过 `tanh(scale)` 逐步放大；早期训练不会被 node_ctx 强行带偏，有利于 `KL(P||pred)` 收敛。
- **node 更新直接吃 edge_attr**：`SGATLayer(..., edge_dim=H)`，注意力里真的用到边特征；这会同时提升图指标（更像“关系驱动”的聚类）。
- **保留 pooled residual**：raw/upd edge mean-pool 到 node，再用 gate 做残差；这条“强基线通路”让模型即使 edge↔edge 贡献不大也不至于退化。

为什么在综合体系下占优：

- `collapse_rate=0` 说明这种“去噪→小步注入→残差兜底”的结构最不容易把结构分支推向单簇塌缩。
- 但 KL 并不是最小（`kl_p_pred_last_mean` 中等），说明它更像“稳定的结构增益”，而不是“强行对齐到 AE 的 P”。

### 2) v15：`v15edge_ee_aux_context_similarity_denoise`（有真值 profile 排名 #1 / 无真值 #2）

结构关键点：

- **相似度加权的 edge↔edge 平滑**：不是直接 GAT 混合，而是用 `edge_key = edge_raw + s*node_ctx` 计算相似度权重，再做加权平均平滑（避免在异质 edge↔edge 邻域里“乱搅”）。这会显著改善图划分指标与一致性。（`DLAA_NEW.py:2945`）
- **同样是“denoise→context 残差→SGAT”**：整体思路比 v10 更偏“自适应平滑”，表达力更强。
- **带 edge-level auxiliary head**：本结构里有 `edge_within_lin`，但注意本次 sweep 配置里 `SDCN_EDGE_AUX_WEIGHT=0.0`，所以 aux head 其实没有参与训练（结构上有、优化上没用上）。

为什么表现“强但不稳”：

- v15 往往带来更高的 `consistency_nmi` 和更好的图指标（结构更强、平滑更有针对性），因此在 `labeled_dev`（acc/nmi/ari/f1 权重大）会排第一。
- 但在本配置下它的 `collapse_rate` 较高（2/3 seed 塌缩），说明“更强的结构分支”在某些随机初始化/早期阶段会把 `pred` 推向极端解；这在 `unlabeled_city`（更看重稳定）会被显式惩罚，因此不如 v10。

### 3) v5：`v5edge_pool_residual`（无真值 #3）

结构关键点（偏“强基线/稳”）：

- **node attention 直接用 raw edge_attr**（V2 哲学），不依赖 edge embedding 的正确性，训练更稳。（`DLAA_NEW.py:682`）
- **edge↔edge 更新存在，但主要通过 pooling residual 影响 node**：edge embedding 不直接进入 attention，而是走 mean-pool 残差通路；这减少了 edge↔edge 噪声对注意力 logits 的干扰。
- **raw + upd pooling + learned gate**：当 edge embedding 有帮助时才“慢慢加”，否则模型会退回到 raw edge + node attention 的稳定解。

为什么综合分高：

- KL 往往更小（结构分支更容易被 `P` 拉住），塌缩概率也低于更激进的融合变体；但图指标提升幅度有限，所以通常排在 v10/v15 后。

### 4) v7：`v7edge_attr_fusion`（无真值 #4）

结构关键点（v5 的“更强注入版”）：

- 在 v5 的基础上，把 **edge↔edge 产物 `edge_feat_1` 直接加回 node attention 的 edge_attr**：
  `edge_attr_att = dist_feat + scale*norm(edge_feat_1)`（可 detach，可 LayerNorm）。（`DLAA_NEW.py:931`）
- 优点：edge↔edge 不再只通过 pooling 影响 node，而是直接调制注意力，理论上更强。
- 风险：如果 edge↔edge 邻域异质或更新不稳，注意力会被“错误边语义”干扰，导致图指标不一定提升（本次 modularity 低、conductance 偏高）。

为什么还能排前：

- 这次 `fuse_scale=0.1`（较小）+ LayerNorm + 仍保留 pooling residual，使它没有“强到失控”，因此在稳定性上没输太多；同时 KL 很小（容易对齐）。

### 5) v22：`v22stable_denoise_scalar_fuse`（无真值 #5）

结构关键点（“保守 + 稳定工程化”）：

- 用 v15 的相似度平滑做 edge_out，但 **node attention 仍从 raw edge_attr 起步**；
- 将 edge_out 注入 attention 时用 **每条边的 scalar gate**（并且 `gate_scale` 初值 0、`gate_lin` 0-init），训练一开始几乎等价于 v5；只有当 edge_out 确实有用时才逐步放大。（`DLAA_NEW.py:3113`）
- 额外的“防抖”：在最后 logits 层（`out_activation is F.leaky_relu`）直接跳过 denoise/context 混合，避免在分类头空间做复杂传播引发不稳定。

为什么它的图指标（尤其 conductance）可能更好：

- 这种“raw 主通路 + gated 注入 + logits 层保护”更像把 edge_out 当作“可选的结构先验”，在不破坏主训练动力学的前提下改善边界质量；但一致性不一定最高（本次 `consistency_nmi` 较低）。

---

## 一句话总结（可操作的结构结论）

- **如果目标是“真实城市网络（无真值）先求稳、再求好”**：优先 v10/v5/v22 这类“edge↔edge 当去噪器 + 门控注入 + 残差兜底”的结构。
- **如果目标是“有真值对照追 acc/nmi/ari”**：v15/v14 这类“相似度平滑 + 强融合（concat/更强 edge_out）”更容易冲高分，但需要额外的稳定化手段（更严格的门槛筛选、多 seed、或改训练配方）。
