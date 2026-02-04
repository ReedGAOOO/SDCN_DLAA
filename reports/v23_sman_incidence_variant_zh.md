# v23sman_incidence：更贴近 SMAN “incidence(edge-node) 图” 的节点更新版本

> 目的：把 “edge embedding 作为一等公民” 的思路再推进一步：不仅做 edge↔edge（线图）传播，还在 **node+edge 的二部图/关联图（incidence graph）** 上，用 **edge-aware attention** 从 *edge-nodes → node-nodes* 聚合更新节点表征。

---

## 1) 对应关系：SMAN 原版在做什么

- SMAN 的核心算子在 `Reference/SMAN_ORIGINAL/SMAN_layers.py:120`：`SpatialConv(...)`（step1 更新 edge，step2 用 `sgat` 更新 node）。

在本仓库已有的“SMAN-port”：

- `DLAA_NEW.py:212`：`SpatialConvV1Original`（最直接的移植）
- `DLAA_NEW.py:328`：`SpatialConvV2EdgeSingleLayer`（更稳定：node 更新不再“洗掉”edge 行）
- `DLAA_NEW.py:400`：`SpatialConvV3EdgeCrossLayers`（把 refined edge embedding 作为 node attention 的 edge_attr）

---

## 2) v23 新增了什么

实现：`DLAA_NEW.py:460` → `SpatialConvV23SmanIncidenceBipartite`

选择方式：

- `SPATIALCONV_VARIANT=v23` 或 `v23sman_incidence`

关键点：

- 把每条原始边 `e` 当作一个 **edge-node**（id = `N + e`）
- 构造 incidence 有向边：`(N+e)->src` 与 `(N+e)->dst`
- 在 `(N+E)` 个节点上跑一次 `SGATLayer`，用（投影后的）raw `dist_feat` 作为 incidence 边的 `edge_attr`
- 返回仍保持 SDCN 的契约：`concat([node_out, edge_feat])`

---

## 3) 一次诊断集结果（edge_edge_denoise_nonknn）

对比命令（与你之前 v22 报告同一套“压测 pred 稳定性”的配置）：

```bash
conda run -n gnn python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/sweep_v23_sman_compare \
  --variants v2edge_single_layer,v3edge_cross_layers,v23sman_incidence,v5edge_pool_residual,v16edge_ee_residual_aux_fusion \
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

摘要（来自 `/tmp/sweep_v23_sman_compare/aggregate.json`）：

- `v2edge_single_layer`：`acc_mean≈0.3167`，`collapse=2/3`
- `v3edge_cross_layers`：`acc_mean≈0.2806`，`collapse=2/3`
- `v23sman_incidence`：`acc_mean≈0.3153`，`collapse=2/3`
- `v5edge_pool_residual`：`acc_mean≈0.4111`，`collapse=1/3`
- `v16edge_ee_residual_aux_fusion`：`acc_mean≈0.3292`，`collapse=2/3`

就这个诊断集而言：v23 没有优于 v2/v5；它更像一个“结构更贴近 SMAN incidence 思路”的对照实验版本。

