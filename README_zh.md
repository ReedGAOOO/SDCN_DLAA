# SDCN_DLAA

SDCN + Dual-Level Attentive Aggregation (DLAA)。本仓库在 PyTorch Geometric 中实现了类似 `SpatialConv` 的节点↔边、边↔边交互式消息传递，用于将边语义（距离/关系强度等）融入节点聚类。

## 项目亮点（创新点简述）

- **边信息融入聚类（SDCN + DLAA）**：通过双层聚合（节点↔边、边↔边）让边语义影响节点表示与聚类。
- **默认最佳版本（`v5edge_pool_residual`）**：在 edge 语义主导的合成数据上最鲁棒（见 `reports/realistic_synthetic_ablation_zh.md`）。
- **可切换的 SpatialConv 版本**：`v1original`、`v2edge_single_layer`、`v3edge_cross_layers`、`v4edge_pool_fusion`、`v5edge_pool_residual`，用 `SPATIALCONV_VARIANT` 方便做消融/对比。
- **可选 edge message 注入**：设置 `SDCN_EDGE_MESSAGE=1`，让 `edge_attr` 以“消息内容”参与节点更新（不仅是调注意力权重），适合 node_features 很弱的场景。
- **实验辅助工具**：`SDCN_SEED` / `SDCN_EPOCHS` + `tools/` 的概念/合成数据对比脚本。

## 推荐默认配置

- 默认版本：**`v5edge_pool_residual`**（在 edge 语义主导的合成数据上可稳定超过强 baseline，详见 `reports/realistic_synthetic_ablation_zh.md`）。
- 对“边信息驱动聚类”的常用组合：
  - `SDCN_Q_SOURCE=h4`
  - `SDCN_EDGE_MESSAGE=1`
  - `SDCN_FINAL_ASSIGN=p`
  - profile 类边特征建议加 `--edge_attr_norm zscore_clip`（在 `tools/test_conceptual_data.py` / `tools/sweep_stability.py` 中设置）

## v5 结构详解（推荐）

`v5edge_pool_residual` 面向 **“边信息驱动聚类”**：当边承载的语义（关系类型、交互统计、多维 profile）比节点特征更关键时，v5 更容易把边语义“兑现”为节点可聚类的表征。

**核心思路**：保留两条互补路径，分别从不同角度把边信息注入到节点表征里。

**单个 SpatialConv block 的前向流程（示意）**：

```text
输入：x（节点特征）, edge_index, dist_feat（raw edge_attr）, dist_feat_order, edge_to_edge_index

edge_feat_0 = MLP([x_src, x_dst, dist_feat_order])
edge_feat_1 = ee_gat(edge_feat_0, edge_to_edge_index)            # edge↔edge 上下文

node_att    = SGAT(x, edge_index, edge_attr=dist_feat)            # raw edge 直接参与注意力
pooled      = mean_pool(dist_feat) + mean_pool(edge_feat_1)       # edge→node residual（两端点聚合）
node_out    = node_att + sigmoid(gate([node_att, pooled])) * proj(pooled)
```

1) **node attention 直接使用 raw edge（v2 思路）**  
节点更新使用 `SGATLayer(GATConv(edge_dim=...))`，其中 `edge_attr = dist_feat` 直接参与注意力计算，避免 edge embedding 在早期被“洗掉”。

2) **edge↔edge 更新（局部一致性/上下文）**  
边向量先由 `(x_src, x_dst, dist_feat_order)` 初始化，再在 `edge_to_edge_index`（共享端点的边构成的图）上用 `ee_gat` 做更新，捕捉边之间的上下文。

3) **显式 edge→node pooling residual（类似强 baseline，但可学习）**  
把边特征 mean-pool 到节点（两端点都聚合），再用门控融合：

- pooled = mean_pool(raw_edge) + mean_pool(updated_edge)
- node_out = node_att + sigmoid(gate([node_att, pooled])) * proj(pooled)

从消融实验来看，这个 residual 在 edge 语义数据上经常是“结构必要项”（关掉会明显掉点/塌缩）。

**默认选择**
- 当前仓库默认就是 `v5edge_pool_residual`（见 `DLAA_NEW.py`），也可手动指定：
  - `export SPATIALCONV_VARIANT=v5edge_pool_residual`

**v5 推荐组合**
- `SDCN_Q_SOURCE=h4`（用图分支 embedding 做 q/p 自训练）
- `SDCN_FINAL_ASSIGN=p`（当 `pred` 头落后时用 p 作为最终聚类输出）
- `SDCN_EDGE_MESSAGE=1`（让 edge_attr 以“消息内容”额外注入）

**结构消融开关（研究用）**
- `SDCN_POOL_RESIDUAL=0/1`（关闭/开启 pooling residual）
- `SDCN_EDGE_EE=0/1`（关闭/开启 edge↔edge 更新）
- `SDCN_POOL_GATE_MODE=learned|one|zero`（门控行为）

## 目录结构

- `sdcn_dlaa_NEW.py`: 主训练/评估入口（SDCN 风格自监督聚类）。
- `DLAA_NEW.py`: DLAA / `SpatialConv` 及其版本实现。
- `preprocess_distance_matrix.py`: 从距离矩阵构建稀疏图并生成边特征。
- `NEWDATA/`: 示例原始数据与预处理输出目录。
- `experiments/`: 可直接运行的入口脚本（Sparse KNN / Threshold 等）。
- `archive/`: 已归档的实验模型与脚本（AMP/hetero/hiddensize），保留以便回溯。
- `tools/`: 概念数据生成与三版本对比脚本。

## 模型结构（概览）

- **AE 自编码器**：从节点特征学习内容表示。
- **图编码器（SpatialConv 堆叠）**：利用 `edge_index` + 距离类边特征做结构聚合。
- **聚类头（SDCN）**：Student-t 软分配 + 目标分布（sharpen）自监督优化。

## 快速开始

### 1) 数据预处理（KNN 稀疏图）

```bash
python preprocess_distance_matrix.py --output_dir NEWDATA/processed_knn_k10 --method knn --k 10
```

默认读取：
- 节点特征：`NEWDATA/X_simplize.CSV`
- 距离矩阵：`NEWDATA/A.csv`

需要自定义时用 `--node_features` / `--distance_matrix` 指定。

### 2) 运行（Sparse KNN）

```bash
python experiments/test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10
```

常用参数：
- `--heads`: 注意力头数
- `--edge_dim`: 边特征维度（需与预处理一致）
- `--max_edges_per_node`: 控制 edge-to-edge 图的稠密度

## SpatialConv 三个版本（v1/v2/v3）

通过环境变量在 import 时选择（默认：`v5edge_pool_residual`）：

```bash
export SPATIALCONV_VARIANT=v5edge_pool_residual  # v1original | v2edge_single_layer | v3edge_cross_layers | v4edge_pool_fusion | v5edge_pool_residual
export SDCN_Q_SOURCE=h4                          # z | h4 | fused
export SDCN_FINAL_ASSIGN=p                       # pred | q | p（选择最终聚类输出来自哪个头）
export SDCN_SEED=0                              # 可选：复现实验
export SDCN_EPOCHS=30                           # 可选：覆盖训练轮数
export SDCN_EDGE_MESSAGE=1                      # 可选：edge_attr 作为消息内容注入
python experiments/test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10 --heads 1
```

版本含义（简述）：
- `v1original`: 旧版基线(初版设计)。
- `v2edge_single_layer`: 小改动修复（保证边特征进注意力，避免 edge 行被“洗掉”）。
- `v3edge_cross_layers`: 将更新后的 edge embedding 作为 `edge_attr` 参与 node attention。
- `v4edge_pool_fusion`: 在 v3 基础上加入显式的 edge→node pooling residual（带门控）。
- `v5edge_pool_residual`（推荐）: 延续 v2 思路（node attention 用 raw edge），同时加入 edge→node pooling residual；在 edge 语义主导任务上更鲁棒。

## 其他运行方式

```bash
# Threshold 稀疏化图
python experiments/test_sdcn_dlaa_NEW_sparse_threshold.py --data_dir NEWDATA/processed_threshold_0.5
```

## 概念数据对比（可选）

```bash
python tools/generate_conceptual_data.py --output_dir /tmp/sdcn_dlaa_concept_data --seed 0
python tools/compare_spatialconv_variants.py \
  --data_dir /tmp/sdcn_dlaa_concept_data \
  --out_dir /tmp/sdcn_dlaa_variant_compare \
  --seeds 0,1,2 \
  --epochs 30 \
  --variants v1original,v2edge_single_layer,v3edge_cross_layers,v4edge_pool_fusion,v5edge_pool_residual \
  --heads 1
```

模拟数据套件（多数据集 + 传统 baseline 对比）：

```bash
python tools/generate_synthetic_suite.py --output_root /tmp/sdcn_dlaa_synth_suite --seed 0
python tools/benchmark_synthetic_suite.py \
  --suite_dir /tmp/sdcn_dlaa_synth_suite \
  --out_dir /tmp/sdcn_dlaa_synth_results \
  --seeds 0,1,2 \
  --epochs 30 \
  --variants v2edge_single_layer,v3edge_cross_layers \
  --baselines kmeans_x,spectral_adj_binary,spectral_edge_distance \
  --heads 1
```

示例结果表与更详细的说明已整合在本文档中。
