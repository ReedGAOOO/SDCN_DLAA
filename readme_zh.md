# SDCN_DLAA

SDCN + Dual-Level Attentive Aggregation (DLAA)。本仓库在 PyTorch Geometric 中实现了类似 `SpatialConv` 的节点↔边、边↔边交互式消息传递，用于将边语义（距离/关系强度等）融入节点聚类。

完整版说明见：
- `readme_zh_full.md`（中文）
- `readme_full.md`（EN）

## 项目亮点（创新点简述）

- **边信息融入聚类（SDCN + DLAA）**：通过双层聚合（节点↔边、边↔边）让边语义影响节点表示与聚类。
- **可切换的 SpatialConv 版本**：`v1original`、`v2edge_single_layer`（默认）、`v3edge_cross_layers`、`v4edge_pool_fusion`、`v5edge_pool_residual`，用 `SPATIALCONV_VARIANT` 方便做消融/对比。
- **可选 edge message 注入**：设置 `SDCN_EDGE_MESSAGE=1`，让 `edge_attr` 以“消息内容”参与节点更新（不仅是调注意力权重），适合 node_features 很弱的场景。
- **实验辅助工具**：`SDCN_SEED` / `SDCN_EPOCHS` + `tools/` 的概念/合成数据对比脚本。

## 目录结构

- `sdcn_dlaa_NEW.py`: 主训练/评估入口（SDCN 风格自监督聚类）。
- `DLAA_NEW.py`: DLAA / `SpatialConv` 及其版本实现。
- `preprocess_distance_matrix.py`: 从距离矩阵构建稀疏图并生成边特征。
- `NEWDATA/`: 示例原始数据与预处理输出目录。
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
python test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10
```

常用参数：
- `--heads`: 注意力头数
- `--edge_dim`: 边特征维度（需与预处理一致）
- `--max_edges_per_node`: 控制 edge-to-edge 图的稠密度

## SpatialConv 三个版本（v1/v2/v3）

通过环境变量在 import 时选择（默认：`v2edge_single_layer`）：

```bash
export SPATIALCONV_VARIANT=v2edge_single_layer  # v1original | v2edge_single_layer | v3edge_cross_layers | v4edge_pool_fusion | v5edge_pool_residual
export SDCN_FINAL_ASSIGN=pred                 # pred | q | p（选择最终聚类输出来自哪个头）
export SDCN_SEED=0                              # 可选：复现实验
export SDCN_EPOCHS=30                           # 可选：覆盖训练轮数
export SDCN_EDGE_MESSAGE=1                      # 可选：edge_attr 作为消息内容注入
python test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10 --heads 1
```

版本含义（简述）：
- `v1original`: 旧版基线。
- `v2edge_single_layer`: 小改动修复（保证边特征进注意力，避免 edge 行被“洗掉”）。
- `v3edge_cross_layers`: 将更新后的 edge embedding 作为 `edge_attr` 参与 node attention。

## 其他运行方式

```bash
# Threshold 稀疏化图
python test_sdcn_dlaa_NEW_sparse_threshold.py --data_dir NEWDATA/processed_threshold_0.5
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

示例结果表与更详细的说明见 `readme_zh_full.md`。
