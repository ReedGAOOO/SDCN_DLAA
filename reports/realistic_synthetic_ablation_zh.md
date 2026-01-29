# 更真实合成数据：v2/v3/v4/v5 多轮对比 + v4/v5 消融

> 生成时间：2026-01-27  
> 运行环境：WSL + `conda env: gnn`（GPU: NVIDIA GeForce RTX 4080 SUPER 32GB）

## 1) 复现实验（本次实际跑的配置）

### 数据集 suite
- suite 根目录：`/tmp/sdcn_suite_realistic_v2_20260127_191759`
- 数据集种子：`ds_seed=0/1/2`
- 每个 `ds_seed` 下包含 4 个数据集：
  - `real_social_topics_nonknn`：非空间社交交互风格（edge_attr 含 topic/interaction 统计），`random_k` 图
  - `relational_cycle_nonknn`：关系类型（相对偏移）风格，`random_k` 图
  - `rich_edge_profiles`：含距离/关系 profile，`knn` 图
  - `rich_edge_semantic_only_nonknn`：edge 语义信号强（用于验证“边特征聚类设计是否能生效”），`random_k` 图

生成命令（每个 seed 一次）：
```bash
conda run -n gnn python -B tools/generate_synthetic_suite.py \
  --output_root /tmp/sdcn_suite_realistic_v2_20260127_191759/seed0 \
  --seed 0 \
  --presets rich_edge_profiles,real_social_topics_nonknn,relational_cycle_nonknn,rich_edge_semantic_only_nonknn \
  --random_edges_per_node 0
```

### 训练/评测超参（固定，不做搜索）
用于 v2–v5 的 sweep（每个数据集目录都会跑 `seed=0/1/2`）：
- `SDCN_EPOCHS=120`
- `lr=5e-4`, `dropout=0`, `heads=4`, `n_z=32`
- `SDCN_Q_SOURCE=h4`（用图分支 embedding 做 q/p）
- `SDCN_EDGE_MESSAGE=1`
- `SDCN_FINAL_ASSIGN=p`（用 target distribution `p` 作为最终输出）
- `edge_attr_norm`：
  - `rich_edge_semantic_only_nonknn`: `none`
  - 其他 3 个：`zscore_clip`（clip=5）

模型 sweep 运行命令示例：
```bash
conda run -n gnn python -B tools/sweep_stability.py \
  --data_dir /tmp/sdcn_suite_realistic_v2_20260127_191759/seed0/real_social_topics_nonknn \
  --out_dir /tmp/realistic_benchmark_v2_20260127_191838/models/real_social_topics_nonknn_ds0 \
  --variants v2edge_single_layer,v3edge_cross_layers,v4edge_pool_fusion,v5edge_pool_residual \
  --seeds 0,1,2 \
  --epochs 120 --lrs 5e-4 --dropouts 0.0 --heads 4 --n_z 32 \
  --q_sources h4 --edge_messages 1 \
  --edge_attr_norms zscore_clip \
  --final_assign p
```

Baseline 输出目录：
- baselines：`/tmp/realistic_benchmark_v2_20260127_191838/baselines/*.json`

## 2) 多轮结果（mean ± std, 共 9 次：ds_seed=0/1/2 × seed=0/1/2）

### real_social_topics_nonknn

**Baselines (Top-3 by acc)**

| method | acc |
|---|---:|
| kmeans_edge_mean | 0.4782 ± 0.0524 |
| kmeans_x_edge_mean | 0.4514 ± 0.0534 |
| spectral_node_edge_rbf | 0.4065 ± 0.0340 |

**Models (v2–v5)**

| variant | acc | collapse_rate |
|---|---:|---:|
| v2edge_single_layer | 0.3713 ± 0.0581 | 0.00 |
| v3edge_cross_layers | 0.2958 ± 0.0696 | 1.00 |
| v4edge_pool_fusion | 0.4653 ± 0.0514 | 0.78 |
| v5edge_pool_residual | 0.4477 ± 0.0705 | 0.22 |

结论：v4/v5 接近最强 baseline，但总体仍未超过；v3 在该类数据上几乎必塌缩。

### relational_cycle_nonknn

**Baselines (Top-3 by acc)**

| method | acc |
|---|---:|
| kmeans_x_edge_mean | 0.3130 ± 0.0187 |
| spectral_adj_binary | 0.3111 ± 0.0108 |
| kmeans_x | 0.3106 ± 0.0198 |

**Models (v2–v5)**

| variant | acc | collapse_rate |
|---|---:|---:|
| v2edge_single_layer | 0.2986 ± 0.0157 | 0.00 |
| v3edge_cross_layers | 0.2898 ± 0.0110 | 1.00 |
| v4edge_pool_fusion | 0.3005 ± 0.0185 | 0.00 |
| v5edge_pool_residual | 0.3074 ± 0.0196 | 0.22 |

结论：该数据更像“相对关系约束/同步”问题，当前训练目标下 v4/v5 仅能接近 baseline，且 v3 仍显著不稳定。

### rich_edge_profiles

**Baselines (Top-3 by acc)**

| method | acc |
|---|---:|
| spectral_edge_distance | 0.6821 ± 0.0442 |
| spectral_node_edge_rbf | 0.6630 ± 0.0594 |
| spectral_edge_l2 | 0.6611 ± 0.0614 |

**Models (v2–v5)**

| variant | acc | collapse_rate |
|---|---:|---:|
| v2edge_single_layer | 0.4395 ± 0.0467 | 0.00 |
| v3edge_cross_layers | 0.4080 ± 0.0794 | 0.89 |
| v4edge_pool_fusion | 0.4321 ± 0.0684 | 0.00 |
| v5edge_pool_residual | 0.4265 ± 0.0433 | 0.00 |

结论：谱聚类 baseline 在“距离/几何”主导的数据上非常强（~0.68），目前 v2–v5 难以超过；但 v4/v5 至少能避免 v3 那种高塌缩率。

### rich_edge_semantic_only_nonknn

**Baselines (Top-3 by acc)**

| method | acc |
|---|---:|
| spectral_node_edge_rbf | 0.8148 ± 0.0409 |
| kmeans_x_edge_mean | 0.7698 ± 0.1452 |
| kmeans_edge_mean | 0.7562 ± 0.1301 |

**Models (v2–v5)**

| variant | acc | collapse_rate |
|---|---:|---:|
| v2edge_single_layer | 0.6932 ± 0.0881 | 0.00 |
| v3edge_cross_layers | 0.3790 ± 0.0250 | 0.56 |
| v4edge_pool_fusion | 0.6340 ± 0.1089 | 0.00 |
| v5edge_pool_residual | 0.9179 ± 0.0722 | 0.00 |

结论：在“边语义信号足够强、图结构不靠 KNN”的场景，v5 能明显超过强 baseline（~0.918 vs ~0.815），说明 **v5 的边特征聚类设计在机制上是能生效的**。

## 3) v4/v5 结构消融（seed0 数据集，train seed=0/1/2）

### A) `rich_edge_semantic_only_nonknn`（edge_dim=16, nonknn）

| variant | edge_ee | pool_residual | pool_gate_mode | acc | collapse_rate |
|---|---:|---:|---|---:|---:|
| v4edge_pool_fusion | 1 | 1 | learned | 0.6648 ± 0.1331 | 0.00 |
| v4edge_pool_fusion | 1 | 1 | one | 0.9278 ± 0.0309 | 0.00 |
| v4edge_pool_fusion | 0 | 1 | one | 0.9833 ± 0.0096 | 0.00 |
| v4edge_pool_fusion | 1 | 0 | learned | 0.4111 ± 0.0807 | 0.67 |
| v5edge_pool_residual | 1 | 1 | learned | 0.9648 ± 0.0225 | 0.00 |
| v5edge_pool_residual | 1 | 0 | learned | 0.6167 ± 0.2705 | 0.33 |

要点：
- **关掉 pooling residual（`SDCN_POOL_RESIDUAL=0`）会显著掉点/塌缩**（v4: ~0.66 → ~0.41；v5: ~0.96 → ~0.62，且方差暴增）。
- v4 的 learned gate 在该数据上容易“学不到合适尺度”，固定 `SDCN_POOL_GATE_MODE=one` 明显更稳更强。
- `SDCN_EDGE_EE`（edge↔edge 更新）并非总是收益：在该数据上关掉 edge↔edge + gate=one 反而更好，提示 edge↔edge 可能会把强语义 edge 信号“洗掉”。

### B) `rich_edge_profiles`（edge_dim=16, knn）

| variant | edge_ee | pool_residual | pool_gate_mode | acc | collapse_rate |
|---|---:|---:|---|---:|---:|
| v4edge_pool_fusion | 1 | 1 | learned | 0.4741 ± 0.0210 | 0.00 |
| v4edge_pool_fusion | 1 | 0 | learned | 0.4130 ± 0.0085 | 1.00 |
| v5edge_pool_residual | 1 | 1 | learned | 0.4704 ± 0.0170 | 0.00 |
| v5edge_pool_residual | 1 | 0 | learned | 0.4630 ± 0.0085 | 0.00 |
| v5edge_pool_residual | 0 | 1 | learned | 0.4759 ± 0.0128 | 0.00 |

要点：
- v4：`pool_residual` 是“防塌缩”关键（关掉后 collapse_rate=1.0）。
- v5：对 `pool_residual` 更不敏感（说明 v5 的“raw edge 用于注意力”的路径更稳定），但收益也相对小。

## 4) 总结与建议（如何理解“可靠性/必要性”）

1) **v5 在 edge-driven 场景下更可靠**
   - 在 `rich_edge_semantic_only_nonknn` 上稳定超过强 baseline，且多轮无塌缩。
   - 在 `real_social_topics_nonknn` / `relational_cycle_nonknn` 上虽未超过 baseline，但塌缩率明显低于 v3，整体更稳。

2) **Pooling residual 是结构必要项（尤其对 v4）**
   - 多个数据集的消融都显示：关掉 `SDCN_POOL_RESIDUAL` 会带来显著掉点，甚至直接塌缩（v4 最明显）。

3) **v4 的 gate 与 edge↔edge 更新需要策略化使用**
   - v4 在部分数据上 learned gate 不稳定，推荐优先尝试：`SDCN_POOL_GATE_MODE=one`（把 pooling residual 变成强直通路径）。
   - `SDCN_EDGE_EE` 并非总收益：对“语义 edge 强”的数据可能反而有害；对 profile/几何类数据可能有益。建议作为开关做数据自适应。

4) **为什么难以超过谱聚类（以 `rich_edge_profiles` 为例）**
   - baseline 已经直接用距离构造 affinity 并做 spectral 分解，这是针对该数据分布的“强归纳偏置”；而当前 SDCN 类训练目标更偏向自训练聚类头 + 表征学习，未显式逼近谱分解/图割目标，所以在该类任务上天然吃亏。

## 5) 三条“互补通路”的更严格消融（v5）

为把 v5 的三条通路拆得更干净，我们新增了一个开关：`SDCN_NODE_ATT_EDGE`（是否让 node attention 真的接收 raw edge 作为 `edge_attr` 参与注意力）。

下面基于 `rich_edge_semantic_only_nonknn`（seed=0）做消融（train seed=0/1/2；`SDCN_EDGE_MESSAGE=0`；`SDCN_POOL_GATE_MODE=one`；`SDCN_FINAL_ASSIGN=p`）：

### A) 注意力通路（node_att 是否使用 raw edge）

| 设置 | acc | collapse_rate |
|---|---:|---:|
| `SDCN_NODE_ATT_EDGE=1` | 0.9833 ± 0.0056 | 0.00 |
| `SDCN_NODE_ATT_EDGE=0` | 0.8907 ± 0.0479 | 0.00 |

结论：raw edge 参与注意力能显著提升效果，但即使关掉，v5 仍可依靠 pooling residual 保持较强性能（不塌缩）。

### B) 强基线通路（pool residual）

| 设置 | acc | collapse_rate |
|---|---:|---:|
| `SDCN_POOL_RESIDUAL=1` | 0.9833 ± 0.0056 | 0.00 |
| `SDCN_POOL_RESIDUAL=0` | 0.3444 ± 0.0192 | 1.00 |

结论：在 edge 语义数据上，pool residual 是**结构必要项**（关掉基本直接塌缩）。

### C) 边表征通路（edge↔edge + pool_upd）

| `SDCN_EDGE_EE` | `SDCN_POOL_UPD` | acc | collapse_rate |
|---:|---:|---:|---:|
| 1 | 1 | 0.9833 ± 0.0056 | 0.00 |
| 1 | 0 | 0.9926 ± 0.0032 | 0.00 |
| 0 | 1 | 0.9796 ± 0.0128 | 0.00 |
| 0 | 0 | 0.9926 ± 0.0032 | 0.00 |

结论：在该数据上，edge↔edge 与 “updated edge pooling” 不是关键瓶颈，收益更依赖 node-att(raw edge) + pool residual 这两条主干。
