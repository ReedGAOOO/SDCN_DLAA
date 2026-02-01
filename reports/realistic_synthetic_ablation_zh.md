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

## 6) 让 edge↔edge “稳定正向”地帮助聚类：加噪诊断 + 新结构尝试（v6/v7/v8）

> 背景：前面多轮结果显示 edge↔edge（`SDCN_EDGE_EE`）并非总是收益，有些数据上甚至会“洗掉”强 edge 语义。  
> 这里做一个更可定位的诊断：**人为给 edge_attr 加高斯噪声**（`--edge_noise_std`），看哪些结构能把 edge↔edge 变成稳定的“去噪/正则”而不是不稳定的“额外自由度”。

### 6.1 数据与固定设置

数据集（新生成，seed=0）：
- `real_social_topics_nonknn`：`/tmp/sdcn_dlaa_real_suite_20260130_001308/real_social_topics_nonknn`（N=240, edge_dim=32）

加噪方式：
- 在 `tools/test_conceptual_data.py` 中对 `edge_attr` 做：`edge_attr += Normal(0, edge_noise_std)`（seeded by `SDCN_SEED`）

固定超参（不做搜索）：
- `SDCN_EPOCHS=60`
- `lr=1e-3`, `dropout=0.2`, `heads=1`, `n_z=10`
- `SDCN_Q_SOURCE=h4`, `SDCN_EDGE_MESSAGE=0`, `SDCN_FINAL_ASSIGN=p`
- `edge_attr_norm=zscore_clip`（clip=5）
- `SDCN_POOL_RESIDUAL=1`, `SDCN_POOL_RAW=1`, `SDCN_POOL_UPD=1`, `SDCN_POOL_GATE_MODE=one`
- train seed：`0/1/2`
- `edge_noise_std ∈ {0, 0.2, 0.5}`

运行命令（示例）：
```bash
conda run -n gnn python -B tools/sweep_stability.py \
  --data_dir /tmp/sdcn_dlaa_real_suite_20260130_001308/real_social_topics_nonknn \
  --out_dir /tmp/sdcn_dlaa_v5v8_noise_sweep_xxx \
  --variants v5edge_pool_residual,v8edge_denoise_attr \
  --seeds 0,1,2 \
  --epochs 60 --lrs 1e-3 --dropouts 0.2 --heads 1 --n_z 10 \
  --q_sources h4 --edge_messages 0 --node_att_edges 1 --edge_ees 0,1 \
  --edge_attr_norms zscore_clip --edge_attr_clip 5 \
  --edge_noise_stds 0,0.2,0.5 \
  --final_assign p \
  --pool_residuals 1 --pool_raws 1 --pool_upds 1 --pool_gate_modes one
```

### 6.2 v6（EE-aux）：“把 edge↔edge 变成可优化目标”但存在不稳定反馈

v6 思路：在 v5 三通路基础上加一个 edge-level head（within-edge logit），并在训练中加入
`BCE(edge_logit, same_prob(p_u,p_v))`（可选再加 edge↔edge 平滑），试图让 edge↔edge “被聚类目标驱动”。

本次设置（固定）：
- `SPATIALCONV_VARIANT=v6edge_ee_aux`
- `SDCN_EDGE_AUX_WEIGHT=0.1`
- `SDCN_EDGE_AUX_SMOOTH_WEIGHT=0.05`
- `SDCN_EDGE_AUX_WARMUP_EPOCHS=20`

结果（mean±std, seed=0/1/2）：

| variant | edge_ee | edge_noise_std | acc | collapse_final_rate |
|---|---:|---:|---:|---:|
| v5 | 0 | 0.0 | 0.4014 ± 0.0171 | 0.00 |
| v5 | 0 | 0.2 | 0.3819 ± 0.0239 | 0.00 |
| v5 | 0 | 0.5 | 0.3861 ± 0.0142 | 0.00 |
| v5 | 1 | 0.0 | 0.4167 ± 0.0059 | 0.00 |
| v5 | 1 | 0.2 | 0.4181 ± 0.0193 | 0.00 |
| v5 | 1 | 0.5 | 0.4083 ± 0.0508 | 0.00 |
| v6 | 0 | 0.0 | 0.4319 ± 0.0241 | 0.00 |
| v6 | 0 | 0.2 | 0.4264 ± 0.0109 | 0.00 |
| v6 | 0 | 0.5 | 0.4264 ± 0.0187 | 0.00 |
| v6 | 1 | 0.0 | 0.4014 ± 0.0277 | 0.33 |
| v6 | 1 | 0.2 | 0.4014 ± 0.0193 | 0.33 |
| v6 | 1 | 0.5 | 0.3958 ± 0.0223 | 0.33 |

解读（机制层面）：
- v6 在该数据上呈现 **“edge_ee=1 时更不稳”**：出现 `collapse_final_rate≈0.33`，且均值不如 `edge_ee=0`。
- 直觉原因：`same_prob(p_u,p_v)` 来自当前聚类 target `p`（即便 `detach`），当 `p` 偏尖或走向塌缩时，会把 edge head 推向“所有边都像 within-edge”（或相反），再通过 `pool_upd`/edge 表征反馈到 node，形成**正反馈环**。
- 结论：**“给 edge↔edge 直接挂聚类伪标签”不是一个天然稳定的做法**，需要更强的防反馈设计（例如用更软的 `q` 作为 target、温度/熵约束、负采样对比等）。

### 6.3 v7（edge_attr fusion）：让 edge↔edge 直接进入注意力，但收益依赖尺度

v7 思路：仍保留 raw edge 用于 node attention，同时把 edge↔edge 学到的 `edge_feat_1`（layernorm 后）以加性方式融入 `edge_attr`：
`edge_attr_att = dist_feat + fuse_scale * norm(edge_feat_1)`，使 edge↔edge 影响注意力权重，而不是只走 pooling。

本次设置：
- `SPATIALCONV_VARIANT=v7edge_attr_fusion`
- `SDCN_EDGE_ATTR_FUSE=1`
- `SDCN_EDGE_ATTR_FUSE_SCALE=0.5`
- `SDCN_EDGE_FUSE_NORM=1`

结果（mean±std, seed=0/1/2）：

| variant | edge_ee | edge_noise_std | acc |
|---|---:|---:|---:|
| v5 | 1 | 0.2 | 0.4097 ± 0.0275 |
| v7 | 1 | 0.2 | 0.4208 ± 0.0207 |
| v5 | 1 | 0.5 | 0.4181 ± 0.0425 |
| v7 | 1 | 0.5 | 0.3944 ± 0.0208 |

解读：
- 在中等噪声（0.2）下 v7 有小幅收益，但在更大噪声（0.5）下反而变差，说明**“把 edge↔edge 表征直接加进注意力”对尺度/归一化更敏感**（需要进一步做 fuse_scale 的搜索与 warmup/门控）。

### 6.4 v8（edge-edge denoise）：把 edge↔edge 明确定位成“对 raw edge_attr 去噪”

v8 思路：不再额外造“边语义 head”，而是把 edge↔edge 作为 **edge_attr 的去噪器/平滑器**：
- `edge_raw = proj(dist_feat)`
- `edge_upd = ee_gat(edge_raw, edge_to_edge_index)`
- `edge_denoised = (1-α)*edge_raw + α*edge_upd`
- node attention / pooling 用 `edge_denoised`

本次设置：
- `SPATIALCONV_VARIANT=v8edge_denoise_attr`
- `SDCN_EDGE_DENOISE_ALPHA=0.5`
- `SDCN_EDGE_DENOISE_NORM=1`

结果（mean±std, seed=0/1/2）：

| variant | edge_ee | edge_noise_std | acc |
|---|---:|---:|---:|
| v8 | 0 | 0.0 | 0.3667 ± 0.0136 |
| v8 | 1 | 0.0 | 0.4056 ± 0.0341 |
| v8 | 0 | 0.2 | 0.3681 ± 0.0187 |
| v8 | 1 | 0.2 | 0.4097 ± 0.0399 |
| v8 | 0 | 0.5 | 0.3708 ± 0.0238 |
| v8 | 1 | 0.5 | 0.3958 ± 0.0368 |

解读（关键结论）：
- 在 v8 这个“把 edge↔edge 明确当成去噪器”的结构里，`edge_ee=1` **在 3 个噪声强度下都稳定优于** `edge_ee=0`（约 +0.02～+0.04）。
- 但 v8 的整体均值仍未超过 v5：说明该数据上“单靠对 raw edge_attr 平滑”还不够，需要把 **node-pair 上下文 + edge 去噪** 结合起来（下一步可做 v8 的增强版：edge_raw = f(dist_feat, x_src, x_dst) 再去噪）。

### 6.5 小结：edge↔edge 想要稳定正向，最好扮演“保守的正则/去噪器”

从这组加噪诊断可以得到一个更清晰的机制判断：
1) **把 edge↔edge 直接挂到聚类伪标签（v6）容易形成正反馈，稳定性差**（尤其当 `p/q` 本身会尖化/塌缩）。
2) **把 edge↔edge 当成对 edge_attr 的“去噪/平滑/一致性正则”（v8）更稳定**，且在该结构内确实能观察到 edge↔edge 的稳定正向贡献。
3) 如果希望它在整体上超过 v5/baseline，下一步需要把 “edge 去噪” 与 “edge 语义表达（node-pair 上下文）” 统一到同一条信息流里，而不是只做平滑或只做语义 head。

## 7) 顺着结论继续：把“node-pair 上下文 + edge 去噪”统一起来（v9/v10）与更广泛噪声诊断

> 目标：验证“edge↔edge 作为保守去噪器”在更复杂的信息流里是否仍然稳定正向；并定位它在什么噪声强度/数据类型下值得开启。

### 7.1 v9（edge_context_denoise）：直接在 edge_raw 里混入 node-pair 上下文会不稳定

我们先尝试 v9：`edge_raw = f(dist_feat, x_src, x_dst)` 后在 edge↔edge 图上更新并混合。

初版 v9（`relu(Wd(dist)+Wn(nodes))`）在 `real_social_topics_nonknn` 上出现大量塌缩（acc≈0.25），说明：
- **把 node-pair 上下文以“强非线性”直接并入 edge_raw，会引入很大的自由度**；
- 再叠加 edge↔edge 更新，容易让 edge_attr 的尺度/分布失控，从而触发 SDCN 自训练头塌缩。

因此我们把 v9 改为“更保守”的形式（已合入代码）：  
`edge_raw = dist_feat + s * Wn([x_src, x_dst])`，其中 `s` 是可学习标量且初始化为 0（训练初期退化为 v8 风格，更稳）。

### 7.2 v10（base denoise + context）：让 edge↔edge 专注去噪，再叠加 node-pair 上下文

v10 的关键改动是把“去噪”和“语义表达”拆开再合并：
1) 先只对 raw edge_attr 做 edge↔edge 去噪：`base_denoised = (1-α)*dist + α*ee(dist)`
2) 再加 node-pair 上下文残差：`edge_attr = norm(base_denoised + s*Wn([x_src,x_dst]))`

这样 edge↔edge 的职责更清晰：**它只做“保守的平滑/去噪器”**，不会被 node_ctx 的大自由度“拖进来”。

### 7.3 多数据集 × 多噪声强度（alpha=0.1）对比：v5 vs v10

数据（seed=0）：
- `/tmp/sdcn_dlaa_noise_suite_20260130_013834/{real_social_topics_nonknn,relational_cycle_nonknn,rich_edge_profiles}`

固定设置：
- `SDCN_EPOCHS=60`, `lr=1e-3`, `dropout=0.2`, `heads=1`, `n_z=10`
- `SDCN_Q_SOURCE=h4`, `SDCN_EDGE_MESSAGE=0`, `SDCN_FINAL_ASSIGN=p`
- `edge_attr_norm=zscore_clip`（clip=5）
- `SDCN_POOL_RESIDUAL=1`, `SDCN_POOL_RAW=1`, `SDCN_POOL_UPD=1`, `SDCN_POOL_GATE_MODE=one`
- `SDCN_EDGE_DENOISE_ALPHA=0.1`（“保守去噪”）
- train seed：0/1/2
- `edge_noise_std ∈ {0, 0.2, 0.5, 1.0}`

**A) real_social_topics_nonknn（edge_dim=32）**

| variant | edge_ee | noise=0.0 | noise=0.5 | noise=1.0 |
|---|---:|---:|---:|---:|
| v5 | 0 | 0.3958 ± 0.0156 | 0.3736 ± 0.0199 | 0.3833 ± 0.0102 |
| v5 | 1 | 0.4139 ± 0.0039 | 0.4097 ± 0.0483 | 0.4000 ± 0.0189 |
| v10 | 0 | 0.3694 ± 0.0264 | 0.3597 ± 0.0216 | **0.4083 ± 0.0123** |
| v10 | 1 | 0.3639 ± 0.0161 | 0.3875 ± 0.0148 | **0.4028 ± 0.0193** |

解读：
- 在低/中噪声下 v5 更强；但在高噪声（1.0）下 v10 明显更稳更强，符合“去噪器”定位。
- v10 中 `edge_ee=1` 的收益不总是大，但在高噪声下至少不再系统性伤害（相比 v6/v7 那种不稳定反馈）。

**B) relational_cycle_nonknn（edge_dim=16）**

| variant | edge_ee | noise=0.0 | noise=0.5 | noise=1.0 |
|---|---:|---:|---:|---:|
| v5 | 0 | 0.3153 ± 0.0119 | 0.3069 ± 0.0406 | 0.3014 ± 0.0175 |
| v10 | 0 | **0.3333 ± 0.0090** | **0.3236 ± 0.0039** | 0.2958 ± 0.0059 |
| v10 | 1 | **0.3361 ± 0.0052** | 0.3083 ± 0.0123 | 0.2931 ± 0.0052 |

解读：
- 该数据整体更难（acc 低），但 v10 在低噪声下略有收益；高噪声时所有方法都下降。

**C) rich_edge_profiles（edge_dim=16, knn）**

| variant | edge_ee | noise=0.0 | noise=0.5 | noise=1.0 |
|---|---:|---:|---:|---:|
| v5 | 0 | **0.4833 ± 0.0045** | **0.4519 ± 0.0210** | **0.4741 ± 0.0139** |
| v10 | 0 | 0.4537 ± 0.0052 | 0.4426 ± 0.0069 | 0.4537 ± 0.0094 |
| v10 | 1 | 0.4611 ± 0.0045 | 0.4444 ± 0.0120 | 0.4370 ± 0.0114 |

解读：
- 在 profile/几何类边特征上，v5 的归纳偏置更对味，v10 的“去噪 + 上下文”并不占优。

### 7.4 当前阶段可得的操作性结论

1) **edge↔edge 作为“保守去噪器”的定位更接近正确答案**：v10 在高噪声边特征下能显著改善（至少在 `real_social_topics_nonknn` 上验证了这一点）。
2) **不要让 edge↔edge 过度介入“语义表达自由度”**：直接对 `f(dist,x_src,x_dst)` 做 edge↔edge（v9 初版）会不稳定。
3) **下一步实验建议（更有可能“稳定超越 v5/baseline”）**
   - 对 v10 做 `α` 随噪声自适应或 warmup（小 α 更像正则；大 α 更像重写 edge_attr，风险大）。
   - 增加“edge 置信度/噪声估计”门控：只对低置信度边更依赖 ee 去噪。
   - 把 baseline（kmeans_edge_mean）也加噪做对照：验证 v10 的提升是否来自“对 edge 噪声更鲁棒”的真实增益。

### 7.5 继续往前推：门控/相似度约束，让 edge↔edge 更“保守”

#### A) v11（adaptive_denoise）：按 edge↔edge 局部不一致性自适应去噪强度

v11 思路：对每条边计算在 edge↔edge 邻域内的局部不一致性 `inco_i`，把它映射成 gate（不一致性越大→去噪越强）：
`alpha_i = alpha * sigmoid(beta * (inco_i/mean(inco) - 1))`。

在 `real_social_topics_nonknn` 上（`alpha=0.5, beta=5.0`）我们观察到：
- v11 在 **高噪声（noise=1.0）** 时可达 `acc≈0.41`（且不塌缩），说明“edge 去噪/上下文残差”在极噪 edge_attr 下确实能提供鲁棒性收益。
- 但 **v11 并没有呈现稳定的 `edge_ee=1 > edge_ee=0`**：中低噪声下开启 edge↔edge 仍可能不收益，说明 gate 仍不足以避免“混入不相似边”。

解释：即便有 gate，如果 edge↔edge 邻域本身包含大量“语义不相似”的边（random_k 图的 line-graph 更容易发生），那么 ee 去噪仍难成为净收益。

#### B) Baselines 加噪对照：强 baseline 对噪声并不总敏感

为避免“模型抬升只是 baseline 同步掉点”的错觉，我们对 baselines 也加入了相同的
`edge_attr_norm=zscore_clip` + `edge_noise_std`。

以 `real_social_topics_nonknn` 为例（seed=0，baseline seed=0/1/2）：
- `kmeans_edge_mean` 在 `noise=0/0.2/0.5/1.0` 上 acc 仍约 `0.44~0.48`

这意味着：在这套设置下，baseline 并没有因为简单加噪而全面失效；模型若要“战胜 baseline”，需要在 **信息流上更有效地利用结构/上下文**，而不是只做平滑。

#### C) v12（similarity_denoise）：只混合“相似边”，避免 ee 误伤

v12 思路：在 edge↔edge 图上用非参数相似度权重（RBF）做加权平滑：
`w_ij = exp(-gamma * ||e_i - e_j||^2)`，只把“相似边”混合进来，再用 `alpha` 做残差混合。

在 `real_social_topics_nonknn` 上做快速验证（`alpha=0.1, gamma=1.0`，noise=0/0.5/1.0）：
- v12 在 **高噪声（noise=1.0）** 下达到 `acc≈0.415`，并出现 `edge_ee=1` 对 v12 的小幅正增益；
- 低噪声时 `edge_ee=1` 仍可能不如 `edge_ee=0`，符合“ee 去噪是噪声条件下的工具，而不是永远开着更好”。

这条结果更接近我们要的机制链条：
- edge↔edge 的正向贡献来自“只在相似边之间做保守平滑”；
- 当 edge↔edge 邻域混杂时（random_k），必须加相似度约束/门控，否则容易把信号洗掉。

**v12 网格结果摘要（train seed=0/1/2；edge_noise_std=0/0.2/0.5/1.0；从 `alpha∈{0.05,0.1,0.2}`、`gamma∈{0.5,1,2}` 里选 `edge_ee=1` 的最优均值）：**

| dataset | noise | v5 (`edge_ee=1`) acc | v12 best (`edge_ee=1`) acc | best (alpha,gamma) |
|---|---:|---:|---:|---|
| real_social_topics_nonknn | 0.0 | 0.4028 ± 0.0137 | 0.3875 ± 0.0068 | (0.1, 0.5) |
| real_social_topics_nonknn | 0.2 | 0.4278 ± 0.0086 | 0.3750 ± 0.0090 | (0.1, 2.0) |
| real_social_topics_nonknn | 0.5 | 0.4139 ± 0.0327 | 0.3931 ± 0.0104 | (0.2, 2.0) |
| real_social_topics_nonknn | 1.0 | 0.3986 ± 0.0316 | **0.4222 ± 0.0039** | (0.1, 0.5) |
| relational_cycle_nonknn | 0.0 | 0.3069 ± 0.0079 | **0.3431 ± 0.0187** | (0.2, 2.0) |
| relational_cycle_nonknn | 0.2 | 0.3139 ± 0.0109 | **0.3431 ± 0.0071** | (0.2, 0.5) |
| relational_cycle_nonknn | 0.5 | 0.3014 ± 0.0086 | **0.3125 ± 0.0034** | (0.2, 2.0) |
| relational_cycle_nonknn | 1.0 | 0.3028 ± 0.0175 | **0.3139 ± 0.0142** | (0.1, 2.0) |
| rich_edge_profiles | 0.0 | 0.4500 ± 0.0164 | 0.4667 ± 0.0045 | (0.2, 0.5) |
| rich_edge_profiles | 0.2 | 0.4444 ± 0.0120 | 0.4722 ± 0.0091 | (0.2, 2.0) |
| rich_edge_profiles | 0.5 | 0.4519 ± 0.0139 | 0.4722 ± 0.0198 | (0.05, 1.0) |
| rich_edge_profiles | 1.0 | 0.4463 ± 0.0114 | 0.4926 ± 0.0250 | (0.1, 0.5) |

结论（更具体、更可操作）：
- v12 的收益具有明显的**条件性**：在 `real_social_topics_nonknn` 上，只有高噪声（1.0）出现稳定提升；低噪声时反而可能“过平滑/误混合”。
- `relational_cycle_nonknn` 上 v12 在多噪声强度下均优于 v5（但整体 acc 仍偏低，任务更难）。
- `rich_edge_profiles` 上 v12 能达到不错的 acc，但 v5 在 `edge_ee=0` 下仍更强（说明 profile/几何类任务上，v5 的归纳偏置更契合）。

### 7.6（先做 2）把 edge↔edge 的“邻域”也做对：`SDCN_EE_GRAPH=edge_sim`

前面 v11/v12 的分析反复出现一个核心问题：**edge↔edge 的邻域如果是“混杂的”，那么任何 ee 去噪/平滑都可能变成误伤**。

原实现的 `edge_to_edge_index` 是“共享端点（incidence）”构图：两条边只要共享一个 node 就会连起来。  
这对分子图/几何图往往合理，但对 random-k 或关系边语义很复杂的数据，line-graph 邻域会塞进大量“语义不相似边”。

因此我在代码里新增了一个可选开关：直接用 edge_attr 的相似度来建 edge↔edge 图（top-k cosine），从源头减少邻域混杂：
- `SDCN_EE_GRAPH=incidence|edge_sim|none`
- `SDCN_EE_TOPK`：edge_sim 的 top-k（默认用 `max_edges_per_node`）
- `SDCN_EE_SIM_MAX_EDGES`：当 E 太大时自动回退到 incidence（避免 O(E^2)）
- `SDCN_EE_SIM_CHUNK`：分块计算 cosine top-k（控制显存/内存）

#### A/B：incidence vs edge_sim（v12，固定 α=0.1, γ=0.5；seed=0/1）

目录：
- real_social：`/tmp/sdcn_dlaa_eegraph_ab2_20260130_114147/`
- cycle：`/tmp/sdcn_dlaa_eegraph_ab2_20260130_cycle_114308/`

| dataset | noise | v12 + incidence | v12 + edge_sim |
|---|---:|---:|---:|
| real_social_topics_nonknn | 0.5 | 0.4750 ± 0.0000 | **0.4833 ± 0.0042** |
| real_social_topics_nonknn | 1.0 | **0.4563 ± 0.0021** | 0.4521 ± 0.0021 |
| relational_cycle_nonknn | 0.5 | **0.3063 ± 0.0021** | 0.3021 ± 0.0021 |
| relational_cycle_nonknn | 1.0 | 0.3104 ± 0.0063 | **0.3188 ± 0.0021** |

解读（非常关键）：
- “把邻域做对”并不会无条件提升：它更像是 **把 ee 从‘可能误伤’变成‘更可控的正则工具’**。
- edge_sim 在某些条件（例如 cycle 的高噪声）能带来更明显的稳定收益；但在某些条件下也可能略弱（例如 real_social 的 noise=1.0）。

### 7.7（再做 1）最终冲刺：high-noise 更保守 α + 更大 γ + baseline 同噪声对照

#### A) v12（edge_sim）在 high-noise 的 best-of-grid（seed=0/1/2）

目录：`/tmp/sdcn_dlaa_v12_sprint_20260130_114504/`

- real_social（pool=none）：
  - noise=0.5：best `α=0.05, γ=2` → **0.4694 ± 0.0052**
  - noise=1.0：best `α=0.02, γ=1` → **0.4264 ± 0.0187**

结论：即使把 `α` 调得更保守、把 `γ` 扫到更大，real_social 上仍然无法逼近最强 baseline（见下一节）。

#### B) baselines 同噪声对照（zscore_clip + noise；seed=0/1/2）

real_social baselines 目录：`/tmp/sdcn_dlaa_baselines_real_social_20260130_120250/`

- noise=0.5：best_overall 基本由 `kmeans_edge_mean` 取得 → **0.5083 ± 0.0295**
- noise=1.0：best_overall 仍由 `kmeans_edge_mean` 取得 → **0.4931 ± 0.0264**

这说明：在 real_social 这一类数据上，baseline 的“直接池化 edge_attr 再聚类”非常强，  
模型若想稳定超越，必须让 SDCN 的表征学习真正“吃到”同等强度的 edge 统计信号。

#### C) 给 AE/SDCN 更强的 edge 可见性：`node_edge_pool=mean_concat`（探索性）

动机：baseline 的最强项（`kmeans_edge_mean`）几乎只用 edge 信号；而 SDCN 的 AE/q 主要从 node features 学。  
因此我们尝试在输入侧把 “per-node mean(edge_attr)” 拼进 x（`--node_edge_pool mean_concat`），让 AE/q 也能直接看到 edge 统计量。

探索目录（seed=0/1，部分网格）：`/tmp/sdcn_dlaa_v12_pool_sprint_20260130_223842/`

观察：
- noise=1.0 下，`mean_concat` 的 best mean（seed=0/1）可到 **≈0.4792**（例如 `α=0.05, γ=8`），明显强于 pool=none 的 ≈0.45。
- 但与 baseline 的 **0.4931 ± 0.0264** 仍有差距：说明“把 edge 统计量喂给 AE”能显著缩小差距，但还不足以稳定反超。

#### D) relational_cycle：v12 在 high-noise 已接近/达到 baseline

- v12（edge_sim）目录：`/tmp/sdcn_dlaa_v12_sprint_cycle_20260130_120341/`
  - noise=1.0：best 可到 **0.3208**（例如 `α=0.05, γ=8`，seed=0/1/2）
- cycle baselines 目录：`/tmp/sdcn_dlaa_baselines_cycle_20260130_223736/`
  - noise=1.0：best_overall mean 约 **0.3208 ± 0.0034**

这意味着：在 cycle 这一类“结构更占主导”的数据上，edge↔edge（作为保守去噪/相似度约束）能把模型推到接近 baseline 的水平；  
但在 real_social 这种“edge 统计本身就近似可线性分簇”的数据上，baseline 仍更像一个强先验/上界。

> 代码备注（影响回溯对比）：近期还修正了 v1/v2 的 edge↔edge index offset（避免把 edge↔edge 图错误地作用到 node rows），以及把 `initial_edge_proj` 改成 `Identity()` 以保证 edge_attr 在训练中不漂移（否则 edge_sim 的缓存/邻域会被破坏）。

### 7.8（延续 2→1 的思路）把“邻域做对”继续往前推：hybrid 邻域 + 更强的 edge 统计通路（v14）

在 7.6/7.7 之后，我们继续沿着“**不要把 edge↔edge 绑到伪标签形成正反馈**，而是把它当成 **保守的去噪/正则器**”这个主线推进。

#### A) `SDCN_EE_GRAPH=hybrid`：incidence ∪ edge_sim（更稳的邻域）

实现：`hybrid = incidence ∪ edge_sim`（去重后合并），动机是：
- incidence：给“结构邻域”，避免 edge_sim 把相似边连到图上很远的区域导致“过全局化”
- edge_sim：给“语义相似邻域”，避免 incidence 邻域里混入大量不相似边导致“误混合”

（代码：`sdcn_dlaa_NEW.py` 支持 `SDCN_EE_GRAPH=hybrid`）

#### B) v13（context-key sim denoise）对比：hybrid 往往更优，但跨 seed 方差仍大

目录：
- real_social：`/tmp/sdcn_dlaa_eegraph_v12v13_20260131_010447_real_social/`
- cycle：`/tmp/sdcn_dlaa_eegraph_v12v13_20260131_010802_cycle/`

设置：noise=1.0，`α=0.05, γ=8`，seed=0/1/2，pool=none

| dataset | variant | ee_graph | acc (mean ± std) |
|---|---|---|---:|
| real_social_topics_nonknn | v13 | incidence | 0.4236 ± 0.0275 |
| real_social_topics_nonknn | v13 | edge_sim | 0.4208 ± 0.0272 |
| real_social_topics_nonknn | v13 | **hybrid** | **0.4375 ± 0.0345** |
| relational_cycle_nonknn | v12 | incidence | 0.3014 ± 0.0071 |
| relational_cycle_nonknn | v12 | edge_sim | 0.3000 ± 0.0090 |
| relational_cycle_nonknn | v12 | **hybrid** | **0.3139 ± 0.0039** |

解读：
- hybrid 在两个数据上都更有机会成为“稳定正则工具”（尤其是 cycle）
- 但在 real_social 上，v12/v13 仍出现明显 seed 方差，说明“**SDCN 的训练策略/表征通路**”仍可能把强 edge 统计信号吃掉（这是无法稳定超过 `kmeans_edge_mean` 的关键原因）

#### C) 关键改动：v14（edge-pool concat fusion）让“强基线通路”成为可分离子空间

直觉：7.7 已经观察到 “把 per-node mean(edge_attr) 喂给 AE（mean_concat）” 能显著缩小差距。  
但这属于“输入侧 hack”。更合理的是在模型内部把“edge 统计量”作为一条稳定的信息通路保留。

因此做 v14：
- edge↔edge：仍按 v13 的 **相似度加权去噪**（保守）
- edge→node：不再 `node_att + gate*pooled`（容易被 gate 或表征混合吃掉），而是
  - `node_out = W([node_att, pool_raw, pool_upd])`（concat 再线性融合）
  - 并把 `W` 初始化成接近加法（起步就像 “node_att + pool_raw + pool_upd”）

这使得 “pool_raw/pool_upd” 成为 **显式可见的子空间**，更接近 `kmeans_edge_mean` 的归纳偏置。

#### D) v14 在 real_social（noise=1.0）显著提升“跨 seed 的均值与稳定性”，并观察到 edge↔edge 的稳定正增益

v14 网格（`α∈{0.02,0.05,0.1}, γ∈{2,4,8,16}`，`SDCN_EE_GRAPH=hybrid`）：
- 目录：`/tmp/sdcn_dlaa_v14_sprint_20260131_015837_real_social_noise1/`
- best（mean over seed=0/1/2）：`α=0.05, γ=2` → **0.4722 ± 0.0205**

edge↔edge 消融（同一设置 `α=0.05, γ=2, hybrid`）：
- 目录：`/tmp/sdcn_dlaa_v14_edgeee_ab_20260131_020621_real_social_noise1/`
- `edge_ee=0`：0.4389 ± 0.0052
- `edge_ee=1`：**0.4625 ± 0.0189**（≈ +0.024）

ee_graph 消融（同一设置 `α=0.05, γ=2, edge_ee=1`）：
- 目录：`/tmp/sdcn_dlaa_v14_eegraph_ab_20260131_020739_real_social_noise1/`
- incidence：0.4333 ± 0.0328
- edge_sim：0.4361 ± 0.0379
- **hybrid：0.4694 ± 0.0208**

结论（对“edge↔edge 是否真的生效”的直接回答）：
- 在 v14 这种“**edge↔edge=保守去噪器 + 强基线通路=concat 可见子空间**”结构里，
  - `edge_ee=1` 对均值有 **稳定正增益**（≈ +0.02～+0.03）
  - `hybrid` 邻域在该数据上也更稳定（std 更小）

但也要诚实：目前 v14 的 best mean（≈0.47）仍略低于 real_social 上最强 baseline（`kmeans_edge_mean` ≈0.49），说明要“稳定反超”还需要进一步把 SDCN 的训练目标与 edge 统计优势对齐（例如调 `KL/CE/RE` 权重、引入更强的 balanced-constraint、或只在高噪时开启 denoise 等）。

#### E) 补齐 noise=0.5（同样的 v14 网格 + 消融 + baseline 对照）

我们在 noise=1.0 之外补齐 noise=0.5，观察 edge↔edge 的“条件性”更明显：
- 同样是 v14，在不同噪声强度/数据类型下，`ee_graph` 与 `edge_ee` 的正负贡献不一致；
- 这更支持“edge↔edge 是工具而非总开关”：需要结合邻域质量与 edge 信号形态。

##### E.1 real_social_topics_nonknn（noise=0.5）

v14 网格（`α∈{0.02,0.05,0.1}, γ∈{2,4,8,16}`, `ee_graph=hybrid`, seed=0/1/2）：
- 目录：`/tmp/sdcn_dlaa_v14_sprint_20260131_024625_real_social_noise05/`
- best（mean over seed=0/1/2）：`α=0.02, γ=4` → **0.4722 ± 0.0109**（`α=0.02, γ=8` 也同均值）

edge↔edge 消融（同一 best 设置 `α=0.02, γ=4`）：
- 目录：`/tmp/sdcn_dlaa_v14_edgeee_ab_20260131_025319_real_social_noise05/`
- `edge_ee=0`：0.4556 ± 0.0137
- `edge_ee=1`：0.4583 ± 0.0118

ee_graph 消融（同一 best 设置 `α=0.02, γ=4, edge_ee=1`）：
- 目录：`/tmp/sdcn_dlaa_v14_eegraph_ab_20260131_025433_real_social_noise05/`
- incidence：0.4458 ± 0.0223
- **edge_sim：0.4667 ± 0.0148**
- hybrid：0.4611 ± 0.0104

baseline（同噪声）：
- 目录：`/tmp/sdcn_dlaa_baselines_real_social_20260131_025625_noise05/`
- best_overall：`kmeans_edge_mean` → **0.5083 ± 0.0295**

解读：
- noise=0.5 下，edge↔edge 的“净贡献”变小（对比 noise=1.0 时更明显的正增益），且 `edge_sim` 反而略优于 `hybrid`；
- 这符合预期：噪声更低时，过强的“结构邻域混入”（hybrid 的 incidence 部分）可能带来轻微误混合，`edge_sim` 更“纯”。

##### E.2 relational_cycle_nonknn（noise=0.5）

v14 网格：
- 目录：`/tmp/sdcn_dlaa_v14_sprint_20260131_025651_cycle_noise05/`
- best（mean over seed=0/1/2）：≈ **0.3125 ± 0.0148**（多个 `(α,γ)` 并列，如 `α=0.05, γ=8`）

edge↔edge 消融（`α=0.05, γ=8, ee_graph=hybrid`）：
- 目录：`/tmp/sdcn_dlaa_v14_edgeee_ab_20260131_031146_cycle_noise05/`
- `edge_ee=0`：0.3083 ± 0.0266
- `edge_ee=1`：0.3042 ± 0.0212（此处略弱，提示“不是所有数据/噪声都该开 ee”）

ee_graph 消融（`α=0.05, γ=8, edge_ee=1`）：
- 目录：`/tmp/sdcn_dlaa_v14_eegraph_ab_20260131_031258_cycle_noise05/`
- incidence：0.3097 ± 0.0199
- **edge_sim：0.3194 ± 0.0246**
- hybrid：0.3083 ± 0.0223

baseline（同噪声）：
- 目录：`/tmp/sdcn_dlaa_baselines_cycle_20260131_030406_noise05/`
- best_overall：`spectral_node_edge_rbf` → **0.3250 ± 0.0207**

##### E.3 rich_edge_profiles（noise=0.5）

v14 网格：
- 目录：`/tmp/sdcn_dlaa_v14_sprint_20260131_030420_profiles_noise05/`
- best（mean over seed=0/1/2）：**0.4111 ± 0.0045**（如 `α=0.05, γ=4`）

edge↔edge 消融（`α=0.05, γ=4, ee_graph=hybrid`）：
- 目录：`/tmp/sdcn_dlaa_v14_edgeee_ab_20260131_031445_profiles_noise05/`
- `edge_ee=0`：0.4037 ± 0.0069
- `edge_ee=1`：0.4037 ± 0.0026（几乎无差异）

baseline（同噪声）：
- 目录：`/tmp/sdcn_dlaa_baselines_profiles_20260131_031058_noise05/`
- best_overall：`spectral_edge_distance` → **0.6907 ± 0.0069**

解释（非常重要，避免误判“模型不行”）：
- `spectral_edge_distance` 只用 `edge_attr[:,0]` 构造权重；在该数据生成方式下，`edge_attr[:,0]` 很可能是“距离型强信号”，因此谱聚类几乎相当于“使用了正确的先验”；
- 这类数据对 DLAA/SDCN 的难点不在“edge↔edge 去噪”，而在于：如何在不破坏距离先验的情况下，把更高维的 profile 语义也兑现为聚类优势（否则 baseline 本身就接近上界）。

> 工程备注：在 cycle 的 sweep 中偶发 `SIGSEGV`（来自底层 BLAS/OMP 线程栈的不稳定），用 `tools/sweep_stability.py --resume` 可无损续跑；本报告的结果目录均可复现/续跑。

---

### F) 继续推进（中断续跑后）：更“保守”的 ee 邻域 & 损失对齐（real_social_topics_nonknn, noise=0.5）

这部分的动机是：在 noise=0.5 时，v14 仍明显落后于最强 baseline（`kmeans_edge_mean`≈0.508），因此我们尝试两条路径：
1) **先把 ee 邻域定义得更“保守”**（mutual / 阈值），避免误混合；
2) **把训练目标与强 edge 统计信号对齐**（额外重建项 / 调 loss 权重），看能否把均值推过 baseline。

#### F.1 先看“loss 网格”的当前上限（复盘）

- 目录：`/tmp/sdcn_dlaa_v14_lossgrid_20260131_110935_real_social_noise05/`
- 设置：v14 + `ee_graph=edge_sim` + `edge_ee=1` + `α=0.02, γ=4` + `edge_attr_norm=zscore_clip` + noise=0.5；train seed=0/1/2
- best（mean over seed=0/1/2）：**0.4917 ± 0.0207**（`ce=2.0, re=1.0, q_balance=1.0`）

> 结论：通过 “KL/CE/RE/q_balance” 的小网格，我们能把均值推进到 ≈0.49，但仍低于 baseline ≈0.51。

#### F.2 ee_graph 更保守：mutual topk + 相似度阈值（未带来提升）

代码层新增：
- `SDCN_EE_SIM_MUTUAL=1`：只保留 mutual-topk 的相似边（再做无向化）
- `SDCN_EE_SIM_MIN_SIM=...`：cosine 相似度阈值（只保留 ≥ 阈值的邻居）

实验（固定 loss：`ce=2, re=1, q_balance=1`；`α∈{0.01,0.02}`, `γ∈{4,8,16}`；noise=0.5/1.0；seed=0/1/2）：
- 目录：`/tmp/sdcn_dlaa_v14_mutual_smallgrid_20260131_112951_real_social/`
- noise=0.5 best：**0.4903 ± 0.0258**（`α=0.02, γ=8`）
- noise=1.0 best：0.4667 ± 0.0223（明显不如之前 v14 在 noise=1.0 的 best ≈0.47+）

结论：
- “mutual + 阈值” 并未把 mean 推过 ≈0.49；
- 在高噪声下，阈值过强反而可能把 ee 邻域变得过稀，收益下降。

#### F.3 尝试把训练目标对齐强 baseline：额外重建项（效果不显著/略负）

新增两类可选损失（均通过 env 开关，默认关闭）：
- **edge-level 重建**：`SDCN_EDGE_RE_WEIGHT`（把 layer-4 的 edge latent 线性映射回 edge_attr）
- **node-level pooled edge 统计重建**：`SDCN_POOL_RE_WEIGHT`（把 `mean_pool(edge_attr)` 作为 node-level 目标重建，直接对齐 `kmeans_edge_mean` 的归纳偏置）

结果（noise=0.5, seed=0/1/2，固定 v14 + `α=0.02, γ=8` + `ce=2,re=1,q_balance=1`）：
- edge 重建 sweep：`/tmp/sdcn_dlaa_v14_edge_re_20260131_113933_real_social_noise05/`
  - best：**0.4903 ± 0.0232**（`SDCN_EDGE_RE_WEIGHT=0`，即不开）
- pooled 重建 sweep：`/tmp/sdcn_dlaa_v14_pool_re_20260131_114612_real_social_noise05/`
  - best：**0.4903 ± 0.0196**（`SDCN_POOL_RE_WEIGHT=0.1, warmup=10`，与不开几乎持平）
  - 较大权重会带来塌缩/明显掉点（出现 `collapse_final=True`）

结论：
- 在当前设置下，“额外重建”并没有带来可复现的均值提升；
- 过强地把表征拉向 edge 统计，反而可能与 SDCN 的自训练头产生冲突，诱发塌缩。

#### F.4 q-head 直接引入 pooled edge 统计（仍未优于 h4）

新增 `SDCN_Q_SOURCE` 选项：
- `pool`：`q_input = W(pool_mean(edge_attr))`
- `h4_pool`：`q_input = h4 + s * W(pool_mean(edge_attr))`（`s` 可学习，初始化为 0）

对比（noise=0.5, seed=0/1/2；其余同上）：
- 目录：`/tmp/sdcn_dlaa_v14_qsource2_20260131_115353_real_social_noise05/`
- `h4`：0.4958 ± 0.0238
- `h4_pool`：0.4917 ± 0.0266
- `pool`：0.4931 ± 0.0193（seed 方差偏大）

结论：
- 在当前训练配方下，`h4` 仍是最稳的 q 来源；
- “把 pooled edge 直接塞进 q” 的直觉虽然更贴近 baseline，但也更容易把目标与图编码器解耦，导致不稳定/方差上升。

#### F.5 阶段性结论（为什么仍难稳定反超 `kmeans_edge_mean`）

到目前为止，这些结果反过来支持一个更清晰的判断：
1) **baseline 的优势来自“直接、无学习”的统计通路**（`mean_pool(edge_attr)` + kmeans），几乎不受自训练/塌缩机制影响；
2) 我们的 edge↔edge 目前更像“正则/去噪器”，对鲁棒性有帮助，但 **不足以改变最终聚类上界**；
3) 想稳定反超 baseline，下一步更可能需要：
   - 让“edge 统计通路”以更稳定的方式参与训练（例如把 pooled edge 作为显式分支输入到预测头，而不是通过额外重建硬拉表征）；
   - 或者构造更贴近真实任务的 edge 语义（让 `kmeans_edge_mean` 不再是几乎最优的上界），再检验 edge↔edge 的创新价值。
