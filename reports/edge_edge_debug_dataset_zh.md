# edge↔edge 调试诊断数据集（edge_edge_denoise_nonknn）与调优结论

> 目的：在缺少真实数据集时，先用一个“专门放大 edge↔edge 贡献”的合成数据集，快速定位为什么 edge↔edge 往往“不显著/甚至有害”，并给出一套可复现的调参 recipe。

## 1. 现有合成数据生成逻辑（简述）

合成套件由 `tools/generate_synthetic_suite.py` 生成，每个数据集目录包含：
- `node_features.npy` / `labels.npy`
- `binary_adj.npz`（CSR，0/1）
- `edge_attr.npy`（与 CSR `nnz` 顺序对齐）
- `edge_index.npy`（与 `edge_attr` 对齐）
- `data_info.json`

关键点：
- KNN 图：由 `coords` 的 KNN 建图并对称化。
- nonknn 图：`random_k`（每个节点随机采 k 个邻居）并对称化。
- `rich_edge` 类数据集默认会额外注入随机边（`--random_edges_per_node` + `--random_edges_within_prob`）。
- 多个 preset 在 `edge_attr[:,0]` 放“随机噪声”以避免 `spectral_edge_distance` baseline 走捷径。

## 2. 新增诊断数据集：edge_edge_denoise_nonknn

代码位置：`tools/generate_synthetic_suite.py`（preset 名：`edge_edge_denoise_nonknn`）

设计动机（核心）：
- 每个节点的邻域里**同时存在**：
  - 同簇边：`edge_attr` 共享同一个 cluster prototype（低噪声、彼此相似）
  - 跨簇边：`edge_attr` 大多是高熵噪声（与同簇 prototype 不相似）
- 因此：
  - `SDCN_EE_GRAPH=incidence`：edge↔edge 会把同簇边和噪声边“硬混合”，经常洗掉信号
  - `SDCN_EE_GRAPH=incidence_sim` + `SDCN_EE_SIM_MIN_SIM`：只把**相似且相邻（共享端点）**的边连起来，更保守、更利于“边去噪/一致性传播”

数据集信号检查（seed=0 的一次生成）：
- homophily ≈ 0.30（图结构本身不够强）
- `edge_attr[:,0]` 的 within/between AUC ≈ 0.50（防止距离/单通道捷径）
- `coords` / `x` silhouette < 0（节点本身几乎不可分）

## 3. 在该数据集上让 edge↔edge “显著有效”的关键调优点

### 3.1 q 的来源必须跟随“边驱动”

默认 `SDCN_Q_SOURCE=z`（AE latent）。在 node_features 很弱时，`z` 很难承载聚类信号，导致不稳定/塌缩。

建议使用：
- `SDCN_Q_SOURCE=h4`（用图分支 embedding 做 q）
- 并把 `SDCN_SIGMA` 设小一些（减少 AE 分支噪声注入），例如 `SDCN_SIGMA=0.2`

### 3.2 防塌缩：balance 正则是“放大 edge↔edge 贡献”的必要条件

在该诊断集上，为了让不同 seed 下的 edge↔edge 对比稳定，建议开启：
- `SDCN_Q_BALANCE_WEIGHT=0.1`
- `SDCN_PRED_BALANCE_WEIGHT=0.1`

这会明显减少“某个簇被吃空/单簇占比过大”导致的随机性。

### 3.3 incidence_sim 需要相似度阈值（min_sim）

`incidence_sim` 若不加阈值，仍会把“最相似的一些噪声边”硬连起来（尤其在高噪声边较多时），从而削弱 edge↔edge。

建议：
- `SDCN_EE_GRAPH=incidence_sim`
- `SDCN_EE_TOPK=4`（更稀疏、更保守）
- `SDCN_EE_SIM_MIN_SIM=0.4`

## 4. 可复现命令（推荐 recipe）

### 4.1 生成诊断数据集

```bash
python tools/generate_synthetic_suite.py \
  --output_root /tmp/sdcn_edge_debug_suite \
  --seed 0 \
  --presets edge_edge_denoise_nonknn
```

### 4.2 edge↔edge 消融（3 seeds）

```bash
python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/sweep_edge_edge_denoise \
  --variants v5edge_pool_residual \
  --seeds 0,1,2 \
  --epochs 60 \
  --q_sources h4 \
  --sigmas 0.2 \
  --edge_messages 1 \
  --edge_ees 0,1 \
  --ee_graphs incidence_sim \
  --ee_topks 4 \
  --ee_sim_min_sims 0.4 \
  --q_balance_weights 0.1 \
  --pred_balance_weights 0.1
```

> 注：`tools/test_conceptual_data.py` 已记录 `SDCN_EE_GRAPH/SDCN_EE_TOPK/SDCN_EE_SIM_MIN_SIM`，方便追溯配置。

## 5. 观察到的结果（本次环境中的一次跑法）

在上述 recipe 下（`edge_edge_denoise_nonknn`, seed=0/1/2, epoch=60）：
- 关闭 edge↔edge（`SDCN_EDGE_EE=0`）：acc 平均约 **0.356**
- 开启 edge↔edge（`SDCN_EDGE_EE=1`）：acc 平均约 **0.408**

同样设置下对比 ee 图结构（`SDCN_EDGE_EE=1`）：
- `SDCN_EE_GRAPH=incidence`：acc 平均约 **0.358**
- `SDCN_EE_GRAPH=incidence_sim`：acc 平均约 **0.413**

> 结论：在“同簇边一致、跨簇边高噪声”的典型调试场景下，edge↔edge 的关键不在“有没有 EE”，而在 **EE 邻域是不是足够保守（incidence_sim + min_sim）**，以及训练是否足够稳定（balance 正则 + 合适的 q_source/sigma）。

## 6. 新架构有效性：v16（edge↔edge residual + edge_attr fusion）消融

本仓库新增实验版 `v16edge_ee_residual_aux_fusion`（见 `DLAA_NEW.py`），核心是把“边表征（edge embedding）”以一个可控的 fusion 形式注入 node attention 的 `edge_attr`，从而让 edge 信息更直接进入节点表示；同时保留 edge↔edge 更新与 edge-level aux head（aux 需要在训练 loop 中显式启用）。

### 6.1 v16 与 v5 的对比（同一 recipe）

对比命令（与第 4 节一致，仅替换 variant）：

```bash
python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/abl_edge_edge_v16_gated_quick \
  --variants v16edge_ee_residual_aux_fusion \
  --seeds 0,1,2 --epochs 60 \
  --q_sources h4 --sigmas 0.2 \
  --edge_messages 1 \
  --edge_ees 0,1 \
  --ee_graphs incidence_sim --ee_topks 4 --ee_sim_min_sims 0.4 \
  --q_balance_weights 0.1 --pred_balance_weights 0.1 \
  --edge_attr_fuses 1 --edge_attr_fuse_scales 0.1 --edge_attr_fuse_detaches 0 \
  --edge_aux_weights 0.0 \
  --gat_input_dropouts 0.1
```

本次跑法下的现象（seed=0/1/2）：
- **v5**：`SDCN_EDGE_EE=0 → 1` 会带来明显提升（mean acc 约 `0.353 → 0.408`），说明“edge↔edge + incidence_sim”在该诊断集上确实有效。
- **v16**：整体 acc 均值可略高于 v5，但 `SDCN_EDGE_EE=0/1` 的差距不明显；主要收益来自 **edge_attr fusion**（见 6.2）。

### 6.2 v16 的结构消融：fusion 是否必要？

保持其它超参不变，仅把 `SDCN_EDGE_ATTR_FUSE_SCALE` 从 `0.1` 改为 `0.0`（等价于关闭 fusion）：

```bash
python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/abl_edge_edge_v16_gated_fuse0 \
  --variants v16edge_ee_residual_aux_fusion \
  --seeds 0,1,2 --epochs 60 \
  --q_sources h4 --sigmas 0.2 \
  --edge_messages 1 \
  --edge_ees 0,1 \
  --ee_graphs incidence_sim --ee_topks 4 --ee_sim_min_sims 0.4 \
  --q_balance_weights 0.1 --pred_balance_weights 0.1 \
  --edge_attr_fuses 1 --edge_attr_fuse_scales 0.0 --edge_attr_fuse_detaches 0 \
  --edge_aux_weights 0.0 \
  --gat_input_dropouts 0.1
```

结论（本次环境结果）：
- `SDCN_EDGE_ATTR_FUSE_SCALE=0.0` 时，v16 的 mean acc 降到约 `0.367` 左右；
- `SDCN_EDGE_ATTR_FUSE_SCALE=0.1` 时，v16 的 mean acc 提升到约 `0.419` 左右；
- 说明在该诊断集上，**fusion 路径是 v16 的主要贡献点**（比“额外加 EE 更新”更关键）。

### 6.3 v16 的注意事项（当前版本）

- v16 若使用默认 `SDCN_FINAL_ASSIGN=pred`，在该诊断集上容易出现“某个簇被吃空”（cluster_distribution 少一个簇）的情况，导致 `collapse_final=True`。
- **推荐做法：把最终输出改为 `SDCN_FINAL_ASSIGN=q` 或 `p`**。在本次环境中（seed=0/1/2, epoch=60）：
  - `final_assign=q`：acc 约 `[0.625, 0.488, 0.521]`，mean≈`0.544`，且 `collapse_final=False`（3/3 seeds）。
  - `final_assign=p`：acc 约 `[0.604, 0.488, 0.542]`，mean≈`0.544`，且 `collapse_final=False`（3/3 seeds）。
- 说明 v16 的“图分支表示（q）”在该数据集上明显强于 `pred` 头；用 `q/p` 作为最终聚类输出能同时提升性能与稳定性。

如果你**必须**使用 `final_assign=pred`（只看 pred head 的可用性），一个有效的稳定化手段是增强 `pred→p` 的对齐强度：
- `SDCN_CE_WEIGHT=1.0`
- `SDCN_CE_WARMUP_EPOCHS=20`

在本次环境中该设置能让 `final_assign=pred` 也达到 `collapse_final=False`（3/3 seeds），mean acc≈`0.435`。

若要进一步提升稳定性/泛化，建议下一步继续 sweep：
  - `SDCN_Q_BALANCE_WEIGHT / SDCN_PRED_BALANCE_WEIGHT`（例如 0.1→0.3）
  - `SDCN_EDGE_ATTR_FUSE_SCALE`（0.05/0.1/0.2）
  - 学习率（`--lrs 1e-3,5e-4`）

## 7. 创新探索：其他边特征利用方式（v17~v21）

本节尝试在 v5/v16 的“边信息入口”之外，探索更不同的 edge_attr/edge→node 路径。实现位置：`DLAA_NEW.py`。

### 7.1 新增结构简述

- **v17edge_attr_gate**：把 refined edge embedding 当作“门控信号”，对 node attention 使用的 `edge_attr` 做**乘性调制**（更保守，适合噪声 edge_attr）：
  - `edge_attr_att = dist_feat * exp(fuse_scale * tanh(W(LN(edge_feat_1))))`
- **v18edge_attr_mlp_fuse**：对 `[dist_feat, edge_feat_1]` 拼接做一个残差 MLP，让模型自己学“怎么把 refined edge 注入 edge_attr”：
  - `edge_attr_att = dist_feat + fuse_scale * MLP([dist_feat, LN(edge_feat_1)])`
  - MLP 以 “近似恒等映射” 初始化（delta≈0），理论上更稳，但可能需要更合适的 scale/训练配方才会出收益。
- **v19edge_attn_pool**：替代 mean pooling，用 **node→incident-edge attention pooling** 聚合边特征（意图：抑制噪声边、突出“有用边”）：
  - `pooled_i = Σ softmax(score(i,e)) * V(edge_feat_e)`（对每个节点在 incident edges 上做 softmax）
  - 该路径用 `tanh(scale)` 控幅，初始化接近 0，避免一上来就破坏 baseline。
- **v20edge_attr_scalar_gate**：把 refined edge embedding 融入 `edge_attr`，但只用 **每条边一个标量 gate**（更鲁棒）：
  - `gate_e = sigmoid(w^T LN(edge_feat_1))`
  - `edge_attr_att = dist_feat + fuse_scale * tanh(scale) * gate_e * LN(edge_feat_1)`（`scale` 可学习，初始化为 0）
- **v21dual_sgat_edge_attr**：**双分支 SGAT**：raw edge_attr 分支 + fused edge_attr 分支，然后在 node 表征层面做残差融合：
  - `node_raw = SGAT(x, edge_attr=dist_feat)`
  - `node_fused = SGAT(x, edge_attr=dist_feat + fuse_scale * LN(edge_feat_1))`
  - `node_att = node_raw + tanh(scale) * node_fused`（`scale` 初始化为 0）

### 7.2 快速对比实验（edge_edge_denoise_nonknn, final_assign=pred）

固定 recipe（与前文 v16 对比风格一致，主要看 pred head 的可用性）：

```bash
python tools/generate_synthetic_suite.py \
  --output_root /tmp/sdcn_edge_debug_suite \
  --seed 0 \
  --presets edge_edge_denoise_nonknn

python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/sweep_edge_arch_explore_v20_v21_pred \
  --variants v5edge_pool_residual,v7edge_attr_fusion,v16edge_ee_residual_aux_fusion,v17edge_attr_gate,v18edge_attr_mlp_fuse,v19edge_attn_pool,v20edge_attr_scalar_gate,v21dual_sgat_edge_attr \
  --seeds 0,1,2 --epochs 60 \
  --q_sources h4 --sigmas 0.2 \
  --edge_messages 1 \
  --edge_ees 1 \
  --ee_graphs incidence_sim --ee_topks 4 --ee_sim_min_sims 0.4 \
  --q_balance_weights 0.1 --pred_balance_weights 0.1 \
  --edge_attr_fuses 1 --edge_attr_fuse_scales 0.1 --edge_attr_fuse_detaches 0 \
  --edge_aux_weights 0.0 \
  --gat_input_dropouts 0.1
```

结果汇总（mean±std over 3 seeds；`collapse_final` 为 3 个 seed 里出现“吃空簇”的次数）：

| variant | acc | nmi | ari | f1 | collapse_final |
|---|---:|---:|---:|---:|---:|
| v5edge_pool_residual | 0.4083±0.0223 | 0.1174 | 0.0886 | 0.3539 | 0/3 |
| v7edge_attr_fusion | 0.3986±0.0449 | 0.1225 | 0.0846 | 0.3546 | 1/3 |
| v16edge_ee_residual_aux_fusion | 0.4208±0.0445 | 0.1335 | 0.1075 | 0.3459 | 1/3 |
| v17edge_attr_gate | 0.4222±0.0431 | 0.1294 | 0.1166 | 0.3512 | 0/3 |
| v18edge_attr_mlp_fuse | 0.3181±0.0599 | 0.0426 | 0.0279 | 0.2544 | 1/3 |
| v19edge_attn_pool | 0.3722±0.0682 | 0.0886 | 0.0623 | 0.3063 | 1/3 |
| v20edge_attr_scalar_gate | 0.3417±0.0450 | 0.0771 | 0.0328 | 0.2767 | 0/3 |
| **v21dual_sgat_edge_attr** | **0.4569±0.0227** | **0.1677** | **0.1299** | **0.4282** | **0/3** |

结论（当前这组 quick recipe）：
- **v21（dual-SGAT）在 `final_assign=pred` 下表现最好且稳定**（0/3 collapse），说明“同时保留 raw edge_attr 与 fused edge_attr 两套注意力路径”对 pred head 更友好。
- v17（乘性门控）依然是一个稳定的方向（0/3 collapse），但在 pred 指标上不如 v21。
- v18/v19 仍偏不稳定/偏弱（1/3 seeds collapse），更适合在 `final_assign=q/p` 或更细的超参 sweep 下再评估。

### 7.3 同一 recipe，但 final_assign=q（更贴近“聚类分配”）

```bash
python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/sweep_edge_arch_explore_v20_v21_q \
  --variants v5edge_pool_residual,v7edge_attr_fusion,v16edge_ee_residual_aux_fusion,v17edge_attr_gate,v18edge_attr_mlp_fuse,v19edge_attn_pool,v20edge_attr_scalar_gate,v21dual_sgat_edge_attr \
  --seeds 0,1,2 --epochs 60 \
  --q_sources h4 --sigmas 0.2 \
  --edge_messages 1 \
  --edge_ees 1 \
  --ee_graphs incidence_sim --ee_topks 4 --ee_sim_min_sims 0.4 \
  --q_balance_weights 0.1 --pred_balance_weights 0.1 \
  --edge_attr_fuses 1 --edge_attr_fuse_scales 0.1 --edge_attr_fuse_detaches 0 \
  --edge_aux_weights 0.0 \
  --gat_input_dropouts 0.1 \
  --final_assign q
```

结果汇总（mean±std over 3 seeds）：

| variant | acc | nmi | ari | f1 | collapse_final |
|---|---:|---:|---:|---:|---:|
| v5edge_pool_residual | 0.3889±0.0129 | 0.0939 | 0.0646 | 0.3316 | 0/3 |
| v7edge_attr_fusion | 0.5097±0.0258 | 0.2485 | 0.2107 | 0.4710 | 0/3 |
| **v16edge_ee_residual_aux_fusion** | **0.5444±0.0586** | 0.2425 | **0.2198** | **0.5187** | 0/3 |
| v17edge_attr_gate | 0.4972±0.0529 | 0.2284 | 0.2052 | 0.4401 | 0/3 |
| v18edge_attr_mlp_fuse | 0.4417±0.0312 | 0.1574 | 0.1299 | 0.3849 | 0/3 |
| v19edge_attn_pool | 0.3333±0.0335 | 0.0554 | 0.0345 | 0.2909 | 0/3 |
| v20edge_attr_scalar_gate | 0.5042±0.0180 | 0.1997 | 0.1728 | 0.4449 | 0/3 |
| v21dual_sgat_edge_attr | 0.4861±0.0119 | 0.2096 | 0.1830 | 0.4257 | 0/3 |

结论（final_assign=q）：
- 与 6.3 的观察一致：**final_assign=q 明显更稳**（本组里所有 variant 都是 0/3 collapse）。
- v16 依然是该诊断集上的强基线（mean acc≈0.544）。
- v20（标量门控）在 q 指标下变得更有竞争力（acc≈0.504 且 std 更小），但 pred 指标不占优，说明它更偏向“提升图分支聚类”而非 pred head。
