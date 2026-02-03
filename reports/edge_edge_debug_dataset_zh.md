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
