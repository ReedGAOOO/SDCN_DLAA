# 合成数据多轮测试分析（v1/v2/v3 vs Baselines）

本报告基于 `tools/generate_synthetic_suite.py` 生成的合成数据集套件，对比：
- **模型**：`v1original`、`v2edge_single_layer`、`v3edge_cross_layers`
- **传统 baseline**：`kmeans_x`、`spectral_adj_binary`、`spectral_edge_distance`（只用 `edge_attr[:,0]` 作为“距离”权重）

原始跑分表见：
- `reports/synthetic_benchmark_report_zh.md`（suite_seed=0, epochs=30）
- `reports/synthetic_benchmark_report_zh_epochs60.md`（suite_seed=0, epochs=60）
- `reports/synthetic_benchmark_report_zh_suite_seed1.md`（suite_seed=1, epochs=30）

补充：为了更可解释地回答“为什么谱聚类 baseline 更强”，新增了**数据集信号分析表**：
- `reports/suite_signal_seed0.md` / `reports/suite_signal_seed0.json`
- `reports/suite_signal_seed1.md` / `reports/suite_signal_seed1.json`

## 实验设置

- 数据套件：`/tmp/sdcn_dlaa_suite_seed0`（6 个数据集：3 个 distance_1d + 3 个 rich_edge）
- 多轮：`seeds=0,1,2`
- 训练轮数：`epochs=30` 与 `epochs=60`
- 其他：`heads=1`，`max_edges_per_node=10`

补充：为了验证“数据生成随机性”的影响，也额外跑了 `suite_seed=1, epochs=30`（同样 seeds=0,1,2），总体趋势与 suite_seed=0 一致。

## 关键结论（当前阶段）

1. **在 distance_1d（只有距离 1D 边特征）上，传统谱聚类是非常强的上界**  
   - `spectral_edge_distance` 在 `dist_two_moons` 上达到 **1.0**（完美分割），在 `dist_blobs_easy` 上接近 **0.99+**。  
   - 这类数据本质上“图结构/距离”已经足够，深模型未必能超过强基线。

2. **v2/v3 在多个数据集上出现明显的“塌缩”现象**（见 report 里的 `collapse_rate`）  
   - 典型表现：`cluster_distribution` 极度偏斜（如几乎所有点落到同一簇），导致 acc/nmi/ari 很低。  
   - `v2` 在多个数据集上 `collapse_rate` 偏高；`v3`相对更“有机会”跑出非塌缩结果，但仍不稳定。

3. **增加 epochs 对部分数据集有帮助，但不是普遍提升**  
   - `dist_two_moons`：`v3` 从 epochs=30 的中等水平提升到 epochs=60 的 **0.72±0.13（且塌缩率 0）**，说明训练轮数/收敛程度会显著影响结果。  
   - 但在 `rich_edge_profiles / rich_multirelation` 上，增大 epochs 并没有根本解决塌缩问题（仍出现大量极端簇分布）。

4. **当前 rich_edge 数据集中，baseline 依然能在部分数据上达到很高分**  
   - `rich_geo_temporal`、`rich_multirelation` 的谱聚类结果接近 **0.93~0.95**，说明“空间 KNN 图”本身就携带了很强的可分信息。  
   - 这会掩盖“高维边特征”的增益：即使模型能用 edge feature，提升空间也被 baseline 吃掉了。

## 补充实验：用“信号指标”解释 baseline 为什么强

新增脚本 `tools/analyze_suite_signal.py`，对每个数据集计算：
- **homophily**：边连接同标签点的比例（越高越“谱聚类友好”）
- **dist_auc / effect_d**：把 `edge_attr[:,0]` 当“距离”特征时，同簇/异簇边是否可分（越高越“距离谱聚类友好”）
- **coords_sil / x_sil**：仅从 `coords` 或 `node_features` 看标签的可分性（辅助诊断）

这能把“baseline 强”拆成可解释的两步：
- 数据本身是否已经满足“高同配性 + 距离可分”？
- baseline 是否正好用到了这个信号（`spectral_adj_binary` 用同配性，`spectral_edge_distance` 用距离可分）？

示例（见 `reports/suite_signal_seed0.md`）：
- `dist_blobs_easy`：homophily≈0.99 且 dist_auc≈0.97 → `spectral_edge_distance` 接近最优几乎是必然结果
- `rich_multirelation`：homophily≈0.80 且 dist_auc≈0.82 → 谱聚类依然很强（尽管它没用到“高维语义边特征”）

## 补充实验：构造“距离无效、语义边有效”的数据集

为验证“高维边特征是否真的带来优势”，新增合成 preset：`rich_edge_semantic_only`（`tools/generate_synthetic_suite.py`）。
- 设计目标：**坐标/距离不再携带标签信息**，但 `edge_attr` 含可区分的语义通道
- 信号分析（seed=0）：homophily≈0.36、dist_auc≈0.44、coords_sil≈-0.03、x_sil≈-0.02（距离/节点特征都几乎无效）

对比结果（epochs=60, seeds=0/1/2）：
- baseline：`spectral_*` 下降到约 **0.36~0.38**（符合预期）
- v1/v2/v3：依然接近随机且频繁塌缩（典型 `cluster_distribution` 为 178/1/1）

这说明当前实现下，模型**并没有在“只有 edge_attr 提供信号”的场景里有效利用边语义**。

### 更新：注入 edge message 后的改观（定位“edge 只调权重”的瓶颈）

在 `DLAA_NEW.py` 的 `SGATLayer` 中增加了一个可选开关 `SDCN_EDGE_MESSAGE=1`：除了用 `edge_attr` 参与注意力权重外，再额外把 `edge_attr` 作为**消息内容**聚合到节点（`mean_{dst}(Linear(edge_attr))`），让“边语义”可以直接写入节点表示。

快速验证（`rich_edge_semantic_only`, seed=0, epochs=30, AE 预训练 200, `--node_edge_pool mean_concat`, `lr=5e-4`, `dropout=0`, `heads=1`）：
- `SDCN_EDGE_MESSAGE=0`：final `pred` 约 **0.34**（接近随机），且 `pred` hard-assign 迅速塌缩到单簇/双簇
- `SDCN_EDGE_MESSAGE=1`：final `pred` 可到 **0.48**（显著高于谱聚类 baseline 的 ~0.36~0.38）

这基本确认了此前的关键瓶颈：**仅“调注意力权重”不足以在 node_features 无效时“创造可聚类的节点表示”**；需要让 edge_attr 以“内容消息”的形式进入节点更新。

## 补充实验：sigma 扫描（验证 AE↔GNN 融合是否是主因）

新增环境变量 `SDCN_SIGMA`（覆盖融合系数 sigma，范围 [0,1]），并在 `rich_edge_semantic_only` 上扫：
- `sigmas = 0, 0.25, 0.5, 0.75, 1`（epochs=60, seeds=0/1/2）
- 现象：`v2/v3` 基本对 sigma 不敏感，绝大多数配置仍塌缩到单簇；`v1` 偶尔略好但依然远低于有效聚类水平

结论：**“sigma 混合比例”不是解释当前失败的主要旋钮**，问题更可能在于“信息流/损失约束/稳定性”。

## 补充实验：edge 特征消融（判断模型到底有没有用到高维边）

给 `tools/test_conceptual_data.py` 增加 `--edge_ablation`（none/distance_only/shuffle_rows/zeros），在两个 rich_edge 数据上做快速对照：

- `rich_edge_profiles`：无论 edge_attr 置零/打乱/只留距离，v2/v3 结果几乎完全一致且塌缩（表明该设置下模型**基本没用到边特征**）
- `rich_multirelation`：baseline 的谱聚类可达 **0.95**，而 v2/v3 明显更低；且 v3 在 `shuffle_rows/zeros` 下反而更好，提示“某些 edge_attr 形态可能诱发不稳定/塌缩”

补充：`rich_edge_profiles` 的 edge_attr 中包含 `1/(dist+1e-3)` 这类极端尺度通道（最大可到 ~1e3）。当把 edge_attr 池化进节点特征（`--node_edge_pool mean_concat`）时，会导致 AE 重构的 MSE 量级显著增大、训练更不稳。为方便做“尺度公平”的对照，在 `tools/test_conceptual_data.py` 增加了 `--edge_attr_norm`（`none|zscore|zscore_clip|minmax`）用于可控归一化。

### 进一步实验：`SDCN_Q_SOURCE` 对塌缩的影响（更“可定位”的机制验证）

核心假设：**塌缩不只是“模型弱”，还可能来自 SDCN 的自训练目标与信息源不一致**。

- `q/p` 的目标来自 `q_input`（`SDCN_Q_SOURCE` 控制：`z`/`h4`/`fused`），而 `pred` 的预测来自最终图分支输出（`h5`）。
- 当任务是 **edge 主导**（节点特征与距离几乎无信号）时，若 `q_input=z`（纯 AE latent）无法承载 edge 语义，则 `p` 会对训练产生“错误牵引”，使 `pred` 更容易退化到“常数输出 + 单簇 argmax”的稳定坏解。

在两个代表性数据集上用小网格验证该现象（均为 `v2edge_single_layer`，`epochs=30`，`seeds=0/1/2`，`lr=5e-4`，`dropout=0`，`heads=1`，`n_z=10`，`SDCN_PRETRAIN_EPOCHS=200`，并使用 `--node_edge_pool mean_concat`）：

**A) `rich_edge_semantic_only`（距离无效、语义边有效）**

| q_source | edge_message | mean acc | collapse_rate |
|---|---:|---:|---:|
| z | 0 | 0.3444 | 1.00 |
| h4 | 0 | 0.3537 | 0.67 |
| fused | 0 | 0.3500 | 1.00 |
| z | 1 | 0.4315 | 0.33 |
| **h4** | **1** | **0.4185** | **0.00** |
| fused | 1 | 0.3870 | 0.33 |

结论：**开启 `SDCN_EDGE_MESSAGE=1` 是必要条件**；同时把 `SDCN_Q_SOURCE` 从 `z` 改为 **`h4`** 能显著降低塌缩（3 个 seed 全部不塌）。

**B) `rich_edge_profiles`（包含极端尺度通道；此处固定 `--edge_attr_norm zscore_clip`）**

| q_source | edge_message | mean acc | collapse_rate |
|---|---:|---:|---:|
| z | 1 | 0.3944 | 0.33 |
| **h4** | **1** | **0.4333** | **0.00** |
| fused | 1 | 0.3889 | 0.33 |

补充验证（同配置下，仅切换 `edge_message`，固定 `q_source=h4`）：
- `edge_message=0`：mean acc 0.3889，collapse_rate 0.67
- `edge_message=1`：mean acc 0.4333，collapse_rate 0.00

解释：`h4` 更接近“图/边驱动”的表示空间，`p` 的目标与 `pred` 的信息源更一致，自训练不再强迫模型去拟合一个与图分支无关的 `p`，因此更稳定。

## 对“为什么 baseline 更强 / 为什么 v3 优势不明显”的综合解释

结合上述信号诊断 + 消融现象，主要有两类原因：

1. **任务上界/数据设计原因**：当前 suite 很多数据天然满足“同配性高 + 距离可分”，谱聚类接近最优；此时深模型很难再抬高上界，反而可能因训练噪声/超参不稳而落后。
2. **模型机制与稳定性原因**：  
   - 当前实现的 edge 参与方式更像“调注意力权重”（GATConv 的 `edge_dim`），而**信息内容仍主要来自节点特征**；当 `node_features` 本身弱/无效时（如 `rich_edge_semantic_only`），仅靠调权重很难从边语义中“创造”可聚类的节点表示。  
   - SDCN 的自监督聚类对塌缩敏感：当 q/p 早期偏斜时，后续会被目标分布进一步放大，导致单簇解更稳定（需要额外正则/预训练/损失设计来抑制）。

## 复现命令（关键增量）

- 生成 `rich_edge_semantic_only`：`python tools/generate_synthetic_suite.py --output_root /tmp/suite_semantic_test --seed 0 --presets rich_edge_semantic_only`
- 信号诊断表：`python tools/analyze_suite_signal.py --suite_dir /tmp/suite_semantic_test --out_md /tmp/suite_semantic_test_signal.md --out_json /tmp/suite_semantic_test_signal.json`
- v1/v2/v3 vs baselines：`python tools/benchmark_synthetic_suite.py --suite_dir /tmp/suite_semantic_test --out_dir /tmp/sdcn_semantic_only_bench --epochs 60 --seeds 0,1,2 --variants v1original,v2edge_single_layer,v3edge_cross_layers`
- sigma 扫描：`python tools/sweep_sigma.py --data_dir /tmp/suite_semantic_test/rich_edge_semantic_only --out_dir /tmp/sigma_sweep_semantic_only --sigmas 0,0.25,0.5,0.75,1 --seeds 0,1,2 --epochs 60 --variants v1original,v2edge_single_layer,v3edge_cross_layers`
- edge 消融：`python tools/test_conceptual_data.py --data_dir <DATASET_DIR> --edge_ablation none|distance_only|shuffle_rows|zeros`

## Suite 全量复跑：h4 推荐组合 + 稳定性策略（抑制塌缩）

为了把“塌缩”从现象变成可操作的工程问题，我们把 suite 全量跑法固化在 `tools/benchmark_synthetic_suite.py`：
- `--recommended_h4`：统一设置 `SDCN_Q_SOURCE=h4`，并按数据类别自动设置 `SDCN_EDGE_MESSAGE`（`distance_1d` 默认 0，`rich_edge*` 默认 1）
- `--recommended_auto`：按数据集自动选择 `SDCN_Q_SOURCE`（目前启发式：`distance_1d` 若 `K<=2` 用 `z`，否则用 `h4`；`rich_edge*` 用 `h4`）
- `--edge_message_policy auto|on|off`：当启用 recommended 组合时，控制 `SDCN_EDGE_MESSAGE` 的策略
  - 经验上：`v2edge_single_layer` 在 rich_edge 更受益于 `edge_message=1`（把 edge_attr 作为**消息内容**注入节点）
  - 但 `v3edge_cross_layers` 的 `edge_attr` 已经是“更新后的 edge embedding”，再额外注入 `edge_message` 在多数设置下容易过平滑并提升塌缩率；因此当前 `auto` 策略下 **v3 默认强制 `SDCN_EDGE_MESSAGE=0`**
    - 注意：在 `rich_multirelation` 这类“边语义更强、节点特征更弱”的任务里，如果再配合 `--node_edge_pool mean_replace`（把 edge_attr 池化进节点输入）与小 AE（`SDCN_ENC_DIMS=256,256,512`），`edge_message=1` 反而能显著降低单簇塌缩并提升分数（见下文“定点排查”）。
- `rich_edge_profiles` 自动加 `--edge_attr_norm zscore_clip`（避免极端尺度通道干扰）
- 新增 `--strategy_rich_only`：把稳定性策略只应用在 `rich_edge` 数据集（可避免在 distance_1d 上“过正则”）

同时实现了三类“机制性抑制塌缩”的训练策略（均通过环境变量控制，默认关闭，不影响旧实验复现）：
- `SDCN_CE_WARMUP_EPOCHS`：CE loss 权重 warmup
- `SDCN_P_SMOOTHING`：把 target distribution `p` 向 uniform 做 label-smoothing（降低早期极端尖锐的自训练牵引）
- `SDCN_PRED_MI_WEIGHT`：对 `pred` 加 mutual-information 正则（鼓励“整体均衡 + 单样本更自信”，直接抑制“pred 全部近均匀 → argmax 单簇”的坏解）
- `SDCN_Q_MI_WEIGHT`：对 `q` 做同样的 MI 正则（用于抑制“q 退化到近均匀 → hard argmax 单簇”的坏解；在部分 rich_edge 上对 v2 有帮助）

另外修复了一个会导致 suite 跑批中途崩溃的问题：`evaluation.py` 的 `cluster_acc` 现在对“预测簇数≠真实簇数”（典型塌缩场景）也能稳定返回 acc/f1，不再直接 `return None`。

### suite_seed0（`/tmp/sdcn_dlaa_suite_seed0`）对照

产物：
- 基线（h4rec）：`reports/synthetic_benchmark_report_zh_h4rec.md`
- strategy1（p_smooth=0.1 + ce_warmup=10 + mi=0.1）：`reports/synthetic_benchmark_report_zh_h4rec_strategy1.md`
- 消融：`reports/synthetic_benchmark_report_zh_h4rec_strategy2.md`（p_smooth+warmup）、`reports/synthetic_benchmark_report_zh_h4rec_strategy3.md`（MI-only）

结论（重点看 collapse_rate 变化）：
- **strategy1 明显降低 v2 在 `dist_two_moons / rich_edge_profiles / rich_geo_temporal / rich_multirelation` 的塌缩率**，同时把 `rich_multirelation` 的 v3 塌缩率从 0.67 降到 0.33。
- 消融结果显示：**MI（strategy3）对多个数据集提升很明显**；而 **p_smooth + warmup（strategy2）更偏向改善 `dist_two_moons` 这类容易“一簇吞噬”的场景**；两者组合通常更稳。
- v3 在 `rich_edge_profiles` 仍顽固塌缩（需要进一步从 v3 的 edge→node 信息利用方式/超参入手）。

### suite_seed1（`/tmp/sdcn_dlaa_suite_seed1`）复核

产物：
- 基线（h4rec）：`reports/synthetic_benchmark_report_zh_h4rec_suite_seed1.md`
- strategy1（全量应用）：`reports/synthetic_benchmark_report_zh_h4rec_strategy1_suite_seed1.md`
- strategy1_richonly（只对 rich_edge 应用）：`reports/synthetic_benchmark_report_zh_h4rec_strategy1_richonly_suite_seed1.md`

结论：
- 在 suite_seed1 上，strategy1 仍能降低 `rich_edge_profiles / rich_geo_temporal / rich_multirelation` 中 v2 的塌缩率，但对部分 `distance_1d` 数据集存在“过正则”风险。
- 因此更推荐实际跑批时使用 `--strategy_rich_only`，把稳定性策略先限定在 `rich_edge` 任务上（符合“按数据决定”的思路），再逐步把策略扩展到 `distance_1d` 并做更细的参数扫。

### 更新：`recommended_auto + pretrain=200 + strategyA(richonly) + v3 edge_message=off`

为了把“按数据/按版本决定开关”也纳入可复现实验，我们又跑了一组更贴近“实际推荐跑法”的组合：
- `SDCN_PRETRAIN_EPOCHS=200`（自监督聚类对初始化很敏感；不预训练时整体明显更差）
- `--recommended_auto`（二分类 distance 数据用 `q_source=z` 更稳；rich_edge 用 `h4` 更一致）
- `strategyA`（仅 rich_edge 生效）：`SDCN_P_SMOOTHING=0.1`、`SDCN_CE_WARMUP_EPOCHS=10`、`SDCN_PRED_MI_WEIGHT=0.1`、`SDCN_Q_MI_WEIGHT=0.1`
- `edge_message`：auto 策略下对 v3 强制关闭（避免“双重 edge→node 注入”）

产物：
- suite_seed0（models-only）：`reports/synthetic_benchmark_report_zh_autorec2_pretrain200_strategyA_richonly_emv3off.md`
- suite_seed1（models-only）：`reports/synthetic_benchmark_report_zh_suite_seed1_autorec2_pretrain200_strategyA_richonly_emv3off.md`

观察：
- suite_seed0 上，`v3edge_cross_layers` 在 `dist_two_moons` 的 **collapse_rate 降到 0**，且在 `rich_geo_temporal` 的均值 acc 明显高于 v2（但在 `rich_multirelation` 仍有较高塌缩）。
- suite_seed1 上仍存在“数据生成随机性/初始化敏感”导致的波动：例如 `dist_two_moons` 对 v2/v3 都比较不稳，`rich_edge_profiles` 上 v3 仍可能顽固塌缩。

### 定点排查：`rich_multirelation` 的 v3 “均匀塌缩（uniform trap）”

这类失败模式和“一簇吞噬”不同：它更像是 **q/p/pred 接近均匀分布（熵≈logK）+ KL(P||Q)≈0** 的“零梯度坏解”。  
现象上看起来可能是：
- `pred` 的 soft 概率几乎均匀，但因为极小偏置导致所有样本 argmax 落到同一个簇（或少数几个簇）→ 表现为“塌缩/少簇”。
- trace 中常见：`q_entropy_mean≈logK`、`p_entropy_mean≈logK`、`kl_p_q_mean≈0`。

为更可控地复现/定位，我们用 `tools/sweep_stability.py` 固定数据集 `/tmp/sdcn_dlaa_suite_seed0/rich_multirelation`，只扫 v3 的关键超参，并记录每轮 `trace.jsonl`。

#### 1) 仅扫 `sigma/q_source/ce_weight`（`edge_message=0`）不足以解决

此前 stage1a（不池化边特征、默认大 AE）里，`sigma/q_source/ce_weight` 的组合基本都陷入上述“均匀坏解”，几乎全程 `KL≈0`，很难靠损失自身拉出。

#### 2) `--node_edge_pool mean_replace` + 小 AE 能大幅降低“单簇”概率，但容易“少簇（2~3）”

当把 `edge_attr` 池化进节点输入（`--node_edge_pool mean_replace`）并把 AE 缩小（`SDCN_ENC_DIMS=256,256,512`），在 v3 上常能把“全点单簇”的比例明显压下去，但仍经常出现“少簇”：
- 典型表现：`cluster_distribution` 用到了 2~3 个簇（`max_frac` 不高），但缺失某些簇 id。
- 这更像是“欠分簇/合并簇”而不是“塌到一簇”，需要进一步增强分离/置信度。

#### 3) 在该数据集上，`edge_message=1` 反而是有效开关（与默认 auto 策略相反）

在同样的“小 AE + mean_replace”前提下，我们扫 `sigma∈{0.5,0.75,0.9,1.0}`、`q_source∈{z,h4,fused}` 并开启 `SDCN_EDGE_MESSAGE=1`（其余固定：`pretrain=200, lr=5e-4, dropout=0, heads=4, n_z=10, ce_warmup=10`），得到一批**非塌缩且高分**的点：
- seed0：`sigma=0.9, q_source=fused` → acc≈0.70，且 4 簇都被使用（`/tmp/sweep_rich_multirelation_v3_stage1b8/aggregate.json`）
- seed1：`sigma=0.75, q_source=h4` → acc≈0.76，且簇分布较均衡（同上）
- seed2：仍最难，但 `sigma=1.0, q_source=h4` 可得到非塌缩点（acc≈0.49，簇分布偏斜）

复现示例（单次跑一个点）：
- `SDCN_PRETRAIN_EPOCHS=200 SDCN_ENC_DIMS=256,256,512 SDCN_SIGMA=0.75 SDCN_Q_SOURCE=h4 SDCN_EDGE_MESSAGE=1 SDCN_CE_WARMUP_EPOCHS=10 SDCN_CE_WEIGHT=1 python tools/test_conceptual_data.py --data_dir /tmp/sdcn_dlaa_suite_seed0/rich_multirelation --lr 5e-4 --dropout 0 --heads 4 --n_z 10 --node_edge_pool mean_replace`

> 备注：本段的 “collapse” 使用当前工具的严格定义（少簇也算），因此建议同时参考 `cluster_distribution` 的 `max_frac` 来区分“单簇塌缩”与“欠分簇”。

## 下一步建议（建议继续多轮迭代）

### A) 设计“edge feature 必须有用”的更强合成任务

目标：让 `spectral_adj_binary` 和 `spectral_edge_distance` 明显变差，而 edge-aware 模型有机会提升。

可行方向（保持“现实意义”）：
- **多关系社交图**：拓扑接近随机/均匀度分布，但边类型（relation type/互动强度）强相关于社区；距离不再与社区相关。  
- **同空间不同语义的地理路网**：坐标高度重叠（距离无区分），但道路等级/通行时间/方向性特征与簇相关。  
- **异配性（heterophily）图**：同类节点更可能通过某些 edge type 连接，而另一些 edge type 连接异类；仅看二值图会“混”，但看 edge type 才能分。

### B) 在现有模型上做稳定性/可解释性排查

建议优先做“可定位”的实验：
- 固定某一个数据集（如 `rich_edge_profiles`），扫 `lr / dropout / epochs / heads / n_z`，记录塌缩率与 acc 曲线。
- 记录每轮 `cluster_distribution`、`q/p` 的熵或 KL，定位是“聚类头塌缩”还是“图编码器塌缩”。
- 尝试降低网络规模（hidden size）或减少层数，避免小图上过拟合/数值不稳。

对应的最小可复现实验工具（已加入代码）：
- 超参扫描：`tools/sweep_stability.py#L1`（输出 `aggregate.json`，每个 run 目录下含 `trace.jsonl`）
- 每轮 trace：`sdcn_dlaa_NEW.py#L879`（在 eval 阶段写入 `q/p/pred` 的熵、KL(P||Q)、hard 分簇分布、collapse flag）
- 隐藏层规模（AE/SpatialConv 同步）覆盖：环境变量 `SDCN_ENC_DIMS="256,256,512"`（仅影响 `train_sdcn_dlaa_custom`）

示例（固定 `rich_edge_profiles` 做一个小网格）：
- `python tools/sweep_stability.py --data_dir /tmp/suite_ablation/rich_edge_profiles --out_dir /tmp/sweep_rich_edge_profiles --variants v2edge_single_layer,v3edge_cross_layers --seeds 0,1,2 --epochs 30,60 --lrs 1e-3,5e-4 --dropouts 0.0,0.2 --heads 1,4 --n_z 10,20`

### C) 增加 baseline（更公平地利用高维边特征）

目前 baseline 对 rich_edge 只用到了 `edge_attr[:,0]`（距离）。为了评估“边特征信息量”的理论上界，可新增：
- 由高维 `edge_attr` 映射到权重的谱聚类（如用 `w = exp(-||edge_attr - c||)` 或简单线性组合），作为“非深度”但“用到边特征”的对照。
