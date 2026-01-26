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
