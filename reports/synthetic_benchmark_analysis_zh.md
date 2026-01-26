# 合成数据多轮测试分析（v2/v3 vs Baselines）

本报告基于 `tools/generate_synthetic_suite.py` 生成的合成数据集套件，对比：
- **模型**：`v2edge_single_layer`、`v3edge_cross_layers`
- **传统 baseline**：`kmeans_x`、`spectral_adj_binary`、`spectral_edge_distance`（只用 `edge_attr[:,0]` 作为“距离”权重）

原始跑分表见：
- `reports/synthetic_benchmark_report_zh.md`（suite_seed=0, epochs=30）
- `reports/synthetic_benchmark_report_zh_epochs60.md`（suite_seed=0, epochs=60）
- `reports/synthetic_benchmark_report_zh_suite_seed1.md`（suite_seed=1, epochs=30）

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

## 对“为什么暂时看不到 v3 明显优势”的解释

结合当前实现（`sdcn_dlaa_NEW.py` 的 SDCN 训练逻辑 + `DLAA_NEW.py` 的 v2/v3 信息流）和现象，主要有两类原因：

1. **任务上界/数据设计原因**：rich_edge 数据仍然以“空间距离构图”为主，图结构本身已经很可分，导致传统谱聚类很强；要验证“边特征真正带来提升”，需要构造“距离/拓扑无法区分，但边语义可区分”的场景。
2. **训练稳定性原因**：SDCN 自监督聚类在小数据/强噪声/强随机边场景下更容易出现 q/p 的塌缩或聚类中心漂移；v2/v3 虽然增强了 edge feature 的参与，但并不自动解决自监督聚类的稳定性问题（需要更细的超参/预训练策略/正则化）。

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

### C) 增加 baseline（更公平地利用高维边特征）

目前 baseline 对 rich_edge 只用到了 `edge_attr[:,0]`（距离）。为了评估“边特征信息量”的理论上界，可新增：
- 由高维 `edge_attr` 映射到权重的谱聚类（如用 `w = exp(-||edge_attr - c||)` 或简单线性组合），作为“非深度”但“用到边特征”的对照。
