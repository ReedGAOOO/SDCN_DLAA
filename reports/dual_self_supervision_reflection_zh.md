# 双重自监督聚类：从 DEC/SCAN/CC 到本仓库 SDCN_DLAA（v2/v5/v16）的一次“对齐式”反思

> 目标：把你提供的材料里“**双重自监督**”的核心机制抽象成一个可复用的框架，然后把它一一映射到本仓库当前的实现（尤其是 v2/v5/v16 的 edge↔edge 机制），明确：哪些是“同一范式的不同实现”，哪些是“其实在做另一类自监督”，以及为什么某些版本会稳定、某些会塌缩/不显著。

---

## 0. 我实际读取到的材料

### 0.1 PDF：✅已完整读取并提炼

来源：`Reference/双重自监督机制在聚类任务中的起源与发展.pdf`  
我已将其内容抽取为纯文本（6 页），并基于全文提炼出“二重自监督”的谱系与关键动机（见下文第 1 节）。

### 0.2 分享对话：⚠️当前环境无法直接还原正文

链接：`https://chatgpt.com/share/6981a4e9-2ae8-800b-800f-278080b5e828`  
在 CLI 环境下只能拿到页面壳（HTML/React 组件流），真正的对话消息需要通过浏览器侧 JS 加载，并且 `/continue` 会触发 Cloudflare challenge（403），因此无法可靠提取对话正文。

> 如果你希望我把“分享对话”的观点也纳入反思框架：请你把对话内容（或关键段落）直接粘贴到这里，或导出为纯文本/Markdown。

---

## 1. 从 PDF 提炼出来的“二重自监督”统一框架

PDF 的主旨不是给出一个严格数学定义，而是给出一条**可追溯的设计谱系**：聚类在没有标签时容易遇到两类问题：

1) **表示学不好**：学不到可聚类的表示，或学到“低级捷径”（颜色/噪声/局部纹理）。  
2) **聚类会塌缩**：所有样本被分到一个簇（或少数簇），或者聚类头输出不均衡。

所谓“双重自监督”，核心是用**两类互补的自监督信号**同时约束网络，常见是“局部/实例级” + “全局/簇级”，或“表示学习” + “自训练（pseudo-label/目标分布）”，让模型同时具备：

- **可分性**（区分不同样本/不同簇）
- **一致性**（同一对象的不同视角/近邻应一致）
- **均衡性**（避免塌缩，保持簇占比合理）

---

## 2. 关键范式（按 PDF 的时间线）与“第二个自监督信号”到底在干什么

### 2.1 DEC（2016）：重构 + 目标分布 KL（自训练）

- 自监督 1：AE 重构（让表示保留信息）
- 自监督 2：通过软分配 `Q` 构造更“尖锐”的目标分布 `P`，最小化 `KL(Q || P)`（让聚类逐步自我强化）

**本质**：用“自训练目标分布”把聚类当作另一个监督源。

### 2.2 IMSAT / IIC（2017/2019）：增强一致性 + 互信息最大化

- 自监督 1：同一样本的两种增强，输出应一致（invariance）
- 自监督 2：最大化互信息（鼓励输出既“自信”又“均衡”）

**本质**：用“一致性”抑制噪声/捷径，用“信息最大化”防塌缩并提升可分性。

### 2.3 SCAN（2020）：两阶段（表示预训练） + 近邻一致性 + 熵/均衡正则

PDF 强调 SCAN 的洞察：端到端聚类容易粘在低级特征上，因此先用自监督表示学习预训练，再做聚类细化。

- 自监督 1（阶段 1）：实例级自监督预训练（如对比学习）
- 自监督 2（阶段 2）：近邻一致性 + 全局均衡约束（避免塌缩）

### 2.4 SDCN（2020）：AE 分支 + GCN 分支，用同一个 P 同时监督两路

这是跟本仓库最直接相关的一段：SDCN 把“二重自监督”搬到图聚类中，核心是：

- 分支 A：AE（属性/内容）
- 分支 B：GCN（结构）
- 同一个 `P` 同时监督两路输出（两个 KL），并保留 AE 的重构误差

**关键点**：二重自监督不仅是“两个 loss”，也常是“两个视角/两路网络”。

### 2.5 Contrastive Clustering / SACC（2021~）：实例对比 + 簇对比（原型）

PDF 把这一类总结为：显式同时优化

- 实例级：InfoNCE（区分不同样本）
- 簇级：原型/簇对比（强调全局聚类结构）

并指出加入簇级目标能显著改善“只有实例对比”的不足（只会记住每个样本的差异但不形成语义簇）。

---

## 3. 映射到本仓库：SDCN_DLAA 的“二重/多重自监督”到底是什么

本仓库 `sdcn_dlaa_NEW.py` 的训练目标（按默认习惯）已经是一个“多目标叠加”的体系：

1) **AE 重构**：`re_loss = MSE(x_bar, x)`（DEC/SDCN 谱系）
2) **自训练 KL（q→p）**：`kl_loss = KL(q || p)`，其中 `p = target_distribution(q)`（DEC/SDCN 谱系）
3) **预测头对齐（pred→p）**：`ce_loss = KL(pred || p)`（可理解为“第二个聚类头”的协同自监督）
4) **防塌缩/均衡正则（可选）**：
   - `SDCN_Q_BALANCE_WEIGHT / SDCN_PRED_BALANCE_WEIGHT`
   - `SDCN_Q_MI_WEIGHT / SDCN_PRED_MI_WEIGHT`
   - `SDCN_Q_ENTROPY_WEIGHT / SDCN_PRED_ENTROPY_WEIGHT`
   这些与 PDF 中“熵/互信息/均衡”用于防塌缩的动机是一致的（SCAN/IMSAT/IIC/CC 的那条线）。
5) **边相关自监督（可选）**：
   - `SDCN_EDGE_RE_WEIGHT`：用 `z` 去回归 edge_attr（把“边语义”显式变成重构目标）
   - `SDCN_POOL_RE_WEIGHT`：用 `z` 去回归 per-node mean(edge_attr)
   - `SDCN_EDGE_AUX_WEIGHT`：edge-level 辅助头预测“within-cluster 概率”（在 v6/v15/v16 等 variant 里由 `SpatialConv` 暴露 `_last_edge_within_logit`）

因此，从“二重自监督”的视角看，本仓库其实已经同时具备：

- **重构类自监督**（AE / edge_recon / pool_recon）
- **自训练类自监督**（q→p，pred→p）
- **一致性/均衡类自监督**（MI/balance/entropy 等正则）
- **结构视角**（图分支 vs AE 分支；以及 DLAA 内部 node↔edge↔edge 的局部一致性先验）

---

## 4. v2 / v5 / v16 的 edge↔edge：还是不是“同一个机制”？

结论先行：**三者都使用同一个“edge↔edge 图”（`data.edge_to_edge_index`）作为边之间消息传递的载体**，但“边更新如何进入节点表征/聚类”差别很大，因此表现会非常不一样。

> edge↔edge 图的构建方式由 `sdcn_dlaa_NEW.py::_prepare_pyg_data()` 控制：
> - `SDCN_EE_GRAPH=incidence`：共享端点的边相连（line graph）
> - `SDCN_EE_GRAPH=incidence_sim`：共享端点 + 相似度 topk/min_sim（更保守，见 `reports/edge_edge_debug_dataset_zh.md`）

### 4.1 v2：最小改动修复版（edge↔edge 主要“更新边”，节点主要吃 raw edge_attr）

实现：`DLAA_NEW.py::SpatialConvV2EdgeSingleLayer`

- edge 初始化：`edge_feat = MLP([x_src, x_dst, dist_feat_order])`
- edge↔edge：在 `edge_to_edge_index` 上做 `ee_gat` 得到 refined edge
- node 更新：`SGAT(x, edge_attr=dist_feat)`（**注意力里用 raw dist_feat**）
- **没有 v5 的 pooling residual，也没有 v7/v16 的 edge_attr 融合**

含义：edge↔edge 主要影响“输出的边表征”，它对节点聚类的作用相对间接；如果节点端最终聚类强依赖 edge 信息，v2 往往不够“把 edge↔edge 的收益兑现出来”。

### 4.2 v5：把 edge↔edge 的收益“通过 pooling residual 送回节点”

实现：`DLAA_NEW.py::SpatialConvV5EdgePoolResidual`

仍然保留 v2 的关键取舍：node attention 用 raw edge_attr（避免 edge embedding 早期被洗掉），但额外加了：

- `pooled = mean_pool(dist_feat) + mean_pool(edge_feat_1)`
- `node_out = node_att + gate(node_att, pooled) * proj(pooled)`

含义：edge↔edge 的收益主要通过 **edge→node pooling residual** 进入节点表征；这与 PDF 里“局部一致性信号需要一个路径影响最终聚类”的观点是一致的——否则它只是中间表征的“漂亮但无用的平滑”。

### 4.3 v16：把 edge↔edge 直接写进 node attention（并让 edge↔edge 本身可被显式优化）

实现：`DLAA_NEW.py::SpatialConvV16EdgeEeResidualAuxFusion`

v16 的三个关键改变：

1) **edge↔edge residual**（防止“洗掉 edge_feat_0”）  
   `edge_feat_1 = edge_feat_0 + tanh(scale) * ee(edge_feat_0)`

2) **edge_attr fusion**（edge↔edge 直接调制 node attention）  
   `edge_attr_att = dist_feat + fuse_scale * norm(edge_feat_1)`  
   这相当于把 edge↔edge 从“只做平滑”升级为“直接改注意力里的边条件”。

3) **edge-level 辅助头**（让 edge↔edge 有可对齐的训练信号）  
   暴露 `_last_edge_within_logit`，训练时用 `SDCN_EDGE_AUX_WEIGHT>0` 让它对齐 `same_prob = (p[src]*p[dst]).sum()` 这样的软目标。

含义：v16 更接近“二重自监督里常见的双头/双分支设计”：除了 node 的聚类自训练，还显式让 edge 的表征去拟合“簇内关系”的自监督信号，理论上更符合“互补约束”的路线。

---

## 5. 用“二重自监督”的语言重新审视 edge↔edge：它到底应该扮演什么角色？

从 PDF 的脉络看，一个自监督信号要长期稳定地提升聚类，通常要满足：

1) **互补性**：它约束的是“另一个维度”的错误（局部 vs 全局、结构 vs 属性、实例 vs 簇）。  
2) **可达性**：它必须有一条清晰路径影响最终聚类输出（否则只是中间层的装饰）。  
3) **防捷径/防塌缩**：它最好还能抑制某类常见失败模式（例如簇坍塌、噪声边误混合）。

把这三条套到 edge↔edge：

- 如果 `edge_to_edge_index` 太“激进”（例如 incidence 把同端点的所有边都连起来），在“同簇边 + 跨簇噪声边”共存时会产生**误混合**；这时 edge↔edge 反而会破坏信号（见 `reports/edge_edge_debug_dataset_zh.md` 的诊断结论）。
- 如果 edge↔edge 只更新边表征，但节点聚类完全感受不到它，那么它就违背了“可达性”。v5 通过 pooling residual、v16 通过 edge_attr fusion 解决的就是这个问题。
- 如果 edge↔edge 没有明确的优化目标，它更像“归纳偏置”而不是“自监督信号”。v6/v16 引入 edge_aux，就是把它拉回到“自监督可对齐”的范式里。

---

## 6. 结合 PDF 的谱系：下一步“更像论文、也更可能稳定有效”的方向（供你选）

下面这些方向的共同点：它们都把 edge/node/graph 当作“不同视图”，显式构造第二个（或多个）自监督信号，且目标直指“防塌缩 + 语义聚类”。

1) **Graph 版 IIC/IMSAT**：两种图增强视图（edge dropout / feature masking），最大化两视图的簇分配互信息。  
2) **Graph 版 CC（实例对比 + 簇对比）**：在 node embedding 上做 InfoNCE，同时在 cluster prototype 上做对比，避免“只实例对比不成簇”。  
3) **SCAN 风格两阶段**：先用图自监督预训练（DGI/GraphCL/BGRL 一类），再用“近邻一致性 + 均衡正则”细化聚类头。  
4) **Teacher-Student（动量教师）稳定自训练**：用 EMA teacher 产生更平滑的 soft targets（对应 PDF 提到的“动量教师稳定训练”那条线），缓解 DEC/SDCN 风格自训练的确认偏差。

这些我可以在你确认优先级后，按“改动最小 → 风险最低”的顺序实现到当前代码里，并在合成 suite + 真实数据上做统一消融。

---

## 7. 原论文 SDCN（WWW 2020）到底在坚持什么：从论文 + 作者代码抽取的“设计不变量”

材料来源：

- 论文：`Reference/SDCN_ORIGINAL/SDCN.pdf`
- 代码：`Reference/SDCN_ORIGINAL/sdcn.py`（以及 `Reference/SDCN_ORIGINAL/GNN.py` / `Reference/SDCN_ORIGINAL/utils.py`）

把论文的叙述压缩成“可对照的关键点”，我认为 SDCN 有几条非常明确的设计不变量（你可以把它当作“原教旨”定义）：

### 7.1 Delivery operator（逐层交付）不是装饰：它是为了解决 GCN 的结构性问题

论文 Eq.(7)(8) 的核心：在第 `l` 层，把 GCN 的中间表示 `Z^{l-1}` 与 AE 的中间表示 `H^{l-1}` 做线性混合：

- `\tilde Z^{l-1} = (1-ε) Z^{l-1} + ε H^{l-1}`（论文里 ε=0.5）
- 再用 `\tilde Z^{l-1}` 进入下一层 GCN 卷积（Eq.(8)）

论文的理论分析（Sec.3.5）强调两件事：

1) **GCN 相当于给 AE 表征加了高阶图正则**（近似二阶图正则 / common-neighbor similarity）。  
2) **Delivery operator 缓解 over-smoothing**：没有交付时，多层 GCN 很容易“把表示抹平”，尤其在 KNN 图上更明显。

这条原则的含义是：**逐层注入 AE 表征**不是“随便加个 skip”，而是整个方法把结构信息引入深度聚类的关键接口。

### 7.2 Dual self-supervised module（双自监督）在做什么：P 同时监督 Q 与 Z

论文定义（Sec.3.4）：

- 用 AE 最后一层表示 `H^L` 计算 `Q`（Student-t, Eq.(11)）
- 由 `Q` 构造目标分布 `P`（Eq.(12)）
- `L_clu = KL(P || Q)`（Eq.(13)）：让 AE 表征聚向簇中心（“自训练/自监督”）
- `L_cn  = KL(P || Z)`（Eq.(14)）：用同一个 `P` 去“温和地”监督 GCN 输出分布 `Z`

关键点：论文明确说 KL 的好处是“更 gentle”，而且把 AE 与 GCN 统一到同一个 target `P` 上，形成“强耦合”。

### 7.3 总损失与权重：L = L_res + α L_clu + β L_cn（α=0.1, β=0.01）

论文 Eq.(15)；作者代码 `Reference/SDCN_ORIGINAL/sdcn.py` 里对应：

- `loss = re_loss + 0.1 * kl_loss + 0.01 * ce_loss`
- 其中 `kl_loss = KL(q || p)`、`ce_loss = KL(pred || p)`（实现上用 `F.kl_div(log(), p)`）

### 7.4 “最终用谁输出聚类结果”并非一刀切：Z 是默认，但 Q 在噪声图上可能更好

论文算法（Algorithm 1, Step 18）写的是：最终用 `Z`（GCN 输出分布）做聚类标签。

但论文实验分析（Sec.4.3）也明确提到：在 Reuters 这种 **KNN 图很噪** 的场景，`SDCN Q`（用 Q 的 variant）反而显著更好——因为错误结构信息会误导 GCN。

这点很关键：SDCN 从来不是“结构一定好”，它要求 **图结构足够干净/有意义** 才能让结构正则带来净收益。

### 7.5 训练 recipe：预训练 AE + KMeans 初始化中心

论文 Sec.4.2 的参数设置，以及作者代码都采用：

- 先单独训练 AE（30 epoch）
- 用预训练 AE 的 `z` 做 KMeans 初始化聚类中心 `μ`
- 再进入端到端训练

这不是形式主义：它让初始 `Q/P` 至少不是纯噪声，否则自训练很容易产生确认偏差并触发塌缩。

---

## 8. 回看本仓库：哪些地方是“对齐/扩展”，哪些地方确实容易背离 SDCN 原理

下面的判断标准是：**是否仍然满足第 7 节的不变量**，以及是否引入了与原假设冲突的新信号。

### 8.1 明确对齐（仍在 SDCN 范式内的扩展）

在 `sdcn_dlaa_NEW.py` 中，整体训练结构仍然与论文/原代码高度同构：

- **delivery operator**：`data.x = (1-sigma)*h + sigma*tra`（逐层把 AE 中间表征注入图分支；σ 默认 0.5）。  
- **dual self-supervision**：`p = target_distribution(q)`，然后 `KL(q||p)` 与 `KL(pred||p)` 两路对齐 + `re_loss`。  
- **损失权重默认一致**：默认仍是 `0.1*KL + 0.01*CE + RE`。  
- **输出也允许用图分支 pred**（对应论文的 Z 作为最终输出）。

因此，从“是不是还叫 SDCN”角度看：**大体没有背离**；你做的是“把 GCN 换成更强的 message passing（SpatialConv/attention），并加入 edge 相关 inductive bias”。

### 8.2 明确可能背离（会改变 SDCN 的核心动力学/假设）

下面这些改动并不是“不可以做”，但它们确实改变了 SDCN 的关键假设；如果不加约束，表现为“不稳定/不显著/塌缩”，就容易让实验看起来像是“背离了双重自监督”。

#### (A) 取消/弱化 AE 预训练（或等价地：让初始 Q/P 太噪）

原 SDCN 强依赖“预训练 AE + KMeans init”。  
而 `sdcn_dlaa_NEW.py` 当前代码路径里**没有加载预训练权重**（对比 `Reference/SDCN_ORIGINAL/sdcn.py` 的 `load_state_dict(args.pretrain_path)`）。

影响：训练初期的 `q/p` 极不稳定，后续你不得不靠更强的 balance/entropy/MI 正则与各种 trick 来“硬拉住”，这会把研究重心从 SDCN 的原理转移到“防塌缩工程”。

#### (B) 改变 P 的来源（`SDCN_Q_SOURCE != z`）

论文定义里，`Q/P` 明确由 AE 的 `H^L`（z）产生；它是“AE→(P)→监督 AE+GCN”的耦合。

当你设置：

- `SDCN_Q_SOURCE=h4` / `pool` / `fused`

本质上是在改变 teacher：`P` 不再是“纯 AE 表征的自训练目标”，而变成“图分支/边统计/混合表征的自训练目标”。这会改变两路一致性的来源，可能带来：

- 更适合 edge-driven/节点特征弱的诊断集（这是你报告里观察到的）
- 也可能带来更强的确认偏差（尤其当同一分支既产生 Q 又被 P 监督时）

所以这不是简单的“超参”，而是 **范式变体**：建议把它当作 ablation 的第一主轴，而不是默认随便改。

#### (C) 结构假设从“同质图”变成“带噪/异质/边语义主导”

原 SDCN 强调：KNN 图噪声会让结构信息“有害”，Reuters 上 SDCN Q 更好就是例证。  
而本仓库大量实验是“edge 语义主导 + 结构可能是 nonknn/噪声”，这已经超出了论文默认假设。

这并不意味着你做错了，反而意味着你在研究更一般的 setting（attributed edges / noisy structure）。但在这种 setting 下：

- **edge↔edge 图构建必须保守**（`incidence_sim + min_sim` 更像是在恢复论文要求的“结构干净”前提）
- **final_assign 选择 q/p**（在噪声结构下更稳）并不背离论文，反而是呼应论文的分析

#### (D) 额外自监督头（edge_aux / edge_recon / pool_recon）权重过大

从“二重自监督”的大框架看，增加第三/第四个自监督任务完全合理（PDF 里也提到 DCCM/CC/SACC 的多重约束）。

但从 SDCN 原理看，所有额外任务都应该满足第 5 节提的三条（互补性/可达性/防塌缩），否则就会出现：

- 额外任务学到了“捷径”，反向牵引主聚类目标
- loss 之间梯度冲突，使得 `P` 无法作为统一 target 产生凝聚力

因此额外 head 最好都配合：小权重 + warmup + 明确的 ablation（否则很难判断到底是“结构创新有效”还是“正则恰好压住塌缩”）。

---

## 9. 一个更“原教旨 + 可验证”的自检清单（建议你用来判断是否背离）

如果你希望“实验版仍然是 SDCN 的延伸，而不是另起炉灶”，我建议每个新 variant 都回答下面 5 个问题：

1) **P 的来源是谁？**（AE z / graph h4 / pooled edge / fused）  
2) **delivery operator 是否仍逐层发生？σ 是否为 0？**（σ=0 基本等价于砍掉核心接口）  
3) **最终输出用谁？pred(Z)/q/p？在噪声图上是否出现“SDCN Q 更好”的同类现象？**  
4) **结构是否“干净”？**（incidence 是否会误混合；incidence_sim/min_sim 是否显著改善）  
5) **额外自监督是否可达且互补？**（能否证明它通过 fusion/pooling/aux head 真实影响最终分配，而不是只美化中间表征）

只要这 5 条能自洽，你的工作通常不是“背离”，而是“带约束的扩展”；反之，很多“掉点/塌缩/随机性”其实是因为违背了 SDCN 在论文里反复强调的先决条件（图噪声、over-smoothing、Q/P 初始化）。
