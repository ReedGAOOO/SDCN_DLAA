## SDCN_DLAA:Structural Deep Clustering Network with Dual-Level Attentive Aggregation Mechanism

## INTRODUCTION

创新点一：融合深度边信息建模的图自监督聚类框架(sdcn_dlaa_NEW.py)
1.本项目创新性地将双层图聚合机制（源自 SMAN的 SpatialConv 思想）引入图自监督聚类框架 (SDCN)。该机制通过显式的、迭代的节点↔边和边↔边信息传递，实现了对边特征的深度建模与动态利用。这显著区别于主流 GNN 层（如 PyG 中的 GATConv/GCNConv）通常采用的边信息静态处理（如仅用于注意力计算或简单拼接）方式。
2.这种深度边建模使模型能够更精确地捕捉由连接强度、类型或属性定义的复杂节点间关系，从而进行更科学、更符合底层图结构的节点聚类。该框架特别适用于地理空间网络分析（如轨迹点聚类，考虑连接道路等级、通行时间等）、社交网络挖掘（分析用户社群，考虑关系类型、亲密度）、分子属性预测（考虑化学键类型、键能）等边信息具有丰富语义的应用场景。

创新点二：PyTorch Geometric 框架下的双层聚合实现与应用（DLAA_NEW.py）
本项目首次在 PyTorch Geometric (PyG) 框架下完整、高效地实现了双层图聚合机制（核心思想来自SMAN模型）。针对 PyG 的数据表示和消息传递（定制message passing）特性进行了适配与优化 (如内存优化、并行化改进)，为在主流 PyG 生态中应用此类交互式节点-边联合建模技术提供了重要的实现基础和应用范例，特别是在自监督聚类任务中验证了其有效性。

这个项目依然是一份实验性质的，目前正处于解决训练时的学习率不稳定的情况（参数调整阶段）。但是理论上证明了SDCN_DLAA框架的可行性（详细请见engineering note https://docs.google.com/document/d/1qZmEbDUiWt8VqjI-uQMnlMk5Skk1pxMWrIq58-LcI8I/edit?usp=sharing），并且从目前在test_sdcn_dlaa_NEW_sparse.py上运行的过程来看，其已经表现出了能够将EDGE FEATURE融入NODE CLUSTERING中，并且捕获有意义的信息的能力。
## QUICK START
以下是快速运行模型的基本步骤：

1.  **数据预处理 (使用 KNN, K=10):**

    ```bash
    python preprocess_distance_matrix.py --output_dir NEWDATA/processed_knn_k10 --method knn --k 10
    ```

2.  **模型测试 (使用 KNN 预处理的数据):**

    ```bash
    python test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10
    ```

*注意: 请确保 `NEWDATA/X_simplize.CSV` 和 `NEWDATA/A.csv` 文件存在于默认路径，或使用 `--node_features` 和 `--distance_matrix` 参数指定路径。详细参数请参考后续章节。*

## 数据预处理

在使用模型进行训练或测试之前，需要使用 `preprocess_distance_matrix.py` 脚本对原始数据进行预处理。该脚本主要完成以下任务：

1.  **加载数据**: 读取节点特征（例如 `NEWDATA/X_simplize.CSV`）和节点间的距离矩阵（例如 `NEWDATA/A.csv`）。
2.  **图构建**: 根据指定的稀疏化方法（KNN 或 Threshold）从距离矩阵构建图结构。
    *   **KNN**: 为每个节点保留 K 个最近邻居，生成无向图。
    *   **Threshold**: 保留距离小于或等于指定阈值 `theta` 的边。
3.  **边特征生成**: 提取与图结构对应的原始边距离，进行归一化（可选），并扩展到指定的维度，生成最终的边特征。
4.  **保存输出**: 将处理后的节点特征、图结构（边索引 `edge_index`、稀疏邻接矩阵 `binary_adj.npz`）和边特征 (`edge_attr`) 保存到指定的输出目录中，格式为 `.npy` 和 `.pt` 文件。同时会生成一个 `data_info.txt` 文件记录数据信息。

### 使用方法

通过命令行运行脚本，并指定相关参数。

**输入文件:**

*   `--node_features`: 节点特征矩阵的 CSV 文件路径 (默认: `NEWDATA/X_simplize.CSV`)。
*   `--distance_matrix`: 包含节点间实际距离的邻接矩阵 CSV 文件路径 (默认: `NEWDATA/A.csv`)。

**主要参数:**

*   `--output_dir`: 指定保存处理后数据的目录。
*   `--method`: 图稀疏化方法，可选 `'knn'` 或 `'threshold'` (默认: `'knn'`)。
*   `--k`: 当 `method='knn'` 时，指定 K 值 (默认: `10`)。
*   `--theta`: 当 `method='threshold'` 时，指定距离阈值 (必须提供)。
*   `--normalize`: 边距离归一化方法，可选 `'minmax'`, `'standard'`, `'none'` (默认: `'minmax'`)。
*   `--edge_dim`: 最终边特征的目标维度 (默认: `10`)。
*   `--visualize`: 添加此标志以生成数据可视化图表。

**示例命令:**

1.  **使用 KNN 方法 (K=15):**

    ```bash
    python preprocess_distance_matrix.py `
    --node_features NEWDATA/X_simplize.CSV `
    --distance_matrix NEWDATA/A.csv `
    --output_dir NEWDATA/processed_knn_k15 `
    --method knn `
    --k 15 `
    --normalize minmax `
    --edge_dim 10 `
    --visualize
    ```

2.  **使用 Threshold 方法 (阈值 theta=0.5):**
    *(注意: `theta` 的值需要根据数据实际情况调整)*

    ```bash
    python preprocess_distance_matrix.py `
        --node_features NEWDATA/X_simplize.CSV `
        --distance_matrix NEWDATA/A.csv `
        --output_dir NEWDATA/processed_threshold_0.5 `
        --method threshold `
        --theta 0.5 `
        --normalize minmax `
        --edge_dim 10 `
        --visualize
    ```

处理完成后，生成的 `.pt` 文件可直接用于后续的模型训练和评估脚本。

## 模型测试

预处理完成后，可以使用以下脚本对 SDCN-DLAA 模型进行测试，这些脚本会加载预处理后的数据：

*   **`test_sdcn_dlaa_NEW_sparse_KNN.py`**: 用于测试基于 **KNN** 方法预处理的数据。
*   **`test_sdcn_dlaa_NEW_sparse_threshold.py`**: 用于测试基于 **Threshold** 方法预处理的数据。

### 使用方法

通过命令行运行相应的测试脚本。两个脚本都接受一个关键参数 `--data_dir` 来指定包含预处理数据（`node_features.npy`, `binary_adj.npz`, `edge_attr.npy`）的目录。

**主要参数:**

*   `--data_dir`: 包含预处理数据的目录路径。
*   `--lr`: 学习率 (默认: `1e-3`)。
*   `--n_clusters`: 目标聚类数 (默认: `3`)。
*   `--n_z`: 嵌入维度 (默认: `10`)。
*   `--dropout`: Dropout 比率 (默认: `0.2`)。
*   `--heads`: GAT 中的注意力头数 (默认: `4`)。
*   `--edge_dim`: 输入的边特征维度 (应与预处理脚本中的 `--edge_dim` 匹配, 默认: `10`)。
*   `--max_edges_per_node`: 在构建边到边图时每个节点考虑的最大边数 (默认: `10`)。

**示例命令:**

1.  **测试 KNN 数据 (假设数据在 `NEWDATA/processed_knn_k15`):**

    ```bash
    python test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k15
    ```

2.  **测试 Threshold 数据 (假设数据在 `NEWDATA/processed_threshold_0.5`):**

    ```bash
    python test_sdcn_dlaa_NEW_sparse_threshold.py --data_dir NEWDATA/processed_threshold_0.5
    ```

脚本将加载指定目录中的数据，训练模型，并在日志文件（位于 `logs/` 目录下，文件名包含时间戳和方法名）和控制台中输出训练过程和聚类结果。


## EXP TEST CODE

本部分介绍用于实验性测试和分析的代码变体，主要用于解决使用稠密图作为输入时导致的OOM问题。

### 混合精度训练 (AMP)

*   **相关文件**: `sdcn_dlaa_NEW_amp.py`, `test_sdcn_dlaa_NEW_amp.py`, `run_batch_test_amp.py`
*   **目的**: 这些带有 `_amp` 后缀的文件利用了自动混合精度 (Automatic Mixed Precision, AMP) 技术进行训练。AMP 使用较低精度的浮点数（如 FP16）进行部分计算，同时保持关键部分的 FP32 精度，从而在不显著牺牲模型性能的情况下，有效减少 GPU 内存占用和计算时间。这对于处理节点数量多、边连接稠密的图数据尤其有用，可以缓解内存瓶颈问题。

### 异构图卷积 (HeteroConv)

*   **相关文件**: `DLAA_NEW_hetero.py`, `sdcn_dlaa_NEW_hetero.py`, `test_sdcn_dlaa_NEW_hetero.py`
*   **目的**: 这些带有 `_hetero` 后缀的文件采用了 PyTorch Geometric (PyG) 库中的 `HeteroConv` 模块。与原始模型中使用的 `SpatialConv`（隐式建模边信息的卷积层）不同，`HeteroConv` 允许显式地定义和处理不同类型的节点和边及其关系。这提供了更灵活的方式来建模图中复杂的交互，可能有助于捕捉更精细的结构信息。通过重塑 `SpatialConv` 的隐式关系建模方式，探索不同的图信息聚合策略。同时使用`HeteroConv`将规避`SpatialConv`中拼接node_edge向量过大，导致OOM的问题。

### 参数敏感性测试

为了定位和理解模型在处理稠密图时可能遇到的内存瓶颈，设计了以下测试脚本：

*   **Batch Test (GAT Heads)**
    *   **相关文件**: `run_batch_test.py`, `run_batch_test_amp.py`
    *   **目的**: 这些脚本通过系统性地改变图注意力网络 (GAT) 层中的注意力头数 (`heads` 参数) 来进行一系列测试。改变 `heads` 会影响模型的复杂度和计算量，运行这些测试有助于分析不同注意力头数对模型性能和内存消耗的影响，特别是在处理大规模或稠密图时的表现。
*   **Hidden Size Test**
    *   **相关文件**: `run_hidden_size_test.py`, `sdcn_dlaa_NEW_hiddensize.py`, `test_sdcn_dlaa_NEW_hiddensize.py`
    *   **目的**: 这些脚本专注于测试模型内部不同层隐藏表示的维度 (`hidden_size` 参数) 对性能和资源消耗的影响。通过改变这些中间向量的大小，可以评估模型容量与内存需求之间的权衡，帮助确定在处理特定规模（尤其是稠密）图数据时最优或可行的隐藏层维度配置。

## 理论分析：
1. SDCN 自监督聚类核心原理回顾
原始的 SDCN（以及许多深度聚类方法）的自监督核心在于构建一个高置信度的目标分布 P 来指导当前软聚类分配 Q 的学习。其流程通常如下：

获取节点表示 (Latent Representation z): 通过 Autoencoder (AE) 提取节点自身特征的低维表示 z，同时通过 GNN (如图卷积 GCN) 提取考虑了邻域结构的表示 h。z 和 h 可能以某种方式融合（例如相加、拼接后降维），或者直接使用 AE 的 z 作为聚类输入。
计算当前软分配 (Soft Assignment q): 基于节点表示 z 和一组可学习的聚类中心 μ，使用某种相似度度量（如 Student's t-distribution）计算每个节点 i 属于每个簇 j 的概率 q_ij。 q_ij = (1 + ||z_i - μ_j||^2 / ν)^(-(ν+1)/2) / Σ_{k} (1 + ||z_i - μ_k||^2 / ν)^(-(ν+1)/2) [Source 105]
构建目标分布 (Target Distribution p): 通过对当前软分配 q 进行锐化（sharpening）来生成一个更高置信度的目标分布 p_ij。通常做法是提高高概率分配的权重，降低低概率分配的权重。 p_ij = (q_ij^2 / Σ_i q_ij) / Σ_{k} (q_ik^2 / Σ_i q_ik) [Source 105, target_distribution function]
自监督损失 (Self-Supervision Loss): 核心损失是 KL 散度损失 L_kl，用于最小化当前分配 q 与目标分布 p 之间的差异。这驱使模型学习到的节点表示 z 在特征空间中形成更紧凑、分离度更好的簇。 L_kl = KL(P || Q) = Σ_i Σ_j p_ij * log(p_ij / q_ij) [Source 105]
辅助损失:
AE 重建损失 L_rec: 保证 z 能够保留节点原始特征信息。 L_rec = ||x - x_bar||^2 [Source 105]
GNN 预测一致性损失 L_ce (可选): 让 GNN 的直接预测输出 predict 也去拟合目标分布 p。 L_ce = KL(P || Predict) [Source 105]
关键点: 这个自监督过程的核心在于 q 和 p 的相互迭代优化。p 为 q 提供了学习目标，而 q 的改进又会生成更好的 p。这个循环的有效性依赖于节点表示 z 的质量——它需要既能保留节点内容信息，又能反映图的结构信息。
2. 引入边特征 (DLAA/SpatialConv) 的影响
在 SDCN-DLAA 中，主要的改动是用 SpatialConv 替换了原始 SDCN 中可能使用的标准 GCN/GAT 层。SpatialConv 的核心是深度、动态地建模边特征。
信息来源的变化: SpatialConv 层输出的节点表示 h1, h2, h3, h4, h5 不仅考虑了邻居节点的特征，还显式地、动态地考虑了连接边的特征和状态（通过 Node↔Edge, Edge↔Edge, Edge→Node 的交互）。
融合表示的变化: 这些由 SpatialConv 产生的、蕴含了更丰富边信息的结构表示 h，会与 AE 产生的表示 tra1, tra2, tra3, z 进行融合（通常是加权求和）。
最终聚类输入 z 的变化: 虽然最终用于计算 q 的还是名为 z 的潜在表示（或者是一个融合了 z 的表示），但这个 z 间接受到了来自 SpatialConv 增强的结构信息的影响。
3. 论证有效性：为何自监督依然有效？
核心自监督循环未变: 最关键的自监督机制——即从节点表示 z 计算 q，从 q 推导目标 p，再用 L_kl(P||Q) 来优化模型参数（包括 AE 和 GNN/SpatialConv 的参数以改进 z）——这个核心循环保持不变。模型依然在通过最小化 q 和 p 的差异来学习聚类结构。
输入表示质量的提升: DLAA/SpatialConv 的目的是提供更高质量、信息更丰富的结构表示 h。通过深度建模边特征，h 能够更精确地反映节点间的复杂关系（不仅仅是连接关系，还有连接的性质）。当这些更优的 h 融合到最终用于聚类的表示 z 中时，理论上应该使得 z 更能区分那些在结构上（现在包括边定义的结构）应该分开的节点。
自监督目标的增强: 一个更高质量的 z 会产生一个初始区分度可能更好的 q。基于这个更好的 q 生成的目标 p 也会更可靠。因此，KL 散度损失 L_kl 会在一个更优化的起点上指导模型参数的更新，有望收敛到更准确、更符合图真实结构（包括边信息）的聚类结果。
重建损失的约束: AE 的重建损失 L_rec 依然存在，它确保了即使 GNN 部分引入了复杂的边信息，模型也不会丢失节点自身的基本特征信息，保持了表示的基础有效性。

## Feference
### SDCN 
PAPER https://arxiv.org/abs/2002.01633
GITHUB LINKS https://github.com/bdy9527/SDCN

### SMAN 
PAPER https://arxiv.org/abs/2012.09624
GITHUB LINKS https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/drug_target_interaction/sman