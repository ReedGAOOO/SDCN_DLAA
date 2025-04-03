
好的，这是 `preprocess_distance_matrix.py` 脚本的使用指南：

**脚本目的:**

该脚本用于预处理节点特征数据和节点间的距离矩阵，生成适用于图神经网络（如 SDCN_Spatial、DLAA 等）的输入数据。主要功能包括：

1.  加载节点特征和距离矩阵。
2.  根据指定的稀疏化方法（KNN 或 Threshold）从距离矩阵构建图结构（边索引 `edge_index` 和二进制稀疏邻接矩阵 `sparse_adj_binary`）。
3.  提取与图结构对应的原始边距离。
4.  对原始边距离进行归一化处理。
5.  将归一化后的边距离扩展到指定的维度，生成最终的边特征 `edge_attr`。
6.  保存处理后的节点特征、图结构和边特征为 `.npy` 和 `.pt` 文件。
7.  （可选）生成数据可视化图表。

**命令行用法:**

```bash
python preprocess_distance_matrix.py [参数选项]
```

**参数说明:**

*   `--node_features <路径>`:
    *   指定包含节点特征的 CSV 文件路径。
    *   默认值: `'NEWDATA/X_simplize.CSV'`
*   `--distance_matrix <路径>`:
    *   指定包含节点间实际距离的邻接矩阵 CSV 文件路径（通常无表头）。
    *   默认值: `'NEWDATA/A.csv'`
*   `--output_dir <目录>`:
    *   指定保存处理后数据的输出目录。如果目录不存在，脚本会自动创建。
    *   默认值: `'NEWDATA/processed_sparse'`
*   `--method <方法>`:
    *   选择图稀疏化的方法。
    *   可选值:
        *   `'knn'`: 基于 K 最近邻构建图。对于每个节点，只保留指向其最近 K 个邻居的边。**经过我们之前的修改，此方法现在会生成无向图**。
        *   `'threshold'`: 基于距离阈值构建图。只保留距离小于或等于 `--theta` 指定阈值的边。如果输入的距离矩阵是对称的，则生成的图也是无向的。
    *   默认值: `'knn'`
*   `--k <整数>`:
    *   当 `--method` 为 `'knn'` 时使用。指定每个节点保留的最近邻居数量 K。
    *   必须是正整数。
    *   默认值: `10`
*   `--theta <浮点数>`:
    *   当 `--method` 为 `'threshold'` 时使用。指定距离阈值 Theta。
    *   **如果 `--method` 设置为 `'threshold'`，则此参数必须指定**。
    *   默认值: `None`
*   `--normalize <方法>`:
    *   选择对提取出的原始边距离进行归一化的方法。
    *   可选值:
        *   `'minmax'`: 最小-最大归一化，将距离缩放到 [0, 1] 区间。
        *   `'standard'`: 标准化（Z-score），使距离均值为 0，标准差为 1。
        *   `'none'`: 不进行归一化。
    *   默认值: `'minmax'`
*   `--edge_dim <整数>`:
    *   指定最终输出的边特征的目标维度。脚本会基于归一化后的距离（通常是1维）通过预定义的函数（平方、指数衰减、倒数等）扩展到此维度。
    *   默认值: `10`
*   `--visualize`:
    *   可选标志。如果添加此参数，脚本会在输出目录下的 `visualization` 子目录中生成并保存一些数据可视化图表（如特征分布、度分布、邻接矩阵热图等）。
    *   默认不启用。

**使用示例:**

1.  **使用 KNN 方法 (K=15)，不进行归一化，输出到 `output_knn` 目录，并生成可视化:**
    ```bash
    python preprocess_distance_matrix.py \
        --node_features path/to/your/node_features.csv \
        --distance_matrix path/to/your/distance_matrix.csv \
        --output_dir output_knn \
        --method knn \
        --k 15 \
        --normalize none \
        --edge_dim 5 \
        --visualize
    ```

2.  **使用 Threshold 方法 (阈值=0.5)，进行 MinMax 归一化，输出到 `output_threshold` 目录:**
    ```bash
    python preprocess_distance_matrix.py \
        --node_features data/features.csv \
        --distance_matrix data/distances.csv \
        --output_dir output_threshold \
        --method threshold \
        --theta 0.5 \
        --normalize minmax \
        --edge_dim 10
    ```

**输出文件 (在 `--output_dir` 中):**

*   `node_features.npy`, `node_features.pt`: 保存处理后的节点特征。
*   `binary_adj.npz`: 保存二进制稀疏邻接矩阵 (Scipy sparse matrix)。
*   `edge_index.npy`, `edge_index.pt`: 保存图的边索引 (核心结构)。
*   `edge_attr.npy`, `edge_attr.pt`: 保存最终处理（归一化+扩展）后的边特征。
*   `feature_names.txt`: （如果原始节点特征文件有表头）保存节点特征的名称。
*   `data_info.txt`: 包含处理后数据的基本信息（节点数、边数、特征维度、稀疏度等）。
*   `visualization/` (如果使用了 `--visualize`): 包含各种可视化图表文件（.png）。

确保根据你的实际文件路径和需求调整参数。