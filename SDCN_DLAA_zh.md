
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
    python preprocess_distance_matrix.py \
        --node_features NEWDATA/X_simplize.CSV \
        --distance_matrix NEWDATA/A.csv \
        --output_dir NEWDATA/processed_knn_k15 \
        --method knn \
        --k 15 \
        --normalize minmax \
        --edge_dim 10 \
        --visualize
    ```

2.  **使用 Threshold 方法 (阈值 theta=0.5):**
    *(注意: `theta` 的值需要根据数据实际情况调整)*

    ```bash
    python preprocess_distance_matrix.py \
        --node_features NEWDATA/X_simplize.CSV \
        --distance_matrix NEWDATA/A.csv \
        --output_dir NEWDATA/processed_threshold_0.5 \
        --method threshold \
        --theta 0.5 \
        --normalize minmax \
        --edge_dim 10 \
        --visualize
    ```

处理完成后，生成的 `.pt` 文件可直接用于后续的模型训练和评估脚本。