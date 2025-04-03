#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预处理脚本：将包含实际距离的邻接矩阵转换为边特征(edge_attr)和稀疏邻接图结构
适用于SDCN_Spatial和DLAA模型

输入:
- 节点特征矩阵 (CSV文件，如X_simplize.CSV)
- 距离邻接矩阵 (CSV文件，如A.csv，包含节点间的实际距离)

输出:
- 处理后的节点特征矩阵 (numpy数组)
- 图结构的二进制邻接矩阵 (scipy稀疏矩阵) - 用于兼容或可视化
- 边索引 (numpy数组 [2, num_edges]) - 核心图结构
- 边特征矩阵 (numpy数组 [num_edges, edge_dim]) - 基于距离生成
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_undirected
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_node_features(file_path):
    """加载节点特征矩阵"""
    print(f"正在加载节点特征: {file_path}")
    df = pd.read_csv(file_path)
    print(f"节点特征数据集形状: {df.shape}")
    print(f"节点特征列名: {df.columns.tolist()}")
    # print(f"前5行数据:\n{df.head()}") # Optional: Less verbose
    feature_names = df.columns.tolist()
    node_features = df.values
    print(f"节点特征矩阵形状: {node_features.shape}")
    return node_features, feature_names


def load_distance_matrix(file_path):
    """加载距离邻接矩阵"""
    print(f"正在加载距离邻接矩阵: {file_path}")
    df = pd.read_csv(file_path, header=None)
    print(f"距离邻接矩阵形状: {df.shape}")
    # print(f"前5行5列数据:\n{df.iloc[:5, :5]}") # Optional: Less verbose
    distance_matrix = df.values
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "距离矩阵必须是方阵"
    print(f"距离邻接矩阵形状: {distance_matrix.shape}")
    return distance_matrix

# --- Method 1: Threshold-based Graph Creation ---
def create_threshold_graph(distance_matrix, theta):
    """
    从距离矩阵创建基于阈值的稀疏图 (保留 <= theta 的边)

    Args:
        distance_matrix: 距离邻接矩阵 (NumPy)
        theta: 距离阈值

    Returns:
        sparse_adj_binary: 二进制邻接矩阵 (scipy稀疏矩阵)
        edge_index: 边索引 [2, num_edges] (NumPy)
        edge_attr_dist: 边上的原始距离 [num_edges, 1] (NumPy)
    """
    print(f"正在创建基于阈值的图 (theta={theta})")
    num_nodes = distance_matrix.shape[0]

    # 找到距离在 (0, theta] 范围内的边
    # 注意：我们排除距离为0的边（通常是对角线）
    rows, cols = np.where((distance_matrix > 0) & (distance_matrix <= theta))

    if rows.size == 0:
         print("警告：在当前阈值下没有找到边！")
         # 返回空图结构
         return sp.csr_matrix((num_nodes, num_nodes)), np.zeros((2, 0), dtype=int), np.zeros((0, 1))

    # 提取对应的距离作为初始边特征
    distances = distance_matrix[rows, cols]

    # 创建边索引
    edge_index = np.vstack((rows, cols))

    # 创建原始距离边特征
    edge_attr_dist = distances.reshape(-1, 1)

    # 创建二进制稀疏邻接矩阵 (用于保存或可视化)
    sparse_adj_binary = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))

    print(f"基于阈值的图 - 边数量: {edge_index.shape[1]}")
    sparsity = 1.0 - (sparse_adj_binary.nnz / (num_nodes * num_nodes))
    print(f"邻接矩阵稀疏度: {sparsity:.6f}")

    return sparse_adj_binary, edge_index, edge_attr_dist


# --- Method 2: K-Nearest Neighbors (KNN) Graph Creation ---
def create_knn_graph(distance_matrix, k=10):
    """
    从距离矩阵创建KNN图 (只保留每个节点最近的k个邻居)。
    注意：生成的图可能不是对称的，这里只保留 i -> j 的边，如果 j 是 i 的KNN。

    Args:
        distance_matrix: 距离邻接矩阵 (NumPy稠密矩阵)
        k: 每个节点保留的最近邻数量

    Returns:
        sparse_adj_binary: 二进制邻接矩阵 (scipy稀疏矩阵)
        edge_index: 边索引 [2, num_edges] (NumPy)
        edge_attr_dist: 边上的原始距离 [num_edges, 1] (NumPy)
    """
    print(f"正在创建KNN图 (k={k})")
    num_nodes = distance_matrix.shape[0]
    rows = []
    cols = []
    distances = []

    # 复制一份距离矩阵，并将对角线设为无穷大，以忽略自环
    dist_matrix_no_diag = distance_matrix.copy()
    np.fill_diagonal(dist_matrix_no_diag, np.inf)

    for i in range(num_nodes):
        # 找到第i个节点的所有距离，并获取排序后的索引
        neighbor_distances = dist_matrix_no_diag[i, :]
        nearest_indices = np.argsort(neighbor_distances) # 升序排序

        # 选择最近的k个邻居
        selected_neighbors = nearest_indices[:k]

        # 添加边和距离
        for neighbor_idx in selected_neighbors:
           # 确保距离是有效的 (非原始对角线处的inf)
           dist = neighbor_distances[neighbor_idx] # 使用修改后的距离检查
           if np.isfinite(dist):
                rows.append(i)
                cols.append(neighbor_idx)
                distances.append(distance_matrix[i, neighbor_idx]) # 存储原始距离

    if not rows:
         print("警告：KNN图未生成任何边！检查 K 值或距离矩阵。")
         return sp.csr_matrix((num_nodes, num_nodes)), np.zeros((2, 0), dtype=int), np.zeros((0, 1))

    # 创建边索引
    edge_index = np.vstack((rows, cols))

    # 创建原始距离边特征
    edge_attr_dist = np.array(distances).reshape(-1, 1)

    # 创建二进制稀疏邻接矩阵 (用于保存或可视化)
    sparse_adj_binary = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))

    print(f"KNN图 - 边数量: {edge_index.shape[1]}")
    average_degree = edge_index.shape[1] / num_nodes
    print(f"平均度: {average_degree:.2f}")
    sparsity = 1.0 - (sparse_adj_binary.nnz / (num_nodes * num_nodes))
    print(f"邻接矩阵稀疏度: {sparsity:.6f}")

    return sparse_adj_binary, edge_index, edge_attr_dist


def normalize_edge_features(edge_attr, method='minmax'):
    """归一化边特征 (输入应为原始距离)"""
    print(f"正在使用 {method} 方法归一化边特征 (原始距离)")

    if edge_attr.shape[0] == 0:
        print("边特征为空，跳过归一化。")
        return edge_attr

    if method == 'none':
        return edge_attr

    if method == 'minmax':
        min_val = np.min(edge_attr)
        max_val = np.max(edge_attr)
        # 避免除以零
        if max_val == min_val:
             print("警告：所有边特征值相同，MinMax归一化结果为0。")
             normalized_edge_attr = np.zeros_like(edge_attr)
        else:
             normalized_edge_attr = (edge_attr - min_val) / (max_val - min_val)
    elif method == 'standard':
        mean_val = np.mean(edge_attr)
        std_val = np.std(edge_attr)
         # 避免除以零
        if std_val == 0:
            print("警告：所有边特征值相同，标准化结果为0。")
            normalized_edge_attr = np.zeros_like(edge_attr)
        else:
            normalized_edge_attr = (edge_attr - mean_val) / std_val
    else:
        raise ValueError(f"不支持的归一化方法: {method}")

    print(f"归一化前边特征(距离)范围: [{np.min(edge_attr):.4f}, {np.max(edge_attr):.4f}]")
    if edge_attr.shape[0] > 0:
        print(f"归一化后边特征范围: [{np.min(normalized_edge_attr):.4f}, {np.max(normalized_edge_attr):.4f}]")

    return normalized_edge_attr


def expand_edge_features(normalized_edge_attr, dim=10):
    """扩展边特征维度 (输入应为归一化后的特征)"""
    print(f"正在扩展边特征维度到 {dim}")

    if normalized_edge_attr.shape[0] == 0:
        print("边特征为空，跳过扩展。")
        return np.zeros((0, dim))
    if normalized_edge_attr.shape[1] == dim:
        print("边特征维度已满足要求，无需扩展。")
        return normalized_edge_attr
    if normalized_edge_attr.shape[1] != 1 :
        print(f"警告：期望输入1维特征进行扩展，但收到了{normalized_edge_attr.shape[1]}维特征。将只使用第一维。")


    if dim == 1:
        return normalized_edge_attr[:, 0:1] # 确保是 [N, 1]

    num_edges = normalized_edge_attr.shape[0]
    expanded_edge_attr = np.zeros((num_edges, dim))

    # 获取第一维特征（假设是归一化后的距离或其表示）
    base_feature = normalized_edge_attr[:, 0].flatten()

    # 将基础特征放在第一列
    expanded_edge_attr[:, 0] = base_feature

    # 使用基础特征的函数生成其他特征
    for i in range(1, dim):
        if i % 3 == 0:
            expanded_edge_attr[:, i] = np.power(base_feature, 2) # 平方
        elif i % 3 == 1:
            expanded_edge_attr[:, i] = np.exp(-base_feature * 5) # 指数衰减 (乘以5增加区分度)
        else:
            # 避免除以0，加上一个小值
            expanded_edge_attr[:, i] = 1.0 / (base_feature + 1e-6) # 倒数

    print(f"扩展后的边特征形状: {expanded_edge_attr.shape}")
    return expanded_edge_attr


def save_processed_data(output_dir, node_features, sparse_adj_binary, edge_index, edge_attr_final, feature_names=None):
    """保存处理后的数据"""
    print(f"正在保存处理后的数据到 {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 保存节点特征
    np.save(os.path.join(output_dir, 'node_features.npy'), node_features)
    torch.save(torch.FloatTensor(node_features), os.path.join(output_dir, 'node_features.pt'))

    # 保存二进制邻接矩阵 (稀疏)
    sp.save_npz(os.path.join(output_dir, 'binary_adj.npz'), sparse_adj_binary)

    # 保存边索引
    np.save(os.path.join(output_dir, 'edge_index.npy'), edge_index)
    torch.save(torch.LongTensor(edge_index), os.path.join(output_dir, 'edge_index.pt'))

    # 保存最终的边特征 (归一化 + 扩展后)
    np.save(os.path.join(output_dir, 'edge_attr.npy'), edge_attr_final)
    torch.save(torch.FloatTensor(edge_attr_final), os.path.join(output_dir, 'edge_attr.pt'))

    # 保存特征名称
    if feature_names is not None:
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")

    # 保存处理信息
    with open(os.path.join(output_dir, 'data_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f"节点数量: {node_features.shape[0]}\n")
        f.write(f"节点特征维度: {node_features.shape[1]}\n")
        f.write(f"边数量: {edge_index.shape[1]}\n")
        f.write(f"最终边特征维度: {edge_attr_final.shape[1] if edge_attr_final.ndim > 1 else 1}\n")
        if sparse_adj_binary.shape[0] > 0:
            sparsity = 1.0 - (sparse_adj_binary.nnz / (sparse_adj_binary.shape[0] * sparse_adj_binary.shape[1]))
            f.write(f"邻接矩阵稀疏度: {sparsity:.6f}\n")
        else:
             f.write(f"邻接矩阵稀疏度: N/A (空图)\n")
        # Record method used
        # We need to pass the method name or args here, omitted for simplicity now

    print("数据保存完成")


def visualize_data(output_dir, node_features, sparse_adj_binary, edge_attr_final):
    """可视化处理后的数据"""
    print("正在生成数据可视化")
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 节点特征分布 (前5个)
    plt.figure(figsize=(12, 8))
    num_node_feat_to_plot = min(5, node_features.shape[1])
    for i in range(num_node_feat_to_plot):
        plt.subplot(2, 3, i+1)
        plt.hist(node_features[:, i], bins=30)
        plt.title(f'Node Feature {i} Dist')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'node_features_dist.png'))
    plt.close()

    # 2. 邻接矩阵热图 (如果节点数 <= 100)
    if sparse_adj_binary.shape[0] <= 100 and sparse_adj_binary.nnz > 0:
        plt.figure(figsize=(10, 10))
        sns.heatmap(sparse_adj_binary.toarray(), cmap='Blues', cbar=False)
        plt.title('Sparse Adjacency Matrix')
        plt.savefig(os.path.join(vis_dir, 'adjacency_heatmap.png'))
        plt.close()
    elif sparse_adj_binary.shape[0] > 100:
         print("节点数量 > 100，跳过邻接矩阵热图可视化。")
    elif sparse_adj_binary.nnz == 0:
         print("图为空，跳过邻接矩阵热图可视化。")


    # 3. 最终边特征分布 (前5个维度)
    if edge_attr_final.shape[0] > 0: # Check if edges exist
        plt.figure(figsize=(12, 8))
        num_edge_feat_to_plot = min(5, edge_attr_final.shape[1] if edge_attr_final.ndim > 1 else 1)
        for i in range(num_edge_feat_to_plot):
            plt.subplot(2, 3, i+1)
            if edge_attr_final.ndim > 1:
                plt.hist(edge_attr_final[:, i], bins=30)
            else: # Handle 1D case
                plt.hist(edge_attr_final, bins=30)
            plt.title(f'Final Edge Feature Dim {i} Dist')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'final_edge_features_dist.png'))
        plt.close()
    else:
        print("图为空，跳过边特征分布可视化。")


    # 4. 节点度分布
    if sparse_adj_binary.nnz > 0:
        # 计算出度和入度 (对于非对称 KNN 可能不同)
        out_degrees = sparse_adj_binary.sum(axis=1).A1 # .A1 converts matrix to 1D array
        in_degrees = sparse_adj_binary.sum(axis=0).A1

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(out_degrees, bins=max(1, int(np.max(out_degrees)) // 2 if np.max(out_degrees)>0 else 1) )
        plt.title('Node Out-Degree Distribution')
        plt.xlabel('Out-Degree')
        plt.ylabel('Node Count')

        plt.subplot(1, 2, 2)
        plt.hist(in_degrees, bins=max(1, int(np.max(in_degrees)) // 2 if np.max(in_degrees)>0 else 1))
        plt.title('Node In-Degree Distribution')
        plt.xlabel('In-Degree')
        plt.ylabel('Node Count')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'degree_dist.png'))
        plt.close()
    else:
         print("图为空，跳过度分布可视化。")


    print(f"可视化结果已保存到 {vis_dir}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将距离邻接矩阵转换为稀疏图结构和边特征')
    parser.add_argument('--node_features', type=str, default='NEWDATA/X_simplize.CSV', help='节点特征CSV文件路径')
    parser.add_argument('--distance_matrix', type=str, default='NEWDATA/A.csv', help='距离邻接矩阵CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='NEWDATA/processed_sparse', help='输出目录') # Changed default output dir

    # Graph sparsification method arguments
    parser.add_argument('--method', type=str, default='knn', choices=['knn', 'threshold'], help='图稀疏化方法')
    parser.add_argument('--k', type=int, default=10, help='KNN方法中每个节点保留的邻居数量')
    parser.add_argument('--theta', type=float, default=None, help='Threshold方法中的距离阈值 (需要根据数据分布设定)') # Default None, must be set if method='threshold'

    # Edge feature processing arguments
    parser.add_argument('--normalize', type=str, default='minmax', choices=['minmax', 'standard', 'none'], help='边特征(距离)归一化方法')
    parser.add_argument('--edge_dim', type=int, default=10, help='最终边特征的目标维度')
    parser.add_argument('--visualize', action='store_true', help='是否生成数据可视化')

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.method == 'threshold' and args.theta is None:
        parser.error("--method 'threshold' requires --theta to be set.")
    if args.method == 'knn' and args.k <= 0:
         parser.error("--k must be a positive integer for method 'knn'.")

    print("--- 参数配置 ---")
    print(f"节点特征文件: {args.node_features}")
    print(f"距离矩阵文件: {args.distance_matrix}")
    print(f"输出目录: {args.output_dir}")
    print(f"图稀疏化方法: {args.method}")
    if args.method == 'knn':
        print(f"  K 值: {args.k}")
    else:
        print(f"  Theta 阈值: {args.theta}")
    print(f"边特征(距离)归一化: {args.normalize}")
    print(f"最终边特征维度: {args.edge_dim}")
    print(f"是否可视化: {args.visualize}")
    print("-----------------")


    # 加载节点特征
    node_features, feature_names = load_node_features(args.node_features)

    # 加载距离邻接矩阵
    distance_matrix = load_distance_matrix(args.distance_matrix)

    # --- Graph Sparsification ---
    sparse_adj_binary = None
    edge_index = None
    edge_attr_distances = None # Store raw distances first
    
    num_nodes = distance_matrix.shape[0] # 获取节点数量

    if args.method == 'knn':
        # 1. 先创建有向 KNN 图
        print("步骤 1: 创建初始有向 KNN 图...")
        # 使用临时变量存储有向图的结果
        sparse_adj_binary_directed, edge_index_directed, edge_attr_distances_directed = create_knn_graph(distance_matrix, args.k)

        # 2. 将有向边索引转换为无向 (取并集)
        print("步骤 2: 将有向图转换为无向图...")
        edge_index_tensor = torch.from_numpy(edge_index_directed).long()

        # 调用 to_undirected 对边索引进行对称化
        edge_index_undirected_tensor = to_undirected(edge_index_tensor, num_nodes=num_nodes)
        
        # 将对称化后的边索引转回 numpy 格式，作为最终的 edge_index
        edge_index = edge_index_undirected_tensor.numpy()
        print(f"无向化后的边数量: {edge_index.shape[1]}")

        # 3. 重新提取与无向边对应的距离特征 (重要！)
        #   因为 to_undirected 可能添加了新的边 (j, i)，我们需要为这些边找到对应的距离
        print("步骤 3: 重新提取与无向边对应的距离特征...")
        rows_undirected, cols_undirected = edge_index[0], edge_index[1]
        # 直接从原始距离矩阵中查找新 edge_index 对应的距离
        edge_attr_distances = distance_matrix[rows_undirected, cols_undirected].reshape(-1, 1)
        print(f"重新提取的边特征形状: {edge_attr_distances.shape}")

        # 4. 创建最终的二进制稀疏邻接矩阵 (基于无向边)
        print("步骤 4: 创建最终的无向二进制稀疏邻接矩阵...")
        sparse_adj_binary = sp.csr_matrix((np.ones(edge_index.shape[1]), (rows_undirected, cols_undirected)),
                                          shape=(num_nodes, num_nodes))
        sparsity = 1.0 - (sparse_adj_binary.nnz / (num_nodes * num_nodes))
        print(f"最终无向邻接矩阵稀疏度: {sparsity:.6f}")
        

    elif args.method == 'threshold':
        # Threshold 方法如果距离矩阵对称，则结果已经是无向的，无需额外处理
        sparse_adj_binary, edge_index, edge_attr_distances = create_threshold_graph(distance_matrix, args.theta)
        print("Threshold 方法生成图，假设输入距离矩阵对称，图已为无向。")
    else:
        raise ValueError(f"未知的稀疏化方法: {args.method}")

    # --- Edge Feature Processing ---
    # 后续处理现在使用对称化后的 edge_index 和重新提取的 edge_attr_distances
    print("\n--- 开始处理边特征 ---")
    edge_attr_normalized = normalize_edge_features(edge_attr_distances, args.normalize)
    edge_attr_final = expand_edge_features(edge_attr_normalized, args.edge_dim)

    # --- 保存和可视化 ---
    # 使用的是最终（可能已无向化）的 sparse_adj_binary, edge_index, edge_attr_final
    print("\n--- 开始保存数据 ---")
    save_processed_data(args.output_dir, node_features, sparse_adj_binary, edge_index, edge_attr_final, feature_names)

    # 可视化数据
    if args.visualize:
        print("\n--- 开始可视化数据 ---")
        visualize_data(args.output_dir, node_features, sparse_adj_binary, edge_attr_final)

    print(f"\n数据预处理 ({args.method} 方法) 完成")


if __name__ == "__main__":
    main()