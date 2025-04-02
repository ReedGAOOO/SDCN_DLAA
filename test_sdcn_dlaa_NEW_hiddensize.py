import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
# 注意：这里我们假设 sdcn_dlaa_NEW_hiddensize.py 稍后会被创建并包含修改后的 train_sdcn_dlaa_custom
from sdcn_dlaa_NEW_hiddensize import SDCN_DLAA, target_distribution, eva, train_sdcn_dlaa_custom
from sklearn.cluster import KMeans
import argparse
import pandas as pd
import os
from datetime import datetime
import sys
from collections import defaultdict
import random

# 创建日志记录器
class Logger(object):
    def __init__(self, filename="Default.log", terminal_mode="normal"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")  # 添加UTF-8编码
        self.terminal_mode = terminal_mode

    def write(self, message):
        # 写入日志文件
        self.log.write(message)

        # 终端输出
        self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 自定义数据集类
class CustomDataset:
    def __init__(self, node_features_path, edge_attr_path=None, device=None):
        """
        初始化自定义数据集

        Args:
            node_features_path: 节点特征文件路径
            edge_attr_path: 边特征文件路径 (可选)
            device: 设备 (CPU或GPU)
        """
        self.device = device

        # 加载节点特征
        if node_features_path.endswith('.pt'):
            self.x = torch.load(node_features_path).numpy()
        else:
            self.x = np.load(node_features_path)

        # 获取节点数量和特征维度
        self.num_nodes, self.num_features = self.x.shape
        print(f"节点特征形状: {self.x.shape}")

        # 我们没有标签，使用全0作为初始标签（仅用于训练过程）
        # 假设有3个聚类类别
        self.num_clusters = 3 # 默认值，会被 args.n_clusters 覆盖
        self.y = np.zeros(self.num_nodes, dtype=int)

        # 如果提供了边特征路径，加载边特征
        self.edge_attr = None
        if edge_attr_path:
            if edge_attr_path.endswith('.pt'):
                self.edge_attr = torch.load(edge_attr_path)
            else:
                edge_attr_np = np.load(edge_attr_path)
                self.edge_attr = torch.from_numpy(edge_attr_np).float()

            # 将边特征移动到指定设备
            if self.device is not None:
                self.edge_attr = self.edge_attr.to(self.device)

            print(f"边特征形状: {self.edge_attr.shape}")

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx))

# 将scipy稀疏矩阵转换为torch稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转换为torch稀疏张量

    Args:
        sparse_mx: scipy稀疏矩阵

    Returns:
        torch稀疏张量
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# 加载稀疏邻接矩阵
def load_sparse_adj(path, device=None):
    """
    加载稀疏邻接矩阵

    Args:
        path: 稀疏邻接矩阵文件路径
        device: 设备 (CPU或GPU)

    Returns:
        torch稀疏张量形式的邻接矩阵
    """
    sparse_adj = sp.load_npz(path)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(sparse_adj)

    # 将张量移动到指定设备
    if device is not None:
        adj_tensor = adj_tensor.to(device)

    return adj_tensor

# 预处理边到边图关系（性能优化的关键）
def precompute_edge_to_edge_graph(adj, max_edges_per_node=10, device=None):
    """
    预处理边到边图关系，避免在每次前向传播中重新计算

    Args:
        adj: 邻接矩阵（torch稀疏张量）
        max_edges_per_node: 每个节点考虑的最大边数
        device: 设备 (CPU或GPU)

    Returns:
        edge_index: 节点到节点的边索引 [2, num_edges]
        edge_to_edge_index: 边到边的连接索引 [2, num_edge_edges]
    """
    print("预处理边到边图关系（一次性操作）...")

    # 转换邻接矩阵为边索引
    if adj.is_sparse:
        adj = adj.coalesce()
        edge_index = adj.indices()
    else:
        # 假设 dense_to_sparse 是一个可用的函数
        # from torch_geometric.utils import dense_to_sparse
        edge_index, _ = dense_to_sparse(adj)

    # 移动到指定设备
    if device is not None:
        edge_index = edge_index.to(device)

    num_edges = edge_index.size(1)

    # 构建节点到边的映射
    node_to_edges = defaultdict(list)
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        node_to_edges[src].append(i)
        node_to_edges[dst].append(i)

    # 构建边到边的连接
    edge_to_edge_list = []
    for node, connected_edges in node_to_edges.items():
        if len(connected_edges) > 1:
            # 超过最大边数限制时进行随机采样
            if len(connected_edges) > max_edges_per_node:
                sampled_edges = random.sample(connected_edges, max_edges_per_node)
            else:
                sampled_edges = connected_edges

            # 连接共享节点的所有边对
            for i in range(len(sampled_edges)):
                for j in range(i+1, len(sampled_edges)):
                    edge_i = sampled_edges[i]
                    edge_j = sampled_edges[j]
                    # 为无向图添加双向连接
                    edge_to_edge_list.append([edge_i, edge_j])
                    edge_to_edge_list.append([edge_j, edge_i])

    # 转换为张量形式
    if len(edge_to_edge_list) > 0:
        edge_to_edge_index = torch.tensor(edge_to_edge_list, dtype=torch.long).t()
        if device is not None:
            edge_to_edge_index = edge_to_edge_index.to(device)
    else:
        # 如果没有边到边的连接，创建空张量
        edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long)
        if device is not None:
            edge_to_edge_index = edge_to_edge_index.to(device)

    print(f"边到边图构建完成: {edge_to_edge_index.shape[1]} 条边对连接")

    return edge_index, edge_to_edge_index

if __name__ == "__main__":
    # 创建日志目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 修改日志文件名以反映是 hiddensize 测试
    log_filename = f'logs/sdcn_dlaa_hiddensize_run_{timestamp}.txt'

    # 重定向stdout到控制台和文件
    sys.stdout = Logger(log_filename)

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='使用NEWDATA/processed中处理好的数据训练可变隐藏层大小的SDCN_DLAA模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=4) # 默认值会被 runner 覆盖
    parser.add_argument('--edge_dim', type=int, default=10)
    parser.add_argument('--max_edges_per_node', type=int, default=10)

    # --- 新增参数 ---
    parser.add_argument('--hs1', type=int, default=500, help='Hidden size for AE layer 1 & SpatialConv 1')
    parser.add_argument('--hs2', type=int, default=500, help='Hidden size for AE layer 2 & SpatialConv 2')
    parser.add_argument('--hs3', type=int, default=2000, help='Hidden size for AE layer 3 & SpatialConv 3')
    # --- 结束新增参数 ---

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("使用CUDA: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # 设置文件路径
    node_features_path = 'NEWDATA/processed/node_features.npy'
    binary_adj_path = 'NEWDATA/processed/binary_adj.npz'
    edge_attr_path = 'NEWDATA/processed/edge_attr.npy'

    # 创建数据集，并指定设备
    dataset = CustomDataset(node_features_path, edge_attr_path, device=args.device)
    # 更新数据集中的聚类数量
    dataset.num_clusters = args.n_clusters

    # 加载邻接矩阵，并指定设备
    adj = load_sparse_adj(binary_adj_path, device=args.device)

    # 设置特征尺寸
    args.n_input = dataset.num_features

    # 加载边特征
    edge_attr = dataset.edge_attr

    # 打印信息
    print("\n--- 参数配置 ---")
    print(f"学习率 (lr): {args.lr}")
    print(f"聚类数量 (n_clusters): {args.n_clusters}")
    print(f"潜在空间维度 (n_z): {args.n_z}")
    print(f"Dropout率: {args.dropout}")
    print(f"GAT头数 (heads): {args.heads}")
    print(f"边特征维度 (edge_dim): {args.edge_dim}")
    print(f"每个节点最大边数 (max_edges_per_node): {args.max_edges_per_node}")
    print(f"隐藏层大小 1 (hs1): {args.hs1}")
    print(f"隐藏层大小 2 (hs2): {args.hs2}")
    print(f"隐藏层大小 3 (hs3): {args.hs3}")
    print("--- 数据信息 ---")
    print(f"节点数量: {dataset.num_nodes}")
    print(f"特征维度 (n_input): {args.n_input}")
    if edge_attr is not None:
        print(f"边特征形状: {edge_attr.shape}")
    else:
        print("未使用边特征")
    print("-----------------\n")


    # 训练模型
    print("\n开始训练可变隐藏层大小的SDCN_DLAA模型...")
    try:
        # 调用修改后的训练函数 (来自 sdcn_dlaa_NEW_hiddensize.py)
        model, results, clusters = train_sdcn_dlaa_custom(dataset, adj, args, edge_attr)

        # 分析聚类结果
        cluster_counts = np.bincount(clusters)
        print("\n聚类分布:")
        for i, count in enumerate(cluster_counts):
            print(f"聚类 {i}: {count} 个节点 ({count/len(clusters)*100:.2f}%)")
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        # 打印完整的异常堆栈跟踪，帮助调试
        import traceback
        traceback.print_exc()

    print("\n训练完成！")