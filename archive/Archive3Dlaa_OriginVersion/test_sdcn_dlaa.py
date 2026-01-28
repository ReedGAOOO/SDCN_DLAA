import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
from Archive3.sdcn_dlaa import SDCN_DLAA, target_distribution, eva, train_sdcn_dlaa_custom
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
        self.num_clusters = 3
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

# 自定义SDCN_DLAA模型训练函数
def train_sdcn_dlaa_custom_main(dataset, adj, args, edge_attr=None):
    """
    训练SDCN_DLAA模型
    
    Args:
        dataset: 数据集对象，包含特征和标签
        adj: 邻接矩阵（torch稀疏张量）
        args: 训练参数
        edge_attr: 边特征 [num_edges, edge_dim]
    """
    
    # 检查是否提供了边特征，如果没有，创建简单的边特征
    if edge_attr is None:
        print("未提供边特征，使用随机初始化的边特征")
        num_edges = adj._nnz()
        edge_attr = torch.ones(num_edges, args.edge_dim).to(args.device)
    else:
        # 确保边特征在正确的设备上
        edge_attr = edge_attr.to(args.device)
    
    # 性能优化：预处理边到边图结构
    print("性能优化：预计算图结构...")
    edge_index, edge_to_edge_index = precompute_edge_to_edge_graph(
        adj, 
        max_edges_per_node=args.max_edges_per_node,
        device=args.device
    )
    
    print(f"预计算完成：节点到节点边数量: {edge_index.shape[1]}, 边到边连接数量: {edge_to_edge_index.shape[1]}")
    
    # 创建使用预计算图结构的模型
    model = SDCN_DLAA(
        500, 500, 2000, 2000, 500, 500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1.0,
        dropout=args.dropout,
        edge_dim=args.edge_dim,
        heads=args.heads,
        max_edges_per_node=args.max_edges_per_node,
        precomputed_edge_index=edge_index,
        precomputed_edge_to_edge_index=edge_to_edge_index
    ).to(args.device)
    
    print(model)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # 将邻接矩阵移动到指定设备
    adj = adj.to(args.device)
    
    # 准备数据
    data = torch.Tensor(dataset.x).to(args.device)
    y = dataset.y
    
    # 使用预训练的自编码器初始化聚类中心
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    
    # 使用K-means进行初始聚类
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    
    # 评估初始聚类结果
    if len(np.unique(y)) > 1:  # 如果有真实标签
        eva(y, y_pred, 'pae')
    else:
        print(f"初始聚类完成。聚类数量: {args.n_clusters}")
    
    # 创建保存结果的列表
    results = []
    
    # 训练循环
    for epoch in range(200):
        # 更新当前epoch
        model.current_epoch = epoch
        
        if epoch % 1 == 0:
            # 评估模型
            try:
                _, tmp_q, pred, _, _ = model(data, adj, edge_attr)
                
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)
                
                res1 = tmp_q.cpu().numpy().argmax(1)  # Q
                res2 = pred.data.cpu().numpy().argmax(1)  # Z
                res3 = p.data.cpu().numpy().argmax(1)  # P
                
                # 评估每轮的聚类指标
                if len(np.unique(y)) > 1:  # 如果有真实标签
                    acc1, f1_1, nmi1, ari1 = eva(y, res1, f'{epoch}Q')
                    acc2, f1_2, nmi2, ari2 = eva(y, res2, f'{epoch}Z')
                    acc3, f1_3, nmi3, ari3 = eva(y, res3, f'{epoch}P')
                    
                    # 保存每轮的聚类结果
                    results.append([epoch, acc1, f1_1, nmi1, ari1, acc2, f1_2, nmi2, ari2, acc3, f1_3, nmi3, ari3])
                else:
                    # 无标签情况下，只保存聚类结果，不计算评估指标
                    cluster_distribution = np.bincount(res2)
                    print(f"Epoch {epoch}, 聚类分布: {cluster_distribution}")
                    results.append([epoch] + [0] * 12)  # 占位填充
            except Exception as e:
                print(f"Epoch {epoch} 评估出错: {str(e)}")
                continue
        
        # 前向传播
        try:
            x_bar, q, pred, _, _ = model(data, adj, edge_attr)
            
            # 计算目标分布
            p = target_distribution(q.data)
            
            # 计算损失
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, data)
            
            # 综合损失，使用与原始SDCN相同的权重
            loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 每10个epoch打印一次损失信息
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, KL: {kl_loss.item():.4f}, CE: {ce_loss.item():.4f}, RE: {re_loss.item():.4f}")
        except Exception as e:
            print(f"Epoch {epoch} 训练出错: {str(e)}")
            continue
    
    # 获取最终聚类结果
    try:
        _, _, final_pred, _, _ = model(data, adj, edge_attr)
        final_clusters = final_pred.data.cpu().numpy().argmax(1)
    except Exception as e:
        print(f"获取最终聚类结果出错: {str(e)}")
        # 如果出错，使用最后一次成功的聚类结果
        if 'res2' in locals():
            final_clusters = res2
        else:
            # 如果没有任何成功的聚类结果，返回全0
            final_clusters = np.zeros(dataset.num_nodes, dtype=int)
    
    # 保存结果
    column_names = ['Epoch', 'Acc_Q', 'F1_Q', 'NMI_Q', 'ARI_Q', 'Acc_Z', 'F1_Z', 'NMI_Z', 'ARI_Z', 'Acc_P', 'F1_P', 'NMI_P', 'ARI_P']
    results_df = pd.DataFrame(results, columns=column_names)
    results_df.to_csv('sdcn_dlaa_training_results.csv', index=False)
    
    print("训练完成。结果已保存到 'sdcn_dlaa_training_results.csv'.")
    
    final_results_df = pd.DataFrame({'节点ID': np.arange(len(final_clusters)), '聚类ID': final_clusters})
    final_results_df.to_csv('sdcn_dlaa_final_cluster_results.csv', index=False)
    
    print("最终聚类结果已保存到 'sdcn_dlaa_final_cluster_results.csv'.")
    
    return model, results_df, final_clusters

if __name__ == "__main__":
    # 创建日志目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/sdcn_dlaa_run_{timestamp}.txt'
    
    # 重定向stdout到控制台和文件
    sys.stdout = Logger(log_filename)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='使用NEWDATA/processed中处理好的数据训练SDCN_DLAA模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--edge_dim', type=int, default=10)
    parser.add_argument('--max_edges_per_node', type=int, default=10)
    
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
    
    # 加载邻接矩阵，并指定设备
    adj = load_sparse_adj(binary_adj_path, device=args.device)
    
    # 设置特征尺寸
    args.n_input = dataset.num_features
    
    # 加载边特征
    edge_attr = dataset.edge_attr
    
    # 打印信息
    print(f"节点数量: {dataset.num_nodes}")
    print(f"特征维度: {dataset.num_features}")
    print(f"边特征维度: {args.edge_dim}")
    print(f"聚类数量: {args.n_clusters}")
    
    # 训练模型
    print("\n开始训练SDCN_DLAA模型...")
    try:
        model, results, clusters = train_sdcn_dlaa_custom_main(dataset, adj, args, edge_attr)
        
        # 分析聚类结果
        cluster_counts = np.bincount(clusters)
        print("\n聚类分布:")
        for i, count in enumerate(cluster_counts):
            print(f"聚类 {i}: {count} 个节点 ({count/len(clusters)*100:.2f}%)")
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
    
    print("\n训练完成！")