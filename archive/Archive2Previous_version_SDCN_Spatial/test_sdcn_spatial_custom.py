import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
from sdcn_spatial import SDCN_Spatial, target_distribution, eva
from sklearn.cluster import KMeans
import argparse
import pandas as pd
import os
from datetime import datetime
import sys

# 创建日志记录器
class Logger(object):
    def __init__(self, filename="Default.log", terminal_mode="normal"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
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
    def __init__(self, node_features_path, edge_attr_path=None):
        """
        初始化自定义数据集
        
        Args:
            node_features_path: 节点特征文件路径
            edge_attr_path: 边特征文件路径 (可选)
        """
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
                self.edge_attr = torch.from_numpy(np.load(edge_attr_path))
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
def load_sparse_adj(path):
    """
    加载稀疏邻接矩阵
    
    Args:
        path: 稀疏邻接矩阵文件路径
        
    Returns:
        torch稀疏张量形式的邻接矩阵
    """
    sparse_adj = sp.load_npz(path)
    return sparse_mx_to_torch_sparse_tensor(sparse_adj)

# 自定义训练函数
def train_sdcn_spatial_custom(dataset, adj, args, edge_attr=None):
    """
    训练SDCN_Spatial模型
    
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
        edge_attr = torch.ones(num_edges, args.edge_dim)
    
    # 创建模型
    model = SDCN_Spatial(
        500, 500, 2000, 2000, 500, 500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1.0,
        dropout=args.dropout,
        edge_dim=args.edge_dim,
        heads=args.heads,
        max_edges_per_node=args.max_edges_per_node
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
        
        # 前向传播
        x_bar, q, pred, _, _ = model(data, adj, edge_attr)
        
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
    
    # 获取最终聚类结果
    _, _, final_pred, _, _ = model(data, adj, edge_attr)
    final_clusters = final_pred.data.cpu().numpy().argmax(1)
    
    # 保存结果
    column_names = ['Epoch', 'Acc_Q', 'F1_Q', 'NMI_Q', 'ARI_Q', 'Acc_Z', 'F1_Z', 'NMI_Z', 'ARI_Z', 'Acc_P', 'F1_P', 'NMI_P', 'ARI_P']
    results_df = pd.DataFrame(results, columns=column_names)
    results_df.to_csv('spatial_training_results_custom.csv', index=False)
    
    print("训练完成。结果已保存到 'spatial_training_results_custom.csv'.")
    
    final_results_df = pd.DataFrame({'节点ID': np.arange(len(final_clusters)), '聚类ID': final_clusters})
    final_results_df.to_csv('spatial_final_cluster_results_custom.csv', index=False)
    
    print("最终聚类结果已保存到 'spatial_final_cluster_results_custom.csv'.")
    
    return model, results_df, final_clusters

if __name__ == "__main__":
    # 创建日志目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/sdcn_spatial_custom_run_{timestamp}.txt'
    
    # 重定向stdout到控制台和文件
    sys.stdout = Logger(log_filename)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='使用NEWDATA/processed中处理好的数据训练SDCN_Spatial模型',
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
    
    # 创建数据集
    dataset = CustomDataset(node_features_path, edge_attr_path)
    
    # 加载邻接矩阵
    adj = load_sparse_adj(binary_adj_path)
    
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
    print("\n开始训练SDCN_Spatial模型...")
    model, results, clusters = train_sdcn_spatial_custom(dataset, adj, args, edge_attr)
    
    # 分析聚类结果
    cluster_counts = np.bincount(clusters)
    print("\n聚类分布:")
    for i, count in enumerate(cluster_counts):
        print(f"聚类 {i}: {count} 个节点 ({count/len(clusters)*100:.2f}%)")
    
    print("\n训练完成！")