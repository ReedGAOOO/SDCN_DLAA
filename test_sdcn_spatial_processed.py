import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
from sdcn_spatial import SDCN_Spatial, train_sdcn_spatial
import argparse
from sklearn.preprocessing import OneHotEncoder
import os

# 创建自定义数据集类来使用我们处理好的数据
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
        
        # 我们没有标签，创建假标签（所有节点的类别为0）
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

if __name__ == "__main__":
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
    args.name = 'custom_data'  # 设置一个自定义名称
    args.k = None  # 不使用K-nearest neighbor
    
    # 加载边特征
    edge_attr = dataset.edge_attr
    
    # 打印信息
    print(f"节点数量: {dataset.num_nodes}")
    print(f"特征维度: {dataset.num_features}")
    print(f"边特征维度: {args.edge_dim}")
    print(f"聚类数量: {args.n_clusters}")
    
    # 训练模型
    print("开始训练SDCN_Spatial模型...")
    model, results = train_sdcn_spatial(dataset, args, edge_attr)
    
    print("训练完成！")