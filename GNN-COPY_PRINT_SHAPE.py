import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        # 合并稀疏矩阵以确保可以访问索引
        adj = adj.coalesce()

        # 打印邻接矩阵
        print("邻接矩阵 (adj):")
        print(adj)

        # 统计稀疏矩阵的元素个数
        sparse_indices = adj.indices()
        sparse_values = adj.values()

        num_indices = sparse_indices.numel()  # 统计索引的总元素个数
        num_values = sparse_values.numel()  # 统计值的总元素个数

        print("稀疏矩阵的索引个数:", num_indices)
        print("稀疏矩阵的值个数:", num_values)

        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)


        if active:
            output = F.relu(output)
        return output

