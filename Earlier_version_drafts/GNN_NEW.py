import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_scatter import scatter_add
from collections import defaultdict


# 假设 SpatialEmbedding, NodeToEdgeAggregation, EdgeToEdgeAggregation, EdgeToNodeAggregation 已在之前定义
# 如果尚未定义，请参考前一个回答中的定义

class SpatialConv(nn.Module):
    """
    Spatial Graph Convolution Layer
    Implements the spatial graph convolution layer for molecular graphs.
    """

    def __init__(self, node_in_channels, edge_in_channels, hidden_size, embed_size, num_heads=4, dropout=0.2):
        super(SpatialConv, self).__init__()
        self.node_to_edge = NodeToEdgeAggregation(node_in_channels, edge_in_channels, hidden_size, embed_size)
        self.edge_to_edge = EdgeToEdgeAggregation(hidden_size, hidden_size, num_heads=num_heads, dropout=dropout)
        self.edge_to_node = EdgeToNodeAggregation(node_in_channels, hidden_size, hidden_size, num_heads=num_heads,
                                                  dropout=dropout)

    def forward(self, data, edge_to_edge_index, node_to_node_index):
        """
        Args:
            data (Data): PyG Data object containing:
                - x: [num_nodes, node_in_channels]
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges, edge_in_channels]
            edge_to_edge_index (Tensor): [2, num_edge_edges] connections between edges
            node_to_node_index (Tensor): [2, num_connections] connections between nodes and edges
        Returns:
            Updated node features and edge features
        """
        node_feat = data.x  # [num_nodes, node_in_channels]
        edge_index = data.edge_index  # [2, num_edges]
        edge_attr = data.edge_attr  # [num_edges, edge_in_channels]

        # Step 1: Update edge features using node features
        updated_edge_feat = self.node_to_edge(node_feat, edge_index, edge_attr)  # [num_edges, hidden_size]

        # Step 2: Update edge features using edge-to-edge aggregation
        if edge_to_edge_index.numel() > 0:
            updated_edge_feat = self.edge_to_edge(updated_edge_feat,
                                                  edge_to_edge_index)  # [num_edge_edges, hidden_size * heads]
            # 平均多头输出
            updated_edge_feat = updated_edge_feat.view(updated_edge_feat.size(0), -1).mean(dim=1,
                                                                                           keepdim=True)  # [num_edge_edges, hidden_size]

        # Step 3: Update node features using edge features via edge-to-node aggregation
        # 将节点特征和边特征合并
        node_edge_feat = torch.cat([node_feat, updated_edge_feat], dim=0)  # [num_nodes + num_edges, ...]
        updated_node_feat = self.edge_to_node(node_edge_feat,
                                              node_to_node_index)  # [num_connections, hidden_size * heads]
        # 平均多头输出
        updated_node_feat = updated_node_feat.view(updated_node_feat.size(0), -1).mean(
            dim=1)  # [num_connections, hidden_size]

        # 聚合节点更新
        # 假设 node_to_node_index 的目标是节点索引
        src, dst = node_to_node_index
        num_nodes = node_feat.size(0)
        aggregated_node_feat = scatter_add(updated_node_feat, dst, dim=0,
                                           dim_size=num_nodes)  # [num_nodes, hidden_size]

        return aggregated_node_feat, updated_edge_feat


class SMANModel(nn.Module):
    """
    The main SMAN model that integrates all components.
    """

    def __init__(self, node_in_channels, edge_in_channels, hidden_size, embed_size, num_layers=3, num_heads=4,
                 dropout=0.2):
        super(SMANModel, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = SpatialConv(node_in_channels, edge_in_channels, hidden_size, embed_size, num_heads, dropout)
            self.layers.append(conv)
            node_in_channels = hidden_size  # Update for the next layer
            edge_in_channels = hidden_size  # Assuming edge features are also updated to hidden_size

        self.pool = global_mean_pool  # or global_add_pool, depending on your needs
        self.fc = nn.Linear(hidden_size, 1)  # For regression tasks

    def forward(self, data, edge_to_edge_index, node_to_node_index):
        x = data.x
        for layer in self.layers:
            x, edge_attr = layer(data, edge_to_edge_index, node_to_node_index)
            data.x = x  # Update node features in data for the next layer
            data.edge_attr = edge_attr  # Update edge features in data for the next layer

        # Pooling
        x = self.pool(x, data.batch)  # [batch_size, hidden_size]

        # Fully connected layer
        x = self.fc(x)  # [batch_size, 1]
        return x


class SMANLayer(nn.Module):
    """
    GNN Layer that uses the SMANModel internally.
    This layer is designed to replace the simple GNNLayer.
    """

    def __init__(self, in_features, out_features, hidden_size, embed_size, num_layers=3, num_heads=4, dropout=0.2):
        super(SMANLayer, self).__init__()
        self.sman = SMANModel(node_in_channels=in_features,
                              edge_in_channels=1,  # Assume default edge_attr=1
                              hidden_size=hidden_size,
                              embed_size=embed_size,
                              num_layers=num_layers,
                              num_heads=num_heads,
                              dropout=dropout)
        self.out_features = out_features
        self.activation = nn.ReLU()

    def forward(self, features, adj):
        """
        Args:
            features (Tensor): Node features [num_nodes, in_features]
            adj (SparseTensor or Tensor): Adjacency matrix [num_nodes, num_nodes]
        Returns:
            Tensor: Updated node features [num_nodes, out_features]
        """
        # Convert adjacency matrix to edge_index
        if isinstance(adj, torch.Tensor):
            adj = adj.coalesce()
            edge_index = adj.indices()  # [2, num_edges]
        elif isinstance(adj, SparseTensor):
            edge_index = adj.coo().indices()
        else:
            raise ValueError("Adjacency matrix must be a torch.Tensor or torch_sparse.SparseTensor")

        # Assign edge_attr as ones
        num_edges = edge_index.size(1)
        edge_attr = torch.ones((num_edges, 1), device=features.device)

        # Create a Data object
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

        # Define edge_to_edge_index
        src, dst = edge_index
        num_edges = src.size(0)

        # Build edge_to_edge connections: edges sharing a common node
        node_to_edges = defaultdict(list)
        for e in range(num_edges):
            node_to_edges[src[e].item()].append(e)
            node_to_edges[dst[e].item()].append(e)

        edge_to_edge = []
        for e in range(num_edges):
            connected_edges = node_to_edges[src[e].item()] + node_to_edges[dst[e].item()]
            connected_edges = [ce for ce in connected_edges if ce != e]
            for ce in connected_edges:
                edge_to_edge.append([e, ce])

        if len(edge_to_edge) == 0:
            edge_to_edge_index = torch.empty((2, 0), dtype=torch.long, device=features.device)
        else:
            edge_to_edge_index = torch.tensor(edge_to_edge, dtype=torch.long, device=features.device).t().contiguous()

        # Define node_to_node_index
        # Edge nodes are indexed from num_nodes to num_nodes + num_edges -1
        num_nodes = features.size(0)
        edge_nodes = torch.arange(num_nodes, num_nodes + num_edges, device=features.device)
        # Each edge node connects to its source and target nodes
        edge_source = edge_nodes.repeat_interleave(2)  # [2*num_edges]
        edge_target = torch.cat([src, dst], dim=0)  # [2*num_edges]
        node_to_node_index = torch.stack([edge_source, edge_target], dim=0)  # [2, 2*num_edges]

        # Pass to SMANModel
        output = self.sman(data, edge_to_edge_index, node_to_node_index)

        # Apply activation if needed
        output = self.activation(output)

        # Optionally, project to out_features if hidden_size != out_features
        if self.sman.fc.out_features != self.out_features:
            output = F.linear(output, self.sman.fc.weight, self.sman.fc.bias)

        return output


class SimpleGNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, hidden_size, embed_size, num_layers=3, num_heads=4,
                 dropout=0.2):
        super(SimpleGNNModel, self).__init__()
        self.sman_layer = SMANLayer(in_features, hidden_features, hidden_size, embed_size, num_layers, num_heads,
                                    dropout)
        self.output_layer = nn.Linear(hidden_features, out_features)

    def forward(self, features, adj):
        # features: [num_nodes, in_features]
        # adj: [num_nodes, num_nodes] 稀疏张量
        updated_features = self.sman_layer(features, adj)  # [num_nodes, hidden_features]
        output = self.output_layer(updated_features)  # [num_nodes, out_features]
        return output


# 示例数据
num_nodes = 100
in_features = 64
hidden_features = 128
out_features = 10
hidden_size = 128
embed_size = 32
num_layers = 3
num_heads = 4
dropout = 0.2

# 随机生成节点特征
features = torch.randn((num_nodes, in_features))

# 随机生成邻接矩阵
edge_indices = torch.randint(0, num_nodes, (2, 500))  # 500 条边
adj = SparseTensor(row=edge_indices[0], col=edge_indices[1], sparse_sizes=(num_nodes, num_nodes))

# 创建模型
model = SimpleGNNModel(in_features, hidden_features, out_features, hidden_size, embed_size, num_layers, num_heads,
                       dropout)

# 前向传播
output = model(features, adj)

print(output.shape)  # 应为 [num_nodes, out_features]
