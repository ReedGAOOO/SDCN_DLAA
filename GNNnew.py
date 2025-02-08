import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool


class SpatialEmbedding(nn.Module):
    """
    Spatial Embedding Layer
    Encodes one-hot distance features into embedding representations.
    """

    def __init__(self, dist_dim, embed_size):
        super(SpatialEmbedding, self).__init__()
        self.embedding = nn.Linear(dist_dim, embed_size, bias=False)

    def forward(self, dist_feat):
        dist_feat = self.embedding(dist_feat)
        return dist_feat


class NodeToEdgeAggregation(nn.Module):
    """
    Node-to-Edge Aggregation Layer
    Aggregates node features and spatial features to update edge embeddings.
    """

    def __init__(self, node_in_channels, edge_in_channels, hidden_size, embed_size):
        super(NodeToEdgeAggregation, self).__init__()
        self.spatial_embedding = SpatialEmbedding(edge_in_channels, embed_size)
        self.fc = nn.Linear(2 * node_in_channels + embed_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, node_feat, edge_index, edge_attr):
        # node_feat: [num_nodes, node_in_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_in_channels]

        src, dst = edge_index
        src_feat = node_feat[src]  # [num_edges, node_in_channels]
        dst_feat = node_feat[dst]  # [num_edges, node_in_channels]

        # Spatial embedding of edge attributes
        dist_feat = self.spatial_embedding(edge_attr)  # [num_edges, embed_size]

        # Concatenate source, destination, and spatial features
        edge_input = torch.cat([src_feat, dst_feat, dist_feat], dim=-1)  # [num_edges, 2*node_in_channels + embed_size]

        # Linear transformation and activation
        edge_feat = self.fc(edge_input)  # [num_edges, hidden_size]
        edge_feat = self.activation(edge_feat)
        return edge_feat


class EdgeToEdgeAggregation(nn.Module):
    """
    Edge-to-Edge Aggregation Layer using GATv2Conv
    """

    def __init__(self, edge_in_channels, hidden_size, num_heads=4, dropout=0.2):
        super(EdgeToEdgeAggregation, self).__init__()
        self.gatv2 = GATv2Conv(in_channels=edge_in_channels,
                               out_channels=hidden_size,
                               heads=num_heads,
                               dropout=dropout,
                               concat=True)

    def forward(self, edge_feat, edge_to_edge_index):
        # edge_feat: [num_edges, edge_in_channels]
        # edge_to_edge_index: [2, num_edge_edges]
        updated_edge_feat = self.gatv2(edge_feat, edge_to_edge_index)  # [num_edge_edges, hidden_size * heads]
        return updated_edge_feat


class EdgeToNodeAggregation(nn.Module):
    """
    Edge-to-Node Aggregation Layer using GATv2Conv
    """

    def __init__(self, node_in_channels, edge_in_channels, hidden_size, num_heads=4, dropout=0.2):
        super(EdgeToNodeAggregation, self).__init__()
        self.gatv2 = GATv2Conv(in_channels=edge_in_channels,
                               out_channels=hidden_size,
                               heads=num_heads,
                               dropout=dropout,
                               concat=True)

    def forward(self, node_edge_feat, node_to_node_index):
        # node_edge_feat: [num_nodes + num_edges, edge_in_channels]
        # node_to_node_index: [2, num_connections]
        updated_node_feat = self.gatv2(node_edge_feat, node_to_node_index)  # [num_connections, hidden_size * heads]
        return updated_node_feat


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
        if edge_to_edge_index is not None:
            updated_edge_feat = self.edge_to_edge(updated_edge_feat,
                                                  edge_to_edge_index)  # [num_edge_edges, hidden_size * heads]
            # Optionally, you might want to reshape or reduce the heads dimension
            updated_edge_feat = updated_edge_feat.mean(dim=1)  # [num_edge_edges, hidden_size]

        # Step 3: Update node features using edge features via edge-to-node aggregation
        # Concatenate node features and edge features if necessary
        # Here, assuming node_to_node_index connects nodes with their corresponding edges
        node_edge_feat = torch.cat([node_feat, updated_edge_feat], dim=0)  # [num_nodes + num_edge_edges, ...]
        updated_node_feat = self.edge_to_node(node_edge_feat,
                                              node_to_node_index)  # [num_connections, hidden_size * heads]
        updated_node_feat = updated_node_feat.mean(dim=1)  # [num_connections, hidden_size]

        # Aggregate the updated node features
        # This step may vary based on how node_to_node_index is defined
        # Here, assuming scatter_add over node indices
        # Example:
        # updated_node_feat = scatter_add(updated_node_feat, node_indices, dim=0, dim_size=num_nodes)

        # For simplicity, assume node_to_node_index defines connections properly
        # and the aggregation is handled within the GATv2Conv layer

        return updated_node_feat, updated_edge_feat


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
        """
        Args:
            data (Data): PyG Data object containing:
                - x: [num_nodes, node_in_channels]
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges, edge_in_channels]
                - batch: [num_nodes] batch indices
            edge_to_edge_index (Tensor): [2, num_edge_edges] connections between edges
            node_to_node_index (Tensor): [2, num_connections] connections between nodes and edges
        Returns:
            Output of the model
        """
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


# Example usage
# Assuming you have a PyG Data object `data` with the necessary attributes
# and edge_to_edge_index and node_to_node_index are predefined tensors

# Define model parameters
node_in_channels = 64
edge_in_channels = 16
hidden_size = 128
embed_size = 32
num_layers = 3
num_heads = 4
dropout = 0.2

# Instantiate the model
model = SMANModel(node_in_channels, edge_in_channels, hidden_size, embed_size, num_layers, num_heads, dropout)

# Example data (you need to prepare `data`, `edge_to_edge_index`, `node_to_node_index`)
# data = Data(x=torch.randn(num_nodes, node_in_channels),
#             edge_index=torch.randint(0, num_nodes, (2, num_edges)),
#             edge_attr=torch.randn(num_edges, edge_in_channels),
#             batch=torch.zeros(num_nodes, dtype=torch.long))

# edge_to_edge_index = torch.randint(0, num_edges, (2, num_edge_edges))
# node_to_node_index = torch.randint(0, num_nodes + num_edge_edges, (2, num_connections))

# Forward pass
# output = model(data, edge_to_edge_index, node_to_node_index)
