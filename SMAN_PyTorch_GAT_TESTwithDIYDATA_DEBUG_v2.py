## 1. Import Necessary Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.optim as optim
import numpy as np

## 2. Define the SMAN Model and Layers

class SMANModel(nn.Module):
    """
    Spatial-aware Molecular Graph Attention Network for DTA Prediction
    """
    def __init__(self, args, n_output=1):
        super(SMANModel, self).__init__()
        self.args = args
        self.num_layers = args['num_layers']
        self.hidden_size = args['hid_dim']
        self.pool_type = args['pool_type']  # 'sum' or 'mean'
        self.dropout_prob = args['drop']
        self.dist_dim = args['dist_dim']
        self.n_output = n_output

        # Define spatial embedding layer
        self.spatial_embedding = SpatialEmbedding(input_dim=self.dist_dim, embed_size=self.hidden_size)

        # Define spatial convolution layers
        self.spatial_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.spatial_convs.append(
                SpatialConv(
                    hidden_size=self.hidden_size,
                    num_heads=4,
                    dropout=0.2,
                    alpha=0.2,
                    activation='relu'
                )
            )

        # Define fully connected layers and dropout
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.fc2 = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.fc3 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.n_output)
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # Define activation function
        self.relu = nn.ReLU()

        # Define loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, data, return_intermediate=False):
        """
        Forward pass with optional return of intermediate outputs for debugging.
        Args:
            data: torch_geometric.data.Data or torch_geometric.data.Batch object
            return_intermediate (bool): Whether to return intermediate outputs for debugging.
        Returns:
            output: Predicted binding affinity
            intermediates (dict): A dictionary of intermediate outputs if return_intermediate=True.
        """
        intermediates = {}

        # Extract input features
        edges_dist = data.edges_dist  # Tensor: [num_edges, dist_dim]
        dist_feat_order = data.edges_dist_order  # Tensor: [num_edges, dist_dim], optional

        # Spatial embedding
        dist_feat, dist_feat_order = self.spatial_embedding(edges_dist, dist_feat_order)

        intermediates['spatial_embedding'] = (dist_feat.shape, dist_feat_order.shape if dist_feat_order is not None else None)

        node_edge_feat = data.x  # Node and edge features, Tensor: [num_nodes + num_edges, feat_size]
        feat_size = node_edge_feat.size(-1)
        intermediates['initial_node_edge_feat'] = node_edge_feat.shape

        # Multi-layer spatial convolution
        for i in range(self.num_layers):
            node_edge_feat = self.spatial_convs[i](
                data=data,
                node_edge_feat=node_edge_feat,
                dist_feat_order=dist_feat_order,
                dist_feat=dist_feat
            )
            intermediates[f'spatial_conv_{i}'] = node_edge_feat.shape

        # Node feature aggregation
        node_feat = node_edge_feat[data.nids]  # [num_nodes, hidden_size]
        if self.pool_type == 'sum':
            pooled_h = global_add_pool(node_feat, data.batch)
        elif self.pool_type == 'mean':
            pooled_h = global_mean_pool(node_feat, data.batch)
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")

        intermediates['pooled_h'] = pooled_h.shape

        # Fully connected layers and dropout
        output = self.relu(self.fc1(pooled_h))
        intermediates['fc1_output'] = output.shape
        output = self.dropout(output)
        output = self.relu(self.fc2(output))
        intermediates['fc2_output'] = output.shape
        output = self.dropout(output)
        output = self.relu(self.fc3(output))
        intermediates['fc3_output'] = output.shape
        output = self.dropout(output)
        output = self.fc_out(output)

        intermediates['final_output'] = output.shape

        if return_intermediate:
            return output.squeeze(), intermediates
        else:
            return output.squeeze()

class SpatialEmbedding(nn.Module):
    """
    Spatial Embedding Layer
    """
    def __init__(self, input_dim, embed_size):
        super(SpatialEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embed_size)

    def forward(self, dist_feat, dist_feat_order=None):
        """
        Args:
            dist_feat (Tensor): [num_edges, dist_dim]
            dist_feat_order (Tensor, optional): [num_edges, dist_dim]
        Returns:
            Tuple[Tensor, Tensor]: Embedded distance features
        """
        dist_feat = self.fc(dist_feat)
        if dist_feat_order is not None:
            dist_feat_order = self.fc(dist_feat_order)
            return dist_feat, dist_feat_order
        return dist_feat, None

class AggregateEdgesFromNodes(nn.Module):
    """
    Aggregates edge features from node features
    """
    def __init__(self, hidden_size):
        super(AggregateEdgesFromNodes, self).__init__()
        self.fc = nn.Linear(2 * hidden_size + hidden_size, hidden_size)

    def forward(self, node_edge_feat, dist_feat, srcs, dsts):
        """
        Args:
            node_edge_feat (Tensor): [num_nodes + num_edges, hidden_size]
            dist_feat (Tensor): [num_edges, hidden_size]
            srcs (Tensor): [num_edges]
            dsts (Tensor): [num_edges]
        Returns:
            Tensor: [num_edges, hidden_size]
        """
        src_feat = node_edge_feat[srcs]  # [num_edges, hidden_size]
        dst_feat = node_edge_feat[dsts]  # [num_edges, hidden_size]
        feat_h = torch.cat([src_feat, dst_feat, dist_feat], dim=-1)  # [num_edges, 3 * hidden_size]
        feat_h = F.relu(self.fc(feat_h))  # [num_edges, hidden_size]
        return feat_h

def concat_node_edge_feat(node_feat, edge_feat):
    """
    Concatenates node and edge features to form a node-edge feature matrix.
    Args:
        node_feat (Tensor): [num_nodes, feat_size]
        edge_feat (Tensor): [num_edges, feat_size]
    Returns:
        Tensor: [num_nodes + num_edges, feat_size]
    """
    node_edge_feat = torch.cat([node_feat, edge_feat], dim=0)
    return node_edge_feat

class GATLayer(nn.Module):
    """
    Manually implemented Graph Attention Layer for updating edge features
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2, alpha=0.2, activation='relu'):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.activation = activation

        self.W = nn.Parameter(torch.Tensor(in_features, out_features * num_heads))
        self.a_src = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        self.a_dst = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

    def forward(self, edge_index, h):
        """
        Args:
            edge_index (Tensor): Edge indices, [2, num_edges]
            h (Tensor): Node features, [num_nodes + num_edges, in_features]
        Returns:
            Tensor: Updated edge features
        """
        Wh = torch.matmul(h, self.W)  # [N, num_heads * out_features]
        N = Wh.size(0)
        Wh = Wh.view(N, self.num_heads, self.out_features)  # [N, num_heads, out_features]

        src = edge_index[0]  # [num_edges]
        dst = edge_index[1]  # [num_edges]

        Wh_src = Wh[src]  # [num_edges, num_heads, out_features]
        Wh_dst = Wh[dst]  # [num_edges, num_heads, out_features]

        # Compute attention scores
        e = self.leakyrelu(
            (Wh_src * self.a_src).sum(dim=-1) + (Wh_dst * self.a_dst).sum(dim=-1)
        )  # [num_edges, num_heads]

        # Softmax normalization
        attention = F.softmax(e, dim=0)  # [num_edges, num_heads]
        attention = self.dropout(attention)

        # Weighted feature aggregation
        h_prime = torch.zeros_like(Wh)
        for head in range(self.num_heads):
            h_prime[:, head, :] = h_prime[:, head, :].index_add(
                0, dst, attention[:, head].unsqueeze(-1) * Wh_src[:, head, :]
            )

        if self.activation == 'relu':
            h_prime = F.relu(h_prime)
        elif self.activation == 'elu':
            h_prime = F.elu(h_prime)
        # Reshape back to [N, num_heads * out_features]
        h_prime = h_prime.view(N, self.num_heads * self.out_features)
        return h_prime

class SGATLayer(nn.Module):
    """
    Manually implemented Spatial Graph Attention Layer for updating node features
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2, alpha=0.2, activation='relu'):
        super(SGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.activation = activation

        self.W = nn.Parameter(torch.Tensor(in_features, out_features * num_heads))
        self.W_e = nn.Parameter(torch.Tensor(in_features, out_features * num_heads))
        self.a_src = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        self.a_dst = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        self.a_edge = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_e.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_edge.data, gain=1.414)

    def forward(self, edge_index, h, edge_feat):
        """
        Args:
            edge_index (Tensor): Edge indices, [2, num_edges]
            h (Tensor): Node features, [num_nodes + num_edges, in_features]
            edge_feat (Tensor): Edge features, [num_edges, in_features]
        Returns:
            Tensor: Updated node features
        """
        Wh = torch.matmul(h, self.W)  # [N, num_heads * out_features]
        We = torch.matmul(edge_feat, self.W_e)  # [num_edges, num_heads * out_features]
        N = Wh.size(0)
        Wh = Wh.view(N, self.num_heads, self.out_features)  # [N, num_heads, out_features]
        We = We.view(-1, self.num_heads, self.out_features)  # [num_edges, num_heads, out_features]

        src = edge_index[0]  # [num_edges]
        dst = edge_index[1]  # [num_edges]

        Wh_src = Wh[src]  # [num_edges, num_heads, out_features]
        Wh_dst = Wh[dst]  # [num_edges, num_heads, out_features]

        # Compute attention scores including edge features
        e = self.leakyrelu(
            (Wh_src * self.a_src).sum(dim=-1) +
            (Wh_dst * self.a_dst).sum(dim=-1) +
            (We * self.a_edge).sum(dim=-1)
        )  # [num_edges, num_heads]

        # Softmax normalization
        attention = F.softmax(e, dim=0)  # [num_edges, num_heads]
        attention = self.dropout(attention)

        # Weighted feature aggregation
        h_prime = torch.zeros_like(Wh)
        for head in range(self.num_heads):
            h_prime[:, head, :] = h_prime[:, head, :].index_add(
                0, dst, attention[:, head].unsqueeze(-1) * Wh_src[:, head, :]
            )

        if self.activation == 'relu':
            h_prime = F.relu(h_prime)
        elif self.activation == 'elu':
            h_prime = F.elu(h_prime)
        # Reshape back to [N, num_heads * out_features]
        h_prime = h_prime.view(N, self.num_heads * self.out_features)
        return h_prime

class SpatialConv(nn.Module):
    """
    Spatial Graph Convolution Layer combining edge and node features for feature updating
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.2, alpha=0.2, activation='relu'):
        super(SpatialConv, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.gat_edge = GATLayer(
            in_features=hidden_size,
            out_features=hidden_size // num_heads,  # Adjusted here
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            activation=activation
        )
        self.sgat_node = SGATLayer(
            in_features=hidden_size,
            out_features=hidden_size // num_heads,  # Adjusted here
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            activation=activation
        )
        self.aggregate_edges_from_nodes = AggregateEdgesFromNodes(hidden_size)

    def forward(self, data, node_edge_feat, dist_feat_order, dist_feat):
        """
        Args:
            data: torch_geometric.data.Data or torch_geometric.data.Batch object
            node_edge_feat (Tensor): [num_nodes + num_edges, hidden_size]
            dist_feat_order (Tensor): [num_edges, hidden_size]
            dist_feat (Tensor): [num_edges, hidden_size]
        Returns:
            Tensor: Updated node and edge feature matrix
        """
        srcs = data.srcs  # [num_edges]
        dsts = data.dsts  # [num_edges]
        nids = data.nids  # [num_nodes]
        eids = data.eids  # [num_edges]

        # Step 1: Update edge features
        node_feat = node_edge_feat[nids]  # [num_nodes, hidden_size]
        edge_feat = self.aggregate_edges_from_nodes(node_edge_feat, dist_feat_order, srcs, dsts)  # [num_edges, hidden_size]
        node_edge_feat_lod = concat_node_edge_feat(node_feat, edge_feat)  # [num_nodes + num_edges, hidden_size]
        node_edge_feat = self.gat_edge(data.edge_index_e2e, node_edge_feat_lod)  # Edge-to-edge graph

        # Step 2: Update node features
        edge_feat = node_edge_feat[eids]  # [num_edges, hidden_size * num_heads]
        node_edge_feat_lod = concat_node_edge_feat(node_feat, edge_feat)  # [num_nodes + num_edges, hidden_size * num_heads]
        node_edge_feat = self.sgat_node(data.edge_index_e2n, node_edge_feat_lod, dist_feat)  # Edge-to-node graph

        # Update node and edge feature matrix
        node_feat = node_edge_feat[nids]  # [num_nodes, hidden_size * num_heads]
        node_edge_feat = concat_node_edge_feat(node_feat, edge_feat)  # [num_nodes + num_edges, hidden_size * num_heads]

        return node_edge_feat

## 3. Data Preparation

### 3.1. Generating Synthetic Graph Data

def generate_synthetic_data(num_graphs, num_nodes_per_graph, num_edges_per_graph, node_feat_dim, edge_feat_dim, dist_dim):
    data_list = []
    for _ in range(num_graphs):
        num_nodes = num_nodes_per_graph
        num_edges = num_edges_per_graph

        # Node features
        node_features = torch.randn(num_nodes, node_feat_dim)

        # Edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Edge features
        edge_features = torch.randn(num_edges, edge_feat_dim)

        # Distance features
        edges_dist = torch.randn(num_edges, dist_dim)
        edges_dist_order = edges_dist.clone()  # Assuming same as edges_dist for simplicity

        # Node and edge IDs
        nids = torch.arange(num_nodes)
        eids = torch.arange(num_edges)

        # Source and destination node indices for edges
        srcs = edge_index[0]
        dsts = edge_index[1]

        # Edge indices for edge-to-edge graph (simplified as self-loop for each edge)
        edge_index_e2e = torch.stack([eids, eids], dim=0)

        # Edge indices for edge-to-node graph (edges pointing from edges to nodes)
        edge_index_e2n = torch.stack([eids, dsts], dim=0)

        # Combine node and edge features
        x = concat_node_edge_feat(node_features, edge_features)

        # Remove manual batch assignment

        # Target value (synthetic)
        pk = torch.randn(1)

        data = Data(
            x=x,
            edge_index=edge_index,
            edges_dist=edges_dist,
            edges_dist_order=edges_dist_order,
            nids=nids,
            eids=eids,
            srcs=srcs,
            dsts=dsts,
            edge_index_e2e=edge_index_e2e,
            edge_index_e2n=edge_index_e2n,
            pk=pk
        )

        data_list.append(data)

    return data_list

def custom_collate_fn(data_list):
    batch = Batch.from_data_list(data_list)

    # Initialize offsets
    node_offset = 0
    edge_offset = 0
    nids_list = []
    eids_list = []
    srcs_list = []
    dsts_list = []

    for data in data_list:
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1)

        # Adjust custom indices by the current offset
        nids_list.append(data.nids + node_offset)
        eids_list.append(data.eids + edge_offset)
        srcs_list.append(data.srcs + node_offset)
        dsts_list.append(data.dsts + node_offset)

        # Update offsets
        node_offset += num_nodes
        edge_offset += num_edges

    # Concatenate all adjusted indices
    batch.nids = torch.cat(nids_list, dim=0)
    batch.eids = torch.cat(eids_list, dim=0)
    batch.srcs = torch.cat(srcs_list, dim=0)
    batch.dsts = torch.cat(dsts_list, dim=0)

    return batch

# Parameters
num_graphs = 100
num_nodes_per_graph = 10
num_edges_per_graph = 20
node_feat_dim = 32  # Set to hidden_size
edge_feat_dim = 32  # Set to hidden_size
dist_dim = 8  # This can remain as is

# Generate synthetic data
data_list = generate_synthetic_data(
    num_graphs=num_graphs,
    num_nodes_per_graph=num_nodes_per_graph,
    num_edges_per_graph=num_edges_per_graph,
    node_feat_dim=node_feat_dim,
    edge_feat_dim=edge_feat_dim,
    dist_dim=dist_dim
)

# Create DataLoader with custom collate function
batch_size = 32  # You can set to 1 for debugging or higher for efficiency
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Model arguments
args = {
    'num_layers': 2,
    'hid_dim': 32,         # Must match node_feat_dim and edge_feat_dim
    'pool_type': 'mean',
    'drop': 0.5,
    'dist_dim': dist_dim
}

# Initialize the model
model = SMANModel(args)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_data in loader:
        # Move data to the same device as the model
        batch_data = batch_data.to(device)

        optimizer.zero_grad()
        try:
            # Run the model with intermediate outputs for debugging
            output, intermediates = model(batch_data, return_intermediate=True)

            # Print intermediate output shapes
            print(f"Batch size: {batch_size}")
            for name, shape in intermediates.items():
                print(f"{name}: {shape}")

            # Compute loss and backpropagate
            loss = F.mse_loss(output, batch_data.pk.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise e

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
