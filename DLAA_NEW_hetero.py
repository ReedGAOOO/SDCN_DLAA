
"""
This file implements custom PyTorch Geometric layers for graph neural networks,
specifically focusing on heterogeneous graph processing and spatial attention mechanisms.
Key components include:
- GATLayer: Standard Graph Attention Network layer using PyG's GATConv.
- SGATLayer: Spatial Graph Attention Network layer incorporating edge features.
- EdgeInitLayer: Computes initial edge features based on connected nodes and distance.
- HeteroSpatialConv: A heterogeneous spatial graph convolution layer that processes
  nodes and edges separately, avoiding node+edge feature concatenation and handling
  node-edge and edge-edge interactions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing # Import MessagePassing
from torch_geometric.utils import softmax


class GATLayer(nn.Module):
    """Graph Attention Network Layer
    
    Implementation of graph attention networks (GAT) using PyG's GATConv.
    This replaces the original gat function.
    
    Args:
        in_channels (int): Size of input features
        out_channels (int): Size of output features
        heads (int): Number of attention heads
        dropout (float): Dropout probability
        negative_slope (float): LeakyReLU negative slope
        activation (callable): Activation function
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2, 
                 negative_slope=0.2, activation=F.relu):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(
            in_channels, 
            out_channels, 
            heads=heads,
            dropout=dropout,
            negative_slope=negative_slope,
            concat=False  # Use mean aggregation by default
        )
        self.activation = activation
        
        # Create bias parameter similar to original implementation
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.zeros_(self.bias)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (Tensor): Graph connectivity [2, num_edges]
            edge_attr (Tensor, optional): Edge features [num_edges, edge_dim]
            
        Returns:
            Tensor: Updated node features [num_nodes, out_channels]
        """
        # Apply dropout to input features if in training mode
        if self.training:
            x = F.dropout(x, p=0.2)
            
        # Apply GATConv
        out = self.gat_conv(x, edge_index)
        
        # Add bias and apply activation
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out
    
class SGATLayer(nn.Module):
    """Spatial Graph Attention Network Layer
    
    Explicitly considers edge features in the attention calculation.
    Uses PyG's built-in GATConv by specifying edge_dim during construction.
    
    Args:
        in_channels (int): Dimensionality of input features
        out_channels (int): Dimensionality of output features
        heads (int): Number of attention heads
        dropout (float): Dropout probability
        negative_slope (float): Negative slope for LeakyReLU
        combine (str): Method to combine multi-head results: 'mean', 'max', or 'dense'
        activation (callable): Activation function
        edge_dim (int): Dimensionality of edge features (required for attention)
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2, 
                 negative_slope=0.2, combine='mean', activation=F.relu, edge_dim=None):
        super(SGATLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.combine = combine
        self.activation = activation
        self.dropout = dropout
        
        # Directly use PyG's GATConv and specify edge_dim
        # concat=True means the output shape is [num_nodes, heads * out_channels]
        # We will manually handle mean/max/dense aggregation for multi-head results later
        self.gat_conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            negative_slope=negative_slope,
            edge_dim=edge_dim,   # Key: Specify edge feature dimension
            concat=True
        )
        
        # If combine == 'dense', add a linear layer to map heads*out_channels -> out_channels
        if self.combine == 'dense':
            self.dense_combine = nn.Linear(heads * out_channels, out_channels, bias=False)
        
        # Keep a bias parameter, similar to before
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x (Tensor): Node features [N, in_channels]
            edge_index (Tensor): Graph connectivity [2, E]
            edge_attr (Tensor, optional): Edge features [E, edge_dim]
        Returns:
            (Tensor): Updated node features. Shape depends on 'combine':
                      - 'mean'/'max': [N, out_channels]
                      - 'dense':      [N, out_channels]
                      - default:      [N, heads*out_channels]
        """
        if self.training:
            x = F.dropout(x, p=self.dropout)
            if edge_attr is not None:
                edge_attr = F.dropout(edge_attr, p=self.dropout)
        
        # Call GATConv, passing edge features
        out = self.gat_conv(x, edge_index, edge_attr)
        # out shape: [N, heads*out_channels]
        
        if self.combine in ['mean', 'max']:
            # Reshape to separate heads: [N, heads, out_channels]
            out = out.view(-1, self.heads, self.out_channels)
            if self.combine == 'mean':
                out = out.mean(dim=1)  # [N, out_channels]
            else:  # 'max'
                out, _ = out.max(dim=1)  # [N, out_channels]
        elif self.combine == 'dense':
            # Apply the dense combination layer (input is already [N, heads*out_channels])
            out = self.dense_combine(out)  # [N, out_channels]
        else:
            # Default: do nothing, keep shape [N, heads*out_channels]
            pass
        
        # Add bias and apply activation
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out

# --- Custom Layer to Replicate Initial Edge Feature Calculation ---
# This layer computes edge features based on connected nodes and distance features
# It replaces the original edge_fc logic without explicit node-edge concatenation

class EdgeInitLayer(MessagePassing):
    """
    Computes initial edge features based on source node, destination node,
    and distance features using the logic similar to the original edge_fc.
    """
    def __init__(self, node_hidden_size, edge_feature_dim, output_edge_hidden_size):
        super().__init__(aggr=None) # We compute per edge, no aggregation needed here
        self.node_hs = node_hidden_size
        self.edge_dim = edge_feature_dim
        self.out_hs = output_edge_hidden_size

        # Replicates the original edge_fc layer's input logic
        # Input to fc: node_src (hs) + node_dst (hs) + dist_feat (edge_dim)
        self.edge_fc = nn.Linear(node_hidden_size * 2 + edge_feature_dim, output_edge_hidden_size)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, dist_feat_order):
        """
        Args:
            x (Tensor): Node features [N, node_hidden_size]
            edge_index (Tensor): Node-node connectivity [2, E]
            dist_feat_order (Tensor): Distance features [E, edge_feature_dim]

        Returns:
            Tensor: Initial edge features [E, output_edge_hidden_size]
        """
        # propagate initiates the message passing process
        # We pass all necessary features that message() will need
        return self.propagate(edge_index, x=x, dist_feat_order=dist_feat_order, size=(x.size(0), x.size(0)))

    def message(self, x_i, x_j, dist_feat_order):
        """
        Computes the feature for a single edge.
        x_i: Features of the target node [E, node_hidden_size]
        x_j: Features of the source node [E, node_hidden_size]
        dist_feat_order: Distance features for these edges [E, edge_feature_dim]
        """
        # Concatenate features for each edge
        feat_h = torch.cat([x_j, x_i, dist_feat_order], dim=-1) # [E, 2*node_hs + edge_dim]
        
        # Apply the linear layer and activation
        edge_output = self.act(self.edge_fc(feat_h)) # [E, out_hs]
        return edge_output

    # No aggregation or update needed as we compute directly per edge
    # def aggregate(...): pass
    # def update(...): pass


# --- The Heterogeneous SpatialConv Layer ---

class HeteroSpatialConv(nn.Module):
    """
    Heterogeneous Spatial Graph Convolution Layer.

    Processes nodes and edges separately, avoiding N+E concatenation.
    Follows the interaction logic:
    1. Initialize edge features based on nodes and distances.
    2. Update edge features using edge-edge interactions (GAT).
    3. Update node features using node-edge interactions (Spatial GAT),
       incorporating updated edge features.
    """
    def __init__(self, hidden_size, edge_dim, dropout, heads):
        super().__init__()
        self.hidden_size = hs = hidden_size
        self.edge_dim = edge_dim # Original edge dimension from input data

        # Layer 1: Initialize Edge Features
        # Takes node features (hs), distance features (edge_dim) -> edge features (hs)
        self.edge_init = EdgeInitLayer(hs, edge_dim, hs)

        # Layer 2: Edge-Edge Interaction (operates on edge features)
        self.ee_gat = GATLayer(
            hs, hs, heads=heads, dropout=dropout
        )

        # Layer 3: Edge-Node Interaction (updates node features)
        # Takes node features (hs), edge features (hs) + dist_feat (edge_dim)
        # We need SGATLayer to accept edge features of dimension hs + edge_dim
        # OR we project dist_feat to hs and concatenate with updated edge features
        
        # Option A: Project dist_feat and concatenate
        self.dist_feat_proj_en = nn.Linear(edge_dim, hs) # Project dist_feat for en_gat
        sgat_edge_dim = hs + hs # updated_edge_feat (hs) + projected_dist_feat (hs)
        
        # Option B: Modify SGATLayer (more complex) - Let's stick with Option A

        self.en_gat = SGATLayer(
            in_channels=hs,     # Input node feature dimension
            out_channels=hs,    # Output node feature dimension
            heads=heads,
            dropout=dropout,
            combine='mean',     # As in original
            edge_dim=sgat_edge_dim # Dimension of concatenated edge features for attention
        )

    def forward(self, data):
        """
        Args:
            data: A PyG Data or similar object containing:
                - x: Node features [N, hs] (Assume input projected to hs)
                - edge_index: Node-node connectivity [2, E]
                - edge_attr: Original edge attributes [E, edge_dim] (unused directly?)
                - dist_feat: Distance features for en_gat [E, edge_dim]
                - dist_feat_order: Distance features for edge_init [E, edge_dim]
                - edge_to_edge_index: Edge-to-edge connectivity [2, M]

        Returns:
            Tuple[Tensor, Tensor]:
                - x_updated: Updated node features [N, hs]
                - edge_feat_updated: Updated edge features [E, hs]
        """
        x = data.x
        edge_index = data.edge_index
        # edge_attr = data.edge_attr # Might not be needed if dist features are used
        dist_feat = data.dist_feat
        dist_feat_order = data.dist_feat_order
        edge_to_edge_index = data.edge_to_edge_index

        # 1. Initialize edge features using nodes and dist_feat_order
        # edge_init takes node features 'x' (N, hs) and dist_feat_order (E, edge_dim)
        edge_feat_0 = self.edge_init(x, edge_index, dist_feat_order) # Output: (E, hs)

        # 2. Update edge features using edge-to-edge interactions
        # ee_gat takes edge features 'edge_feat_0' (E, hs)
        edge_feat_1 = self.ee_gat(edge_feat_0, edge_to_edge_index) # Output: (E, hs)

        # 3. Update node features using edge-to-node interactions
        # en_gat takes node features 'x' (N, hs)
        # It needs edge features for attention. We combine updated edge features
        # and projected distance features (dist_feat).

        # Project dist_feat for en_gat input
        dist_feat_proj = self.dist_feat_proj_en(dist_feat) # (E, hs)

        # Concatenate edge features for SGATLayer's attention mechanism
        en_edge_features = torch.cat([edge_feat_1, dist_feat_proj], dim=-1) # (E, hs + hs)

        # en_gat updates node features 'x' using 'edge_index' and 'en_edge_features'
        x_1 = self.en_gat(x, edge_index, en_edge_features) # Output: (N, hs)

        # Return the updated node and edge features separately
        return x_1, edge_feat_1