
"""
This file implements custom PyTorch Geometric layers for graph neural networks,
specifically focusing on spatial attention mechanisms, adapted from the S-MAN model.
Key components include:
- GATLayer: Standard Graph Attention Network layer using PyG's GATConv.
- SGATLayer: Spatial Graph Attention Network layer incorporating edge features.
- SpatialConv: A spatial graph convolution layer that processes node and edge features,
  including interactions between nodes and edges, and between edges themselves.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
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

class SpatialConv(nn.Module):
    """Spatial Graph Convolution Layer
    
    This layer implements the function of the spatial graph convolution layer for molecular graph.
    
    Args:
        hidden_size (int): Size of hidden features
        edge_dim (int, optional): Dimension of edge features. If None, will use hidden_size
        dropout (float): Dropout probability
        heads (int): Number of attention heads
    """
    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4):
        super(SpatialConv, self).__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        
        # Add edge dimension projection if needed
        self.edge_dim_proj = None
        if self.edge_dim != hidden_size:
            self.edge_dim_proj = nn.Linear(self.edge_dim, hidden_size)
        
        # Define linear layer for edge feature aggregation
        self.edge_fc = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
  # After projection, edge dim is always hidden_size
        
        # Define GAT layers for edge-to-edge and edge-to-node aggregation
        self.ee_gat = GATLayer(
            hidden_size, 
            hidden_size, 
            heads=heads, 
            dropout=dropout
        )
        
        self.en_gat = SGATLayer(
            hidden_size, 
            hidden_size, 
            heads=heads, 
            dropout=dropout,
            combine='mean'
        )
        
    def forward(self, data):
        """
        Args:
            data: A PyG Data object containing:
                - x: Node features [num_nodes, feature_size]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, feature_size]
                - dist_feat: Distance features for node-node graph [num_edges, embedding_size]
                - dist_feat_order: Distance features for edge-edge graph [num_edges, embedding_size]
                - edge_to_edge_index: Edge-to-edge graph connectivity [2, num_edge_edges]
                
        Returns:
            Tensor: Updated node-edge feature matrix
        """
        # Extract data components
        x = data.x  # Node features
        edge_index = data.edge_index  # Node-to-node connectivity
        edge_attr = data.edge_attr  # Edge features
        dist_feat = data.dist_feat  # Distance features for node-node graph
        dist_feat_order = data.dist_feat_order  # Distance features for edge-edge graph
        edge_to_edge_index = data.edge_to_edge_index  # Edge-to-edge connectivity
        
        num_nodes = x.shape[0]
        
        # Step 1: Update edge features
        # Get source and target node indices
        srcs, dsts = edge_index[0], edge_index[1]
        
        # Aggregate node features to update edge features
        # Project edge features if dimensions don't match
        if self.edge_dim_proj is not None and dist_feat_order.shape[1] != self.hidden_size:
            dist_feat_order = self.edge_dim_proj(dist_feat_order)
            
        src_feat = x[srcs]
        dst_feat = x[dsts]
        feat_h = torch.cat([src_feat, dst_feat, dist_feat_order], dim=1)
        edge_feat = F.relu(self.edge_fc(feat_h))
        
        # Concatenate node and edge features
        node_edge_feat = torch.cat([x, edge_feat], dim=0)
        
        # Update edge features using edge-to-edge graph
        node_edge_feat = self.ee_gat(node_edge_feat, edge_to_edge_index)
        
        # Step 2: Update node features
        # Extract updated edge features
        updated_edge_feat = node_edge_feat[num_nodes:]
        
        # Project edge features for node-edge graph if needed
        if self.edge_dim_proj is not None and dist_feat.shape[1] != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)
            
        updated_node_feat = node_edge_feat[:num_nodes]
        
        # Concatenate updated node and edge features
        node_edge_feat = torch.cat([updated_node_feat, updated_edge_feat], dim=0)
        
        # Update node features using edge-to-node graph
        node_edge_feat = self.en_gat(node_edge_feat, edge_index, dist_feat)
        
        # Extract final node features and edge features
        final_node_feat = node_edge_feat[:num_nodes]
        final_edge_feat = node_edge_feat[num_nodes:]
        
        # Concatenate final node and edge features
        final_node_edge_feat = torch.cat([final_node_feat, final_edge_feat], dim=0)
        
        return final_node_edge_feat