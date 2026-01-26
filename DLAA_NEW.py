
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
import os
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax


_UNSET = object()


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
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=4,
        dropout=0.2,
        negative_slope=0.2,
        combine='mean',
        activation=F.relu,
        edge_dim=None,
        edge_message: bool | None = None,
    ):
        super(SGATLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.combine = combine
        self.activation = activation
        self.dropout = dropout
        if edge_message is None:
            edge_message = os.getenv("SDCN_EDGE_MESSAGE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        self.edge_message = bool(edge_message)
        
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

        self.edge_msg_lin = None
        self.edge_msg_scale = None
        if self.edge_message:
            if edge_dim is None:
                raise ValueError("edge_message=True requires edge_dim to be set")
            if self.combine not in {"mean", "max", "dense"}:
                raise ValueError(f"edge_message=True currently supports combine in {{mean,max,dense}}, got: {self.combine!r}")
            self.edge_msg_lin = nn.Linear(int(edge_dim), out_channels, bias=False)
            self.edge_msg_scale = nn.Parameter(torch.tensor(1.0))
        
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

        if self.edge_message and edge_attr is not None:
            if self.edge_msg_lin is None or self.edge_msg_scale is None:
                raise RuntimeError("edge_message is enabled but edge_msg_lin/edge_msg_scale is not initialized")
            dst = edge_index[1]
            edge_msg = self.edge_msg_lin(edge_attr)  # [E, out_channels]
            node_sum = torch.zeros((x.size(0), self.out_channels), device=x.device, dtype=edge_msg.dtype)
            node_sum.index_add_(0, dst, edge_msg)
            node_cnt = torch.zeros((x.size(0), 1), device=x.device, dtype=edge_msg.dtype)
            ones = torch.ones((edge_msg.size(0), 1), device=x.device, dtype=edge_msg.dtype)
            node_cnt.index_add_(0, dst, ones)
            node_mean = node_sum / node_cnt.clamp(min=1.0)
            out = out + self.edge_msg_scale * node_mean
        
        # Add bias and apply activation
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out

class SpatialConvV1Original(nn.Module):
    """Spatial Graph Convolution Layer
    
    This layer implements the function of the spatial graph convolution layer for molecular graph.
    
    Args:
        hidden_size (int): Size of hidden features
        edge_dim (int, optional): Dimension of edge features. If None, will use hidden_size
        dropout (float): Dropout probability
        heads (int): Number of attention heads
    """
    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        out_activation = activation if out_activation is _UNSET else out_activation
        
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
            dropout=dropout,
            activation=activation,
        )
        
        self.en_gat = SGATLayer(
            hidden_size, 
            hidden_size, 
            heads=heads, 
            dropout=dropout,
            combine='mean',
            edge_dim=hidden_size,
            activation=out_activation,
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
        return torch.cat([final_node_feat, final_edge_feat], dim=0)


class SpatialConvV2EdgeSingleLayer(nn.Module):
    """
    A minimal-change SpatialConv variant:
    - Keeps the original (node+edge concatenation) edge-edge update flow.
    - Ensures dist_feat actually participates in node attention via `edge_dim`.
    - Avoids passing edge rows through node update to prevent washing edge features.
    """
    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_dim_proj = None
        if self.edge_dim != hidden_size:
            self.edge_dim_proj = nn.Linear(self.edge_dim, hidden_size)

        # edge init: [x_src, x_dst, dist_feat_order] -> edge_feat
        self.edge_fc = nn.Linear(hidden_size * 3, hidden_size)

        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

        # Ensure dist_feat participates in attention
        self.en_gat = SGATLayer(
            hidden_size,
            hidden_size,
            heads=heads,
            dropout=dropout,
            combine='mean',
            edge_dim=hidden_size,
            activation=out_activation,
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        dist_feat_order = data.dist_feat_order
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)

        # ---- Step 1: edge init (same as original) ----
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat_order.size(1) != self.hidden_size:
            dist_feat_order = self.edge_dim_proj(dist_feat_order)  # -> [E, H]

        src_feat = x[srcs]
        dst_feat = x[dsts]
        feat_h = torch.cat([src_feat, dst_feat, dist_feat_order], dim=1)  # [E, 3H]
        edge_feat = F.relu(self.edge_fc(feat_h))  # [E, H]

        # ---- Step 2: edge-to-edge update (keep original info flow) ----
        node_edge_feat = torch.cat([x, edge_feat], dim=0)  # [(N+E), H]
        node_edge_feat = self.ee_gat(node_edge_feat, edge_to_edge_index)

        updated_node_feat = node_edge_feat[:num_nodes]   # [(N), H]
        updated_edge_feat = node_edge_feat[num_nodes:]   # [(E), H]

        # ---- Step 3: node update using dist_feat (keep original info flow) ----
        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)  # -> [E, H]

        # Only update nodes; keep edge features from Step 2.
        final_node_feat = self.en_gat(updated_node_feat, edge_index, dist_feat)  # [N, H]

        return torch.cat([final_node_feat, updated_edge_feat], dim=0)  # [(N+E), H]


class SpatialConvV3EdgeCrossLayers(nn.Module):
    """
    Edge-centric SpatialConv variant:
    - Computes edge embeddings from (src, dst, dist_feat_order).
    - Updates edges via edge-edge graph (ee_gat) on edges only.
    - Updates nodes via SGATLayer using the updated edge embeddings as `edge_attr`.
    """
    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_dim_proj = None
        if self.edge_dim != hidden_size:
            self.edge_dim_proj = nn.Linear(self.edge_dim, hidden_size)

        self.edge_fc = nn.Linear(hidden_size * 3, hidden_size)

        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

        self.en_gat = SGATLayer(
            hidden_size,
            hidden_size,
            heads=heads,
            dropout=dropout,
            combine='mean',
            edge_dim=hidden_size,
            activation=out_activation,
        )

    def forward(self, data):
        x = data.x  # [N, H]
        edge_index = data.edge_index  # [2, E]
        dist_feat_order = data.dist_feat_order  # [E, edge_dim]
        edge_to_edge_index = data.edge_to_edge_index  # [2, M] (edge indices)

        num_edges = edge_index.size(1)

        # Step 1: Initialize edge features from node features and raw distance features.
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat_order.shape[1] != self.hidden_size:
            dist_feat_order = self.edge_dim_proj(dist_feat_order)

        src_feat = x[srcs]
        dst_feat = x[dsts]
        feat_h = torch.cat([src_feat, dst_feat, dist_feat_order], dim=1)
        edge_feat_0 = F.relu(self.edge_fc(feat_h))  # [E, H]

        # Step 2: Update edge features with edge-to-edge interactions (operate on edges only).
        edge_feat_1 = self.ee_gat(edge_feat_0, edge_to_edge_index)  # [E, H]

        # Step 3: Update node features with node-to-node graph, incorporating updated edge features as attention inputs.
        node_feat_1 = self.en_gat(x, edge_index, edge_feat_1)  # [N, H]

        # Keep the original return contract: concatenate node and edge features.
        return torch.cat([node_feat_1, edge_feat_1], dim=0)


_DEFAULT_SPATIALCONV_VARIANT = "v2edge_single_layer"
SPATIALCONV_VARIANT_SELECTED = os.getenv("SPATIALCONV_VARIANT", _DEFAULT_SPATIALCONV_VARIANT).strip().lower()

if SPATIALCONV_VARIANT_SELECTED in {
    "v1",
    "v1original",
    "v1_original",
    "orig",
    "original",
    "legacy",
}:
    SpatialConv = SpatialConvV1Original
elif SPATIALCONV_VARIANT_SELECTED in {
    "v2",
    "v2edge_single_layer",
    "v2_edge_single_layer",
    "edge_single_layer",
    "single_layer",
    "small",
    "smallfix",
    "small_fix",
    "minimal",
}:
    SpatialConv = SpatialConvV2EdgeSingleLayer
elif SPATIALCONV_VARIANT_SELECTED in {
    "v3",
    "v3edge_cross_layers",
    "v3_edge_cross_layers",
    "edge_cross_layers",
    "cross_layers",
    "edge_only",
    "edgeonly",
    "edge",
}:
    SpatialConv = SpatialConvV3EdgeCrossLayers
else:
    raise ValueError(
        f"Unknown SPATIALCONV_VARIANT={SPATIALCONV_VARIANT_SELECTED!r}. "
        f"Use one of: v1original, v2edge_single_layer, v3edge_cross_layers."
    )
