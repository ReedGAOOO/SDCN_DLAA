
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


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


class SpatialConvV4EdgePoolFusion(nn.Module):
    """
    Edge-pool fusion SpatialConv variant:
    - Treats raw edge features as a first-class signal (similar to strong pooling baselines).
    - Updates edges via edge-edge graph (ee_gat) on edges only.
    - Updates nodes via SGAT attention with updated edge embeddings.
    - Adds an explicit edge->node mean-pooling residual (gated) to ensure edge semantics
      can directly shape node embeddings for clustering.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.pool_residual = _env_flag("SDCN_POOL_RESIDUAL", True)
        self.pool_raw = _env_flag("SDCN_POOL_RAW", True)
        self.pool_upd = _env_flag("SDCN_POOL_UPD", True)
        self.pool_gate_mode = os.getenv("SDCN_POOL_GATE_MODE", "learned").strip().lower()
        if self.pool_gate_mode not in {"learned", "one", "zero"}:
            raise ValueError(
                f"Unknown SDCN_POOL_GATE_MODE={self.pool_gate_mode!r}. Use one of: learned, one, zero."
            )

        self.edge_dim_proj = None
        if self.edge_dim != hidden_size:
            self.edge_dim_proj = nn.Linear(self.edge_dim, hidden_size)
            # Stable init: pad/truncate identity so raw edge channels survive early training.
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        # Edge init: combine raw edge features with optional node-pair context.
        self.edge_from_dist = nn.Linear(hidden_size, hidden_size)
        self.edge_from_nodes = nn.Linear(hidden_size * 2, hidden_size)

        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

        # Node update (no activation here; apply once after fusion for stability).
        self.en_gat = SGATLayer(
            hidden_size,
            hidden_size,
            heads=heads,
            dropout=dropout,
            combine="mean",
            edge_dim=hidden_size,
            activation=None,
        )

        self.pool_proj = nn.Linear(hidden_size, hidden_size)
        self.pool_gate = nn.Linear(hidden_size * 2, hidden_size)

        # Initialize pool projection close to identity so pooling has a strong, stable path.
        nn.init.eye_(self.pool_proj.weight)
        nn.init.zeros_(self.pool_proj.bias)

    @staticmethod
    def _pool_edges_to_nodes_mean(edge_feat: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        node_sum = torch.zeros((num_nodes, edge_feat.size(1)), device=edge_feat.device, dtype=edge_feat.dtype)
        node_cnt = torch.zeros((num_nodes, 1), device=edge_feat.device, dtype=edge_feat.dtype)

        node_sum.index_add_(0, src, edge_feat)
        node_sum.index_add_(0, dst, edge_feat)

        ones = torch.ones((edge_feat.size(0), 1), device=edge_feat.device, dtype=edge_feat.dtype)
        node_cnt.index_add_(0, src, ones)
        node_cnt.index_add_(0, dst, ones)

        return node_sum / node_cnt.clamp(min=1.0)

    def forward(self, data):
        x = data.x  # [N, H]
        edge_index = data.edge_index  # [2, E]
        dist_feat_order = data.dist_feat_order  # [E, edge_dim]
        edge_to_edge_index = data.edge_to_edge_index  # [2, M]

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat_order.size(1) != self.hidden_size:
            dist_feat_order = self.edge_dim_proj(dist_feat_order)  # -> [E, H]

        # Raw edge signal (baseline-style) and optional node-pair context.
        edge_dist = self.edge_from_dist(dist_feat_order)
        edge_nodes = self.edge_from_nodes(torch.cat([x[srcs], x[dsts]], dim=1))
        edge_feat_0 = F.relu(edge_dist + edge_nodes)  # [E, H]

        # Edge-to-edge update (edges only).
        edge_feat_1 = self.ee_gat(edge_feat_0, edge_to_edge_index) if self.edge_ee else edge_feat_0  # [E, H]

        # Node update via attention with edge embeddings.
        node_att = self.en_gat(x, edge_index, edge_feat_1)  # [N, H]

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(dist_feat_order, edge_index, num_nodes=num_nodes)  # [N, H]
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_feat_1, edge_index, num_nodes=num_nodes)  # [N, H]

            if self.pool_gate_mode == "learned":
                gate = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate = torch.ones_like(node_att)
            else:
                gate = torch.zeros_like(node_att)

            node_out = node_att + gate * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_feat_1], dim=0)


class SpatialConvV5EdgePoolResidual(nn.Module):
    """
    V5 (derived from V2 philosophy):
    - Node attention uses the *raw* edge features (dist_feat) as edge_attr (like V2).
    - Edge embeddings are still computed and refined via edge-edge message passing.
    - Adds an explicit edge->node mean-pooling residual (gated) to match strong edge-pooling baselines.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        # Whether to pass raw edge features into node attention as edge_attr (for ablation).
        # If disabled, SGAT falls back to node-only attention (no edge_attr in attention, no edge_message injection).
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)
        self.pool_residual = _env_flag("SDCN_POOL_RESIDUAL", True)
        self.pool_raw = _env_flag("SDCN_POOL_RAW", True)
        self.pool_upd = _env_flag("SDCN_POOL_UPD", True)
        self.pool_gate_mode = os.getenv("SDCN_POOL_GATE_MODE", "learned").strip().lower()
        if self.pool_gate_mode not in {"learned", "one", "zero"}:
            raise ValueError(
                f"Unknown SDCN_POOL_GATE_MODE={self.pool_gate_mode!r}. Use one of: learned, one, zero."
            )

        self.edge_dim_proj = None
        if self.edge_dim != hidden_size:
            self.edge_dim_proj = nn.Linear(self.edge_dim, hidden_size)
            # Stable init: pad/truncate identity so raw edge channels survive early training.
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        # Edge init: [x_src, x_dst, dist_feat_order] -> edge_feat
        self.edge_fc = nn.Linear(hidden_size * 3, hidden_size)

        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

        # Node update uses raw dist_feat (as edge_attr). Defer activation to the final fusion.
        self.en_gat = SGATLayer(
            hidden_size,
            hidden_size,
            heads=heads,
            dropout=dropout,
            combine="mean",
            edge_dim=hidden_size,
            activation=None,
        )

        self.pool_proj = nn.Linear(hidden_size, hidden_size)
        self.pool_gate = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.eye_(self.pool_proj.weight)
        nn.init.zeros_(self.pool_proj.bias)

    @staticmethod
    def _pool_edges_to_nodes_mean(edge_feat: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        node_sum = torch.zeros((num_nodes, edge_feat.size(1)), device=edge_feat.device, dtype=edge_feat.dtype)
        node_cnt = torch.zeros((num_nodes, 1), device=edge_feat.device, dtype=edge_feat.dtype)

        node_sum.index_add_(0, src, edge_feat)
        node_sum.index_add_(0, dst, edge_feat)

        ones = torch.ones((edge_feat.size(0), 1), device=edge_feat.device, dtype=edge_feat.dtype)
        node_cnt.index_add_(0, src, ones)
        node_cnt.index_add_(0, dst, ones)

        return node_sum / node_cnt.clamp(min=1.0)

    def forward(self, data):
        x = data.x  # [N, H]
        edge_index = data.edge_index  # [2, E]
        dist_feat = data.dist_feat  # [E, edge_dim]
        dist_feat_order = data.dist_feat_order  # [E, edge_dim]
        edge_to_edge_index = data.edge_to_edge_index  # [2, M]

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        # Project raw edge features into hidden space if needed.
        if self.edge_dim_proj is not None:
            if dist_feat_order.size(1) != self.hidden_size:
                dist_feat_order = self.edge_dim_proj(dist_feat_order)
            if dist_feat.size(1) != self.hidden_size:
                dist_feat = self.edge_dim_proj(dist_feat)

        # ---- Step 1: edge init (V2-style) ----
        edge_feat_0 = F.relu(self.edge_fc(torch.cat([x[srcs], x[dsts], dist_feat_order], dim=1)))  # [E, H]

        # ---- Step 2: edge-to-edge update (edges only) ----
        edge_feat_1 = self.ee_gat(edge_feat_0, edge_to_edge_index) if self.edge_ee else edge_feat_0  # [E, H]

        # ---- Step 3: node update (V2 philosophy: optionally use raw dist_feat as edge_attr) ----
        node_edge_attr = dist_feat if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)  # [N, H]

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(dist_feat, edge_index, num_nodes=num_nodes)
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_feat_1, edge_index, num_nodes=num_nodes)

            if self.pool_gate_mode == "learned":
                gate = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate = torch.ones_like(node_att)
            else:
                gate = torch.zeros_like(node_att)

            node_out = node_att + gate * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_feat_1], dim=0)


_DEFAULT_SPATIALCONV_VARIANT = "v5edge_pool_residual"
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
elif SPATIALCONV_VARIANT_SELECTED in {
    "v4",
    "v4edge_pool_fusion",
    "v4_edge_pool_fusion",
    "edge_pool_fusion",
    "pool_fusion",
    "edge_pool",
    "pool",
}:
    SpatialConv = SpatialConvV4EdgePoolFusion
elif SPATIALCONV_VARIANT_SELECTED in {
    "v5",
    "v5edge_pool_residual",
    "v5_edge_pool_residual",
    "edge_pool_residual",
    "pool_residual",
    "v5v2",
    "v5_from_v2",
}:
    SpatialConv = SpatialConvV5EdgePoolResidual
else:
    raise ValueError(
        f"Unknown SPATIALCONV_VARIANT={SPATIALCONV_VARIANT_SELECTED!r}. "
        f"Use one of: v1original, v2edge_single_layer, v3edge_cross_layers, v4edge_pool_fusion, v5edge_pool_residual."
    )
