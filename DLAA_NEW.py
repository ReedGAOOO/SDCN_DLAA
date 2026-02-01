
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
        input_dropout_env = os.getenv("SDCN_GAT_INPUT_DROPOUT", "").strip()
        # Backward compatible default: the original DLAA drafts used a fixed 0.2 feature-dropout here.
        # This dropout is independent of GATConv's internal attention-dropout (controlled by `dropout`).
        self.input_dropout = float(input_dropout_env) if input_dropout_env else 0.2
        self.input_dropout = max(0.0, min(1.0, float(self.input_dropout)))
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
        if self.training and self.input_dropout > 0:
            x = F.dropout(x, p=self.input_dropout)
            
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

        # Update edge features using edge-to-edge graph.
        # Note: `data.edge_to_edge_index` is defined over edge ids [0..E-1], while `node_edge_feat`
        # is concatenated as [nodes; edges]. Offset to address the edge rows.
        if edge_to_edge_index.numel() > 0:
            edge_to_edge_index = edge_to_edge_index + num_nodes
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
        if edge_to_edge_index.numel() > 0:
            edge_to_edge_index = edge_to_edge_index + num_nodes
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


class SpatialConvV6EdgeEeAux(nn.Module):
    """
    V6 (EE-aux, derived from V5):
    - Keeps V5's three-path design (node_att(raw edge), edge↔edge update, pooling residual).
    - Adds an *edge-level auxiliary head* (within-edge probability) to make edge↔edge modeling
      directly optimizable for clustering (loss computed in the SDCN training loop).

    Exposes (per forward call):
      - self._last_edge_within_logit: [E] logits (float)
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.edge_fc = nn.Linear(hidden_size * 3, hidden_size)
        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

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

        # Edge-level auxiliary head: predict within-edge probability (logit).
        self.edge_within_lin = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.edge_within_lin.bias)

        self._last_edge_within_logit: torch.Tensor | None = None

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
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        dist_feat_order = data.dist_feat_order
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None:
            if dist_feat_order.size(1) != self.hidden_size:
                dist_feat_order = self.edge_dim_proj(dist_feat_order)
            if dist_feat.size(1) != self.hidden_size:
                dist_feat = self.edge_dim_proj(dist_feat)

        edge_feat_0 = F.relu(self.edge_fc(torch.cat([x[srcs], x[dsts], dist_feat_order], dim=1)))
        edge_feat_1 = self.ee_gat(edge_feat_0, edge_to_edge_index) if self.edge_ee else edge_feat_0

        # Store edge auxiliary logits for the training loop to consume.
        self._last_edge_within_logit = self.edge_within_lin(edge_feat_1).squeeze(-1)

        node_edge_attr = dist_feat if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

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


class SpatialConvV7EdgeAttrFusion(nn.Module):
    """
    V7 (edge_attr fusion, designed to make edge↔edge stably help clustering):
    - Still uses raw edge features (dist_feat) for node attention like V5/V2.
    - Learns an edge embedding via (x_src, x_dst, dist_feat_order) and refines it via edge↔edge GAT.
    - *Fuses* the refined edge embedding into the node-attention edge_attr:
        edge_attr_att = dist_feat_proj + fuse_scale * norm(edge_feat_1)
      so edge↔edge can directly modulate node attention (not only via pooling).
    - Keeps the pooling residual path (raw + updated edges) to match strong edge-pooling baselines.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)
        self.edge_attr_fuse = _env_flag("SDCN_EDGE_ATTR_FUSE", True)
        self.edge_attr_fuse_detach = _env_flag("SDCN_EDGE_ATTR_FUSE_DETACH", False)
        fuse_scale_env = os.getenv("SDCN_EDGE_ATTR_FUSE_SCALE", "").strip()
        self.edge_attr_fuse_scale = float(fuse_scale_env) if fuse_scale_env else 0.5
        self.edge_fuse_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_FUSE_NORM", True) else None

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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        # Edge embedding from (src, dst, raw edge feat).
        self.edge_fc = nn.Linear(hidden_size * 3, hidden_size)
        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

        # Node update via attention with edge features (activation deferred to the end).
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
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        dist_feat_order = data.dist_feat_order
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None:
            if dist_feat_order.size(1) != self.hidden_size:
                dist_feat_order = self.edge_dim_proj(dist_feat_order)
            if dist_feat.size(1) != self.hidden_size:
                dist_feat = self.edge_dim_proj(dist_feat)

        # Edge embedding and edge↔edge refinement.
        edge_feat_0 = F.relu(self.edge_fc(torch.cat([x[srcs], x[dsts], dist_feat_order], dim=1)))
        edge_feat_1 = self.ee_gat(edge_feat_0, edge_to_edge_index) if self.edge_ee else edge_feat_0

        # Node attention edge_attr (raw + fused refined edge embedding).
        node_edge_attr = dist_feat if self.node_att_edge else None
        if node_edge_attr is not None and self.edge_attr_fuse and self.edge_attr_fuse_scale != 0.0:
            fuse = edge_feat_1.detach() if self.edge_attr_fuse_detach else edge_feat_1
            if self.edge_fuse_norm is not None:
                fuse = self.edge_fuse_norm(fuse)
            node_edge_attr = node_edge_attr + float(self.edge_attr_fuse_scale) * fuse

        node_att = self.en_gat(x, edge_index, node_edge_attr)

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


class SpatialConvV8EdgeDenoiseAttr(nn.Module):
    """
    V8 (edge-edge denoising on raw edge_attr):
    - Projects raw edge features (dist_feat) into hidden space.
    - Runs edge↔edge message passing on *raw edge features* (edge_raw) to obtain edge_denoised.
    - Uses edge_denoised as edge_attr for node attention and as the "updated edge" for pooling.
    This is aimed at the regime where edge attributes are high-dimensional but noisy.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)

        alpha_env = os.getenv("SDCN_EDGE_DENOISE_ALPHA", "").strip()
        self.denoise_alpha = float(alpha_env) if alpha_env else 1.0
        denoise_mode = os.getenv("SDCN_EDGE_DENOISE_MODE", "gat").strip().lower()
        if denoise_mode in {"", "default"}:
            denoise_mode = "gat"
        if denoise_mode in {"sim"}:
            denoise_mode = "similarity"
        if denoise_mode not in {"gat", "similarity"}:
            raise ValueError(
                f"Unknown SDCN_EDGE_DENOISE_MODE={denoise_mode!r}. Use one of: gat, similarity."
            )
        self.denoise_mode = denoise_mode
        gamma_env = os.getenv("SDCN_EDGE_SIM_GAMMA", "").strip()
        self.sim_gamma = float(gamma_env) if gamma_env else 1.0
        self.edge_denoise_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_DENOISE_NORM", True) else None

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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

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

    @staticmethod
    def _edge_similarity_smooth(edge_raw: torch.Tensor, edge_to_edge_index: torch.Tensor, gamma: float) -> torch.Tensor:
        if edge_to_edge_index is None or edge_to_edge_index.numel() == 0:
            return edge_raw
        s = edge_to_edge_index[0]
        t = edge_to_edge_index[1]
        diff = edge_raw[s] - edge_raw[t]
        d2 = (diff * diff).mean(dim=1)  # [M]
        w = torch.exp(-float(gamma) * d2).clamp(min=1e-8)  # [M]

        n_edges = edge_raw.size(0)
        w_sum = torch.zeros((n_edges,), device=edge_raw.device, dtype=edge_raw.dtype)
        w_sum.index_add_(0, s, w)

        out = torch.zeros_like(edge_raw)
        out.index_add_(0, s, edge_raw[t] * w.unsqueeze(1))
        out = out / w_sum.clamp(min=1e-8).unsqueeze(1)
        return out

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)

        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)

        edge_raw = dist_feat

        if self.edge_ee:
            if self.denoise_mode == "gat":
                edge_upd = self.ee_gat(edge_raw, edge_to_edge_index)
            else:
                edge_upd = self._edge_similarity_smooth(edge_raw, edge_to_edge_index, gamma=float(self.sim_gamma))
        else:
            edge_upd = edge_raw

        a = float(self.denoise_alpha)
        if a <= 0.0:
            edge_denoised = edge_raw
        elif a >= 1.0:
            edge_denoised = edge_upd
        else:
            edge_denoised = (1.0 - a) * edge_raw + a * edge_upd

        if self.edge_denoise_norm is not None:
            edge_denoised = self.edge_denoise_norm(edge_denoised)

        node_edge_attr = edge_denoised if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_raw, edge_index, num_nodes=num_nodes)
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_denoised, edge_index, num_nodes=num_nodes)

            if self.pool_gate_mode == "learned":
                gate = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate = torch.ones_like(node_att)
            else:
                gate = torch.zeros_like(node_att)

            node_out = node_att + gate * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_denoised], dim=0)


class SpatialConvV9EdgeContextDenoise(nn.Module):
    """
    V9 (context-aware edge denoising):
    - Builds a raw edge representation from raw edge_attr plus a *gated* node-pair context residual:
        edge_raw = dist_feat + s * Wn([x_src, x_dst]),  where s starts near 0 for stability.
    - Applies edge↔edge message passing on edge_raw (line-graph) and blends with a residual:
        edge_denoised = (1-α)*edge_raw + α*ee(edge_raw)
    - Uses edge_denoised as edge_attr for node attention AND as the "updated edge" for pooling.

    This directly targets the next step suggested by the v8 conclusion:
    unify "node-pair semantics" + "edge-edge denoise" in a single information flow.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)

        alpha_env = os.getenv("SDCN_EDGE_DENOISE_ALPHA", "").strip()
        self.denoise_alpha = float(alpha_env) if alpha_env else 0.5
        self.edge_denoise_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_DENOISE_NORM", True) else None

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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.node_ctx_lin = nn.Linear(hidden_size * 2, hidden_size)
        self.node_ctx_scale = nn.Parameter(torch.tensor(0.0))
        self.edge_raw_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_RAW_NORM", True) else None

        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

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
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)

        node_ctx = self.node_ctx_lin(torch.cat([x[srcs], x[dsts]], dim=1))
        node_ctx_scale = torch.tanh(self.node_ctx_scale)
        edge_raw = dist_feat + node_ctx_scale * node_ctx
        if self.edge_raw_norm is not None:
            edge_raw = self.edge_raw_norm(edge_raw)

        if self.edge_ee:
            edge_upd = self.ee_gat(edge_raw, edge_to_edge_index)
        else:
            edge_upd = edge_raw

        a = float(self.denoise_alpha)
        if a <= 0.0:
            edge_denoised = edge_raw
        elif a >= 1.0:
            edge_denoised = edge_upd
        else:
            edge_denoised = (1.0 - a) * edge_raw + a * edge_upd

        if self.edge_denoise_norm is not None:
            edge_denoised = self.edge_denoise_norm(edge_denoised)

        node_edge_attr = edge_denoised if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(dist_feat, edge_index, num_nodes=num_nodes)
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_denoised, edge_index, num_nodes=num_nodes)

            if self.pool_gate_mode == "learned":
                gate = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate = torch.ones_like(node_att)
            else:
                gate = torch.zeros_like(node_att)

            node_out = node_att + gate * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_denoised], dim=0)


class SpatialConvV10EdgeBaseDenoisePlusContext(nn.Module):
    """
    V10 (base denoise + context):
    - First denoise ONLY the raw edge_attr (dist_feat) via edge↔edge message passing:
        base_upd = ee(dist_feat)
        base_denoised = (1-α)*dist_feat + α*base_upd
    - Then add a gated node-pair context residual:
        edge_attr = norm(base_denoised + s*Wn([x_src, x_dst]))

    Motivation:
    keep edge↔edge focused as a *conservative denoiser/regularizer* on the raw edge signal,
    while still letting node-pair context complement edge_attr without destabilizing the ee pathway.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)

        alpha_env = os.getenv("SDCN_EDGE_DENOISE_ALPHA", "").strip()
        self.denoise_alpha = float(alpha_env) if alpha_env else 0.5
        self.edge_denoise_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_DENOISE_NORM", True) else None

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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.node_ctx_lin = nn.Linear(hidden_size * 2, hidden_size)
        self.node_ctx_scale = nn.Parameter(torch.tensor(0.0))

        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)

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
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)

        edge_raw = dist_feat

        if self.edge_ee:
            edge_upd = self.ee_gat(edge_raw, edge_to_edge_index)
        else:
            edge_upd = edge_raw

        a = float(self.denoise_alpha)
        if a <= 0.0:
            edge_base = edge_raw
        elif a >= 1.0:
            edge_base = edge_upd
        else:
            edge_base = (1.0 - a) * edge_raw + a * edge_upd

        node_ctx = self.node_ctx_lin(torch.cat([x[srcs], x[dsts]], dim=1))
        edge_attr = edge_base + torch.tanh(self.node_ctx_scale) * node_ctx

        if self.edge_denoise_norm is not None:
            edge_attr = self.edge_denoise_norm(edge_attr)

        node_edge_attr = edge_attr if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_raw, edge_index, num_nodes=num_nodes)
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_attr, edge_index, num_nodes=num_nodes)

            if self.pool_gate_mode == "learned":
                gate = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate = torch.ones_like(node_att)
            else:
                gate = torch.zeros_like(node_att)

            node_out = node_att + gate * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_attr], dim=0)


class SpatialConvV11EdgeAdaptiveDenoiseContext(nn.Module):
    """
    V11 (adaptive edge-edge denoising + context):
    - Uses edge↔edge as a *conditional* denoiser: edges with high local inconsistency (w.r.t. their
      edge↔edge neighbors) get stronger denoising, while consistent edges stay closer to raw input.

    Pipeline:
      edge_raw = dist_feat
      edge_upd = ee(edge_raw)
      inco_i = mean_j ||edge_raw[i] - edge_raw[j]||^2   over edge↔edge neighbors
      gate_i = sigmoid(beta * (inco_i / mean(inco) - 1))
      alpha_i = alpha * gate_i
      edge_base = edge_raw + alpha_i * (edge_upd - edge_raw)
      edge_attr = norm(edge_base + s*Wn([x_src, x_dst]))

    This targets the regime where only a subset of edges are noisy/outlier-ish.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)

        alpha_env = os.getenv("SDCN_EDGE_DENOISE_ALPHA", "").strip()
        self.denoise_alpha = float(alpha_env) if alpha_env else 0.5
        beta_env = os.getenv("SDCN_EDGE_ADAPT_BETA", "").strip()
        self.adapt_beta = float(beta_env) if beta_env else 5.0

        self.edge_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_DENOISE_NORM", True) else None

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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.node_ctx_lin = nn.Linear(hidden_size * 2, hidden_size)
        self.node_ctx_scale = nn.Parameter(torch.tensor(0.0))

        self.ee_gat = GATLayer(hidden_size, hidden_size, heads=heads, dropout=dropout, activation=activation)
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

    @staticmethod
    def _edge_inconsistency(edge_feat: torch.Tensor, edge_to_edge_index: torch.Tensor) -> torch.Tensor:
        if edge_to_edge_index is None or edge_to_edge_index.numel() == 0:
            return torch.zeros((edge_feat.size(0),), device=edge_feat.device, dtype=edge_feat.dtype)
        s = edge_to_edge_index[0]
        t = edge_to_edge_index[1]
        diff = edge_feat[s] - edge_feat[t]
        d = (diff * diff).mean(dim=1)  # [M]

        n_edges = edge_feat.size(0)
        sum_d = torch.zeros((n_edges,), device=edge_feat.device, dtype=edge_feat.dtype)
        cnt_d = torch.zeros((n_edges,), device=edge_feat.device, dtype=edge_feat.dtype)
        ones = torch.ones_like(d)

        sum_d.index_add_(0, s, d)
        sum_d.index_add_(0, t, d)
        cnt_d.index_add_(0, s, ones)
        cnt_d.index_add_(0, t, ones)

        return sum_d / cnt_d.clamp(min=1.0)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)

        edge_raw = dist_feat

        if self.edge_ee:
            edge_upd = self.ee_gat(edge_raw, edge_to_edge_index)
            inco = self._edge_inconsistency(edge_raw, edge_to_edge_index)
            mean_inco = inco.mean().clamp(min=1e-8)
            gate = torch.sigmoid(float(self.adapt_beta) * (inco / mean_inco - 1.0))  # [E]
            a = float(self.denoise_alpha) * gate  # [E]
            edge_base = edge_raw + a.unsqueeze(1) * (edge_upd - edge_raw)
        else:
            edge_base = edge_raw

        node_ctx = self.node_ctx_lin(torch.cat([x[srcs], x[dsts]], dim=1))
        edge_attr = edge_base + torch.tanh(self.node_ctx_scale) * node_ctx

        if self.edge_norm is not None:
            edge_attr = self.edge_norm(edge_attr)

        node_edge_attr = edge_attr if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_raw, edge_index, num_nodes=num_nodes)
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_attr, edge_index, num_nodes=num_nodes)

            if self.pool_gate_mode == "learned":
                gate_n = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate_n = torch.ones_like(node_att)
            else:
                gate_n = torch.zeros_like(node_att)

            node_out = node_att + gate_n * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_attr], dim=0)


class SpatialConvV12EdgeSimilarityDenoise(nn.Module):
    """
    V12 (similarity-weighted edge-edge denoise):
    - Treats edge↔edge as a conservative smoothing operator conditioned on edge similarity.
    - On the edge↔edge graph, compute weights from edge_raw differences:
        w_ij = exp(-gamma * ||e_i - e_j||^2)
      then row-normalize and smooth:
        edge_upd_i = sum_j w_ij * e_j
      and blend with residual:
        edge_base = (1-α)*edge_raw + α*edge_upd

    Compared to a learned ee_gat, this is much less likely to mix incompatible edges
    (important when edge_to_edge_index includes many heterogeneous neighbors).
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)

        alpha_env = os.getenv("SDCN_EDGE_DENOISE_ALPHA", "").strip()
        self.denoise_alpha = float(alpha_env) if alpha_env else 0.1
        gamma_env = os.getenv("SDCN_EDGE_SIM_GAMMA", "").strip()
        self.sim_gamma = float(gamma_env) if gamma_env else 1.0

        self.edge_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_DENOISE_NORM", True) else None

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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.node_ctx_lin = nn.Linear(hidden_size * 2, hidden_size)
        self.node_ctx_scale = nn.Parameter(torch.tensor(0.0))

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

    @staticmethod
    def _edge_similarity_smooth(edge_raw: torch.Tensor, edge_to_edge_index: torch.Tensor, gamma: float) -> torch.Tensor:
        if edge_to_edge_index is None or edge_to_edge_index.numel() == 0:
            return edge_raw
        s = edge_to_edge_index[0]
        t = edge_to_edge_index[1]
        diff = edge_raw[s] - edge_raw[t]
        d2 = (diff * diff).mean(dim=1)  # [M]
        w = torch.exp(-float(gamma) * d2).clamp(min=1e-8)  # [M]

        n_edges = edge_raw.size(0)
        w_sum = torch.zeros((n_edges,), device=edge_raw.device, dtype=edge_raw.dtype)
        w_sum.index_add_(0, s, w)

        out = torch.zeros_like(edge_raw)
        out.index_add_(0, s, edge_raw[t] * w.unsqueeze(1))
        out = out / w_sum.clamp(min=1e-8).unsqueeze(1)

        return out

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)

        edge_raw = dist_feat
        if self.edge_ee:
            edge_upd = self._edge_similarity_smooth(edge_raw, edge_to_edge_index, gamma=float(self.sim_gamma))
            a = float(self.denoise_alpha)
            edge_base = (1.0 - a) * edge_raw + a * edge_upd
        else:
            edge_base = edge_raw

        node_ctx = self.node_ctx_lin(torch.cat([x[srcs], x[dsts]], dim=1))
        edge_attr = edge_base + torch.tanh(self.node_ctx_scale) * node_ctx
        if self.edge_norm is not None:
            edge_attr = self.edge_norm(edge_attr)

        node_edge_attr = edge_attr if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_raw, edge_index, num_nodes=num_nodes)
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_attr, edge_index, num_nodes=num_nodes)

            if self.pool_gate_mode == "learned":
                gate_n = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate_n = torch.ones_like(node_att)
            else:
                gate_n = torch.zeros_like(node_att)

            node_out = node_att + gate_n * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_attr], dim=0)


class SpatialConvV13EdgeContextSimilarityDenoise(nn.Module):
    """
    V13 (context-aware similarity denoise):
    - Keeps v12's conservative similarity-weighted edge↔edge smoothing.
    - Uses node-pair context as a *key* to compute similarity weights (not as a large free-form edge rewrite):
        edge_key = edge_raw + tanh(k)*Wn([x_src,x_dst])
      and smooth edge_raw using weights from edge_key differences.

    This aims to make edge↔edge mixing "only among semantically compatible edges"
    when the default edge↔edge neighborhood (e.g., incidence) is heterogeneous.
    Both context scales initialize to 0, so training starts close to v12 (stable).
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)

        alpha_env = os.getenv("SDCN_EDGE_DENOISE_ALPHA", "").strip()
        self.denoise_alpha = float(alpha_env) if alpha_env else 0.1
        gamma_env = os.getenv("SDCN_EDGE_SIM_GAMMA", "").strip()
        self.sim_gamma = float(gamma_env) if gamma_env else 1.0

        self.edge_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_DENOISE_NORM", True) else None

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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.node_ctx_lin = nn.Linear(hidden_size * 2, hidden_size)
        self.node_ctx_scale = nn.Parameter(torch.tensor(0.0))
        self.sim_key_ctx_scale = nn.Parameter(torch.tensor(0.0))

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

    @staticmethod
    def _edge_similarity_smooth_key(
        edge_value: torch.Tensor, edge_key: torch.Tensor, edge_to_edge_index: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        if edge_to_edge_index is None or edge_to_edge_index.numel() == 0:
            return edge_value
        s = edge_to_edge_index[0]
        t = edge_to_edge_index[1]
        diff = edge_key[s] - edge_key[t]
        d2 = (diff * diff).mean(dim=1)  # [M]
        w = torch.exp(-float(gamma) * d2).clamp(min=1e-8)  # [M]

        n_edges = edge_value.size(0)
        w_sum = torch.zeros((n_edges,), device=edge_value.device, dtype=edge_value.dtype)
        w_sum.index_add_(0, s, w)

        out = torch.zeros_like(edge_value)
        out.index_add_(0, s, edge_value[t] * w.unsqueeze(1))
        out = out / w_sum.clamp(min=1e-8).unsqueeze(1)

        return out

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)

        edge_raw = dist_feat

        node_ctx = self.node_ctx_lin(torch.cat([x[srcs], x[dsts]], dim=1))
        edge_key = edge_raw + torch.tanh(self.sim_key_ctx_scale) * node_ctx

        if self.edge_ee:
            edge_upd = self._edge_similarity_smooth_key(edge_raw, edge_key, edge_to_edge_index, gamma=float(self.sim_gamma))
            a = float(self.denoise_alpha)
            edge_base = (1.0 - a) * edge_raw + a * edge_upd
        else:
            edge_base = edge_raw

        edge_attr = edge_base + torch.tanh(self.node_ctx_scale) * node_ctx
        if self.edge_norm is not None:
            edge_attr = self.edge_norm(edge_attr)

        node_edge_attr = edge_attr if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_raw, edge_index, num_nodes=num_nodes)
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_attr, edge_index, num_nodes=num_nodes)

            if self.pool_gate_mode == "learned":
                gate_n = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate_n = torch.ones_like(node_att)
            else:
                gate_n = torch.zeros_like(node_att)

            node_out = node_att + gate_n * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_attr], dim=0)


class SpatialConvV14EdgePoolConcatFusion(nn.Module):
    """
    V14 (edge-pool concat fusion):
    - Same edge↔edge role as v13: similarity-weighted denoise with an optional context key.
    - Makes the "strong baseline path" explicit by *concatenating* pooled edge features into node features,
      instead of adding them as a gated residual:
        node_out = W([node_att, pool_raw, pool_upd])

    Motivation: on edge-dominant datasets, additive residual can be under-utilized or unstable;
    concat keeps the edge statistics as a separable subspace that downstream layers can reliably use.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)

        alpha_env = os.getenv("SDCN_EDGE_DENOISE_ALPHA", "").strip()
        self.denoise_alpha = float(alpha_env) if alpha_env else 0.1
        gamma_env = os.getenv("SDCN_EDGE_SIM_GAMMA", "").strip()
        self.sim_gamma = float(gamma_env) if gamma_env else 1.0

        self.edge_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_DENOISE_NORM", True) else None

        self.pool_raw = _env_flag("SDCN_POOL_RAW", True)
        self.pool_upd = _env_flag("SDCN_POOL_UPD", True)

        self.edge_dim_proj = None
        if self.edge_dim != hidden_size:
            self.edge_dim_proj = nn.Linear(self.edge_dim, hidden_size)
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.node_ctx_lin = nn.Linear(hidden_size * 2, hidden_size)
        self.node_ctx_scale = nn.Parameter(torch.tensor(0.0))
        self.sim_key_ctx_scale = nn.Parameter(torch.tensor(0.0))

        self.en_gat = SGATLayer(
            hidden_size,
            hidden_size,
            heads=heads,
            dropout=dropout,
            combine="mean",
            edge_dim=hidden_size,
            activation=None,
        )

        fuse_parts = 1 + int(self.pool_raw) + int(self.pool_upd)
        self.fuse_lin = nn.Linear(hidden_size * fuse_parts, hidden_size, bias=False)
        with torch.no_grad():
            self.fuse_lin.weight.zero_()
            # Start close to an additive fusion: node_att + pool_raw + pool_upd (when enabled).
            w = self.fuse_lin.weight
            w[:, :hidden_size].copy_(torch.eye(hidden_size, device=w.device))
            offset = hidden_size
            if self.pool_raw:
                w[:, offset : offset + hidden_size].copy_(torch.eye(hidden_size, device=w.device))
                offset += hidden_size
            if self.pool_upd:
                w[:, offset : offset + hidden_size].copy_(torch.eye(hidden_size, device=w.device))

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

    @staticmethod
    def _edge_similarity_smooth_key(
        edge_value: torch.Tensor, edge_key: torch.Tensor, edge_to_edge_index: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        if edge_to_edge_index is None or edge_to_edge_index.numel() == 0:
            return edge_value
        s = edge_to_edge_index[0]
        t = edge_to_edge_index[1]
        diff = edge_key[s] - edge_key[t]
        d2 = (diff * diff).mean(dim=1)
        w = torch.exp(-float(gamma) * d2).clamp(min=1e-8)

        n_edges = edge_value.size(0)
        w_sum = torch.zeros((n_edges,), device=edge_value.device, dtype=edge_value.dtype)
        w_sum.index_add_(0, s, w)

        out = torch.zeros_like(edge_value)
        out.index_add_(0, s, edge_value[t] * w.unsqueeze(1))
        out = out / w_sum.clamp(min=1e-8).unsqueeze(1)

        return out

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)

        edge_raw = dist_feat
        node_ctx = self.node_ctx_lin(torch.cat([x[srcs], x[dsts]], dim=1))
        edge_key = edge_raw + torch.tanh(self.sim_key_ctx_scale) * node_ctx

        if self.edge_ee:
            edge_upd = self._edge_similarity_smooth_key(edge_raw, edge_key, edge_to_edge_index, gamma=float(self.sim_gamma))
            a = float(self.denoise_alpha)
            edge_base = (1.0 - a) * edge_raw + a * edge_upd
        else:
            edge_base = edge_raw

        edge_attr = edge_base + torch.tanh(self.node_ctx_scale) * node_ctx
        if self.edge_norm is not None:
            edge_attr = self.edge_norm(edge_attr)

        node_edge_attr = edge_attr if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

        parts = [node_att]
        if self.pool_raw:
            parts.append(self._pool_edges_to_nodes_mean(edge_raw, edge_index, num_nodes=num_nodes))
        if self.pool_upd:
            parts.append(self._pool_edges_to_nodes_mean(edge_attr, edge_index, num_nodes=num_nodes))

        node_out = self.fuse_lin(torch.cat(parts, dim=1))
        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_attr], dim=0)


class SpatialConvV15EdgeEeAuxContextSimilarityDenoise(nn.Module):
    """
    V15 (context-aware similarity denoise + edge-level auxiliary head):
    - Uses v13's conservative, context-aware similarity smoothing on edge features.
    - Adds an edge-level auxiliary head (like v6) that predicts within-cluster probability per edge.

    Motivation:
    - In many datasets, plain edge↔edge GAT either does not help or even hurts because the edge↔edge neighborhood
      is heterogeneous and the training objective does not directly supervise edge representations.
    - v15 makes edge↔edge updates more conservative (similarity smoothing) and adds an explicit objective to
      align edge↔edge modeling with clustering (enabled via SDCN_EDGE_AUX_WEIGHT / SDCN_EDGE_AUX_SMOOTH_WEIGHT).

    Exposes (per forward call):
      - self._last_edge_within_logit: [E] logits (float) for the training loop to consume.
    """

    def __init__(self, hidden_size, edge_dim=None, dropout=0.2, heads=4, activation=F.relu, out_activation=_UNSET):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim if edge_dim is not None else hidden_size
        self.out_activation = activation if out_activation is _UNSET else out_activation

        self.edge_ee = _env_flag("SDCN_EDGE_EE", True)
        self.node_att_edge = _env_flag("SDCN_NODE_ATT_EDGE", True)

        alpha_env = os.getenv("SDCN_EDGE_DENOISE_ALPHA", "").strip()
        self.denoise_alpha = float(alpha_env) if alpha_env else 0.1
        gamma_env = os.getenv("SDCN_EDGE_SIM_GAMMA", "").strip()
        self.sim_gamma = float(gamma_env) if gamma_env else 1.0

        self.edge_norm = nn.LayerNorm(hidden_size) if _env_flag("SDCN_EDGE_DENOISE_NORM", True) else None

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
            with torch.no_grad():
                self.edge_dim_proj.weight.zero_()
                self.edge_dim_proj.bias.zero_()
                m = min(int(self.edge_dim), int(hidden_size))
                self.edge_dim_proj.weight[:m, :m].copy_(torch.eye(m, device=self.edge_dim_proj.weight.device))

        self.node_ctx_lin = nn.Linear(hidden_size * 2, hidden_size)
        self.node_ctx_scale = nn.Parameter(torch.tensor(0.0))
        self.sim_key_ctx_scale = nn.Parameter(torch.tensor(0.0))

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

        # Edge-level auxiliary head: predict within-edge probability (logit).
        self.edge_within_lin = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.edge_within_lin.bias)
        self._last_edge_within_logit: torch.Tensor | None = None

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

    @staticmethod
    def _edge_similarity_smooth_key(
        edge_value: torch.Tensor, edge_key: torch.Tensor, edge_to_edge_index: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        if edge_to_edge_index is None or edge_to_edge_index.numel() == 0:
            return edge_value
        s = edge_to_edge_index[0]
        t = edge_to_edge_index[1]
        diff = edge_key[s] - edge_key[t]
        d2 = (diff * diff).mean(dim=1)  # [M]
        w = torch.exp(-float(gamma) * d2).clamp(min=1e-8)  # [M]

        n_edges = edge_value.size(0)
        w_sum = torch.zeros((n_edges,), device=edge_value.device, dtype=edge_value.dtype)
        w_sum.index_add_(0, s, w)

        out = torch.zeros_like(edge_value)
        out.index_add_(0, s, edge_value[t] * w.unsqueeze(1))
        out = out / w_sum.clamp(min=1e-8).unsqueeze(1)

        return out

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        dist_feat = data.dist_feat
        edge_to_edge_index = data.edge_to_edge_index

        num_nodes = x.size(0)
        srcs, dsts = edge_index[0], edge_index[1]

        if self.edge_dim_proj is not None and dist_feat.size(1) != self.hidden_size:
            dist_feat = self.edge_dim_proj(dist_feat)

        edge_raw = dist_feat
        node_ctx = self.node_ctx_lin(torch.cat([x[srcs], x[dsts]], dim=1))
        edge_key = edge_raw + torch.tanh(self.sim_key_ctx_scale) * node_ctx

        if self.edge_ee:
            edge_upd = self._edge_similarity_smooth_key(edge_raw, edge_key, edge_to_edge_index, gamma=float(self.sim_gamma))
            a = float(self.denoise_alpha)
            edge_base = (1.0 - a) * edge_raw + a * edge_upd
        else:
            edge_base = edge_raw

        edge_attr = edge_base + torch.tanh(self.node_ctx_scale) * node_ctx
        if self.edge_norm is not None:
            edge_attr = self.edge_norm(edge_attr)

        # Store edge auxiliary logits for the training loop to consume.
        self._last_edge_within_logit = self.edge_within_lin(edge_base).squeeze(-1)

        node_edge_attr = edge_attr if self.node_att_edge else None
        node_att = self.en_gat(x, edge_index, node_edge_attr)

        node_out = node_att
        if self.pool_residual and (self.pool_raw or self.pool_upd):
            pooled = 0.0
            if self.pool_raw:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_raw, edge_index, num_nodes=num_nodes)
            if self.pool_upd:
                pooled = pooled + self._pool_edges_to_nodes_mean(edge_attr, edge_index, num_nodes=num_nodes)

            if self.pool_gate_mode == "learned":
                gate_n = torch.sigmoid(self.pool_gate(torch.cat([node_att, pooled], dim=1)))
            elif self.pool_gate_mode == "one":
                gate_n = torch.ones_like(node_att)
            else:
                gate_n = torch.zeros_like(node_att)

            node_out = node_att + gate_n * self.pool_proj(pooled)

        if self.out_activation is not None:
            node_out = self.out_activation(node_out)

        return torch.cat([node_out, edge_attr], dim=0)


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
elif SPATIALCONV_VARIANT_SELECTED in {
    "v6",
    "v6edge_ee_aux",
    "v6_edge_ee_aux",
    "edge_ee_aux",
    "ee_aux",
    "v6ee",
}:
    SpatialConv = SpatialConvV6EdgeEeAux
elif SPATIALCONV_VARIANT_SELECTED in {
    "v7",
    "v7edge_attr_fusion",
    "v7_edge_attr_fusion",
    "edge_attr_fusion",
    "edge_fusion",
    "fuse",
}:
    SpatialConv = SpatialConvV7EdgeAttrFusion
elif SPATIALCONV_VARIANT_SELECTED in {
    "v8",
    "v8edge_denoise_attr",
    "v8_edge_denoise_attr",
    "edge_denoise_attr",
    "denoise",
}:
    SpatialConv = SpatialConvV8EdgeDenoiseAttr
elif SPATIALCONV_VARIANT_SELECTED in {
    "v9",
    "v9edge_context_denoise",
    "v9_edge_context_denoise",
    "edge_context_denoise",
    "context_denoise",
    "ctx_denoise",
}:
    SpatialConv = SpatialConvV9EdgeContextDenoise
elif SPATIALCONV_VARIANT_SELECTED in {
    "v10",
    "v10edge_base_denoise_plus_context",
    "v10_edge_base_denoise_plus_context",
    "edge_base_denoise_plus_context",
    "base_denoise_ctx",
}:
    SpatialConv = SpatialConvV10EdgeBaseDenoisePlusContext
elif SPATIALCONV_VARIANT_SELECTED in {
    "v11",
    "v11edge_adaptive_denoise_context",
    "v11_edge_adaptive_denoise_context",
    "edge_adaptive_denoise_context",
    "adaptive_denoise",
}:
    SpatialConv = SpatialConvV11EdgeAdaptiveDenoiseContext
elif SPATIALCONV_VARIANT_SELECTED in {
    "v12",
    "v12edge_similarity_denoise",
    "v12_edge_similarity_denoise",
    "edge_similarity_denoise",
    "sim_denoise",
}:
    SpatialConv = SpatialConvV12EdgeSimilarityDenoise
elif SPATIALCONV_VARIANT_SELECTED in {
    "v13",
    "v13edge_context_similarity_denoise",
    "v13_edge_context_similarity_denoise",
    "edge_context_similarity_denoise",
    "ctx_sim_denoise",
    "context_sim_denoise",
}:
    SpatialConv = SpatialConvV13EdgeContextSimilarityDenoise
elif SPATIALCONV_VARIANT_SELECTED in {
    "v15",
    "v15edge_ee_aux_context_similarity_denoise",
    "v15_edge_ee_aux_context_similarity_denoise",
    "edge_ee_aux_ctx_sim_denoise",
    "ee_aux_ctx_sim_denoise",
}:
    SpatialConv = SpatialConvV15EdgeEeAuxContextSimilarityDenoise
elif SPATIALCONV_VARIANT_SELECTED in {
    "v14",
    "v14edge_pool_concat_fusion",
    "v14_edge_pool_concat_fusion",
    "edge_pool_concat_fusion",
    "pool_concat_fusion",
    "concat_fusion",
}:
    SpatialConv = SpatialConvV14EdgePoolConcatFusion
else:
    raise ValueError(
        f"Unknown SPATIALCONV_VARIANT={SPATIALCONV_VARIANT_SELECTED!r}. "
        f"Use one of: v1original, v2edge_single_layer, v3edge_cross_layers, v4edge_pool_fusion, "
        f"v5edge_pool_residual, v6edge_ee_aux, v7edge_attr_fusion, v8edge_denoise_attr, "
        f"v9edge_context_denoise, v10edge_base_denoise_plus_context, v11edge_adaptive_denoise_context, "
        f"v12edge_similarity_denoise, v13edge_context_similarity_denoise, "
        f"v14edge_pool_concat_fusion, v15edge_ee_aux_context_similarity_denoise."
    )
