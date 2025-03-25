# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file implement some layers for S-MAN using PyTorch Geometric.
Converted from the original PGL implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
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
        if self.training and hasattr(F, 'dropout'):
            x = F.dropout(x, p=0.2)
            
        # Apply GATConv
        x = self.gat_conv(x, edge_index)
        
        # Add bias and apply activation
        x = x + self.bias
        
        if self.activation is not None:
            x = self.activation(x)
            
        return x


class CustomGATConv(MessagePassing):
    """Custom GAT Convolution with edge features support
    
    This is a custom implementation of GAT that supports edge features in the attention mechanism.
    It's used when we need to incorporate edge attributes (like distance features) into the attention calculation.
    
    Args:
        in_channels (int): Size of input features
        out_channels (int): Size of output features
        heads (int): Number of attention heads
        dropout (float): Dropout probability
        negative_slope (float): LeakyReLU negative slope
        edge_dim (int, optional): Edge feature dimension
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2, 
                 negative_slope=0.2, edge_dim=None):
        super(CustomGATConv, self).__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim
        
        # Linear transformation for node features
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters for source and target nodes
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        
        # Edge feature projection if edge_dim is provided
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.empty(1, heads, out_channels))
        
        # Bias parameter
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if hasattr(self, 'lin_edge'):
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
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
        # Apply feature dropout during training
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout)
            if edge_attr is not None:
                edge_attr = F.dropout(edge_attr, p=self.dropout)
        
        # Transform node features
        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)
        
        # Process edge features if available
        edge_embedding = None
        if edge_attr is not None and self.edge_dim is not None:
            edge_embedding = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        
        # Start propagating messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        
        # Combine heads (mean aggregation)
        out = out.mean(dim=1)
        
        # Add bias
        out = out + self.bias
        
        return out
        
    def message(self, x_i, x_j, edge_attr, index, size_i):
        """Compute messages and attention weights
        
        Args:
            x_i (Tensor): Features of target nodes
            x_j (Tensor): Features of source nodes
            edge_attr (Tensor, optional): Edge features
            index (Tensor): Target node indices
            size_i (int): Size of target nodes
            
        Returns:
            Tensor: Messages with attention weights applied
        """
        # Compute attention coefficients
        alpha = (x_i * self.att_src).sum(-1) + (x_j * self.att_dst).sum(-1)
        
        # Add edge feature attention if available
        if edge_attr is not None:
            alpha = alpha + (edge_attr * self.att_edge).sum(-1)
        
        # Apply LeakyReLU and normalize
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, size_i)
        
        # Apply dropout to attention weights during training
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout)
        
        # Apply attention weights to source node features
        return x_j * alpha.unsqueeze(-1)


class SGATLayer(nn.Module):
    """Spatial Graph Attention Network Layer
    
    This layer aggregates the edge-neighbors of node to update the node embedding.
    It replaces the original sgat function.
    
    Args:
        in_channels (int): Size of input features
        out_channels (int): Size of output features
        heads (int): Number of attention heads
        dropout (float): Dropout probability
        negative_slope (float): LeakyReLU negative slope
        combine (str): Method to combine multi-head results ('mean', 'max', or 'dense')
        activation (callable): Activation function
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2, 
                 negative_slope=0.2, combine='mean', activation=F.relu):
        super(SGATLayer, self).__init__()
        
        self.heads = heads
        self.combine = combine
        self.activation = activation
        
        # Use custom GAT implementation to support edge features
        self.gat_conv = CustomGATConv(
            in_channels, 
            out_channels, 
            heads=heads,
            dropout=dropout,
            negative_slope=negative_slope,
            edge_dim=in_channels  # Assuming edge features have same dimension
        )
        
        # For 'dense' combine method, add an extra linear layer
        if combine == 'dense':
            self.dense_layer = nn.Linear(out_channels * heads, out_channels)
            
        # Create bias parameter
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (Tensor): Graph connectivity [2, num_edges]
            edge_attr (Tensor, optional): Edge features [num_edges, edge_dim]
            
        Returns:
            Tensor: Updated node features [num_nodes, out_channels]
        """
        # Apply feature dropout during training
        if self.training:
            x = F.dropout(x, p=0.2)
            if edge_attr is not None:
                edge_attr = F.dropout(edge_attr, p=0.2)
        
        # Apply GAT convolution
        out = self.gat_conv(x, edge_index, edge_attr)
        
        # Combine multi-head results based on the specified method
        if self.combine == 'mean':
            # Mean aggregation is already done in CustomGATConv
            pass
        elif self.combine == 'max':
            # Reshape to get heads dimension and apply max pooling
            out = out.view(-1, self.heads, out.shape[1] // self.heads)
            out = out.max(dim=1)[0]
        elif self.combine == 'dense':
            # Apply dense layer to combine heads
            out = self.dense_layer(out)
        
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