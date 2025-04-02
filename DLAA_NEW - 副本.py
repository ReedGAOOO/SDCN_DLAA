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
    
    在注意力计算中显式考虑边特征。
    只要在构造时指定 edge_dim=边特征的维度，即可使用 PyG 内置的 GATConv。
    
    Args:
        in_channels (int): 输入特征维度
        out_channels (int): 输出特征维度
        heads (int): 注意力头数
        dropout (float): Dropout 概率
        negative_slope (float): LeakyReLU 的负斜率
        combine (str): 多头结果融合方式：'mean', 'max' or 'dense'
        activation (callable): 激活函数
        edge_dim (int): 边特征维度（必填，用于注意力时）
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
        
        # 直接用 PyG 的 GATConv 并指定 edge_dim
        # concat=True 表示输出形状为 [num_nodes, heads * out_channels]
        # 后续我们再手动对多头做 mean/max/dense
        self.gat_conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            negative_slope=negative_slope,
            edge_dim=edge_dim,   # 关键：指定边特征维度
            concat=True
        )
        
        # 如果 combine == 'dense'，再加一层线性把 heads*out_channels -> out_channels
        if self.combine == 'dense':
            self.dense_combine = nn.Linear(heads * out_channels, out_channels, bias=False)
        
        # 和之前一样，保留一个 bias
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x (Tensor): 节点特征 [N, in_channels]
            edge_index (Tensor): [2, E]
            edge_attr (Tensor, optional): [E, edge_dim]
        Returns:
            (Tensor): 更新后的节点特征，形状根据 combine 而定:
                      - mean/max -> [N, out_channels]
                      - dense    -> [N, out_channels]
                      - default  -> [N, heads*out_channels]
        """
        if self.training:
            x = F.dropout(x, p=self.dropout)
            if edge_attr is not None:
                edge_attr = F.dropout(edge_attr, p=self.dropout)
        
        # 调用 GATConv，传入边特征
        out = self.gat_conv(x, edge_index, edge_attr)
        # out 形状: [N, heads*out_channels]
        
        if self.combine in ['mean', 'max']:
            # 把多头维度分离出来: [N, heads, out_channels]
            out = out.view(-1, self.heads, self.out_channels)
            if self.combine == 'mean':
                out = out.mean(dim=1)  # [N, out_channels]
            else:  # 'max'
                out, _ = out.max(dim=1)  # [N, out_channels]
        elif self.combine == 'dense':
            # 先 flatten 多头维度
            out = self.dense_combine(out)  # [N, out_channels]
        else:
            # 默认什么都不做, 保持 [N, heads*out_channels]
            pass
        
        # 加上 bias 并激活
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