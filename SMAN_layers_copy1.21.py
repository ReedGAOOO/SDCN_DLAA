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
This file implement some layers for S-MAN.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import numpy as np


class GraphPooling(nn.Module):
    """PyTorch Geometric implementation of graph pooling layer"""
    def __init__(self, pool_type='sum'):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, node_feat, batch):
        """
        Args:
            node_feat: Tensor of shape (num_nodes, feature_size)
            batch: LongTensor mapping nodes to their graph in batch
        Returns:
            Pooled graph features tensor of shape (num_graphs, feature_size)
        """
        return scatter(node_feat, batch, dim=0, reduce=self.pool_type)

    def extra_repr(self):
        return f'pool_type={self.pool_type}'


class SpatialEmbedding(nn.Module):
    """PyTorch implementation of spatial embedding layer"""
    def __init__(self, dist_dim, embed_size):
        super().__init__()
        self.dist_w = nn.Parameter(torch.Tensor(dist_dim, embed_size))
        nn.init.xavier_uniform_(self.dist_w)

    def forward(self, dist_feat, dist_feat_order=None):
        """
        Args:
            dist_feat: Tensor of shape (num_edges, dist_dim)
            dist_feat_order: Tensor of shape (num_edges, dist_dim) or None
        Returns:
            Tuple of embedded distance features
        """
        dist_feat = torch.matmul(dist_feat, self.dist_w)
        if dist_feat_order is not None:
            dist_feat_order = torch.matmul(dist_feat_order, self.dist_w)
        return dist_feat, dist_feat_order


def aggregate_edges_from_nodes(node_edge_feat, dist_feat, srcs, dsts):
    """
    ** Node-to-Edge Aggregation Layer **
    This function can aggregate the two node features and spatial features to update the edge embedding.
    Args:
        node_edge_feat(Variable): A tensor with shape (num_nodes + num_edges, feature_size).
        dist_feat(Variable): The ispatial distance feature for the edges of node-node graph, the shape is (num_edges, embedding_size).
        srcs(Variable): Source indices of edges with shape (num_edges, 1) to gather source features.
        dsts(Variable): Target indices of edges with shape (num_edges, 1) to gather target features.
    Returns:
        Variable: The updated edge features after aggregating embeddings of nodes.
    """
    hidden_size = node_edge_feat.shape[-1]
    src_feat = L.gather(node_edge_feat, srcs)
    dst_feat = L.gather(node_edge_feat, dsts)
    feat_h = L.concat([src_feat, dst_feat, dist_feat], axis=-1)
    feat_h = L.fc(input=feat_h, size=hidden_size, act="relu")
    return feat_h


def aggregate_edges_from_edges(gw, node_feat, hidden_size, name):
    """The gat function can aggregate the edge-neighbors of edge to update the edfe embedding."""
    node_edge_feat = gat(gw,
                        node_feat,
                        hidden_size,
                        dist_feat=None,
                        activation="relu",
                        name=name,
                        num_heads=4,
                        feat_drop=0.2,
                        attn_drop=0.2,
                        is_test=False)
    return node_edge_feat


def aggregate_nodes_from_edges(gw, node_feat, edge_feat, hidden_size, name):
    """The sgat function can aggregate the edge-neighbors of node to update the node embedding."""
    node_edge_feat = sgat(gw,
                        node_feat,
                        edge_feat,
                        hidden_size,
                        name=name,
                        activation='relu',
                        num_heads=4,
                        feat_drop=0.2,
                        attn_drop=0.2,
                        is_test=False)
    return node_edge_feat


class NodeEdgeConcat(nn.Module):
    """PyTorch implementation of node-edge feature concatenation"""
    def __init__(self):
        super().__init__()

    def forward(self, node_feat, edge_feat, node_batch, edge_batch):
        """
        Args:
            node_feat: Tensor of shape (num_nodes, feature_size)
            edge_feat: Tensor of shape (num_edges, feature_size)
            node_batch: LongTensor mapping nodes to their graph
            edge_batch: LongTensor mapping edges to their graph
        Returns:
            Concatenated tensor of shape (num_nodes + num_edges, feature_size)
        """
        # Validate batch indices match
        assert torch.all(node_batch[1:] >= node_batch[:-1]), "Node batch indices must be sorted"
        assert torch.all(edge_batch[1:] >= edge_batch[:-1]), "Edge batch indices must be sorted"

        return torch.cat([node_feat, edge_feat], dim=0)

    @torch.jit.script
    def test_method():
        node_feat = torch.randn(4, 64)
        edge_feat = torch.randn(6, 64)
        return torch.cat([node_feat, edge_feat], dim=0).shape


class SpatialConv(nn.Module):
    """PyG implementation of spatial graph convolution layer"""
    def __init__(self, hidden_size, edge_dim):
        super().__init__()
        self.edge_aggregator = EdgeAggregator(hidden_size)
        self.edge_conv = EdgeEnhancedGAT(hidden_size, hidden_size, edge_dim)
        self.node_conv = SpatialGATConv(hidden_size, hidden_size, edge_dim)
        self.concat = NodeEdgeConcat()

    def forward(self, node_feat, edge_feat, edge_index, dist_feat):
        """
        Args:
            node_feat: Tensor of shape (num_nodes, hidden_size)
            edge_feat: Tensor of shape (num_edges, hidden_size)
            edge_index: LongTensor of shape (2, num_edges)
            dist_feat: Tensor of shape (num_edges, edge_dim)
        Returns:
            Updated node-edge features tensor
        """
        # Step 1: Update edge features
        edge_feat = self.edge_aggregator(
            self.concat(node_feat, edge_feat, None, None),
            dist_feat,
            edge_index
        )

        # Step 2: Edge-enhanced node update
        node_feat = self.node_conv(
            node_feat,
            edge_index,
            self.edge_conv(node_feat, edge_index, edge_feat)
        )

        return self.concat(node_feat, edge_feat, None, None)


def sgat(gw,
        node_feat,
        edge_feat,
        hidden_size,
        name,
        activation='relu',
        combine='mean',
        num_heads=4,
        feat_drop=0.2,
        attn_drop=0.2,
        is_test=False):
    """
    The sgat function can aggregate the edge-neighbors of node to update the node embedding.
    Adapted from https://github.com/PaddlePaddle/PGL/blob/main/pgl/layers/conv.py.
    Args:
        gw(GraphWrapper): A graph wrapper for edge-node graph.
        node_feat(Variable): A tensor of node-edge features with shape (num_nodes + num_nodes, feature_size).
        edge_feat(Variable): A tensor of spatial distance features with shape (num_edges, feature_size).
        combine(str): The choice of combining multi-head embeddings. It can be mean, max or dense.

        hidden_size: The hidden size for gat.
        activation: The activation for the output.
        name: Gat layer names.
        num_heads: The head number in gat.
        feat_drop: Dropout rate for feature.
        attn_drop: Dropout rate for attention.
        is_test: Whether in test phrase.
    Returns:
        Variable: The updated node-edge feature matrix with shape (num_nodes + num_edges, feature_size).
    """

    def send_attention(src_feat, dst_feat, edge_feat):
        output = src_feat["left_a"] + dst_feat["right_a"]
        if 'edge_a' in edge_feat:
            output += edge_feat["edge_a"]
        output = L.leaky_relu(
            output, alpha=0.2)  # (num_edges, num_heads)
        return {"alpha": output, "h": src_feat["h"]}

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["h"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h
        h = L.reshape(h, [-1, num_heads, hidden_size])
        alpha = L.reshape(alpha, [-1, num_heads, 1])
        if attn_drop > 1e-15:
            alpha = L.dropout(
                alpha,
                dropout_prob=attn_drop,
                is_test=is_test,
                dropout_implementation="upscale_in_train")
        h = h * alpha
        h = L.reshape(h, [-1, num_heads * hidden_size])
        h = L.lod_reset(h, old_h)
        return L.sequence_pool(h, "sum")

    if feat_drop > 1e-15:
        node_feat = L.dropout(
                node_feat,
                dropout_prob=feat_drop,
                is_test=is_test,
                dropout_implementation='upscale_in_train')
        edge_feat = L.dropout(
                edge_feat,
                dropout_prob=feat_drop,
                is_test=is_test,
                dropout_implementation='upscale_in_train') 

    ft = L.fc(node_feat,
                         hidden_size * num_heads,
                         bias_attr=False,
                         param_attr=fluid.ParamAttr(name=name + '_weight'))
    left_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_l_A')
    right_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_r_A')
    reshape_ft = L.reshape(ft, [-1, num_heads, hidden_size])
    left_a_value = L.reduce_sum(reshape_ft * left_a, -1)
    right_a_value = L.reduce_sum(reshape_ft * right_a, -1)

    fd = L.fc(edge_feat,
            size=hidden_size * num_heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_fc_eW'))
    edge_a = L.create_parameter(
            shape=[num_heads, hidden_size],
            dtype='float32',
            name=name + '_gat_d_A')
    fd = L.reshape(fd, [-1, num_heads, hidden_size])
    edge_a_value = L.reduce_sum(fd * edge_a, -1)
    efeat_list = [('edge_a', edge_a_value)]
        
    msg = gw.send(
        send_attention,
        nfeat_list=[("h", ft), ("left_a", left_a_value),
                    ("right_a", right_a_value)], efeat_list=efeat_list)
    output = gw.recv(msg, reduce_attention)
    
    if combine == 'mean':
        output = L.reshape(output, [-1, num_heads, hidden_size])
        output = L.reduce_mean(output, dim=1)
        num_heads = 1
    if combine == 'max':
        output = L.reshape(output, [-1, num_heads, hidden_size])
        output = L.reduce_max(output, dim=1)
        num_heads = 1
    if combine == 'dense':
        output = L.fc(output, hidden_size, bias_attr=False, param_attr=fluid.ParamAttr(name=name + '_dense_combine'))
        num_heads = 1

    bias = L.create_parameter(
        shape=[hidden_size * num_heads],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    bias.stop_gradient = True
    output = L.elementwise_add(output, bias, act=activation)
    return output


def gat(gw,
        feature,
        hidden_size,
        activation,
        name,
        dist_feat=None,
        num_heads=4,
        feat_drop=0.2,
        attn_drop=0.2,
        is_test=False):
    """Implementation of graph attention networks (GAT)
    Adapted from https://github.com/PaddlePaddle/PGL/blob/main/pgl/layers/conv.py.
    """

    def send_attention(src_feat, dst_feat, edge_feat):
        output = src_feat["left_a"] + dst_feat["right_a"]
        if 'dist_a' in edge_feat:
            output += edge_feat["dist_a"]
        output = L.leaky_relu(
            output, alpha=0.2)  # (num_edges, num_heads)
        return {"alpha": output, "h": src_feat["h"]}

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["h"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h
        h = L.reshape(h, [-1, num_heads, hidden_size])
        alpha = L.reshape(alpha, [-1, num_heads, 1])
        if attn_drop > 1e-15:
            alpha = L.dropout(
                alpha,
                dropout_prob=attn_drop,
                is_test=is_test,
                dropout_implementation="upscale_in_train")
        h = h * alpha
        h = L.reshape(h, [-1, num_heads * hidden_size])
        h = L.lod_reset(h, old_h)
        return L.sequence_pool(h, "sum")

    if feat_drop > 1e-15:
        feature = L.dropout(
            feature,
            dropout_prob=feat_drop,
            is_test=is_test,
            dropout_implementation='upscale_in_train')
        if dist_feat:
           dist_feat = L.dropout(
                       dist_feat,
                       dropout_prob=feat_drop,
                       is_test=is_test,
                       dropout_implementation='upscale_in_train') 

    ft = L.fc(feature,
                         hidden_size * num_heads,
                         bias_attr=False,
                         param_attr=fluid.ParamAttr(name=name + '_weight'))
    left_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_l_A')
    right_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_r_A')
    reshape_ft = L.reshape(ft, [-1, num_heads, hidden_size])
    left_a_value = L.reduce_sum(reshape_ft * left_a, -1)
    right_a_value = L.reduce_sum(reshape_ft * right_a, -1)
    efeat_list = []

    if dist_feat:
        fd = L.fc(dist_feat,
                  size=hidden_size * num_heads,
                  bias_attr=False,
                  param_attr=fluid.ParamAttr(name=name + '_fc_eW'))
        dist_a = L.create_parameter(
            shape=[num_heads, hidden_size],
            dtype='float32',
            name=name + '_gat_d_A')
        fd = L.reshape(fd, [-1, num_heads, hidden_size])
        dist_a_value = L.reduce_sum(fd * dist_a, -1)
        efeat_list = [('dist_a', dist_a_value)]
        
    msg = gw.send(
        send_attention,
        nfeat_list=[("h", ft), ("left_a", left_a_value),
                    ("right_a", right_a_value)], efeat_list=efeat_list)
    output = gw.recv(msg, reduce_attention)

    
    output = L.reshape(output, [-1, num_heads, hidden_size])
    output = L.reduce_mean(output, dim=1)
    num_heads = 1

    bias = L.create_parameter(
        shape=[hidden_size * num_heads],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    bias.stop_gradient = True
    output = L.elementwise_add(output, bias, act=activation)
    return output

def test_forward_pass():
    """Test function for verifying converted layer behavior"""
    # Create sample data
    num_nodes = 10
    num_edges = 30
    feat_dim = 64
    edge_dim = 32
    heads = 4

    # Initialize layers
    gat_conv = SpatialGATConv(feat_dim, feat_dim, edge_dim, heads=heads)
    sgat_conv = EdgeEnhancedGAT(feat_dim, feat_dim, edge_dim, heads=heads)
    hetero_conv = HeteroSpatialConv(feat_dim, edge_dim, num_types=3, heads=heads)

    # Create dummy data
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)
    node_types = torch.randint(0, 3, (num_nodes,))

    # Test forward passes
    assert gat_conv(x, edge_index, edge_attr).shape == (num_nodes, feat_dim)
    assert sgat_conv(x, edge_index, edge_attr).shape == (num_nodes, feat_dim)
    assert hetero_conv(x, edge_index, edge_attr, node_types).shape == (num_nodes, feat_dim)

    print("All forward passes completed successfully!")
