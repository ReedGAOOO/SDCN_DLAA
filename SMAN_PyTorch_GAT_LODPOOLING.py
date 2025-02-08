import torch
import torch.nn as nn
import torch.nn.functional as F

def graph_pooling(node_feat, graph_lod, pool_type='sum'):
    """
    对节点特征进行图级别的池化操作。

    Args:
        node_feat (Tensor): [num_nodes, feat_size]
        graph_lod (List[int]): 每个图包含的节点数量
        pool_type (str): 'sum' 或 'mean'

    Returns:
        Tensor: [num_graphs, feat_size]
    """
    graph_feat_list = []
    start_idx = 0
    for num_nodes in graph_lod:
        end_idx = start_idx + num_nodes
        nodes = node_feat[start_idx:end_idx]
        if pool_type == 'sum':
            graph_feat = torch.sum(nodes, dim=0)
        elif pool_type == 'mean':
            graph_feat = torch.mean(nodes, dim=0)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
        graph_feat_list.append(graph_feat)
        start_idx = end_idx
    graph_feat = torch.stack(graph_feat_list)
    return graph_feat

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
            dist_feat_order (Tensor, optional): [num_order_edges, dist_dim]
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
    从节点特征聚合更新边特征的层
    """
    def __init__(self, hidden_size):
        super(AggregateEdgesFromNodes, self).__init__()
        # 输入特征大小为 2 * hidden_size + dist_dim
        # 假设 dist_feat 的维度与 hidden_size 相同
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

class GATLayer(nn.Module):
    """
    手动实现的图注意力层，用于从边的邻居聚合更新边特征
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
        nn.init.xavier_unifor
        m_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

    def forward(self, graph, h):
        """
        Args:
            graph (GraphWrapper): 图结构，包含 edge_index
            h (Tensor): 节点特征，[num_nodes + num_edges, in_features]
        Returns:
            Tensor: 更新后的边特征
        """
        Wh = torch.matmul(h, self.W)  # [N, num_heads * out_features]
        N = Wh.size(0)
        Wh = Wh.view(N, self.num_heads, self.out_features)  # [N, num_heads, out_features]

        edge_index = graph.edge_index  # [2, E]
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        Wh_src = Wh[src]  # [E, num_heads, out_features]
        Wh_dst = Wh[dst]  # [E, num_heads, out_features]

        # 计算注意力分数
        e = self.leakyrelu(
            (Wh_src * self.a_src).sum(dim=-1) + (Wh_dst * self.a_dst).sum(dim=-1)
        )  # [E, num_heads]

        # softmax 归一化
        attention = F.softmax(e, dim=1)  # [E, num_heads]
        attention = self.dropout(attention)

        # 加权特征聚合
        h_prime = torch.zeros_like(Wh)
        for head in range(self.num_heads):
            h_prime[:, head, :] = torch.zeros_like(Wh[:, head, :])
            h_prime[:, head, :].index_add_(
                0, dst, attention[:, head].unsqueeze(-1) * Wh_src[:, head, :]
            )

        if self.activation == 'relu':
            h_prime = F.relu(h_prime)
        elif self.activation == 'elu':
            h_prime = F.elu(h_prime)
        # 其他激活函数可以根据需要添加

        h_prime = h_prime.view(N, self.num_heads * self.out_features)  # [N, num_heads * out_features]
        return h_prime

class SGATLayer(nn.Module):
    """
    手动实现的空间图注意力层，用于从边的邻居聚合更新节点特征
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

    def forward(self, graph, h, edge_feat):
        """
        Args:
            graph (GraphWrapper): 图结构，包含 edge_index
            h (Tensor): 节点特征，[num_nodes + num_edges, in_features]
            edge_feat (Tensor): 边特征，[num_edges, in_features]
        Returns:
            Tensor: 更新后的节点特征
        """
        Wh = torch.matmul(h, self.W)  # [N, num_heads * out_features]
        We = torch.matmul(edge_feat, self.W_e)  # [E, num_heads * out_features]
        N = Wh.size(0)
        Wh = Wh.view(N, self.num_heads, self.out_features)  # [N, num_heads, out_features]
        We = We.view(-1, self.num_heads, self.out_features)  # [E, num_heads, out_features]

        edge_index = graph.edge_index  # [2, E]
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        Wh_src = Wh[src]  # [E, num_heads, out_features]
        Wh_dst = Wh[dst]  # [E, num_heads, out_features]

        # 计算注意力分数，包括边特征
        e = self.leakyrelu(
            (Wh_src * self.a_src).sum(dim=-1) +
            (Wh_dst * self.a_dst).sum(dim=-1) +
            (We * self.a_edge).sum(dim=-1)
        )  # [E, num_heads]

        # softmax 归一化
        attention = F.softmax(e, dim=1)  # [E, num_heads]
        attention = self.dropout(attention)

        # 加权特征聚合
        h_prime = torch.zeros_like(Wh)
        for head in range(self.num_heads):
            h_prime[:, head, :] = torch.zeros_like(Wh[:, head, :])
            h_prime[:, head, :].index_add_(
                0, dst, attention[:, head].unsqueeze(-1) * Wh_src[:, head, :]
            )

        if self.activation == 'relu':
            h_prime = F.relu(h_prime)
        elif self.activation == 'elu':
            h_prime = F.elu(h_prime)
        # 其他激活函数可以根据需要添加

        h_prime = h_prime.view(N, self.num_heads * self.out_features)  # [N, num_heads * out_features]
        return h_prime

def concat_node_edge_feat(node_feat, edge_feat):
    """
    拼接节点特征和边特征，形成节点-边特征矩阵。

    Args:
        node_feat (Tensor): [num_nodes, feat_size]
        edge_feat (Tensor): [num_edges, feat_size]
    Returns:
        Tensor: [num_nodes + num_edges, feat_size]
    """
    node_edge_feat = torch.cat([node_feat, edge_feat], dim=0)
    return node_edge_feat

class SpatialConv(nn.Module):
    """
    空间图卷积层，结合边特征和节点特征进行特征更新
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.2, alpha=0.2, activation='relu'):
        super(SpatialConv, self).__init__()
        self.hidden_size = hidden_size
        self.gat_edge = GATLayer(hidden_size, hidden_size, num_heads=num_heads, dropout=dropout, alpha=alpha, activation=activation)
        self.sgat_node = SGATLayer(hidden_size, hidden_size, num_heads=num_heads, dropout=dropout, alpha=alpha, activation=activation)
        self.aggregate_edges_from_nodes = AggregateEdgesFromNodes(hidden_size)

    def forward(self, e2n_gw, e2e_gw, srcs, dsts, node_edge_feat, dist_feat_order, dist_feat, nids, eids, nlod, elod):
        """
        Args:
            e2n_gw (GraphWrapper): 边到节点的图结构
            e2e_gw (GraphWrapper): 边到边的图结构
            srcs (Tensor): [num_edges]
            dsts (Tensor): [num_edges]
            node_edge_feat (Tensor): [num_nodes + num_edges, hidden_size]
            dist_feat_order (Tensor): [num_order_edges, hidden_size]
            dist_feat (Tensor): [num_edges, hidden_size]
            nids (Tensor): [num_nodes]
            eids (Tensor): [num_edges]
            nlod (List[int]): 每个图的节点数量
            elod (List[int]): 每个图的边数量
        Returns:
            Tensor: 更新后的节点和边特征矩阵
        """
        # 步骤1：更新边特征
        node_feat = node_edge_feat[nids]  # [num_nodes, hidden_size]
        edge_feat = self.aggregate_edges_from_nodes(node_edge_feat, dist_feat, srcs, dsts)  # [num_edges, hidden_size]
        node_edge_feat_lod = concat_node_edge_feat(node_feat, edge_feat)  # [num_nodes + num_edges, hidden_size]
        node_edge_feat = self.gat_edge(e2e_gw, node_edge_feat_lod)  # [num_nodes + num_edges, hidden_size * num_heads]

        # 步骤2：更新节点特征
        edge_feat = node_edge_feat[eids]  # [num_edges, hidden_size * num_heads]
        node_edge_feat_lod = concat_node_edge_feat(node_feat, edge_feat)  # [num_nodes + num_edges, hidden_size * num_heads]
        node_edge_feat = self.sgat_node(e2n_gw, node_edge_feat_lod, dist_feat)  # [num_nodes + num_edges, hidden_size * num_heads]

        # 更新节点和边的特征矩阵
        node_feat = node_edge_feat[nids]  # [num_nodes, hidden_size * num_heads]
        node_edge_feat = concat_node_edge_feat(node_feat, edge_feat)  # [num_nodes + num_edges, hidden_size * num_heads]

        return node_edge_feat
