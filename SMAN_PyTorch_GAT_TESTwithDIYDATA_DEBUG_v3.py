# 1. 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter  # 用于 TensorBoard
import numpy as np
from torch_geometric.loader import DataLoader


# 2. 定义模型和相关层
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

        # 定义空间嵌入层
        self.spatial_embedding = SpatialEmbedding(input_dim=self.dist_dim, embed_size=self.hidden_size)

        # 定义空间卷积层
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

        # 定义全连接层和 dropout
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.fc2 = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.fc3 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.n_output)
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # 定义激活函数
        self.relu = nn.ReLU()

        # 定义损失函数
        self.loss_fn = nn.MSELoss()

    def forward(self, data, return_intermediate=False):
        """
        前向传播，可选择返回中间输出以进行调试。
        """
        intermediates = {}

        # 提取输入特征
        edges_dist = data.edges_dist  # Tensor: [num_edges, dist_dim]
        dist_feat_order = data.edges_dist_order  # Tensor: [num_edges, dist_dim], optional

        # 空间嵌入
        dist_feat, dist_feat_order = self.spatial_embedding(edges_dist, dist_feat_order)
        intermediates[
            'spatial_embedding'] = dist_feat.shape, dist_feat_order.shape if dist_feat_order is not None else None

        # 使用 TensorBoard 记录张量形状
        writer.add_text('Shapes/spatial_embedding/dist_feat', str(dist_feat.shape))
        if dist_feat_order is not None:
            writer.add_text('Shapes/spatial_embedding/dist_feat_order', str(dist_feat_order.shape))

        node_edge_feat = data.x  # 节点和边的特征，Tensor: [num_nodes + num_edges, feat_size]
        intermediates['initial_node_edge_feat'] = node_edge_feat.shape
        writer.add_text('Shapes/initial_node_edge_feat', str(node_edge_feat.shape))

        # 多层空间卷积
        for i in range(self.num_layers):
            node_edge_feat = self.spatial_convs[i](
                data=data,
                node_edge_feat=node_edge_feat,
                dist_feat_order=dist_feat_order,
                dist_feat=dist_feat
            )
            intermediates[f'spatial_conv_{i}'] = node_edge_feat.shape
            writer.add_text(f'Shapes/spatial_conv_{i}', str(node_edge_feat.shape))

        # 节点特征聚合
        node_feat = node_edge_feat[data.nids]  # 假设 nids 是节点索引
        if self.pool_type == 'sum':
            pooled_h = global_add_pool(node_feat, data.batch)
        elif self.pool_type == 'mean':
            pooled_h = global_mean_pool(node_feat, data.batch)
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")
        intermediates['pooled_h'] = pooled_h.shape
        writer.add_text('Shapes/pooled_h', str(pooled_h.shape))

        # 全连接层和 dropout
        output = self.relu(self.fc1(pooled_h))
        intermediates['fc1_output'] = output.shape
        writer.add_text('Shapes/fc1_output', str(output.shape))
        output = self.dropout(output)

        output = self.relu(self.fc2(output))
        intermediates['fc2_output'] = output.shape
        writer.add_text('Shapes/fc2_output', str(output.shape))
        output = self.dropout(output)

        output = self.relu(self.fc3(output))
        intermediates['fc3_output'] = output.shape
        writer.add_text('Shapes/fc3_output', str(output.shape))
        output = self.dropout(output)

        output = self.fc_out(output)
        intermediates['final_output'] = output.shape
        writer.add_text('Shapes/final_output', str(output.shape))

        if return_intermediate:
            return output.squeeze(), intermediates
        else:
            return output.squeeze()


class SpatialEmbedding(nn.Module):
    """
    空间嵌入层
    """

    def __init__(self, input_dim, embed_size):
        super(SpatialEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embed_size)

    def forward(self, dist_feat, dist_feat_order=None):
        dist_feat = self.fc(dist_feat)
        if dist_feat_order is not None:
            dist_feat_order = self.fc(dist_feat_order)
            return dist_feat, dist_feat_order
        return dist_feat, None


class AggregateEdgesFromNodes(nn.Module):
    """
    从节点特征聚合边特征
    """

    def __init__(self, hidden_size):
        super(AggregateEdgesFromNodes, self).__init__()
        self.fc = nn.Linear(2 * hidden_size + hidden_size, hidden_size)

    def forward(self, node_edge_feat, dist_feat, srcs, dsts):
        src_feat = node_edge_feat[srcs]  # [num_edges, hidden_size]
        dst_feat = node_edge_feat[dsts]  # [num_edges, hidden_size]
        feat_h = torch.cat([src_feat, dst_feat, dist_feat], dim=-1)  # [num_edges, 3 * hidden_size]
        feat_h = F.relu(self.fc(feat_h))  # [num_edges, hidden_size]
        return feat_h


def concat_node_edge_feat(node_feat, edge_feat):
    """
    连接节点和边的特征以形成节点-边特征矩阵。
    """
    node_edge_feat = torch.cat([node_feat, edge_feat], dim=0)
    return node_edge_feat


class GATLayer(nn.Module):
    """
    手动实现的图注意力层，用于更新边特征
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
        前向传播。
        """
        Wh = torch.matmul(h, self.W)  # [N, num_heads * out_features]
        N = Wh.size(0)
        Wh = Wh.view(N, self.num_heads, self.out_features)  # [N, num_heads, out_features]

        src = edge_index[0]  # [num_edges]
        dst = edge_index[1]  # [num_edges]

        Wh_src = Wh[src]  # [num_edges, num_heads, out_features]
        Wh_dst = Wh[dst]  # [num_edges, num_heads, out_features]

        # 计算注意力分数
        e = self.leakyrelu(
            (Wh_src * self.a_src).sum(dim=-1) + (Wh_dst * self.a_dst).sum(dim=-1)
        )  # [num_edges, num_heads]

        # Softmax 归一化
        attention = F.softmax(e, dim=0)  # [num_edges, num_heads]
        attention = self.dropout(attention)

        # 加权特征聚合
        h_prime = torch.zeros_like(Wh)
        for head in range(self.num_heads):
            h_prime[:, head, :] = h_prime[:, head, :].index_add(
                0, dst, attention[:, head].unsqueeze(-1) * Wh_src[:, head, :]
            )

        if self.activation == 'relu':
            h_prime = F.relu(h_prime)
        elif self.activation == 'elu':
            h_prime = F.elu(h_prime)

        # 重塑回 [N, num_heads * out_features]
        h_prime = h_prime.view(N, self.num_heads * self.out_features)
        return h_prime


class SGATLayer(nn.Module):
    """
    手动实现的空间图注意力层，用于更新节点特征
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
        前向传播。
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

        # 计算包括边特征的注意力分数
        e = self.leakyrelu(
            (Wh_src * self.a_src).sum(dim=-1) +
            (Wh_dst * self.a_dst).sum(dim=-1) +
            (We * self.a_edge).sum(dim=-1)
        )  # [num_edges, num_heads]

        # Softmax 归一化
        attention = F.softmax(e, dim=0)  # [num_edges, num_heads]
        attention = self.dropout(attention)

        # 加权特征聚合
        h_prime = torch.zeros_like(Wh)
        for head in range(self.num_heads):
            h_prime[:, head, :] = h_prime[:, head, :].index_add(
                0, dst, attention[:, head].unsqueeze(-1) * Wh_src[:, head, :]
            )

        if self.activation == 'relu':
            h_prime = F.relu(h_prime)
        elif self.activation == 'elu':
            h_prime = F.elu(h_prime)

        # 重塑回 [N, num_heads * out_features]
        h_prime = h_prime.view(N, self.num_heads * self.out_features)
        return h_prime


class SpatialConv(nn.Module):
    """
    结合边和节点特征的空间图卷积层，用于特征更新
    """

    def __init__(self, hidden_size, num_heads=4, dropout=0.2, alpha=0.2, activation='relu'):
        super(SpatialConv, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.gat_edge = GATLayer(
            in_features=hidden_size,
            out_features=hidden_size // num_heads,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            activation=activation
        )
        self.sgat_node = SGATLayer(
            in_features=hidden_size,
            out_features=hidden_size // num_heads,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            activation=activation
        )
        self.aggregate_edges_from_nodes = AggregateEdgesFromNodes(hidden_size)

    def forward(self, data, node_edge_feat, dist_feat_order, dist_feat):
        """
        前向传播。
        """
        srcs = data.srcs  # [num_edges]
        dsts = data.dsts  # [num_edges]
        nids = data.nids  # [num_nodes]
        eids = data.eids  # [num_edges]

        # 第一步：更新边特征
        node_feat = node_edge_feat[nids]  # [num_nodes, hidden_size]
        edge_feat = self.aggregate_edges_from_nodes(node_edge_feat, dist_feat_order, srcs,
                                                    dsts)  # [num_edges, hidden_size]

        # 记录张量形状到 TensorBoard
        writer.add_text('Shapes/SpatialConv/edge_feat', str(edge_feat.shape))

        node_edge_feat_lod = concat_node_edge_feat(node_feat, edge_feat)  # [num_nodes + num_edges, hidden_size]
        writer.add_text('Shapes/SpatialConv/node_edge_feat_lod', str(node_edge_feat_lod.shape))

        node_edge_feat = self.gat_edge(data.edge_index_e2e, node_edge_feat_lod)  # Edge-to-edge graph
        writer.add_text('Shapes/SpatialConv/after_gat_edge', str(node_edge_feat.shape))

        # 第二步：更新节点特征
        edge_feat = node_edge_feat[eids]  # [num_edges, hidden_size * num_heads]
        writer.add_text('Shapes/SpatialConv/edge_feat_after_gat_edge', str(edge_feat.shape))

        node_edge_feat_lod = concat_node_edge_feat(node_feat,
                                                   edge_feat)  # [num_nodes + num_edges, hidden_size * num_heads]
        writer.add_text('Shapes/SpatialConv/node_edge_feat_lod_after_concat', str(node_edge_feat_lod.shape))

        node_edge_feat = self.sgat_node(data.edge_index_e2n, node_edge_feat_lod, dist_feat)  # Edge-to-node graph
        writer.add_text('Shapes/SpatialConv/after_sgat_node', str(node_edge_feat.shape))

        # 更新节点和边特征矩阵
        node_feat = node_edge_feat[nids]  # [num_nodes, hidden_size * num_heads]
        node_edge_feat = concat_node_edge_feat(node_feat, edge_feat)  # [num_nodes + num_edges, hidden_size * num_heads]
        writer.add_text('Shapes/SpatialConv/node_edge_feat_final', str(node_edge_feat.shape))

        return node_edge_feat


# 3. 生成合成数据
def generate_synthetic_data(num_graphs, num_nodes_per_graph, num_edges_per_graph, node_feat_dim, edge_feat_dim,
                            dist_dim):
    data_list = []
    for _ in range(num_graphs):
        num_nodes = num_nodes_per_graph
        num_edges = num_edges_per_graph

        # 节点特征
        node_features = torch.randn(num_nodes, node_feat_dim)

        # 边索引
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # 边特征
        edge_features = torch.randn(num_edges, edge_feat_dim)

        # 距离特征
        edges_dist = torch.randn(num_edges, dist_dim)
        edges_dist_order = edges_dist.clone()  # 简单起见，假设与 edges_dist 相同

        # 节点和边的 ID
        nids = torch.arange(num_nodes)
        eids = torch.arange(num_edges)

        # 边的源和目标节点索引
        srcs = edge_index[0]
        dsts = edge_index[1]

        # 边到边的边索引（简化为每条边的自环）
        edge_index_e2e = torch.stack([eids, eids], dim=0)

        # 边到节点的边索引（边指向目标节点）
        edge_index_e2n = torch.stack([eids, dsts], dim=0)

        # 连接节点和边的特征
        x = concat_node_edge_feat(node_features, edge_features)

        # 批次索引（由于每次处理一个图，所有节点的批次索引为 0）
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # 目标值（合成）
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
            batch=batch,
            pk=pk
        )

        data_list.append(data)

    return data_list


# 4. 自定义数据加载器
def custom_collate_fn(data_list):
    batch = Batch.from_data_list(data_list)

    # 调整自定义索引
    node_offset = 0
    edge_offset = 0
    nids_list = []
    eids_list = []
    srcs_list = []
    dsts_list = []

    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1)

        nids_list.append(data.nids + node_offset)
        eids_list.append(data.eids + edge_offset)
        srcs_list.append(data.srcs + node_offset)
        dsts_list.append(data.dsts + node_offset)

        node_offset += num_nodes
        edge_offset += num_edges

    batch.nids = torch.cat(nids_list, dim=0)
    batch.eids = torch.cat(eids_list, dim=0)
    batch.srcs = torch.cat(srcs_list, dim=0)
    batch.dsts = torch.cat(dsts_list, dim=0)

    return batch


# 5. 集成 torchsummary 和 TensorBoard
# 初始化 TensorBoard 的 SummaryWriter
writer = SummaryWriter('runs/model_shape')

# 参数设置
num_graphs = 100
num_nodes_per_graph = 10
num_edges_per_graph = 20
node_feat_dim = 32  # 设置为 hid_dim
edge_feat_dim = 32  # 设置为 hid_dim
dist_dim = 8  # 可以保持原样

# 生成合成数据
data_list = generate_synthetic_data(
    num_graphs=num_graphs,
    num_nodes_per_graph=num_nodes_per_graph,
    num_edges_per_graph=num_edges_per_graph,
    node_feat_dim=node_feat_dim,
    edge_feat_dim=edge_feat_dim,
    dist_dim=dist_dim
)

# 创建 DataLoader
batch_size = 32
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# 模型参数
args = {
    'num_layers': 2,
    'hid_dim': 32,
    'pool_type': 'mean',
    'drop': 0.5,
    'dist_dim': dist_dim
}

# 初始化模型
model = SMANModel(args)


# 尝试使用 torchsummary
# 由于 torchsummary 需要一个标准的张量输入，我们无法直接使用 Data 对象
# 这里提供一个替代方法，定义一个函数来返回模型的总参数量
def model_summary(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    return total_params


model_summary(model)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录模型图到 TensorBoard
# 由于模型接受的是 Data 对象，我们需要创建一个示例输入
sample_batch = next(iter(loader))
try:
    writer.add_graph(model, sample_batch)
    print("Model graph added to TensorBoard.")
except Exception as e:
    print(f"Failed to add graph to TensorBoard: {e}")

# 6. 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch_data in enumerate(loader):
        optimizer.zero_grad()
        # 运行模型并获取中间输出
        output, intermediates = model(batch_data, return_intermediate=True)

        # 在训练循环中记录中间张量形状
        for name, shape in intermediates.items():
            writer.add_text(f'Shapes/Epoch_{epoch + 1}_Batch_{batch_idx + 1}/{name}', str(shape))

        # 计算损失并进行反向传播
        loss = F.mse_loss(output, batch_data.pk.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 将平均损失记录到 TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch + 1)

# 关闭 TensorBoard 的 writer
writer.close()
