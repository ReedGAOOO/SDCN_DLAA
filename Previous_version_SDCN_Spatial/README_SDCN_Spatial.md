# SDCN_Spatial：融合SMAN双聚合机制与SDCN聚类框架

本项目实现了一个改进版的SDCN（结构化深度聚类网络），通过融合SMAN（空间多注意力网络）的双聚合机制来增强其图结构感知能力。

## 核心改进

### 1. 双聚合机制的整合

原始SDCN使用简单的GNN层处理节点特征，但忽略了边特征和复杂的空间关系。我们引入了SMAN的双聚合机制：

- **节点→边聚合**：边特征更新考虑其连接的源节点和目标节点的特征
- **边→边聚合**：边特征进一步考虑与其共享节点的其他边的特征
- **边→节点聚合**：节点特征通过聚合连接到它的边的特征进行更新

这种双聚合机制能够更充分地利用图的结构信息，特别是边特征和空间关系。

### 2. 空间一致性约束

为解决"在SMAN_layers中没有利用空间信息引入额外约束"的问题，我们设计了空间一致性损失函数：

```python
def spatial_consistency_loss(z, edge_index, edge_attr=None, margin=0.5):
    """鼓励连接紧密的节点在嵌入空间中也更接近"""
    src, dst = edge_index
    src_z = z[src]
    dst_z = z[dst]
    
    # 计算连接节点之间的成对距离
    node_dists = F.pairwise_distance(src_z, dst_z, p=2)
    
    # 如果提供了边特征，用它们加权损失
    if edge_attr is not None:
        # 计算边相似度（距离的倒数）
        edge_sim = 1.0 / (1.0 + torch.norm(edge_attr, dim=1))
        
        # 加权损失：通过相似边连接的节点应该更接近
        loss = torch.mean(node_dists * edge_sim)
    else:
        # 简单版本：所有连接的节点在嵌入空间中应该接近
        loss = torch.mean(F.relu(node_dists - margin))
    
    return loss
```

这个损失函数确保了空间上相邻或通过重要边连接的节点在嵌入空间中也应当保持接近，从而显式地利用了空间信息。

### 3. 边特征学习的直接约束

为解决"损失函数中未对边特征的学习进行直接约束"的问题，我们设计了：

1. **边重构器（EdgeDecoder）**：从节点嵌入预测边特征
2. **边一致性损失**：包含两部分
   - 边重构损失：确保边特征可以从节点嵌入中重建
   - 边平滑性损失：确保相邻层中的边特征保持一致性

```python
def edge_consistency_loss(edge_feat_dict):
    """计算边特征一致性损失以规范化边特征学习"""
    # 边重构损失
    edge_recon_loss = F.mse_loss(edge_feat_dict['pred_edge_feat'], edge_feat_dict['orig_edge_feat'])
    
    # 边特征平滑性损失（连续层应有相似的边特征）
    smoothness_loss = 0.0
    edge_feats = [
        edge_feat_dict['edge_feat1'],
        edge_feat_dict['edge_feat2'],
        edge_feat_dict['edge_feat3'],
        edge_feat_dict['edge_feat4'],
        edge_feat_dict['edge_feat5']
    ]
    
    for i in range(len(edge_feats) - 1):
        smoothness_loss += F.mse_loss(edge_feats[i], edge_feats[i+1])
    
    smoothness_loss /= (len(edge_feats) - 1)
    
    # 组合两个损失
    return edge_recon_loss + 0.1 * smoothness_loss
```

### 4. 动态权重调整策略

我们采用了动态调整损失权重的策略，在训练初期小幅度引入空间和边特征约束，随训练进行逐步增加权重：

```python
# 损失权重调度器
spatial_weight = 0.01  # 从小权重开始
edge_weight = 0.01

# 在训练循环中逐渐增加权重
if epoch < 20:
    spatial_weight = min(0.05, 0.01 + epoch * 0.002)
    edge_weight = min(0.05, 0.01 + epoch * 0.002)
```

这种策略确保了模型首先学习基本的数据表示，然后逐渐加强空间结构和边特征的影响。

## 改进型损失函数

改进的SDCN_Spatial损失函数包含五个组件：

```python
# 原始SDCN损失
kl_loss = F.kl_div(q.log(), p, reduction='batchmean')  # 聚类分配与目标分布之间的KL散度
ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')  # 预测与目标分布之间的KL散度
re_loss = F.mse_loss(x_bar, data)  # 重构损失

# 新增的空间与边特征约束
sc_loss = spatial_consistency_loss(z, edge_index, edge_features['orig_edge_feat'])  # 空间一致性损失
edge_loss = edge_consistency_loss(edge_features)  # 边特征一致性损失

# 组合损失
loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + spatial_weight * sc_loss + edge_weight * edge_loss
```

## 实现架构

改进后的模型包含以下主要组件：

1. **自编码器（AE）**：与原始SDCN相同，用于学习节点特征的低维表示
2. **边解码器（EdgeDecoder）**：从节点表示重构边特征
3. **空间卷积层（SpatialConv）**：从SMAN中移植，实现双聚合机制
4. **空间嵌入层（SpatialEmbedding）**：处理边特征的嵌入

## 与原始模型的区别

| 特性 | 原始SDCN_Spatial | 改进版SDCN_Spatial |
|-----|-----------------|------------------|
| 边特征处理 | 隐式学习 | 显式学习并有专门的损失约束 |
| 空间信息利用 | 无显式约束 | 有专门的空间一致性损失 |
| 损失函数 | 3个组件（固定权重） | 5个组件（动态权重） |
| 边特征重构 | 无 | 有专门的边解码器 |

## 使用方法

直接运行比较脚本可比较原始版本和改进版本的性能差异：

```bash
python compare_sdcn_models.py --name usps --run_original --run_improved
```

也可以单独运行改进版本：

```bash
python spatial_sdcn_improved.py --name usps
```

## 预期效果

1. **聚类精度提升**：通过更好地利用边特征和空间关系，改进模型应当在聚类准确率和F1分数上有所提升。
2. **收敛速度加快**：显式的空间和边特征约束有助于模型更快地收敛到更好的表示。
3. **更鲁棒的表示**：双聚合机制和多重约束使得模型学习到的表示更加鲁棒，对噪声和异常值更不敏感。

## 总结

改进的SDCN_Spatial模型通过以下方式解决了原有模型的两个主要问题：

1. 引入专门的空间一致性损失来增强空间特性的学习
2. 设计边特征重构和平滑损失来直接约束边特征的学习

这些改进使得模型能够更充分地利用图的拓扑结构和空间关系，提升聚类性能。同时，动态权重调整策略确保了模型训练的平衡性和稳定性。