# SDCN_Spatial: Integrating SMAN's Dual Aggregation Mechanism with the SDCN Clustering Framework

This project implements an improved version of SDCN (Structural Deep Clustering Network) by integrating the dual aggregation mechanism from SMAN (Spatial Multi-Attention Network) to enhance its graph structure awareness.

## Core Improvements

### 1. Integration of Dual Aggregation Mechanism

The original SDCN uses simple GNN layers to process node features but ignores edge features and complex spatial relationships. We introduce SMAN's dual aggregation mechanism:

- **Node→Edge Aggregation**: Edge features are updated considering the features of their connected source and target nodes.
- **Edge→Edge Aggregation**: Edge features are further refined by considering features of other edges sharing common nodes.
- **Edge→Node Aggregation**: Node features are updated by aggregating the features of edges connected to them.

This dual aggregation mechanism allows for a more comprehensive utilization of the graph's structural information, especially edge features and spatial relationships.

### 2. Spatial Consistency Constraint

To address the issue that "spatial information was not utilized to introduce additional constraints in SMAN_layers," we designed a spatial consistency loss function:

```python
def spatial_consistency_loss(z, edge_index, edge_attr=None, margin=0.5):
    """Encourages closely connected nodes to be closer in the embedding space"""
    src, dst = edge_index
    src_z = z[src]
    dst_z = z[dst]

    # Calculate pairwise distances between connected nodes
    node_dists = F.pairwise_distance(src_z, dst_z, p=2)

    # If edge features are provided, weight the loss by them
    if edge_attr is not None:
        # Calculate edge similarity (inverse of distance)
        edge_sim = 1.0 / (1.0 + torch.norm(edge_attr, dim=1))

        # Weighted loss: nodes connected by similar edges should be closer
        loss = torch.mean(node_dists * edge_sim)
    else:
        # Simple version: all connected nodes should be close in the embedding space
        loss = torch.mean(F.relu(node_dists - margin))

    return loss
```

This loss function ensures that nodes that are spatially adjacent or connected by important edges remain close in the embedding space, thus explicitly utilizing spatial information.

### 3. Direct Constraint on Edge Feature Learning

To address the issue that "the loss function did not directly constrain the learning of edge features," we designed:

1.  **EdgeDecoder**: Predicts edge features from node embeddings.
2.  **Edge Consistency Loss**: Consists of two parts:
    *   Edge Reconstruction Loss: Ensures edge features can be reconstructed from node embeddings.
    *   Edge Smoothness Loss: Ensures edge features remain consistent across adjacent layers.

```python
def edge_consistency_loss(edge_feat_dict):
    """Calculates edge feature consistency loss to regularize edge feature learning"""
    # Edge reconstruction loss
    edge_recon_loss = F.mse_loss(edge_feat_dict['pred_edge_feat'], edge_feat_dict['orig_edge_feat'])

    # Edge feature smoothness loss (consecutive layers should have similar edge features)
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

    # Combine the two losses
    return edge_recon_loss + 0.1 * smoothness_loss
```

### 4. Dynamic Weight Adjustment Strategy

We adopted a strategy of dynamically adjusting loss weights, introducing spatial and edge feature constraints with small weights initially and gradually increasing them as training progresses:

```python
# Loss weight scheduler
spatial_weight = 0.01  # Start with small weights
edge_weight = 0.01

# Gradually increase weights in the training loop
if epoch < 20:
    spatial_weight = min(0.05, 0.01 + epoch * 0.002)
    edge_weight = min(0.05, 0.01 + epoch * 0.002)
```

This strategy ensures the model first learns basic data representations and then gradually strengthens the influence of spatial structure and edge features.

## Improved Loss Function

The improved SDCN_Spatial loss function includes five components:

```python
# Original SDCN losses
kl_loss = F.kl_div(q.log(), p, reduction='batchmean')  # KL divergence between cluster assignment and target distribution
ce_loss = F.kl_div(pred.log(), p, reduction='batchmean') # KL divergence between prediction and target distribution
re_loss = F.mse_loss(x_bar, data)  # Reconstruction loss

# Newly added spatial and edge feature constraints
sc_loss = spatial_consistency_loss(z, edge_index, edge_features['orig_edge_feat'])  # Spatial consistency loss
edge_loss = edge_consistency_loss(edge_features)  # Edge feature consistency loss

# Combined loss
loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + spatial_weight * sc_loss + edge_weight * edge_loss
```

## Implementation Architecture

The improved model includes the following main components:

1.  **Autoencoder (AE)**: Same as the original SDCN, used to learn low-dimensional representations of node features.
2.  **EdgeDecoder**: Reconstructs edge features from node representations.
3.  **SpatialConv Layer**: Ported from SMAN, implements the dual aggregation mechanism.
4.  **SpatialEmbedding Layer**: Handles the embedding of edge features.

## Differences from the Original Model

| Feature             | Original SDCN_Spatial | Improved SDCN_Spatial        |
|---------------------|-----------------------|------------------------------|
| Edge Feature Handling | Implicit learning     | Explicit learning with dedicated loss constraints |
| Spatial Info Usage  | No explicit constraint| Dedicated spatial consistency loss |
| Loss Function       | 3 components (fixed weights) | 5 components (dynamic weights) |
| Edge Feature Recon. | None                  | Dedicated EdgeDecoder        |

## Usage

Run the comparison script directly to compare the performance difference between the original and improved versions:

```bash
python compare_sdcn_models.py --name usps --run_original --run_improved
```

You can also run the improved version separately:

```bash
python spatial_sdcn_improved.py --name usps
```

## Expected Effects

1.  **Improved Clustering Accuracy**: By better utilizing edge features and spatial relationships, the improved model should show improvements in clustering accuracy and F1 score.
2.  **Faster Convergence**: Explicit spatial and edge feature constraints help the model converge faster to a better representation.
3.  **More Robust Representation**: The dual aggregation mechanism and multiple constraints make the learned representation more robust and less sensitive to noise and outliers.

## Summary

The improved SDCN_Spatial model addresses the two main issues of the original model by:

1.  Introducing a dedicated spatial consistency loss to enhance the learning of spatial properties.
2.  Designing edge feature reconstruction and smoothness losses to directly constrain edge feature learning.

These improvements enable the model to more fully utilize the graph's topological structure and spatial relationships, enhancing clustering performance. Additionally, the dynamic weight adjustment strategy ensures the balance and stability of model training.