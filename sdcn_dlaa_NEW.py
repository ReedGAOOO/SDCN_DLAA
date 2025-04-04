from __future__ import print_function, division
import argparse
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index, dense_to_sparse
from Reference.SDCN_ORIGINAL.utils import load_data, load_graph
from evaluation import eva
import sys
import os
from datetime import datetime

# Import SpatialConv from DLAA
from DLAA_NEW import SpatialConv


class AE(nn.Module):
    """
    Autoencoder module for SDCN, same as the original implementation
    """
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
 
    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        # Store shapes in a dictionary without printing
        self.layer_shapes = {
            'autoencoder': {
                'Encoder Layer 1': enc_h1.shape,
                'Encoder Layer 2': enc_h2.shape,
                'Encoder Layer 3': enc_h3.shape,
                'Latent Space': z.shape,
                'Decoder Layer 1': dec_h1.shape,
                'Decoder Layer 2': dec_h2.shape,
                'Decoder Layer 3': dec_h3.shape,
                'Output Layer': x_bar.shape
            }
        }

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN_DLAA(nn.Module):
    """
    SDCN_DLAA (Spatial Deep Clustering Network with Deep Learning-based Attentional Aggregation)
    
    A performance-enhanced version of SDCN that incorporates spatial graph attention from DLAA
    with optimized graph structure preprocessing. This model is based on the design principles
    of the original SMAN architecture, pre-computing graph structures for efficient processing.
    """
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                n_input, n_z, n_clusters, v=1, dropout=0.2, heads=4, edge_dim=None, 
                max_edges_per_node=10, precomputed_edge_index=None, precomputed_edge_to_edge_index=None):
        super(SDCN_DLAA, self).__init__()
        # Initialize epoch tracking variables
        self.current_epoch = 0
        self.last_logged_epoch = -1
        
        # Hyperparameters
        self.n_input = n_input
        self.n_z = n_z
        self.n_clusters = n_clusters
        self.v = v
        self.dropout = dropout
        self.heads = heads
        self.edge_dim = edge_dim if edge_dim is not None else n_input
        self.max_edges_per_node = max_edges_per_node

        # Cache for graph structures - this is the key optimization
        self.precomputed_edge_index = precomputed_edge_index
        self.precomputed_edge_to_edge_index = precomputed_edge_to_edge_index
        self.graph_cache = {}
        
        # Autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        
        # SpatialConv layers replacing GNNLayers
        self.spatial_conv1 = SpatialConv(n_enc_1, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        self.spatial_conv2 = SpatialConv(n_enc_2, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        self.spatial_conv3 = SpatialConv(n_enc_3, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        self.spatial_conv4 = SpatialConv(n_z, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        self.spatial_conv5 = SpatialConv(n_clusters, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        
        # Projection layers to match dimensions between layers
        self.proj1 = nn.Linear(n_input, n_enc_1)
        self.proj2 = nn.Linear(n_enc_1, n_enc_2)
        self.proj3 = nn.Linear(n_enc_2, n_enc_3)
        self.proj4 = nn.Linear(n_enc_3, n_z)
        self.proj5 = nn.Linear(n_z, n_clusters)
        
        # Cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
        # Add edge feature projection layer for initial edge features
        self.initial_edge_proj = None
        if edge_dim is not None and edge_dim != n_input:
            self.initial_edge_proj = nn.Linear(edge_dim, edge_dim)

    def _prepare_pyg_data(self, x, adj, edge_attr, max_edges_per_node=10):
        """
        Optimized method to prepare PyG Data object from node features and adjacency matrix
        Caches graph structure to avoid redundant computation
        
        Args:
            x: Node features [num_nodes, feature_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
            edge_attr: Edge features [num_edges, edge_dim]
            max_edges_per_node: Maximum number of edges to consider per node
            
        Returns:
            data: PyG Data object
        """
        num_nodes = x.size(0)
        
        # Important fix: Add validation for node count and feature dimensions
        if num_nodes < 4:
            raise ValueError(f"Number of nodes must be greater than 3, current is {num_nodes}")
            
        # Check if x contains valid features
        if not torch.is_floating_point(x) or torch.isnan(x).any():
            raise ValueError("Node features contain invalid values (NaN or non-float numbers)")
            
        # Check if we already have precomputed graph structures from initialization
        if self.precomputed_edge_index is not None and self.precomputed_edge_to_edge_index is not None:
            # Important fix: Validate precomputed edge indices don't exceed node range
            max_node_idx = self.precomputed_edge_index.max().item()
            if max_node_idx >= num_nodes:
                print(f"Warning: Precomputed edge index ({max_node_idx}) exceeds current node count ({num_nodes}), recalculating...")
                # Clear precomputed results and recalculate
                self.precomputed_edge_index = None
                self.precomputed_edge_to_edge_index = None
                # Continue with recalculation logic below
            else:
                # If precomputed edge indices are valid, use them to create data object
                data = Data(
                    x=x,
                    edge_index=self.precomputed_edge_index,
                    edge_attr=edge_attr,
                    dist_feat=edge_attr,
                    dist_feat_order=edge_attr,
                    edge_to_edge_index=self.precomputed_edge_to_edge_index
                )
                return data
        
        # Create a cache key based on adjacency matrix properties and parameters
        # For sparse tensors, use a hash of indices and values
        if adj.is_sparse:
            adj_id = f"{adj._indices().sum().item()}_{adj._values().sum().item()}"
        else:
            adj_id = f"{adj.sum().item()}"
            
        # Important fix: Include node count and feature dimensions in cache key
        cache_key = f"{adj_id}_{max_edges_per_node}_{num_nodes}_{x.size(1)}"
        
        # Check if we have cached this graph structure
        if cache_key in self.graph_cache:
            cached = self.graph_cache[cache_key]
            
            # Important fix: Validate cached edge indices don't exceed node range
            max_node_idx = cached['edge_index'].max().item()
            if max_node_idx >= num_nodes:
                print(f"Warning: Cached edge index ({max_node_idx}) exceeds current node count ({num_nodes}), recalculating...")
                # Remove invalid cache entry
                self.graph_cache.pop(cache_key)
                # Continue with recalculation logic below
            else:
                # If cached edge indices are valid, use them to create data object
                data = Data(
                    x=x,
                    edge_index=cached['edge_index'],
                    edge_attr=edge_attr,
                    dist_feat=edge_attr,
                    dist_feat_order=edge_attr,
                    edge_to_edge_index=cached['edge_to_edge_index']
                )
                return data
            
        # If not cached, compute the graph structure (first time only)
        # Convert adjacency matrix to edge_index
        if adj.is_sparse:
            adj = adj.coalesce()
            edge_index = adj.indices()
        else:
            edge_index, _ = dense_to_sparse(adj)
        
        # Validate edge indices
        max_index = edge_index.max().item()
        
        if max_index >= num_nodes:
            print(f"Warning: Edge index contains indices ({max_index}) that exceed the number of nodes ({num_nodes})")
            print(f"Filtering edges to only include those with valid node indices...")
            
            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_edges_mask]
            
            if edge_index.size(1) == 0:
                print("Error: No valid edges remain after filtering!")
                edge_index = torch.zeros((2, 1), dtype=torch.long).to(x.device)
                edge_index[0, 0] = 0
                edge_index[1, 0] = min(1, num_nodes-1)
        
        num_edges = edge_index.size(1)
        
        # Process edge features
        dist_feat = edge_attr
        
        # Apply initial projection if needed
        if self.initial_edge_proj is not None:
            dist_feat = self.initial_edge_proj(dist_feat)
        
        # Create edge-to-edge graph more efficiently
        print("Building edge-to-edge graph (one-time operation)...")
        
        # Build a mapping from nodes to their connected edges
        node_to_edges = defaultdict(list)
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            node_to_edges[src.item()].append(i)
            node_to_edges[dst.item()].append(i)
        
        # Create edge connections based on shared nodes
        edge_to_edge_list = []
        
        # For each node, connect all edges that share this node
        for node, connected_edges in node_to_edges.items():
            # Only process if the node connects multiple edges
            if len(connected_edges) > 1:
                # If node connects too many edges, randomly sample to limit
                if len(connected_edges) > max_edges_per_node:
                    sampled_edges = random.sample(connected_edges, max_edges_per_node)
                else:
                    sampled_edges = connected_edges
                
                # Connect all pairs of edges that share this node
                for i in range(len(sampled_edges)):
                    for j in range(i+1, len(sampled_edges)):
                        edge_i = sampled_edges[i]
                        edge_j = sampled_edges[j]
                        # Add both directions for undirected graph
                        edge_to_edge_list.append([edge_i, edge_j])
                        edge_to_edge_list.append([edge_j, edge_i])
        
        # Convert to tensor representation
        if len(edge_to_edge_list) > 0:
            edge_to_edge_index = torch.tensor(edge_to_edge_list, dtype=torch.long).t().to(x.device)
        else:
            # If no edge-to-edge connections, create empty tensor
            edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long).to(x.device)
        
        # Store in cache for future use
        self.graph_cache[cache_key] = {
            'edge_index': edge_index,
            'edge_to_edge_index': edge_to_edge_index
        }
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=dist_feat,
            dist_feat=dist_feat,
            dist_feat_order=dist_feat,
            edge_to_edge_index=edge_to_edge_index
        )
        
        # Also store as precomputed values for future use
        self.precomputed_edge_index = edge_index
        self.precomputed_edge_to_edge_index = edge_to_edge_index
        
        return data

    def forward(self, x, adj, edge_attr=None):
        """
        Forward pass of the model
        
        Args:
            x: Node features [num_nodes, n_input]
            adj: Adjacency matrix
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            x_bar: Reconstructed features
            q: Soft assignment
            predict: Cluster prediction
            z: Latent representation
            spatial_shapes: Dictionary of layer shapes
        """
        original_nodes = x.size(0)
        
        # Get autoencoder outputs
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        # Prepare PyG Data object (using cached graph structure if available)
        data = self._prepare_pyg_data(x, adj, edge_attr)
        
        # Store shapes for logging
        spatial_shapes = {}
        
        # Apply SpatialConv layers with fusion of AE features
        sigma = 0.5  # Fusion coefficient (same as original SDCN)
        
        # Layer 1: Process input features
        data.x = F.relu(self.proj1(x))
        node_edge_feat1 = self.spatial_conv1(data)
        h1 = node_edge_feat1[:original_nodes]  # Extract node features
        spatial_shapes['Layer 1'] = h1.shape
        
        # Layer 2: Fuse with AE features
        data.x = (1 - sigma) * h1 + sigma * tra1
        data.x = F.relu(self.proj2(data.x))
        node_edge_feat2 = self.spatial_conv2(data)
        h2 = node_edge_feat2[:original_nodes]
        spatial_shapes['Layer 2'] = h2.shape
        
        # Layer 3
        data.x = (1 - sigma) * h2 + sigma * tra2
        data.x = F.relu(self.proj3(data.x))
        node_edge_feat3 = self.spatial_conv3(data)
        h3 = node_edge_feat3[:original_nodes]
        spatial_shapes['Layer 3'] = h3.shape
        
        # Layer 4
        data.x = (1 - sigma) * h3 + sigma * tra3
        data.x = F.relu(self.proj4(data.x))
        node_edge_feat4 = self.spatial_conv4(data)
        h4 = node_edge_feat4[:original_nodes]
        spatial_shapes['Layer 4'] = h4.shape
        
        # Layer 5 (no activation for final layer)
        data.x = (1 - sigma) * h4 + sigma * z   # data.x.shape = [original_nodes, n_z]
        
        # Project node features: map each node's features from n_z to n_clusters
        projected_features = self.proj5(data.x)   # projected_features.shape = [original_nodes, n_clusters]
        
        # Create new Data object while maintaining original_nodes count
        updated_data = Data(
            x=projected_features,                
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            dist_feat=data.dist_feat,
            dist_feat_order=data.dist_feat_order,
            edge_to_edge_index=data.edge_to_edge_index
        )

        # Properly pass data to SpatialConv
        node_edge_feat5 = self.spatial_conv5(updated_data)
        h5 = node_edge_feat5[:original_nodes]
        spatial_shapes['Layer 5'] = h5.shape
        
        # Store shapes
        self.ae.layer_shapes['spatial'] = spatial_shapes
        
        # Apply softmax to get prediction
        predict = F.softmax(h5, dim=1)
        
        # Only print shapes once per epoch during training
        if self.training and self.current_epoch != self.last_logged_epoch:
            print(f"\nEpoch {self.current_epoch}")
            print("=" * 50)
            print("\nAutoencoder Architecture:")
            print("-" * 30)
            for layer_name, shape in self.ae.layer_shapes['autoencoder'].items():
                print(f"{layer_name}: {shape}")
            
            print("\nSpatial Architecture:")
            print("-" * 30)
            for layer_name, shape in spatial_shapes.items():
                print(f"{layer_name}: {shape}")
            print()
            self.last_logged_epoch = self.current_epoch
        
        # Calculate soft assignment (q) using Student's t-distribution
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return x_bar, q, predict, z, spatial_shapes


def target_distribution(q):
    """
    Calculate the target distribution p
    
    Args:
        q: Soft assignment (Student's t-distribution)
        
    Returns:
        p: Target distribution
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn_dlaa(dataset, args, edge_attr=None):
    """
    Train SDCN_DLAA model
    
    Args:
        dataset: Dataset object containing features and labels
        args: Arguments for training
        edge_attr: Edge features [num_edges, edge_dim]
    """
    
    # Check if edge_attr is provided, if not, create simple edge features
    if edge_attr is None:
        # Load KNN Graph to get number of edges
        adj = load_graph(args.name, args.k)
        edge_index, _ = dense_to_sparse(adj)
        num_edges = edge_index.size(1)
        
        # Create simple edge features (all ones)
        print(f"No edge features provided. Creating simple edge features with dimension {args.edge_dim}")
        edge_attr = torch.ones(num_edges, args.edge_dim)
    
    # Load KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.to(args.device)
    
    # Precompute graph structures for optimization
    print("Precomputing graph structures...")
    if adj.is_sparse:
        adj = adj.coalesce()
        edge_index = adj.indices()
    else:
        edge_index, _ = dense_to_sparse(adj)
    
    # Build edge-to-edge graph (once, not in every forward pass)
    num_edges = edge_index.size(1)
    
    # Build mapping from nodes to edges
    node_to_edges = defaultdict(list)
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        node_to_edges[src].append(i)
        node_to_edges[dst].append(i)
    
    # Build edge-to-edge connections
    print("Building edge-to-edge connections...")
    edge_to_edge_list = []
    max_edges_per_node = args.max_edges_per_node if hasattr(args, 'max_edges_per_node') else 10
    
    for node, connected_edges in node_to_edges.items():
        if len(connected_edges) > 1:
            if len(connected_edges) > max_edges_per_node:
                sampled_edges = random.sample(connected_edges, max_edges_per_node)
            else:
                sampled_edges = connected_edges
            
            for i in range(len(sampled_edges)):
                for j in range(i+1, len(sampled_edges)):
                    edge_i = sampled_edges[i]
                    edge_j = sampled_edges[j]
                    edge_to_edge_list.append([edge_i, edge_j])
                    edge_to_edge_list.append([edge_j, edge_i])
    
    if len(edge_to_edge_list) > 0:
        edge_to_edge_index = torch.tensor(edge_to_edge_list, dtype=torch.long).t().to(args.device)
    else:
        edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long).to(args.device)
    
    # Create model with precomputed graph structures
    model = SDCN_DLAA(
        500, 500, 2000, 2000, 500, 500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1.0,
        dropout=args.dropout,
        edge_dim=args.edge_dim,
        heads=4,
        max_edges_per_node=max_edges_per_node,
        precomputed_edge_index=edge_index,
        precomputed_edge_to_edge_index=edge_to_edge_index
    ).to(args.device)
    
    print(model)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Prepare data
    data = torch.Tensor(dataset.x).to(args.device)
    y = dataset.y
    
    # Initialize cluster centers using pretrained autoencoder
    # ---> Use no_grad here too for initialization inference <---
    model.eval() # Set model to eval mode for initialization inference
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    model.train() # Switch back to train mode

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    # Check if y has enough classes for evaluation metrics
    if len(np.unique(y)) > 1:
        eva(y, y_pred, 'pae')
    else:
        print(f"Initial clustering (pae) completed. Cluster distribution may not be diverse.")
        print(f"Initial y_pred counts: {np.bincount(y_pred)}")
    
    # Create a list to store results
    results = []
    
    # Training loop
    for epoch in range(60):
        # Update the current epoch
        model.current_epoch = epoch
        
        if epoch % 1 == 0:
            # ---> Set model to evaluation mode <---
            model.eval()
            # Evaluate the model
            try:
                # ---> Use torch.no_grad() for evaluation inference <---
                with torch.no_grad():
                    _, tmp_q, pred, _, _ = model(data, adj, edge_attr)

                # The rest of the calculations generally don't need gradients
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

                res1 = tmp_q.cpu().numpy().argmax(1)  # Q
                res2 = pred.data.cpu().numpy().argmax(1)  # Z
                res3 = p.data.cpu().numpy().argmax(1)  # P

                # Check if y has enough classes for evaluation metrics
                if len(np.unique(y)) > 1:
                    acc1, f1_1, nmi1, ari1 = eva(y, res1, f'{epoch}Q')
                    acc2, f1_2, nmi2, ari2 = eva(y, res2, f'{epoch}Z')
                    acc3, f1_3, nmi3, ari3 = eva(y, res3, f'{epoch}P')
                    # Save clustering results for each round
                    results.append([epoch, acc1, f1_1, nmi1, ari1, acc2, f1_2, nmi2, ari2, acc3, f1_3, nmi3, ari3])
                else:
                    # Handle case with insufficient classes in y
                    print(f"Epoch {epoch} evaluation skipped due to insufficient ground truth classes.")
                    # Append placeholders or skip appending
                    results.append([epoch] + [0] * 12) # Example: Append zeros

            except Exception as e:
                print(f"Epoch {epoch} evaluation error: {str(e)}")
                # ---> Ensure model returns to train mode even if eval fails <---
                model.train()
                continue # Skip to next epoch if evaluation fails
            finally:
                # ---> Switch model back to training mode AFTER evaluation try block <---
                model.train()
        
        # Forward pass (Training) - already wrapped in its own try-except
        # No changes needed here unless train mode was not set correctly
        try:
            # Ensure model is in train mode before forward pass for training
            model.train() # Redundant if correctly placed after eval, but safe

            x_bar, q, pred, _, _ = model(data, adj, edge_attr)

            # Calculate target distribution
            p = target_distribution(q.data)
            
            # Calculate loss
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, data)
            
            # Combined loss with the same weights as original SDCN
            loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, KL: {kl_loss.item():.4f}, CE: {ce_loss.item():.4f}, RE: {re_loss.item():.4f}")
        except Exception as e:
            print(f"Epoch {epoch} training error: {str(e)}")
            continue
    
    # Get final clustering results
    # ---> Use model.eval() and no_grad() for final inference <---
    model.eval()
    try:
        with torch.no_grad():
            _, _, final_pred, _, _ = model(data, adj, edge_attr)
        final_clusters = final_pred.data.cpu().numpy().argmax(1)
    except Exception as e:
        print(f"Error getting final clustering results: {str(e)}")
        # Fallback logic...
        if 'res2' in locals() and res2 is not None:
             final_clusters = res2
        elif len(results) > 0 and len(results[-1]) > 6 : # Check if previous eval results exist
             # Attempt to reconstruct from last valid 'res2' if possible (needs storing res2)
             # As a simple fallback, use the last recorded P prediction if available
             if 'res3' in locals() and res3 is not None:
                 final_clusters = res3
             else: # Last resort: zeros
                 final_clusters = np.zeros(len(dataset.x), dtype=int)
             print("Warning: Using fallback for final clustering results.")
        else:
            final_clusters = np.zeros(len(dataset.x), dtype=int)
            print("Warning: Using zeros for final clustering results due to errors.")
    
    # Save results
    column_names = ['Epoch', 'Acc_Q', 'F1_Q', 'NMI_Q', 'ARI_Q', 'Acc_Z', 'F1_Z', 'NMI_Z', 'ARI_Z', 'Acc_P', 'F1_P', 'NMI_P', 'ARI_P']
    if len(results) > 0 and len(results[0]) != len(column_names): # Adjust columns if only epoch was saved
        column_names = ['Epoch'] + [f'Metric_{i}' for i in range(len(results[0]) - 1)]

    results_df = pd.DataFrame(results, columns=column_names)
    results_df.to_csv('sdcn_dlaa_training_results.csv', index=False)

    print("Training complete. Results saved to 'sdcn_dlaa_training_results.csv'.")

    final_results_df = pd.DataFrame({'Node': np.arange(len(final_clusters)), 'Cluster': final_clusters})
    final_results_df.to_csv('sdcn_dlaa_final_cluster_results.csv', index=False)

    print("Final clustering results saved to 'sdcn_dlaa_final_cluster_results.csv'.")

    return model, results_df


class Logger(object):
    def __init__(self, filename="Default.log", terminal_mode="normal"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")  # 添加UTF-8编码
        self.terminal_mode = terminal_mode

    def write(self, message):
        # Always write everything to log file
        self.log.write(message)
        
        # For terminal, only show important information
        if self.terminal_mode == "minimal":
            # Only print to terminal if message contains important keywords
            if any(key in message for key in [
                'acc', 'nmi', 'ari', 'f1',  # Metrics
                'Training complete',         # Important status
                'Final clustering',          # Final results
                'use cuda',                  # Hardware info
                'Epoch'                      # Epoch progress
            ]):
                # Skip layer shape information even in epoch headers
                if not any(shape in message for shape in [
                    'Layer', 'Shape', 'Architecture'
                ]):
                    self.terminal.write(message)
        else:
            # Normal mode - print everything
            self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def train_sdcn_dlaa_custom(dataset, adj, args, edge_attr=None):
    """
    Optimized training function - for custom datasets
    
    Args:
        dataset: Dataset object containing features and labels
        adj: Adjacency matrix (torch sparse tensor)
        args: Training parameters
        edge_attr: Edge features [num_edges, edge_dim]
    """
    
    # Check if edge features are provided, if not create simple edge features
    if edge_attr is None:
        print("Edge features not provided, using randomly initialized edge features")
        num_edges = adj._nnz()
        edge_attr = torch.ones(num_edges, args.edge_dim).to(args.device)
    else:
        # Ensure edge features are on the correct device
        edge_attr = edge_attr.to(args.device)
    
    # Performance optimization: Preprocess edges into edge graph structure
    print("Performance optimization: Precomputing graph structure...")
    if adj.is_sparse:
        adj = adj.coalesce()
        edge_index = adj.indices()
    else:
        edge_index, _ = dense_to_sparse(adj)
        
    # Validate edge indices
    num_nodes = dataset.num_nodes
    max_index = edge_index.max().item()
    
    if max_index >= num_nodes:
        print(f"Warning: Edge indices contain out-of-range values (max: {max_index}, num_nodes: {num_nodes})")
        print(f"Filtering invalid edges...")
        
        valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        edge_index = edge_index[:, valid_edges_mask]
        # Update edge features
        if edge_attr is not None:
            edge_attr = edge_attr[valid_edges_mask]
            
        if edge_index.size(1) == 0:
            print("Error: No valid edges after filtering!")
            # Create minimal valid graph
            edge_index = torch.zeros((2, 1), dtype=torch.long).to(args.device)
            edge_index[0, 0] = 0
            edge_index[1, 0] = min(1, num_nodes-1)  # Connect to self if only 1 node
            
            # Update edge features
            if edge_attr is not None:
                edge_attr = torch.ones(1, args.edge_dim).to(args.device)
    
    # Build edge-to-edge connections
    print("Building edge-to-edge connections...")
    num_edges = edge_index.size(1)
    
    # Build node-to-edge mapping
    node_to_edges = defaultdict(list)
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        node_to_edges[src].append(i)
        node_to_edges[dst].append(i)
    
    # Create edge-to-edge connections
    edge_to_edge_list = []
    for node, connected_edges in node_to_edges.items():
        if len(connected_edges) > 1:
            # Random sampling when exceeding max edges per node
            if len(connected_edges) > args.max_edges_per_node:
                sampled_edges = random.sample(connected_edges, args.max_edges_per_node)
            else:
                sampled_edges = connected_edges
            
            # Connect all edge pairs sharing the same node
            for i in range(len(sampled_edges)):
                for j in range(i+1, len(sampled_edges)):
                    edge_i = sampled_edges[i]
                    edge_j = sampled_edges[j]
                    # Add bidirectional connections for undirected graph
                    edge_to_edge_list.append([edge_i, edge_j])
                    edge_to_edge_list.append([edge_j, edge_i])
    
    # Convert to tensor format
    if len(edge_to_edge_list) > 0:
        edge_to_edge_index = torch.tensor(edge_to_edge_list, dtype=torch.long).t().to(args.device)
    else:
        # Create empty tensor if no edge-to-edge connections
        edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long).to(args.device)
        
    print(f"Precomputation complete: node-to-node edges: {edge_index.shape[1]}, edge-to-edge connections: {edge_to_edge_index.shape[1]}")
    
    # Create model using precomputed graph structure
    model = SDCN_DLAA(
        500, 500, 2000, 2000, 500, 500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1.0,
        dropout=args.dropout,
        edge_dim=args.edge_dim,
        heads=args.heads,
        max_edges_per_node=args.max_edges_per_node,
        precomputed_edge_index=edge_index,
        precomputed_edge_to_edge_index=edge_to_edge_index
    ).to(args.device)
    
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    adj = adj.to(args.device)

    data = torch.Tensor(dataset.x).to(args.device)
    y = dataset.y

    # ---> Use no_grad here too for initialization inference <---
    model.eval() # Set model to eval mode for initialization inference
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    model.train() # Switch back to train mode


    # Perform initial clustering using K-means
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    # Ensure z is on CPU for KMeans
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    # Evaluate initial clustering results
    if len(np.unique(y)) > 1:  # If ground truth labels exist
        eva(y, y_pred, 'pae')
    else:
        print(f"Initial clustering complete. Number of clusters: {args.n_clusters}")

    results = []

    for epoch in range(60):
        model.current_epoch = epoch
        
        if epoch % 1 == 0:
            # ---> Set model to evaluation mode <---
            model.eval()
            try:
                # ---> Use torch.no_grad() for evaluation inference <---
                with torch.no_grad():
                     _, tmp_q, pred, _, _ = model(data, adj, edge_attr)

                # The rest of the calculations generally don't need gradients
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

                res1 = tmp_q.cpu().numpy().argmax(1)  # Q
                res2 = pred.data.cpu().numpy().argmax(1)  # Z
                res3 = p.data.cpu().numpy().argmax(1)  # P
                last_successful_res2 = res2 # Store the latest successful result

                if len(np.unique(y)) > 1: 
                    acc1, f1_1, nmi1, ari1 = eva(y, res1, f'{epoch}Q')
                    acc2, f1_2, nmi2, ari2 = eva(y, res2, f'{epoch}Z')
                    acc3, f1_3, nmi3, ari3 = eva(y, res3, f'{epoch}P')

                    results.append([epoch, acc1, f1_1, nmi1, ari1, acc2, f1_2, nmi2, ari2, acc3, f1_3, nmi3, ari3])
                else:
                    # Without labels, only save clustering results without computing evaluation metrics
                    cluster_distribution = np.bincount(res2)
                    print(f"Epoch {epoch}, Cluster distribution: {cluster_distribution}")
                    results.append([epoch] + [0] * 12)  # Placeholder padding
            except Exception as e:
                print(f"Epoch {epoch} Evaluation error: {str(e)}")
                 # ---> Ensure model returns to train mode even if eval fails <---
                model.train()
                continue # Skip to next epoch if evaluation fails
            finally:
                 # ---> Switch model back to training mode AFTER evaluation try block <---
                model.train()
        
        # Forward pass (training) - already wrapped in its own try-except
        try:
            # Ensure model is in train mode
            model.train()

            x_bar, q, pred, _, _ = model(data, adj, edge_attr)

            p = target_distribution(q.data)

            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, data)
 
            loss = 1.0 * kl_loss + 0.1 * ce_loss + re_loss
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, KL: {kl_loss.item():.4f}, CE: {ce_loss.item():.4f}, RE: {re_loss.item():.4f}")
        except Exception as e:
            print(f"Epoch {epoch} Training error: {str(e)}")
            continue
    
    # ---> Use model.eval() and no_grad() for final inference <---
    model.eval()
    try:
        with torch.no_grad():
            _, _, final_pred, _, _ = model(data, adj, edge_attr)
        final_clusters = final_pred.data.cpu().numpy().argmax(1)
    except Exception as e:
        print(f"Error getting final clustering results: {str(e)}")
        # If error occurs, use last successful clustering results
        if last_successful_res2 is not None:
            print("Warning: Using last successful evaluation result for final clusters.")
            final_clusters = last_successful_res2
        else:
            print("Warning: Using zeros for final clusters due to errors.")
            final_clusters = np.zeros(dataset.num_nodes, dtype=int)
    
    # Save results
    column_names = ['Epoch', 'Acc_Q', 'F1_Q', 'NMI_Q', 'ARI_Q', 'Acc_Z', 'F1_Z', 'NMI_Z', 'ARI_Z', 'Acc_P', 'F1_P', 'NMI_P', 'ARI_P']
    # Handle case where no results were appended if all evaluations failed early
    if not results:
         print("Warning: No evaluation results were recorded during training.")
         # Optionally create an empty DataFrame or handle as needed
         results_df = pd.DataFrame(columns=column_names)
    elif len(results[0]) != len(column_names): # Adjust columns if only epoch was saved
        column_names = ['Epoch'] + [f'Metric_{i}' for i in range(len(results[0]) - 1)]
        results_df = pd.DataFrame(results, columns=column_names)
    else:
        results_df = pd.DataFrame(results, columns=column_names)

    # Use specific filenames if running hiddensize test, otherwise use defaults
    if hasattr(args, 'hs1'): # Check if hiddensize args exist
        results_filename = f'sdcn_dlaa_hiddensize_training_results_hs{args.hs1}-{args.hs2}-{args.hs3}_heads{args.heads}.csv'
        final_clusters_filename = f'sdcn_dlaa_hiddensize_final_clusters_hs{args.hs1}-{args.hs2}-{args.hs3}_heads{args.heads}.csv'
    else:
        results_filename = 'sdcn_dlaa_training_results.csv'
        final_clusters_filename = 'sdcn_dlaa_final_cluster_results.csv'

    results_df.to_csv(results_filename, index=False)
    print(f"Training completed. Results saved to '{results_filename}'.")
   
    final_results_df = pd.DataFrame({'NodeID': np.arange(len(final_clusters)), 'ClusterID': final_clusters})
    final_results_df.to_csv(final_clusters_filename, index=False)
    print(f"Final clustering results saved to '{final_clusters_filename}'.")

    return model, results_df, final_clusters


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/sdcn_dlaa_run_{timestamp}.txt'
    
    # Redirect stdout to both console and file, with minimal terminal output
    sys.stdout = Logger(log_filename, terminal_mode="minimal")
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='train SDCN_DLAA with optimized SpatialConv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='reut')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--edge_dim', type=int, default=None, help='Dimension of edge features. If None, will use n_input')
    parser.add_argument('--use_edge_attr', action='store_true', help='Use edge attributes from dataset if available')
    parser.add_argument('--max_edges_per_node', type=int, default=10, help='Maximum number of edges to consider per node for edge-to-edge connections')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)
    
    # Check if edge attributes are available in the dataset
    edge_attr = None
    if hasattr(dataset, 'edge_attr') and args.use_edge_attr:
        edge_attr = dataset.edge_attr
        if args.edge_dim is None:
            args.edge_dim = edge_attr.shape[1]
    
    # Set dataset-specific parameters
    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256
    
    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561
    
    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000
    
    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870
    
    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334
    
    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703
    
    # If edge_dim is still None, set it to n_input
    if args.edge_dim is None:
        args.edge_dim = args.n_input
        print(f"Setting edge_dim to n_input: {args.edge_dim}")
    
    print(args)
    
    # Train the model
    model, results = train_sdcn_dlaa(dataset, args, edge_attr)