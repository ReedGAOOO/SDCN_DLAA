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
print(f"DEBUG: Type of eva right after import: {type(eva)}") # Debug print
import sys
import os
import traceback
from datetime import datetime

# Import HeteroSpatialConv from the new DLAA file
from DLAA_NEW_hetero import HeteroSpatialConv # Changed import


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
    
    Version using HeteroSpatialConv to avoid N+E concatenation.
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
        # Ensure edge_dim is set correctly, default to n_input if None
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
        
        # HeteroSpatialConv layers replacing SpatialConv
        # Note: hidden_size matches the corresponding AE encoder layer output
        self.spatial_conv1 = HeteroSpatialConv(n_enc_1, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        self.spatial_conv2 = HeteroSpatialConv(n_enc_2, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        self.spatial_conv3 = HeteroSpatialConv(n_enc_3, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        self.spatial_conv4 = HeteroSpatialConv(n_z, edge_dim=self.edge_dim, dropout=dropout, heads=heads)
        # Layer 5 input/output is n_clusters
        self.spatial_conv5 = HeteroSpatialConv(n_clusters, edge_dim=self.edge_dim, dropout=dropout, heads=heads) 
        
        # Projection layers to match dimensions between layers
        self.proj1 = nn.Linear(n_input, n_enc_1)
        self.proj2 = nn.Linear(n_enc_1, n_enc_2)
        self.proj3 = nn.Linear(n_enc_2, n_enc_3)
        self.proj4 = nn.Linear(n_enc_3, n_z)
        self.proj5 = nn.Linear(n_z, n_clusters) # Projects final node features to cluster space
        
        # Cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
        # Add edge feature projection layer for initial edge features (if needed by _prepare_pyg_data)
        # This might be redundant if HeteroSpatialConv handles edge_dim internally
        self.initial_edge_proj = None
        if edge_dim is not None and edge_dim != self.edge_dim: # Check against self.edge_dim
             print(f"Warning: Initializing initial_edge_proj, but HeteroSpatialConv might handle this. Check logic.")
             self.initial_edge_proj = nn.Linear(edge_dim, self.edge_dim)


    def _prepare_pyg_data(self, x, adj, edge_attr, max_edges_per_node=10):
        """
        Optimized method to prepare PyG Data object from node features and adjacency matrix
        Caches graph structure to avoid redundant computation.
        Ensures edge_attr matches self.edge_dim.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
            edge_attr: Edge features [num_edges, original_edge_dim]
            max_edges_per_node: Maximum number of edges to consider per node
            
        Returns:
            data: PyG Data object with edge_attr matching self.edge_dim
        """
        num_nodes = x.size(0)
        
        if num_nodes < 4:
            raise ValueError(f"节点数量必须大于3，当前为{num_nodes}")
            
        if not torch.is_floating_point(x) or torch.isnan(x).any():
            raise ValueError("节点特征包含无效值(NaN或非浮点数)")
            
        # Handle edge_attr: Ensure it exists and matches self.edge_dim
        num_expected_edges = 0 # Will be determined later
        if self.precomputed_edge_index is not None:
             num_expected_edges = self.precomputed_edge_index.size(1)
        elif adj is not None:
             if adj.is_sparse:
                 num_expected_edges = adj._nnz()
             else:
                 # Estimate from non-zero elements if dense
                 num_expected_edges = (adj != 0).sum().item() 
                 # This might be inaccurate if adj is not binary

        processed_edge_attr = None
        if edge_attr is not None:
            if edge_attr.size(0) != num_expected_edges and num_expected_edges > 0:
                 print(f"Warning: Provided edge_attr rows ({edge_attr.size(0)}) do not match expected edges ({num_expected_edges}). Will attempt to use.")
            
            # Project edge_attr if its dimension doesn't match self.edge_dim
            if edge_attr.size(1) != self.edge_dim:
                if self.initial_edge_proj is None:
                    # Dynamically create projection layer if needed and not present
                    print(f"Dynamically creating initial_edge_proj: {edge_attr.size(1)} -> {self.edge_dim}")
                    self.initial_edge_proj = nn.Linear(edge_attr.size(1), self.edge_dim).to(x.device) # Ensure on correct device
                processed_edge_attr = self.initial_edge_proj(edge_attr)
            else:
                processed_edge_attr = edge_attr
        else:
            # If no edge_attr provided, create dummy ones with the correct dimension
            if num_expected_edges > 0:
                 print(f"Creating dummy edge_attr with dimension {self.edge_dim}")
                 processed_edge_attr = torch.ones(num_expected_edges, self.edge_dim).to(x.device)
            else:
                 # Handle case with no edges
                 processed_edge_attr = torch.empty(0, self.edge_dim).to(x.device)


        # Check if we already have precomputed graph structures from initialization
        if self.precomputed_edge_index is not None and self.precomputed_edge_to_edge_index is not None:
            max_node_idx = self.precomputed_edge_index.max().item()
            if max_node_idx >= num_nodes:
                print(f"警告：预计算的边索引({max_node_idx})超出了当前节点数量({num_nodes})，重新计算中...")
                self.precomputed_edge_index = None
                self.precomputed_edge_to_edge_index = None
            else:
                # Ensure processed_edge_attr matches precomputed edge_index size
                if processed_edge_attr.size(0) != self.precomputed_edge_index.size(1):
                     print(f"Warning: Corrected processed_edge_attr size ({processed_edge_attr.size(0)}) does not match precomputed_edge_index size ({self.precomputed_edge_index.size(1)}). Recreating dummy attributes.")
                     processed_edge_attr = torch.ones(self.precomputed_edge_index.size(1), self.edge_dim).to(x.device)

                data = Data(
                    x=x,
                    edge_index=self.precomputed_edge_index,
                    edge_attr=processed_edge_attr, # Use processed_edge_attr
                    dist_feat=processed_edge_attr, # Use processed_edge_attr
                    dist_feat_order=processed_edge_attr, # Use processed_edge_attr
                    edge_to_edge_index=self.precomputed_edge_to_edge_index
                )
                return data
        
        # Create a cache key
        if adj.is_sparse:
            adj_id = f"{adj._indices().sum().item()}_{adj._values().sum().item()}"
        else:
            adj_id = f"{adj.sum().item()}"
        cache_key = f"{adj_id}_{max_edges_per_node}_{num_nodes}_{x.size(1)}"
        
        # Check cache
        if cache_key in self.graph_cache:
            cached = self.graph_cache[cache_key]
            max_node_idx = cached['edge_index'].max().item()
            if max_node_idx >= num_nodes:
                print(f"警告：缓存的边索引({max_node_idx})超出了当前节点数量({num_nodes})，重新计算中...")
                self.graph_cache.pop(cache_key)
            else:
                 # Ensure processed_edge_attr matches cached edge_index size
                 if processed_edge_attr.size(0) != cached['edge_index'].size(1):
                     print(f"Warning: Corrected processed_edge_attr size ({processed_edge_attr.size(0)}) does not match cached edge_index size ({cached['edge_index'].size(1)}). Recreating dummy attributes.")
                     processed_edge_attr = torch.ones(cached['edge_index'].size(1), self.edge_dim).to(x.device)

                 data = Data(
                    x=x,
                    edge_index=cached['edge_index'],
                    edge_attr=processed_edge_attr, # Use processed_edge_attr
                    dist_feat=processed_edge_attr, # Use processed_edge_attr
                    dist_feat_order=processed_edge_attr, # Use processed_edge_attr
                    edge_to_edge_index=cached['edge_to_edge_index']
                 )
                 return data
            
        # If not cached, compute the graph structure
        print("Computing graph structure (first time)...")
        if adj.is_sparse:
            adj = adj.coalesce()
            edge_index = adj.indices()
        else:
            edge_index, _ = dense_to_sparse(adj)
        
        # Validate edge indices
        max_index = edge_index.max().item()
        if max_index >= num_nodes:
            print(f"Warning: Edge index contains indices ({max_index}) > num_nodes ({num_nodes}). Filtering...")
            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_edges_mask]
            # Filter edge_attr accordingly
            if processed_edge_attr is not None and processed_edge_attr.size(0) == valid_edges_mask.size(0):
                 processed_edge_attr = processed_edge_attr[valid_edges_mask]
            elif processed_edge_attr is not None:
                 print(f"Warning: Could not filter processed_edge_attr due to size mismatch.")
                 # Recreate dummy attributes if filtering failed
                 processed_edge_attr = torch.ones(edge_index.size(1), self.edge_dim).to(x.device)


            if edge_index.size(1) == 0:
                print("Error: No valid edges remain after filtering!")
                edge_index = torch.zeros((2, 1), dtype=torch.long).to(x.device)
                edge_index[0, 0] = 0
                edge_index[1, 0] = min(1, num_nodes-1)
                processed_edge_attr = torch.ones(1, self.edge_dim).to(x.device) # Create dummy for the single edge
        
        num_edges = edge_index.size(1)

        # Ensure processed_edge_attr matches the final edge_index size
        if processed_edge_attr is None or processed_edge_attr.size(0) != num_edges:
             print(f"Warning: Final processed_edge_attr size mismatch ({processed_edge_attr.size(0) if processed_edge_attr is not None else 'None'}) vs edge_index size ({num_edges}). Recreating dummy attributes.")
             processed_edge_attr = torch.ones(num_edges, self.edge_dim).to(x.device)

        
        # Create edge-to-edge graph efficiently
        print("Building edge-to-edge graph...")
        node_to_edges = defaultdict(list)
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            node_to_edges[src.item()].append(i)
            node_to_edges[dst.item()].append(i)
        
        edge_to_edge_list = []
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
            edge_to_edge_index = torch.tensor(edge_to_edge_list, dtype=torch.long).t().to(x.device)
        else:
            edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long).to(x.device)
        
        # Store in cache
        self.graph_cache[cache_key] = {
            'edge_index': edge_index,
            'edge_to_edge_index': edge_to_edge_index
        }
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=processed_edge_attr, # Use processed_edge_attr
            dist_feat=processed_edge_attr, # Use processed_edge_attr
            dist_feat_order=processed_edge_attr, # Use processed_edge_attr
            edge_to_edge_index=edge_to_edge_index
        )
        
        # Store as precomputed values
        self.precomputed_edge_index = edge_index
        self.precomputed_edge_to_edge_index = edge_to_edge_index
        
        return data

    def forward(self, x, adj, edge_attr=None):
        """
        Forward pass using HeteroSpatialConv.
        
        Args:
            x: Node features [num_nodes, n_input]
            adj: Adjacency matrix
            edge_attr: Edge features [num_edges, original_edge_dim]
            
        Returns:
            x_bar: Reconstructed features
            q: Soft assignment
            predict: Cluster prediction
            z: Latent representation (from AE)
            spatial_shapes: Dictionary of layer shapes
        """
        original_nodes = x.size(0)
        
        # Get autoencoder outputs
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        # Prepare PyG Data object (handles caching and edge_attr processing)
        # Pass original edge_attr, _prepare_pyg_data handles projection/dummy creation
        pyg_data = self._prepare_pyg_data(x, adj, edge_attr, self.max_edges_per_node) 
        
        spatial_shapes = {}
        sigma = 0.5
        
        # --- Layer 1 ---
        h0 = F.relu(self.proj1(x)) 
        pyg_data.x = h0 # Set initial node features for the layer
        # HeteroSpatialConv expects pyg_data with x, edge_index, dist_feat, dist_feat_order, edge_to_edge_index
        # dist_feat and dist_feat_order are set to the processed edge_attr in _prepare_pyg_data
        x1, edge_feat1 = self.spatial_conv1(pyg_data) # Returns updated node and edge features
        spatial_shapes['Layer 1 Node'] = x1.shape
        spatial_shapes['Layer 1 Edge'] = edge_feat1.shape # Log edge shapes too

        # --- Layer 2 ---
        h1_fused = (1 - sigma) * x1 + sigma * tra1 # Fuse node features
        h1_fused_proj = F.relu(self.proj2(h1_fused))
        pyg_data.x = h1_fused_proj # Update node features for the next layer
        # Reuse edge_index, edge_to_edge_index, dist_feat, dist_feat_order from pyg_data
        x2, edge_feat2 = self.spatial_conv2(pyg_data)
        spatial_shapes['Layer 2 Node'] = x2.shape
        spatial_shapes['Layer 2 Edge'] = edge_feat2.shape
        
        # --- Layer 3 ---
        h2_fused = (1 - sigma) * x2 + sigma * tra2
        h2_fused_proj = F.relu(self.proj3(h2_fused))
        pyg_data.x = h2_fused_proj
        x3, edge_feat3 = self.spatial_conv3(pyg_data)
        spatial_shapes['Layer 3 Node'] = x3.shape
        spatial_shapes['Layer 3 Edge'] = edge_feat3.shape

        # --- Layer 4 (Latent Space) ---
        h3_fused = (1 - sigma) * x3 + sigma * tra3 
        h3_fused_proj = F.relu(self.proj4(h3_fused)) # Output dim n_z
        pyg_data.x = h3_fused_proj 
        # spatial_conv4 input node dim is n_z, output node dim is n_z
        x4, edge_feat4 = self.spatial_conv4(pyg_data) 
        spatial_shapes['Layer 4 Node'] = x4.shape
        spatial_shapes['Layer 4 Edge'] = edge_feat4.shape

        # --- Layer 5 (Cluster Prediction) ---
        h4_fused = (1 - sigma) * x4 + sigma * z # Fuse with AE latent z (dim n_z)
        # Project fused features to n_clusters dimension BEFORE the final spatial layer
        h4_fused_proj = self.proj5(h4_fused) # Dim becomes n_clusters (NO ReLU here usually)
        pyg_data.x = h4_fused_proj # Node features are now n_clusters dim
        # spatial_conv5 input node dim is n_clusters, output node dim is n_clusters
        x5, edge_feat5 = self.spatial_conv5(pyg_data) 
        spatial_shapes['Layer 5 Node'] = x5.shape
        spatial_shapes['Layer 5 Edge'] = edge_feat5.shape
        
        # Final prediction comes from the updated node features of the last layer (x5)
        predict = F.softmax(x5, dim=1) 

        # --- Logging Shapes ---
        self.ae.layer_shapes['spatial'] = spatial_shapes
        if self.training and self.current_epoch != self.last_logged_epoch:
            print(f"\nEpoch {self.current_epoch}")
            print("=" * 50)
            print("\nAutoencoder Architecture:")
            print("-" * 30)
            for layer_name, shape in self.ae.layer_shapes['autoencoder'].items():
                print(f"{layer_name}: {shape}")
            
            print("\nSpatial Architecture (Hetero):") # Indicate Hetero
            print("-" * 30)
            for layer_name, shape in spatial_shapes.items():
                print(f"{layer_name}: {shape}")
            print()
            self.last_logged_epoch = self.current_epoch
        
        # --- Clustering Soft Assignment (q) ---
        # Uses AE's latent 'z', not the GNN output
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        # Return values needed for loss calculation
        # Return original z from AE for clustering loss
        return x_bar, q, predict, z, spatial_shapes 


def target_distribution(q):
    """
    Calculate the target distribution p
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn_dlaa(dataset, adj, args, edge_attr=None): # Added adj parameter
    """
    Train SDCN_DLAA model using HeteroSpatialConv
    """
    # adj is now passed as an argument, no need to load it here.
    # Ensure adj is on the correct device (already handled in test script, but good practice)
    
    # Determine expected edge count from the provided adj
    if adj.is_sparse:
        num_expected_edges = adj._nnz()
    else:
        num_expected_edges = (adj != 0).sum().item()

    # Check if edge_attr is provided, if not, create simple edge features
    processed_edge_attr = None
    if edge_attr is not None:
         if edge_attr.size(0) != num_expected_edges:
              print(f"Warning: Provided edge_attr rows ({edge_attr.size(0)}) do not match expected edges from loaded graph ({num_expected_edges}).")
         # We will let _prepare_pyg_data handle projection if needed
         processed_edge_attr = edge_attr.to(args.device) 
    else:
        # Create simple edge features (all ones) if none provided
        print(f"No edge features provided. Creating simple edge features with dimension {args.edge_dim}")
        processed_edge_attr = torch.ones(num_expected_edges, args.edge_dim).to(args.device)
    
    # Precompute graph structures (edge_index, edge_to_edge_index)
    # This part remains largely the same, as it prepares inputs for the model
    print("Precomputing graph structures...")
    if adj.is_sparse:
        adj_coalesced = adj.coalesce() # Use a different variable name
        edge_index = adj_coalesced.indices()
    else:
        edge_index, _ = dense_to_sparse(adj)
    
    num_edges = edge_index.size(1)
    
    # Build mapping from nodes to edges
    node_to_edges = defaultdict(list)
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        # Ensure indices are within bounds (although _prepare_pyg_data also checks)
        if src < dataset.x.shape[0] and dst < dataset.x.shape[0]:
             node_to_edges[src].append(i)
             node_to_edges[dst].append(i)
        else:
             print(f"Warning: Skipping edge {i} with invalid indices ({src}, {dst}) during precomputation.")

    
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
        edge_dim=args.edge_dim, # Pass the target edge_dim
        heads=args.heads,
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
    model.eval() 
    with torch.no_grad():
        # Pass processed_edge_attr for initialization if needed by forward pass logic
        # Note: AE part doesn't use adj or edge_attr, so they can be None here
        _, _, _, _, z = model.ae(data) 
    model.train() 

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    if len(np.unique(y)) > 1:
        eva(y, y_pred, 'pae')
    else:
        print(f"Initial clustering (pae) completed. Cluster distribution may not be diverse.")
        print(f"Initial y_pred counts: {np.bincount(y_pred)}")
    
    # Create a list to store results
    results = []
    
    # Training loop
    for epoch in range(60):
        model.current_epoch = epoch
        
        if epoch % 1 == 0:
            model.eval()
            try:
                with torch.no_grad():
                    # Pass processed_edge_attr during evaluation
                    _, tmp_q, pred, _, _ = model(data, adj, processed_edge_attr) 

                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

                res1 = tmp_q.cpu().numpy().argmax(1)  # Q
                res2 = pred.data.cpu().numpy().argmax(1)  # Z (Prediction from GNN)
                res3 = p.data.cpu().numpy().argmax(1)  # P
                
                if len(np.unique(y)) > 1:
                    acc1, f1_1, nmi1, ari1 = eva(y, res1, f'{epoch}Q') # Reverted to eva
                    acc2, f1_2, nmi2, ari2 = eva(y, res2, f'{epoch}Z') # Reverted to eva
                    acc3, f1_3, nmi3, ari3 = eva(y, res3, f'{epoch}P') # Reverted to eva
                    results.append([epoch, acc1, f1_1, nmi1, ari1, acc2, f1_2, nmi2, ari2, acc3, f1_3, nmi3, ari3])
                else:
                    print(f"Epoch {epoch} evaluation skipped due to insufficient ground truth classes.")
                    results.append([epoch] + [0] * 12) 

            except Exception as e:
                traceback.print_exc() # Print full traceback
                print(f"Epoch {epoch} evaluation error: {str(e)}")
                model.train() # Ensure return to train mode
                continue 
            finally:
                model.train() # Ensure return to train mode
        
        # Forward pass (Training)
        try:
            model.train() 
            # Pass processed_edge_attr during training
            x_bar, q, pred, _, _ = model(data, adj, processed_edge_attr) 

            p = target_distribution(q.data)
            
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean') # Use GNN prediction 'pred'
            re_loss = F.mse_loss(x_bar, data)
            
            # Combined loss
            loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, KL: {kl_loss.item():.4f}, CE: {ce_loss.item():.4f}, RE: {re_loss.item():.4f}")
            traceback.print_exc() # Print full traceback
        except Exception as e:
            print(f"Epoch {epoch} training error: {str(e)}")
            continue
    
    # Get final clustering results
    model.eval()
    final_clusters = None # Initialize
    try:
        with torch.no_grad():
             # Pass processed_edge_attr for final inference
            _, _, final_pred, _, _ = model(data, adj, processed_edge_attr) 
        final_clusters = final_pred.data.cpu().numpy().argmax(1)
        traceback.print_exc() # Print full traceback
    except Exception as e:
        print(f"Error getting final clustering results: {str(e)}")
        # Fallback logic...
        if 'res2' in locals() and res2 is not None:
             final_clusters = res2
             print("Warning: Using last successful evaluation result (res2) for final clusters.")
        elif len(results) > 0 and len(results[-1]) > 6 : 
             if 'res3' in locals() and res3 is not None:
                 final_clusters = res3
                 print("Warning: Using last successful evaluation result (res3) for final clusters.")
             else: 
                 final_clusters = np.zeros(len(dataset.x), dtype=int)
                 print("Warning: Using zeros for final clustering results due to errors and lack of fallback.")
        else:
            final_clusters = np.zeros(len(dataset.x), dtype=int)
            print("Warning: Using zeros for final clustering results due to errors.")
    
    # Save results
    results_filename = 'sdcn_dlaa_hetero_training_results.csv' # New filename
    final_clusters_filename = 'sdcn_dlaa_hetero_final_cluster_results.csv' # New filename

    column_names = ['Epoch', 'Acc_Q', 'F1_Q', 'NMI_Q', 'ARI_Q', 'Acc_Z', 'F1_Z', 'NMI_Z', 'ARI_Z', 'Acc_P', 'F1_P', 'NMI_P', 'ARI_P']
    if not results:
         print("Warning: No evaluation results were recorded during training.")
         results_df = pd.DataFrame(columns=column_names)
    elif len(results[0]) != len(column_names): 
        column_names = ['Epoch'] + [f'Metric_{i}' for i in range(len(results[0]) - 1)]
        results_df = pd.DataFrame(results, columns=column_names)
    else:
        results_df = pd.DataFrame(results, columns=column_names)

    results_df.to_csv(results_filename, index=False)
    print(f"Training complete. Results saved to '{results_filename}'.")

    if final_clusters is not None:
         final_results_df = pd.DataFrame({'Node': np.arange(len(final_clusters)), 'Cluster': final_clusters})
         final_results_df.to_csv(final_clusters_filename, index=False)
         print(f"Final clustering results saved to '{final_clusters_filename}'.")
    else:
         print("Could not save final clustering results as they are None.")


    return model, results_df


class Logger(object):
    def __init__(self, filename="Default.log", terminal_mode="normal"):
        self.terminal = sys.stdout
        # Ensure logs directory exists
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
             os.makedirs(log_dir)
        self.log = open(filename, "a", encoding="utf-8")
        self.terminal_mode = terminal_mode

    def write(self, message):
        self.log.write(message)
        if self.terminal_mode == "minimal":
            if any(key in message for key in ['acc', 'nmi', 'ari', 'f1', 'Training complete', 'Final clustering', 'use cuda', 'Epoch', 'Error', 'Warning']):
                 # Skip detailed layer shape info even in epoch headers
                 if not any(shape_key in message for shape_key in ['Layer', 'Shape', 'Architecture', 'Encoder', 'Decoder', 'Latent', 'Output', 'Node', 'Edge']):
                      self.terminal.write(message)
        else:
            self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Note: train_sdcn_dlaa_custom is removed as train_sdcn_dlaa now handles the logic


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use a specific name for hetero version log
    log_filename = f'logs/sdcn_dlaa_hetero_run_{timestamp}.txt' 
    
    # Redirect stdout
    sys.stdout = Logger(log_filename, terminal_mode="minimal")
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='train SDCN_DLAA with HeteroSpatialConv', # Updated description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='reut')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    # pretrain_path is not strictly needed if AE is trained end-to-end
    # parser.add_argument('--pretrain_path', type=str, default='pkl') 
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=4)
    # edge_dim will be determined from data or set to n_input
    parser.add_argument('--edge_dim', type=int, default=None, help='Target dimension for edge features after potential projection. If None, will use n_input.') 
    parser.add_argument('--use_edge_attr', action='store_true', help='Use edge attributes from dataset if available')
    parser.add_argument('--max_edges_per_node', type=int, default=10, help='Max edges per node for edge-to-edge graph')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    # args.pretrain_path = 'data/{}.pkl'.format(args.name) # Not used currently
    dataset = load_data(args.name)
    
    # Check for edge attributes in the dataset
    edge_attr_from_data = None
    if hasattr(dataset, 'edge_attr') and args.use_edge_attr:
        edge_attr_from_data = dataset.edge_attr
        print(f"Loaded edge attributes from dataset with shape: {edge_attr_from_data.shape}")
        # If edge_dim is not specified, infer from loaded attributes
        if args.edge_dim is None:
             args.edge_dim = edge_attr_from_data.shape[1]
             print(f"Inferred edge_dim from dataset: {args.edge_dim}")

    
    # Set dataset-specific parameters (same as before)
    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256
    elif args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561
    elif args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000
    elif args.name == 'acm':
        args.k = None # Adjacency matrix is expected in the dataset for these
        args.n_clusters = 3
        args.n_input = 1870
    elif args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334
    elif args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703
    
    # If edge_dim is still None after checking data and defaults, set it to n_input
    if args.edge_dim is None:
        args.edge_dim = args.n_input
        print(f"Setting edge_dim to n_input: {args.edge_dim}")
    
    print(args)
    
    # Train the model, passing edge_attr_from_data
    model, results_df = train_sdcn_dlaa(dataset, args, edge_attr=edge_attr_from_data)