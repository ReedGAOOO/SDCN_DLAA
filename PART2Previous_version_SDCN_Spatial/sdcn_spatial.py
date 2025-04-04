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
from SDCN_ORIGINAL.evaluation import eva
import sys
import os
from datetime import datetime

# Import SpatialConv from DLAA
from Archive3.DLAA import SpatialConv


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


class SDCN_Spatial(nn.Module):
    """
    SDCN model with SpatialConv layers replacing GNNLayers
    """
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                n_input, n_z, n_clusters, v=1, dropout=0.2, heads=4, edge_dim=None, 
                max_edges_per_node=10):
        super(SDCN_Spatial, self).__init__()
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
        self.max_edges_per_node = max_edges_per_node  # 每个节点最大考虑的边数

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
        
        # No longer needed: spatial embedding layer
        
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
        Prepare PyG Data object from node features and adjacency matrix
        
        Args:
            x: Node features [num_nodes, feature_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
            edge_attr: Edge features [num_edges, edge_dim]
            max_edges_per_node: Maximum number of edges to consider per node for edge-to-edge connections
        Returns:
            data: PyG Data object
        """
        # Convert adjacency matrix to edge_index
        # Check if adj is already a sparse tensor
        if adj.is_sparse:
            # If it's sparse, directly get indices
            adj = adj.coalesce()
            edge_index = adj.indices()
        else:
            edge_index, _ = dense_to_sparse(adj)
        
        # 验证边索引是否在有效范围内
        num_nodes = x.size(0)
        max_index = edge_index.max().item()
        
        if max_index >= num_nodes:
            print(f"Warning: Edge index contains indices ({max_index}) that exceed the number of nodes ({num_nodes})")
            print(f"Filtering edges to only include those with valid node indices...")
            
            # 过滤边，只保留有效的节点索引
            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_edges_mask]
            
            if edge_index.size(1) == 0:
                print("Error: No valid edges remain after filtering!")
                # 创建一个最小有效的边索引以避免错误
                edge_index = torch.zeros((2, 1), dtype=torch.long).to(x.device)
                edge_index[0, 0] = 0
                edge_index[1, 0] = min(1, num_nodes-1)  # 将节点0连接到节点1（如果只有1个节点则连接到自身）
        
        # 更新边数量
        num_edges = edge_index.size(1)
        
        # Process edge features
        dist_feat = edge_attr
        
        # Apply initial projection if needed
        if self.initial_edge_proj is not None:
            dist_feat = self.initial_edge_proj(dist_feat)
        
        # Create edge-to-edge graph more efficiently
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
                # 如果节点连接的边数超过阈值，则随机采样限制数量
                if len(connected_edges) > max_edges_per_node:
                    # 使用随机采样来限制边的数量
                    import random
                    sampled_edges = random.sample(connected_edges, max_edges_per_node)
                else:
                    sampled_edges = connected_edges
                
                # Connect all pairs of edges that share this node (使用采样后的边列表)
                for i in range(len(sampled_edges)):
                    for j in range(i+1, len(sampled_edges)):
                        edge_i = sampled_edges[i]
                        edge_j = sampled_edges[j]
                        # Add both directions for undirected graph
                        edge_to_edge_list.append([edge_i, edge_j])
                        edge_to_edge_list.append([edge_j, edge_i])
        
        # 转换为稀疏张量表示
        if len(edge_to_edge_list) > 0:
            edge_to_edge_index = torch.tensor(edge_to_edge_list, dtype=torch.long).t().to(x.device)
        else:
            # 如果没有边对边连接，创建一个空的边索引张量
            edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long).to(x.device)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=dist_feat,  # Edge features
            dist_feat=dist_feat,  # Distance features for node-node graph
            dist_feat_order=dist_feat,  # Distance features for edge-edge graph
            edge_to_edge_index=edge_to_edge_index  # Edge-to-edge connectivity
        )
        
        return data

    def forward(self, x, adj, edge_attr=None):
        # Get autoencoder outputs
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        # Prepare PyG Data object
        data = self._prepare_pyg_data(x, adj, edge_attr)
        
        # Store shapes for logging
        spatial_shapes = {}
        
        # Apply SpatialConv layers with fusion of AE features
        sigma = 0.5  # Fusion coefficient (same as original SDCN)
        
        # Layer 1: Process input features
        data.x = F.relu(self.proj1(x))
        node_edge_feat1 = self.spatial_conv1(data)
        h1 = node_edge_feat1[:x.size(0)]  # Extract node features
        spatial_shapes['Layer 1'] = h1.shape
        
        # Layer 2: Fuse with AE features
        data.x = (1 - sigma) * h1 + sigma * tra1
        data.x = F.relu(self.proj2(data.x))
        node_edge_feat2 = self.spatial_conv2(data)
        h2 = node_edge_feat2[:x.size(0)]
        spatial_shapes['Layer 2'] = h2.shape
        
        # Layer 3
        data.x = (1 - sigma) * h2 + sigma * tra2
        data.x = F.relu(self.proj3(data.x))
        node_edge_feat3 = self.spatial_conv3(data)
        h3 = node_edge_feat3[:x.size(0)]
        spatial_shapes['Layer 3'] = h3.shape
        
        # Layer 4
        data.x = (1 - sigma) * h3 + sigma * tra3
        data.x = F.relu(self.proj4(data.x))
        node_edge_feat4 = self.spatial_conv4(data)
        h4 = node_edge_feat4[:x.size(0)]
        spatial_shapes['Layer 4'] = h4.shape
        
        # Layer 5 (no activation for final layer)
        data.x = (1 - sigma) * h4 + sigma * z
        data.x = self.proj5(data.x)
        node_edge_feat5 = self.spatial_conv5(data)
        h5 = node_edge_feat5[:x.size(0)]
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


def train_sdcn_spatial(dataset, args, edge_attr=None):
    """
    Train SDCN_Spatial model
    
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
    
    # Create model
    model = SDCN_Spatial(
        500, 500, 2000, 2000, 500, 500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1.0,
        dropout=args.dropout,
        edge_dim=args.edge_dim,
        heads=4,
        max_edges_per_node=args.max_edges_per_node if hasattr(args, 'max_edges_per_node') else 10
    ).to(args.device)
    
    print(model)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Load KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.to(args.device)
    
    # Prepare data
    data = torch.Tensor(dataset.x).to(args.device)
    y = dataset.y
    
    # Initialize cluster centers using pretrained autoencoder
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)
    eva(y, y_pred, 'pae')
    
    # Create a list to store results
    results = []
    
    # Training loop
    for epoch in range(200):
        # Update the current epoch
        model.current_epoch = epoch
        
        if epoch % 1 == 0:
            # Evaluate the model
            _, tmp_q, pred, _, _ = model(data, adj, edge_attr)
            
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            
            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            
            # Get evaluation metrics for each round
            acc1, f1_1, nmi1, ari1 = eva(y, res1, f'{epoch}Q')
            acc2, f1_2, nmi2, ari2 = eva(y, res2, f'{epoch}Z')
            acc3, f1_3, nmi3, ari3 = eva(y, res3, f'{epoch}P')
            
            # Save clustering results for each round
            results.append([epoch, acc1, f1_1, acc2, f1_2, acc3, f1_3])
        
        # Forward pass
        x_bar, q, pred, _, _ = model(data, adj, edge_attr)
        
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
    
    # Get final clustering results
    _, _, final_pred, _, _ = model(data, adj, edge_attr)
    final_clusters = final_pred.data.cpu().numpy().argmax(1)
    
    # Save results
    results_df = pd.DataFrame(results, columns=['Epoch', 'Acc_Q', 'F1_Q', 'Acc_Z', 'F1_Z', 'Acc_P', 'F1_P'])
    results_df.to_csv('spatial_training_results.csv', index=False)
    
    print("Training complete. Results saved to 'spatial_training_results.csv'.")
    
    final_results_df = pd.DataFrame({'Node': np.arange(len(final_clusters)), 'Cluster': final_clusters})
    final_results_df.to_csv('spatial_final_cluster_results.csv', index=False)
    
    print("Final clustering results saved to 'spatial_final_cluster_results.csv'.")
    
    return model, results_df


class Logger(object):
    def __init__(self, filename="Default.log", terminal_mode="normal"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
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


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/sdcn_spatial_run_{timestamp}.txt'
    
    # Redirect stdout to both console and file, with minimal terminal output
    sys.stdout = Logger(log_filename, terminal_mode="minimal")
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='train SDCN with SpatialConv',
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
    model, results = train_sdcn_spatial(dataset, args, edge_attr)