import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
from sdcn_dlaa_NEW import SDCN_DLAA, target_distribution, eva, train_sdcn_dlaa_custom
from sklearn.cluster import KMeans
import argparse
import pandas as pd
import os
from datetime import datetime
import sys
from collections import defaultdict
import random

# Create logger
class Logger(object):
    def __init__(self, filename="Default.log", terminal_mode="normal"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")  # Add UTF-8 encoding
        self.terminal_mode = terminal_mode

    def write(self, message):
        # Write to log file
        self.log.write(message)
        
        # Console output
        self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Custom dataset class
class CustomDataset:
    def __init__(self, node_features_path, edge_attr_path=None, device=None):
        """
        Initialize custom dataset
        
        Args:
            node_features_path: Path to node features file
            edge_attr_path: Path to edge features file (optional)
            device: Device (CPU or GPU)
        """
        self.device = device
        
        # Load node features
        if node_features_path.endswith('.pt'):
            self.x = torch.load(node_features_path).numpy()
        else:
            self.x = np.load(node_features_path)
            
        # Get number of nodes and feature dimensions
        self.num_nodes, self.num_features = self.x.shape
        print(f"Node features shape: {self.x.shape}")
        
        # We don't have labels, use all zeros as initial labels (for training only)
        # Assuming 3 cluster classes
        self.num_clusters = 3
        self.y = np.zeros(self.num_nodes, dtype=int)
        
        # If edge features path is provided, load edge features
        self.edge_attr = None
        if edge_attr_path:
            if edge_attr_path.endswith('.pt'):
                self.edge_attr = torch.load(edge_attr_path)
            else:
                edge_attr_np = np.load(edge_attr_path)
                self.edge_attr = torch.from_numpy(edge_attr_np).float()
                
            # Move edge features to target device
            if self.device is not None:
                self.edge_attr = self.edge_attr.to(self.device)
                
            print(f"Edge features shape: {self.edge_attr.shape}")
            
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx))

# Convert scipy sparse matrix to torch sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy sparse matrix to torch sparse tensor
    
    Args:
        sparse_mx: scipy sparse matrix
        
    Returns:
        torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# Load sparse adjacency matrix
def load_sparse_adj(path, device=None):
    """
    Load sparse adjacency matrix
    
    Args:
        path: Path to sparse adjacency matrix file
        device: Device (CPU or GPU)
        
    Returns:
        Adjacency matrix as torch sparse tensor
    """
    sparse_adj = sp.load_npz(path)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(sparse_adj)
    
    # Move tensor to target device
    if device is not None:
        adj_tensor = adj_tensor.to(device)
        
    return adj_tensor

# Precompute edge-to-edge graph relations (key performance optimization)
def precompute_edge_to_edge_graph(adj, max_edges_per_node=10, device=None):
    """
    Precompute edge-to-edge graph relations to avoid recomputing in each forward pass
    
    Args:
        adj: Adjacency matrix (torch sparse tensor)
        max_edges_per_node: Maximum edges per node to consider
        device: Device (CPU or GPU)
        
    Returns:
        edge_index: Node-to-node edge indices [2, num_edges]
        edge_to_edge_index: Edge-to-edge connection indices [2, num_edge_edges]
    """
    print("Precomputing edge-to-edge graph relations (one-time operation)...")
    
    # Convert adjacency matrix to edge indices
    if adj.is_sparse:
        adj = adj.coalesce()
        edge_index = adj.indices()
    else:
        edge_index, _ = dense_to_sparse(adj)
    
    # Move to target device
    if device is not None:
        edge_index = edge_index.to(device)
    
    num_edges = edge_index.size(1)
    
    # Build node-to-edge mapping
    node_to_edges = defaultdict(list)
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        node_to_edges[src].append(i)
        node_to_edges[dst].append(i)
    
    # Build edge-to-edge connections
    edge_to_edge_list = []
    for node, connected_edges in node_to_edges.items():
        if len(connected_edges) > 1:
            # Random sampling when exceeding max edges per node
            if len(connected_edges) > max_edges_per_node:
                sampled_edges = random.sample(connected_edges, max_edges_per_node)
            else:
                sampled_edges = connected_edges
            
            # Connect all edge pairs sharing nodes
            for i in range(len(sampled_edges)):
                for j in range(i+1, len(sampled_edges)):
                    edge_i = sampled_edges[i]
                    edge_j = sampled_edges[j]
                    # Add bidirectional connections for undirected graph
                    edge_to_edge_list.append([edge_i, edge_j])
                    edge_to_edge_list.append([edge_j, edge_i])
    
    # Convert to tensor format
    if len(edge_to_edge_list) > 0:
        edge_to_edge_index = torch.tensor(edge_to_edge_list, dtype=torch.long).t()
        if device is not None:
            edge_to_edge_index = edge_to_edge_index.to(device)
    else:
        # Create empty tensor if no edge-to-edge connections
        edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long)
        if device is not None:
            edge_to_edge_index = edge_to_edge_index.to(device)
    
    print(f"Edge-to-edge graph construction completed: {edge_to_edge_index.shape[1]} edge pairs connected")
    
    return edge_index, edge_to_edge_index

if __name__ == "__main__":
    # Create log directory if not exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/sdcn_dlaa_fixed_run_{timestamp}.txt'
    
    # Redirect stdout to both console and file
    sys.stdout = Logger(log_filename)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train fixed SDCN_DLAA model using preprocessed data from NEWDATA/processed_sparse',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--edge_dim', type=int, default=10)
    parser.add_argument('--max_edges_per_node', type=int, default=10)
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("Use CUDA: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    # Set file paths
    node_features_path = 'NEWDATA/processed_sparse/node_features.npy'
    binary_adj_path = 'NEWDATA/processed_sparse/binary_adj.npz'
    edge_attr_path = 'NEWDATA/processed_sparse/edge_attr.npy'
    
    # Create dataset with specified device
    dataset = CustomDataset(node_features_path, edge_attr_path, device=args.device)
    
    # Load adjacency matrix with specified device
    adj = load_sparse_adj(binary_adj_path, device=args.device)
    
    # Set feature dimensions
    args.n_input = dataset.num_features
    
    # Load edge features
    edge_attr = dataset.edge_attr
    
    # Print information
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Feature dimensions: {dataset.num_features}")
    print(f"Edge feature dimensions: {args.edge_dim}")
    print(f"Number of clusters: {args.n_clusters}")
    
    # Train model
    print("\nStarting training for fixed SDCN_DLAA model...")
    try:
        model, results, clusters = train_sdcn_dlaa_custom(dataset, adj, args, edge_attr)
        
        # Analyze clustering results
        cluster_counts = np.bincount(clusters)
        print("\nCluster distribution:")
        for i, count in enumerate(cluster_counts):
            print(f"Cluster {i}: {count} nodes ({count/len(clusters)*100:.2f}%)")
    except Exception as e:
        print(f"Error occurred during training: {str(e)}")
        # Print full exception traceback for debugging
        import traceback
        traceback.print_exc()
    
    print("\nTraining completed!")