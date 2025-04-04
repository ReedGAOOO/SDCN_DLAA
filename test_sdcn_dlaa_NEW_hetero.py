import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
# Import from the new hetero file
from sdcn_dlaa_NEW_hetero import SDCN_DLAA, target_distribution, train_sdcn_dlaa # Removed eva import
from sklearn.cluster import KMeans
import argparse
import pandas as pd
import os
from datetime import datetime
import sys
from collections import defaultdict
import random

# Create logger (Keep Logger class)
class Logger(object):
    def __init__(self, filename="Default.log", terminal_mode="normal"):
        self.terminal = sys.stdout
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
             os.makedirs(log_dir)
        self.log = open(filename, "a", encoding="utf-8")
        self.terminal_mode = terminal_mode

    def write(self, message):
        self.log.write(message)
        # Minimal terminal output for testing script as well
        if self.terminal_mode == "minimal":
            if any(key in message for key in ['acc', 'nmi', 'ari', 'f1', 'Training complete', 'Final clustering', 'use cuda', 'Epoch', 'Error', 'Warning', '开始训练', '聚类分布']):
                 if not any(shape_key in message for shape_key in ['Layer', 'Shape', 'Architecture', 'Encoder', 'Decoder', 'Latent', 'Output', 'Node', 'Edge']):
                      self.terminal.write(message)
        else:
            self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Custom dataset class (Keep CustomDataset class)
class CustomDataset:
    def __init__(self, node_features_path, edge_attr_path=None, device=None):
        self.device = device
        
        if node_features_path.endswith('.pt'):
            self.x = torch.load(node_features_path).numpy()
        else:
            self.x = np.load(node_features_path)
            
        self.num_nodes, self.num_features = self.x.shape
        print(f"Node feature shape: {self.x.shape}")
        
        # Dummy labels for unsupervised task
        self.num_clusters = 3 # Default, can be overridden by args
        self.y = np.zeros(self.num_nodes, dtype=int)
        
        self.edge_attr = None
        if edge_attr_path:
            if edge_attr_path.endswith('.pt'):
                self.edge_attr = torch.load(edge_attr_path)
            else:
                edge_attr_np = np.load(edge_attr_path)
                self.edge_attr = torch.from_numpy(edge_attr_np).float()
                
            if self.device is not None:
                self.edge_attr = self.edge_attr.to(self.device)
                
            print(f"Edge feature shape: {self.edge_attr.shape}")
            
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        # This is not typically used in the current training loop which processes the whole graph
        return torch.from_numpy(self.x[idx]), torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx))

# Convert scipy sparse matrix to torch sparse tensor (Keep sparse conversion)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# Load sparse adjacency matrix (Keep loading function)
def load_sparse_adj(path, device=None):
    sparse_adj = sp.load_npz(path)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(sparse_adj)
    if device is not None:
        adj_tensor = adj_tensor.to(device)
    return adj_tensor

# Remove precompute_edge_to_edge_graph function as it's handled internally now

if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use a specific log filename for the hetero test
    log_filename = f'logs/test_sdcn_dlaa_hetero_run_{timestamp}.txt' 

    # Use minimal output for cleaner test logs
    sys.stdout = Logger(log_filename, terminal_mode="minimal") 

    parser = argparse.ArgumentParser(
        description='Test SDCN_DLAA (Hetero) model', # Updated description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Keep relevant arguments, match those in sdcn_dlaa_NEW_hetero.py main block if needed
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=4)
    # edge_dim will be determined later
    parser.add_argument('--edge_dim', type=int, default=None, help="Target edge feature dimension. If None, inferred from data or set to n_input")
    parser.add_argument('--max_edges_per_node', type=int, default=10)
    # Add arguments for data paths
    parser.add_argument('--node_features_path', type=str, default='NEWDATA/processed/node_features.npy')
    parser.add_argument('--adj_path', type=str, default='NEWDATA/processed/binary_adj.npz')
    parser.add_argument('--edge_attr_path', type=str, default='NEWDATA/processed/edge_attr.npy', help="边特征文件路径 (可选)")
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("使用CUDA: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    # Set file paths
    node_features_path = args.node_features_path
    binary_adj_path = args.adj_path
    # Use None if path is empty string or not provided effectively
    edge_attr_path = args.edge_attr_path if args.edge_attr_path else None 
    
    # Create dataset
    dataset = CustomDataset(node_features_path, edge_attr_path, device=args.device)
    
    # Load adjacency matrix
    adj = load_sparse_adj(binary_adj_path, device=args.device)
    
    # Set model input/output dimensions
    args.n_input = dataset.num_features
    args.n_clusters = dataset.num_clusters # Use clusters from dataset (even if dummy)
    
    # Determine edge_dim
    edge_attr = dataset.edge_attr # Get potentially loaded edge_attr
    if edge_attr is not None:
        if args.edge_dim is None: # If not specified by user, infer from data
            args.edge_dim = edge_attr.shape[1]
            print(f"Inferred edge_dim from data: {args.edge_dim}")
        elif args.edge_dim != edge_attr.shape[1]:
            print(f"Warning: Specified edge_dim ({args.edge_dim}) doesn't match loaded edge feature dimension ({edge_attr.shape[1]}). Model will attempt projection.")
            # Model's _prepare_pyg_data will handle projection
    elif args.edge_dim is None: # No edge_attr loaded and not specified
        args.edge_dim = args.n_input # Default to n_input
        print(f"No edge features provided and edge_dim not specified, setting to n_input: {args.edge_dim}")

    # Print information
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Feature dimension (n_input): {args.n_input}")
    print(f"Target edge feature dimension (edge_dim): {args.edge_dim}")
    print(f"Number of clusters (n_clusters): {args.n_clusters}")
    print(f"Latent space dimension (n_z): {args.n_z}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print(f"Number of attention heads: {args.heads}")
    print(f"Maximum edges per node (edge-to-edge): {args.max_edges_per_node}")
    
    # Train model - Call the main training function from the imported module
    print("\nStarting training SDCN_DLAA (Hetero) model...")
    try:
        # Pass dataset object, args, and the loaded edge_attr
        # The train_sdcn_dlaa function now handles precomputation and training loop
        # Note: We pass the original dataset.edge_attr; train_sdcn_dlaa handles processing/projection
        model, results_df = train_sdcn_dlaa(dataset, adj, args, edge_attr=dataset.edge_attr) # Added adj here
        
        # train_sdcn_dlaa now saves results internally, so no need to save here
        # We can still print final metrics if needed, though they are in the log/CSV
        if not results_df.empty:
             print("\nFinal evaluation metrics recorded during training:")
             print(results_df.iloc[-1])
        
    except Exception as e:
        print(f"Error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nTest script execution completed! Log and result files have been generated.")