#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing script: Convert adjacency matrix with actual distances into edge features (edge_attr) and sparse adjacency graph structure
Compatible with SDCN_Spatial and DLAA models

Input:
- Node feature matrix (CSV file, e.g. X_simplize.CSV)
- Distance adjacency matrix (CSV file, e.g. A.csv, containing actual distances between nodes)

Output:
- Processed node feature matrix (numpy array)
- Binary adjacency matrix of graph structure (scipy sparse matrix) - for compatibility/visualization
- Edge index (numpy array [2, num_edges]) - core graph structure
- Edge feature matrix (numpy array [num_edges, edge_dim]) - generated from distances
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_undirected
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_node_features(file_path):
    """Load node feature matrix"""
    print(f"Loading node features: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Node feature dataset shape: {df.shape}")
    print(f"Node feature column names: {df.columns.tolist()}")
    # print(f"First 5 rows:\n{df.head()}") # Optional: Less verbose
    feature_names = df.columns.tolist()
    node_features = df.values
    print(f"Node feature matrix shape: {node_features.shape}")
    return node_features, feature_names


def load_distance_matrix(file_path):
    """Load distance adjacency matrix"""
    print(f"Loading distance adjacency matrix: {file_path}")
    df = pd.read_csv(file_path, header=None)
    print(f"Distance adjacency matrix shape: {df.shape}")
    # print(f"First 5 rows and 5 columns:\n{df.iloc[:5, :5]}") # Optional: Less verbose
    distance_matrix = df.values
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "Distance matrix must be square"
    print(f"Distance adjacency matrix shape: {distance_matrix.shape}")
    return distance_matrix

# --- Method 1: Threshold-based Graph Creation ---
def create_threshold_graph(distance_matrix, theta):
    """
    Create threshold-based sparse graph from distance matrix (keep edges <= theta)
    
    Args:
        distance_matrix: Distance adjacency matrix (NumPy)
        theta: Distance threshold
        
    Returns:
        sparse_adj_binary: Binary adjacency matrix (scipy sparse matrix)
        edge_index: Edge index [2, num_edges] (NumPy)
        edge_attr_dist: Raw distances on edges [num_edges, 1] (NumPy)
    """
    print(f"Creating threshold-based graph (theta={theta})")
    num_nodes = distance_matrix.shape[0]

    # Find edges with distance in (0, theta] range
    # Note: We exclude edges with distance=0 (typically diagonal)
    rows, cols = np.where((distance_matrix > 0) & (distance_matrix <= theta))

    if rows.size == 0:
         print("Warning: No edges found with current threshold!")
         # Return empty graph structure
         return sp.csr_matrix((num_nodes, num_nodes)), np.zeros((2, 0), dtype=int), np.zeros((0, 1))

    # Extract corresponding distances as initial edge features
    distances = distance_matrix[rows, cols]

    # Create edge index
    edge_index = np.vstack((rows, cols))

    # Create raw distance edge features
    edge_attr_dist = distances.reshape(-1, 1)

    # Create binary sparse adjacency matrix (for saving/visualization)
    sparse_adj_binary = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))

    print(f"Threshold-based graph - Number of edges: {edge_index.shape[1]}")
    sparsity = 1.0 - (sparse_adj_binary.nnz / (num_nodes * num_nodes))
    print(f"Adjacency matrix sparsity: {sparsity:.6f}")

    return sparse_adj_binary, edge_index, edge_attr_dist


# --- Method 2: K-Nearest Neighbors (KNN) Graph Creation ---
def create_knn_graph(distance_matrix, k=10):
    """
    Create KNN graph from distance matrix (keep only k nearest neighbors for each node).
    Note: Resulting graph may not be symmetric - here we only keep i -> j edges if j is i's KNN.
    
    Args:
        distance_matrix: Distance adjacency matrix (NumPy dense matrix)
        k: Number of nearest neighbors to keep per node
        
    Returns:
        sparse_adj_binary: Binary adjacency matrix (scipy sparse matrix)
        edge_index: Edge index [2, num_edges] (NumPy)
        edge_attr_dist: Raw distances on edges [num_edges, 1] (NumPy)
    """
    print(f"Creating KNN graph (k={k})")
    num_nodes = distance_matrix.shape[0]
    rows = []
    cols = []
    distances = []

    # Make copy of distance matrix and set diagonal to infinity to ignore self-loops
    dist_matrix_no_diag = distance_matrix.copy()
    np.fill_diagonal(dist_matrix_no_diag, np.inf)

    for i in range(num_nodes):
        # Find all distances for ith node and get sorted indices
        neighbor_distances = dist_matrix_no_diag[i, :]
        nearest_indices = np.argsort(neighbor_distances) # Ascending order

        # Select k nearest neighbors
        selected_neighbors = nearest_indices[:k]

        # Add edges and distances
        for neighbor_idx in selected_neighbors:
           # Ensure distance is valid (not original diagonal inf)
           dist = neighbor_distances[neighbor_idx] # Using modified distance check
           if np.isfinite(dist):
                rows.append(i)
                cols.append(neighbor_idx)
                distances.append(distance_matrix[i, neighbor_idx]) # Store original distance

    if not rows:
         print("Warning: KNN graph generated no edges! Check K value or distance matrix.")
         return sp.csr_matrix((num_nodes, num_nodes)), np.zeros((2, 0), dtype=int), np.zeros((0, 1))

    # Create edge index
    edge_index = np.vstack((rows, cols))

    # Create raw distance edge features
    edge_attr_dist = np.array(distances).reshape(-1, 1)

    # Create binary sparse adjacency matrix (for saving/visualization)
    sparse_adj_binary = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))

    print(f"KNN graph - Number of edges: {edge_index.shape[1]}")
    average_degree = edge_index.shape[1] / num_nodes
    print(f"Average degree: {average_degree:.2f}")
    sparsity = 1.0 - (sparse_adj_binary.nnz / (num_nodes * num_nodes))
    print(f"Adjacency matrix sparsity: {sparsity:.6f}")

    return sparse_adj_binary, edge_index, edge_attr_dist


def normalize_edge_features(edge_attr, method='minmax'):
    """Normalize edge features (input should be raw distances)"""
    print(f"Normalizing edge features (raw distances) using {method} method")

    if edge_attr.shape[0] == 0:
        print("Edge features empty, skipping normalization.")
        return edge_attr

    if method == 'none':
        return edge_attr

    if method == 'minmax':
        min_val = np.min(edge_attr)
        max_val = np.max(edge_attr)
        # Avoid division by zero
        if max_val == min_val:
             print("Warning: All edge feature values identical, MinMax normalization results in 0.")
             normalized_edge_attr = np.zeros_like(edge_attr)
        else:
             normalized_edge_attr = (edge_attr - min_val) / (max_val - min_val)
    elif method == 'standard':
        mean_val = np.mean(edge_attr)
        std_val = np.std(edge_attr)
         # Avoid division by zero
        if std_val == 0:
            print("Warning: All edge feature values identical, standardization results in 0.")
            normalized_edge_attr = np.zeros_like(edge_attr)
        else:
            normalized_edge_attr = (edge_attr - mean_val) / std_val
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    print(f"Edge feature (distance) range before normalization: [{np.min(edge_attr):.4f}, {np.max(edge_attr):.4f}]")
    if edge_attr.shape[0] > 0:
        print(f"Edge feature range after normalization: [{np.min(normalized_edge_attr):.4f}, {np.max(normalized_edge_attr):.4f}]")

    return normalized_edge_attr


def expand_edge_features(normalized_edge_attr, dim=10):
    """Expand edge feature dimensions (input should be normalized features)"""
    print(f"Expanding edge feature dimensions to {dim}")

    if normalized_edge_attr.shape[0] == 0:
        print("Edge features empty, skipping expansion.")
        return np.zeros((0, dim))
    if normalized_edge_attr.shape[1] == dim:
        print("Edge feature dimensions already meet requirements, no expansion needed.")
        return normalized_edge_attr
    if normalized_edge_attr.shape[1] != 1 :
        print(f"Warning: Expected 1D features for expansion but received {normalized_edge_attr.shape[1]}D features. Will only use first dimension.")


    if dim == 1:
        return normalized_edge_attr[:, 0:1] # Ensure [N, 1] shape

    num_edges = normalized_edge_attr.shape[0]
    expanded_edge_attr = np.zeros((num_edges, dim))

    # Get first dimension feature (assumed to be normalized distance or its representation)
    base_feature = normalized_edge_attr[:, 0].flatten()

    # Put base feature in first column
    expanded_edge_attr[:, 0] = base_feature

    # Generate other features using functions of base feature
    for i in range(1, dim):
        if i % 3 == 0:
            expanded_edge_attr[:, i] = np.power(base_feature, 2) # Square
        elif i % 3 == 1:
            expanded_edge_attr[:, i] = np.exp(-base_feature * 5) # Exponential decay (multiply by 5 for better distinction)
        else:
            # Avoid division by zero, add small value
            expanded_edge_attr[:, i] = 1.0 / (base_feature + 1e-6) # Reciprocal

    print(f"Expanded edge feature shape: {expanded_edge_attr.shape}")
    return expanded_edge_attr


def save_processed_data(output_dir, node_features, sparse_adj_binary, edge_index, edge_attr_final, feature_names=None):
    """Save processed data"""
    print(f"Saving processed data to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save node features
    np.save(os.path.join(output_dir, 'node_features.npy'), node_features)
    torch.save(torch.FloatTensor(node_features), os.path.join(output_dir, 'node_features.pt'))

    # Save binary adjacency matrix (sparse)
    sp.save_npz(os.path.join(output_dir, 'binary_adj.npz'), sparse_adj_binary)

    # Save edge index
    np.save(os.path.join(output_dir, 'edge_index.npy'), edge_index)
    torch.save(torch.LongTensor(edge_index), os.path.join(output_dir, 'edge_index.pt'))

    # Save final edge features (normalized + expanded)
    np.save(os.path.join(output_dir, 'edge_attr.npy'), edge_attr_final)
    torch.save(torch.FloatTensor(edge_attr_final), os.path.join(output_dir, 'edge_attr.pt'))

    # Save feature names
    if feature_names is not None:
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")

    # Save processing info
    with open(os.path.join(output_dir, 'data_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Number of nodes: {node_features.shape[0]}\n")
        f.write(f"Node feature dimensions: {node_features.shape[1]}\n")
        f.write(f"Number of edges: {edge_index.shape[1]}\n")
        f.write(f"Final edge feature dimensions: {edge_attr_final.shape[1] if edge_attr_final.ndim > 1 else 1}\n")
        if sparse_adj_binary.shape[0] > 0:
            sparsity = 1.0 - (sparse_adj_binary.nnz / (sparse_adj_binary.shape[0] * sparse_adj_binary.shape[1]))
            f.write(f"Adjacency matrix sparsity: {sparsity:.6f}\n")
        else:
             f.write(f"Adjacency matrix sparsity: N/A (empty graph)\n")
        # Record method used
        # We need to pass the method name or args here, omitted for simplicity now

    print("Data saved successfully")


def visualize_data(output_dir, node_features, sparse_adj_binary, edge_attr_final):
    """Visualize processed data"""
    print("Generating data visualizations")
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Node feature distributions (first 5)
    plt.figure(figsize=(12, 8))
    num_node_feat_to_plot = min(5, node_features.shape[1])
    for i in range(num_node_feat_to_plot):
        plt.subplot(2, 3, i+1)
        plt.hist(node_features[:, i], bins=30)
        plt.title(f'Node Feature {i} Dist')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'node_features_dist.png'))
    plt.close()

    # 2. Adjacency matrix heatmap (if num_nodes <= 100)
    if sparse_adj_binary.shape[0] <= 100 and sparse_adj_binary.nnz > 0:
        plt.figure(figsize=(10, 10))
        sns.heatmap(sparse_adj_binary.toarray(), cmap='Blues', cbar=False)
        plt.title('Sparse Adjacency Matrix')
        plt.savefig(os.path.join(vis_dir, 'adjacency_heatmap.png'))
        plt.close()
    elif sparse_adj_binary.shape[0] > 100:
         print("Number of nodes > 100, skipping adjacency matrix heatmap visualization.")
    elif sparse_adj_binary.nnz == 0:
         print("Graph is empty, skipping adjacency matrix heatmap visualization.")


    # 3. Final edge feature distributions (first 5 dimensions)
    if edge_attr_final.shape[0] > 0: # Check if edges exist
        plt.figure(figsize=(12, 8))
        num_edge_feat_to_plot = min(5, edge_attr_final.shape[1] if edge_attr_final.ndim > 1 else 1)
        for i in range(num_edge_feat_to_plot):
            plt.subplot(2, 3, i+1)
            if edge_attr_final.ndim > 1:
                plt.hist(edge_attr_final[:, i], bins=30)
            else: # Handle 1D case
                plt.hist(edge_attr_final, bins=30)
            plt.title(f'Final Edge Feature Dim {i} Dist')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'final_edge_features_dist.png'))
        plt.close()
    else:
        print("Graph is empty, skipping edge feature distribution visualization.")


    # 4. Node degree distribution
    if sparse_adj_binary.nnz > 0:
        # Calculate out-degrees and in-degrees (may differ for asymmetric KNN)
        out_degrees = sparse_adj_binary.sum(axis=1).A1 # .A1 converts matrix to 1D array
        in_degrees = sparse_adj_binary.sum(axis=0).A1

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(out_degrees, bins=max(1, int(np.max(out_degrees)) // 2 if np.max(out_degrees)>0 else 1) )
        plt.title('Node Out-Degree Distribution')
        plt.xlabel('Out-Degree')
        plt.ylabel('Node Count')

        plt.subplot(1, 2, 2)
        plt.hist(in_degrees, bins=max(1, int(np.max(in_degrees)) // 2 if np.max(in_degrees)>0 else 1))
        plt.title('Node In-Degree Distribution')
        plt.xlabel('In-Degree')
        plt.ylabel('Node Count')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'degree_dist.png'))
        plt.close()
    else:
         print("Graph is empty, skipping degree distribution visualization.")


    print(f"Visualizations saved to {vis_dir}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert distance adjacency matrix to sparse graph structure and edge features')
    parser.add_argument('--node_features', type=str, default='NEWDATA/X_simplize.CSV', help='Path to node features CSV file')
    parser.add_argument('--distance_matrix', type=str, default='NEWDATA/A.csv', help='Path to distance adjacency matrix CSV file')
    parser.add_argument('--output_dir', type=str, default='NEWDATA/processed_sparse', help='Output directory') # Changed default output dir

    # Graph sparsification method arguments
    parser.add_argument('--method', type=str, default='knn', choices=['knn', 'threshold'], help='Graph sparsification method')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors to keep per node for KNN method')
    parser.add_argument('--theta', type=float, default=None, help='Distance threshold for Threshold method (should be set based on data distribution)') # Default None, must be set if method='threshold'

    # Edge feature processing arguments
    parser.add_argument('--normalize', type=str, default='minmax', choices=['minmax', 'standard', 'none'], help='Edge feature (distance) normalization method')
    parser.add_argument('--edge_dim', type=int, default=10, help='Target dimension for final edge features')
    parser.add_argument('--visualize', action='store_true', help='Whether to generate data visualizations')

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.method == 'threshold' and args.theta is None:
        parser.error("--method 'threshold' requires --theta to be set.")
    if args.method == 'knn' and args.k <= 0:
         parser.error("--k must be a positive integer for method 'knn'.")

    print("--- Configuration Parameters ---")
    print(f"Node features file: {args.node_features}")
    print(f"Distance matrix file: {args.distance_matrix}")
    print(f"Output directory: {args.output_dir}")
    print(f"Graph sparsification method: {args.method}")
    if args.method == 'knn':
        print(f"  K value: {args.k}")
    else:
        print(f"  Theta threshold: {args.theta}")
    print(f"Edge feature (distance) normalization: {args.normalize}")
    print(f"Final edge feature dimensions: {args.edge_dim}")
    print(f"Generate visualizations: {args.visualize}")
    print("-----------------")


    # Load node features
    node_features, feature_names = load_node_features(args.node_features)

    # Load distance adjacency matrix
    distance_matrix = load_distance_matrix(args.distance_matrix)

    # --- Graph Sparsification ---
    sparse_adj_binary = None
    edge_index = None
    edge_attr_distances = None # Store raw distances first
    
    num_nodes = distance_matrix.shape[0] # Get number of nodes

    if args.method == 'knn':
        # 1. First create directed KNN graph
        print("Step 1: Creating initial directed KNN graph...")
        # Use temp variables to store directed graph results
        sparse_adj_binary_directed, edge_index_directed, edge_attr_distances_directed = create_knn_graph(distance_matrix, args.k)

        # 2. Convert directed edge index to undirected (take union)
        print("Step 2: Converting directed graph to undirected...")
        edge_index_tensor = torch.from_numpy(edge_index_directed).long()

        # Call to_undirected to symmetrize edge index
        edge_index_undirected_tensor = to_undirected(edge_index_tensor, num_nodes=num_nodes)
        
        # Convert symmetrized edge index back to numpy format as final edge_index
        edge_index = edge_index_undirected_tensor.numpy()
        print(f"Number of edges after undirected conversion: {edge_index.shape[1]}")

        # 3. Re-extract distance features corresponding to undirected edges (important!)
        #    Because to_undirected may have added new edges (j, i), we need to find corresponding distances
        print("Step 3: Re-extracting distance features for undirected edges...")
        rows_undirected, cols_undirected = edge_index[0], edge_index[1]
        # Look up distances for new edge_index directly from original distance matrix
        edge_attr_distances = distance_matrix[rows_undirected, cols_undirected].reshape(-1, 1)
        print(f"Re-extracted edge feature shape: {edge_attr_distances.shape}")

        # 4. Create final binary sparse adjacency matrix (based on undirected edges)
        print("Step 4: Creating final undirected binary sparse adjacency matrix...")
        sparse_adj_binary = sp.csr_matrix((np.ones(edge_index.shape[1]), (rows_undirected, cols_undirected)),
                                          shape=(num_nodes, num_nodes))
        sparsity = 1.0 - (sparse_adj_binary.nnz / (num_nodes * num_nodes))
        print(f"Final undirected adjacency matrix sparsity: {sparsity:.6f}")
        

    elif args.method == 'threshold':
        # Threshold method results are already undirected if distance matrix is symmetric
        sparse_adj_binary, edge_index, edge_attr_distances = create_threshold_graph(distance_matrix, args.theta)
        print("Threshold method graph, assuming input distance matrix is symmetric, graph is already undirected.")
    else:
        raise ValueError(f"Unknown sparsification method: {args.method}")

    # --- Edge Feature Processing ---
    # Subsequent processing now uses symmetrized edge_index and re-extracted edge_attr_distances
    print("\n--- Processing edge features ---")
    edge_attr_normalized = normalize_edge_features(edge_attr_distances, args.normalize)
    edge_attr_final = expand_edge_features(edge_attr_normalized, args.edge_dim)

    # --- Saving and Visualization ---
    # Using final (possibly undirected) sparse_adj_binary, edge_index, edge_attr_final
    print("\n--- Saving data ---")
    save_processed_data(args.output_dir, node_features, sparse_adj_binary, edge_index, edge_attr_final, feature_names)

    # Visualize data
    if args.visualize:
        print("\n--- Generating data visualizations ---")
        visualize_data(args.output_dir, node_features, sparse_adj_binary, edge_attr_final)

    print(f"\nData preprocessing ({args.method} method) completed")


if __name__ == "__main__":
    main()