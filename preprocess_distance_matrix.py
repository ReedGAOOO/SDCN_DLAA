#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing script: Converts an adjacency matrix containing actual distances into edge features (edge_attr) and a binary adjacency matrix.
Suitable for SDCN_Spatial and DLAA models.

Input:
- Node feature matrix (CSV file, e.g., X_simplize.CSV)
- Distance adjacency matrix (CSV file, e.g., A.csv, containing actual distances between nodes)

Output:
- Processed node feature matrix (numpy array)
- Binary adjacency matrix (scipy sparse matrix)
- Edge feature matrix (numpy array)
- Edge index (source and target node pairs)
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.utils import dense_to_sparse
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_node_features(file_path):
    """
    Loads the node feature matrix.

    Args:
        file_path: Path to the node feature CSV file.

    Returns:
        node_features: Node feature matrix (numpy array).
        feature_names: List of feature names.
    """
    print(f"Loading node features: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print dataset information
    print(f"Node feature dataset shape: {df.shape}")
    print(f"Node feature column names: {df.columns.tolist()}")
    print(f"First 5 rows of data:\n{df.head()}")
    
    # Extract feature names
    feature_names = df.columns.tolist()
    
    # Convert to numpy array
    node_features = df.values
    
    print(f"Node feature matrix shape: {node_features.shape}")
    return node_features, feature_names


def load_distance_matrix(file_path):
    """
    Loads the distance adjacency matrix.

    Args:
        file_path: Path to the distance adjacency matrix CSV file.

    Returns:
        distance_matrix: Distance adjacency matrix (numpy array).
    """
    print(f"Loading distance adjacency matrix: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Print dataset information
    print(f"Distance adjacency matrix shape: {df.shape}")
    print(f"First 5 rows and columns of data:\n{df.iloc[:5, :5]}")
    
    # Convert to numpy array
    distance_matrix = df.values
    
    # Ensure the matrix is square
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "The distance matrix must be a square matrix"
    
    print(f"Distance adjacency matrix shape: {distance_matrix.shape}")
    return distance_matrix


def create_binary_adjacency(distance_matrix, threshold=0):
    """
    Creates a binary adjacency matrix from the distance matrix.

    Args:
        distance_matrix: Distance adjacency matrix.
        threshold: Distance threshold, edges with values greater than this will be kept.

    Returns:
        binary_adj: Binary adjacency matrix (scipy sparse matrix).
    """
    print(f"Creating binary adjacency matrix (threshold: {threshold})")
    
    # Create binary adjacency matrix (1 for edge, 0 for no edge)
    binary_adj = (distance_matrix > threshold).astype(np.float32)
    
    # Set diagonal to 0 (remove self-loops)
    np.fill_diagonal(binary_adj, 0)
    
    # Calculate the number of edges
    num_edges = np.sum(binary_adj)
    print(f"Number of edges: {num_edges}")
    
    # Convert to sparse matrix
    sparse_adj = sp.csr_matrix(binary_adj)
    
    # Calculate sparsity
    sparsity = 1.0 - (sparse_adj.nnz / (sparse_adj.shape[0] * sparse_adj.shape[1]))
    print(f"Adjacency matrix sparsity: {sparsity:.4f}")
    
    return sparse_adj


def extract_edge_features(distance_matrix, binary_adj):
    """
    Extracts edge features from the distance matrix.

    Args:
        distance_matrix: Distance adjacency matrix.
        binary_adj: Binary adjacency matrix.

    Returns:
        edge_index: Edge index [2, num_edges].
        edge_attr: Edge features [num_edges, 1].
    """
    print("Extracting edge features")
    
    # Get edge indices (source and target nodes)
    rows, cols = np.where(binary_adj > 0)
    
    # Extract corresponding distances as edge features
    distances = distance_matrix[rows, cols]
    
    # Create edge index
    edge_index = np.vstack((rows, cols))
    
    # Create edge features (distances)
    edge_attr = distances.reshape(-1, 1)
    
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge feature shape: {edge_attr.shape}")
    
    return edge_index, edge_attr


def normalize_edge_features(edge_attr, method='minmax'):
    """
    Normalizes edge features.

    Args:
        edge_attr: Edge features.
        method: Normalization method ('minmax', 'standard', 'none').

    Returns:
        normalized_edge_attr: Normalized edge features.
    """
    print(f"Normalizing edge features using {method} method")
    
    if method == 'none':
        return edge_attr
    
    if method == 'minmax':
        # Min-Max normalization
        min_val = np.min(edge_attr)
        max_val = np.max(edge_attr)
        normalized_edge_attr = (edge_attr - min_val) / (max_val - min_val)
    elif method == 'standard':
        # Standardization (Z-score)
        mean_val = np.mean(edge_attr)
        std_val = np.std(edge_attr)
        normalized_edge_attr = (edge_attr - mean_val) / std_val
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    print(f"Edge feature range before normalization: [{np.min(edge_attr)}, {np.max(edge_attr)}]")
    print(f"Edge feature range after normalization: [{np.min(normalized_edge_attr)}, {np.max(normalized_edge_attr)}]")
    
    return normalized_edge_attr


def expand_edge_features(edge_attr, dim=10):
    """
    Expands the dimension of edge features.

    Args:
        edge_attr: Edge features [num_edges, 1].
        dim: Target dimension.

    Returns:
        expanded_edge_attr: Expanded edge features [num_edges, dim].
    """
    print(f"Expanding edge feature dimension to {dim}")
    
    if dim == 1:
        return edge_attr
    
    # Get the number of edges
    num_edges = edge_attr.shape[0]
    
    # Create expanded edge features
    expanded_edge_attr = np.zeros((num_edges, dim))
    
    # Place the original distance feature in the first column
    expanded_edge_attr[:, 0] = edge_attr.flatten()
    
    # Generate other features using functions of distance
    for i in range(1, dim):
        # Different functions can be used, such as powers, exponentials, logarithms of the distance, etc.
        if i % 3 == 0:
            # Square of the distance
            expanded_edge_attr[:, i] = np.power(edge_attr.flatten(), 2)
        elif i % 3 == 1:
            # Exponential decay of the distance
            expanded_edge_attr[:, i] = np.exp(-edge_attr.flatten())
        else:
            # Inverse of the distance
            expanded_edge_attr[:, i] = 1.0 / (edge_attr.flatten() + 1e-6)
    
    print(f"Expanded edge feature shape: {expanded_edge_attr.shape}")
    return expanded_edge_attr


def save_processed_data(output_dir, node_features, binary_adj, edge_index, edge_attr, feature_names=None):
    """
    Saves the processed data.

    Args:
        output_dir: Output directory.
        node_features: Node feature matrix.
        binary_adj: Binary adjacency matrix.
        edge_index: Edge index.
        edge_attr: Edge features.
        feature_names: List of feature names.
    """
    print(f"Saving processed data to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save node features
    np.save(os.path.join(output_dir, 'node_features.npy'), node_features)
    
    # Save binary adjacency matrix
    sp.save_npz(os.path.join(output_dir, 'binary_adj.npz'), binary_adj)
    
    # Save edge index
    np.save(os.path.join(output_dir, 'edge_index.npy'), edge_index)
    
    # Save edge features
    np.save(os.path.join(output_dir, 'edge_attr.npy'), edge_attr)
    
    # Save as PyTorch format
    torch_node_features = torch.FloatTensor(node_features)
    torch_edge_index = torch.LongTensor(edge_index)
    torch_edge_attr = torch.FloatTensor(edge_attr)
    
    # Save PyTorch tensors
    torch.save(torch_node_features, os.path.join(output_dir, 'node_features.pt'))
    torch.save(torch_edge_index, os.path.join(output_dir, 'edge_index.pt'))
    torch.save(torch_edge_attr, os.path.join(output_dir, 'edge_attr.pt'))
    
    # Save feature names
    if feature_names is not None:
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
    
    # Save processing information
    with open(os.path.join(output_dir, 'data_info.txt'), 'w') as f:
        f.write(f"Number of nodes: {node_features.shape[0]}\n")
        f.write(f"Number of features: {node_features.shape[1]}\n")
        f.write(f"Number of edges: {edge_index.shape[1]}\n")
        f.write(f"Dimension of edge features: {edge_attr.shape[1]}\n")
        f.write(f"Adjacency matrix sparsity: {1.0 - (binary_adj.nnz / (binary_adj.shape[0] * binary_adj.shape[1])):.4f}\n")
    
    print("Data saving completed")


def visualize_data(output_dir, node_features, binary_adj, edge_attr):
    """
    Visualizes the processed data.

    Args:
        output_dir: Output directory.
        node_features: Node feature matrix.
        binary_adj: Binary adjacency matrix.
        edge_attr: Edge features.
    """
    print("Generating data visualizations")
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Node feature distribution
    plt.figure(figsize=(12, 8))
    for i in range(min(5, node_features.shape[1])):
        plt.subplot(2, 3, i+1)
        plt.hist(node_features[:, i], bins=20)
        plt.title(f'Feature {i} distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'node_features_dist.png'))
    plt.close()
    
    # 2. Adjacency matrix heatmap (if the number of nodes is not too large)
    if binary_adj.shape[0] <= 100:
        plt.figure(figsize=(10, 10))
        sns.heatmap(binary_adj.toarray(), cmap='Blues', cbar=True)
        plt.title('Binary Adjacency Matrix')
        plt.savefig(os.path.join(vis_dir, 'adjacency_heatmap.png'))
        plt.close()
    
    # 3. Edge feature distribution
    plt.figure(figsize=(12, 8))
    for i in range(min(5, edge_attr.shape[1])):
        plt.subplot(2, 3, i+1)
        plt.hist(edge_attr[:, i], bins=20)
        plt.title(f'Edge feature {i} distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'edge_features_dist.png'))
    plt.close()
    
    # 4. Node degree distribution
    degrees = np.sum(binary_adj.toarray(), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=20)
    plt.title('Node Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.savefig(os.path.join(vis_dir, 'degree_dist.png'))
    plt.close()
    
    print(f"Visualization results saved to {vis_dir}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert distance adjacency matrix to edge features and binary adjacency matrix')
    parser.add_argument('--node_features', type=str, default='NEWDATA/X_simplize.CSV',
                        help='Path to the node feature CSV file')
    parser.add_argument('--distance_matrix', type=str, default='NEWDATA/A.csv',
                        help='Path to the distance adjacency matrix CSV file')
    parser.add_argument('--output_dir', type=str, default='NEWDATA/processed',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Distance threshold, edges with values greater than this will be kept')
    parser.add_argument('--normalize', type=str, default='minmax',
                        choices=['minmax', 'standard', 'none'],
                        help='Edge feature normalization method')
    parser.add_argument('--edge_dim', type=int, default=10,
                        help='Edge feature dimension')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to generate data visualizations')
    
    args = parser.parse_args()
    
    # Load node features
    node_features, feature_names = load_node_features(args.node_features)
    
    # Load distance adjacency matrix
    distance_matrix = load_distance_matrix(args.distance_matrix)
    
    # Create binary adjacency matrix
    binary_adj = create_binary_adjacency(distance_matrix, args.threshold)
    
    # Extract edge features
    edge_index, edge_attr = extract_edge_features(distance_matrix, binary_adj.toarray())
    
    # Normalize edge features
    edge_attr = normalize_edge_features(edge_attr, args.normalize)
    
    # Expand edge feature dimension
    edge_attr = expand_edge_features(edge_attr, args.edge_dim)
    
    # Save processed data
    save_processed_data(args.output_dir, node_features, binary_adj, edge_index, edge_attr, feature_names)
    
    # Visualize data
    if args.visualize:
        visualize_data(args.output_dir, node_features, binary_adj, edge_attr)
    
    print("Data preprocessing completed")


if __name__ == "__main__":
    main()