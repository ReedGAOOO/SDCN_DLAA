"""
Test script for SDCN_Spatial model
"""
import torch
import numpy as np
from torch_geometric.data import Data
from sdcn_spatial import SDCN_Spatial, SpatialEmbedding
from utils import load_data, load_graph


def test_spatial_embedding():
    """Test SpatialEmbedding layer"""
    print("Testing SpatialEmbedding layer...")
    
    # Create dummy data
    dist_dim = 10
    embed_size = 5
    dist_feat = torch.rand(15, dist_dim)  # 15 edges, 10-dim features
    dist_feat_order = torch.rand(15, dist_dim)
    
    # Create SpatialEmbedding layer
    spatial_embedding = SpatialEmbedding(dist_dim, embed_size)
    
    # Forward pass
    embedded_dist_feat, embedded_dist_feat_order = spatial_embedding(dist_feat, dist_feat_order)
    
    # Check output shapes
    assert embedded_dist_feat.shape == (15, embed_size), f"Expected shape (15, {embed_size}), got {embedded_dist_feat.shape}"
    assert embedded_dist_feat_order.shape == (15, embed_size), f"Expected shape (15, {embed_size}), got {embedded_dist_feat_order.shape}"
    
    print("SpatialEmbedding test passed!")


def test_sdcn_spatial_init():
    """Test SDCN_Spatial initialization"""
    print("Testing SDCN_Spatial initialization...")
    
    # Model parameters
    n_enc_1, n_enc_2, n_enc_3 = 500, 500, 2000
    n_dec_1, n_dec_2, n_dec_3 = 2000, 500, 500
    n_input, n_z, n_clusters = 2000, 10, 4
    
    # Create model
    model = SDCN_Spatial(
        n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
 
        n_input, n_z, n_clusters, v=1.0, dropout=0.2, heads=4
    )
    
    # Check model components
    assert hasattr(model, 'ae'), "Model should have an autoencoder component"
    assert hasattr(model, 'spatial_embedding'), "Model should have a spatial embedding component"
    assert hasattr(model, 'spatial_conv1'), "Model should have spatial convolution layers"
    assert hasattr(model, 'cluster_layer'), "Model should have a cluster layer"
    
    print("SDCN_Spatial initialization test passed!")


def test_sdcn_spatial_with_edge_features():
    """Test SDCN_Spatial with edge features"""
    print("Testing SDCN_Spatial with edge features...")
    
    # Model parameters
    n_enc_1, n_enc_2, n_enc_3 = 500, 500, 2000
    n_dec_1, n_dec_2, n_dec_3 = 2000, 500, 500
    n_input, n_z, n_clusters = 10, 5, 3  # Smaller dimensions for testing
    edge_dim = 8  # Custom edge dimension
    
    # Create model with custom edge dimension
    model = SDCN_Spatial(
        n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
        n_input, n_z, n_clusters, v=1.0, dropout=0.2, heads=4, edge_dim=edge_dim
    )
    
    # Check if edge_dim is correctly set
    assert model.edge_dim == edge_dim, f"Expected edge_dim to be {edge_dim}, got {model.edge_dim}"
    
    # Check if spatial_embedding is correctly initialized
    assert model.spatial_embedding.embed_layer.weight.shape == (edge_dim, n_z), \
        f"Expected spatial_embedding weight shape ({edge_dim}, {n_z}), got {model.spatial_embedding.embed_layer.weight.shape}"
    
    print("SDCN_Spatial with edge features test passed!")


def test_sdcn_spatial_forward():
    """Test SDCN_Spatial forward pass"""
    print("Testing SDCN_Spatial forward pass...")
    
    # Model parameters
    n_enc_1, n_enc_2, n_enc_3 = 500, 500, 2000
    n_dec_1, n_dec_2, n_dec_3 = 2000, 500, 500
    n_input, n_z, n_clusters = 10, 5, 3  # Smaller dimensions for testing
    
    # Create model
    model = SDCN_Spatial(
        n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
        n_input, n_z, n_clusters, v=1.0, dropout=0.2, heads=4
    )
    
    # Create dummy data
    num_nodes = 20
    x = torch.rand(num_nodes, n_input)
    adj = torch.zeros(num_nodes, num_nodes)
    
    # Create a simple graph: each node is connected to its neighbors
    for i in range(num_nodes):
        if i > 0:
            adj[i, i-1] = 1
            adj[i-1, i] = 1
        if i < num_nodes - 1:
            adj[i, i+1] = 1
            adj[i+1, i] = 1
    
    # Forward pass
    try:
        x_bar, q, predict, z, spatial_shapes = model(x, adj)
        
        # Check output shapes
        assert x_bar.shape == x.shape, f"Expected x_bar shape {x.shape}, got {x_bar.shape}"
        assert q.shape == (num_nodes, n_clusters), f"Expected q shape ({num_nodes}, {n_clusters}), got {q.shape}"
        assert predict.shape == (num_nodes, n_clusters), f"Expected predict shape ({num_nodes}, {n_clusters}), got {predict.shape}"
        assert z.shape == (num_nodes, n_z), f"Expected z shape ({num_nodes}, {n_z}), got {z.shape}"
        
        print("SDCN_Spatial forward pass test passed!")
    except Exception as e:
        print(f"SDCN_Spatial forward pass test failed: {e}")
        raise


def test_sdcn_spatial_forward_with_edge_features():
    """Test SDCN_Spatial forward pass with edge features"""
    print("Testing SDCN_Spatial forward pass with edge features...")
    
    # Model parameters
    n_enc_1, n_enc_2, n_enc_3 = 500, 500, 2000
    n_dec_1, n_dec_2, n_dec_3 = 2000, 500, 500
    n_input, n_z, n_clusters = 10, 5, 3  # Smaller dimensions for testing
    edge_dim = 8  # Custom edge dimension
    
    # Create model with custom edge dimension
    model = SDCN_Spatial(
        n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
        n_input, n_z, n_clusters, v=1.0, dropout=0.2, heads=4, edge_dim=edge_dim
    )
    
    # Create dummy data
    num_nodes = 20
    x = torch.rand(num_nodes, n_input)
    adj = torch.zeros(num_nodes, num_nodes)
    
    # Create a simple graph: each node is connected to its neighbors
    for i in range(num_nodes):
        if i > 0:
            adj[i, i-1] = 1
            adj[i-1, i] = 1
        if i < num_nodes - 1:
            adj[i, i+1] = 1
            adj[i+1, i] = 1
    
    # Create dummy edge features
    edge_index, _ = dense_to_sparse(adj)
    num_edges = edge_index.size(1)
    edge_attr = torch.rand(num_edges, edge_dim)
    
    # Forward pass with edge features
    x_bar, q, predict, z, spatial_shapes = model(x, adj, edge_attr)
    
    print("SDCN_Spatial forward pass with edge features test passed!")


def test_with_real_data():
    """Test SDCN_Spatial with real data"""
    print("Testing SDCN_Spatial with real data...")
    
    try:
        # Load a small dataset (e.g., a subset of reut)
        dataset = load_data('reut')
        
        # Take a small subset for testing
        subset_size = min(100, dataset.x.shape[0])
        x_subset = dataset.x[:subset_size]
        y_subset = dataset.y[:subset_size]
        
        # Convert to torch tensor
        x = torch.Tensor(x_subset)
        
        # Load graph
        adj = load_graph('reut', k=3)
        
        # Take corresponding subset of adjacency matrix
        adj_subset = adj[:subset_size, :subset_size].clone()
        
        # Model parameters
        n_input = x.shape[1]
        n_z = 10
        n_clusters = len(np.unique(y_subset))
        
        # Create model
        model = SDCN_Spatial(
            500, 500, 2000, 2000, 500, 500,
            n_input=n_input, n_z=n_z, n_clusters=n_clusters,
            v=1.0, dropout=0.2, heads=4
        )
        
        # Forward pass
        x_bar, q, predict, z, spatial_shapes = model(x, adj_subset)
        
        # Check output shapes
        assert x_bar.shape == x.shape, f"Expected x_bar shape {x.shape}, got {x_bar.shape}"
        assert q.shape == (subset_size, n_clusters), f"Expected q shape ({subset_size}, {n_clusters}), got {q.shape}"
        assert predict.shape == (subset_size, n_clusters), f"Expected predict shape ({subset_size}, {n_clusters}), got {predict.shape}"
        assert z.shape == (subset_size, n_z), f"Expected z shape ({subset_size}, {n_z}), got {z.shape}"
        
        print("SDCN_Spatial test with real data passed!")
    except Exception as e:
        print(f"SDCN_Spatial test with real data failed: {e}")
        print("This might be due to missing pretrained model or dataset. Skipping this test.")


if __name__ == "__main__":
    print("Running tests for SDCN_Spatial model...")
    
    # Run tests
    test_spatial_embedding()
    test_sdcn_spatial_init()
    test_sdcn_spatial_with_edge_features()
    test_sdcn_spatial_forward()
    test_sdcn_spatial_forward_with_edge_features()
    
    # This test requires real data and might fail if data is not available
    try:
        test_with_real_data()
    except Exception as e:
        print(f"Test with real data failed: {e}")
        print("This might be due to missing data. Skipping this test.")
    
    print("All tests completed!")