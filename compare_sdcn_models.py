import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_data, load_graph
from sdcn_spatial import train_sdcn_spatial
from spatial_sdcn_improved import train_sdcn_spatial_improved
import pandas as pd
import os
from datetime import datetime

def setup_args():
    parser = argparse.ArgumentParser(
        description='Compare SDCN models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='usps', help='Dataset name: usps, hhar, reut, acm, dblp, cite')
    parser.add_argument('--k', type=int, default=3, help='KNN neighbors')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_clusters', default=None, type=int, help='Number of clusters (auto-determined if None)')
    parser.add_argument('--n_z', default=10, type=int, help='Dimension of latent space')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--edge_dummy_type', type=str, default="onehot", choices=["onehot", "uniform"], 
                        help='Type of dummy edge features to use when real edge features are not available')
    parser.add_argument('--max_edges_per_node', type=int, default=10, 
                        help='Maximum number of edges to consider per node for edge-to-edge connections')
    parser.add_argument('--use_edge_attr', action='store_true', help='Use edge attributes from dataset if available')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--run_original', action='store_true', help='Run original SDCN_Spatial model')
    parser.add_argument('--run_improved', action='store_true', help='Run improved SDCN_Spatial model')
    
    args = parser.parse_args()
    
    # If neither is explicitly selected, run both
    if not args.run_original and not args.run_improved:
        args.run_original = True
        args.run_improved = True
    
    # Set dataset-specific parameters
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
        args.k = None
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
    
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    return args

def plot_comparison(original_results, improved_results, save_path='comparison_results'):
    """
    Plot comparison metrics between original and improved models
    
    Args:
        original_results: DataFrame with original model results
        improved_results: DataFrame with improved model results
        save_path: Path to save the plots
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 8))
    
    # Original model metrics
    plt.plot(original_results['Epoch'], original_results['Acc_Q'], 'b-', label='Original Q Accuracy')
    plt.plot(original_results['Epoch'], original_results['Acc_Z'], 'g-', label='Original Z Accuracy')
    plt.plot(original_results['Epoch'], original_results['Acc_P'], 'r-', label='Original P Accuracy')
    
    # Improved model metrics
    plt.plot(improved_results['Epoch'], improved_results['Acc_Q'], 'b--', label='Improved Q Accuracy')
    plt.plot(improved_results['Epoch'], improved_results['Acc_Z'], 'g--', label='Improved Z Accuracy')
    plt.plot(improved_results['Epoch'], improved_results['Acc_P'], 'r--', label='Improved P Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: Original vs Improved SDCN_Spatial')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/accuracy_comparison.png')
    
    # Plot F1 score comparison
    plt.figure(figsize=(12, 8))
    
    # Original model metrics
    plt.plot(original_results['Epoch'], original_results['F1_Q'], 'b-', label='Original Q F1')
    plt.plot(original_results['Epoch'], original_results['F1_Z'], 'g-', label='Original Z F1')
    plt.plot(original_results['Epoch'], original_results['F1_P'], 'r-', label='Original P F1')
    
    # Improved model metrics
    plt.plot(improved_results['Epoch'], improved_results['F1_Q'], 'b--', label='Improved Q F1')
    plt.plot(improved_results['Epoch'], improved_results['F1_Z'], 'g--', label='Improved Z F1')
    plt.plot(improved_results['Epoch'], improved_results['F1_P'], 'r--', label='Improved P F1')
    
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison: Original vs Improved SDCN_Spatial')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/f1_comparison.png')
    
    # Create summary table
    summary = pd.DataFrame({
        'Metric': ['Q Accuracy', 'Z Accuracy', 'P Accuracy', 'Q F1', 'Z F1', 'P F1'],
        'Original Max': [
            original_results['Acc_Q'].max(),
            original_results['Acc_Z'].max(),
            original_results['Acc_P'].max(),
            original_results['F1_Q'].max(),
            original_results['F1_Z'].max(),
            original_results['F1_P'].max()
        ],
        'Improved Max': [
            improved_results['Acc_Q'].max(),
            improved_results['Acc_Z'].max(),
            improved_results['Acc_P'].max(),
            improved_results['F1_Q'].max(),
            improved_results['F1_Z'].max(),
            improved_results['F1_P'].max()
        ],
        'Improvement': [
            f"{(improved_results['Acc_Q'].max() - original_results['Acc_Q'].max()) * 100:.2f}%",
            f"{(improved_results['Acc_Z'].max() - original_results['Acc_Z'].max()) * 100:.2f}%",
            f"{(improved_results['Acc_P'].max() - original_results['Acc_P'].max()) * 100:.2f}%",
            f"{(improved_results['F1_Q'].max() - original_results['F1_Q'].max()) * 100:.2f}%",
            f"{(improved_results['F1_Z'].max() - original_results['F1_Z'].max()) * 100:.2f}%",
            f"{(improved_results['F1_P'].max() - original_results['F1_P'].max()) * 100:.2f}%"
        ]
    })
    
    # Save summary table
    summary.to_csv(f'{save_path}/performance_summary.csv', index=False)
    print("\nPerformance Summary:")
    print(summary)
    
    # Return summary for further analysis
    return summary

def main():
    args = setup_args()
    print(f"Running comparison on {args.name} dataset with device: {args.device}")
    
    # Load dataset
    dataset = load_data(args.name)
    
    # Check if edge attributes are available
    edge_attr = None
    if hasattr(dataset, 'edge_attr') and args.use_edge_attr:
        edge_attr = dataset.edge_attr
        print(f"Using edge attributes with shape: {edge_attr.shape}")
        args.edge_dim = edge_attr.shape[1]
    else:
        args.edge_dim = args.n_input
        print(f"No edge attributes found. Using dummy edge features of dimension: {args.edge_dim}")
    
    # Create timestamp for saving results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'comparison_results_{args.name}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save arguments
    with open(f'{save_dir}/args.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
    
    original_results = None
    improved_results = None
    
    # Run original model if specified
    if args.run_original:
        print("\n=============== Running Original SDCN_Spatial ===============")
        _, original_results = train_sdcn_spatial(dataset, args, edge_attr)
        original_results.to_csv(f'{save_dir}/original_results.csv', index=False)
        print(f"Original model results saved to {save_dir}/original_results.csv")
    else:
        # Load previous results if available
        try:
            original_results = pd.read_csv('spatial_training_results.csv')
            print("Loaded existing original model results")
        except:
            print("No original model results found and --run_original not specified")
    
    # Run improved model if specified
    if args.run_improved:
        print("\n=============== Running Improved SDCN_Spatial ===============")
        _, improved_results = train_sdcn_spatial_improved(dataset, args, edge_attr)
        improved_results.to_csv(f'{save_dir}/improved_results.csv', index=False)
        print(f"Improved model results saved to {save_dir}/improved_results.csv")
    else:
        # Load previous results if available
        try:
            improved_results = pd.read_csv('spatial_improved_training_results.csv')
            print("Loaded existing improved model results")
        except:
            print("No improved model results found and --run_improved not specified")
    
    # Generate comparison plots and summary if both results are available
    if original_results is not None and improved_results is not None:
        summary = plot_comparison(original_results, improved_results, save_dir)
        print(f"\nComparison results saved to {save_dir}")
    else:
        print("\nCannot generate comparison - need both original and improved results")

if __name__ == "__main__":
    main()