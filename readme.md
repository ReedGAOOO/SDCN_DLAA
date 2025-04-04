# SDCN_DLAA: Structural Deep Clustering Network with Dual-Level Attentive Aggregation Mechanism

## INTRODUCTION

### Innovation 1: Graph Self-Supervised Clustering Framework Integrating Deep Edge Information Modeling (sdcn_dlaa_NEW.py)
1. This project innovatively introduces a dual-level graph aggregation mechanism (inspired by SMAN's SpatialConv concept) into the graph self-supervised clustering framework (SDCN). This mechanism achieves deep modeling and dynamic utilization of edge features through explicit, iterative information passing between nodes↔edges and edges↔edges. This significantly differs from mainstream GNN layers (like GATConv/GCNConv in PyG) which typically handle edge information statically (e.g., only for attention calculation or simple concatenation).
2. This deep edge modeling enables the model to more accurately capture complex inter-node relationships defined by connection strength, type, or attributes, leading to more scientific node clustering that better aligns with the underlying graph structure. The framework is particularly suitable for application scenarios where edge information carries rich semantics, such as geospatial network analysis (e.g., trajectory point clustering, considering road grades, travel times), social network mining (analyzing user communities, considering relationship types, intimacy), and molecular property prediction (considering chemical bond types, bond energies).

### Innovation 2: Implementation and Application of Dual-Level Aggregation under the PyTorch Geometric Framework (DLAA_NEW.py)
This project implements the dual-level graph aggregation mechanism (core idea from the SMAN model) completely and efficiently within the PyTorch Geometric (PyG) framework for the first time. Adaptations and optimizations (e.g., memory optimization, parallelization improvements) have been made for PyG's data representation and message passing features (custom message passing). This provides an important implementation foundation and application example for applying such interactive node-edge joint modeling techniques within the mainstream PyG ecosystem, especially validating its effectiveness in self-supervised clustering tasks.

This project is still experimental and currently in the parameter tuning phase to address unstable learning rates during training. However, the feasibility of the SDCN_DLAA framework has been theoretically demonstrated (see engineering note for details: https://docs.google.com/document/d/1qZmEbDUiWt8VqjI-uQMnlMk5Skk1pxMWrIq58-LcI8I/edit?usp=sharing). Furthermore, current runs on `test_sdcn_dlaa_NEW_sparse.py` indicate its ability to incorporate EDGE FEATURES into NODE CLUSTERING and capture meaningful information.

## QUICK START
Here are the basic steps to quickly run the model:

1.  **Data Preprocessing (using KNN, K=10):**

    ```bash
    python preprocess_distance_matrix.py --output_dir NEWDATA/processed_knn_k10 --method knn --k 10
    ```

2.  **Model Testing (using KNN preprocessed data):**

    ```bash
    python test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10
    ```

*Note: Ensure `NEWDATA/X_simplize.CSV` and `NEWDATA/A.csv` files exist in the default paths, or specify their paths using the `--node_features` and `--distance_matrix` arguments. Refer to subsequent sections for detailed parameters.*

## Data Preprocessing

Before using the model for training or testing, the raw data needs to be preprocessed using the `preprocess_distance_matrix.py` script. This script primarily performs the following tasks:

1.  **Load Data**: Reads node features (e.g., `NEWDATA/X_simplize.CSV`) and the distance matrix between nodes (e.g., `NEWDATA/A.csv`).
2.  **Graph Construction**: Builds the graph structure from the distance matrix based on the specified sparsification method (KNN or Threshold).
    *   **KNN**: Retains the K nearest neighbors for each node, generating an undirected graph.
    *   **Threshold**: Retains edges with distances less than or equal to the specified threshold `theta`.
3.  **Edge Feature Generation**: Extracts the original edge distances corresponding to the graph structure, normalizes them (optional), and expands them to the specified dimension to generate the final edge features.
4.  **Save Output**: Saves the processed node features, graph structure (edge index `edge_index`, sparse adjacency matrix `binary_adj.npz`), and edge features (`edge_attr`) to the specified output directory in `.npy` and `.pt` formats. A `data_info.txt` file recording data information is also generated.

### Usage

Run the script via the command line and specify the relevant parameters.

**Input Files:**

*   `--node_features`: Path to the CSV file for the node feature matrix (Default: `NEWDATA/X_simplize.CSV`).
*   `--distance_matrix`: Path to the CSV file for the adjacency matrix containing actual distances between nodes (Default: `NEWDATA/A.csv`).

**Main Parameters:**

*   `--output_dir`: Specify the directory to save the processed data.
*   `--method`: Graph sparsification method, options: `'knn'` or `'threshold'` (Default: `'knn'`).
*   `--k`: Specify the K value when `method='knn'` (Default: `10`).
*   `--theta`: Specify the distance threshold when `method='threshold'` (Required).
*   `--normalize`: Edge distance normalization method, options: `'minmax'`, `'standard'`, `'none'` (Default: `'minmax'`).
*   `--edge_dim`: Target dimension for the final edge features (Default: `10`).
*   `--visualize`: Add this flag to generate data visualization charts.

**Example Commands:**

1.  **Using KNN method (K=15):**

    ```bash
    python preprocess_distance_matrix.py `
    --node_features NEWDATA/X_simplize.CSV `
    --distance_matrix NEWDATA/A.csv `
    --output_dir NEWDATA/processed_knn_k15 `
    --method knn `
    --k 15 `
    --normalize minmax `
    --edge_dim 10 `
    --visualize
    ```

2.  **Using Threshold method (threshold theta=0.5):**
    *(Note: The value of `theta` needs to be adjusted based on the actual data)*

    ```bash
    python preprocess_distance_matrix.py `
        --node_features NEWDATA/X_simplize.CSV `
        --distance_matrix NEWDATA/A.csv `
        --output_dir NEWDATA/processed_threshold_0.5 `
        --method threshold `
        --theta 0.5 `
        --normalize minmax `
        --edge_dim 10 `
        --visualize
    ```

After processing, the generated `.pt` files can be directly used for subsequent model training and evaluation scripts.

## Model Testing

After preprocessing, the following scripts can be used to test the SDCN-DLAA model, which load the preprocessed data:

*   **`test_sdcn_dlaa_NEW_sparse_KNN.py`**: Used for testing data preprocessed with the **KNN** method.
*   **`test_sdcn_dlaa_NEW_sparse_threshold.py`**: Used for testing data preprocessed with the **Threshold** method.

### Usage

Run the corresponding test script via the command line. Both scripts accept a key argument `--data_dir` to specify the directory containing the preprocessed data (`node_features.npy`, `binary_adj.npz`, `edge_attr.npy`).

**Main Parameters:**

*   `--data_dir`: Path to the directory containing preprocessed data.
*   `--lr`: Learning rate (Default: `1e-3`).
*   `--n_clusters`: Target number of clusters (Default: `3`).
*   `--n_z`: Embedding dimension (Default: `10`).
*   `--dropout`: Dropout rate (Default: `0.2`).
*   `--heads`: Number of attention heads in GAT (Default: `4`).
*   `--edge_dim`: Input edge feature dimension (should match `--edge_dim` in the preprocessing script, Default: `10`).
*   `--max_edges_per_node`: Maximum number of edges considered per node when building the edge-to-edge graph (Default: `10`).

**Example Commands:**

1.  **Testing KNN data (assuming data is in `NEWDATA/processed_knn_k15`):**

    ```bash
    python test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k15
    ```

2.  **Testing Threshold data (assuming data is in `NEWDATA/processed_threshold_0.5`):**

    ```bash
    python test_sdcn_dlaa_NEW_sparse_threshold.py --data_dir NEWDATA/processed_threshold_0.5
    ```

The script will load data from the specified directory, train the model, and output the training process and clustering results to a log file (located in the `logs/` directory, filename includes timestamp and method name) and the console.


## EXPERIMENTAL TEST CODE

This section introduces code variants used for experimental testing and analysis, primarily aimed at addressing Out-Of-Memory (OOM) issues encountered when using dense graphs as input.

### Mixed Precision Training (AMP)

*   **Related Files**: `sdcn_dlaa_NEW_amp.py`, `test_sdcn_dlaa_NEW_amp.py`, `run_batch_test_amp.py`
*   **Purpose**: These files with the `_amp` suffix utilize Automatic Mixed Precision (AMP) technology for training. AMP uses lower-precision floating-point numbers (like FP16) for some computations while maintaining FP32 precision for critical parts. This effectively reduces GPU memory usage and computation time without significantly sacrificing model performance. It is particularly useful for handling graph data with a large number of nodes and dense edge connections, alleviating memory bottlenecks.

### Heterogeneous Graph Convolution (HeteroConv)

*   **Related Files**: `DLAA_NEW_hetero.py`, `sdcn_dlaa_NEW_hetero.py`, `test_sdcn_dlaa_NEW_hetero.py`
*   **Purpose**: These files with the `_hetero` suffix employ the `HeteroConv` module from the PyTorch Geometric (PyG) library. Unlike the `SpatialConv` used in the original model (which implicitly models edge information in the convolution layer), `HeteroConv` allows for the explicit definition and handling of different types of nodes and edges and their relationships. This offers a more flexible way to model complex interactions within the graph, potentially helping to capture finer structural details. By reshaping the implicit relationship modeling of `SpatialConv`, it explores different graph information aggregation strategies. Additionally, using `HeteroConv` avoids the OOM issue caused by concatenating large node-edge vectors in `SpatialConv`.

### Parameter Sensitivity Testing

To locate and understand potential memory bottlenecks when the model processes dense graphs, the following test scripts were designed:

*   **Batch Test (GAT Heads)**
    *   **Related Files**: `run_batch_test.py`, `run_batch_test_amp.py`
    *   **Purpose**: These scripts perform a series of tests by systematically varying the number of attention heads (`heads` parameter) in the Graph Attention Network (GAT) layer. Changing `heads` affects the model's complexity and computational load. Running these tests helps analyze the impact of different numbers of attention heads on model performance and memory consumption, especially when dealing with large-scale or dense graphs.
*   **Hidden Size Test**
    *   **Related Files**: `run_hidden_size_test.py`, `sdcn_dlaa_NEW_hiddensize.py`, `test_sdcn_dlaa_NEW_hiddensize.py`
    *   **Purpose**: These scripts focus on testing the impact of the dimension of hidden representations (`hidden_size` parameter) in different layers of the model on performance and resource consumption. By varying the size of these intermediate vectors, the trade-off between model capacity and memory requirements can be evaluated, helping to determine the optimal or feasible hidden layer dimension configuration when processing graph data of a specific scale (especially dense graphs).

## Theoretical Analysis:
### Review of SDCN Self-Supervised Clustering Core Principles
The core of self-supervision in the original SDCN (and many deep clustering methods) lies in constructing a high-confidence target distribution P to guide the learning of the current soft clustering assignment Q. The process is typically as follows:
Obtain Node Representation (Latent Representation z): Extract a low-dimensional representation z of the node's own features using an Autoencoder (AE), and simultaneously extract a representation h that considers the neighborhood structure using a GNN (e.g., Graph Convolutional Network, GCN). z and h might be fused in some way (e.g., addition, concatenation followed by dimensionality reduction), or the AE's z might be used directly as input for clustering.
Calculate Current Soft Assignment (q): Based on the node representation z and a set of learnable cluster centers μ, calculate the probability q_ij that each node i belongs to each cluster j using a similarity measure (e.g., Student's t-distribution).
q_ij = (1 + ||z_i - μ_j||^2 / ν)^(-(ν+1)/2) / Σ_{k} (1 + ||z_i - μ_k||^2 / ν)^(-(ν+1)/2) [Source 105]
Construct Target Distribution (p): Generate a higher-confidence target distribution p_ij by sharpening the current soft assignment q. This is typically done by increasing the weight of high-probability assignments and decreasing the weight of low-probability assignments.
p_ij = (q_ij^2 / Σ_i q_ij) / Σ_{k} (q_ik^2 / Σ_i q_ik) [Source 105, target_distribution function]
Self-Supervision Loss (L_kl): The core loss is the KL divergence loss L_kl, used to minimize the difference between the current assignment q and the target distribution p. This drives the model to learn node representations z that form more compact and well-separated clusters in the feature space.
L_kl = KL(P || Q) = Σ_i Σ_j p_ij * log(p_ij / q_ij) [Source 105]
Auxiliary Losses:
AE Reconstruction Loss (L_rec): Ensures that z retains the original feature information of the nodes.
L_rec = ||x - x_bar||^2 [Source 105]
GNN Prediction Consistency Loss (L_ce) (Optional): Makes the direct prediction output `predict` of the GNN also fit the target distribution p.
L_ce = KL(P || Predict) [Source 105]
Key Point: The core of this self-supervised process lies in the iterative optimization between q and p. p provides the learning target for q, and improvements in q lead to a better p. The effectiveness of this cycle depends on the quality of the node representation z—it needs to preserve node content information while also reflecting the graph's structural information.
### Impact of Introducing Edge Features (DLAA/SpatialConv)
In SDCN-DLAA, the main modification is replacing the standard GCN/GAT layers potentially used in the original SDCN with SpatialConv. The core of SpatialConv is the deep and dynamic modeling of edge features.
Change in Information Source: The node representations h1, h2, h3, h4, h5 output by the SpatialConv layer consider not only the features of neighboring nodes but also explicitly and dynamically consider the features and states of connecting edges (through Node↔Edge, Edge↔Edge, Edge→Node interactions).
Change in Fused Representation: These structural representations h, generated by SpatialConv and containing richer edge information, are fused (usually through weighted summation) with the representations tra1, tra2, tra3, z produced by the AE.
Change in Final Clustering Input z: Although the latent representation ultimately used to calculate q is still named z (or a representation fused with z), this z is indirectly influenced by the enhanced structural information from SpatialConv.
### Justifying Effectiveness: Why is Self-Supervision Still Effective?
Core Self-Supervision Loop Unchanged: The most crucial self-supervision mechanism—calculating q from node representation z, deriving target p from q, and then using L_kl(P||Q) to optimize model parameters (including AE and GNN/SpatialConv parameters to improve z)—this core loop remains unchanged. The model still learns the clustering structure by minimizing the difference between q and p.
Improvement in Input Representation Quality: The purpose of DLAA/SpatialConv is to provide higher-quality, more informative structural representations h. By deeply modeling edge features, h can more accurately reflect complex relationships between nodes (not just connectivity, but also the nature of the connections). When these superior h representations are fused into the final representation z used for clustering, theoretically, z should be better able to distinguish nodes that should be separated structurally (now including structure defined by edges).
Enhancement of Self-Supervision Target: A higher-quality z will likely produce an initial q with better discriminative power. The target p generated based on this better q will also be more reliable. Therefore, the KL divergence loss L_kl will guide the model parameter updates from a more optimized starting point, potentially converging to more accurate clustering results that better conform to the true graph structure (including edge information).
Constraint of Reconstruction Loss: The AE's reconstruction loss L_rec still exists, ensuring that even if the GNN part introduces complex edge information, the model does not lose the basic feature information of the nodes themselves, maintaining the fundamental validity of the representation.

## References
### SDCN
#### PAPER: https://arxiv.org/abs/2002.01633
#### GITHUB: https://github.com/bdy9527/SDCN

### SMAN
#### PAPER: https://arxiv.org/abs/2012.09624
#### GITHUB: https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/drug_target_interaction/sman