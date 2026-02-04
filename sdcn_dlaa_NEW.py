from __future__ import print_function, division
import argparse
import json
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
import math

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
        self.precomputed_edge_to_edge_kind = None
        self.precomputed_edge_to_edge_sig = None
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
        self.spatial_conv5 = SpatialConv(n_clusters, edge_dim=self.edge_dim, dropout=dropout, heads=heads, out_activation=F.leaky_relu)
        
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
        # NOTE: Edge features are already projected inside SpatialConv variants (edge_dim_proj),
        # so we keep the raw edge_attr stable here. A trainable projection would change edge_attr
        # over training and break caching for edge-sim ee graphs.
        if edge_dim is not None and edge_dim != n_input:
            self.initial_edge_proj = nn.Identity()

        # Optional: edge reconstruction head (aligns training objective with edge_attr signal).
        # Enabled by setting SDCN_EDGE_RE_WEIGHT>0 during training.
        self.edge_recon_head = nn.Linear(int(n_z), int(self.edge_dim))
        self._last_edge_recon = None
        self._last_edge_recon_target = None

        # Optional: reconstruct per-node pooled edge statistics (matches strong kmeans_edge_mean prior).
        # Enabled by setting SDCN_POOL_RE_WEIGHT>0 during training.
        self.pool_recon_head = nn.Linear(int(n_z), int(self.edge_dim))
        self._last_pool_recon = None
        self._last_pool_target = None

        # Optional: use pooled edge_attr to drive the clustering q-head (bridges to kmeans_edge_mean baseline).
        self.pool_q_proj = nn.Linear(int(self.edge_dim), int(n_z), bias=False)
        self.pool_q_scale = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _pool_edges_to_nodes_mean(edge_feat: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        node_sum = torch.zeros((num_nodes, edge_feat.size(1)), device=edge_feat.device, dtype=edge_feat.dtype)
        node_cnt = torch.zeros((num_nodes, 1), device=edge_feat.device, dtype=edge_feat.dtype)

        node_sum.index_add_(0, src, edge_feat)
        node_sum.index_add_(0, dst, edge_feat)

        ones = torch.ones((edge_feat.size(0), 1), device=edge_feat.device, dtype=edge_feat.dtype)
        node_cnt.index_add_(0, src, ones)
        node_cnt.index_add_(0, dst, ones)

        return node_sum / node_cnt.clamp(min=1.0)

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
        ee_graph = os.getenv("SDCN_EE_GRAPH", "incidence").strip().lower()
        if ee_graph in {"", "default"}:
            ee_graph = "incidence"
        if ee_graph not in {"incidence", "incidence_sim", "inc_sim", "edge_sim", "sim", "hybrid", "none", "off", "0"}:
            raise ValueError(
                f"Unknown SDCN_EE_GRAPH={ee_graph!r}. Use one of: incidence, incidence_sim, edge_sim, hybrid, none."
            )
        if ee_graph in {"sim"}:
            ee_graph = "edge_sim"
        if ee_graph in {"inc_sim"}:
            ee_graph = "incidence_sim"
        if ee_graph in {"none", "off", "0"}:
            ee_graph = "none"

        num_nodes = x.size(0)
        
        # Important fix: Add validation for node count and feature dimensions
        if num_nodes < 4:
            raise ValueError(f"Number of nodes must be greater than 3, current is {num_nodes}")
            
        # Check if x contains valid features
        if not torch.is_floating_point(x) or torch.isnan(x).any():
            raise ValueError("Node features contain invalid values (NaN or non-float numbers)")

        if edge_attr is None:
            raise ValueError("edge_attr must not be None for SDCN_DLAA SpatialConv variants")

        # Process edge features (including optional initial projection) consistently,
        # regardless of whether graph indices are cached/precomputed.
        dist_feat = edge_attr
        if self.initial_edge_proj is not None:
            dist_feat = self.initial_edge_proj(dist_feat)

        def _edge_sig(t: torch.Tensor) -> tuple[int, int, float, float]:
            # Signature for caching edge-dependent ee graphs within a run.
            # Uses exact sums (not rounded) so repeated calls match.
            return (int(t.size(0)), int(t.size(1)), float(t.sum().item()), float(t.abs().sum().item()))

        def _build_edge_to_edge_incidence(edge_index: torch.Tensor, num_edges: int) -> torch.Tensor:
            node_to_edges = defaultdict(list)
            for i in range(num_edges):
                src, dst = edge_index[0, i], edge_index[1, i]
                node_to_edges[src.item()].append(i)
                node_to_edges[dst.item()].append(i)

            edge_to_edge_list: list[list[int]] = []
            for _, connected_edges in node_to_edges.items():
                if len(connected_edges) <= 1:
                    continue
                if len(connected_edges) > max_edges_per_node:
                    sampled_edges = random.sample(connected_edges, max_edges_per_node)
                else:
                    sampled_edges = connected_edges
                for i in range(len(sampled_edges)):
                    for j in range(i + 1, len(sampled_edges)):
                        edge_i = sampled_edges[i]
                        edge_j = sampled_edges[j]
                        edge_to_edge_list.append([edge_i, edge_j])
                        edge_to_edge_list.append([edge_j, edge_i])
            if edge_to_edge_list:
                return torch.tensor(edge_to_edge_list, dtype=torch.long, device=x.device).t()
            return torch.zeros((2, 0), dtype=torch.long, device=x.device)

        def _build_edge_to_edge_incidence_sim(edge_index: torch.Tensor, edge_feat: torch.Tensor, num_edges: int) -> torch.Tensor:
            ee_topk_env = os.getenv("SDCN_EE_TOPK", "").strip()
            ee_topk = int(ee_topk_env) if ee_topk_env else int(max_edges_per_node)
            ee_topk = max(1, int(ee_topk))

            mutual = os.getenv("SDCN_EE_SIM_MUTUAL", "").strip().lower() in {"1", "true", "yes", "y", "on"}
            min_sim_env = os.getenv("SDCN_EE_SIM_MIN_SIM", "").strip()
            min_sim: float | None = float(min_sim_env) if min_sim_env else None

            # Build node->edges adjacency (incidence).
            node_to_edges = defaultdict(list)
            for i in range(num_edges):
                src, dst = edge_index[0, i], edge_index[1, i]
                node_to_edges[src.item()].append(i)
                node_to_edges[dst.item()].append(i)

            # Pre-normalize all edge features for cosine similarity.
            feat_norm = F.normalize(edge_feat, p=2, dim=1)

            all_src: list[torch.Tensor] = []
            all_dst: list[torch.Tensor] = []
            for _, connected_edges in node_to_edges.items():
                if len(connected_edges) <= 1:
                    continue
                # Cap per-node edge set to keep compute bounded and deterministic.
                if len(connected_edges) > max_edges_per_node:
                    connected_edges = connected_edges[: int(max_edges_per_node)]

                m = len(connected_edges)
                if m <= 1:
                    continue
                k = min(int(ee_topk), m - 1)
                if k <= 0:
                    continue

                idx = torch.tensor(connected_edges, dtype=torch.long, device=edge_feat.device)
                f = feat_norm.index_select(0, idx)  # [m, d]
                sims = f @ f.t()  # [m, m]
                sims.fill_diagonal_(-float("inf"))

                vals, jj = torch.topk(sims, k=k, dim=1)
                src = idx.unsqueeze(1).expand(-1, k)  # [m, k]
                dst = idx.index_select(0, jj.reshape(-1)).reshape(m, k)

                if min_sim is not None:
                    mask = vals >= float(min_sim)
                    if not bool(mask.any()):
                        continue
                    all_src.append(src[mask].reshape(-1))
                    all_dst.append(dst[mask].reshape(-1))
                else:
                    all_src.append(src.reshape(-1))
                    all_dst.append(dst.reshape(-1))

            if not all_src:
                return torch.zeros((2, 0), dtype=torch.long, device=edge_feat.device)

            src = torch.cat(all_src, dim=0)
            dst = torch.cat(all_dst, dim=0)

            ids = src.to(torch.int64) * int(num_edges) + dst.to(torch.int64)
            ids = torch.unique(ids)
            src = (ids // int(num_edges)).to(torch.long)
            dst = (ids % int(num_edges)).to(torch.long)
            e2e = torch.stack([src, dst], dim=0)

            if mutual:
                rev = dst.to(torch.int64) * int(num_edges) + src.to(torch.int64)
                keep = torch.isin(ids, rev)
                ids = ids[keep]
                if ids.numel() == 0:
                    return torch.zeros((2, 0), dtype=torch.long, device=edge_feat.device)
                src = (ids // int(num_edges)).to(torch.long)
                dst = (ids % int(num_edges)).to(torch.long)
                return torch.stack([src, dst], dim=0)

            # Make it effectively undirected (helps stabilize ee message passing).
            e2e_rev = torch.stack([dst, src], dim=0)
            merged = torch.cat([e2e, e2e_rev], dim=1)
            ids = merged[0].to(torch.int64) * int(num_edges) + merged[1].to(torch.int64)
            ids = torch.unique(ids)
            return torch.stack([ids // int(num_edges), ids % int(num_edges)], dim=0).to(edge_feat.device)

        def _build_edge_to_edge_edge_sim(edge_feat: torch.Tensor, num_edges: int) -> torch.Tensor:
            ee_topk_env = os.getenv("SDCN_EE_TOPK", "").strip()
            ee_topk = int(ee_topk_env) if ee_topk_env else int(max_edges_per_node)
            ee_topk = max(1, min(int(ee_topk), max(1, num_edges - 1)))

            mutual = os.getenv("SDCN_EE_SIM_MUTUAL", "").strip().lower() in {"1", "true", "yes", "y", "on"}
            min_sim_env = os.getenv("SDCN_EE_SIM_MIN_SIM", "").strip()
            min_sim: float | None = float(min_sim_env) if min_sim_env else None

            max_edges_env = os.getenv("SDCN_EE_SIM_MAX_EDGES", "").strip()
            max_edges = int(max_edges_env) if max_edges_env else 5000
            if num_edges > max_edges:
                print(
                    f"Warning: num_edges={num_edges} exceeds SDCN_EE_SIM_MAX_EDGES={max_edges}; "
                    "falling back to incidence ee graph."
                )
                return _build_edge_to_edge_incidence(edge_index, num_edges)

            chunk_env = os.getenv("SDCN_EE_SIM_CHUNK", "").strip()
            chunk = int(chunk_env) if chunk_env else 1024
            chunk = max(16, int(chunk))

            feat = F.normalize(edge_feat, p=2, dim=1)
            all_src = []
            all_dst = []
            for start in range(0, num_edges, chunk):
                end = min(num_edges, start + chunk)
                sims = feat[start:end] @ feat.t()  # [B, E]
                # Mask diagonal for these rows to avoid self-neighbors.
                row = torch.arange(start, end, device=edge_feat.device)
                sims[torch.arange(end - start, device=edge_feat.device), row] = -float("inf")
                vals, idx = torch.topk(sims, k=ee_topk, dim=1)
                src = row.unsqueeze(1).expand(-1, ee_topk)
                if min_sim is not None:
                    mask = vals >= float(min_sim)
                    if mask.any():
                        all_src.append(src[mask].reshape(-1))
                        all_dst.append(idx[mask].reshape(-1))
                else:
                    all_src.append(src.reshape(-1))
                    all_dst.append(idx.reshape(-1))
            if not all_src:
                return torch.zeros((2, 0), dtype=torch.long, device=x.device)
            src = torch.cat(all_src, dim=0)
            dst = torch.cat(all_dst, dim=0)
            if mutual:
                src_cpu = src.detach().cpu().tolist()
                dst_cpu = dst.detach().cpu().tolist()
                neigh: dict[int, set[int]] = {}
                for s, d in zip(src_cpu, dst_cpu):
                    neigh.setdefault(int(s), set()).add(int(d))
                undirected: set[tuple[int, int]] = set()
                for s, d in zip(src_cpu, dst_cpu):
                    s = int(s)
                    d = int(d)
                    if s == d:
                        continue
                    if s in neigh.get(d, set()):
                        a, b = (s, d) if s < d else (d, s)
                        undirected.add((a, b))
                if not undirected:
                    return torch.zeros((2, 0), dtype=torch.long, device=x.device)
                src_u = torch.tensor([a for a, _ in undirected], dtype=torch.long, device=edge_feat.device)
                dst_u = torch.tensor([b for _, b in undirected], dtype=torch.long, device=edge_feat.device)
                e2e = torch.stack([src_u, dst_u], dim=0)
                e2e_rev = torch.stack([dst_u, src_u], dim=0)
                return torch.cat([e2e, e2e_rev], dim=1)

            e2e = torch.stack([src, dst], dim=0)
            # Make it effectively undirected.
            e2e_rev = torch.stack([dst, src], dim=0)
            return torch.cat([e2e, e2e_rev], dim=1)
            
        # Reuse precomputed edge_index when available (edge_index is adj-dependent).
        edge_index = None
        if self.precomputed_edge_index is not None:
            max_node_idx = self.precomputed_edge_index.max().item()
            if max_node_idx >= num_nodes:
                print(
                    f"Warning: Precomputed edge index ({max_node_idx}) exceeds current node count ({num_nodes}), "
                    "recalculating..."
                )
                self.precomputed_edge_index = None
                edge_index = None
            else:
                edge_index = self.precomputed_edge_index
        
        # Create a cache key based on adjacency matrix properties and parameters
        # For sparse tensors, use a hash of indices and values
        if adj.is_sparse:
            adj_id = f"{adj._indices().sum().item()}_{adj._values().sum().item()}"
        else:
            adj_id = f"{adj.sum().item()}"

        # Important fix: Include node count and feature dimensions in cache key
        edge_sig = _edge_sig(dist_feat) if ee_graph in {"edge_sim", "hybrid", "incidence_sim"} else None
        cache_key = f"{adj_id}_{max_edges_per_node}_{num_nodes}_{x.size(1)}_{ee_graph}"
        if edge_sig is not None:
            cache_key = f"{cache_key}_{edge_sig[0]}_{edge_sig[1]}_{edge_sig[2]}_{edge_sig[3]}"

        # If edge indices and edge-to-edge indices were precomputed externally and match the current setting,
        # reuse them (this is used by training helpers that precompute incidence graphs for speed).
        pre_kind = self.precomputed_edge_to_edge_kind or "incidence"
        if (
            edge_index is not None
            and self.precomputed_edge_to_edge_index is not None
            and pre_kind == ee_graph
            and (ee_graph not in {"edge_sim", "hybrid", "incidence_sim"} or self.precomputed_edge_to_edge_sig == edge_sig)
        ):
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=dist_feat,
                dist_feat=dist_feat,
                dist_feat_order=dist_feat,
                edge_to_edge_index=self.precomputed_edge_to_edge_index,
            )
            return data
        
        # Check if we have cached this graph structure
        if cache_key in self.graph_cache:
            cached = self.graph_cache[cache_key]
            max_node_idx = cached["edge_index"].max().item()
            if max_node_idx >= num_nodes:
                print(
                    f"Warning: Cached edge index ({max_node_idx}) exceeds current node count ({num_nodes}), "
                    "recalculating..."
                )
                self.graph_cache.pop(cache_key)
            else:
                edge_index = cached["edge_index"]
                edge_to_edge_index = cached["edge_to_edge_index"]
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=dist_feat,
                    dist_feat=dist_feat,
                    dist_feat_order=dist_feat,
                    edge_to_edge_index=edge_to_edge_index,
                )
                return data
            
        # If not cached (or cache invalid), compute edge_index if we didn't reuse a precomputed one.
        if edge_index is None:
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
        
        # Create edge-to-edge graph more efficiently
        print("Building edge-to-edge graph (one-time operation)...")
        if ee_graph == "none":
            edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        elif ee_graph == "incidence_sim":
            edge_to_edge_index = _build_edge_to_edge_incidence_sim(edge_index, dist_feat, num_edges)
        elif ee_graph == "edge_sim":
            edge_to_edge_index = _build_edge_to_edge_edge_sim(dist_feat, num_edges)
        elif ee_graph == "hybrid":
            inc = _build_edge_to_edge_incidence(edge_index, num_edges)
            sim = _build_edge_to_edge_edge_sim(dist_feat, num_edges)
            if inc.numel() == 0:
                edge_to_edge_index = sim
            elif sim.numel() == 0:
                edge_to_edge_index = inc
            else:
                merged = torch.cat([inc, sim], dim=1)
                ids = merged[0].to(torch.int64) * int(num_edges) + merged[1].to(torch.int64)
                uniq = torch.unique(ids)
                edge_to_edge_index = torch.stack([uniq // int(num_edges), uniq % int(num_edges)], dim=0).to(x.device)
        else:
            edge_to_edge_index = _build_edge_to_edge_incidence(edge_index, num_edges)
        
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
        self.precomputed_edge_to_edge_kind = ee_graph
        self.precomputed_edge_to_edge_sig = edge_sig
        
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
        # Expose graph indices for optional auxiliary losses/diagnostics.
        self._last_edge_index = data.edge_index
        self._last_edge_to_edge_index = data.edge_to_edge_index
        
        # Store shapes for logging
        spatial_shapes = {}
        
        # Apply SpatialConv layers with fusion of AE features
        sigma_env = os.getenv("SDCN_SIGMA")
        sigma = float(sigma_env) if sigma_env is not None and sigma_env.strip() != "" else 0.5
        sigma = max(0.0, min(1.0, sigma))
        
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
        # Expose optional edge auxiliary logits from SpatialConv variants that support it (e.g., v6edge_ee_aux).
        self._last_edge_within_logit = getattr(self.spatial_conv4, "_last_edge_within_logit", None)

        # Optional edge reconstruction signal (from edge latents at layer-4).
        edge_re_w_env = os.getenv("SDCN_EDGE_RE_WEIGHT", "").strip()
        edge_re_w = float(edge_re_w_env) if edge_re_w_env else 0.0
        if edge_re_w != 0.0:
            edge_latent = node_edge_feat4[original_nodes:]  # [E, n_z]
            self._last_edge_recon = self.edge_recon_head(edge_latent)  # [E, edge_dim]
            self._last_edge_recon_target = data.dist_feat  # [E, edge_dim]
        else:
            self._last_edge_recon = None
            self._last_edge_recon_target = None

        # Optional per-node pooled edge_attr reconstruction (node-level edge statistics).
        pool_re_w_env = os.getenv("SDCN_POOL_RE_WEIGHT", "").strip()
        pool_re_w = float(pool_re_w_env) if pool_re_w_env else 0.0
        if pool_re_w != 0.0:
            pool_target = self._pool_edges_to_nodes_mean(data.dist_feat, data.edge_index, num_nodes=original_nodes)  # [N, edge_dim]
            self._last_pool_target = pool_target
            self._last_pool_recon = self.pool_recon_head(h4)  # [N, edge_dim]
        else:
            self._last_pool_target = None
            self._last_pool_recon = None
        
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
        
        # Calculate soft assignment (q) using Student's t-distribution.
        # Default (SDCN original): use AE latent z. For edge-driven tasks, you may want to
        # use graph-aware embeddings (h4 / fused embedding) via SDCN_Q_SOURCE.
        q_source = os.getenv("SDCN_Q_SOURCE", "z").strip().lower()
        if q_source in {"z", "ae", "latent"}:
            q_input = z
        elif q_source in {"h4", "graph", "gnn", "spatial"}:
            q_input = h4
        elif q_source in {"pool", "edge_mean", "edge"}:
            pool_raw = self._pool_edges_to_nodes_mean(data.dist_feat, data.edge_index, num_nodes=original_nodes)  # [N, edge_dim]
            q_input = self.pool_q_proj(pool_raw)  # [N, n_z]
        elif q_source in {"h4_pool", "h4pool", "mix_pool"}:
            pool_raw = self._pool_edges_to_nodes_mean(data.dist_feat, data.edge_index, num_nodes=original_nodes)  # [N, edge_dim]
            q_input = h4 + self.pool_q_scale * self.pool_q_proj(pool_raw)  # [N, n_z]
        elif q_source in {"fused", "mix", "datax", "x"}:
            q_input = data.x  # [(1-sigma)*h4 + sigma*z] with shape [N, n_z]
        else:
            raise ValueError(f"Unknown SDCN_Q_SOURCE={q_source!r}. Use one of: z, h4, h4_pool, pool, fused.")

        # Expose the q-input embedding for initialization/diagnostics (no graph retained).
        self._last_q_input = q_input.detach()

        if q_input.size(1) != self.cluster_layer.size(1):
            raise ValueError(
                f"q_input dim mismatch: q_input.shape={tuple(q_input.shape)} vs "
                f"cluster_layer.shape={tuple(self.cluster_layer.shape)}. "
                f"Check n_z and SDCN_Q_SOURCE={q_source!r}."
            )

        q = 1.0 / (1.0 + torch.sum(torch.pow(q_input.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
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

    # Optional reproducibility (mainly for experiments/debugging).
    seed_env = os.getenv("SDCN_SEED")
    seed = int(seed_env) if seed_env is not None and seed_env.strip() != "" else None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
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
    
    ee_graph = os.getenv("SDCN_EE_GRAPH", "incidence").strip().lower()
    if ee_graph in {"", "default"}:
        ee_graph = "incidence"
    if ee_graph in {"sim"}:
        ee_graph = "edge_sim"
    if ee_graph in {"inc_sim"}:
        ee_graph = "incidence_sim"
    if ee_graph in {"none", "off", "0"}:
        ee_graph = "none"
    if ee_graph not in {"incidence", "incidence_sim", "edge_sim", "hybrid", "none"}:
        raise ValueError(
            f"Unknown SDCN_EE_GRAPH={ee_graph!r}. Use one of: incidence, incidence_sim, edge_sim, hybrid, none."
        )

    edge_to_edge_index = None
    if ee_graph in {"edge_sim", "hybrid", "incidence_sim"}:
        # Build inside the model from edge features (it depends on edge_attr).
        print(f"Skipping incidence edge-to-edge precompute (SDCN_EE_GRAPH={ee_graph}).")
    elif ee_graph == "none":
        edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long).to(args.device)
    else:
        print("Building edge-to-edge connections (incidence)...")
        edge_to_edge_list = []
        max_edges_per_node = args.max_edges_per_node if hasattr(args, "max_edges_per_node") else 10
        for _, connected_edges in node_to_edges.items():
            if len(connected_edges) <= 1:
                continue
            if len(connected_edges) > max_edges_per_node:
                sampled_edges = random.sample(connected_edges, max_edges_per_node)
            else:
                sampled_edges = connected_edges
            for i in range(len(sampled_edges)):
                for j in range(i + 1, len(sampled_edges)):
                    edge_i = sampled_edges[i]
                    edge_j = sampled_edges[j]
                    edge_to_edge_list.append([edge_i, edge_j])
                    edge_to_edge_list.append([edge_j, edge_i])
        if edge_to_edge_list:
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
    model.precomputed_edge_to_edge_kind = ee_graph
    
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

    n_init_env = os.getenv("SDCN_KMEANS_N_INIT", "").strip()
    n_init = int(n_init_env) if n_init_env else 20
    kmeans_kwargs = {"n_clusters": args.n_clusters, "n_init": n_init}
    if seed is not None:
        kmeans_kwargs["random_state"] = seed
    kmeans = KMeans(**kmeans_kwargs)
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
    epochs_env = os.getenv("SDCN_EPOCHS")
    num_epochs = int(epochs_env) if epochs_env is not None and epochs_env.strip() != "" else 60
    for epoch in range(num_epochs):
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

            # Optional: smooth target distribution towards uniform (reduces early peaking / collapse).
            p_smooth_env = os.getenv("SDCN_P_SMOOTHING", "").strip()
            p_smooth = float(p_smooth_env) if p_smooth_env else 0.0
            if p_smooth > 0:
                p_smooth = max(0.0, min(1.0, p_smooth))
                k = int(p.size(1))
                uniform = torch.full_like(p, 1.0 / max(k, 1))
                p = (1.0 - p_smooth) * p + p_smooth * uniform
                p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-10)
            
            # Calculate loss (numerically stable: avoid log(0) in KL computations)
            eps = 1e-10
            q_safe = torch.clamp(q, min=eps)
            q_safe = q_safe / q_safe.sum(dim=1, keepdim=True)
            pred_safe = torch.clamp(pred, min=eps)
            pred_safe = pred_safe / pred_safe.sum(dim=1, keepdim=True)

            kl_loss = F.kl_div(q_safe.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred_safe.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, data)
            
            # Optional CE warmup (avoid overfitting to noisy targets early).
            ce_warmup_env = os.getenv("SDCN_CE_WARMUP_EPOCHS", "").strip()
            ce_warmup = int(ce_warmup_env) if ce_warmup_env else 0
            ce_scale = 1.0
            if ce_warmup > 0:
                ce_scale = min(1.0, float(epoch + 1) / float(ce_warmup))

            # Combined loss with the same weights as original SDCN
            loss = 0.1 * kl_loss + (0.01 * ce_scale) * ce_loss + re_loss

            # Optional mutual-information regularizer on pred: balanced yet confident assignments.
            mi_w_env = os.getenv("SDCN_PRED_MI_WEIGHT", "").strip()
            mi_w = float(mi_w_env) if mi_w_env else 0.0
            if mi_w != 0.0:
                mean_pred = pred_safe.mean(dim=0)  # [K]
                ent_mean = -(mean_pred.clamp(min=eps) * mean_pred.clamp(min=eps).log()).sum()
                ent_cond = -(pred_safe * pred_safe.log()).sum(dim=1).mean()
                mi_loss = ent_cond - ent_mean
                loss = loss + float(mi_w) * mi_loss
            
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
    final_assign = os.getenv("SDCN_FINAL_ASSIGN", "pred").strip().lower()
    if final_assign not in {"pred", "q", "p"}:
        raise ValueError(f"Unknown SDCN_FINAL_ASSIGN={final_assign!r}. Use one of: pred, q, p.")

    model.eval()
    try:
        with torch.no_grad():
            _, q_final, pred_final, _, _ = model(data, adj, edge_attr)
            p_final = target_distribution(q_final.data)

        if final_assign == "q":
            final_clusters = q_final.data.cpu().numpy().argmax(1)
        elif final_assign == "p":
            final_clusters = p_final.data.cpu().numpy().argmax(1)
        else:
            final_clusters = pred_final.data.cpu().numpy().argmax(1)
    except Exception as e:
        print(f"Error getting final clustering results: {str(e)}")
        # Fallback logic...
        if 'res2' in locals() and res2 is not None:
             final_clusters = res2
        elif len(results) > 0 and len(results[-1]) > 6 : # Check if previous eval results exist
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

    # Optional reproducibility (mainly for experiments/debugging).
    seed_env = os.getenv("SDCN_SEED")
    seed = int(seed_env) if seed_env is not None and seed_env.strip() != "" else None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
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
    
    ee_graph = os.getenv("SDCN_EE_GRAPH", "incidence").strip().lower()
    if ee_graph in {"", "default"}:
        ee_graph = "incidence"
    if ee_graph in {"sim"}:
        ee_graph = "edge_sim"
    if ee_graph in {"inc_sim"}:
        ee_graph = "incidence_sim"
    if ee_graph in {"none", "off", "0"}:
        ee_graph = "none"
    if ee_graph not in {"incidence", "incidence_sim", "edge_sim", "hybrid", "none"}:
        raise ValueError(
            f"Unknown SDCN_EE_GRAPH={ee_graph!r}. Use one of: incidence, incidence_sim, edge_sim, hybrid, none."
        )

    edge_to_edge_index = None
    if ee_graph in {"edge_sim", "hybrid", "incidence_sim"}:
        print(f"Skipping incidence edge-to-edge precompute (SDCN_EE_GRAPH={ee_graph}).")
    elif ee_graph == "none":
        edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long).to(args.device)
    else:
        # Build edge-to-edge connections (incidence)
        print("Building edge-to-edge connections (incidence)...")
        num_edges = edge_index.size(1)
        node_to_edges = defaultdict(list)
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            node_to_edges[src].append(i)
            node_to_edges[dst].append(i)
        edge_to_edge_list = []
        for _, connected_edges in node_to_edges.items():
            if len(connected_edges) <= 1:
                continue
            if len(connected_edges) > args.max_edges_per_node:
                sampled_edges = random.sample(connected_edges, args.max_edges_per_node)
            else:
                sampled_edges = connected_edges
            for i in range(len(sampled_edges)):
                for j in range(i + 1, len(sampled_edges)):
                    edge_i = sampled_edges[i]
                    edge_j = sampled_edges[j]
                    edge_to_edge_list.append([edge_i, edge_j])
                    edge_to_edge_list.append([edge_j, edge_i])
        if edge_to_edge_list:
            edge_to_edge_index = torch.tensor(edge_to_edge_list, dtype=torch.long).t().to(args.device)
        else:
            edge_to_edge_index = torch.zeros((2, 0), dtype=torch.long).to(args.device)
        
    if edge_to_edge_index is None:
        print(f"Precomputation complete: node-to-node edges: {edge_index.shape[1]}, edge-to-edge: deferred ({ee_graph})")
    else:
        print(f"Precomputation complete: node-to-node edges: {edge_index.shape[1]}, edge-to-edge connections: {edge_to_edge_index.shape[1]}")

    # Optional AE hidden size override for small-graph stability experiments.
    # Format: "500,500,2000" (n_enc_1,n_enc_2,n_enc_3); decoder dims are mirrored.
    enc_dims = [500, 500, 2000]
    enc_dims_env = os.getenv("SDCN_ENC_DIMS", "").strip()
    if enc_dims_env:
        parts = [p.strip() for p in enc_dims_env.split(",") if p.strip() != ""]
        if len(parts) != 3:
            raise ValueError(f"SDCN_ENC_DIMS must have 3 ints like '256,256,512', got: {enc_dims_env!r}")
        enc_dims = [int(p) for p in parts]
    dec_dims = [enc_dims[2], enc_dims[1], enc_dims[0]]
    
    # Create model using precomputed graph structure
    model = SDCN_DLAA(
        enc_dims[0], enc_dims[1], enc_dims[2], dec_dims[0], dec_dims[1], dec_dims[2],
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
    model.precomputed_edge_to_edge_kind = ee_graph
    
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    adj = adj.to(args.device)

    data = torch.Tensor(dataset.x).to(args.device)
    y = dataset.y

    # Optional: pretrain AE on reconstruction loss (for custom datasets without pretrained weights).
    pretrain_epochs_env = os.getenv("SDCN_PRETRAIN_EPOCHS", "").strip()
    pretrain_epochs = int(pretrain_epochs_env) if pretrain_epochs_env else 0
    if pretrain_epochs > 0:
        pretrain_lr_env = os.getenv("SDCN_PRETRAIN_LR", "").strip()
        pretrain_lr = float(pretrain_lr_env) if pretrain_lr_env else float(args.lr)
        pretrain_log_every_env = os.getenv("SDCN_PRETRAIN_LOG_EVERY", "").strip()
        pretrain_log_every = int(pretrain_log_every_env) if pretrain_log_every_env else 50

        print(f"AE pretraining: epochs={pretrain_epochs}, lr={pretrain_lr:g}")
        ae_optim = Adam(model.ae.parameters(), lr=pretrain_lr)
        model.ae.train()
        for ep in range(pretrain_epochs):
            ae_optim.zero_grad()
            x_bar, _, _, _, _ = model.ae(data)
            loss_ae = F.mse_loss(x_bar, data)
            loss_ae.backward()
            ae_optim.step()
            if pretrain_log_every > 0 and (ep % pretrain_log_every == 0 or ep == pretrain_epochs - 1):
                print(f"[AE pretrain] ep={ep:04d} mse={loss_ae.item():.6f}")

    # ---> Use no_grad here too for initialization inference <---
    q_source = os.getenv("SDCN_Q_SOURCE", "z").strip().lower()
    model.eval()  # Set model to eval mode for initialization inference
    with torch.no_grad():
        if q_source in {"z", "ae", "latent"}:
            _, _, _, _, z = model.ae(data)
            q_embed = z
        else:
            # Run a forward pass to populate model._last_q_input (h4 / fused embedding).
            model(data, adj, edge_attr)
            q_embed = getattr(model, "_last_q_input", None)
            if q_embed is None:
                raise RuntimeError("Expected model._last_q_input to be set in forward() when using non-z SDCN_Q_SOURCE")
    model.train()  # Switch back to train mode

    def _pool_edges_to_nodes_mean(edge_feat: torch.Tensor, ei: torch.Tensor, n: int) -> torch.Tensor:
        src = ei[0]
        dst = ei[1]
        node_sum = torch.zeros((n, edge_feat.size(1)), device=edge_feat.device, dtype=edge_feat.dtype)
        node_cnt = torch.zeros((n, 1), device=edge_feat.device, dtype=edge_feat.dtype)
        node_sum.index_add_(0, src, edge_feat)
        node_sum.index_add_(0, dst, edge_feat)
        ones = torch.ones((edge_feat.size(0), 1), device=edge_feat.device, dtype=edge_feat.dtype)
        node_cnt.index_add_(0, src, ones)
        node_cnt.index_add_(0, dst, ones)
        return node_sum / node_cnt.clamp(min=1.0)

    init_method = os.getenv("SDCN_INIT_METHOD", "kmeans_q").strip().lower()

    # Perform initial clustering using K-means.
    n_init_env = os.getenv("SDCN_KMEANS_N_INIT", "").strip()
    n_init = int(n_init_env) if n_init_env else 20
    kmeans_kwargs = {"n_clusters": args.n_clusters, "n_init": n_init}
    if seed is not None:
        kmeans_kwargs["random_state"] = seed

    if init_method in {"edge_mean", "kmeans_edge_mean", "edge"}:
        # Initialize clusters from baseline-style per-node mean(edge_attr), then map labels into q-embedding space.
        edge_pool = _pool_edges_to_nodes_mean(edge_attr, edge_index, int(data.size(0)))
        kmeans = KMeans(**kmeans_kwargs)
        y_pred = kmeans.fit_predict(edge_pool.detach().cpu().numpy())

        centers = []
        for k in range(int(args.n_clusters)):
            mask = torch.as_tensor(y_pred == k, device=q_embed.device)
            if int(mask.sum().item()) == 0:
                centers.append(q_embed.mean(dim=0))
            else:
                centers.append(q_embed[mask].mean(dim=0))
        model.cluster_layer.data = torch.stack(centers, dim=0).to(args.device)
    else:
        # Default: initialize in the same embedding space used to compute q.
        kmeans = KMeans(**kmeans_kwargs)
        y_pred = kmeans.fit_predict(q_embed.data.cpu().numpy())
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    # Evaluate initial clustering results
    if len(np.unique(y)) > 1:  # If ground truth labels exist
        eva(y, y_pred, 'pae')
    else:
        print(f"Initial clustering complete. Number of clusters: {args.n_clusters}")

    results = []

    epochs_env = os.getenv("SDCN_EPOCHS")
    num_epochs = int(epochs_env) if epochs_env is not None and epochs_env.strip() != "" else 60

    # Optional per-epoch trace for stability/interpretability diagnostics (JSONL).
    trace_jsonl = getattr(args, "trace_jsonl", "") if args is not None else ""
    trace_env = os.getenv("SDCN_TRACE_JSONL", "")
    trace_path = (trace_jsonl or trace_env).strip()
    trace_f = None
    if trace_path:
        trace_f = open(trace_path, "w", encoding="utf-8")

    def _prob_stats(prob: torch.Tensor, eps: float = 1e-10) -> dict:
        p = torch.clamp(prob, min=eps)
        p = p / p.sum(dim=1, keepdim=True)
        ent = -(p * p.log()).sum(dim=1).mean()
        maxp = p.max(dim=1).values.mean()
        return {"entropy": float(ent.item()), "max_prob_mean": float(maxp.item())}

    def _cluster_dist(assign: np.ndarray, n_clusters: int) -> dict[int, int]:
        counts = np.bincount(assign.astype(np.int64), minlength=int(n_clusters))
        return {int(i): int(counts[i]) for i in range(int(n_clusters))}

    def _collapse_flag(dist: dict[int, int], n_nodes: int, n_clusters: int) -> bool:
        if n_nodes <= 0:
            return False
        counts = np.asarray(list(dist.values()), dtype=np.int64)
        if counts.size == 0:
            return True
        effective_k = int((counts > 0).sum())
        max_frac = float(counts.max() / max(n_nodes, 1))
        return effective_k < int(n_clusters) or max_frac >= 0.90

    def _trace_write(record: dict) -> None:
        if trace_f is None:
            return
        trace_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        trace_f.flush()

    for epoch in range(num_epochs):
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

                # ---- Stability diagnostics (always available) ----
                q_stats = _prob_stats(tmp_q)
                p_stats = _prob_stats(p)
                pred_stats = _prob_stats(pred)

                # KL(P||Q) / KL(P||Pred) on eval outputs (same direction as training loss).
                eps = 1e-10
                q_safe = torch.clamp(tmp_q, min=eps)
                q_safe = q_safe / q_safe.sum(dim=1, keepdim=True)
                p_safe = torch.clamp(p, min=eps)
                p_safe = p_safe / p_safe.sum(dim=1, keepdim=True)
                pred_safe = torch.clamp(pred, min=eps)
                pred_safe = pred_safe / pred_safe.sum(dim=1, keepdim=True)

                kl_p_q = float(F.kl_div(q_safe.log(), p_safe, reduction="batchmean").item())
                kl_p_pred = float(F.kl_div(pred_safe.log(), p_safe, reduction="batchmean").item())

                dist_q = _cluster_dist(res1, args.n_clusters)
                dist_pred = _cluster_dist(res2, args.n_clusters)
                dist_p = _cluster_dist(res3, args.n_clusters)

                collapse_q = _collapse_flag(dist_q, n_nodes=int(dataset.num_nodes), n_clusters=int(args.n_clusters))
                collapse_pred = _collapse_flag(dist_pred, n_nodes=int(dataset.num_nodes), n_clusters=int(args.n_clusters))
                collapse_p = _collapse_flag(dist_p, n_nodes=int(dataset.num_nodes), n_clusters=int(args.n_clusters))

                # Alignment between the two self-supervised branches (no ground truth needed).
                align_nmi_q_pred = float(nmi_score(res1, res2))
                align_ari_q_pred = float(ari_score(res1, res2))
                align_nmi_p_pred = float(nmi_score(res3, res2))
                align_ari_p_pred = float(ari_score(res3, res2))

                trace_record = {
                    "epoch": int(epoch),
                    "n_nodes": int(dataset.num_nodes),
                    "n_clusters": int(args.n_clusters),
                    "q": {**q_stats},
                    "p": {**p_stats},
                    "pred": {**pred_stats},
                    "kl_p_q": kl_p_q,
                    "kl_p_pred": kl_p_pred,
                    "align_nmi_q_pred": align_nmi_q_pred,
                    "align_ari_q_pred": align_ari_q_pred,
                    "align_nmi_p_pred": align_nmi_p_pred,
                    "align_ari_p_pred": align_ari_p_pred,
                    "hard_q": dist_q,
                    "hard_pred": dist_pred,
                    "hard_p": dist_p,
                    "collapse_q": bool(collapse_q),
                    "collapse_pred": bool(collapse_pred),
                    "collapse_p": bool(collapse_p),
                }

                if len(np.unique(y)) > 1: 
                    acc1, f1_1, nmi1, ari1 = eva(y, res1, f'{epoch}Q')
                    acc2, f1_2, nmi2, ari2 = eva(y, res2, f'{epoch}Z')
                    acc3, f1_3, nmi3, ari3 = eva(y, res3, f'{epoch}P')

                    results.append([epoch, acc1, f1_1, nmi1, ari1, acc2, f1_2, nmi2, ari2, acc3, f1_3, nmi3, ari3])
                    trace_record["metrics"] = {
                        "q": {"acc": float(acc1), "f1": float(f1_1), "nmi": float(nmi1), "ari": float(ari1)},
                        "pred": {"acc": float(acc2), "f1": float(f1_2), "nmi": float(nmi2), "ari": float(ari2)},
                        "p": {"acc": float(acc3), "f1": float(f1_3), "nmi": float(nmi3), "ari": float(ari3)},
                    }
                else:
                    # Without labels, only save clustering results without computing evaluation metrics
                    cluster_distribution = np.bincount(res2)
                    print(f"Epoch {epoch}, Cluster distribution: {cluster_distribution}")
                    results.append([epoch] + [0] * 12)  # Placeholder padding

                _trace_write(trace_record)
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

            # Optional: smooth target distribution towards uniform (reduces early peaking / collapse).
            p_smooth_env = os.getenv("SDCN_P_SMOOTHING", "").strip()
            p_smooth = float(p_smooth_env) if p_smooth_env else 0.0
            if p_smooth > 0:
                p_smooth = max(0.0, min(1.0, p_smooth))
                k = int(p.size(1))
                uniform = torch.full_like(p, 1.0 / max(k, 1))
                p = (1.0 - p_smooth) * p + p_smooth * uniform
                p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-10)

            # Numerical stability: avoid log(0) in KL computations
            eps = 1e-10
            q_safe = torch.clamp(q, min=eps)
            q_safe = q_safe / q_safe.sum(dim=1, keepdim=True)
            pred_safe = torch.clamp(pred, min=eps)
            pred_safe = pred_safe / pred_safe.sum(dim=1, keepdim=True)

            kl_loss = F.kl_div(q_safe.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred_safe.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, data)
 
            kl_w_env = os.getenv("SDCN_KL_WEIGHT", "").strip()
            ce_w_env = os.getenv("SDCN_CE_WEIGHT", "").strip()
            re_w_env = os.getenv("SDCN_RE_WEIGHT", "").strip()
            kl_w = float(kl_w_env) if kl_w_env else 1.0
            ce_w = float(ce_w_env) if ce_w_env else 0.1
            re_w = float(re_w_env) if re_w_env else 1.0

            # Optional CE warmup (avoid overfitting to noisy targets early).
            ce_warmup_env = os.getenv("SDCN_CE_WARMUP_EPOCHS", "").strip()
            ce_warmup = int(ce_warmup_env) if ce_warmup_env else 0
            ce_scale = 1.0
            if ce_warmup > 0:
                ce_scale = min(1.0, float(epoch + 1) / float(ce_warmup))

            loss = kl_w * kl_loss + (ce_w * ce_scale) * ce_loss + re_w * re_loss

            # Optional: edge reconstruction loss (stabilizes edge-utilization; aligns with strong edge-mean baselines).
            edge_re_w_env = os.getenv("SDCN_EDGE_RE_WEIGHT", "").strip()
            edge_re_w = float(edge_re_w_env) if edge_re_w_env else 0.0
            if edge_re_w != 0.0:
                edge_pred = getattr(model, "_last_edge_recon", None)
                edge_tgt = getattr(model, "_last_edge_recon_target", None)
                if edge_pred is not None and edge_tgt is not None:
                    edge_re_loss = F.mse_loss(edge_pred, edge_tgt)
                    warm_env = os.getenv("SDCN_EDGE_RE_WARMUP_EPOCHS", "").strip()
                    warm = int(warm_env) if warm_env else 0
                    scale = 1.0
                    if warm > 0:
                        scale = min(1.0, float(epoch + 1) / float(warm))
                    loss = loss + (edge_re_w * scale) * edge_re_loss

            # Optional: node-level pooled edge_attr reconstruction loss (aligns to kmeans_edge_mean-style signal).
            pool_re_w_env = os.getenv("SDCN_POOL_RE_WEIGHT", "").strip()
            pool_re_w = float(pool_re_w_env) if pool_re_w_env else 0.0
            if pool_re_w != 0.0:
                pool_pred = getattr(model, "_last_pool_recon", None)
                pool_tgt = getattr(model, "_last_pool_target", None)
                if pool_pred is not None and pool_tgt is not None:
                    pool_re_loss = F.mse_loss(pool_pred, pool_tgt)
                    warm_env = os.getenv("SDCN_POOL_RE_WARMUP_EPOCHS", "").strip()
                    warm = int(warm_env) if warm_env else 0
                    scale = 1.0
                    if warm > 0:
                        scale = min(1.0, float(epoch + 1) / float(warm))
                    loss = loss + (pool_re_w * scale) * pool_re_loss

            # Optional: edge-edge auxiliary loss (v6 / EE-aux). This ties edge↔edge modeling to clustering.
            edge_aux_w_env = os.getenv("SDCN_EDGE_AUX_WEIGHT", "").strip()
            edge_aux_w = float(edge_aux_w_env) if edge_aux_w_env else 0.0
            edge_aux_smooth_env = os.getenv("SDCN_EDGE_AUX_SMOOTH_WEIGHT", "").strip()
            edge_aux_smooth_w = float(edge_aux_smooth_env) if edge_aux_smooth_env else 0.0
            if edge_aux_w != 0.0 or edge_aux_smooth_w != 0.0:
                edge_logit = getattr(model, "_last_edge_within_logit", None)
                edge_index = getattr(model, "_last_edge_index", None)
                edge_to_edge_index = getattr(model, "_last_edge_to_edge_index", None)
                if edge_logit is not None and edge_index is not None:
                    src = edge_index[0]
                    dst = edge_index[1]
                    same_prob = (p[src] * p[dst]).sum(dim=1).detach()  # [E]

                    # BCE on soft targets: edge head predicts within-cluster probability per edge.
                    edge_bce = F.binary_cross_entropy_with_logits(edge_logit, same_prob)

                    # Warmup to avoid destabilizing early clustering.
                    warm_env = os.getenv("SDCN_EDGE_AUX_WARMUP_EPOCHS", "").strip()
                    warm = int(warm_env) if warm_env else 0
                    scale = 1.0
                    if warm > 0:
                        scale = min(1.0, float(epoch + 1) / float(warm))

                    loss = loss + float(edge_aux_w) * float(scale) * edge_bce

                    # Optional: smooth edge head predictions along the edge↔edge graph.
                    if edge_aux_smooth_w != 0.0 and edge_to_edge_index is not None and edge_to_edge_index.numel() > 0:
                        ee_src = edge_to_edge_index[0]
                        ee_dst = edge_to_edge_index[1]
                        edge_prob = torch.sigmoid(edge_logit)
                        smooth = (edge_prob[ee_src] - edge_prob[ee_dst]).pow(2).mean()
                        loss = loss + float(edge_aux_smooth_w) * float(scale) * smooth

            # Optional entropy penalty on q (discourage uniform assignments).
            q_ent_env = os.getenv("SDCN_Q_ENTROPY_WEIGHT", "").strip()
            q_ent_w = float(q_ent_env) if q_ent_env else 0.0
            if q_ent_w != 0.0:
                q_ent = -(q_safe * q_safe.log()).sum(dim=1).mean()
                loss = loss + float(q_ent_w) * q_ent

            # Optional mutual-information regularizer on q: balanced yet confident soft assignments.
            q_mi_w_env = os.getenv("SDCN_Q_MI_WEIGHT", "").strip()
            q_mi_w = float(q_mi_w_env) if q_mi_w_env else 0.0
            if q_mi_w != 0.0:
                mean_q = q_safe.mean(dim=0)  # [K]
                ent_mean = -(mean_q.clamp(min=eps) * mean_q.clamp(min=eps).log()).sum()
                ent_cond = -(q_safe * q_safe.log()).sum(dim=1).mean()
                mi_loss = ent_cond - ent_mean
                loss = loss + float(q_mi_w) * mi_loss

            # Optional balance regularizer on q mean: KL(mean_q || uniform).
            q_bal_env = os.getenv("SDCN_Q_BALANCE_WEIGHT", "").strip()
            q_bal_w = float(q_bal_env) if q_bal_env else 0.0
            if q_bal_w != 0.0:
                mean_q = q_safe.mean(dim=0)
                mean_q = mean_q / mean_q.sum().clamp(min=eps)
                k = int(mean_q.numel())
                logk = float(math.log(max(k, 1)))
                q_bal_loss = (mean_q.clamp(min=eps) * (mean_q.clamp(min=eps).log() + logk)).sum()
                loss = loss + float(q_bal_w) * q_bal_loss

            # Optional entropy penalty on pred (discourage uniform predictions).
            pred_ent_env = os.getenv("SDCN_PRED_ENTROPY_WEIGHT", "").strip()
            pred_ent_w = float(pred_ent_env) if pred_ent_env else 0.0
            if pred_ent_w != 0.0:
                pred_ent = -(pred_safe * pred_safe.log()).sum(dim=1).mean()
                loss = loss + float(pred_ent_w) * pred_ent

            # Optional mutual-information regularizer on pred: balanced yet confident assignments.
            mi_w_env = os.getenv("SDCN_PRED_MI_WEIGHT", "").strip()
            mi_w = float(mi_w_env) if mi_w_env else 0.0
            if mi_w != 0.0:
                mean_pred = pred_safe.mean(dim=0)  # [K]
                ent_mean = -(mean_pred.clamp(min=eps) * mean_pred.clamp(min=eps).log()).sum()
                ent_cond = -(pred_safe * pred_safe.log()).sum(dim=1).mean()
                mi_loss = ent_cond - ent_mean
                loss = loss + float(mi_w) * mi_loss

            # Optional balance regularizer on pred mean: KL(mean_pred || uniform).
            pred_bal_env = os.getenv("SDCN_PRED_BALANCE_WEIGHT", "").strip()
            pred_bal_w = float(pred_bal_env) if pred_bal_env else 0.0
            if pred_bal_w != 0.0:
                mean_pred = pred_safe.mean(dim=0)
                mean_pred = mean_pred / mean_pred.sum().clamp(min=eps)
                k = int(mean_pred.numel())
                logk = float(math.log(max(k, 1)))
                pred_bal_loss = (mean_pred.clamp(min=eps) * (mean_pred.clamp(min=eps).log() + logk)).sum()
                loss = loss + float(pred_bal_w) * pred_bal_loss
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, KL: {kl_loss.item():.4f}, CE: {ce_loss.item():.4f}, RE: {re_loss.item():.4f}")
        except Exception as e:
            print(f"Epoch {epoch} Training error: {str(e)}")
            continue

    if trace_f is not None:
        trace_f.close()
    
    # ---> Use model.eval() and no_grad() for final inference <---
    final_assign = os.getenv("SDCN_FINAL_ASSIGN", "pred").strip().lower()
    if final_assign not in {"pred", "q", "p"}:
        raise ValueError(f"Unknown SDCN_FINAL_ASSIGN={final_assign!r}. Use one of: pred, q, p.")

    model.eval()
    try:
        with torch.no_grad():
            _, q_final, pred_final, _, _ = model(data, adj, edge_attr)
            p_final = target_distribution(q_final.data)

        if final_assign == "q":
            final_clusters = q_final.data.cpu().numpy().argmax(1)
        elif final_assign == "p":
            final_clusters = p_final.data.cpu().numpy().argmax(1)
        else:
            final_clusters = pred_final.data.cpu().numpy().argmax(1)
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
