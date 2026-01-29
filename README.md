# SDCN_DLAA

Structural Deep Clustering Network (SDCN) with Dual-Level Attentive Aggregation (DLAA). This repo implements a `SpatialConv`-style node↔edge and edge↔edge message passing (PyTorch Geometric) to inject edge semantics into node clustering.

## Highlights (Simplified Innovation)

- **Deep edge-aware clustering (SDCN + DLAA)**: dual-level aggregation (node↔edge, edge↔edge) so edge semantics can influence node clustering.
- **Default best variant (`v5edge_pool_residual`)**: most robust on edge-semantic synthetic benchmarks (see `reports/realistic_synthetic_ablation_zh.md`).
- **Variant-friendly SpatialConv**: `v1original`, `v2edge_single_layer`, `v3edge_cross_layers`, `v4edge_pool_fusion`, `v5edge_pool_residual` via `SPATIALCONV_VARIANT` for ablation/verification.
- **Edge message injection (optional)**: set `SDCN_EDGE_MESSAGE=1` to let `edge_attr` affect node updates as message content (not just attention weights), helpful when node features are weak.
- **Experiment tooling**: `SDCN_SEED` / `SDCN_EPOCHS` and `tools/` scripts for conceptual/synthetic comparison.

## Recommended Defaults

- Default variant: **`v5edge_pool_residual`** (best overall on edge-semantic synthetic benchmark; see `reports/realistic_synthetic_ablation_zh.md`).
- Recommended training knobs for edge-driven datasets:
  - `SDCN_Q_SOURCE=h4`
  - `SDCN_EDGE_MESSAGE=1`
  - `SDCN_FINAL_ASSIGN=p`
  - For profile-like edge features: add `--edge_attr_norm zscore_clip` (via `tools/test_conceptual_data.py` / `tools/sweep_stability.py`)

## v5 Structure (Recommended)

`v5edge_pool_residual` is designed for **edge-semantic clustering**: when edges carry richer information than nodes (relationship types, interaction stats, multi-dim profiles).

**Core idea**: keep two complementary paths that are both edge-aware, but in different ways.

**Forward-pass sketch (one SpatialConv block):**

```text
Inputs: x (node feats), edge_index, dist_feat (raw edge_attr), dist_feat_order, edge_to_edge_index

edge_feat_0 = MLP([x_src, x_dst, dist_feat_order])
edge_feat_1 = ee_gat(edge_feat_0, edge_to_edge_index)          # edge↔edge context

node_att   = SGAT(x, edge_index, edge_attr=dist_feat)          # raw edge participates in attention
pooled     = mean_pool(dist_feat) + mean_pool(edge_feat_1)     # edge→node residual (both endpoints)
node_out   = node_att + sigmoid(gate([node_att, pooled])) * proj(pooled)
```

1) **Node attention uses raw edge features (V2 philosophy)**  
Node updates are computed with `SGATLayer(GATConv(edge_dim=...))`, where `edge_attr = dist_feat` directly participates in attention.

2) **Edges get refined on an edge↔edge graph (local consistency)**  
Edges are embedded from `(x_src, x_dst, dist_feat_order)` and updated by `ee_gat` on `edge_to_edge_index` (edges sharing endpoints).

3) **Explicit edge→node pooling residual (baseline-style, but learnable)**  
Mean-pool edge features to nodes (both endpoints), then fuse into node embeddings with a gate:

- pooled signal = mean_pool(raw_edge) + mean_pool(updated_edge)
- node_out = node_att + sigmoid(gate([node_att, pooled])) * proj(pooled)

This is the part that ablation shows to be “structurally necessary” on edge-semantic data.

**Default runtime selection**
- The repo default is already `v5edge_pool_residual` (see `DLAA_NEW.py`), but you can override:
  - `export SPATIALCONV_VARIANT=v5edge_pool_residual`

**Recommended knobs for v5**
- `SDCN_Q_SOURCE=h4` (cluster self-training uses graph-aware embedding)
- `SDCN_FINAL_ASSIGN=p` (use target distribution head when `pred` lags behind)
- `SDCN_EDGE_MESSAGE=1` (edge_attr also contributes as message content)

**Ablation toggles (for research)**
- `SDCN_POOL_RESIDUAL=0/1` (disable/enable pooling residual)
- `SDCN_EDGE_EE=0/1` (disable/enable edge↔edge update)
- `SDCN_POOL_GATE_MODE=learned|one|zero` (gate behavior)

## Repo Structure

- `sdcn_dlaa_NEW.py`: main training + evaluation entry (SDCN-style self-supervised clustering).
- `DLAA_NEW.py`: DLAA / `SpatialConv` implementations and variants.
- `preprocess_distance_matrix.py`: builds a sparse graph + edge features from a distance matrix.
- `NEWDATA/`: example raw + processed data outputs.
- `experiments/`: runnable entry scripts for sparse KNN / threshold graphs and quick sanity tests.
- `archive/`: archived experimental variants (AMP/hetero/hiddensize) kept for reference.
- `tools/`: synthetic (“conceptual”) dataset generator and variant comparison runner.

## Model Overview (High Level)

- **AE encoder/decoder** learns content embedding from node features.
- **Graph encoder (SpatialConv stack)** aggregates structure using `edge_index` + distance-based edge features.
- **Clustering head (SDCN)** computes soft assignments (Student-t) and optimizes against a sharpened target distribution.

## Quick Start

### 1) Preprocess (KNN graph)

```bash
python preprocess_distance_matrix.py --output_dir NEWDATA/processed_knn_k10 --method knn --k 10
```

Defaults expect:
- node features: `NEWDATA/X_simplize.CSV`
- distance matrix: `NEWDATA/A.csv`

Override with `--node_features` / `--distance_matrix` if needed.

### 2) Run (sparse KNN)

```bash
python experiments/test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10
```

Common knobs:
- `--heads`: attention heads
- `--edge_dim`: edge feature dim (must match preprocessing)
- `--max_edges_per_node`: controls edge-to-edge graph density

## SpatialConv Variants (v1/v2/v3)

Select at import-time via env var (default: `v5edge_pool_residual`):

```bash
export SPATIALCONV_VARIANT=v5edge_pool_residual  # v1original | v2edge_single_layer | v3edge_cross_layers | v4edge_pool_fusion | v5edge_pool_residual
export SDCN_Q_SOURCE=h4                          # z | h4 | fused
export SDCN_FINAL_ASSIGN=p                       # pred | q | p (choose which head to output as final clusters)
export SDCN_SEED=0                              # optional: reproducible runs
export SDCN_EPOCHS=30                           # optional: override epochs
export SDCN_EDGE_MESSAGE=1                      # optional: edge_attr as message content
python experiments/test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10 --heads 1
```

Variant intent:
- `v1original`: legacy baseline(original design).
- `v2edge_single_layer`: minimal fix (ensures edge features participate in attention, avoids washing edge rows).
- `v3edge_cross_layers`: uses updated edge embeddings as `edge_attr` for node attention.
- `v4edge_pool_fusion`: v3 + explicit edge→node pooling residual (gated).
- `v5edge_pool_residual` (recommended): node attention uses raw edge features + explicit edge→node pooling residual; most robust in edge-semantic tests.

## Other Runs

```bash
# Threshold-sparsified graph
python experiments/test_sdcn_dlaa_NEW_sparse_threshold.py --data_dir NEWDATA/processed_threshold_0.5
```

## Synthetic Conceptual Benchmark (Optional)

```bash
python tools/generate_conceptual_data.py --output_dir /tmp/sdcn_dlaa_concept_data --seed 0
python tools/compare_spatialconv_variants.py \
  --data_dir /tmp/sdcn_dlaa_concept_data \
  --out_dir /tmp/sdcn_dlaa_variant_compare \
  --seeds 0,1,2 \
  --epochs 30 \
  --variants v1original,v2edge_single_layer,v3edge_cross_layers,v4edge_pool_fusion,v5edge_pool_residual \
  --heads 1
```

Synthetic suite (multiple datasets + baselines):

```bash
python tools/generate_synthetic_suite.py --output_root /tmp/sdcn_dlaa_synth_suite --seed 0
python tools/benchmark_synthetic_suite.py \
  --suite_dir /tmp/sdcn_dlaa_synth_suite \
  --out_dir /tmp/sdcn_dlaa_synth_results \
  --seeds 0,1,2 \
  --epochs 30 \
  --variants v2edge_single_layer,v3edge_cross_layers \
  --baselines kmeans_x,spectral_adj_binary,spectral_edge_distance \
  --heads 1
```

Example result tables and deeper explanations live in this README.
