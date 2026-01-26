# SDCN_DLAA

Structural Deep Clustering Network (SDCN) with Dual-Level Attentive Aggregation (DLAA). This repo implements a `SpatialConv`-style node↔edge and edge↔edge message passing (PyTorch Geometric) to inject edge semantics into node clustering.

For the full documentation, see:
- `readme_full.md` (EN)
- `readme_zh_full.md` (中文)

## Highlights (Simplified Innovation)

- **Deep edge-aware clustering (SDCN + DLAA)**: dual-level aggregation (node↔edge, edge↔edge) so edge semantics can influence node clustering.
- **Variant-friendly SpatialConv**: `v1original`, `v2edge_single_layer` (default), `v3edge_cross_layers` via `SPATIALCONV_VARIANT` for ablation/verification.
- **Experiment tooling**: `SDCN_SEED` / `SDCN_EPOCHS` and `tools/` scripts for conceptual/synthetic comparison.

## Repo Structure

- `sdcn_dlaa_NEW.py`: main training + evaluation entry (SDCN-style self-supervised clustering).
- `DLAA_NEW.py`: DLAA / `SpatialConv` implementations and variants.
- `preprocess_distance_matrix.py`: builds a sparse graph + edge features from a distance matrix.
- `NEWDATA/`: example raw + processed data outputs.
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
python test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10
```

Common knobs:
- `--heads`: attention heads
- `--edge_dim`: edge feature dim (must match preprocessing)
- `--max_edges_per_node`: controls edge-to-edge graph density

## SpatialConv Variants (v1/v2/v3)

Select at import-time via env var (default: `v2edge_single_layer`):

```bash
export SPATIALCONV_VARIANT=v2edge_single_layer  # v1original | v2edge_single_layer | v3edge_cross_layers
export SDCN_SEED=0                              # optional: reproducible runs
export SDCN_EPOCHS=30                           # optional: override epochs
python test_sdcn_dlaa_NEW_sparse_KNN.py --data_dir NEWDATA/processed_knn_k10 --heads 1
```

Variant intent:
- `v1original`: legacy baseline.
- `v2edge_single_layer`: minimal fix (ensures edge features participate in attention, avoids washing edge rows).
- `v3edge_cross_layers`: uses updated edge embeddings as `edge_attr` for node attention.

## Other Runs

```bash
# Threshold-sparsified graph
python test_sdcn_dlaa_NEW_sparse_threshold.py --data_dir NEWDATA/processed_threshold_0.5
```

## Synthetic Conceptual Benchmark (Optional)

```bash
python tools/generate_conceptual_data.py --output_dir /tmp/sdcn_dlaa_concept_data --seed 0
python tools/compare_spatialconv_variants.py \
  --data_dir /tmp/sdcn_dlaa_concept_data \
  --out_dir /tmp/sdcn_dlaa_variant_compare \
  --seeds 0,1,2 \
  --epochs 30 \
  --variants v1original,v2edge_single_layer,v3edge_cross_layers \
  --heads 1
```

Example result tables and deeper explanations live in `readme_full.md`.
