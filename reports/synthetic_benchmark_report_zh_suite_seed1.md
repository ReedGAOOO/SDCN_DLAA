# Synthetic Benchmark (suite_seed=1, epochs=30)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed1_results/aggregate.json`
- Total runs: `90`
- Run config: suite_seed=1; seeds=0,1,2; epochs=30; heads=1; max_edges_per_node=10; variants=v2edge_single_layer,v3edge_cross_layers; baselines=kmeans_x,spectral_adj_binary,spectral_edge_distance

说明：这是基于 `tools/generate_synthetic_suite.py` 生成的多组合成数据集，
对比了 SDCN_DLAA 的 `SpatialConv` 版本（v2/v3）与传统聚类 baseline（KMeans、谱聚类）。

## 数据集一览

| dataset | category | N | edge_dim | knn_k | note |
|---|---|---:|---:|---:|---|
| dist_blobs_easy | distance_1d | 180 | 1 | 10 |  |
| dist_blobs_overlap | distance_1d | 180 | 1 | 10 |  |
| dist_two_moons | distance_1d | 220 | 1 | 12 |  |
| rich_edge_profiles | rich_edge | 180 | 16 | 10 | weak node features, strong edge profile signal |
| rich_geo_temporal | rich_edge | 180 | 12 | 10 | geo-temporal rich edge attrs |
| rich_multirelation | rich_edge | 200 | 20 | 10 | multi-relation rich edge attrs |

## 结果汇总（mean ± std over seeds）

### dist_blobs_easy (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.8556 | 0.5550 ± 0.0000 | 0.6156 | 0.8555 | 0.00 |
| baseline | spectral_adj_binary | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| baseline | spectral_edge_distance | 0.7870 ± 0.2431 | 0.7973 ± 0.1819 | 0.6954 ± 0.2823 | 0.7641 ± 0.2741 | 0.00 |
| model | v2edge_single_layer | 0.3500 ± 0.0147 | 0.0318 ± 0.0187 | 0.0012 ± 0.0019 | 0.2025 ± 0.0318 | 1.00 |
| model | v3edge_cross_layers | 0.5722 ± 0.1398 | 0.4810 ± 0.2971 | 0.3575 ± 0.2882 | 0.4647 ± 0.1394 | 0.33 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.4611 ± 0.0111 | 0.0442 ± 0.0035 | 0.0348 ± 0.0052 | 0.4626 ± 0.0108 | 0.00 |
| baseline | spectral_adj_binary | 0.7500 | 0.4219 ± 0.0000 | 0.4070 | 0.7543 | 0.00 |
| baseline | spectral_edge_distance | 0.7333 | 0.3944 ± 0.0000 | 0.3673 | 0.7391 | 0.00 |
| model | v2edge_single_layer | 0.3815 ± 0.0548 | 0.0285 ± 0.0151 | 0.0125 ± 0.0216 | 0.2620 ± 0.0894 | 0.67 |
| model | v3edge_cross_layers | 0.4407 ± 0.0758 | 0.0725 ± 0.0419 | 0.0371 ± 0.0542 | 0.3890 ± 0.1235 | 0.00 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.5045 | 0.0001 ± 0.0000 | -0.0045 ± 0.0000 | 0.5042 ± 0.0001 | 0.00 |
| baseline | spectral_adj_binary | 0.8955 | 0.6080 | 0.6239 | 0.8943 ± 0.0000 | 0.00 |
| baseline | spectral_edge_distance | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| model | v2edge_single_layer | 0.5091 ± 0.0079 | 0.0114 ± 0.0045 | 0.0003 ± 0.0005 | 0.3554 ± 0.0209 | 1.00 |
| model | v3edge_cross_layers | 0.5273 ± 0.0394 | 0.0138 ± 0.0088 | 0.0060 ± 0.0103 | 0.4092 ± 0.1140 | 0.67 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.4611 ± 0.0111 | 0.0442 ± 0.0035 | 0.0348 ± 0.0052 | 0.4626 ± 0.0108 | 0.00 |
| baseline | spectral_adj_binary | 0.7889 | 0.4536 ± 0.0000 | 0.4826 ± 0.0000 | 0.7892 | 0.00 |
| baseline | spectral_edge_distance | 0.7556 | 0.4239 ± 0.0000 | 0.4140 ± 0.0000 | 0.7607 | 0.00 |
| model | v2edge_single_layer | 0.3574 ± 0.0225 | 0.0194 ± 0.0027 | 0.0006 ± 0.0012 | 0.2274 ± 0.0648 | 0.67 |
| model | v3edge_cross_layers | 0.3444 ± 0.0000 | 0.0210 | -0.0001 | 0.1899 | 1.00 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3611 | 0.0048 ± 0.0010 | -0.0061 ± 0.0010 | 0.3612 ± 0.0001 | 0.00 |
| baseline | spectral_adj_binary | 0.9611 | 0.8553 ± 0.0000 | 0.8873 | 0.9607 | 0.00 |
| baseline | spectral_edge_distance | 0.9611 | 0.8553 ± 0.0000 | 0.8873 | 0.9607 | 0.00 |
| model | v2edge_single_layer | 0.3833 ± 0.0722 | 0.0456 ± 0.0528 | 0.0226 ± 0.0404 | 0.2566 ± 0.0979 | 0.67 |
| model | v3edge_cross_layers | 0.4074 ± 0.0179 | 0.0481 ± 0.0256 | 0.0211 ± 0.0189 | 0.3176 ± 0.0304 | 0.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3033 ± 0.0076 | 0.0170 ± 0.0070 | -0.0000 ± 0.0044 | 0.2989 ± 0.0094 | 0.00 |
| baseline | spectral_adj_binary | 0.9550 | 0.8512 ± 0.0000 | 0.8824 ± 0.0000 | 0.9550 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.8405 ± 0.0000 | 0.8700 | 0.9500 | 0.00 |
| model | v2edge_single_layer | 0.3167 ± 0.0981 | 0.0821 ± 0.0932 | 0.0478 ± 0.0827 | 0.1820 ± 0.1060 | 0.67 |
| model | v3edge_cross_layers | 0.3267 ± 0.0732 | 0.0737 ± 0.0393 | 0.0271 ± 0.0329 | 0.2281 ± 0.1132 | 0.67 |

