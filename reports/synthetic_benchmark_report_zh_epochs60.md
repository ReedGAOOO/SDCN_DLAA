# Synthetic Benchmark (suite_seed=0, epochs=60)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_epochs60/aggregate.json`
- Total runs: `90`
- Run config: suite_seed=0; seeds=0,1,2; epochs=60; heads=1; max_edges_per_node=10; variants=v2edge_single_layer,v3edge_cross_layers; baselines=kmeans_x,spectral_adj_binary,spectral_edge_distance

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
| baseline | kmeans_x | 0.5500 ± 0.0167 | 0.1636 ± 0.0034 | 0.1748 ± 0.0039 | 0.5382 ± 0.0182 | 0.00 |
| baseline | spectral_adj_binary | 0.9944 | 0.9742 | 0.9833 | 0.9944 | 0.00 |
| baseline | spectral_edge_distance | 0.9944 | 0.9742 | 0.9833 | 0.9944 | 0.00 |
| model | v2edge_single_layer | 0.3759 ± 0.0545 | 0.0537 ± 0.0567 | 0.0133 ± 0.0231 | 0.2379 ± 0.0830 | 0.67 |
| model | v3edge_cross_layers | 0.3926 ± 0.0695 | 0.0677 ± 0.0510 | 0.0256 ± 0.0416 | 0.2664 ± 0.1058 | 0.67 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.5889 | 0.1978 ± 0.0000 | 0.1863 | 0.5890 | 0.00 |
| baseline | spectral_edge_distance | 0.6815 ± 0.0032 | 0.2686 ± 0.0056 | 0.2657 ± 0.0053 | 0.6766 ± 0.0029 | 0.00 |
| model | v2edge_single_layer | 0.3630 ± 0.0321 | 0.0352 ± 0.0246 | 0.0063 ± 0.0110 | 0.2259 ± 0.0623 | 0.67 |
| model | v3edge_cross_layers | 0.3741 ± 0.0432 | 0.0161 ± 0.0036 | 0.0034 ± 0.0063 | 0.2801 ± 0.1132 | 0.67 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.5106 ± 0.0026 | 0.0003 ± 0.0002 | -0.0041 ± 0.0002 | 0.5104 ± 0.0026 | 0.00 |
| baseline | spectral_adj_binary | 0.8955 | 0.5673 | 0.6239 | 0.8947 | 0.00 |
| baseline | spectral_edge_distance | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| model | v2edge_single_layer | 0.5045 | 0.0088 | 0.0000 | 0.3434 | 1.00 |
| model | v3edge_cross_layers | 0.7227 ± 0.1309 | 0.1985 ± 0.1539 | 0.2412 ± 0.1968 | 0.7091 ± 0.1530 | 0.00 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.6833 ± 0.0000 | 0.2470 ± 0.0000 | 0.2765 | 0.6825 | 0.00 |
| baseline | spectral_edge_distance | 0.7000 ± 0.0000 | 0.2817 ± 0.0000 | 0.2990 | 0.6967 | 0.00 |
| model | v2edge_single_layer | 0.3593 ± 0.0257 | 0.0396 ± 0.0323 | 0.0034 ± 0.0060 | 0.2179 ± 0.0485 | 1.00 |
| model | v3edge_cross_layers | 0.3444 ± 0.0000 | 0.0210 | -0.0001 | 0.1899 | 1.00 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.4241 ± 0.0064 | 0.0426 ± 0.0004 | 0.0325 ± 0.0003 | 0.4250 ± 0.0064 | 0.00 |
| baseline | spectral_adj_binary | 0.9296 ± 0.0032 | 0.7620 ± 0.0003 | 0.7970 ± 0.0091 | 0.9304 ± 0.0029 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.7973 ± 0.0000 | 0.8544 | 0.9500 | 0.00 |
| model | v2edge_single_layer | 0.3852 ± 0.0357 | 0.0484 ± 0.0276 | 0.0104 ± 0.0091 | 0.2683 ± 0.0685 | 0.67 |
| model | v3edge_cross_layers | 0.4167 ± 0.0641 | 0.0811 ± 0.0553 | 0.0346 ± 0.0358 | 0.3158 ± 0.1151 | 0.33 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3167 ± 0.0058 | 0.0163 ± 0.0029 | -0.0001 ± 0.0022 | 0.3157 ± 0.0076 | 0.00 |
| baseline | spectral_adj_binary | 0.9500 ± 0.0000 | 0.8457 | 0.8709 | 0.9499 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.8535 ± 0.0000 | 0.8709 ± 0.0000 | 0.9501 ± 0.0000 | 0.00 |
| model | v2edge_single_layer | 0.2767 ± 0.0247 | 0.0437 ± 0.0268 | 0.0028 ± 0.0049 | 0.1503 ± 0.0429 | 1.00 |
| model | v3edge_cross_layers | 0.2750 ± 0.0218 | 0.0482 ± 0.0349 | 0.0018 ± 0.0031 | 0.1481 ± 0.0390 | 1.00 |

