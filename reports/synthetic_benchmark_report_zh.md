# Synthetic Benchmark (suite_seed=0)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results/aggregate.json`
- Total runs: `90`
- Run config: suite_seed=0; seeds=0,1,2; epochs=30; heads=1; max_edges_per_node=10; variants=v2edge_single_layer,v3edge_cross_layers; baselines=kmeans_x,spectral_adj_binary,spectral_edge_distance

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
| model | v2edge_single_layer | 0.4074 ± 0.1043 | 0.0861 ± 0.1057 | 0.0471 ± 0.0815 | 0.2745 ± 0.1299 | 0.67 |
| model | v3edge_cross_layers | 0.5148 ± 0.1476 | 0.3054 ± 0.2728 | 0.2223 ± 0.2511 | 0.4018 ± 0.1582 | 0.33 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.5889 | 0.1978 ± 0.0000 | 0.1863 | 0.5890 | 0.00 |
| baseline | spectral_edge_distance | 0.6815 ± 0.0032 | 0.2686 ± 0.0056 | 0.2657 ± 0.0053 | 0.6766 ± 0.0029 | 0.00 |
| model | v2edge_single_layer | 0.3778 ± 0.0530 | 0.0435 ± 0.0449 | 0.0172 ± 0.0301 | 0.2497 ± 0.0867 | 0.67 |
| model | v3edge_cross_layers | 0.3926 ± 0.0378 | 0.0333 ± 0.0272 | 0.0143 ± 0.0159 | 0.3114 ± 0.0910 | 0.33 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.5106 ± 0.0026 | 0.0003 ± 0.0002 | -0.0041 ± 0.0002 | 0.5104 ± 0.0026 | 0.00 |
| baseline | spectral_adj_binary | 0.8955 | 0.5673 | 0.6239 | 0.8947 | 0.00 |
| baseline | spectral_edge_distance | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| model | v2edge_single_layer | 0.5045 | 0.0088 | 0.0000 | 0.3434 | 1.00 |
| model | v3edge_cross_layers | 0.6015 ± 0.1238 | 0.0730 ± 0.0922 | 0.0802 ± 0.1287 | 0.5231 ± 0.2009 | 0.33 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.6833 ± 0.0000 | 0.2470 ± 0.0000 | 0.2765 | 0.6825 | 0.00 |
| baseline | spectral_edge_distance | 0.7000 ± 0.0000 | 0.2817 ± 0.0000 | 0.2990 | 0.6967 | 0.00 |
| model | v2edge_single_layer | 0.3444 ± 0.0000 | 0.0165 ± 0.0078 | -0.0023 ± 0.0038 | 0.2187 ± 0.0498 | 0.67 |
| model | v3edge_cross_layers | 0.3444 ± 0.0000 | 0.0210 | -0.0001 | 0.1899 | 1.00 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.4241 ± 0.0064 | 0.0426 ± 0.0004 | 0.0325 ± 0.0003 | 0.4250 ± 0.0064 | 0.00 |
| baseline | spectral_adj_binary | 0.9296 ± 0.0032 | 0.7620 ± 0.0003 | 0.7970 ± 0.0091 | 0.9304 ± 0.0029 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.7973 ± 0.0000 | 0.8544 | 0.9500 | 0.00 |
| model | v2edge_single_layer | 0.4148 ± 0.0670 | 0.0751 ± 0.0756 | 0.0248 ± 0.0218 | 0.3167 ± 0.1169 | 0.33 |
| model | v3edge_cross_layers | 0.5093 ± 0.0421 | 0.1789 ± 0.0275 | 0.1139 ± 0.0529 | 0.4645 ± 0.0858 | 0.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3167 ± 0.0058 | 0.0163 ± 0.0029 | -0.0001 ± 0.0022 | 0.3157 ± 0.0076 | 0.00 |
| baseline | spectral_adj_binary | 0.9500 ± 0.0000 | 0.8457 | 0.8709 | 0.9499 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.8535 ± 0.0000 | 0.8709 ± 0.0000 | 0.9501 ± 0.0000 | 0.00 |
| model | v2edge_single_layer | 0.2867 ± 0.0462 | 0.0420 ± 0.0238 | 0.0047 ± 0.0080 | 0.1720 ± 0.0887 | 0.67 |
| model | v3edge_cross_layers | 0.3000 ± 0.0693 | 0.0746 ± 0.0803 | 0.0214 ± 0.0370 | 0.1924 ± 0.1241 | 0.67 |

