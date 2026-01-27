# Synthetic Benchmark Report (suite_seed0 autorec2 + pretrain200)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_autorec2_pretrain200/aggregate.json`
- Total runs: `90`
- Run config: --recommended_auto + edge_message_policy=auto + profiles_edge_attr_norm=zscore_clip + pretrain=200 + epochs=30 + seeds=0,1,2

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
| baseline | kmeans_x | 0.5500 ± 0.0167 | 0.1636 ± 0.0034 | 0.1748 ± 0.0039 | 0.5383 ± 0.0181 | 0.00 |
| baseline | spectral_adj_binary | 0.9944 | 0.9742 | 0.9833 | 0.9944 | 0.00 |
| baseline | spectral_edge_distance | 0.9944 | 0.9742 | 0.9833 | 0.9944 | 0.00 |
| model | v2edge_single_layer | 0.4000 ± 0.0729 | 0.0450 ± 0.0452 | 0.0335 ± 0.0506 | 0.2747 ± 0.1093 | 1.00 |
| model | v3edge_cross_layers | 0.5204 ± 0.1530 | 0.2273 ± 0.1707 | 0.1857 ± 0.1858 | 0.4284 ± 0.1793 | 0.33 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.5889 | 0.1978 ± 0.0000 | 0.1863 | 0.5890 | 0.00 |
| baseline | spectral_edge_distance | 0.6815 ± 0.0032 | 0.2686 ± 0.0056 | 0.2657 ± 0.0053 | 0.6766 ± 0.0029 | 0.00 |
| model | v2edge_single_layer | 0.3944 ± 0.0401 | 0.0398 ± 0.0448 | 0.0245 ± 0.0320 | 0.3045 ± 0.0532 | 0.67 |
| model | v3edge_cross_layers | 0.3722 ± 0.0242 | 0.0162 ± 0.0100 | 0.0017 ± 0.0041 | 0.2993 ± 0.0882 | 0.67 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.5106 ± 0.0026 | 0.0003 ± 0.0002 | -0.0041 ± 0.0002 | 0.5104 ± 0.0026 | 0.00 |
| baseline | spectral_adj_binary | 0.8955 | 0.5673 | 0.6239 | 0.8947 | 0.00 |
| baseline | spectral_edge_distance | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| model | v2edge_single_layer | 0.6030 ± 0.0964 | 0.0522 ± 0.0566 | 0.0646 ± 0.0718 | 0.5450 ± 0.1872 | 0.33 |
| model | v3edge_cross_layers | 0.5924 ± 0.1404 | 0.0736 ± 0.1125 | 0.0842 ± 0.1487 | 0.5367 ± 0.2020 | 0.33 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.6833 ± 0.0000 | 0.2470 ± 0.0000 | 0.2765 | 0.6825 | 0.00 |
| baseline | spectral_edge_distance | 0.7000 ± 0.0000 | 0.2817 ± 0.0000 | 0.2990 | 0.6967 | 0.00 |
| model | v2edge_single_layer | 0.4148 ± 0.0619 | 0.0385 ± 0.0292 | 0.0183 ± 0.0162 | 0.3439 ± 0.1285 | 0.33 |
| model | v3edge_cross_layers | 0.3463 ± 0.0225 | 0.0056 ± 0.0096 | 0.0007 ± 0.0013 | 0.2054 ± 0.0670 | 0.67 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.4241 ± 0.0064 | 0.0426 ± 0.0004 | 0.0325 ± 0.0003 | 0.4250 ± 0.0064 | 0.00 |
| baseline | spectral_adj_binary | 0.9296 ± 0.0032 | 0.7620 ± 0.0003 | 0.7970 ± 0.0091 | 0.9304 ± 0.0029 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.7973 ± 0.0000 | 0.8544 | 0.9500 | 0.00 |
| model | v2edge_single_layer | 0.4204 ± 0.0750 | 0.0826 ± 0.0552 | 0.0443 ± 0.0509 | 0.3055 ± 0.1050 | 0.67 |
| model | v3edge_cross_layers | 0.3926 ± 0.0979 | 0.0497 ± 0.0769 | 0.0406 ± 0.0703 | 0.2513 ± 0.1367 | 1.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3167 ± 0.0058 | 0.0163 ± 0.0029 | -0.0001 ± 0.0022 | 0.3157 ± 0.0076 | 0.00 |
| baseline | spectral_adj_binary | 0.9500 ± 0.0000 | 0.8457 | 0.8709 | 0.9499 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.8535 ± 0.0000 | 0.8709 ± 0.0000 | 0.9501 ± 0.0000 | 0.00 |
| model | v2edge_single_layer | 0.3133 ± 0.0967 | 0.0535 ± 0.0745 | 0.0221 ± 0.0384 | 0.1993 ± 0.1458 | 0.67 |
| model | v3edge_cross_layers | 0.2750 ± 0.0433 | 0.0104 ± 0.0180 | 0.0077 ± 0.0133 | 0.1387 ± 0.0671 | 1.00 |

