# Synthetic Benchmark Report (suite_seed0, h4rec)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_h4rec/aggregate.json`
- Total runs: `90`
- Run config: SDCN_Q_SOURCE=h4; edge_message=auto(rich_edge=1,dist=0); profiles edge_attr_norm=zscore_clip; pretrain=200

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
| model | v2edge_single_layer | 0.4056 ± 0.0674 | 0.0597 ± 0.0382 | 0.0369 ± 0.0539 | 0.2884 ± 0.1000 | 0.33 |
| model | v3edge_cross_layers | 0.5148 ± 0.1445 | 0.2373 ± 0.1585 | 0.1905 ± 0.1797 | 0.4141 ± 0.1636 | 0.33 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.5889 | 0.1978 ± 0.0000 | 0.1863 | 0.5890 | 0.00 |
| baseline | spectral_edge_distance | 0.6815 ± 0.0032 | 0.2686 ± 0.0056 | 0.2657 ± 0.0053 | 0.6766 ± 0.0029 | 0.00 |
| model | v2edge_single_layer | 0.3981 ± 0.0378 | 0.0465 ± 0.0400 | 0.0264 ± 0.0314 | 0.3120 ± 0.0454 | 0.00 |
| model | v3edge_cross_layers | 0.3815 ± 0.0195 | 0.0223 ± 0.0076 | 0.0019 ± 0.0039 | 0.3204 ± 0.0802 | 0.33 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.5106 ± 0.0026 | 0.0003 ± 0.0002 | -0.0041 ± 0.0002 | 0.5104 ± 0.0026 | 0.00 |
| baseline | spectral_adj_binary | 0.8955 | 0.5673 | 0.6239 | 0.8947 | 0.00 |
| baseline | spectral_edge_distance | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| model | v2edge_single_layer | 0.5045 | 0.0088 | 0.0000 | 0.3434 | 1.00 |
| model | v3edge_cross_layers | 0.5076 ± 0.0052 | 0.0061 ± 0.0046 | -0.0009 ± 0.0016 | 0.3901 ± 0.0810 | 0.67 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.6833 ± 0.0000 | 0.2470 ± 0.0000 | 0.2765 | 0.6825 | 0.00 |
| baseline | spectral_edge_distance | 0.7000 ± 0.0000 | 0.2817 ± 0.0000 | 0.2990 | 0.6967 | 0.00 |
| model | v2edge_single_layer | 0.4148 ± 0.0619 | 0.0432 ± 0.0193 | 0.0181 ± 0.0158 | 0.3400 ± 0.1320 | 0.33 |
| model | v3edge_cross_layers | 0.3537 ± 0.0160 | 0.0195 ± 0.0025 | 0.0007 ± 0.0013 | 0.2209 ± 0.0536 | 0.67 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.4241 ± 0.0064 | 0.0426 ± 0.0004 | 0.0325 ± 0.0003 | 0.4250 ± 0.0064 | 0.00 |
| baseline | spectral_adj_binary | 0.9296 ± 0.0032 | 0.7620 ± 0.0003 | 0.7970 ± 0.0091 | 0.9304 ± 0.0029 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.7973 ± 0.0000 | 0.8544 | 0.9500 | 0.00 |
| model | v2edge_single_layer | 0.4222 ± 0.0778 | 0.0874 ± 0.0504 | 0.0443 ± 0.0506 | 0.3098 ± 0.1099 | 0.33 |
| model | v3edge_cross_layers | 0.3981 ± 0.0979 | 0.0613 ± 0.0697 | 0.0403 ± 0.0698 | 0.2633 ± 0.1367 | 0.67 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3167 ± 0.0058 | 0.0163 ± 0.0029 | -0.0001 ± 0.0022 | 0.3157 ± 0.0076 | 0.00 |
| baseline | spectral_adj_binary | 0.9500 ± 0.0000 | 0.8457 | 0.8709 | 0.9499 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.8535 ± 0.0000 | 0.8709 ± 0.0000 | 0.9501 ± 0.0000 | 0.00 |
| model | v2edge_single_layer | 0.3117 ± 0.0808 | 0.0589 ± 0.0527 | 0.0165 ± 0.0287 | 0.2034 ± 0.1252 | 0.67 |
| model | v3edge_cross_layers | 0.2817 ± 0.0375 | 0.0327 ± 0.0077 | 0.0070 ± 0.0121 | 0.1553 ± 0.0597 | 0.67 |

