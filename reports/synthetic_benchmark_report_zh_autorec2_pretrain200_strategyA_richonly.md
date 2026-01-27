# Synthetic Benchmark Report (suite_seed0 autorec2 + pretrain200 + strategyA richonly)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_autorec2_pretrain200_strategyA_richonly/aggregate.json`
- Total runs: `90`
- Run config: pretrain=200; recommended_auto; strategyA: p_smooth=0.1, ce_warmup=10, pred_mi=0.1, q_mi=0.1 (rich_edge only)

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
| model | v2edge_single_layer | 0.4037 ± 0.0754 | 0.0486 ± 0.0487 | 0.0368 ± 0.0550 | 0.2788 ± 0.1116 | 1.00 |
| model | v3edge_cross_layers | 0.5111 ± 0.1417 | 0.2029 ± 0.1731 | 0.1610 ± 0.1855 | 0.4209 ± 0.1606 | 0.33 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.5889 | 0.1978 ± 0.0000 | 0.1863 | 0.5890 | 0.00 |
| baseline | spectral_edge_distance | 0.6815 ± 0.0032 | 0.2686 ± 0.0056 | 0.2657 ± 0.0053 | 0.6766 ± 0.0029 | 0.00 |
| model | v2edge_single_layer | 0.3963 ± 0.0370 | 0.0414 ± 0.0442 | 0.0248 ± 0.0325 | 0.3160 ± 0.0359 | 0.67 |
| model | v3edge_cross_layers | 0.3722 ± 0.0242 | 0.0160 ± 0.0102 | 0.0016 ± 0.0041 | 0.2994 ± 0.0882 | 0.67 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.5106 ± 0.0026 | 0.0003 ± 0.0002 | -0.0041 ± 0.0002 | 0.5104 ± 0.0026 | 0.00 |
| baseline | spectral_adj_binary | 0.8955 | 0.5673 | 0.6239 | 0.8947 | 0.00 |
| baseline | spectral_edge_distance | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| model | v2edge_single_layer | 0.6030 ± 0.0964 | 0.0522 ± 0.0566 | 0.0646 ± 0.0718 | 0.5450 ± 0.1872 | 0.33 |
| model | v3edge_cross_layers | 0.6318 ± 0.1142 | 0.0837 ± 0.0726 | 0.1018 ± 0.0883 | 0.5763 ± 0.2036 | 0.33 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3944 | 0.0128 ± 0.0024 | 0.0034 ± 0.0027 | 0.3921 ± 0.0010 | 0.00 |
| baseline | spectral_adj_binary | 0.6833 ± 0.0000 | 0.2470 ± 0.0000 | 0.2765 | 0.6825 | 0.00 |
| baseline | spectral_edge_distance | 0.7000 ± 0.0000 | 0.2817 ± 0.0000 | 0.2990 | 0.6967 | 0.00 |
| model | v2edge_single_layer | 0.4556 ± 0.0585 | 0.0478 ± 0.0274 | 0.0384 ± 0.0387 | 0.4270 ± 0.0813 | 0.00 |
| model | v3edge_cross_layers | 0.3722 ± 0.0419 | 0.0192 ± 0.0237 | 0.0104 ± 0.0168 | 0.2488 ± 0.0801 | 0.67 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.4241 ± 0.0064 | 0.0426 ± 0.0004 | 0.0325 ± 0.0003 | 0.4250 ± 0.0064 | 0.00 |
| baseline | spectral_adj_binary | 0.9296 ± 0.0032 | 0.7620 ± 0.0003 | 0.7970 ± 0.0091 | 0.9304 ± 0.0029 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.7973 ± 0.0000 | 0.8544 | 0.9500 | 0.00 |
| model | v2edge_single_layer | 0.4259 ± 0.0339 | 0.1254 ± 0.0419 | 0.0383 ± 0.0205 | 0.3248 ± 0.0489 | 0.33 |
| model | v3edge_cross_layers | 0.4426 ± 0.1472 | 0.1363 ± 0.1616 | 0.1065 ± 0.1781 | 0.3065 ± 0.1633 | 0.67 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| baseline | kmeans_x | 0.3167 ± 0.0058 | 0.0163 ± 0.0029 | -0.0001 ± 0.0022 | 0.3157 ± 0.0076 | 0.00 |
| baseline | spectral_adj_binary | 0.9500 ± 0.0000 | 0.8457 | 0.8709 | 0.9499 | 0.00 |
| baseline | spectral_edge_distance | 0.9500 ± 0.0000 | 0.8535 ± 0.0000 | 0.8709 ± 0.0000 | 0.9501 ± 0.0000 | 0.00 |
| model | v2edge_single_layer | 0.3733 ± 0.0388 | 0.0892 ± 0.0386 | 0.0431 ± 0.0225 | 0.2971 ± 0.0659 | 0.67 |
| model | v3edge_cross_layers | 0.3450 ± 0.0557 | 0.0612 ± 0.0538 | 0.0392 ± 0.0389 | 0.2392 ± 0.0298 | 1.00 |

