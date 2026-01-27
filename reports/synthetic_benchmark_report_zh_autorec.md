# Synthetic Benchmark Report (suite_seed0 autorec)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_autorec/aggregate.json`
- Total runs: `36`
- Run config: --recommended_auto (old) + edge_message_policy=auto + profiles_edge_attr_norm=zscore_clip + epochs=30, seeds=0,1,2

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
| model | v2edge_single_layer | 0.3759 ± 0.0479 | 0.0337 ± 0.0549 | 0.0082 ± 0.0167 | 0.2594 ± 0.0813 | 1.00 |
| model | v3edge_cross_layers | 0.5296 ± 0.1419 | 0.3007 ± 0.2138 | 0.2119 ± 0.2170 | 0.4134 ± 0.1397 | 1.00 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3352 ± 0.0032 | 0.0026 ± 0.0045 | -0.0000 ± 0.0001 | 0.1707 ± 0.0070 | 1.00 |
| model | v3edge_cross_layers | 0.4130 ± 0.0740 | 0.0528 ± 0.0531 | 0.0382 ± 0.0579 | 0.2965 ± 0.0951 | 1.00 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.6045 ± 0.0967 | 0.0522 ± 0.0566 | 0.0660 ± 0.0715 | 0.5481 ± 0.1889 | 0.33 |
| model | v3edge_cross_layers | 0.6424 ± 0.1258 | 0.1046 ± 0.1046 | 0.1206 ± 0.1348 | 0.6066 ± 0.1714 | 0.00 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4333 ± 0.0309 | 0.0434 ± 0.0207 | 0.0206 ± 0.0122 | 0.3894 ± 0.0534 | 0.00 |
| model | v3edge_cross_layers | 0.3463 ± 0.0225 | 0.0056 ± 0.0096 | 0.0007 ± 0.0013 | 0.2054 ± 0.0670 | 0.67 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4222 ± 0.0752 | 0.0852 ± 0.0586 | 0.0457 ± 0.0504 | 0.3083 ± 0.1062 | 0.67 |
| model | v3edge_cross_layers | 0.3926 ± 0.0979 | 0.0497 ± 0.0769 | 0.0406 ± 0.0703 | 0.2513 ± 0.1367 | 1.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3100 ± 0.0910 | 0.0519 ± 0.0717 | 0.0219 ± 0.0381 | 0.1973 ± 0.1422 | 0.67 |
| model | v3edge_cross_layers | 0.2750 ± 0.0433 | 0.0104 ± 0.0180 | 0.0077 ± 0.0133 | 0.1387 ± 0.0671 | 1.00 |

