# Synthetic Benchmark Report (suite_seed0 autorec2)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_autorec2/aggregate.json`
- Total runs: `36`
- Run config: --recommended_auto + edge_message_policy=auto + profiles_edge_attr_norm=zscore_clip + epochs=30, seeds=0,1,2

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
| model | v2edge_single_layer | 0.3463 ± 0.0225 | 0.0097 ± 0.0169 | 0.0012 ± 0.0021 | 0.1933 ± 0.0461 | 1.00 |
| model | v3edge_cross_layers | 0.5056 ± 0.1306 | 0.2070 ± 0.1576 | 0.1680 ± 0.1737 | 0.3995 ± 0.1382 | 0.33 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4019 ± 0.0316 | 0.0425 ± 0.0427 | 0.0263 ± 0.0307 | 0.3179 ± 0.0355 | 0.67 |
| model | v3edge_cross_layers | 0.3685 ± 0.0274 | 0.0154 ± 0.0112 | 0.0014 ± 0.0043 | 0.2918 ± 0.0939 | 0.67 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.6030 ± 0.0964 | 0.0522 ± 0.0566 | 0.0646 ± 0.0718 | 0.5450 ± 0.1872 | 0.33 |
| model | v3edge_cross_layers | 0.6576 ± 0.1412 | 0.1172 ± 0.1145 | 0.1502 ± 0.1473 | 0.6014 ± 0.2351 | 0.33 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4296 ± 0.0316 | 0.0405 ± 0.0215 | 0.0204 ± 0.0092 | 0.3798 ± 0.0613 | 0.33 |
| model | v3edge_cross_layers | 0.3463 ± 0.0225 | 0.0056 ± 0.0096 | 0.0007 ± 0.0013 | 0.2054 ± 0.0670 | 0.67 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4204 ± 0.0750 | 0.0826 ± 0.0552 | 0.0443 ± 0.0509 | 0.3055 ± 0.1050 | 0.67 |
| model | v3edge_cross_layers | 0.3926 ± 0.0979 | 0.0497 ± 0.0769 | 0.0406 ± 0.0703 | 0.2513 ± 0.1367 | 1.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3100 ± 0.0910 | 0.0513 ± 0.0707 | 0.0209 ± 0.0362 | 0.1982 ± 0.1437 | 0.67 |
| model | v3edge_cross_layers | 0.2750 ± 0.0433 | 0.0104 ± 0.0180 | 0.0077 ± 0.0133 | 0.1387 ± 0.0671 | 1.00 |

