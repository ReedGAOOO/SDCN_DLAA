# Synthetic Benchmark Report (suite_seed1, h4rec+strategy1)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed1_results_h4rec_strategy1/aggregate.json`
- Total runs: `36`
- Run config: SDCN_Q_SOURCE=h4; edge_message=auto; profiles zscore_clip; pretrain=200; SDCN_P_SMOOTHING=0.1; SDCN_CE_WARMUP_EPOCHS=10; SDCN_PRED_MI_WEIGHT=0.1

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
| model | v2edge_single_layer | 0.4907 ± 0.1363 | 0.2371 ± 0.2134 | 0.1687 ± 0.1463 | 0.3710 ± 0.1770 | 1.00 |
| model | v3edge_cross_layers | 0.6981 ± 0.0545 | 0.7249 ± 0.0152 | 0.5700 ± 0.0024 | 0.6123 ± 0.0984 | 0.67 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4222 ± 0.0147 | 0.0367 ± 0.0095 | 0.0270 ± 0.0130 | 0.3316 ± 0.0195 | 1.00 |
| model | v3edge_cross_layers | 0.4222 ± 0.0338 | 0.0546 ± 0.0433 | 0.0240 ± 0.0187 | 0.3326 ± 0.0419 | 0.67 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.5091 ± 0.0091 | 0.0065 ± 0.0092 | -0.0003 ± 0.0006 | 0.3785 ± 0.0618 | 0.67 |
| model | v3edge_cross_layers | 0.5106 ± 0.0184 | 0.0014 ± 0.0024 | 0.0002 ± 0.0003 | 0.3899 ± 0.0979 | 0.67 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4500 ± 0.0973 | 0.0760 ± 0.0645 | 0.0635 ± 0.0797 | 0.4140 ± 0.1384 | 0.00 |
| model | v3edge_cross_layers | 0.4056 ± 0.0441 | 0.0713 ± 0.0573 | 0.0209 ± 0.0253 | 0.2956 ± 0.0501 | 1.00 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4593 ± 0.1089 | 0.1279 ± 0.1469 | 0.0988 ± 0.1400 | 0.4006 ± 0.1884 | 0.33 |
| model | v3edge_cross_layers | 0.3481 ± 0.0257 | 0.0111 ± 0.0192 | 0.0016 ± 0.0029 | 0.1959 ± 0.0506 | 1.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3317 ± 0.0603 | 0.0722 ± 0.0640 | 0.0331 ± 0.0468 | 0.2445 ± 0.0881 | 0.67 |
| model | v3edge_cross_layers | 0.2750 ± 0.0433 | 0.0237 ± 0.0411 | 0.0065 ± 0.0113 | 0.1388 ± 0.0673 | 1.00 |

