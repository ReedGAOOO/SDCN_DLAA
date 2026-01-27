# Synthetic Benchmark Report (suite_seed1, h4rec+strategy1_richonly)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed1_results_h4rec_strategy1_richonly/aggregate.json`
- Total runs: `36`
- Run config: SDCN_Q_SOURCE=h4; edge_message=auto; profiles zscore_clip; pretrain=200; (rich_edge only) SDCN_P_SMOOTHING=0.1; SDCN_CE_WARMUP_EPOCHS=10; SDCN_PRED_MI_WEIGHT=0.1

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
| model | v2edge_single_layer | 0.5852 ± 0.0128 | 0.4234 ± 0.0617 | 0.3316 ± 0.0294 | 0.5056 ± 0.0041 | 0.33 |
| model | v3edge_cross_layers | 0.6630 ± 0.0504 | 0.4036 ± 0.0674 | 0.3364 ± 0.1289 | 0.6243 ± 0.0643 | 0.00 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4019 ± 0.0570 | 0.0325 ± 0.0218 | 0.0267 ± 0.0285 | 0.2912 ± 0.0986 | 0.67 |
| model | v3edge_cross_layers | 0.3667 ± 0.0441 | 0.0301 ± 0.0508 | 0.0165 ± 0.0322 | 0.2751 ± 0.1153 | 0.67 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.5000 | 0.0000 | 0.0000 | 0.3333 | 1.00 |
| model | v3edge_cross_layers | 0.5091 ± 0.0157 | 0.0010 ± 0.0016 | -0.0002 ± 0.0004 | 0.3901 ± 0.0983 | 0.67 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4500 ± 0.0973 | 0.0760 ± 0.0645 | 0.0635 ± 0.0797 | 0.4140 ± 0.1384 | 0.00 |
| model | v3edge_cross_layers | 0.4278 ± 0.0481 | 0.0723 ± 0.0573 | 0.0384 ± 0.0312 | 0.3256 ± 0.0523 | 1.00 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4630 ± 0.1031 | 0.1320 ± 0.1422 | 0.0991 ± 0.1397 | 0.4078 ± 0.1764 | 0.33 |
| model | v3edge_cross_layers | 0.3519 ± 0.0321 | 0.0139 ± 0.0241 | 0.0027 ± 0.0046 | 0.2014 ± 0.0601 | 1.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.2900 ± 0.0304 | 0.0300 ± 0.0113 | 0.0043 ± 0.0059 | 0.1852 ± 0.0754 | 0.67 |
| model | v3edge_cross_layers | 0.2750 ± 0.0433 | 0.0237 ± 0.0411 | 0.0065 ± 0.0113 | 0.1388 ± 0.0673 | 1.00 |

