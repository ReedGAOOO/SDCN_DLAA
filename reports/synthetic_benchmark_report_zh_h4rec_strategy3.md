# Synthetic Benchmark Report (suite_seed0, h4rec+strategy3)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_h4rec_strategy3/aggregate.json`
- Total runs: `36`
- Run config: SDCN_Q_SOURCE=h4; edge_message=auto; profiles zscore_clip; pretrain=200; SDCN_PRED_MI_WEIGHT=0.1

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
| model | v2edge_single_layer | 0.4241 ± 0.0836 | 0.0641 ± 0.0676 | 0.0549 ± 0.0805 | 0.3126 ± 0.1123 | 0.33 |
| model | v3edge_cross_layers | 0.6259 ± 0.1528 | 0.3801 ± 0.1614 | 0.3101 ± 0.2154 | 0.5643 ± 0.1863 | 0.00 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3722 ± 0.0222 | 0.0207 ± 0.0091 | 0.0030 ± 0.0069 | 0.2714 ± 0.0538 | 0.33 |
| model | v3edge_cross_layers | 0.3667 ± 0.0200 | 0.0173 ± 0.0097 | 0.0006 ± 0.0026 | 0.2908 ± 0.0680 | 0.00 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.5045 | 0.0088 | 0.0000 | 0.3434 | 1.00 |
| model | v3edge_cross_layers | 0.5045 | 0.0088 | 0.0000 | 0.3434 | 1.00 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4333 ± 0.0481 | 0.0338 ± 0.0193 | 0.0218 ± 0.0200 | 0.4085 ± 0.0577 | 0.00 |
| model | v3edge_cross_layers | 0.3426 ± 0.0032 | 0.0210 ± 0.0001 | 0.0000 ± 0.0001 | 0.1863 ± 0.0063 | 1.00 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4333 ± 0.0564 | 0.1084 ± 0.0482 | 0.0445 ± 0.0383 | 0.3292 ± 0.0784 | 0.33 |
| model | v3edge_cross_layers | 0.4833 ± 0.1273 | 0.1537 ± 0.1250 | 0.1275 ± 0.1274 | 0.3661 ± 0.1554 | 0.33 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3500 ± 0.1191 | 0.0752 ± 0.0712 | 0.0420 ± 0.0692 | 0.2678 ± 0.1733 | 0.33 |
| model | v3edge_cross_layers | 0.3000 ± 0.0458 | 0.0446 ± 0.0367 | 0.0097 ± 0.0189 | 0.1988 ± 0.0675 | 0.33 |

