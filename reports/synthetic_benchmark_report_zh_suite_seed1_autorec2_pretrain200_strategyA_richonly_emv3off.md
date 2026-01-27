# Synthetic Benchmark Report (suite_seed1 autorec2 + pretrain200 + strategyA richonly, v3 edge_message off)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed1_results_autorec2_pretrain200_strategyA_richonly_emv3off/aggregate.json`
- Total runs: `36`
- Run config: models-only; suite_seed1; pretrain=200; recommended_auto; strategyA: p_smooth=0.1, ce_warmup=10, pred_mi=0.1, q_mi=0.1 (rich_edge only); edge_message auto but v3 forced off

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
| model | v2edge_single_layer | 0.5537 ± 0.0706 | 0.3617 ± 0.1505 | 0.2577 ± 0.1574 | 0.4722 ± 0.0588 | 0.33 |
| model | v3edge_cross_layers | 0.6833 ± 0.0338 | 0.4365 ± 0.0519 | 0.3617 ± 0.1142 | 0.6419 ± 0.0612 | 0.00 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4519 ± 0.0306 | 0.0624 ± 0.0303 | 0.0624 ± 0.0368 | 0.3653 ± 0.0287 | 0.33 |
| model | v3edge_cross_layers | 0.3667 ± 0.0441 | 0.0301 ± 0.0508 | 0.0165 ± 0.0322 | 0.2751 ± 0.1153 | 0.67 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.5015 ± 0.0026 | 0.0029 ± 0.0051 | 0.0000 | 0.3367 ± 0.0058 | 1.00 |
| model | v3edge_cross_layers | 0.5348 ± 0.0322 | 0.0352 ± 0.0558 | 0.0060 ± 0.0080 | 0.4384 ± 0.0958 | 0.67 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4630 ± 0.0740 | 0.0722 ± 0.0584 | 0.0586 ± 0.0777 | 0.4384 ± 0.0961 | 0.00 |
| model | v3edge_cross_layers | 0.3889 ± 0.0484 | 0.0270 ± 0.0236 | 0.0140 ± 0.0127 | 0.2683 ± 0.0884 | 1.00 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4741 ± 0.0758 | 0.1454 ± 0.1485 | 0.1120 ± 0.1438 | 0.4279 ± 0.1234 | 0.33 |
| model | v3edge_cross_layers | 0.4926 ± 0.0651 | 0.1245 ± 0.1023 | 0.0822 ± 0.0406 | 0.4174 ± 0.0893 | 0.67 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3583 ± 0.0275 | 0.0709 ± 0.0300 | 0.0443 ± 0.0353 | 0.2633 ± 0.0267 | 0.67 |
| model | v3edge_cross_layers | 0.3317 ± 0.0775 | 0.0510 ± 0.0504 | 0.0302 ± 0.0355 | 0.2254 ± 0.1193 | 0.67 |

