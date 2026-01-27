# Synthetic Benchmark Report (suite_seed0 autorec2 + pretrain200 + strategyA richonly, v3 edge_message off)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_autorec2_pretrain200_strategyA_richonly_emv3off/aggregate.json`
- Total runs: `36`
- Run config: models-only; pretrain=200; recommended_auto; strategyA: p_smooth=0.1, ce_warmup=10, pred_mi=0.1, q_mi=0.1 (rich_edge only); edge_message auto but v3 forced off

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
| model | v2edge_single_layer | 0.3722 ± 0.0338 | 0.0344 ± 0.0299 | 0.0069 ± 0.0060 | 0.2398 ± 0.0635 | 1.00 |
| model | v3edge_cross_layers | 0.5074 ± 0.1361 | 0.2206 ± 0.1704 | 0.1803 ± 0.1859 | 0.3995 ± 0.1411 | 0.33 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4000 ± 0.0309 | 0.0425 ± 0.0426 | 0.0260 ± 0.0308 | 0.3147 ± 0.0377 | 0.67 |
| model | v3edge_cross_layers | 0.3759 ± 0.0210 | 0.0195 ± 0.0106 | 0.0011 ± 0.0049 | 0.3120 ± 0.0814 | 0.33 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.6061 ± 0.0972 | 0.0553 ± 0.0562 | 0.0676 ± 0.0712 | 0.5471 ± 0.1884 | 0.33 |
| model | v3edge_cross_layers | 0.6803 ± 0.0728 | 0.1350 ± 0.1141 | 0.1406 ± 0.1077 | 0.6739 ± 0.0685 | 0.00 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4556 ± 0.0619 | 0.0484 ± 0.0288 | 0.0377 ± 0.0403 | 0.4279 ± 0.0846 | 0.00 |
| model | v3edge_cross_layers | 0.4093 ± 0.0703 | 0.0666 ± 0.0697 | 0.0487 ± 0.0444 | 0.3296 ± 0.1501 | 0.33 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4204 ± 0.0274 | 0.1168 ± 0.0270 | 0.0328 ± 0.0137 | 0.3177 ± 0.0406 | 0.33 |
| model | v3edge_cross_layers | 0.5889 ± 0.0309 | 0.2271 ± 0.0302 | 0.2263 ± 0.0271 | 0.5345 ± 0.0851 | 0.33 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3983 ± 0.0759 | 0.1095 ± 0.0777 | 0.0564 ± 0.0360 | 0.3225 ± 0.1081 | 0.67 |
| model | v3edge_cross_layers | 0.3167 ± 0.0675 | 0.0495 ± 0.0461 | 0.0282 ± 0.0394 | 0.2067 ± 0.1128 | 0.67 |

