# Synthetic Benchmark Report (suite_seed1, h4rec)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed1_results_h4rec_fixed/aggregate.json`
- Total runs: `36`
- Run config: SDCN_Q_SOURCE=h4; edge_message=auto; profiles zscore_clip; pretrain=200

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
| model | v2edge_single_layer | 0.5500 ± 0.0484 | 0.3070 ± 0.1080 | 0.1925 ± 0.0853 | 0.4900 ± 0.0586 | 0.33 |
| model | v3edge_cross_layers | 0.6667 ± 0.0641 | 0.4642 ± 0.0701 | 0.3813 ± 0.1172 | 0.6075 ± 0.0792 | 0.00 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4519 ± 0.0306 | 0.0631 ± 0.0320 | 0.0619 ± 0.0370 | 0.3615 ± 0.0227 | 0.67 |
| model | v3edge_cross_layers | 0.3667 ± 0.0441 | 0.0301 ± 0.0508 | 0.0165 ± 0.0322 | 0.2751 ± 0.1153 | 0.67 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.5000 | 0.0000 | 0.0000 | 0.3333 | 1.00 |
| model | v3edge_cross_layers | 0.5091 ± 0.0157 | 0.0010 ± 0.0016 | -0.0002 ± 0.0004 | 0.3901 ± 0.0983 | 0.67 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4500 ± 0.0873 | 0.0851 ± 0.0454 | 0.0615 ± 0.0725 | 0.3973 ± 0.1274 | 0.33 |
| model | v3edge_cross_layers | 0.4241 ± 0.1250 | 0.0746 ± 0.1182 | 0.0712 ± 0.1215 | 0.2952 ± 0.1462 | 1.00 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4037 ± 0.0545 | 0.1216 ± 0.1432 | 0.0648 ± 0.1046 | 0.3008 ± 0.0895 | 0.67 |
| model | v3edge_cross_layers | 0.3630 ± 0.0280 | 0.0267 ± 0.0232 | 0.0046 ± 0.0060 | 0.2265 ± 0.0576 | 1.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3283 ± 0.0693 | 0.0997 ± 0.1157 | 0.0470 ± 0.0709 | 0.2291 ± 0.0973 | 1.00 |
| model | v3edge_cross_layers | 0.2867 ± 0.0635 | 0.0237 ± 0.0410 | 0.0142 ± 0.0247 | 0.1481 ± 0.0833 | 1.00 |

