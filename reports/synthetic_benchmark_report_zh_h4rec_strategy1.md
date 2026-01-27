# Synthetic Benchmark Report (suite_seed0, h4rec+strategy1)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_h4rec_strategy1/aggregate.json`
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
| model | v2edge_single_layer | 0.4130 ± 0.0832 | 0.0696 ± 0.0848 | 0.0410 ± 0.0652 | 0.2990 ± 0.1050 | 0.33 |
| model | v3edge_cross_layers | 0.5259 ± 0.0361 | 0.2250 ± 0.0455 | 0.1464 ± 0.0900 | 0.4522 ± 0.0298 | 0.00 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3611 ± 0.0167 | 0.0192 ± 0.0067 | 0.0014 ± 0.0015 | 0.2433 ± 0.0457 | 0.67 |
| model | v3edge_cross_layers | 0.3926 ± 0.0417 | 0.0357 ± 0.0129 | 0.0157 ± 0.0137 | 0.2806 ± 0.0786 | 0.33 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.5606 ± 0.0681 | 0.0237 ± 0.0284 | 0.0243 ± 0.0398 | 0.5011 ± 0.1469 | 0.33 |
| model | v3edge_cross_layers | 0.5242 ± 0.0341 | 0.0122 ± 0.0059 | 0.0043 ± 0.0075 | 0.4051 ± 0.1069 | 0.67 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4500 ± 0.0547 | 0.0423 ± 0.0294 | 0.0347 ± 0.0398 | 0.4212 ± 0.0745 | 0.00 |
| model | v3edge_cross_layers | 0.3519 ± 0.0128 | 0.0181 ± 0.0050 | -0.0005 ± 0.0008 | 0.2197 ± 0.0516 | 0.67 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4574 ± 0.0479 | 0.1333 ± 0.0704 | 0.0865 ± 0.0801 | 0.3681 ± 0.0561 | 0.00 |
| model | v3edge_cross_layers | 0.4463 ± 0.1441 | 0.1437 ± 0.1593 | 0.1084 ± 0.1820 | 0.3144 ± 0.1570 | 0.67 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3433 ± 0.0580 | 0.0939 ± 0.0356 | 0.0392 ± 0.0448 | 0.2431 ± 0.0660 | 0.33 |
| model | v3edge_cross_layers | 0.3317 ± 0.0621 | 0.0708 ± 0.0416 | 0.0319 ± 0.0276 | 0.2108 ± 0.0780 | 0.33 |

