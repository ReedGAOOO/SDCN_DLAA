# Synthetic Benchmark Report (suite_seed0, h4rec+strategy2)

- Aggregate: `/tmp/sdcn_dlaa_suite_seed0_results_h4rec_strategy2/aggregate.json`
- Total runs: `36`
- Run config: SDCN_Q_SOURCE=h4; edge_message=auto; profiles zscore_clip; pretrain=200; SDCN_P_SMOOTHING=0.1; SDCN_CE_WARMUP_EPOCHS=10

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
| model | v2edge_single_layer | 0.4185 ± 0.0750 | 0.0580 ± 0.0552 | 0.0490 ± 0.0663 | 0.3105 ± 0.1105 | 0.33 |
| model | v3edge_cross_layers | 0.4889 ± 0.0709 | 0.2216 ± 0.0749 | 0.1003 ± 0.0597 | 0.4276 ± 0.1221 | 0.00 |

### dist_blobs_overlap (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4204 ± 0.0306 | 0.0420 ± 0.0189 | 0.0274 ± 0.0218 | 0.3416 ± 0.0153 | 0.00 |
| model | v3edge_cross_layers | 0.4037 ± 0.0695 | 0.0464 ± 0.0472 | 0.0324 ± 0.0536 | 0.3021 ± 0.0766 | 0.00 |

### dist_two_moons (distance_1d)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.5182 ± 0.0236 | 0.0122 ± 0.0060 | 0.0021 ± 0.0037 | 0.3849 ± 0.0719 | 0.67 |
| model | v3edge_cross_layers | 0.5197 ± 0.0262 | 0.0159 ± 0.0123 | 0.0028 ± 0.0049 | 0.3829 ± 0.0685 | 1.00 |

### rich_edge_profiles (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4222 ± 0.0333 | 0.0409 ± 0.0077 | 0.0181 ± 0.0113 | 0.3841 ± 0.0294 | 0.00 |
| model | v3edge_cross_layers | 0.3611 ± 0.0167 | 0.0286 ± 0.0107 | 0.0022 ± 0.0022 | 0.2282 ± 0.0417 | 0.67 |

### rich_geo_temporal (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.4444 ± 0.0855 | 0.1061 ± 0.0515 | 0.0744 ± 0.0813 | 0.3661 ± 0.1083 | 0.33 |
| model | v3edge_cross_layers | 0.3870 ± 0.0378 | 0.0764 ± 0.0595 | 0.0120 ± 0.0118 | 0.2660 ± 0.0668 | 1.00 |

### rich_multirelation (rich_edge)

| approach | name | acc | nmi | ari | f1 | collapse_rate |
|---|---|---:|---:|---:|---:|---:|
| model | v2edge_single_layer | 0.3000 ± 0.0377 | 0.0465 ± 0.0239 | 0.0109 ± 0.0141 | 0.1870 ± 0.0577 | 0.33 |
| model | v3edge_cross_layers | 0.3100 ± 0.0444 | 0.0634 ± 0.0439 | 0.0136 ± 0.0179 | 0.2065 ± 0.0745 | 0.33 |

