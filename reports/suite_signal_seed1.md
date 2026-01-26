# Synthetic Suite Signal Analysis

- suite_dir: `/tmp/sdcn_dlaa_suite_seed1`
- datasets: `6`

该表用于解释“为什么谱聚类 baseline 很强”：若图的同配性（homophily）很高，且 edge_attr[:,0] 的距离特征对同簇/异簇边有很强可分性（AUC 高、effect size 大），谱聚类很容易接近最优。

| dataset | category | N | E | homophily | dist_auc | within_mean | between_mean | effect_d | coords_sil | x_sil |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dist_blobs_easy | distance_1d | 180 | 2480 | 1.0000 | n/a | 0.2343 | n/a | n/a | 0.7690 | 0.2042 |
| dist_blobs_overlap | distance_1d | 180 | 2350 | 0.6834 | 0.3912 | 0.2014 | 0.1497 | 0.3415 | 0.2039 | -0.0105 |
| dist_two_moons | distance_1d | 220 | 3058 | 0.9941 | 0.9687 | 0.2616 | 0.5398 | -2.1531 | 0.3197 | -0.0032 |
| rich_edge_profiles | rich_edge | 180 | 2996 | 0.6342 | 0.5197 | 0.1152 | 0.1607 | -0.3303 | 0.2039 | -0.0105 |
| rich_geo_temporal | rich_edge | 180 | 3052 | 0.8349 | 0.8557 | 0.0952 | 0.3818 | -2.4031 | 0.5343 | -0.0258 |
| rich_multirelation | rich_edge | 200 | 3402 | 0.8248 | 0.8800 | 0.0800 | 0.3448 | -2.4675 | 0.5309 | -0.0319 |

