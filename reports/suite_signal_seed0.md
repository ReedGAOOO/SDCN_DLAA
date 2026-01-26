# Synthetic Suite Signal Analysis

- suite_dir: `/tmp/sdcn_dlaa_suite_seed0`
- datasets: `6`

该表用于解释“为什么谱聚类 baseline 很强”：若图的同配性（homophily）很高，且 edge_attr[:,0] 的距离特征对同簇/异簇边有很强可分性（AUC 高、effect size 大），谱聚类很容易接近最优。

| dataset | category | N | E | homophily | dist_auc | within_mean | between_mean | effect_d | coords_sil | x_sil |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dist_blobs_easy | distance_1d | 180 | 2422 | 0.9934 | 0.9724 | 0.2988 | 0.8048 | -3.1640 | 0.7289 | 0.0664 |
| dist_blobs_overlap | distance_1d | 180 | 2342 | 0.5611 | 0.4072 | 0.2961 | 0.2449 | 0.3019 | 0.1144 | -0.0180 |
| dist_two_moons | distance_1d | 220 | 3134 | 0.9917 | 0.9904 | 0.2894 | 0.6652 | -2.9148 | 0.3214 | 0.0078 |
| rich_edge_profiles | rich_edge | 180 | 2988 | 0.5442 | 0.4747 | 0.1422 | 0.1613 | -0.1324 | 0.1144 | -0.0180 |
| rich_geo_temporal | rich_edge | 180 | 2956 | 0.7903 | 0.7715 | 0.1288 | 0.3534 | -1.5742 | 0.4522 | -0.0160 |
| rich_multirelation | rich_edge | 200 | 3294 | 0.8021 | 0.8195 | 0.1028 | 0.3657 | -2.0539 | 0.4668 | -0.0311 |

