# v22stable_denoise_scalar_fuse：一个更“保守、可控”的稳定版 edge 特征利用结构

> 目标：针对目前常见的两类问题——**edge↔edge 容易误混合导致掉点**、以及 **pred 头在部分 seed 下容易缺簇/塌缩**——实验一个更保守的结构，把“边特征利用”做得更稳、更可控，同时尽量不背离 SDCN 的双重自监督主干（delivery operator + P 同时监督 Q 与 Z）。

---

## 1. 结构概要（实现位于 `DLAA_NEW.py`）

新增 SpatialConv 版本：`SpatialConvV22StableDenoiseScalarFuse`  
选择方式：`SPATIALCONV_VARIANT=v22`（或 `v22stable_denoise_scalar_fuse`）

核心思路（把已有“稳”的设计拼成一条更保守的路径）：

1) **边去噪（保守）**：使用 v15/v13 那条线的 *context-aware similarity smoothing*，避免 incidence 邻域里“同簇边 + 跨簇噪声边”被硬混合。
2) **节点注意力的 base 仍用 raw edge_attr（稳）**：沿用 v5 的取舍，让 `dist_feat` 直接进入 node attention，减少“edge embedding 早期被洗掉/漂移”的风险。
3) **把去噪后的边表征注入注意力（可控）**：用 v20 的 *scalar-gated fusion*（每条边一个标量 gate）把去噪边表征注入 `edge_attr`，并用 `tanh(scale)` 控幅（初始化为 0，训练初期近似不注入，避免不稳）。
4) **保留 pooling residual（稳）**：仍然做 `edge→node` mean pooling（raw + denoised），并用门控融合回节点表征。
5) **logit 层特殊处理（减小扰动）**：当检测到第 5 层（`out_activation=F.leaky_relu`）时，关闭“edge 去噪/上下文混合”，避免在“logit 空间”做不必要的边混合导致 pred 头不稳。

---

## 2. 推荐的“稳态”运行配方（先对齐诊断集，再扩展到真实数据）

### 2.1 诊断集（edge_edge_denoise_nonknn）生成

```bash
conda run -n gnn python tools/generate_synthetic_suite.py \
  --output_root /tmp/sdcn_edge_debug_suite \
  --seed 0 \
  --presets edge_edge_denoise_nonknn
```

### 2.2 对比跑法（v5 vs v16 vs v22）

```bash
conda run -n gnn python tools/sweep_stability.py \
  --data_dir /tmp/sdcn_edge_debug_suite/edge_edge_denoise_nonknn \
  --out_dir /tmp/sweep_v22_compare \
  --variants v5edge_pool_residual,v16edge_ee_residual_aux_fusion,v22stable_denoise_scalar_fuse \
  --seeds 0,1,2 \
  --epochs 60 \
  --lrs 1e-3 \
  --dropouts 0.2 \
  --heads 1 \
  --n_z 10 \
  --sigmas 0.2 \
  --q_sources h4 \
  --edge_messages 1 \
  --edge_ees 1 \
  --ee_graphs incidence_sim \
  --ee_topks 4 \
  --ee_sim_min_sims 0.4 \
  --edge_denoise_alphas 0.1 \
  --edge_sim_gammas 1.0 \
  --q_balance_weights 0.1 \
  --pred_balance_weights 0.1 \
  --edge_attr_fuses 1 \
  --edge_attr_fuse_scales 0.1 \
  --edge_attr_fuse_detaches 0 \
  --edge_aux_weights 0.0 \
  --kl_weights 0.1 \
  --ce_weights 1.0 \
  --ce_warmups 20 \
  --final_assign pred
```

> 说明：这组配方故意用 `final_assign=pred` 来“压测” pred 头稳定性；同时用 `ce_weight + warmup` 与 `balance` 来减轻缺簇/塌缩。

---

## 3. 本次环境的一次结果摘要（来自 `/tmp/sweep_v22_compare/aggregate.json`）

在 `edge_edge_denoise_nonknn`、3 seeds、60 epochs、上述超参下：

- `v5edge_pool_residual`：`acc_mean≈0.418`，`collapse=1/3`
- `v16edge_ee_residual_aux_fusion`：`acc_mean≈0.329`，`collapse=2/3`
- `v22stable_denoise_scalar_fuse`：`acc_mean≈0.343`，`collapse=1/3`

结论（就这个诊断集而言）：

- v22 相比 v16：**更稳（缺簇次数更少）**，均值也略高；
- 但 v5 在该诊断集上仍然是更强的基线（均值更高）。

这与 v22 的定位一致：它不是为了在“edge_pool_residual 强基线”上无条件碾压，而是为了在“edge↔edge 容易误混合/容易不稳”的设置里，提供一条更保守、可控的路径，减少掉点与缺簇风险。

