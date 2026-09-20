# G2 trusted build plan in the complete isolated neighbor pipeline

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 35。

## 目标与路径

本小点把 Point 34 的 `_TrustedBatchCellBuildPlan` 接入已有完整隔离流程：

```text
cell-list build -> native query candidate -> composite-key topology -> geometry
```

重点比较三条路径：

- `full_torch`：Torch cell-list、Torch query、Torch topology/geometry reference；
- `full_native_torch_geometry`：公开 checked native HIP build、native query/topology、Torch
  geometry；
- `full_trusted_torch_geometry`：checked plan initialization 后的 trusted native build、
  native query/topology、Torch geometry。

原 probe 中的 `full_native_native_geometry` 旁路仍保留并通过 parity，但它是 forward-only
候选，不参与训练/梯度路径的性能结论。

## 测量合同

- 设备：HCU 0，BW200，`gfx936`，DTK 26.04，PyTorch HIP `6.3.26093`。
- FP32、capacity 256、mixed PBC、非正交 cell、cutoff `0.6`。
- `46/92 atoms × 32/64 systems`，uniform/clustered，共 8 个 workload；3 warm-up、5 samples。
- 固定 metadata、output buffer、query/topology workspace 和 JIT/module load 不计入稳态区间。
- 每条路径使用独立 geometry output buffer；build、neighbor matrix、shift、count、distance、
  vector 均与 Torch reference 比较。native atom-list 的内部顺序不要求稳定，但要求 cell
  membership 和 public neighbor output 一致。

## 结果

下表为 HIP-event median，factor 相对 `full_torch`；小于 `1.0x` 表示更快。

| atoms × systems | workload | Torch full ms | public native + Torch geometry | trusted plan + Torch geometry |
|---|---|---:|---:|---:|
| 46 × 32 | uniform | 93.089 | 4.661 (`0.0501x`) | 2.352 (`0.0253x`) |
| 46 × 32 | clustered | 114.666 | 4.739 (`0.0413x`) | 2.460 (`0.0214x`) |
| 46 × 64 | uniform | 173.730 | 5.343 (`0.0308x`) | 2.520 (`0.0145x`) |
| 46 × 64 | clustered | 218.184 | 5.432 (`0.0249x`) | 2.640 (`0.0121x`) |
| 92 × 32 | uniform | 103.541 | 4.899 (`0.0473x`) | 2.660 (`0.0257x`) |
| 92 × 32 | clustered | 112.041 | 5.043 (`0.0450x`) | 2.744 (`0.0245x`) |
| 92 × 64 | uniform | 206.682 | 5.549 (`0.0268x`) | 2.737 (`0.0132x`) |
| 92 × 64 | clustered | 225.749 | 5.645 (`0.0250x`) | 2.880 (`0.0128x`) |

跨 8 个 workload：

- public native + Torch geometry / Torch full：`0.0249x--0.0501x`；
- trusted plan + Torch geometry / Torch full：`0.0121x--0.0257x`；
- trusted plan / public native full：`0.472x--0.544x`。

显式同步 API wall-clock 的 trusted/Torch median factor 为 `0.0122x--0.0259x`，与 event
结果同方向。它只说明 build 优化传递到该隔离完整 pipeline；不能扩大为 MACE/FIRE2、
变胞、rebuild/skin 或生产 neighbor API 加速。

## 正确性与边界

- 修正版 probe 为 public 和 trusted plan 使用独立 geometry output buffer；8 workload 全部
  通过 build、topology、neighbor matrix、shift、count、distance、vector parity。
- 46×32 独立-buffer smoke 通过；native forward geometry 旁路也保持 parity。
- Torch geometry 仍是当前训练/力梯度路径；trusted build/query/topology 本身不提供 geometry
  autograd。
- 不接 dispatcher、AOT、`auto`、`hip` capability，也不覆盖 FP64 性能、skin/rebuild、
  half-list、`target_indices`、`pair_fn`/pair outputs、compile/opcheck 或 DomainParallel。

原始 samples：

`artifacts/native-batch-cell-build-query-materialization-trusted-plan-benchmark-v2/`

## 结论与下一步

Point 35 证明 prevalidated build plan 可以安全地进入当前隔离完整邻居 forward pipeline，
并且相对 public native build 保留约 2 倍 full-pipeline 收益。下一步应建立完整 pipeline 的
runtime-facing capability/错误合同，先补 fixed-cell Torch reference 的公开接线和回归，再
决定是否允许显式 `hip` 选择；不得直接把该 isolated factor 写成默认后端性能。
