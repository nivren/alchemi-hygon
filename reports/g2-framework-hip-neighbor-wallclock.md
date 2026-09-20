# G2 Point 39 framework HIP neighbor API wall-clock

日期：2026-09-20

## 目标与范围

测量公开 framework `compute_neighbors` 的 build-inclusive API wall-clock，范围为：

```text
framework selection -> ops dispatcher -> generic executor
-> fixed-cell metadata -> cell-list build -> native query
-> composite topology materialization -> Batch write-back
```

本点只测 topology-only `MATRIX` 输出，不包含距离/向量 geometry，也不代表
MACE/FIRE2、skin/rebuild、half-list、COO、变胞或 `auto` 已支持。Hook 只核对
`skin=0` 成功和 `skin>0` 的明确拒绝边界。

## 环境与测量合同

- 设备：BW200 / UBB BW1000，gfx936；所有 HCU 命令使用
  `HIP_VISIBLE_DEVICES=4`。
- 软件：项目 `.venv`、DTK 26.04、Torch HIP `6.3.26093`。
- dtype：FP32；cutoff `0.6`；固定 `max_neighbors=256`；mixed-PBC 三斜 cell；
  随机输入 seed `20260920`。
- workload：46 atoms/system × 32 systems（1472 atoms），92 atoms/system × 64
  systems（5888 atoms）。
- API wall-clock：每个样本前后调用 `torch.cuda.synchronize()`，用 host
  `perf_counter()` 计时；每个 backend 独立进程，warmup 3 次，采样 7 次。
- cold：独立进程的第一次 public API 调用，包含 runtime/extension load、首次
  workspace/output allocation 和第一次执行；本次使用已编译的 Point38 extension
  cache，因此不把源文件首次编译时间混入 cold 数值。此前使用全新 Point39 cache
  的运行在 180 秒上限内只停留在 DTK 预处理，未作为性能样本。
- warm：预热后的完整 public API 调用，保留全部原始样本于 `artifacts/`。

运行入口为：

```text
probes/framework_neighbor_hip_wallclock.py
```

实际运行命令的共同前缀为：

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR=artifacts/point38-cell-key-count \
NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR=artifacts/point38-cell-scan \
NVALCHEMI_HIP_CELL_FILL_BUILD_DIR=artifacts/point38-cell-fill \
NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR=artifacts/point38-cell-query \
NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_BUILD_DIR=artifacts/point38-topology \
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 \
PYTHONPATH=packages/framework:packages/ops timeout 240 \
.venv/bin/python probes/framework_neighbor_hip_wallclock.py \
  --backend {hip|torch_reference} --atoms-per-system {46|92} \
  --batch-systems {32|64} --warmup 3 --samples 7 \
  --output-dir artifacts/framework-neighbor-hip-wallclock-{46x32|92x64}
```

## 结果

单位均为 ms；factor 定义为 `HIP warm median / Torch reference warm median`，
小于 1 表示 HIP API wall-clock 更短。

| workload | backend | cold | warm median | warm mean | rel. pstdev | HIP/Torch warm factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 46×32 | HIP | 955.521 | 22.389 | 22.962 | 6.32% | 0.09398× |
| 46×32 | Torch reference | 731.852 | 238.222 | 238.900 | 0.70% | — |
| 92×64 | HIP | 972.553 | 39.399 | 39.396 | 0.066% | 0.07839× |
| 92×64 | Torch reference | 1009.494 | 502.571 | 503.581 | 0.393% | — |

原始 warm samples：

- `artifacts/framework-neighbor-hip-wallclock-46x32/hip-46x32-warm-ms.txt`
- `artifacts/framework-neighbor-hip-wallclock-46x32/torch_reference-46x32-warm-ms.txt`
- `artifacts/framework-neighbor-hip-wallclock-92x64/hip-92x64-warm-ms.txt`
- `artifacts/framework-neighbor-hip-wallclock-92x64/torch_reference-92x64-warm-ms.txt`

## 正确性与 Hook 边界

- 46×32 和 92×64 的 HIP public output 均与同一 framework
  `backend="torch_reference", method="cell_list"` 的 matrix/counts/shifts 对照通过。
- `NeighborListHook(backend="hip", method="cell_list", skin=0)` 在两组规模均成功写出
  `MATRIX` 数据。
- `NeighborListHook(..., backend="hip", skin>0)` 明确抛出
  `BackendUnavailableError`，原因是 native HIP executor 尚未实现 skin/rebuild lifecycle。
- 每次性能任务结束后 `/opt/hyhal/bin/hy-smi --showpids` 均为
  `No KFD PIDs currently running`。

## Point39 中修复的两个边界问题

1. Torch reference `batch_cell_list` 的 allocator 按 `max_nbins=8192` 计算总 cell
   容量，但 build 阶段没有沿用该上限；在大 Batch mixed-PBC 输入上会出现
   `cell buffers have capacity 0, need 1`。现在 build 与 allocation 使用同一上限，新增
   `test_large_mixed_pbc_batch_uses_consistent_cell_capacity`，CPU cell-list suite 为
   `14 passed`。
2. HIP public executor 在 topology workspace allocation 前使用未初始化的
   `candidate_counts` 做范围校验，重复调用可能随机失败。现在在 query 写入前显式
   `zero_()`，保证 cold 和 warm 调用共享确定性 output contract。

## 结论与限制

在本次固定 periodic/full/MATRIX、FP32、46×32 和 92×64 workload 上，HIP public
executor 的 steady-state API wall-clock 约为 Torch reference 的 `0.078--0.094×`，
即观测到约 `10.64--12.76×` 的倒数关系。该结果覆盖了 framework 接线和 native
cell-list build/query/topology，但不等于所有邻居模式的性能结论，也不构成 `auto`
准入：当前 HIP 仍未覆盖 no-PBC、half、COO、target/pair、variable-cell、skin/rebuild、
native geometry backward、compile/opcheck、DomainParallel 或 MACE/FIRE2 端到端。

下一点建议是 Point40：在保持 `auto` 不变的前提下，补齐显式 HIP executor 的
capacity/empty/dtype/format/unsupported-request 回归矩阵，并确认一次 Hook 无 skin 的
可重复调用与输出契约；之后再决定是否需要 workspace reuse/lifecycle 改造，而不是直接
把本点的窄性能结果推广为生产默认策略。
