# G2 native HIP Batch neighbor runtime ABI/admission gate

日期：2026-09-20

## 目标

Point 36.3 为已完成的隔离 native HIP stages 定义最小 runtime-facing
ABI/admission gate，但暂不把 `backend="hip"` 注册到 dispatcher 或 `auto`。
这样后续 wrapper 接线可以复用同一份边界，不会在 `compute_neighbors` 与
`NeighborListHook` 中各自复制不完整的 capability 判断。

## 当前候选 ABI

候选流水线固定为：

```text
checked build init
  -> trusted reusable CSR workspace
  -> native unordered candidate query
  -> native composite-key canonical full topology
  -> Torch reference distance/vector geometry
```

其中：

- build workspace 由调用方持有；只有在 shape、dtype、device、alias、cell
  metadata、capacity 和 workspace/storage 不变时才能复用，结构变化必须重新初始化；
- query 的候选顺序不作为 public contract，由 topology materialization 统一规范化；
- topology 是离散、forward-only 的 native stage；Torch geometry 保留连续位置/位移
  路径及一阶、二阶梯度语义；
- ABI 当前只覆盖 Batch、fixed-cell、full-list、float32/float64 和
  `geometry_backend="torch_reference"`。

## 明确拒绝的请求

admission gate 显式拒绝 CPU 请求、float16、非 Batch、variable-cell、half-list、
skin/rebuild、target-index、pair-function output、distributed/DomainParallel，
以及 native HIP geometry 请求。native geometry 仍是 isolated forward-only candidate，
没有进入这个训练/力路径 ABI。

拒绝统一抛出 `BackendUnavailableError`，不静默 fallback 到 Torch。

## 代码与验证

- ABI/gate：`packages/ops/nvalchemiops/_hip_batch_neighbor_runtime.py`
- CPU contract tests：`packages/ops/test/torch/test_hip_batch_neighbor_runtime.py`
- focused result：`15 passed`
- `git diff --check`：通过
- HCU：本点未新增 kernel 或 device execution，因此没有新增 HCU 证据

这只是 semantic contract validation，不代表 native HIP 已注册、已接入 framework、
已通过 MACE/FIRE2 端到端验证或已具备生产性能支持。下一小点应将该 gate 接到一个
不改变现有默认路径的 native wrapper 骨架/显式调用边界，并继续保持未满足条件时的
明确失败。
