# G2 Point 38 framework native HIP neighbor executor boundary

日期：2026-09-20

## 目标

把 Point37 的 ops-side native HIP Batch wrapper 接入正式的

```text
framework selection -> ops dispatcher -> generic executor binding
-> native build/query/topology pipeline
```

只开放固定 cell、周期性、full-list、MATRIX 输出。`backend=None`/Warp、Torch
reference 和 `auto` 的默认行为保持不变。

## 实现

- 新增 `nvalchemiops._hip_neighbor_executor.neighbor_list`，适配公开
  `dispatch_neighbor_list` ABI：规范化 Batch/cell/PBC，生成固定 cell metadata，分配
  CSR/query/topology/output workspace，并调用 Point37 wrapper。
- native query 的候选容量使用显式 `max_neighbors`；未提供时从 Batch 体系大小开始，遇到
  native capacity overflow 在有限保守上界内增长，不截断邻居。
- registry 新增 `hip.neighbor.cell_list-v1`，executor 通过通用 lazy binding 加载；没有
  implementation-ID 分派表，也没有 CPU fallback。
- Point37 wrapper 的 geometry buffers 现在可选：framework topology-only 调用不再为
  未请求的 distance/vector 分配输出；直接 hybrid 调用仍可保留 Torch geometry 路径。
- `NeighborListHook` 对显式 HIP 的 `skin/rebuild` 请求明确失败；Point38 不把 lifecycle
  或 cached rebuild 误报为已支持。

## 支持宽度

已登记的 HIP capability：

- device：HIP PyTorch device（运行时标识为 `cuda`）
- dtype：float32/float64
- topology：Batch、fixed-cell、periodic、full-list、MATRIX
- 不支持：no-PBC、half-list、COO、target indices、pair outputs、variable-cell、
  skin/rebuild、DomainParallel、`auto` 选择和 native geometry backward

这是一个显式功能边界，不是默认后端或生产性能准入。

## CPU 验证

- ops registry/runtime focused suite：`38 passed`
- framework neighbor/executor focused suite：`11 passed, 1 warning`
- adapter `py_compile`、probe `py_compile` 和 `git diff --check`：通过
- CPU 对 explicit HIP 的 unsupported device/output 请求均显式失败，不加载 native
  extension，不 fallback 到 Torch。

## HCU 验证

环境：BW200、gfx936、Torch HIP `6.3.26093`、FP32、`HIP_VISIBLE_DEVICES=4`、capacity
256。HIP framework 调用与独立分层 Torch cell-list build/query ABI 对照；不是性能测试。

| workload | atoms | 结果 |
| --- | ---: | --- |
| 46 atoms/system × 32 systems | 1,472 | framework selection/executor/native build-query-topology parity 通过 |
| 92 atoms/system × 64 systems | 5,888 | framework selection/executor/native build-query-topology parity 通过 |

两组均比较 public neighbor matrix、counts 和 PBC shifts。probe 输出为
`status=passed`。

## 当前边界与下一点

当前仍未完成 MACE/FIRE2 端到端、Hook 无 skin 的长期 lifecycle reuse、half-list、COO、
no-PBC、variable-cell、native geometry backward、compile/opcheck、`auto` 性能准入和
production performance comparison。下一步应先做 Point39：在 `compute_neighbors` 与
适用的 Hook 路径上确认 framework 输出契约/拒绝边界，再测包含 native cell-list build
的 HIP/Torch 完整 API wall-clock；性能测试统一使用 `HIP_VISIBLE_DEVICES=4`。
