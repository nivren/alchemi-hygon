# G2 Batch query topology materialization HIP boundary

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 21。

## 目标

小点 20 的 breakdown 确认 Torch materialization helper 的主要热点是 stable topology
canonicalization/scatter。本小点实现一个隔离的原生 HIP correctness candidate，将 native query
产生的 unordered `(candidate_matrix, candidate_shifts, candidate_counts)` 规范化为 public
`matrix/shifts/counts`，但不计算 distance/vector，也不接入 runtime。

## 实现

新增 `_hip_batch_query_materialize.py` 及对应 `_native/batch_query_materialize.cpp/.cu`：

- 对候选的 active entries 生成 row/column/shift 字段和 int32 order；
- 依照上游 `_sort_pairs` 的等价稳定字段顺序，执行
  `shift_z -> shift_y -> shift_x -> column -> row` 五次 rocPRIM radix sort；
- 对 candidate counts 做 rocPRIM exclusive scan，得到每个 source row 的 public rank；
- 清空 caller-owned public buffers，按最终 order scatter matrix 与 image shifts，并复制 counts；
- 通过 PyTorch mutation-only custom-op/fake 注册和 lazy hipcc loader 提供显式隔离边界。

该 candidate 当前会在 C++ 侧分配临时 sort/scan workspace 和字段 buffer，并通过
`candidate_counts.sum().item()` 取得 scatter 长度，存在 host synchronization；因此本点只验证
语义，不包含性能优化结论。输入 validation 明确拒绝 CPU、非法 active index、capacity overflow、
非 contiguous 和 alias，不静默截断或 CPU fallback。

## 正确性证据

CPU boundary/reference focused test：`21 passed`，其中新增 topology materializer CPU boundary
`2 passed`；`compileall` 与 `git diff --check` 通过。

HCU 0/BW200/gfx936 上使用 DTK 26.04、PyTorch HIP `6.3.26093`，native `.cu` 经 PyTorch
HIP extension loader 转为 `.hip`，实际由 `/opt/dtk-26.04/bin/hipcc` 编译并链接。完整
`probes/native_batch_cell_query_candidate_hip_boundary.py` 通过，覆盖：

- FP32/FP64；
- mixed PBC、full/half query candidate；
- 2-system Batch、非默认 stream、empty candidate/public buffers；
- native topology public matrix/count/shift 与 Torch reference parity；
- 原有 candidate capacity overflow、distance/vector public parity 及一/二阶 geometry gradient。

本机实际目标架构为 `gfx936`；该次 loader 默认命令同时生成多个 DTK offload arch 的 code object，
其中包含 `gfx936`，没有把多架构编译时间或结果解释为额外设备证据。`dcc` 25.10/clang 17 是
`hipcc` 的底层编译工具链，项目入口仍是 `hipcc`。

## 结论与边界

已证明原生 HIP 可以在设备上复现当前 Torch public topology 的 stable row/column/shift 语义，
可作为后续性能实验的 correctness candidate。尚未测该 candidate 的性能，不能宣称比 Torch
helper 快；临时分配和 host sync 也不适合作为最终 production implementation。

仍未覆盖 target/pair outputs、pair-centric/sorted query、compile/opcheck、autograd（topology
本身不可微）、rebuild、FP64 performance、API wall-clock、端到端 MACE/FIRE2、dispatcher、
`auto` 或 `hip` capability。下一小点应在相同 `46/92 × 32/64` 高 Batch 矩阵上测 native topology
candidate 与 Torch topology stage 的稳态 HIP-event device time，并保留计时前后 parity。
