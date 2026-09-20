# G2 native HIP Batch geometry/PBC/key/count fusion boundary

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 9。

## 实现

本小点实现 ADR 0009 的 build 前半段候选，不将单体系的 key/count API 机械串联：

- `_cell_list_abi.build_batch_cell_key_counts_reference_into` 是 Torch oracle。输入是扁平 atom
  positions、`(B,3,3)` inverse cells、`(B,3)` dimensions/PBC、任意 atom 顺序的 `batch_idx` 与
  严格连续的 global `cell_offsets`；输出为每原子的 periodic shifts、cell coordinates、global key
  和完整 `(M,) int32` counts。B=1 是同一 ABI 特例；空 atom input 仍完整清零 counts。
- `nvalchemiops._hip_batch_cell_key_count` 注册 private mutation-only custom-op/fake。native
  `.cpp/.cu` 先在 current stream 用 `hipMemsetAsync` 清零 counts，再由 one-thread-per-atom kernel
  取本体系 geometry/PBC/dimensions/offset，同时写三种 per-atom outputs 和 `atomicAdd` global count。
  只读 inputs 可为 strided，四个 outputs 必须 contiguous；CPU/非 HIP 明确失败，无 fallback。
- `cell_offsets` 必须正好等于每个体系 cell volume 的 exclusive prefix sum，且 `cell_counts` 必须
  恰好覆盖该串接后的 total cells；这禁止重叠、间隙或只为“能跑”而截断异构 Batch 的 counts。

该候选不执行 scan、CSR fill、stable atom ordering、query 或 neighbor materialization；它不接入
`torch_reference_cell_list`、dispatcher、AOT wheel、`auto` 或 `hip` capability。

## 验证证据

CPU focused suite：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_cell_list_abi.py \
  packages/ops/test/torch/test_hip_cell_key_boundary.py \
  packages/ops/test/torch/test_hip_cell_count_boundary.py \
  packages/ops/test/torch/test_hip_cell_scan_boundary.py \
  packages/ops/test/torch/test_hip_batch_cell_key_count_boundary.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py
```

退出码 0，`33 passed`。新 oracle 测试以交错 atom 的异构两体系 Batch 对照逐 system cell-key/count
结果，另覆盖不连续 `cell_offsets` 的显式错误。`compileall` 与 `git diff --check` 均退出码 0。

在主机权限 DTK 26.04 环境中，`/dev/kfd`、`/dev/dri` 可见；以下命令在 HCU 0 上首次 JIT 编译、
链接、加载并执行 native extension：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python probes/native_batch_cell_key_count_hip_boundary.py
```

BW200/gfx936、PyTorch HIP `6.3.26093` 上，257 atoms、两个异构 grid `[[4,5,6],[3,4,2]]` 与
mixed PBC `[[true,false,true],[false,true,false]]` 的 FP32/FP64 shifts/mapping/keys/counts 均与
Torch oracle 逐元素一致。B=1 empty input、strided read-only inputs 和 non-default stream 同样
通过。probe 明确输出 `performance_measured=false`。

## 边界与下一步

这是 fused geometry/PBC/key/count 的 correctness candidate，不是已加速的完整 Batch cell-list。
尚未测量 kernel/API 或 end-to-end 时间，未与已有独立 key+count API 作同条件比较，未测 scan/fill/
query、公开 neighbor order、梯度、compile/opcheck、AOT 或 selective rebuild。因此不改变默认路径、
`auto` 或 `hip` capability。

下一小点应先建立固定、可重复的组合 microbenchmark：相同异构 Batch、相同输出合同、预热后多样本
HIP-event 分别测量“独立 native key + native count”与本融合 candidate。只有确认可重复的差异后，
再决定保留融合实现、调整设计，或进入 CSR fill；仍不接线 runtime。
