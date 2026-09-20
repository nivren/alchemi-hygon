# G2 native HIP cell-start scan boundary

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 8。

## 实现

本小点将 CSR starts 从 count/fill 生命周期中单独固定：

- `_cell_list_abi.build_cell_starts_reference_into` 以 non-negative `(M,) int32` counts 写出
  同形 exclusive starts，并加调用方给定的 `global_atom_offset`。counts 是只读输入；该
  flattened active-cell slice 同时适用于单体系和已应用 per-system cell offset 的 Batch slice。
- `nvalchemiops._hip_cell_scan` 增加显式 native HIP custom-op。它以 rocPRIM 的 two-call
  temporary-storage query 获取 caller-owned `uint8` workspace，再在当前 Torch stream 上执行
  `rocprim::exclusive_scan`；CPU/非 HIP 调用明确失败，不回退 Torch/CPU。
- count、start、workspace 的 dtype、shape、device、alias、offset、non-negative count 与 int32
  range 均有显式检查。CSR convenience API 仍先覆写未初始化的 count output，再调用 scan；因此
  CSR 输入校验不会错误读取调用方尚未写入的 count buffer。

该模块没有接入 `torch_reference_cell_list`、native count、cell-key、fill、query、dispatcher 或
`auto`；也没有 AOT wheel artifact。

## 验证证据

CPU focused suite：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_cell_list_abi.py \
  packages/ops/test/torch/test_hip_cell_key_boundary.py \
  packages/ops/test/torch/test_hip_cell_count_boundary.py \
  packages/ops/test/torch/test_hip_cell_scan_boundary.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py
```

退出码 0，`29 passed`。参考合同覆盖 counts `[1,1,0,2,0,3]`、offset `10`，得到 starts
`[10,11,12,12,14,14]` 且 counts 不变；同时覆盖负 count 与 int32 range 错误。`git diff --check`
和新增 Python 文件的 `compileall` 均退出码 0。

在主机权限 DTK 26.04 环境中，`/dev/kfd` 和 `/dev/dri` 可见。以下 HCU 0 限时 probe 首次 JIT
编译、链接和加载 rocPRIM extension 后退出码 0：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python probes/native_cell_start_scan_hip_boundary.py
```

BW200/gfx936、PyTorch HIP `6.3.26093` 的 probe 确认六个 cells 的 output parity、counts unchanged、
empty input、strided read-only counts 与 non-default stream；其输出明确标记
`performance_measured=false`。随后既有 `probes/pbc_cell_list_reference.py --device cuda` 退出码 0：
FP64 full/half edges 为 `8/4`、pair/shift parity 为 true、二阶梯度有限。

## 边界与下一步

这只是可复用的 global exclusive-scan 边界，不是完整 HIP neighbor backend，也不意味着 native
count+scan 已接入或更快。未测 scan 或完整 build 性能，未测 AOT、fill/query、公开 neighbor order、
梯度、compile/opcheck，尚未实现 geometry/PBC/count 的 Batch-aware fused kernel。因此不登记 `hip`
capability、不改变默认路径或 `auto`。

下一小点应按 ADR 0009 定义并验证 batch-aware fused geometry/PBC/count candidate：先相对现有
cell-key 与 count oracle 验证单体系和 heterogeneous Batch 的输出/错误/stream 合同，再决定是否做
组合 microbenchmark；仍不进入 fill、query 或 runtime 接线。
