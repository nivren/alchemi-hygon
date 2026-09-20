# G2 native HIP cell-key atomic-count boundary

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 5。

## 实现

本小点将共享 CSR ABI 的 count phase 单独固定，并增加显式调用的 HIP 候选实现：

- `_cell_list_abi.build_cell_counts_reference_into(cell_keys, cell_counts)` 是 count-only Torch
  oracle。它完整覆写 `(M,) int32` output，包括空输入；`build_cell_csr_reference_into` 复用它，
  再以 Torch 计算 exclusive starts。
- `nvalchemiops._hip_cell_count` 注册私有 mutation-only
  `nvalchemiops::_cell_key_count_hip` custom-op 和 fake 实现。调用方提供 contiguous `(M,) int32`
  count buffer；CPU 或非 HIP Torch 显式失败，不会 fallback 到 Torch/CPU。
- `_native/cell_key_count.cpp/.cu` 在调用方当前 stream 上以 `hipMemsetAsync` 清零 output，再让每个
  key 对 `cell_counts[key]` 执行一次 `atomicAdd`。只读 `cell_keys` 可为 strided，native boundary
  会连续化其临时读取副本。输入 key 的 dtype、shape、同设备、范围和 int32 count-overflow 上界由
  shared ABI 在 launch 前检查。

该模块没有被 `torch_reference_cell_list` 调用：reference 仍以 Torch `bincount` 完成 count，scan、
sort/fill、query 与 dispatcher 均未改变。当前 loader 仅是 lazy JIT；既有 cell-key AOT wheel 未注入
本模块的 `.so`。

## 验证证据

CPU focused 回归：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_cell_list_abi.py \
  packages/ops/test/torch/test_hip_cell_key_boundary.py \
  packages/ops/test/torch/test_hip_cell_count_boundary.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py
```

退出码 0，`25 passed`。新增 ABI 测试确认 count 覆写 complete active buffer；HIP boundary 导入不
编译/不导入 Warp，CPU 调用明确失败，不触发回退。`git diff --check` 退出码 0。项目 `.venv` 没有
安装 Ruff，故本轮没有 Ruff 执行结论。

主机权限 DTK 26.04 环境中，`/dev/kfd` 与 `/dev/dri` 可见；以
`HIP_VISIBLE_DEVICES=0` 运行以下限时 probe，JIT 编译、链接、加载和设备执行均退出码 0：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python probes/native_cell_key_count_hip_boundary.py
```

PyTorch HIP `6.3.26093` 在 BW200/gfx936 上使用 4099 个 keys、137 个 cells。native count 与
Torch oracle 逐元素一致；空输入、strided read-only keys 和非默认 Torch stream 同样通过。probe
明确 `performance_measured=false`。随后既有
`probes/pbc_cell_list_reference.py --device cuda` 退出码 0：full/half edges 为 `8/4`、
pair/shift parity 为 true、二阶梯度有限，证明 count ABI 抽取未改变 reference PBC cell-list。

## 边界与下一步

这不是 CPU 到 DCU 的迁移：Torch `bincount` 与新 HIP kernel 都在 HCU 上执行。它只是为同一
`cell_keys -> cell_counts` 语义提供专用 HIP 实现候选。尚未测量 isolated count 或完整 build 的
耗时，因此不能声称更快，也不登记 `hip` capability、不改变 `auto` 或默认 Torch reference。

下一小点建议先以固定 keys/cell-occupancy/workload、预热与重复样本测量 Torch `bincount` 和 native
count 的 device 时间；同时报告 dense-cell atomic contention。仅在有结果后决定保留该 candidate 的
后续接线优先级，仍不进入 scan/sort/fill/query 或 dispatcher。
