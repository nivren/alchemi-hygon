# G2 cell-key CSR count/start ABI

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 4。

## 实现

在 shared cell-list ABI 增加独立于排序/query 的 CSR count/start phase：

- `build_cell_csr_reference_into(cell_keys, cell_counts, cell_starts, *, global_atom_offset)`
  是权威 mutation API。`cell_keys` 为 `(N,) int32`，两个输出为同设备 `(M,) int32`；`M` 是
  active cell 数，counts 和 starts 不得 alias。
- counts 是每个 key 的精确桶计数；starts 是 counts 的 exclusive prefix sum 加
  `global_atom_offset`。stable `cell_atom_list` 排序仍在独立阶段，因而该 ABI 不承诺 atom
  order 或完成 fill。
- key 越界、dtype/device、output shape/alias、offset 类型/范围和 int32 count/start overflow
  明确失败。空 key 输入将所有 counts 写 0、所有 starts 写为 offset。
- `torch_reference_cell_list._build_into` 现在复用本 ABI 写 active CSR slices；预分配 buffer
  的尾部清零、capacity 检查、stable sort 和 global atom offset 语义保持原有行为。

## CPU 与 HCU 证据

CPU focused pytest：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_cell_list_abi.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py -k 'cell_key or cell_list'
```

退出码 0，`20 passed`。独立合同例子 `keys=[3,0,3,1]`、`M=5`、offset=10 的 counts 为
`[1,1,0,2,0]`，starts 为 `[10,11,12,12,14]`；空输入 offset=7 的 starts 为 `[7,7,7]`。

在主机权限 DTK 26.04、`HIP_VISIBLE_DEVICES=0`、BW200/gfx936 上，
`probes/cell_key_csr_reference.py --device cuda` 退出码 0，得到同一合同结果。随后
`probes/pbc_cell_list_reference.py --device cuda` 退出码 0：pair/shift dense parity 为 true，
full/half edges 为 `8/4`，layered counts 为 `[1,1]`，二阶梯度有限。

## 边界与下一步

这一步只实现 Torch reference CSR count/start，并用 HCU Torch 执行验证；没有 native HIP
atomic count kernel、没有 device scan、没有 sort/fill/query、没有性能数据，也不改变 AOT wheel
内容、`hip` capability、dispatcher 或 `auto`。

下一小点应针对已冻结的 count output ABI 增加 native HIP atomic count custom-op：显式清零
count buffer、key range 错误、空输入和 stream parity 先过本报告 oracle；starts 仍由 Torch
reference scan 产生，直到单独的 scan 模块有证据。
