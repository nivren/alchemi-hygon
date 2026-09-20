# G2 CSR atom-list fill ABI and Torch oracle

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 11。

## 实现

本小点冻结 scan 后 CSR fill 的可替换边界：

- `build_cell_atom_list_reference_into` 读取 active `cell_keys`、最终 `cell_counts` 和 exclusive
  `cell_starts`，写 caller-owned `cell_atom_list` 与独立 `cell_cursor` scratch。
- `cell_atom_list` 是当前 active atom slice 的 local storage；写入的 atom ID 为
  `stable_argsort(cell_keys) + global_atom_offset`，因此 Batch 的切片可以保留全局 atom ID。
  capacity 可以大于 active atom 数，未使用尾部不被修改。
- `cell_cursor` 完成后等于 final counts，但 public `cell_counts` 和 `cell_starts` 保持不变。
  这对应上游 `bin_atoms` 的 atomic insertion cursor 语义，同时避免把 public count buffer 暴露为
  transient cursor。
- stable sort 保留当前 Torch cell-list 的公共 atom order。未来 HIP atomic fill 若不能复现该顺序，
  必须增加明确 canonicalization，不能未经验证直接放宽 public order。

`torch_reference_cell_list._build_into` 已通过该 ABI 完成 fill；neighbor query、capacity policy 和
Batch orchestration 的公共接口没有改变。

上游对应逻辑是锁定源码中的 `count_atoms -> array_scan -> atoms_per_cell_count.zero_() ->
bin_atoms`：`bin_atoms` 根据 cell start 与 atomic cursor 计算 list slot，并写 atom index。本地
reference 采用 stable sort 作为确定性 oracle，native HIP fill 尚未实现。

## 验证证据

CPU focused suite：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_cell_list_abi.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py
```

退出码 0，`27 passed`。新增测试覆盖 stable order、global atom offset、cursor final state、
public counts/starts 不变、capacity tail preservation、empty input 和 capacity 错误。
`compileall` 与 `git diff --check` 均通过。

主机权限 DTK 26.04、HCU 0（BW200/gfx936）上的限时
`probes/pbc_cell_list_reference.py --device cuda` 退出码 0：FP64 full/half edges 为 `8/4`，
pair/shift parity 为 true，layered counts 为 `[1,1]`，二阶梯度有限。该 probe 验证的是现有
Torch reference 端到端路径已复用 fill ABI，不是 native HIP fill 证据。

## 边界与下一步

本小点没有实现 native HIP fill kernel、没有测 fill 性能、没有处理 atomic fill 的非确定顺序，
也没有接入 `auto`、`hip` capability、AOT wheel 或新的 public neighbor API。fill 的 overflow 由
counts 总量、starts 合同和 caller capacity 检查显式拦截；查询仍使用现有 Torch 路径。

下一小点可在该 ABI 上实现 native HIP fill candidate：复用 counts/starts、使用独立 cursor、写
global atom IDs，并先验证 empty、single-system、heterogeneous Batch、PBC/triclinic、capacity 和
stream；在公共 order 尚未解决前不得把结果接入 runtime。
