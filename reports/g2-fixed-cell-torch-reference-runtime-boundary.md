# G2 fixed-cell Torch reference runtime boundary

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，Point 36 的第一个
runtime-facing slice。

## 目标

闭合 fixed-cell Torch reference cell-list 从 framework neighbor hook 到
force-consuming model 的调用链：

```text
NeighborListHook(backend=torch_reference, method=cell_list)
    -> neighbor_list dispatcher
    -> periodic Torch cell-list
    -> Batch neighbor matrix + image shifts
    -> Torch reference LJ energy/force
```

## 实现与合同

- 新增 periodic fixed-cell `NeighborListHook -> LennardJonesModelWrapper`
  regression，使用 `backend="torch_reference"`、`method="cell_list"` 和 signed
  PBC image shifts。
- 新增 framework boundary regression：显式 `backend="hip"`,
  `method="cell_list"` 仍抛出 `BackendUnavailableError`，不会静默转到 Torch。
- `backend=None`、`backend="warp"` 和 `backend="auto"` 的既有语义未修改；不登记
  native HIP，不接 trusted plan，不接 native forward-only geometry。

## CPU 验证

```bash
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/models/test_lj_torch_reference.py \
  packages/framework/test/models/test_neighbors_torch_reference.py \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py
```

结果：`33 passed, 1 warning`，退出码 `0`。

同时复跑 ops registry/cell-list contract：`31 passed`，退出码 `0`。
`git diff --check` 和新增测试的 `py_compile` 通过。

## 证据边界

本点新增的是 CPU framework wiring/error evidence，不是 HCU 性能证据。Point 35
的 gfx936 isolated full-pipeline parity 仍有效，但不改变本点的 runtime selection：
native HIP 仍未登记，MACE、FIRE2、rebuild/skin、variable-cell、native geometry
autograd、compile/opcheck 和 DomainParallel 仍不在本点范围内。

## 下一步

继续 Point 36 的第二个 slice：把 fixed-cell Torch reference 的 capability/error
contract 收敛到一个可审查的选择记录，并为未来 native HIP wrapper 定义最小输入输出
ABI；在该 ABI 和错误边界确定前，不允许 `backend="hip"` 进入 registry 或 `auto`。
