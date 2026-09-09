# G2 no-PBC Torch reference cell-list：契约与回归

日期：2026-09-08。此次里程碑新增显式 backend
`torch_reference_cell_list`，只覆盖 no-PBC 邻居构建；没有改变默认 `None`/Warp 或
显式 `torch_reference` 路径，也没有让 `auto` 选择 cell-list。

## 实现范围

- 新增 `packages/ops/nvalchemiops/torch_reference_cell_list.py`：以 cutoff 为 cell
  edge，按每个 Batch system 建立 uniform Cartesian cell list，候选只来自 27 个相邻
  cell，随后在设备张量上进行精确距离筛选。
- 支持 full/half、MATRIX/COO、距离/向量、空输入、异构 Batch、确定性 row-major
  `(source,target)` 顺序和显式邻居容量溢出。
- 中央 registry 注册 no-PBC capability；`compute_neighbors(...,
  backend="torch_reference_cell_list")` 已接线。periodic、target rows、pair callbacks、
  scratch 和动态 rebuild 参数显式失败。

## 回归命令与结果

CPU ops：

```bash
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_reference_cell_list.py \
  packages/ops/test/torch/test_backend_registry.py
```

结果：`10 passed`，退出码 `0`。

CPU framework：

```bash
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/models/test_neighbors_torch_reference.py \
  -k 'cell_list or torch_reference_preserves_batch_boundaries or unregistered'
```

结果：`3 passed, 2 deselected`，退出码 `0`。

HCU（source DTK 26.04，`HIP_VISIBLE_DEVICES=0`，设备 `BW200, UBB BW1000`）同一
ops 命令结果为 `10 passed`、退出码 `0`；framework 命令结果为 `3 passed, 2
deselected`、退出码 `0`。

## 尚未宣称的能力

当前只有小型 synthetic 输入的语义证据；真实 `perf_46/92/184/368` 的 cell-list
性能、周期 cell-list、half-list 周期语义、skin/rebuild 的生产性能以及 `auto` 选择
仍未完成。下一步先给 unified benchmark 增加显式 backend 选项，再在 CPU/HCU 上与
现有 dense `torch_reference` 做同口径 no-PBC 计时对比。
