# G2 Point 41 framework HIP neighbor compatibility gate

日期：2026-09-20

## 目标与范围

Point41 是 `TORCH-NEIGHBOR-PBC-CELL` 阶段二的最小收口点：将现有
`framework_neighbor_hip_contract.py` 作为稳定 compatibility gate，并验证
`max_neighbors=None` 的自动容量增长、有限上界和不截断邻居。

本点不新增 kernel、backend capability 或性能策略；不进入 `auto`，也不实现
lifecycle reuse、skin/rebuild、变胞、native geometry backward 或 compile/opcheck。

## 验证命令

CPU gate：

```text
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_hip_neighbor_executor.py
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/models/test_neighbors_torch_reference.py
.venv/bin/python -m py_compile probes/framework_neighbor_hip_contract.py
```

HCU gate（第五张 DCU）：

```text
source scripts/activate_hygon_env.sh project
NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR=artifacts/point38-cell-key-count \
NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR=artifacts/point38-cell-scan \
NVALCHEMI_HIP_CELL_FILL_BUILD_DIR=artifacts/point38-cell-fill \
NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR=artifacts/point38-cell-query \
NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_BUILD_DIR=artifacts/point38-topology \
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 \
PYTHONPATH=packages/framework:packages/ops timeout 300 \
.venv/bin/python probes/framework_neighbor_hip_contract.py \
  --atoms-per-system 46 --batch-systems 32
```

## 结果

- CPU ops registry：`7 passed`。
- CPU framework neighbor：`11 passed, 1 warning`。
- HCU：BW200/gfx936、Torch HIP `6.3.26093`、`HIP_VISIBLE_DEVICES=4`，probe
  `status=passed`。
- 自动容量 fixture：单原子、`cell=2I`、cutoff `2.1`，初始容量 `1`，最终容量
  `8`，最大实际邻居数 `6`，保守上界 `343`；没有截断，且与 Torch reference
  对照通过。
- 这里的 `max_neighbors=None` 是当前 HIP executor 的本地 doubling policy；它只保证有限
  上界内安全增长，不复刻上游 `estimate_max_neighbors` 的最小 `16`/16 对齐布局，也不保证
  与 Torch reference 的 padded capacity 形状相同。该容量布局差异不改变 active neighbor
  集合，但在接入更宽 public/lifecycle capability 前仍需单独决定是否统一。
- Point40 已有的 FP32/FP64、empty、显式 capacity overflow、unsupported request
  和 repeated `skin=0` Hook 检查继续通过。
- gate 结束后 `/opt/hyhal/bin/hy-smi --showpids` 报告
  `No KFD PIDs currently running`。

## 阶段二收口结论

Point41 完成了当前显式 HIP 邻居窄 capability 的最后一项容量安全回归。阶段二在以下
范围内收口：periodic、fixed-cell、Batch、full-list、MATRIX、FP32/FP64、`skin=0`，
并支持 `max_neighbors=None` 自动容量增长。

这不表示完整上游 neighbor capability 或 MACE/FIRE2 生产支持。no-PBC、half-list、COO、
skin/rebuild、变胞、target/pair、native geometry backward、compile/opcheck、DomainParallel
和 `auto` 仍是后续独立任务。
