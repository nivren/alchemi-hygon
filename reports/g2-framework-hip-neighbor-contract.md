# G2 Point 40 framework HIP neighbor contract matrix

日期：2026-09-20

## 目标与范围

补齐 Point39 之后显式 HIP framework neighbor executor 的回归边界：

- 已登记 periodic/fixed-cell/full/MATRIX 的 FP32、FP64 正常路径；
- empty Batch 输入；
- 显式容量溢出；
- no-PBC、half-list、COO unsupported request；
- `NeighborListHook(skin=0)` 的重复调用和输出稳定性。

本点是 correctness/contract 验证，不是性能测试，不改变 `auto`、Torch reference 或
HIP capability 登记宽度。

## 环境与命令

- 设备：BW200 / UBB BW1000，gfx936；`HIP_VISIBLE_DEVICES=4`。
- 软件：项目 `.venv`、DTK 26.04、Torch HIP `6.3.26093`。
- 正常 workload：46 atoms/system × 32 systems，1472 atoms，cutoff `0.6`，
  `max_neighbors=256`，mixed-PBC 三斜 cell。

CPU focused tests：

```text
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_hip_neighbor_executor.py
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/models/test_neighbors_torch_reference.py
```

HCU probe：

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

### CPU selection/error contract

- ops HIP registry tests：`7 passed`；覆盖 registered periodic/full/MATRIX，以及
  no-PBC、half、COO、distances、vectors 的 capability rejection。
- framework neighbor tests：`11 passed, 1 warning`；新增 periodic COO 和 half-list
  显式 HIP rejection。
- probe `py_compile` 与 `git diff --check`：通过。

### HCU functional matrix

| case | result |
| --- | --- |
| FP32 46×32 | HIP public matrix/counts/shifts 与 Torch reference parity 通过 |
| FP64 46×32 | HIP public matrix/counts/shifts 与 Torch reference parity 通过 |
| empty periodic input | matrix `(0,256)`、counts `(0,)`、shifts `(0,256,3)` 通过 |
| `max_neighbors=1` close-pair fixture | `native HIP Batch query capacity overflow` 明确报错 |
| periodic half-list | capability 明确拒绝 |
| periodic COO | capability 明确拒绝 |
| no-PBC | capability 明确拒绝 |
| `NeighborListHook(skin=0)` repeated twice | 两次 matrix/counts/shifts 完全一致 |

HCU probe 输出 `status=passed`，结束后 `/opt/hyhal/bin/hy-smi --showpids` 为
`No KFD PIDs currently running`。

## 结论与剩余范围

Point40 已验证当前 HIP narrow capability 的正常 FP32/FP64、空输入、容量错误、格式/拓扑
拒绝和无 skin Hook 重复调用契约。该证据不表示 HIP 支持 no-PBC、half、COO、skin/rebuild、
variable-cell、target/pair、native geometry backward、compile/opcheck、DomainParallel 或
MACE/FIRE2。

下一点建议是 Point41：把 Point40 的 HCU contract probe 纳入一个稳定的 framework/ops
compatibility gate，并补一组 `max_neighbors=None` 的自动容量增长/overflow 上界验证；仍然
保持显式 HIP、`auto` 不变，再决定是否进入 Hook workspace/lifecycle reuse。
