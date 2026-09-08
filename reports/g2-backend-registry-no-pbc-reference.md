# G2：集中 backend registry 与 no-PBC 邻居 Tier 1

日期：2026-09-08。代码状态以本次工作树和后续提交为准。

## 实现范围

- `nvalchemiops.backend` 从单一 reference 选择器升级为 operation/device/dtype/gradient/features capability registry；`None`/`warp` 只记录 legacy 选择，不触发 Warp 导入。
- framework 的 neighbors、NeighborListHook、LJ、固定晶胞 VV/FIRE、observer 与 periodic helper 统一通过该 registry；`auto` 首次选择会报告实际 backend。
- `LevelStorage` 的 Torch 默认作为正式数据层契约，Warp 对照为 `WARP-EQUIV-001` strict xfail。
- no-PBC reference 邻居将逐原子/逐边 Python 矩阵写回替换为逐 system 的张量 pair geometry、`nonzero`、`bincount`、row-rank 和 scatter；不改变 O(N²) 算法或 cell-list 状态。

## CPU 验证

在项目根目录、项目 `.venv`、`PYTHONPATH=packages/ops` 下：

```bash
.venv/bin/python -m pytest -q packages/ops/test/torch/test_backend_registry.py packages/ops/test/torch/test_torch_reference_backend.py
```

结果：`17 passed`，退出码 `0`。其中覆盖 capability 拒绝、auto 选择记录、legacy 无 Warp probe、no-PBC full/half/MATRIX/COO 行顺序、距离/向量、overlap 与 overflow。

在项目根目录、`PYTHONPATH=packages/framework:packages/ops` 下运行 storage、framework neighbors/Hook、默认路径边界和 public dynamics reference 子集，结果为 `32 passed, 2 xfailed`；LJ、dynamics reference、observer/periodic 和 state/inflight 子集分别为 `7 passed`、`23 passed`、`89 passed`、`22 passed`，退出码均为 `0`。两个 strict xfail 是无 Warp 环境的 `WARP-EQUIV-001` 与 `WARP-DEFAULT-001`，不是 reference 路径 skip。

## HCU smoke

在 DTK 26.04、BW200/gfx936 的 HCU 0 上，以 `HIP_VISIBLE_DEVICES=0`、`OMP_NUM_THREADS=1` 运行同一 ops 回归，结果为 `17 passed, 1 warning in 18.67s`，退出码 `0`。`BackendAutoSelectionWarning` 是预期的 `auto -> torch_reference` 选择记录。

同一设备运行 `probes/no_pbc_neighbor_reference.py --device cuda --dtype float32 --warmup 1 --steady 3`，退出码 `0`。异构 `[46,92]` full/half 分别产生 `890/445` 边，最大 K 为 `16/15`，MATRIX、distance、vector 的 shape 分别为 `[138,91]`、`[138,91]`、`[138,91,3]`；预热后的单次调用约为 `1.90--2.01 ms`。HCU 和 CPU 的随机生成器序列不以相同边数为验收条件；full/half、shape、ops 合同回归均在各自设备实际运行。本次是空闲单卡上的功能 smoke，不能作为发布吞吐基准。

另以两个 no-PBC 双原子 `AtomicData` 构成的 HCU `Batch` 调用 framework `compute_neighbors(..., backend="torch_reference")`，MATRIX 写回为 `[[1], [0], [3], [2]]`、每行计数为 `[1,1,1,1]`，退出码 `0`。这只验证 registry 到 framework Batch 写回的最小集成路径。

## 边界与下一步

`probes/no_pbc_neighbor_reference.py --device cpu --dtype float32 --warmup 1 --steady 2` 退出码为 `0`：异构 `[46,92]` full/half 分别为 `734/367` 边，最大 K 为 `10/9`，两次 steady 为约 `1.22/1.27 ms` 与 `1.33/1.26 ms`。这是小型 synthetic CPU smoke，不是发布性能数据。

no-PBC 的小输入 device-side 装配已获得本轮 HCU smoke 证据，但尚无低干扰、固定结构阶梯的性能基线；periodic 既有性能数字也不外推到该变更。下一步在低干扰窗口用固定结构集重跑 no-PBC full/half/异构 batch，并据此决定是否开始 cell-list；Triton、HIP、周期 half-list、PBC 容量压力和 NVT/Langevin 均未实现。
