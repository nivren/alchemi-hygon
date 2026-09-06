# FIRE2 reference + skin rebuild 功能契约

日期：2026-09-06  
范围：异构 `[3,5]` Batch、固定晶胞、`NeighborListHook(backend="torch_reference", skin=0.5)` 与公共 `FIRE2(backend="torch_reference")` 的组合正确性。此项不测性能。

## 验证内容

测试通过公共 `BaseDynamics.step()`，因此每一步都包含 FIRE2 的坐标更新、模型计算和 `BEFORE_COMPUTE` 邻居 Hook：

- 首步建立 `cutoff + skin` 的 COO 邻居缓存；
- `0.1 < skin/2` 的位移复用缓存；
- 只移动 system 0 超过 `skin/2` 时，仅刷新 system 0 的 reference 坐标；
- `batch_ptr=[0,3,8]`、`batch_idx` 顺序和邻居边体系归属保持不变；
- positions 和 forces 保持 finite。

## 运行证据

CPU 项目环境：

```bash
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_fire2_skin_functional.py
```

结果：退出码 `0`，`1 passed`，约 `5.27 s`。

HCU 项目环境（DTK 26.04，BW200/UBB BW1000，gfx936，单卡可见）：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_fire2_skin_functional.py
```

结果：退出码 `0`，`1 passed`，约 `4.30 s`。

## 结论与边界

该组合契约在 CPU 和 HCU 均通过，证明 FIRE2 reference 的实际 `BaseDynamics` 流程可以驱动 reference skin 缓存，并且局部重建不破坏异构 Batch 边界。它不代表 cell-list、生产 Warp 路径、Triton/HIP kernel、异步 rebuild 或性能基线已经完成；当前 HCU 处于共享负载，性能 profile 仍暂停。
