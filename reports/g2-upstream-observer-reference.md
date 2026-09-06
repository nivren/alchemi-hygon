# 上游 observer hook 的 Torch reference 回归

日期：2026-09-06  
范围：SnapshotHook、ConvergedSnapshotHook、LoggingHook、EnergyDriftMonitorHook 和 hook lifecycle。此轮修复并验证了 observer 数值辅助函数的显式 reference 后端，不涉及性能 profile。

## 实现边界

`scatter_reduce_per_graph` 新增 `backend` 参数：`None`/`"warp"` 保留上游
segmented Warp kernel，`"torch_reference"`/`"auto"` 使用 Torch
`index_add`/`scatter_reduce`。`LoggingHook` 和 `EnergyDriftMonitorHook` 新增
`compute_backend`；未显式设置时会读取 `ctx.workflow.backend`，因此公共
`NVE/FIRE/FIRE2(backend="torch_reference")` 可以让 observer 跟随 reference，
而普通 `BaseDynamics` 仍默认走 Warp。非法 backend 显式报错。

上游 `test_observer_hooks.py` 仅增加 `NVALCHEMI_TEST_BACKEND` 测试载体：设置
`torch_reference` 时将该值传给两个 observer 的计算 backend；未设置时仍为
`None`，不改变默认 Warp 语义。

另有 8 项 framework compatibility 回归覆盖公共 NVE/FIRE/FIRE2 异构 Batch 与
segment 的 sum/amax/amin/mean 及未知 backend 显式错误。

## CPU

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference \
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_observer_hooks.py::TestSnapshotHook \
  packages/framework/test/dynamics/test_observer_hooks.py::TestConvergedSnapshotHook \
  packages/framework/test/dynamics/test_observer_hooks.py::TestLoggingHook \
  packages/framework/test/dynamics/test_observer_hooks.py::TestEnergyDriftMonitorHook \
  packages/framework/test/dynamics/test_observer_hooks.py::TestHookLifecycle
```

退出码 `0`，`44 passed, 33 skipped`，约 `3.20 s`。skip 是 CUDA 参数化用例在
CPU 进程中的既有条件跳过。

## HCU

```bash
source /opt/dtk-26.04/env.sh
NVALCHEMI_TEST_BACKEND=torch_reference \
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
timeout 90 .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_observer_hooks.py::TestSnapshotHook \
  packages/framework/test/dynamics/test_observer_hooks.py::TestConvergedSnapshotHook \
  packages/framework/test/dynamics/test_observer_hooks.py::TestLoggingHook \
  packages/framework/test/dynamics/test_observer_hooks.py::TestEnergyDriftMonitorHook \
  packages/framework/test/dynamics/test_observer_hooks.py::TestHookLifecycle
```

退出码 `0`，`77 passed`，约 `2.32 s`；设备为 DTK 26.04 的 BW200/UBB BW1000
（gfx936）。覆盖 CPU 与 HCU Batch 上的逐图 fmax、temperature、energy drift、
快照 alias 解耦、ConvergedSnapshot 过滤和 hook 生命周期。

## 默认路径反向验证

```bash
env -u NVALCHEMI_TEST_BACKEND \
  PYTHONPATH=packages/framework:packages/ops timeout 60 .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_observer_hooks.py::TestHookLifecycle::test_idempotent_close_via_user_with_and_engine
```

退出码 `1`，在 `_segmented_max` 的 `import warp` 边界失败。该结果是预期的
默认 Warp 反向验证，证明新增 reference 分支没有把默认路径静默改成 Torch。

## workflow backend 自动继承

使用公共 `FIRE(backend="torch_reference")`，不在两个 observer 上显式设置
`compute_backend`，运行一条异构 `[3,5]` Batch 的单步流程：CPU 与 HCU 均写出
1 条逐图日志、`fmax` 为有限值，且 `warp` 不在 `sys.modules`。这确认
`ctx.workflow.backend` 的继承路径可用于真实 reference dynamics；普通
`BaseDynamics` 没有该属性时仍保持默认 Warp。

## 限制

此轮只覆盖 observer 的 Torch reference 计算和 HostMemory/自定义 writer 行为；
Warp 默认 segmented kernel、GPUBuffer/ZarrData、异步长轨迹和完整 checkpoint/restart
仍未验证。该 `compute_backend` 只选择 observer 的数值辅助后端，不改变日志输出
backend（csv/tensorboard/custom）。

compatibility 回归命令：

```bash
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_public_dynamics_reference.py
```

CPU 退出码 `0`，`8 passed`，约 `4.32 s`；同一命令在 DTK 26.04、BW200/gfx936
HCU 退出码 `0`，`8 passed`，约 `17.92 s`。
