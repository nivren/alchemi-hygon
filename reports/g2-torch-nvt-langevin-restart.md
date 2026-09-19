# Torch reference Langevin minimal restart

日期：2026-09-19
分支：`codex/feature-torch-nvt-langevin-restart`

## 范围

本步只实现固定晶胞、普通 Batch 的 Langevin 积分器续跑状态，不引入通用
checkpoint、inflight、`atom_ptr`、`_out` 或分布式状态框架。

`NVTLangevin.state_dict()` 保存：

- restart version；
- `step_count` 和 `random_seed`，用于恢复 `random_seed + step_count` 的随机序列；
- 已初始化时的 per-system `dt`、`temperature` 和 `friction`。

位置、速度、力、模型参数和 Batch 元数据仍由调用方保存。这是积分器 continuation
state，不是完整 checkpoint。

## 验收

测试在第 3 步保存状态和 Batch 快照，创建新的 `NVTLangevin`，恢复状态后再运行 2 步；
与同一初始状态连续运行 5 步比较 positions、velocities、forces 和 energy。

CPU：

```bash
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_langevin_restart.py
```

结果：`2 passed`，退出码 `0`，约 3.57 秒。

HCU 0：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_langevin_restart.py
```

结果：`2 passed`，退出码 `0`，约 22.00 秒。

## 边界

这证明的是同一设备、同一模型、同一普通 Batch 下的积分器续跑语义；不证明完整
上游 checkpoint/restart、模型/优化器联合保存、FusedStage/inflight、Batch 动态补位、
分布式 ownership、跨设备逐位一致或生产后端支持。
