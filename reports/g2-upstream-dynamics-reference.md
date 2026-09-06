# G2 上游 dynamics reference 子集

日期：2026-09-06  
目标：把锁定上游 `packages/framework/test/dynamics/test_ops.py` 中与
Velocity-Verlet、FIRE/FIRE2 对应的行为测试接入显式 Torch reference 运行，
同时验证默认 Warp 选择没有被改成静默回退。

## 测试载体

`test_ops.py` 保持上游测试默认语义，只增加两个测试环境变量：

- `NVALCHEMI_TEST_BACKEND`：作为测试调用的 `backend` 参数；未设置时为
  `None`，即上游 Warp 路径；
- `NVALCHEMI_TEST_DEVICE`：测试设备；未设置时为 `cpu`。

这两个变量只服务于上游测试的 reference/HCU 运行，不改变产品 API 默认值，
也不在 Warp 不可用时自动回退。

## 可重跑命令

CPU reference：

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference NVALCHEMI_TEST_DEVICE=cpu \
  PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_ops.py \
  -k 'VelocityVerlet or fire2 or fire_step'
```

HCU reference：

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference NVALCHEMI_TEST_DEVICE=cuda \
  HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 120 \
  PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_ops.py \
  -k 'VelocityVerlet or fire2 or fire_step'
```

默认路径检查（不设置上述变量）：

```bash
source scripts/activate_hygon_env.sh project
env -u NVALCHEMI_TEST_BACKEND -u NVALCHEMI_TEST_DEVICE \
  PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_ops.py \
  -k 'fire2 or fire_step or velocity_verlet'
```

## 结果

| 运行 | 结果 | 说明 |
|---|---:|---|
| CPU，显式 `torch_reference` | `23 passed, 74 deselected`，退出码 0 | float32/float64、异构 Batch、VV、FIRE/FIRE2 原位行为，以及公共 FIRE 单步测试 |
| HCU，显式 `torch_reference` | `23 passed, 74 deselected`，退出码 0 | 项目 `.venv`、DTK 26.04、BW200/UBB BW1000、gfx936、单卡 |
| CPU，默认 `backend=None` | `2 passed, 13 failed, 82 deselected`，退出码 1 | 失败均在 Warp kernel/import 边界；不是 reference 失败，证明未发生隐式回退 |

HCU 运行耗时约 1.21 秒，CPU 运行耗时约 1.03 秒。测试只覆盖该上游
`test_ops.py` 的算子/固定晶胞 FIRE 子集；Langevin、Nose-Hoover、NPT、变胞
FIRE2、Hooks/DataSink、inflight 和完整 dynamics 套件仍未解除阻断。

## 结论

上游行为测试现在有一个可重复的显式 reference 载体，并已在 CPU 与 HCU
通过。默认路径仍然要求 Warp；后续若要扩大范围，应逐个为 reference 实现
缺失的算子和数据契约，不能把整套上游测试标记为已通过。
