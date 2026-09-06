# FIRE2 reference 与锁定上游 NumPy oracle 对照

日期：2026-09-06  
范围：固定晶胞、坐标 FIRE2 reference；不涉及 HCU 性能或变量晶胞 stress。

## 方法

上游 `packages/ops/test/dynamics/test_fire2.py` 顶层会导入 Warp，不能直接 import 测试模块。新增兼容测试从锁定上游文件的 AST 中提取 `_fire2_reference_step` 函数本体，只执行该纯 NumPy oracle，不复制或改写其算法。

被测实现为 `packages/framework/nvalchemi/_dynamics_reference/fire.py::fire2_step_coord`。测试使用异构 `batch_idx=[0,0,1,1,1,2,2]`，分别覆盖 float32 和 float64；每种 dtype 从同一随机状态连续执行 5 步，逐步比较 positions、velocities、alpha、dt 和 nsteps_inc。

## 验证

```bash
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_fire2_upstream_reference.py
```

结果：退出码 `0`，`2 passed`，耗时 `0.31 s`。float32 使用 `rtol=5e-5, atol=5e-7`，float64 使用 `rtol=1e-12, atol=1e-12`。

## 结论与边界

固定晶胞 FIRE2 reference 的逐式结果已与锁定上游 NumPy oracle 对齐；该证据闭合 A6 的语义对照缺口。它不覆盖变量晶胞、virial/stress、FIRE2 custom op/compile 性能、MACE 物理收敛或 HCU 性能。
