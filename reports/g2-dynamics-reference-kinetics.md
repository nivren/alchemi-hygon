# G2/N2a：Warp-free kinetic energy and temperature reference

日期：2026-09-06  
目标设备：海光 BW200 / UBB BW1000，gfx936

本步新增 `nvalchemi._dynamics_reference.kinetics`，将上游
`dynamics/hooks/_utils.py` 中依赖 Warp 的动能路径提取为 Torch reference：

- 对异构 Batch 按 `batch_idx` 计算 `0.5 * sum(m_i |v_i|²)`，返回 `[B, 1]`；
- 接受质量 `[N]` 或 `[N, 1]`；
- 按上游 `3N` 自由度和 `KB_EV` 计算每体系瞬时温度；
- 空批次、设备/shape/dtype 不一致和非法体系索引显式报错；
- 动能计算使用 `torch.library.custom_op`，供后续 reference hooks/compile 接线。

## 可重跑验证

项目 `.venv` CPU 测试：

```bash
PYTHONPATH=packages/framework .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_velocity_verlet.py
```

结果：`7 passed`，退出码 `0`。其中新增测试覆盖异构 Batch 动能/温度、二维质量、
空 Batch、非法温度原子数和无 Warp 导入。

CPU 探针：

```bash
PYTHONPATH=packages/framework .venv/bin/python \
  probes/dynamics_reference_kinetics.py --device cpu
```

结果：动能 `[7.0, 6.0]`，温度 `[27077.2089507397, 46418.07248698234]`，
`warp_imported=false`，退出码 `0`。

HCU 探针：

```bash
source /opt/dtk-26.04/env.sh
PYTHONPATH=packages/framework HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  timeout 90 .venv/bin/python probes/dynamics_reference_kinetics.py --device cuda
```

结果：退出码 `0`；设备 `BW200, UBB BW1000`，动能 `[7.0, 6.0]`，温度
`[27077.2089507397, 46418.07248698234]`，`warp_imported=false`。

## 边界

本步没有修改 Warp-backed `nvalchemi.dynamics.hooks._utils`，也没有把 reference
函数接入 LoggingHook、EnergyDriftMonitorHook 或公共 NVE。segment max/min/mean
和这些 hooks 的导入隔离仍属于后续 N2b；当前结果只证明 reference 计算本身在
项目 `.venv` CPU/HCU 正确。
