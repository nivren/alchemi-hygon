# G2/N2a：Warp-free fixed-cell FIRE/FIRE2 reference

日期：2026-09-06  
目标设备：海光 BW200 / UBB BW1000，gfx936

本步新增 `nvalchemi._dynamics_reference.fire`，实现固定晶胞、`batch_idx`
路径的 Torch reference：

- FIRE `fire_step`：按体系归约 power/速度平方/力平方，更新 `alpha`、`dt`、
  正功步数，混合速度，执行质量加权 kick 和 `maxstep` 位移限制；
- FIRE `fire_update`：复用同一套归约、混合和参数更新，不执行坐标更新；
- FIRE2 `fire2_step_coord`：按上游 deferred half-step、混合、最大位移和
  uphill 清零语义实现；
- 异构 Batch 的 per-system 参数和体系边界保持独立，所有状态原位更新；
- `fire2_step_coord_cell` 明确拒绝，因为变胞路径仍需要 virial/stress 契约。

## 可重跑验证

项目 `.venv` CPU reference 测试：

```bash
PYTHONPATH=packages/framework .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_fire.py
```

结果：`4 passed`，退出码 `0`。测试包含 FIRE 正功/负功分支、无 MD 的
`fire_update`、FIRE2 独立逐原子标量参考对照、Warp-free 导入和变胞显式失败。

CPU 探针：

```bash
PYTHONPATH=packages/framework .venv/bin/python \
  probes/dynamics_reference_fire.py --device cpu
```

结果：退出码 `0`；`fire_positions_norm=0.03498742631289132`，
`fire2_positions_norm=0.1441066271897306`，`fire2_dt=[0.04, 0.04]`，
`fire2_nsteps=[2, 2]`，`warp_imported=false`。

HCU 探针：

```bash
source /opt/dtk-26.04/env.sh
PYTHONPATH=packages/framework HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  timeout 120 .venv/bin/python probes/dynamics_reference_fire.py --device cuda
```

结果：退出码 `0`；设备 `BW200, UBB BW1000`，上述 FIRE/FIRE2 状态与 CPU
探针一致，`warp_imported=false`。HCU 运行使用 Torch `scatter_reduce_`，没有
依赖 Warp 或 Triton。

## 边界

本步只建立顶层 reference 原语，没有修改公共 `nvalchemi.dynamics._ops.fire`
或优化器类，也没有解除 `nvalchemi.dynamics` 的 eager Warp 导入。FIRE 的
uphill energy-check 分支、FIRE2 变胞、FusedStage/inflight、收敛 Hook 和
MACE 端到端流程留到 N2b/N2c；因此本报告不宣称公共 FIRE/FIRE2 API 已支持。
