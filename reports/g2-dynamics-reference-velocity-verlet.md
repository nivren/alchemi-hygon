# G2/N2a：Warp-free velocity-Verlet reference

日期：2026-09-06  
目标设备：海光 BW200 / UBB BW1000，gfx936  
源码工作树：`codex/g0-initialization`

本步把 velocity-Verlet 的两个原位更新放在
`nvalchemi._dynamics_reference.velocity_verlet`。该顶层模块不导入
`nvalchemi.dynamics`、`nvalchemiops` 或 Warp，避免公共 dynamics 包当前的
eager Warp 导入链。更新支持 `[N, 3]` 原子坐标/速度/力、`[N]` 或 `[N, 1]`
质量、异构批次的 per-system `dt` 和 int32/int64 `batch_idx`：

- position update：位置更新和当前力的半步速度 kick；
- velocity finalize：新力的第二个半步速度 kick；
- 输入 shape、dtype、device 和 batch 索引错误显式报错；
- 使用 `torch.library.custom_op` 形式，保留后续 `torch.compile` 接线的形态。

## 可重跑验证

CPU 单元测试（项目 `.venv`）：

```bash
PYTHONPATH=packages/framework \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_velocity_verlet.py
```

结果：`5 passed`，退出码 `0`；覆盖 Warp-free import、float32/float64、
异构 batch、原位 mutation、finalize 和非法 batch index。

CPU 探针：

```bash
PYTHONPATH=packages/framework \
  .venv/bin/python \
  probes/dynamics_reference_velocity_verlet.py --device cpu
```

结果：`max_position_error=6.938893903907228e-18`，最终速度最大绝对值为
`0.0`，`warp_imported=false`，退出码 `0`。

HCU 探针（必须在可见 `/dev/kfd`/`/dev/dri` 的主机 shell 执行）：

```bash
source /opt/dtk-26.04/env.sh
PYTHONPATH=packages/framework HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  timeout 90 .venv/bin/python \
  probes/dynamics_reference_velocity_verlet.py --device cuda
```

结果：退出码 `0`；设备 `BW200, UBB BW1000`，dtype `torch.float64`，
`max_position_error=6.938893903907228e-18`，最终速度最大绝对值为 `0.0`，
`warp_imported=false`。

## 边界

本步没有修改 `nvalchemi.dynamics._ops.velocity_verlet.py` 或完整 `NVE` 类，
因此上游默认 Warp 路径和公共 dynamics 导入仍保持原状。尚未覆盖 kinetic
energy/temperature、DataSink、FIRE/FIRE2、FusedStage/inflight，也没有将该
独立算子描述为完整 NVE API 已支持。下一条 N2a 小步是提取 Warp-free 的
kinetic energy/temperature 参考；随后再移植固定晶胞 FIRE/FIRE2，最后由 N2b
处理公共 dynamics 的后端委派和导入阻断。
