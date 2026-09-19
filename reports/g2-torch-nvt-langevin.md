# T1 Torch NVTLangevin reference

## 范围和 feature ID

- Feature: `dynamics.integrators`，窄 slice：固定晶胞 BAOAB Langevin NVT。
- 日期：2026-09-19。
- 实现分支：`codex/feature-torch-nvt-langevin`。
- 基线：`a7ce913`；本报告对应本分支已完成的窄 reference slice，合并状态由 Git 历史记录。
- 上游 framework：`4dfe3723def34df3fadb245981081ccf8c94c257`。
- 上游 ops：`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。

## 实现

- 新增 `nvalchemi._dynamics_reference.langevin` 的 Torch `B-A-O-A` 和最终 `B` reference entrypoint。
- 注册 `torch_reference.langevin-v1`，由 generic executor lazy binding 调用。
- `NVTLangevin(backend="torch_reference")` 在首次 concrete batch 上解析一次 selection 并复用；`backend=None` 仍保留 legacy Warp 选择。
- reference 支持 float32/float64、异构 per-system `dt/kT/friction`、原地 positions/velocities 更新和每次调用显式 seed。

## 契约边界

- 输入：positions/velocities/forces `[N,3]`，masses `[N]` 或 `[N,1]`，参数 `[M]`，`batch_idx [N]`。
- `temperature` 沿用上游 ABI 表示内部能量单位的 `kT`；公共 wrapper 将 Kelvin 转为 `kT`。
- 固定晶胞、无 PBC/neighbor 语义；forward-only state transition，不提供随机更新的 autograd 路径。
- 空原子输入有效；非正质量、非正 timestep、负 `kT`/friction、非有限参数和越界 batch index 明确报错。

## 验证

CPU：

```text
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q packages/framework/test/compatibility/test_dynamics_reference_langevin.py
12 passed
```

本轮补充的非统计 contract 覆盖：float32/float64、非均匀系统大小、`masses [N,1]`、
只修改 positions/velocities、finalize 的批索引路由，以及越界 batch index 的显式错误。

CPU probe：7 atoms / 2 systems / float64，friction=0 与 VV 一致、seed 可复现、结果 finite，退出码 0。

HCU：DTK 26.04，BW200/gfx936，`HIP_VISIBLE_DEVICES=0`，float64，7 atoms / 2 systems，使用显式 dispatcher `backend="torch_reference"`，结果 finite、friction=0 与 VV 一致、seed 可复现，退出码 0。

项目 CPU gate：ops 34 passed；framework compatibility 135 passed、1 deselected；state subset 22 passed；VV/FIRE/FIRE2 subset 24 passed；compileall 和 `git diff --check` 通过。

## 未验证范围

- 仅验证了窄 reference slice，不代表完整 `dynamics.integrators` 或生产 DCU dynamics 支持。
- 已覆盖当前 Torch contract 可映射的部分非统计行为；尚未覆盖上游 `test/dynamics/test_langevin.py` 的完整行为/统计套件、长轨迹温度平衡、checkpoint/restart、inflight state replacement、分布式 ownership、跨设备逐位随机一致性或 torch.compile。
- 未实现或验证 Triton/HIP 优化；`backend=None` 的 legacy Warp 路径未作为 HCU 验收目标。
