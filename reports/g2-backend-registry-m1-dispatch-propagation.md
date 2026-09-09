# G2/M1 operation selection propagation

日期：2026-09-09

## 范围

本轮继续完成 M1 的 framework → dispatcher 接线。目标是让 framework 对每个 operation
解析一次 `BackendSelection`，低层 dispatcher 消费该选择，不以原始 backend request
再次解析；同时把 FIRE 与 FIRE2 登记为独立 operation。

## 实现结果

| operation | implementation ID | framework 入口/消费者 |
| --- | --- | --- |
| `lj_energy_forces` | `torch_reference.lj_energy_forces-v1` | `LennardJonesModelWrapper.forward` |
| `velocity_verlet` | `torch_reference.velocity_verlet-v1` | NVE、FIRE/FIRE2 的 VV 阶段 |
| `fire` | `torch_reference.fire-v1` | 固定/变胞 FIRE |
| `fire2` | `torch_reference.fire2-v1` | 固定/变胞 FIRE2 |
| `kinetics` | `torch_reference.kinetics-v1` | Logging、drift monitor、reporting scalar |
| `periodic_wrap` | `torch_reference.periodic_wrap-v1` | `WrapPeriodicHook` |
| `segmented_reduce` | `torch_reference.segmented_reduce-v1` | observer 的逐 graph fmax |

固定晶胞 NVE/FIRE/FIRE2 在首次 state initialization 阶段冻结 selection；LJ 在 forward 入口冻结 selection；
periodic、observer 和 reporting 辅助路径按 operation 解析一次并向下传递。分布式 FIRE
辅助入口也能接收已解析 selection，DomainParallel 的 owned-position wrapping 也缓存
periodic selection，避免重新退回默认 backend。

变胞 FIRE/FIRE2 默认仍使用 legacy Warp。显式请求 Torch reference 时，framework 会
以 `variable_cell` 特征解析并明确报告 capability 不满足；当前没有把固定晶胞 reference
能力扩大成变胞支持。

## 验证

以下均使用项目 `.venv`，CPU，`PYTHONPATH=packages/framework:packages/ops`，未使用
HCU 资源：

- ops registry/reference：`23 passed, 1 warning`，退出码 0。
- selection propagation 回归：`3 passed`，退出码 0。
- framework reference/observer/periodic/LJ slice：`103 passed, 1 deselected`，退出码 0；连同本报告的
  selection propagation `3 passed` 合并运行时为 `106 passed, 1 deselected`。
- NVE/FIRE/FIRE2 state lifecycle：`22 passed, 34 deselected`，退出码 0。
- dynamics import boundary 与 reference wrapper：`15 passed`，退出码 0。
- touched files `compileall` 与 `git diff --check` 通过。

selection propagation 回归包含反向保护：预先取得 selection 后，测试禁止 dispatcher
再次调用中央 resolver；LJ、dynamics、periodic 和 observer 均通过该保护。

## 数值结果

本轮只改变 backend selection 的传递和 operation registry 语义，没有改变 Torch reference
公式、FIRE/VV 更新顺序、邻居集合或 dtype。既有 LJ、VV、FIRE/FIRE2、kinetics 和周期
reference 数值/HCU 证据仍按原报告解释；本轮没有新增 HCU 数值结果。

## 未验证与限制

- 没有新增 Triton/HIP dispatcher 或 BackendProfile/PipelinePlanner；M2 仍未开始。
- 变胞 Torch reference、stress、Langevin/Nose-Hoover/NPT、compile fullgraph 和多卡
  dynamics 仍不在本轮支持范围。
- 本轮 selection 传递回归是 CPU 证据，不能写成完整 DCU backend 支持。
- 完整 framework 上游套件仍会触及尚未迁移的 Warp-only dynamics 模块；本报告只记录
  可隔离、可重跑的 reference slice。
