# MPA-0 FP32 状态验收分层 M8

## 目的与范围

针对 M7 发现的重复约束问题，分别检查模型物理输出和优化器状态。
本阶段获用户确认的范围仅包括 `probes/` 测试/验收代码和文档，
不改 `packages/framework` 或 `packages/ops` 源码、kernel、模型或全局容差。

M7 在 Batch mutable fields 中先按能量/力/应力逐物理量容差核验，随后又将缓存的
`energy`/`forces`/`stress` 当作一般优化器状态套用通用 `atol=2e-5`。
FP32 FIRE2 在该重复比较失败；独立纯 Torch/Torch FIRE2 对照也曾触发相同限制。

## M8 比较规则

- 拓扑 `(i,j,sx,sy,sz)`、体系边界、整数计数/状态/步数继续精确相同。
- `energy`、`forces`、`stress` 由已有 `mace_mpa0_cpu_baseline.compare()` 按精度和物理单位比较。
- mutable `positions`、`cell`、`velocities` 及优化器私有状态继续使用 M7 状态容差：
  FP64 `rtol=2e-8, atol=2e-9`；FP32 `rtol=2e-4, atol=2e-5`。
- FusedStage 最终缓存输出同样走逐物理量比较；位置、晶胞、速度仍走状态比较。
- 新增 CPU 单元测试证明：物理容差内的力差不会被重复状态门拒绝；物理容差外仍失败；
  几何和私有状态仍受原状态阈值约束。

## 验收

- CPU comparator 单测全部通过。
- 对同一真实 CIF 清单、MPA-0 SHA、普通 e3nn eager、cell-list/HIP selection、
  FIRE/FIRE2、FP32/FP64、1029/2030 原子组合，顺序重跑8组 M7 轨迹/补位/性能探针。
- 对照输出保持原 E/F/stress 容差，状态/拓扑门不变；失败组保留日志和退出码。
- 记录实际通过组及所有性能数据；不预设加速门槛，也不将预算退出称为物理收敛。

## 实施状态

比较器CPU单测3项通过。HCU八组中6组完整退出码为0；其余两组的物理输出、轨迹几何/速度和
优化器私有状态比较均通过，最后仅触发既有“所有输入结构的晶胞都必须改变”检查。加上失败
计数诊断后，1000原子FP32 FIRE2为46/47个晶胞变化，2000原子FP32 FIRE2为99/100；不改变该
验收条件。所有八组的初始/step3邻居拓扑与模型物理输出比较通过。详情见
[M8报告](../reports/g2-hip-mpa0-state-validation-m8.md)。M8结果完成，待用户确认收口。
