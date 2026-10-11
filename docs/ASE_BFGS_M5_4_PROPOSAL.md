# 下一阶段提案：M5-4 显式优化后端的产品接入

用户于2026-10-10确认；本阶段已完成实现与限定范围验收，见
[M5-4报告](../reports/g2-ase-bfgs-m5-4.md)。M5-3B的隔离对照保留，默认不改变。
最终选择与收益以 [M5-3B报告](../reports/g2-ase-bfgs-m5-3b.md) 为依据。

## 目标和范围

把已验收的动态 batched Jacobi 与有限跨组调度接入 Ops 显式优化后端，
保留当前 rocSOLVER 和 Torch reference 供独立选择及回归。继续严格 ASE 数学。
不引入 n/B 白名单，不将 worker 数作为 Batch 容量限制，不升级 DTK/Torch。
模型仍整 Batch 计算；各矩阵计算仍在输入 HCU，主机保留控制元数据与标量等待。

以独立 implementation ID `hip.bfgs-ase-jacobi-streams-v1` 注册优化实现。
接口、输入和资源生命周期见[产品合同](HIP_BFGS_JACOBI_STREAMS_CONTRACT.md)。
单个同宽组使用已验收 batched Jacobi；多个尺寸组使用经过实测的 2 worker
调度。具体小规模是否需要串行阈值，以 M5-3B 和本阶段实测为准，不能凭空设置。
原生异构和 4 stream 保留为实验对照，是否注册由证据决定，不扩大默认路径。

## 实施与验收

1. 产品 Ops 复用一份 BFGS 数学表达式，增加先准备全部 H、联合提交分解、
   全部方向通过后提交的内部调度入口；避免长期 ragged State Manager。
2. JIT 原生入口、worker 的创建/关闭、device/stream/event 与张量生命周期有
   明确合同。保留 info、有限值、方程/正交检查，失败等待所有工作后报错。
   原型 API 不是 autograd/compile/fake 产品承诺；按现有机制声明支持范围。
3. Framework 显式传递后端，不修改 Hooks、Filter、restart schema、FIRE/FIRE2。
   未支持 device/dtype/layout/功能仍明确失败；不实现静默回退或 `auto` 选择。
4. CPU原有门和目标HCU完整相关门；动态尺寸/Batch、ASE逐步、变胞、restart、
   冻结/重排/退出/补位、故障原子性和跨stream压力检查。
5. 同输入、同容差、相同输出检查的配对性能门；覆盖 Li异构、T2同宽、大/小
   Batch 的退化案例。补充既有46/92原子两体系的并行后端完整CSP收敛回归；
   Li/T2真实 MLIP 短步与完整收敛分开报告。M5-3B末态位置差达1.72e-8 Å，
   下一阶段保留逐步ASE门，并检查模型反复求值的轨迹差；不以微基准推断全目录吞吐。
6. 同步来源/许可、功能契约、状态矩阵、环境限制、使用说明与命令/退出码报告。
   完成后汇总供用户审阅；不自动提交、合并、推送或改变默认后端。

本阶段的目标是可显式使用和可维护的产品实现。全目录 CSP 成功率、生产规模
压测、默认/auto 选择、其他硬件/构建、NVIDIA、compile 和梯度仍另行确定。
