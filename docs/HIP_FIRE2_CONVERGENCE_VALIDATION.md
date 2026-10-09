# HIP 邻居 FIRE2 小规模完整弛豫验收

日期：2026-10-09。用户确认的补充验证；独立分支
`wangleping/test-hip-fire2-convergence`，产品起点 `ad18998`，两包源码与 develop `2e211ab` 相同。

## 范围与来源

使用已锁定 MACE-MPA-0 medium 和三个既有 Li/Na CIF，FP64/FP32 各运行显式
HIP 和 Torch reference 邻居。优化器、晶胞对齐和应力转换继续使用 Torch reference。
模型为 ordinary e3nn/eager；保持模型权重、输入与进程默认 dtype 一致，冻结权重，
保留位置/应变 autograd。不修改产品源码、数值容差、上游锁或 Torch/DTK。

复用公共 FIRE2VariableCell、AlignCellHook、ConvergenceHook、NeighborListHook、
FusedStage、SizeAwareSampler 和 HostMemory。HIP 为 periodic/full-list、native MATRIX
+ 设备内 Torch COO 打包、skin=0、每次按当前晶胞完整重建；不准入其他 HIP 请求。
来源和原短步边界见 [M3 契约](HIP_VARIABLE_CELL_WORKFLOW_CONTRACT.md)，
物理判据沿用 [MPA-0 M5 契约](TORCH_MACE_MPA0_RELAXATION_CONTRACT.md)。

## 输入和运行约束

- checkpoint SHA256：`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
- Li2MgGe(4)、LiSnCl3(10)、NaLiTl2(4)，记录原文件哈希与 source_index。
- 最大活跃 2 结构 / 14 原子，总共 3 结构，必须实际补位；system_id 与来源索引分别核验。
- HCU4、OMP_NUM_THREADS=1、dt=0.02、零外压；保留 FIRE2 默认控制参数。
- 每体系最多 500 次实际更新，步间墙钟预算 180 秒；每次作业外层 timeout 300 秒。
  最终独立核验另记录，不把墙钟/步数退出算作物理成功。

## 验收标准

1. 三结构均满足 `fmax <= 0.01 eV/Å AND max(abs(stress)) <= 0.1 GPa`；
   四个 dtype/backend 作业分别记录结果、实际步数和退出原因。
2. 每次实际 shared compute 检查 dtype/device、有限性、正体积和体系内邻居；
   按当前几何重建 reference 邻居，atom/image 多重集合精确一致。
3. 初始和新到达结构首个 kick 前有当前 E/F/stress；记录有效来源、Batch ptr、
   首次 admission 与实际更新次数。结果 source_index/system_id 恰好各一次。
4. 终态保存 lossless Torch 张量与 CIF；在同一终态重新求值 HIP/reference E/F/stress，
   对照邻居集合及完整物理量。CIF 重读后另用 CPU/HCU reference 模型核验双阈值。
5. 同几何容差沿用 M2/M5：FP64 rtol2e-8、E/F/stress atol2e-7/2e-7/2e-9；
   FP32 rtol2e-4、atol2e-3/2e-4/2e-6。长非线性轨迹分别收敛，不增加逐步完全相同条件；
   报告两分支终态差异，不用终态差异替代同几何严格门。

先运行相关 CPU 生命周期/判据回归，再顺序运行四个单卡作业。记录精确源码 SHA、
probe 哈希、设备、环境、命令、超时与进程退出码。资源不可用则记录未运行并继续整理证据。

这是固定三样本的完整弛豫证据，不扩展为任意材料/大 Batch、性能、Warp/NVIDIA、
compile、checkpoint 重启、skin 缓存或 DomainParallel 支持。本次不合并或推送。
