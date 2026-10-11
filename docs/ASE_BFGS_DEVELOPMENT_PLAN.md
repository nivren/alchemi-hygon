# ASE-compatible BFGS 分阶段开发记录

2026-10-09：用户已确认 M1–M4 为本轮功能范围，M5 按性能证据另行决定。
BFGS 是本地新增功能；架构参考上游最新 L-BFGS，算法独立对齐 ASE 3.29.0，
开发基于当前 Framework 0.2.0/Ops 0.4.1；不整体升级锁定版本。

| 阶段 | 目标 | 验收 | 当前状态 |
|---|---|---|---|
| M1 | 单体系固定晶胞 FP64 Ops：State、prepare/step、Torch reference、registry、restart | ASE CPU 逐步 H/谱/方向/步长/坐标；CPU 回归；限时 HCU smoke | 已完成实现与窄验收，见 M1 报告；待用户审阅，未提交/集成 |
| M2 | `ASECompatibleBFGS` 接入既有 BaseDynamics、模型/Hooks/收敛/状态生命周期 | CPU/HCU 单体系真实 MLIP，已有工作流回归，初始收敛与严格 fmax 判据 | 用户已确认并完成实现/窄验收，见 M2 报告；未提交/集成 |
| E1（条件阶段） | 消除 HCU eigensolver 库内主机数学路径，适配显式 rocSOLVER 后端 | info/residual/重构/正交/ASE轨迹/stream；小矩阵与 Batch 性能分别实测 | 用户已确认并完成实现/窄验收；显式 hip，当前构建/实测规模见 E1 报告；未提交/集成 |
| M3 | 单体系 ASE UnitCellFilter 的 `3N+9` pack/unpack、广义力及应力转换 | ASE 逐步对照、广义力有限差分、CPU/HCU 联合收敛 | 用户已确认并完成实现/单体系CPU/HCU窄验收，见M3报告；未提交/集成 |
| M4 | 异构 Batch、独立 Hessian、冻结/重排/退出/动态补位、CSP | 与独立运行一致；模型共享 Batch；真实分子晶体弛豫；端到端 profiling | 用户确认后完成 CPU/HCU 窄验收，见 M4 报告；未提交/集成 |
| M5 | 基于 profiling 按需优化 Torch/数学库或 HIP/Triton | 明确端到端收益，全部数值回归 | M5-1/2/3A/3B隔离阶段完成；M5-4显式动态HIP产品接入、目标数值门通过，最终真实工作流和性能见M5-4报告；未集成，生产/auto另行确定 |

M3 Filter 参数参照 ASE：默认 `cell_factor=N, mask=None, scalar_pressure=0,
hydrostatic_strain=False, constant_volume=False`，参考晶胞取进入 Filter 的晶胞；
参数语义和可支持组合须在 M3 合同中逐项验证，不能复用原生 `3N+6` 表示冒充等价。
StrainFilter/FrechetCellFilter 不在本轮。

设备原则：HCU 输入的 H 更新、eigh、方向、步长和坐标计算使用设备内 Torch；
CPU/HCU/NVIDIA 共享数学实现，硬件验证分别记录。M2 已证实 Torch n=138 调用主机
LAPACK；输出驻留 HCU 不等于库内数学全在 HCU。E1 显式 hip 已在实测范围解决，不能把 M2
标为全 HCU 求解支持。M1 单体系并非放弃 Batch；
M4 才实现异构存储/调度，避免未验证的状态混用。不得偷偷 CPU 回退或降精度。

Restart 验收本地 State 连续性，不要求 ASE JSON 互读。每阶段开始前说明范围并等
用户确认；确认后完成该阶段实现与验证；完成后总结，再提出下阶段目标等待确认。
未经授权不推送、不合并 develop、不启动下一阶段。M1 的合同见
[TORCH_ASE_BFGS_CONTRACT](TORCH_ASE_BFGS_CONTRACT.md)，实际证据另记报告。

M1 总结：[M1 报告](../reports/g2-ase-bfgs-m1.md)。M2 的 Framework/解析势/真实
MACE 小体系/性能探针与下一阶段提案见[M2 报告](../reports/g2-ase-bfgs-m2.md)。
E1 保留 Torch reference，仅适配分解实现，不改变 abs 谱、Hessian 或 maxstep。
候选 n<=200 单矩阵 Jacobi 比 Torch 慢，批量有收益；不直接固化产品阈值或 auto。
顺序 E1→M3→M4；用户逐阶段确认后均已完成窄验收。以下 E1/M3 为历史阶段记录，
当前 M4 状态与 solver 修订见本文末节。

E1 结果见[报告](../reports/g2-ase-bfgs-e1.md)与[求解器合同](HIP_BFGS_EIGH_CONTRACT.md)：
CPU gate550/3既有skip，原生HCU31项，解析势/MPA-0与ASE对照及restart；本机
trace无受监测host数学/矩阵回传。产品求解器等维batch已验证，BFGS自身仍是单体系。

## M3 已确认范围与交付

单体系 FP64 ASE UnitCellFilter `3N+9`，独立 pack/unpack/广义力适配，同一个
BFGS 核心与显式 Torch/HIP。按 ASE3.29 签名核实 mask、cell_factor、
hydrostatic_strain、constant_volume、orig_cell、scalar_pressure 及默认值；
保存参考晶胞/Filter 参数和广义历史，复用模型应力与现有 Hooks/neighbor 重建。
ASE逐步 q/H/方向/原子位置/cell/广义力/restart，适用保守情形做广义力有限差分；
投影情形严格比较 ASE 语义，不将 constant_volume 写成严格体积约束。CPU gate
与限时 HCU、真实周期 MPA-0 小体系联合收敛；不在 M3 提前建设异构状态管理，M4
继续交付 Batch/refill/CSP。

M3结果见[报告](../reports/g2-ase-bfgs-m3.md)与[UnitCellFilter合同](TORCH_ASE_UNITCELL_FILTER_CONTRACT.md)：CPU621/3既有skip、HCU变胞8/solver31回归、解析三轨迹ASE/FD、周期Al4 MPA-0 CPU/HCU显式HIP联合收敛及restart；实现未提交/集成。

## M4 已确认目标与方案

固定晶胞与UnitCellFilter的异构Batch BFGS；共享一次模型Batch求值，按稳定system_id
维护独立Hessian/历史/参考晶胞/Filter参数与计数，按自由度分组调用已有Torch或HIP
等维batched eigh，不用零填充改变谱。Hessian、方向、步长及几何更新留在输入设备上；
主机只负责ID/形状/生命周期调度。沿用既有Batch/FusedStage/Hooks/sampler/sinks接口，
实现严格逐体系收敛冻结、重排、退出与动态补位，避免复用上一体系的历史。
验收先CPU与独立ASE逐步/重启一致，再限时单卡HCU分组与生命周期；在现有CSP
实际输入上验证FIRE2→BFGS的真实MPA-0分子晶体批量变胞弛豫及联合收敛，记录模型
求值次数、成功结构总耗时与profiling。若缺少可重复CSP输入/参数，先完成独立工作，
再请用户指定这一验收数据。M5不自动启动，不重构通用引擎、不自动提交/推送/合并。

## M4 实际结果与当时的 M5-1 提案

M4 在 `wangleping/feat-ase-bfgs-m4` 完成独立 ID 状态、异构 DOF 分组、共享模型、
逐体系冻结/重排/退出/补位、Batch 快照及旧单体系快照提升。最终 CPU650 passed /
3既有skip，HCU solver39、Batch/变胞14 passed。真实46/92原子 MPA-0
FIRE2→BFGS 变胞两结构完整收敛，逐步 ASE 坐标/晶胞约1e-14、H约2.55e-11；
四次inflight弛豫全部收敛、退出及补位，约103.37秒。CPU真实固定晶胞通过配对
checkpoint续算收敛；耗时只记续算窗口。全部 M1–M4 工作未提交/集成/推送。

真实285-DOF简并Hessian暴露本机构建strided-batched DC VMFault及info=0错误谱；
当前HIP统一直接使用Jacobi并加设备内分解有效性守卫，不改变ASE算法。
E1旧DC性能不代表当前实现。详见[M4报告](../reports/g2-ase-bfgs-m4.md)及
[Batch合同](TORCH_ASE_BFGS_BATCH_CONTRACT.md)。仅当前HCU构建/小样本，不推导
其他硬件、完整数据集成功率、生产吞吐或通用checkpoint。

建议 M5-1 先做预热后的同输入基准与方案选择：真实Hessian/147与285 DOF、组规模
变化，分离eigh/有效性守卫/同步/copy/调度成本；库调用候选先过简并谱、ASE逐步、
原子失败和stream检查，再进行同配置CSP端到端比较。当前真实预热窗口6次eigh
设备累计约308ms、H更新约7.8ms，已有优化依据。M5仍可选，等待用户确认后启动；
不自动重启已知故障DC、不更改数学、不整体升级锁定依赖。

## M5-1 已确认交付与 M5-2 提案

2026-10-10用户确认M5-1后，在 `wangleping/perf-ase-bfgs-m5-1` 完成性能合同、
147/285 B1/B4同输入三进程×7样本稳态基准、真实异构组核心窗口、hipprof与
三次完整MPA-0 FIRE2→变胞BFGS两结构分项测量。11个BFGS/Filter产品源hash
未变，版本锁/环境不变；没有新增或优化求解器，全部未提交/集成/推送。
285/B1 raw约92.16ms、checked约92.64ms，GPU kernel占trace窗口97%；真实
CSP窗口中位数56.88秒，checked eigh占BFGS更新88.8%/整个窗口47.8%。
数值门、实际命令/退出码及仪表化边界见[M5-1报告](../reports/g2-ase-bfgs-m5-1.md)。

随后用户“继续”确认[M5-2隔离HIP batched Jacobi原型](ASE_BFGS_M5_2_PROPOSAL.md)。
自写FP64 round-robin Jacobi只在probes接入，147/285、B≤4；19例谱/残差/正交/
stream、10步ASE/重启/失败原子性、真实MPA-0变胞ASE对照均通过。
三进程×7样本checked solver耗时下降28.7%–56%，相对当前rocSOLVER Jacobi。
共享八卡作业占用HCU4期间的旧端到端计时序列排除并停止；后续用户确认新窗口
三轮CSP配对全部通过，中位数57.308→46.726秒，耗时下降18.466%；末态差异
与重复波动单独报告，产品11源hash/原型源/版本锁未变。
见[M5-2报告](../reports/g2-ase-bfgs-m5-2.md)与[原型合同](HIP_BFGS_JACOBI_PROTOTYPE_CONTRACT.md)。
M5-2隔离原型验收完成；用户随后确认[M5-3尺寸与Batch扩展](ASE_BFGS_M5_3_PROPOSAL.md)
拆为A/B验收。M5-3A已完成动态尺寸/Batch原型与窄验收，达到1113维/B32子集及
真实561维B8，原API尺寸白名单/固定B上限撤销。CPU专项209/HCU Framework14、
新11体系十步ASE/restart/晚块原子性及16个独立真实CIF短步对照通过；T2共享力
ASE数学不等价于CPU MLIP对照，其独立CPU MLIP门超时124。真实H分解窗口约
47%下降，部分合成规模退化；产品源/锁不改。详见[M5-3A报告](../reports/g2-ase-bfgs-m5-3a.md)。
M5-3A结束时下一步为M5-3B；用户随后确认并完成隔离比选，见[M5-3B报告](../reports/g2-ase-bfgs-m5-3b.md)。
该时点尚未注册产品；当前M5-4产品结果见下节。旧三轮CSP证据仍见[配对报告](../reports/g2-ase-bfgs-m5-2-e2e.md)。

## M5-4 已确认产品接入

用户确认后增加精确选择 `hip.bfgs-ase-jacobi-streams-v1`，单宽batched/多宽
双worker与双stream，无经验n/B白名单；默认/hip/auto不变。共享ASE数学、
Framework/Filter/ID/restart不改；完整候选验证后提交。CPU668/3既有skip，
最终HCU Framework24/native7、32矩阵/9真实组/11体系十步ASE通过，旧rocSOLVER39
回归通过；原生零轮精确对角谱恢复修正保留前后记录。真实CSP、性能配对、
profiler和未覆盖范围见[M5-4报告](../reports/g2-ase-bfgs-m5-4.md)与
[产品合同](HIP_BFGS_JACOBI_STREAMS_CONTRACT.md)。全部仍未提交/集成/推送。
随后用户确认最终交付审阅并完成：共享Batch非有限步长guard、CPU669/5skip、
HCU Ops18/Framework24/11体系ASE门与46/92完整CSP、两个包构建和独立wheel
CPU/HIP JIT验证通过。见[审阅报告](../reports/g2-ase-bfgs-delivery-review.md)及
[提交方案](ASE_BFGS_DELIVERY_PLAN.md)。仍未暂存/提交/集成/推送。
2026-10-11用户确认后按四组完成本地DCO sign-off提交；执行记录见
[交付方案](ASE_BFGS_DELIVERY_PLAN.md)。产品、测试及历史probe内容与最终审阅
一致，仅文档更新交付状态。本文上方“未提交”为各阶段当时的状态；当前个人
分支已本地收口，尚未合入develop/推送。下一阶段本地集成待确认，生产验证另行确定。
