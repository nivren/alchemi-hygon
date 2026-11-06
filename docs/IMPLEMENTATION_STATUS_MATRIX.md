# 实现状态与应用能力矩阵

更新时间：2026-11-06（本次仅更新BFGS集成状态）。本文是面向开发和方案评审的总览，按 framework、ops/后端和应用能力
分别列出当前已经完成的范围。它不替代 [`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)
的逐特性契约；状态判断以 [`STATUS.md`](STATUS.md)、源码、测试和报告为准。

本次集成与候选复验见[报告](../reports/partner-reference-stage1-integration-2026-11-06.md)；不扩大既有科学和性能验证范围。

## 状态口径

| 标记 | 含义 |
|---|---|
| 已验证窄范围 | 代码、测试和指定范围的 CPU/HCU 证据均存在，但不能推广为完整上游或生产能力 |
| 已实现 reference | 已有可运行实现和回归，验证范围较窄或主要是 CPU/reference 证据 |
| forward-only 候选 | 已有前向实现或实验路径，但训练梯度、完整接线或 capability 准入未完成 |
| 计划/未接入 | 当前没有可交付的完整实现，不能用于产品能力承诺 |
| legacy 保留 | 上游 Warp/API 路径仍保留，但不表示已在 Hygon DCU 上验证 |

## 1. Framework 模块

| 模块 | 公共入口/职责 | 当前状态 | 已确认范围 | 尚未覆盖 | 主要证据 |
|---|---|---|---|---|---|
| 数据模型与 Batch | `AtomicData`、`Batch`、`LevelStorage` | 已实现 reference | 无 Warp 基础导入、Torch storage 的 uniform/segmented put、defrag、扩容和 Batch 边界 | 完整上游数据管线、pin-memory 在无设备进程中的 CUDA/HIP 用例、生产多卡数据路径 | [`g1-data-import-audit.md`](../reports/g1-data-import-audit.md)、[`g1-storage-contract.md`](../reports/g1-storage-contract.md) |
| 邻居 framework 接线 | `compute_neighbors`、`NeighborListHook`、中央 `BackendSelection` | 已验证窄范围 | Torch reference 的 periodic/no-PBC、full/half、MATRIX/COO；显式 HIP 的 periodic/Batch/full-list/MATRIX+COO、当前晶胞完整重建、M5批量metadata/M6查询协作两精度变胞性能；M7真实1029/2030原子FP64完整step约2.04–2.37倍；真实MPA-0变胞短步/预算补位；补测FIRE2固定三样本两精度完整联合收敛 | HIP no-PBC/half、skin/rebuild、target/pair、旧 plan 变胞复用、任意材料/大Batch收敛保证、`auto`、DomainParallel、完整上游邻居接口 | [`STATUS.md`](STATUS.md)、[当前晶胞M1](../reports/g2-hip-current-cell-neighbor-m1.md)、[COO/MACE M2](../reports/g2-hip-coo-mace-snapshot-m2.md)、[变胞工作流M3](../reports/g2-hip-variable-cell-workflow-m3.md)、[批量metadata M5](../reports/g2-hip-batch-metadata-m5.md)、[查询协作 M6](../reports/g2-hip-cooperative-query-m6.md)、[真实Batch M7](../reports/g2-hip-real-batch-validation-m7.md)、[完整弛豫补测](../reports/g2-hip-fire2-convergence-2026-10-09.md) |
| 模型包装与 MACE | `MACEWrapper`、真实 checkpoint 推理 | 已验证窄范围 | 真实 MACE checkpoint、异构 Batch、邻居→模型→能量/力短链；固定晶胞 FIRE2 批量弛豫 | 全模型族、训练/微调、长轨迹和生产 MACE 后端 | [`g1-mace-wrapper-batch.md`](../reports/g1-mace-wrapper-batch.md)、[`g2-mace-dynamics-reference.md`](../reports/g2-mace-dynamics-reference.md)、[`g2-mace-fire2-batch32-tier1.md`](../reports/g2-mace-fire2-batch32-tier1.md) |
| MACE-MPA-0 单点 | 现有 `MACEWrapper` + Torch reference 邻居 | 已验证窄 CPU/HCU4 | M1 CPU Batch/ASE/FP64数值导数；M5两精度同几何完整E/F/stress与邻居对照、48份导出核验 | 全目录/完整模型族、默认dtype须匹配；单点一致不代表弛豫精度可互换 | [`g2-mace-mpa0-m1-cpu-baseline.md`](../reports/g2-mace-mpa0-m1-cpu-baseline.md)、[`g2-mpa0-relaxation-m5.md`](../reports/g2-mpa0-relaxation-m5.md) |
| 固定晶胞 dynamics | NVE、FIRE、FIRE2、kinetics、公共 wrapper | 已验证窄范围 | Torch reference 的短流程、异构 Batch、真实 MACE 的短 NVE/FIRE/FIRE2；32×92 固定晶胞 FIRE2 Batch 弛豫 | 长轨迹守恒、完整收敛 Hook/DataSink、变胞 stress、生产后端 | [`g2-dynamics-public-boundary.md`](../reports/g2-dynamics-public-boundary.md)、[`g2-mace-dynamics-reference.md`](../reports/g2-mace-dynamics-reference.md)、[`g2-mace-fire2-batch32-tier1.md`](../reports/g2-mace-fire2-batch32-tier1.md) |
| 固定晶胞 Langevin | `NVTLangevin(backend="torch_reference")` | 已验证窄范围 | BAOAB、float32/64、异构普通 Batch、短统计、最小 `state_dict`/`load_state_dict` continuation；当前 CPU `15 passed` | 完整上游 NVT 行为套件、通用 checkpoint、`atom_ptr`/`_out`、inflight、分布式、compile、生产 Triton/HIP | [`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)、[`g2-torch-nvt-langevin-stat.md`](../reports/g2-torch-nvt-langevin-stat.md)、[`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md) |
| FusedStage、Hooks、inflight | 阶段迁移、sampler/sink、observer、短补位 | 已实现 reference 子集 | FusedStage/inflight 状态收缩、sampler/sink、真实 MACE/FIRE2 短补位和 Hook reference 子集 | NVT/Langevin 完整阶段接入、长轨迹 occupancy、完整 checkpoint/restart、DomainParallel ownership | [`g2-upstream-inflight-reference.md`](../reports/g2-upstream-inflight-reference.md)、[`g2-mace-fire2-inflight-reference.md`](../reports/g2-mace-fire2-inflight-reference.md) |
| 固定晶胞 NHC | `NVTNoseHoover(backend="torch_reference")` | 已验证窄 reference | 异构 Batch、FP32/64、C1–8、Yoshida3/5、scalar refill、FusedStage/Hook、continuation、HCU4短 MACE | Warp/NVIDIA、vector arrivals、完整 checkpoint、DomainParallel、生产后端 | [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md) |
| stress/cell-force 基础 | strain/stress 工具、`stress_to_cell_force` wrapper | 已验证窄 reference | CPU 小模型 stress loss 反向到参数；CPU/HCU4 周期非正交 MACE 应力/晶胞力、FP64 有限差分；保留旧 API/import | 不推进晶胞；完整 MLIP 训练、Warp、HCU 二阶/compile 未验收 | [`g2-torch-cell-stress-force.md`](../reports/g2-torch-cell-stress-force.md) |
| 通用晶胞工具 | 原生 align、六项 pack/unpack、extend | 已验证窄 reference | M2 primitive CPU/HCU；M3 Hook 预处理的周期/active 子集和 F/v/stress 协变、MPA-0 CPU旋转对照 | Warp、compile/梯度、运行中重对齐 | [`g2-cell-tools-m2.md`](../reports/g2-cell-tools-m2.md)、[`g2-variable-cell-fire-m3.md`](../reports/g2-variable-cell-fire-m3.md) |
| 变胞 FIRE/FIRE2 | `FIRE2VariableCell` 六项、fork `FIREVariableCell` 完整矩阵 | 已验证窄 reference | M3单步/解析势；M4生命周期；M5真实MPA-0 HCU4两精度单体系/3结构补位/8结构容量4；后续HIP邻居M3两精度五步/预算补位；补测HIP FIRE2三样本两精度完整联合收敛 | FIRE2 Li(CN)2预算退出、FIRE精度敏感；混合变胞阶段、checkpoint/DD/生产后端、完整HCU gate；HIP完整收敛仅固定三样本，未扩大其他材料/大Batch保证 | [`g2-variable-cell-fire-m3.md`](../reports/g2-variable-cell-fire-m3.md)、[`g2-mpa0-relaxation-m4.md`](../reports/g2-mpa0-relaxation-m4.md)、[`g2-mpa0-relaxation-m5.md`](../reports/g2-mpa0-relaxation-m5.md)、[HIP邻居M3](../reports/g2-hip-variable-cell-workflow-m3.md)、[完整弛豫补测](../reports/g2-hip-fire2-convergence-2026-10-09.md) |
| NPT/NPH | barostat dynamics | 计划/未接入 | 保留原接口 | 原生变胞优化器完成不代表 barostat 接入 | [`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md) |
| ASE-compatible BFGS Framework | 公共ASECompatibleBFGS及VariableCell，固定/UnitCellFilter 3N+9异构Batch、独立ID/共享模型/冻结/重排/退出/补位/配对restart | M1–M4及M5-4目标门已通过，已纳入本地develop；本次未推送 | 最终审阅CPU669/5skip；HCU Framework24/native7，既有rocSOLVER39；11体系10步ASE及真实工作流见报告 | 显式新HIP ID仅当前FP64/eager gfx936构建；Torch库host路径、其他Filter/NVIDIA/DomainParallel/全目录生产仍未验收 | [M5-4报告](../reports/g2-ase-bfgs-m5-4.md)、[本地交付](ASE_BFGS_DELIVERY_PLAN.md)、[产品合同](HIP_BFGS_JACOBI_STREAMS_CONTRACT.md)、[Batch合同](TORCH_ASE_BFGS_BATCH_CONTRACT.md) |
| 分布式/DomainParallel | 多卡通信、单大体系域分解 | 计划/未接入 | 仅有局部通信或上游 reference 审计，不构成应用能力 | 单大体系 ownership、halo/迁移、长程和完整 DomainParallel | [`g2-upstream-inflight-reference.md`](../reports/g2-upstream-inflight-reference.md)、[`STATUS.md`](STATUS.md) |

## 2. Ops 与后端模块

| Ops/后端模块 | 当前状态 | 已验证范围 | 尚未覆盖或准入限制 | 主要证据 |
|---|---|---|---|---|
| Dense neighbor Torch reference | 已验证窄范围 | no-PBC/periodic、full/half、MATRIX/COO、Batch 边界、image shift、距离/向量连续梯度 | 不是完整上游生产邻居后端；target/pair、compile/opcheck、DomainParallel 未完成 | [`g2-torch-reference-pbc-cell-list-core.md`](../reports/g2-torch-reference-pbc-cell-list-core.md) |
| Torch `cell_list` reference | 已验证窄范围 | periodic/no-PBC、full/half、mixed/triclinic Batch、build/query 分层、容量和 selective rebuild | 主要作为 correctness oracle；不进入 `auto`，不代表生产性能 | [`g2-torch-reference-pbc-cell-list-core.md`](../reports/g2-torch-reference-pbc-cell-list-core.md) |
| `hip.neighbor.cell_list-v1` | 已验证窄范围 | gfx936、periodic、Batch、full-list、MATRIX+COO（Torch打包）、调用间当前晶胞完整重建、FP32/FP64、`skin=0`、自动 doubling capacity；M5批量metadata/M6查询协作两精度变胞性能；M7真实1029/2030原子FP64完整step约2.04–2.37倍；M8比较层修正后八组数值/状态比较通过；M9纠正过严的逐结构晶胞移动probe断言，两组FP32 FIRE2复跑通过 | FP32 cutoff舍入边界；不支持 no-PBC、half、skin/rebuild、target/pair、旧 plan 变胞复用、native geometry backward、compile/opcheck、`auto` | [`g2-framework-hip-neighbor-compatibility-gate.md`](../reports/g2-framework-hip-neighbor-compatibility-gate.md)、[当前晶胞M1](../reports/g2-hip-current-cell-neighbor-m1.md)、[COO/MACE M2](../reports/g2-hip-coo-mace-snapshot-m2.md)、[批量metadata M5](../reports/g2-hip-batch-metadata-m5.md)、[查询协作 M6](../reports/g2-hip-cooperative-query-m6.md)、[真实Batch M7](../reports/g2-hip-real-batch-validation-m7.md)、[比较分层 M8](../reports/g2-hip-mpa0-state-validation-m8.md)、[移动断言复核 M9](../reports/g2-hip-fp32-fire2-cell-movement-m9.md) |
| Neighbor geometry | Torch 为训练/梯度路径；native HIP 为 forward-only 候选 | Torch distance/vector 一阶/二阶路径；native HIP forward parity | native HIP backward、完整训练力损失和 compile/opcheck 未完成 | [`g2-batch-query-native-hip-geometry.md`](../reports/g2-batch-query-native-hip-geometry.md)、[`STATUS.md`](STATUS.md) |
| LJ energy/force | 已实现 reference | no-PBC/PBC 的窄 full/half、energy/force 和物理对照 | switching、完整 virial/stress、生产 HIP/Triton 未完成 | [`g1-neighbor-lj-reference.md`](../reports/g1-neighbor-lj-reference.md)、[`g1-pbc-lj-nve-reference.md`](../reports/g1-pbc-lj-nve-reference.md) |
| VV、FIRE、FIRE2、kinetics、periodic/segment helpers | 已验证窄 reference | fixed-cell Torch dispatcher/executor、公共 wrapper 和部分 HCU reference | 生产 Triton/HIP、完整 compile、NPT/NPH 未完成；变胞 FIRE 证据另列 | [`g2-dynamics-reference-velocity-verlet.md`](../reports/g2-dynamics-reference-velocity-verlet.md)、[`g2-dynamics-reference-fire.md`](../reports/g2-dynamics-reference-fire.md)、[`g2-dynamics-reference-kinetics.md`](../reports/g2-dynamics-reference-kinetics.md) |
| Langevin BAOAB | 已验证窄 reference | Torch reference、公共显式 backend、统计和最小 continuation state | 尚无生产 Triton/HIP；完整 NVT/Dynamics capability 仍未完成 | [`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)、[`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md) |
| NHC chain/kick/drift | 已验证窄 reference | Torch functional与原地ABI，12组HCU parity，CPU可逆性/扩展能量，链ABI opcheck/eager compile | MD forward-only；functional v梯度限CPU；Warp、生产HIP/Triton未验收 | [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md) |
| `stress_to_cell_force` | 已验证窄 reference | FP32/64、完整/下三角投影、strided/empty、CPU 一二阶/compile/opcheck、HCU4 一阶/stream；UBF-009 公式修正 | Warp/NVIDIA、HCU 二阶/compile、生产 HIP/Triton 未验收；无晶胞更新 | [`g2-torch-cell-stress-force.md`](../reports/g2-torch-cell-stress-force.md) |
| ASE-compatible BFGS Ops | M1–M4及M5-4已实现并通过目标数值门，已纳入本地develop | FP64固定/UnitCell独立state，prepare/step/batch_step；按DOF batched数学与全组原子提交；新显式hip.bfgs-ase-jacobi-streams-v1，单宽batched/多宽双stream，极小非零谱修正 | 无147/285或B≤4门；稠密H/scratch和整数/内存容量约束；默认/hip不变，不进auto。最终32矩阵/9真实组、1113/B32子集/真实561 B8通过；性能按输入分化，其他构建/NVIDIA/compile/梯度未验收；已纳入本地develop；本次未推送 | [M5-4报告](../reports/g2-ase-bfgs-m5-4.md)、[本地交付](ASE_BFGS_DELIVERY_PLAN.md)、[产品合同](HIP_BFGS_JACOBI_STREAMS_CONTRACT.md) |
| NPT/NPH Ops | 计划/未接入 | 当前没有可发布的 DCU barostat capability | 独立合同/CPU oracle；基础转换不代表算法可用 | [`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md) |
| `cell_tools` | 已验证窄 reference | ops-owned align/pack/unpack/extend；CPU FP32/64 与 HCU4 六组 primitive 对照；异构/zero atoms/empty | 无优化步；Warp/梯度/compile/生产性能未验收；append/interleaved 索引不能混用 | [`g2-cell-tools-m2.md`](../reports/g2-cell-tools-m2.md) |
| `fire2` variable_cell / `fire_cell` | 已验证窄 reference | 六项和全矩阵原生语义；CPU39项oracle、HCU4功率分支/解析势；M5目标模型小样本应用另验收 | forward-only；实际Warp、生产HIP/Triton、DD/compile/完整HCU gate未验收 | [`g2-variable-cell-fire-m3.md`](../reports/g2-variable-cell-fire-m3.md)、[`g2-mpa0-relaxation-m5.md`](../reports/g2-mpa0-relaxation-m5.md) |

## 3. 总体应用能力

本表总结已有验证，不表示所有工作负载或精度均已支持。[项目示例](../examples/README.md)
提供 Batch 与周期邻居入门，其余应用示例仍在整理；probe 和报告是可追溯证据，
不自动等于面向用户的完整示例。

| 应用 | 已有证据与范围 | 限制与证据 |
|---|---|---|
| 数据、Batch 与邻居 | Torch reference 支持周期/非周期、full/half、MATRIX/COO、异构 Batch；显式 HIP 周期邻居已有指定范围 DCU 验证 | HIP 为 full-PBC/full-list、FP32/64、skin=0、当前晶胞完整重建；[能力门](../reports/g2-framework-hip-neighbor-compatibility-gate.md)、[当前胞](../reports/g2-hip-current-cell-neighbor-m1.md) |
| MACE / MPA-0 单点 | 真实 MACE checkpoint 异构 Batch 能量/力短链；MPA-0 小样本 CPU/HCU 能量、力、应力和邻居对照 | 其他模型族及所有材料未验收；[MACE](../reports/g1-mace-wrapper-batch.md)、[MPA-0](../reports/g2-mpa0-relaxation-m5.md) |
| 固定晶胞 NVE | 真实 MACE 小分子 CPU/HCU 短步；LJ/PBC 有 reference 证据 | 不代表长时间能量守恒、完整重启或吞吐；[MACE dynamics](../reports/g2-mace-dynamics-reference.md)、[LJ NVE](../reports/g1-pbc-lj-nve-reference.md) |
| 固定晶胞 Langevin NVT | BAOAB reference、异构 Batch、短统计和最小积分器续算 | 统计证据为解析谐振子，非真实 MACE 长 NVT；[统计](../reports/g2-torch-nvt-langevin-stat.md)、[续算](../reports/g2-torch-nvt-langevin-restart.md) |
| 固定晶胞 NHC NVT | Torch reference 温控链、异构 Batch、scalar refill、短真实 MACE | 长轨迹、vector arrivals、完整 checkpoint 和分布式未验收；[报告](../reports/g2-torch-nvt-nhc.md) |
| 固定晶胞 FIRE/FIRE2 | 真实 MACE-OFF23 small 的 32×92 原子周期结构 FIRE2 达指定 fmax 门槛；异构短轨迹与补位另有证据 | 不保证任意模型、材料或批量的收敛和吞吐；[32体系](../reports/g2-mace-fire2-batch32-tier1.md)、[补位](../reports/g2-mace-fire2-inflight-reference.md) |
| 变胞 FIRE/FIRE2 + MPA-0 | 原生变胞生命周期及两精度小样本；HIP 邻居 FIRE2 三种 Li/Na 结构在两精度下均达力和应力联合阈值 | 个别 Li(CN)2 案例历史上未达应力阈值，FIRE 有精度敏感性；非全目录成功保证；[M5](../reports/g2-mpa0-relaxation-m5.md)、[三样本补测](../reports/g2-hip-fire2-convergence-2026-10-09.md) |
| 显式 HIP 邻居真实 Batch 短步 | 1029/2030 原子、两优化器与两精度的状态比较分层验收；M9 复核后两组受影响 FP32 FIRE2 重跑通过 | 仅指定结构和短步，不等于大批量长时收敛或 auto 准入；[M8](../reports/g2-hip-mpa0-state-validation-m8.md)、[M9](../reports/g2-hip-fp32-fire2-cell-movement-m9.md) |
| ASE-compatible BFGS 与 FIRE2 → BFGS | 本地 develop 已包含固定胞/UnitCellFilter、异构 Batch、配对 restart 及显式 HIP；11体系逐步 ASE 与真实 MPA-0 小样本工作流有记录 | 新 HIP 仅指定 FP64/eager/gfx936 构建；广义力门槛不等同物理原子力门槛；[使用说明](ASE_BFGS_USAGE.md)、[集成复验](../reports/partner-reference-stage1-integration-2026-11-06.md)、[历史CSP](../reports/g2-ase-bfgs-delivery-review.md) |
| 完整弛豫 → 恒温 → MD | 已有 FusedStage、部分 Hooks、短补位等片段 | 完整组合、长期 checkpoint 和跨卡流水线未形成端到端验收；[inflight](../reports/g2-upstream-inflight-reference.md) |
| NPT/NPH、长程、多卡与完整训练 | 保留部分上游接口及通信/梯度基础 | 尚未形成相应完整 DCU 应用验收；[当前状态](STATUS.md)、[特性契约](FEATURE_COMPATIBILITY.yaml) |

历史大 Batch 耗时、容量和性能数据以原报告为准：例如 [4000原子测量](../reports/g2-mpa0-batch4000-timing.md)、
[8000原子容量诊断](../reports/g2-mpa0-batch8000-timing.md)、[HIP较大Batch](../reports/g2-hip-real-batch-validation-m7.md)。
它们使用特定模型、输入和判据，不能直接作为生产容量或通用吞吐承诺。

## 4. 如何使用这张表

- 判断“能不能跑应用”时看第 3 节，并同时查看对应证据；不要只看“代码已存在”。
- 判断某个后端能否接入时看第 2 节和 [`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md)。
- 判断一个功能是否可以进入发布承诺时看 [`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)
  的限制、验证状态和 release gate。
- 新增功能后，先更新对应模块行和报告，再由集成者同步 STATUS、Feature Contract 和本表。
