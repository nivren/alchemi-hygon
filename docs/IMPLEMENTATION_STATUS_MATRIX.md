# 实现状态与应用能力矩阵

更新时间：2026-10-04。本文是面向开发和方案评审的总览，按 framework、ops/后端和应用能力
分别列出当前已经完成的范围。它不替代 [`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)
的逐特性契约；状态判断以 [`STATUS.md`](STATUS.md)、源码、测试和报告为准。

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
| 邻居 framework 接线 | `compute_neighbors`、`NeighborListHook`、中央 `BackendSelection` | 已验证窄范围 | Torch reference 的 periodic/no-PBC、full/half、MATRIX/COO；显式 HIP 的 periodic/fixed-cell/Batch/full-list/MATRIX | HIP no-PBC/half/COO、skin/rebuild、target/pair、变胞、`auto`、DomainParallel、完整上游邻居接口 | [`STATUS.md`](STATUS.md)、[`g2-framework-hip-neighbor-compatibility-gate.md`](../reports/g2-framework-hip-neighbor-compatibility-gate.md) |
| 模型包装与 MACE | `MACEWrapper`、真实 checkpoint 推理 | 已验证窄范围 | 真实 MACE checkpoint、异构 Batch、邻居→模型→能量/力短链；固定晶胞 FIRE2 批量弛豫 | 全模型族、训练/微调、长轨迹和生产 MACE 后端 | [`g1-mace-wrapper-batch.md`](../reports/g1-mace-wrapper-batch.md)、[`g2-mace-dynamics-reference.md`](../reports/g2-mace-dynamics-reference.md)、[`g2-mace-fire2-batch32-tier1.md`](../reports/g2-mace-fire2-batch32-tier1.md) |
| MACE-MPA-0 单点 | 现有 `MACEWrapper` + CPU reference 邻居 | 已验证 CPU 小样本 | 真实 CIF `[4,10,4]`、FP32/64 energy/forces/stress；FP64 Batch/ASE 和力/应力/晶胞力有限差分 | 目标 HCU、变胞优化/收敛、全目录；默认 dtype 须与模型/输入一致 | [`g2-mace-mpa0-m1-cpu-baseline.md`](../reports/g2-mace-mpa0-m1-cpu-baseline.md) |
| 固定晶胞 dynamics | NVE、FIRE、FIRE2、kinetics、公共 wrapper | 已验证窄范围 | Torch reference 的短流程、异构 Batch、真实 MACE 的短 NVE/FIRE/FIRE2；32×92 固定晶胞 FIRE2 Batch 弛豫 | 长轨迹守恒、完整收敛 Hook/DataSink、变胞 stress、生产后端 | [`g2-dynamics-public-boundary.md`](../reports/g2-dynamics-public-boundary.md)、[`g2-mace-dynamics-reference.md`](../reports/g2-mace-dynamics-reference.md)、[`g2-mace-fire2-batch32-tier1.md`](../reports/g2-mace-fire2-batch32-tier1.md) |
| 固定晶胞 Langevin | `NVTLangevin(backend="torch_reference")` | 已验证窄范围 | BAOAB、float32/64、异构普通 Batch、短统计、最小 `state_dict`/`load_state_dict` continuation；当前 CPU `15 passed` | 完整上游 NVT 行为套件、通用 checkpoint、`atom_ptr`/`_out`、inflight、分布式、compile、生产 Triton/HIP | [`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)、[`g2-torch-nvt-langevin-stat.md`](../reports/g2-torch-nvt-langevin-stat.md)、[`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md) |
| FusedStage、Hooks、inflight | 阶段迁移、sampler/sink、observer、短补位 | 已实现 reference 子集 | FusedStage/inflight 状态收缩、sampler/sink、真实 MACE/FIRE2 短补位和 Hook reference 子集 | NVT/Langevin 完整阶段接入、长轨迹 occupancy、完整 checkpoint/restart、DomainParallel ownership | [`g2-upstream-inflight-reference.md`](../reports/g2-upstream-inflight-reference.md)、[`g2-mace-fire2-inflight-reference.md`](../reports/g2-mace-fire2-inflight-reference.md) |
| 固定晶胞 NHC | `NVTNoseHoover(backend="torch_reference")` | 已验证窄 reference | 异构 Batch、FP32/64、C1–8、Yoshida3/5、scalar refill、FusedStage/Hook、continuation、HCU4短 MACE | Warp/NVIDIA、vector arrivals、完整 checkpoint、DomainParallel、生产后端 | [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md) |
| stress/cell-force 基础 | strain/stress 工具、`stress_to_cell_force` wrapper | 已验证窄 reference | CPU 小模型 stress loss 反向到参数；CPU/HCU4 周期非正交 MACE 应力/晶胞力、FP64 有限差分；保留旧 API/import | 不推进晶胞；完整 MLIP 训练、Warp、HCU 二阶/compile 未验收 | [`g2-torch-cell-stress-force.md`](../reports/g2-torch-cell-stress-force.md) |
| 通用晶胞工具 | 原生 align、六项 pack/unpack、extend | 已验证窄 reference | M2 primitive CPU/HCU；M3 Hook 预处理的周期/active 子集和 F/v/stress 协变、MPA-0 CPU旋转对照 | Warp、compile/梯度、运行中重对齐 | [`g2-cell-tools-m2.md`](../reports/g2-cell-tools-m2.md)、[`g2-variable-cell-fire-m3.md`](../reports/g2-variable-cell-fire-m3.md) |
| 变胞 FIRE/FIRE2 | `FIRE2VariableCell` 六项、fork `FIREVariableCell` 完整矩阵 | 已验证窄 reference | 独立单步/CPU解析势；HCU4单步/25步；MPA-0 CPU三CIF、FP32/64各3步及最终单点一致性 | 真实收敛、退出补位/邻居生命周期、MPA-0 HCU、checkpoint/DD/生产后端 | [`g2-variable-cell-fire-m3.md`](../reports/g2-variable-cell-fire-m3.md) |
| NPT/NPH 与 BFGS | barostat dynamics、ASE-compatible BFGS | 计划/未接入 | 保留原接口；分别排队 | 原生变胞优化器完成不代表这些算法已接入 | [`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)、[`adr/0008-ase-compatible-bfgs.md`](../adr/0008-ase-compatible-bfgs.md) |
| 分布式/DomainParallel | 多卡通信、单大体系域分解 | 计划/未接入 | 仅有局部通信或上游 reference 审计，不构成应用能力 | 单大体系 ownership、halo/迁移、长程和完整 DomainParallel | [`g2-upstream-inflight-reference.md`](../reports/g2-upstream-inflight-reference.md)、[`STATUS.md`](STATUS.md) |

## 2. Ops 与后端模块

| Ops/后端模块 | 当前状态 | 已验证范围 | 尚未覆盖或准入限制 | 主要证据 |
|---|---|---|---|---|
| Dense neighbor Torch reference | 已验证窄范围 | no-PBC/periodic、full/half、MATRIX/COO、Batch 边界、image shift、距离/向量连续梯度 | 不是完整上游生产邻居后端；target/pair、compile/opcheck、DomainParallel 未完成 | [`g2-torch-reference-pbc-cell-list-core.md`](../reports/g2-torch-reference-pbc-cell-list-core.md) |
| Torch `cell_list` reference | 已验证窄范围 | periodic/no-PBC、full/half、mixed/triclinic Batch、build/query 分层、容量和 selective rebuild | 主要作为 correctness oracle；不进入 `auto`，不代表生产性能 | [`g2-torch-reference-pbc-cell-list-core.md`](../reports/g2-torch-reference-pbc-cell-list-core.md) |
| `hip.neighbor.cell_list-v1` | 已验证窄范围 | gfx936、periodic/fixed-cell、Batch、full-list、MATRIX、FP32/FP64、`skin=0`、自动 doubling capacity | 不支持 no-PBC、half/COO、skin/rebuild、target/pair、变胞、native geometry backward、compile/opcheck、`auto` | [`g2-framework-hip-neighbor-compatibility-gate.md`](../reports/g2-framework-hip-neighbor-compatibility-gate.md) |
| Neighbor geometry | Torch 为训练/梯度路径；native HIP 为 forward-only 候选 | Torch distance/vector 一阶/二阶路径；native HIP forward parity | native HIP backward、完整训练力损失和 compile/opcheck 未完成 | [`g2-batch-query-native-hip-geometry.md`](../reports/g2-batch-query-native-hip-geometry.md)、[`STATUS.md`](STATUS.md) |
| LJ energy/force | 已实现 reference | no-PBC/PBC 的窄 full/half、energy/force 和物理对照 | switching、完整 virial/stress、生产 HIP/Triton 未完成 | [`g1-neighbor-lj-reference.md`](../reports/g1-neighbor-lj-reference.md)、[`g1-pbc-lj-nve-reference.md`](../reports/g1-pbc-lj-nve-reference.md) |
| VV、FIRE、FIRE2、kinetics、periodic/segment helpers | 已验证窄 reference | fixed-cell Torch dispatcher/executor、公共 wrapper 和部分 HCU reference | 生产 Triton/HIP、完整 compile 和变胞路径未完成 | [`g2-dynamics-reference-velocity-verlet.md`](../reports/g2-dynamics-reference-velocity-verlet.md)、[`g2-dynamics-reference-fire.md`](../reports/g2-dynamics-reference-fire.md)、[`g2-dynamics-reference-kinetics.md`](../reports/g2-dynamics-reference-kinetics.md) |
| Langevin BAOAB | 已验证窄 reference | Torch reference、公共显式 backend、统计和最小 continuation state | 尚无生产 Triton/HIP；完整 NVT/Dynamics capability 仍未完成 | [`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)、[`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md) |
| NHC chain/kick/drift | 已验证窄 reference | Torch functional与原地ABI，12组HCU parity，CPU可逆性/扩展能量，链ABI opcheck/eager compile | MD forward-only；functional v梯度限CPU；Warp、生产HIP/Triton未验收 | [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md) |
| `stress_to_cell_force` | 已验证窄 reference | FP32/64、完整/下三角投影、strided/empty、CPU 一二阶/compile/opcheck、HCU4 一阶/stream；UBF-009 公式修正 | Warp/NVIDIA、HCU 二阶/compile、生产 HIP/Triton 未验收；无晶胞更新 | [`g2-torch-cell-stress-force.md`](../reports/g2-torch-cell-stress-force.md) |
| NPT/NPH、BFGS eigensolver | 计划/未接入 | 当前没有可发布的 DCU ops capability | 独立 contract/CPU oracle；基础转换不代表这些算法可用 | [`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)、[`adr/0008-ase-compatible-bfgs.md`](../adr/0008-ase-compatible-bfgs.md) |
| `cell_tools` | 已验证窄 reference | ops-owned align/pack/unpack/extend；CPU FP32/64 与 HCU4 六组 primitive 对照；异构/zero atoms/empty | 无优化步；Warp/梯度/compile/生产性能未验收；append/interleaved 索引不能混用 | [`g2-cell-tools-m2.md`](../reports/g2-cell-tools-m2.md) |
| `fire2` variable_cell / `fire_cell` | 已验证窄 reference | 六项和全矩阵两种原生语义；CPU39项独立oracle，HCU4 FP32/64四功率分支与解析势 | forward-only；实际Warp、目标模型HCU、生产HIP/Triton、DD/compile未验收 | [`g2-variable-cell-fire-m3.md`](../reports/g2-variable-cell-fire-m3.md) |

## 3. 总体应用能力

| 应用场景 | 当前结论 | 可以做什么 | 不能据此承诺 | 关键组成 |
|---|---|---|---|---|
| 数据导入、Batch 组织和 CPU/Torch reference 验证 | 可以，窄范围 | 构造 AtomicData/Batch，保持体系边界和索引，运行 CPU/Torch reference 回归 | 完整上游数据管线、生产多卡数据路径 | Data/Batch、Torch storage |
| 周期邻居与 LJ correctness | 可以，reference | 验证周期邻居集合、image shift、MATRIX/COO、LJ energy/force 和短 NVE 对照 | 生产默认邻居后端、完整 virial/stress、全量上游邻居接口 | Torch neighbor/cell-list、LJ、Torch geometry |
| 固定晶胞 MACE 推理 | 可以，窄范围 | 使用真实 MACE checkpoint 做异构 Batch 的短 energy/force 推理 | 全模型族、训练/微调、长轨迹性能 | MACEWrapper、Torch neighbor、reference geometry |
| 固定晶胞 MACE NVE/FIRE/FIRE2 | 可以，窄范围 | 运行短固定晶胞 MD/弛豫；已验证 32×92 MACE/FIRE2 Batch 弛豫 | 变胞、长轨迹守恒、生产邻居/HIP MACE 性能 | MACE、Torch neighbor、VV/FIRE/FIRE2 |
| 固定晶胞 Langevin reference NVT | 可以，窄范围 | 运行 BAOAB reference、短统计和最小 integrator continuation | 完整上游 NVT、通用 checkpoint、长 MACE NVT、生产 Triton/HIP | NVTLangevin、Torch reference |
| 固定晶胞 NHC reference NVT | 可以，窄范围 | 短异构 MACE、scalar refill/FusedStage reference、最小积分器续算 | 长真实势 NVT、vector arrivals、完整 checkpoint、分布式、生产性能 | NVTNoseHoover、Torch neighbor/model |
| 双阶段 MACE 弛豫：FIRE2 + BFGS | 当前不能完整完成 | FIRE2 固定晶胞 reference 已可用 | BFGS 尚未实现；无压 BFGS 与变胞/压力阶段尚未形成完整链路 | FIRE2 已完成，BFGS 排队 |
| 周期 MACE 应力与晶胞力计算 | 可以，窄 reference | 在给定晶胞上计算 stress 与 cell-force；真实 checkpoint CPU/HCU4 parity、FP64 有限差分 | 完整模型族、真实 MLIP stress 训练、自动晶胞优化 | MACE strain/stress、Torch neighbor/geometry、stress conversion |
| MPA-0 Li/Na 晶体单点 | 可以，CPU 小样本 | 三个真实 CIF 的 E/F/stress、Batch 对照和 FP64 数值导数 | 目标模型 HCU、全目录处理未验证 | 现有 MACEWrapper、Torch neighbor、stress conversion；M1 报告 |
| MPA-0 原生变胞 FIRE/FIRE2 | 可做 CPU 小样本短步验证 | 三个真实CIF异构Batch，两种精度/优化器各3步，更新后E/F/stress重算一致 | 完整弛豫应用、联合收敛/退出补位、目标HCU、全目录成功率 | M3 reference；M4/M5待确认 |
| 有限外压/NPT/NPH | 当前不能承诺 | stress和晶胞基础可供后续算法使用 | 压力接口及barostat/integrator未移植；M3不是加压应用 | 独立后续队列 |
| 单大体系多卡、DomainParallel、生产训练 | 当前不能承诺 | 可保留上游接口作为后续适配目标 | ownership、halo/迁移、完整分布式训练与生产后端未完成 | DomainParallel/分布式任务暂停 |

## 4. 如何使用这张表

- 判断“能不能跑应用”时看第 3 节，并同时查看对应证据；不要只看“代码已存在”。
- 判断某个后端能否接入时看第 2 节和 [`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md)。
- 判断一个功能是否可以进入发布承诺时看 [`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)
  的限制、验证状态和 release gate。
- 新增功能后，先更新对应模块行和报告，再由集成者同步 STATUS、Feature Contract 和本表。
