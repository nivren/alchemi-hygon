# 实现状态与应用能力矩阵

更新时间：2026-09-20。本文是面向开发和方案评审的总览，按 framework、ops/后端和应用能力
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
| 固定晶胞 dynamics | NVE、FIRE、FIRE2、kinetics、公共 wrapper | 已验证窄范围 | Torch reference 的短流程、异构 Batch、真实 MACE 的短 NVE/FIRE/FIRE2；32×92 固定晶胞 FIRE2 Batch 弛豫 | 长轨迹守恒、完整收敛 Hook/DataSink、变胞 stress、生产后端 | [`g2-dynamics-public-boundary.md`](../reports/g2-dynamics-public-boundary.md)、[`g2-mace-dynamics-reference.md`](../reports/g2-mace-dynamics-reference.md)、[`g2-mace-fire2-batch32-tier1.md`](../reports/g2-mace-fire2-batch32-tier1.md) |
| 固定晶胞 Langevin | `NVTLangevin(backend="torch_reference")` | 已验证窄范围 | BAOAB、float32/64、异构普通 Batch、短统计、最小 `state_dict`/`load_state_dict` continuation；当前 CPU `15 passed` | 完整上游 NVT 行为套件、通用 checkpoint、`atom_ptr`/`_out`、inflight、分布式、compile、生产 Triton/HIP | [`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)、[`g2-torch-nvt-langevin-stat.md`](../reports/g2-torch-nvt-langevin-stat.md)、[`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md) |
| FusedStage、Hooks、inflight | 阶段迁移、sampler/sink、observer、短补位 | 已实现 reference 子集 | FusedStage/inflight 状态收缩、sampler/sink、真实 MACE/FIRE2 短补位和 Hook reference 子集 | NVT/Langevin 完整阶段接入、长轨迹 occupancy、完整 checkpoint/restart、DomainParallel ownership | [`g2-upstream-inflight-reference.md`](../reports/g2-upstream-inflight-reference.md)、[`g2-mace-fire2-inflight-reference.md`](../reports/g2-mace-fire2-inflight-reference.md) |
| NHC、变胞 dynamics 与 BFGS | Nose-Hoover chain、stress/cell-force、variable-cell FIRE2、ASE BFGS | 计划/未接入 | 当前没有可用于产品承诺的完整实现 | NHC、NPT/NPH、变胞 FIRE2、固定晶胞 ASE-compatible BFGS 均按当前计划排队 | [`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)、[`adr/0008-ase-compatible-bfgs.md`](../adr/0008-ase-compatible-bfgs.md) |
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
| NHC、NPT/NPH、变胞 stress/cell-force、BFGS eigensolver | 计划/未接入 | 当前没有可发布的 DCU ops capability | 等待当前并行计划和独立 contract/CPU oracle | [`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)、[`adr/0008-ase-compatible-bfgs.md`](../adr/0008-ase-compatible-bfgs.md) |

## 3. 总体应用能力

| 应用场景 | 当前结论 | 可以做什么 | 不能据此承诺 | 关键组成 |
|---|---|---|---|---|
| 数据导入、Batch 组织和 CPU/Torch reference 验证 | 可以，窄范围 | 构造 AtomicData/Batch，保持体系边界和索引，运行 CPU/Torch reference 回归 | 完整上游数据管线、生产多卡数据路径 | Data/Batch、Torch storage |
| 周期邻居与 LJ correctness | 可以，reference | 验证周期邻居集合、image shift、MATRIX/COO、LJ energy/force 和短 NVE 对照 | 生产默认邻居后端、完整 virial/stress、全量上游邻居接口 | Torch neighbor/cell-list、LJ、Torch geometry |
| 固定晶胞 MACE 推理 | 可以，窄范围 | 使用真实 MACE checkpoint 做异构 Batch 的短 energy/force 推理 | 全模型族、训练/微调、长轨迹性能 | MACEWrapper、Torch neighbor、reference geometry |
| 固定晶胞 MACE NVE/FIRE/FIRE2 | 可以，窄范围 | 运行短固定晶胞 MD/弛豫；已验证 32×92 MACE/FIRE2 Batch 弛豫 | 变胞、长轨迹守恒、生产邻居/HIP MACE 性能 | MACE、Torch neighbor、VV/FIRE/FIRE2 |
| 固定晶胞 Langevin reference NVT | 可以，窄范围 | 运行 BAOAB reference、短统计和最小 integrator continuation | 完整上游 NVT、通用 checkpoint、长 MACE NVT、生产 Triton/HIP | NVTLangevin、Torch reference |
| 双阶段 MACE 弛豫：FIRE2 + BFGS | 当前不能完整完成 | FIRE2 固定晶胞 reference 已可用 | BFGS 尚未实现；无压 BFGS 与变胞/压力阶段尚未形成完整链路 | FIRE2 已完成，BFGS 排队 |
| 变胞/加压弛豫 | 当前不能承诺 | 仅可使用已有固定晶胞 reference 作为前置验证 | stress→cell-force、variable-cell FIRE2、NPT/NPH 尚未完成 | 当前计划第 2 项 |
| 单大体系多卡、DomainParallel、生产训练 | 当前不能承诺 | 可保留上游接口作为后续适配目标 | ownership、halo/迁移、完整分布式训练与生产后端未完成 | DomainParallel/分布式任务暂停 |

## 4. 如何使用这张表

- 判断“能不能跑应用”时看第 3 节，并同时查看对应证据；不要只看“代码已存在”。
- 判断某个后端能否接入时看第 2 节和 [`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md)。
- 判断一个功能是否可以进入发布承诺时看 [`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)
  的限制、验证状态和 release gate。
- 新增功能后，先更新对应模块行和报告，再由集成者同步 STATUS、Feature Contract 和本表。
