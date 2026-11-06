# 当前开发状态

更新：2026-11-06。此页汇总当前基线、已验证范围与限制；完整应用场景见
[实现状态与应用能力矩阵](IMPLEMENTATION_STATUS_MATRIX.md)，逐特性条件见
[特性契约](FEATURE_COMPATIBILITY.yaml)，原始日期的结果见[验证报告索引](../reports/README.md)。

## 当前基线

- 本地 `develop` 已纳入 ASE-compatible BFGS 的四个原始提交，集成记录为 `42f3f19`；
  此集成当时未推送。具体 refs 状态应通过 Git 实时检查。
- 正式源码位于 `packages/framework` 和 `packages/ops`；版本及上游来源见
  [UPSTREAM_LOCK.yaml](UPSTREAM_LOCK.yaml)。`external/` 为只读上游参考，不是安装源。
- 项目环境入口与外部模型、结构文件说明见[环境指南](DEVELOPMENT_ENVIRONMENT.md)。
  面向具体材料的应用入口另属下游 `alchemi-app`。

## 当前能力摘要

| 功能 | 已有窄范围 | 主要限制 |
| --- | --- | --- |
| 邻居与几何 | Torch reference 支持周期/非周期、full/half、MATRIX/COO、Batch 与连续几何梯度；显式 HIP 邻居有指定 HCU 证据 | HIP 仅 full-PBC/full-list、FP32/64、skin=0、当前胞完整重建；COO 用设备内 Torch 打包，不进入 auto；无 no-PBC/half、旧 plan 变胞复用或完整 DomainParallel |
| 真实模型与固定胞 dynamics | MACE/MPA-0 单点、NVE/FIRE/FIRE2 短流程、Langevin 与 NHC reference 有指定 CPU/HCU 证据 | 各模型、dtype 和输入需单独验证；长真实势轨迹、完整 checkpoint 及生产吞吐尚未普遍验收 |
| 变胞弛豫 | stress/cell-force、晶胞工具和 FIRE/FIRE2 工作流有窄验证；HIP 邻居 FIRE2 的三种 Li/Na 结构在指定条件下联合收敛 | 不代表任意材料全目录收敛或 NPT/NPH；个别样本预算退出、精度敏感性需参见报告 |
| ASE-compatible BFGS | 固定胞/UnitCellFilter 3N+9、异构 Batch、按 ID 状态及配对重启；显式 HIP Jacobi 在指定构建通过数值门 | 指定 FP64/eager/gfx936 构建；广义力阈值不同于物理原子力；其他设备、compile、梯度、多卡和生产规模未验收 |
| 分布式、长程与训练 | 保留上游接口、部分通信与梯度基础证据 | 单大体系 DomainParallel、NPT/NPH、长程和真实 MLIP 完整训练流程尚无完整 DCU 应用验收 |

支持范围按操作、设备、dtype、输入和梯度分别判定。Torch reference 在 HCU 运行不等于
优化内核或生产性能通过。具体边界见[后端能力矩阵](BACKEND_CAPABILITY_MATRIX.md)。

## 最近集成与验证

2026-11-06 本地 `develop` 集成 BFGS 四个提交；使用 `e3605c1` 源码的独立候选复验：
CPU reference 669 passed / 5 skipped；HCU4 六项基础 smoke、BFGS Ops 18、Framework 24、
原 rocSOLVER 39 和11体系十步 ASE 比较通过，重启逐位一致。CPU 跳过含三项缺 Warp
对照、两项 HCU 参数；后两项已在 HCU 门通过。

本轮未重新运行历史真实模型完整 CSP、wheel/sdist 或性能套件；其范围和结果仍以原报告为准。
详见[本地集成报告](../reports/partner-reference-stage1-integration-2026-11-06.md)、
[BFGS 最终审阅](../reports/g2-ase-bfgs-delivery-review.md)和
[HIP FIRE2 三样本补测](../reports/g2-hip-fire2-convergence-2026-10-09.md)。

## 当前下一步

项目正在进行参考版本的文档与示例收口。具体工作顺序、暂停项和协作方式见
[当前任务与开发计划](PARALLEL_DEVELOPMENT_PLAN.md)。先以现有项目文档组织入口，
再逐项准备和验证应用示例；范围、目标和结果按里程碑向用户说明，获得明确确认后进入下一项。
示例可依赖外部模型与结构文件，项目环境指南记录 dcu2 已核对的两类 CIF 入口。
NPT/NPH、DomainParallel、生产 Triton/HIP dynamics 等仍需单独规划与验收。

## 验证和设备约定

- CPU 基线：`scripts/check_cpu_reference.sh`；framework 与 ops 测试分进程运行。
- 本机单卡 HCU 历史约定使用 `HIP_VISIBLE_DEVICES=4`；每次运行前仍需确认资源和设备可见性。
- 分别记录代码版本、模型/输入、设备、dtype、命令、退出码、数值结果与未验证范围。
- 性能区分冷启动/JIT、预热、设备执行和端到端，不能从单一窄输入推断普遍收益。

## 文档权威关系

- 架构和后端边界：[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。
- 应用总览：[IMPLEMENTATION_STATUS_MATRIX.md](IMPLEMENTATION_STATUS_MATRIX.md)。
- 操作能力与契约：[BACKEND_CAPABILITY_MATRIX.md](BACKEND_CAPABILITY_MATRIX.md)、
  [FEATURE_COMPATIBILITY.yaml](FEATURE_COMPATIBILITY.yaml)。
- 任务和上游追踪：[PARALLEL_DEVELOPMENT_PLAN.md](PARALLEL_DEVELOPMENT_PLAN.md)、
  [UPSTREAM_LOCK.yaml](UPSTREAM_LOCK.yaml)、[UPSTREAM_SYNC.md](UPSTREAM_SYNC.md)。
- 完整历史：[整理前状态快照](history/STATUS-snapshot-2026-11-06.md)及
  [更早状态历史](history/STATUS.md)；历史记录中的分支、授权和支持状态只表示当时。
