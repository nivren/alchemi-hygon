# 当前并行开发计划

更新时间：2026-10-05。本文只维护当前可启动的任务、边界和验收方式；此前的完整阶段日志与
Point 记录保存在 [`history/PARALLEL_DEVELOPMENT_PLAN.md`](history/PARALLEL_DEVELOPMENT_PLAN.md)。

## 1. 当前基线

- 本轮 NHC 起点 `7a90fcf`，开发分支 `codex/feat-fixed-cell-nhc`；实现提交 `b4050ec`，随本轮交付到本地 `develop`，未推送。
- `TORCH-NEIGHBOR-PBC-CELL` 阶段一 Torch reference 与阶段二 HIP 窄 capability 已收口。
- `TORCH-NVT-LANGEVIN` 固定晶胞 BAOAB 窄 reference slice 已收口，不再作为当前任务。
- `TORCH-CELL-STRESS-FORCE` 基础 slice 位于 `codex/feat-cell-stress-force`，起点
  `e6dd99c`，实现与测试提交 `a76107e`、文档 `35ea133`；未合并/推送。该 slice 排除晶胞及 FIRE/FIRE2。
- MPA-0 + 原生 FIRE/FIRE2 的里程碑1已完成：`codex/test-mace-mpa0-baseline`，起点 `35ea133`。
  [契约](TORCH_MACE_MPA0_RELAXATION_CONTRACT.md)、[CPU 报告](../reports/g2-mace-mpa0-m1-cpu-baseline.md)。
  里程碑2已在 `codex/feat-cell-tools-reference` 完成晶胞工具 CPU/HCU4 窄验证，
  [M2 报告](../reports/g2-cell-tools-m2.md)。里程碑3已在 `codex/feat-variable-cell-fire`
  完成原生六项 FIRE2、fork 完整 FIRE、Hook 预处理和窄 CPU/HCU 验证（实现 `29a70f3`），
  [M3 报告](../reports/g2-variable-cell-fire-m3.md)。里程碑4已在
  `codex/feat-mpa0-relaxation-workflow`（实现 `7a8af50`）完成普通单优化器批量工作流、薄应用与窄
  CPU/合成 HCU 验收，[M4 报告](../reports/g2-mpa0-relaxation-m4.md)；当前等待里程碑5确认。
  后续每个里程碑先说明目的与方案，用户确认后实施。
- 后续单卡 HCU 验证统一使用 `HIP_VISIBLE_DEVICES=4`；历史报告中的卡号不改写。
- 当前任务应从 `develop` 创建短期个人分支，完成一个可独立验证的 slice 后再合并。

## 2. 当前任务队列

| 顺序 | 任务 | 当前目标 | 暂不扩展 |
|---|---|---|---|
| 完成 | MPA-0 M2：晶胞工具 | align_cell、pack/unpack、扩展索引/指针；CPU/HCU4 primitive 通过 | 未含优化器、ASE filters、NPT/NPH |
| 完成 | MPA-0 M3：变胞 FIRE2/FIRE | 单步/公共解析势 CPU/HCU；MPA-0 CPU 三步与旋转协变 | 不代表目标 HCU、完整收敛/补位、新参数化、生产后端 |
| 完成 | MPA-0 M4：批量应用 | 邻居变胞、联合收敛、退出/补位、薄 CIF/JSONL；CPU真实 FP64 收敛与 HCU合成工作流 | 通用完整 checkpoint、流式服务/向量 arrivals、全目录长作业、真实 MPA-0 HCU |
| 1 | MPA-0 M5：目标 HCU 验收 | 单体系→异构小 Batch→扩大批量；FP64 再 FP32，用户确认后启动 | 全目录成功率、生产性能保证 |
| 5 | 固定晶胞 ASE-compatible BFGS | 对齐固定晶胞的 `3N` 优化状态、收敛和 checkpoint 语义 | 变胞 `3N+9` 与并行 planner |

每个任务都先建立 Torch reference 和最小回归测试，再接 framework；只有接口、支持状态或
验证证据发生变化时才同步能力契约、STATUS 和报告。未确认 owner 前不自动创建任务分支。

### Stress/cell-force 已完成基础 reference

转换支持 FP32/64、完整/下三角晶胞力、显式 reference 与单次 selection；已有 strain/stress
工具和小模型 stress loss 混合二阶路径经 CPU 回归验证，真实周期非正交 MACE 经 CPU/HCU4
forward/parity 与 FP64 有限差分验证。证据见 [报告](../reports/g2-torch-cell-stress-force.md)。
该完成项不包含晶胞更新、变胞 FIRE/FIRE2 或完整 MLIP 训练，不自动启动下一项。

### NHC 已完成窄 reference

`TORCH-NVT-NHC` 保留公共接口、Batch/Hook/FusedStage 与 scalar refill，CPU 谐势
统计、continuation、HCU4 FP32/64 parity 和短真实 MACE 已验证；四项修正有源码注释。
证据与边界见 [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md)。
Warp/NVIDIA、vector arrivals、完整 checkpoint、DomainParallel 和生产后端未验收。

### 已完成但不在当前队列：`TORCH-NVT-LANGEVIN`

已完成并合入 `develop` 的固定晶胞 BAOAB Torch reference slice 包括：registry/catalog 和
generic executor binding、公共 `NVTLangevin(backend="torch_reference")` 接线、float32/64、
异构普通 Batch、空输入和显式错误、短谐势统计 oracle，以及 `state_dict()` / `load_state_dict()`
的最小 integrator continuation state。当前 CPU 复核为 `15 passed`；CPU/HCU 证据见
[`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)、
[`g2-torch-nvt-langevin-stat.md`](../reports/g2-torch-nvt-langevin-stat.md) 和
[`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md)。

这不扩大为完整上游 Langevin/NVT：通用 checkpoint、`atom_ptr`/`_out`、inflight refill、
分布式 ownership、完整上游行为套件、跨设备逐位随机一致性、torch.compile 和生产 Triton/HIP
仍是后续能力，不回填本任务。

## 3. 明确暂停项

以下工作不属于当前队列，除非用户重新确认范围：

- M2 planner、NPT/NPH 和 DomainParallel；
- 生产 Triton/HIP dynamics kernel；
- 扩大 HIP neighbor capability（no-PBC、half/COO、skin/rebuild、target/pair、变胞、
  native geometry backward、compile/opcheck 或 `auto`）；
- 仅为追求 benchmark 数字而重写已经有正确性证据的 Torch geometry 路径。

## 4. 并行边界

- NHC 主要修改 `packages/ops` 的 dynamics reference、测试和对应报告。
- stress 基础 reference 与后续 FIRE/FIRE2 晶胞耦合分开验收；后者主要修改
  `packages/ops` 的 cell/dynamics 和 framework wrapper，需要单独的状态与数值契约。
- BFGS 主要修改 dynamics optimizer 与固定晶胞测试；不要顺带引入变胞状态。
- 共享文件 `docs/STATUS.md`、`docs/PROJECT_HANDOFF.md`、`docs/FEATURE_COMPATIBILITY.yaml`
  和能力矩阵由集成者在 slice 完成后统一更新，避免多个分支同时重写当前状态。

## 5. 标准开工和验收流程

1. 从远端同步 `develop`，检查工作树和目标文件 diff。
2. 阅读上游锁定源码、相关 docstring、现有 Torch reference 与测试，先写最小 operation
   contract：shape、dtype、layout、单位、Batch、空输入、mutation/alias、stream、确定性和
   梯度等级。
3. 先实现 CPU/Torch oracle 与 focused test，再接 framework dispatcher/executor。
4. 需要 HCU 时使用 `HIP_VISIBLE_DEVICES=4` 的限时单卡 probe，并区分 compile、device
   execution、数值正确性和性能证据。
5. 完成后运行 `git diff --check`、相关 CPU 测试和可获得的 HCU 测试，记录实现、测试、数值
   结果、性能口径和未验证假设。
6. 一个小功能点一个提交；提交前由集成者复核文档、能力状态和报告索引。

## 6. 暂停条件

遇到上游语义不明、单位/梯度契约未定、缺少独立 oracle、HCU 资源不可用或会扩大公共 API
范围时，先保留窄 slice 并记录阻塞点，不通过静默 fallback、放宽容差或扩大任务范围继续。

相关入口：[`STATUS.md`](STATUS.md)、[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)、
[`BACKEND_PLATFORM_PIPELINE_PLAN.md`](BACKEND_PLATFORM_PIPELINE_PLAN.md)、
[`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)。
