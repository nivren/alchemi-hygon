# 当前并行开发计划

更新时间：2026-09-20。本文只维护当前可启动的任务、边界和验收方式；此前的完整阶段日志与
Point 记录保存在 [`history/PARALLEL_DEVELOPMENT_PLAN.md`](history/PARALLEL_DEVELOPMENT_PLAN.md)。

## 1. 当前基线

- 工作基线：`develop`，最近已推送的文档重构基线为 `97325a4`。
- `TORCH-NEIGHBOR-PBC-CELL` 阶段一 Torch reference 与阶段二 HIP 窄 capability 已收口。
- `TORCH-NVT-LANGEVIN` 固定晶胞 BAOAB 窄 reference slice 已收口，不再作为当前任务。
- 后续单卡 HCU 验证统一使用 `HIP_VISIBLE_DEVICES=4`；历史报告中的卡号不改写。
- 当前任务应从 `develop` 创建短期个人分支，完成一个可独立验证的 slice 后再合并。

## 2. 当前任务队列

| 顺序 | 任务 | 当前目标 | 暂不扩展 |
|---|---|---|---|
| 1 | 固定晶胞 NHC reference | 建立 Nose-Hoover chain 的 Torch/CPU 正确性基线与公共接口 | NPT/NPH、变胞和生产后端 |
| 2 | `TORCH-CELL-STRESS-FORCE` + coupled FIRE2 | 明确 stress 到 cell-force 的单位、shape、梯度和 FIRE2 变胞语义 | 完整 ASE 变胞兼容、后端优化 |
| 3 | 固定晶胞 ASE-compatible BFGS | 对齐固定晶胞的 `3N` 优化状态、收敛和 checkpoint 语义 | 变胞 `3N+9` 与并行 planner |

每个任务都先建立 Torch reference 和最小回归测试，再接 framework；只有接口、支持状态或
验证证据发生变化时才同步能力契约、STATUS 和报告。未确认 owner 前不自动创建任务分支。

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
- stress/FIRE2 主要修改 `packages/ops` 的 cell/stress、dynamics 和 framework wrapper，
  需要同步单位与梯度契约。
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
