# 当前并行开发计划

更新时间：2026-10-09。本文只维护当前可启动的任务、边界和验收方式；此前的完整阶段日志与
Point 记录保存在 [`history/PARALLEL_DEVELOPMENT_PLAN.md`](history/PARALLEL_DEVELOPMENT_PLAN.md)。

## 1. 当前基线

- 本轮 NHC 起点 `7a90fcf`，开发分支 `codex/feat-fixed-cell-nhc`；实现提交 `b4050ec`，已纳入并同步到三个 remote 的 `develop`。
- `TORCH-NEIGHBOR-PBC-CELL` 阶段一 Torch reference 与阶段二 HIP 窄 capability 已收口。
- `TORCH-NVT-LANGEVIN` 固定晶胞 BAOAB 窄 reference slice 已收口，不再作为当前任务。
- `TORCH-CELL-STRESS-FORCE` 基础 slice 位于 `codex/feat-cell-stress-force`，起点
  `e6dd99c`，实现与测试提交 `a76107e`、文档 `35ea133`；已包含于 merge `620a485` 并同步至三个目标 remote。该 slice 排除晶胞及 FIRE/FIRE2。
- MPA-0 + 原生 FIRE/FIRE2 的里程碑1已完成：`codex/test-mace-mpa0-baseline`，起点 `35ea133`。
  [契约](TORCH_MACE_MPA0_RELAXATION_CONTRACT.md)、[CPU 报告](../reports/g2-mace-mpa0-m1-cpu-baseline.md)。
  里程碑2已在 `codex/feat-cell-tools-reference` 完成晶胞工具 CPU/HCU4 窄验证，
  [M2 报告](../reports/g2-cell-tools-m2.md)。里程碑3已在 `codex/feat-variable-cell-fire`
  完成原生六项 FIRE2、fork 完整 FIRE、Hook 预处理和窄 CPU/HCU 验证（实现 `29a70f3`），
  [M3 报告](../reports/g2-variable-cell-fire-m3.md)。里程碑4已在
  `codex/feat-mpa0-relaxation-workflow`（实现 `7a8af50`）完成普通单优化器批量工作流、薄应用与窄
  CPU/合成 HCU 验收，[M4 报告](../reports/g2-mpa0-relaxation-m4.md)。里程碑5已在
  `codex/test-mpa0-hcu-validation` 完成真实 HCU4 窄验收，[M5 报告](../reports/g2-mpa0-relaxation-m5.md)；
  用户确认更大 Batch 测量后，M5收口 `4656cce`；`codex/test-mpa0-batch-timing`
  已完成一个64结构/2287原子的FP64原生FIRE Batch，应用101秒，62收敛/2预算退出，
  [报告](../reports/g2-mpa0-single-batch-timing.md)，收口 `7410445`。只条件估算1600，未跑全量。
  8000诊断分支 `codex/test-mpa0-batch8000-timing` 起点7410445：同一前缀231结构/7937原子，
  初始模型求值在80%/95%/默认100%上限均OOM，无优化更新或弛豫输出。
  [诊断报告](../reports/g2-mpa0-batch8000-timing.md)，收口 `eff753d`；分项观测仅改探针。
  4000测量 `codex/test-mpa0-batch4000-timing` 起点eff753d，收口1dba058：114项/3959原子完整FP64 Batch，
  111收敛/3预算退出，应用160.145秒，模型75.93%/Torch reference邻居19.34%/优化器2.81%。
  [4000报告](../reports/g2-mpa0-batch4000-timing.md)；测量阶段无源码改动。
  当前 `codex/fix-fire-zero-dt-min` 已提交零下限兼容修正 `0e354f4`（LOCAL-004），
  CPU回归和HCU4 FP64短步通过，[报告](../reports/g2-fire-zero-dt-min.md)。
  M1–M5收口审查已完成，[审查报告](../reports/g2-relaxation-port-closure-review.md)；toolkit 边界提交 `730a848` 已由 merge `620a485` 合入并同步，应用工程首提交 `9ae235c` 已同步到三个目标 remote 的 `develop`。
  后续每个里程碑实施前先说明目的与方案，用户确认后实施；完成后先总结并给出
  下一阶段方案，等待用户确认后再收口提交当前阶段、启动下一阶段（用户2026-10-05
  明确的顺序）。不自动推送。M4 已有提交保留，此流程更新在 M5 启动前收口。
- 后续单卡 HCU 验证统一使用 `HIP_VISIBLE_DEVICES=4`；历史报告中的卡号不改写。
- 当前任务应从 `develop` 创建短期个人分支，完成一个可独立验证的 slice 后再合并。

## 2. 当前任务队列

用户已确认 G2 集成收口与 Upstream Sync V1，长期执行规则见
[UPSTREAM_SYNC.md](UPSTREAM_SYNC.md)。A 已固定 `2e211ab` 完成指定 CPU/HCU4 集成基线验收，
真实 MPA-0 八组通过；[报告](../reports/g2-integration-a-2026-10-09.md)和流程文档已获确认并收口提交。
B 已完成只读差异工具/冻结候选记录/[首轮报告](../reports/upstream-delta-2026-10.md)，
14 项 Git/AST CPU 夹具验证通过，用户确认后以 `84b4ba9` 签名收口。
C 已完成 [16 项 UBF 对照](UPSTREAM_BUGFIX_STATUS.yaml)与
[审查/回归报告](../reports/upstream-bugfix-alignment-2026-10.md)，全部 retain；用户已确认 C 收口并授权 D 隔离升级评估。
A–C 完成后恢复正常功能开发；D 可独立并行，不阻塞日常移植。

| 顺序 | 任务 | 当前目标 | 暂不扩展 |
|---|---|---|---|
| 已确认并收口 | A：G2 集成基线验收 | 固定 `2e211ab`，CPU reference 425 passed/3 skipped；HCU ops101/framework77+既有xfail；真实MPA-0八组通过；流程/证据已写入 | 指定 gate，非全上游/NVIDIA/长收敛；未推送 |
| 已确认并收口 | B：Upstream Delta V1 | 工具/候选/报告；14 项夹具验证通过；467/322 文件变化、40/8 潜在交集；候选运行未验证 | 不修改 packages/锁/生产环境，不自动解决冲突 |
| 已确认并收口 | C：UBF 上游对齐 | 16 项官方对照；核心 equivalent 3/partial 5/different 1/unknown 7；全部 retain；PR194 案例 6 passed，本地定向回归与原 MACE 收集失败分别记录 | 未运行候选/HCU/Warp；无补丁删除、产品同步或环境升级 |
| 已确认收口 | D：隔离升级评估 | 依赖/API/三方文本冲突与三原样模块CPU/HCU评估；暂缓整体升级；报告在独立sync worktree | 不作为长期功能主线；候选仍Candidate；与本补测一起整理后续PR |
| 已确认收口 | HIP FIRE2 完整弛豫补测 | HCU4三Li/Na CIF、两精度×两邻居分支；12个结构运行全部联合收敛、真实补位/终态/导出核验；CPU59+探针4通过 | 仅三样本FIRE2；不扩展大Batch/其他材料成功率、性能或候选版本；见补测报告 |
| 完成 | HIP 邻居 M1：当前晶胞 | 显式 full-PBC/Batch/full/MATRIX/FP32/64，每次完整重建；CPU/HCU4 gate；实现 `fecbc39`、分支 `codex/feat-hip-current-cell-neighbors` | 不含 COO/MACE/FIRE 工作流、skin、旧 plan 复用或性能结论 |
| 完成 | HIP 邻居 M2：COO + MACE 单点 | 实现 `3e394d7`；设备内 Torch 适配、ops/framework/Hook CPU/HCU4；真实MPA-0两精度/三快照E/F/stress与FP64差分 | 不直接实现 native COO；变胞 FIRE/FIRE2 和实测优化分别后续验收 |
| 完成 | HIP 邻居 M3：变胞 FIRE/FIRE2 | CPU16+48/HCU4 16测试；真实MPA-0两精度/两优化器五步和三结构预算补位；无新增产品kernel；收口实现 `6b1680a` | 长作业、大Batch、物理收敛、性能准入未新增 |
| 完成实验并收口 | HIP 邻居 M4：性能基线与workspace容量复用 | 基线/trace/候选30测试；配对目标下降1.29%，未达5%门槛，撤回产品缓存；保留重放benchmark | 未新增workspace/性能/auto准入 |
| 完成并收口 | HIP 邻居 M5：批量metadata | CPU ops45/framework21、HCU ops73/framework49；两精度约20,000原子六变胞/补位快照三路径比较，目标下降87.65%；真实模型短轨迹/补位回归 | 实现 `e2d569c`；FP32 cutoff舍入分歧单列，query/native COO/skin/auto未新增 |
| 已收口 | HIP 邻居 M6：查询协作 | CPU ops51/framework21、HCU ops85/framework49；两精度变胞三路径/真实模型/补位通过，few FP64下降11.34%，候选保留 | fe18147/c2a8339；非全库/长收敛/较大真实MLIP吞吐，auto等未新增 |
| 已收口 | HIP 邻居 M7：综合验收 | FP64真实1029/39、2030/92，两优化器step约2.04–2.37倍；短步/补位通过；CPU ops51/framework21、HCU ops85/framework49 | FP32仅1029 FIRE全组通过，其余严格forces门失败；纯Torch对照也失败，未改容差；提交 `00637c0` |
| 已收口 | HIP 邻居 M8：验收分层与复验 | CPU比较器单测3 passed；修正E/F/stress重复状态比较，framework/ops与容差未改 | `04f80bd`；首次八组复验及边界见M8报告 |
| 已收口并集成 | HIP 邻居 M9：FP32 FIRE2逐结构移动断言复核 | `fa1259b`/`b549558` 已随提交链合入 `develop`；未移动项为同一Li3Ge3(MoO6)2；两受影响组复跑通过 | 仅probe/文档；逐结构backend/state与物理容差保留；见M9报告 |
| 完成 | MPA-0 M2：晶胞工具 | align_cell、pack/unpack、扩展索引/指针；CPU/HCU4 primitive 通过 | 未含优化器、ASE filters、NPT/NPH |
| 完成 | MPA-0 M3：变胞 FIRE2/FIRE | 单步/公共解析势 CPU/HCU；MPA-0 CPU 三步与旋转协变 | 不代表目标 HCU、完整收敛/补位、新参数化、生产后端 |
| 完成 | MPA-0 M4：批量应用 | 邻居变胞、联合收敛、退出/补位、薄 CIF/JSONL；CPU真实 FP64 收敛与 HCU合成工作流 | 通用完整 checkpoint、流式服务/向量 arrivals、全目录长作业；目标 HCU由M5补充 |
| 完成 | MPA-0 M5：目标 HCU | 收口4656cce；两精度/两优化器单体系、3结构补位、8结构容量4；48份输出独立CPU/HCU核验 | FIRE2 Li(CN)2预算退出、FIRE精度敏感；非完整HCU gate/全目录/生产性能 |
| 完成 | FP64单个较大Batch | 收口7410445；64结构/2287原子、原生FIRE raw力判据；应用101秒，条件估算1600约41–50分钟 | 没有1600实测/16000原子/持续补位吞吐；与M5及H100判据不同 |
| 完成 | 8000门槛与分项 | 收口eff753d；231结构/7937原子，默认上限初始e3nn前向OOM | 无完整模型求值/优化器更新/弛豫结果；分项不能作稳态或8000吞吐 |
| 完成 | 4000门槛单Batch分项 | 收口1dba058；114结构/3959原子，111收敛/3预算，应用160秒；模型118.8/邻居30.3/优化器4.4秒，独立NumPy/文件核验 | 邻居为Torch reference；原生raw力判据不同于M5/H100，无持续补位/全量1600实测 |
| 完成 | FIRE参数兼容修正 | LOCAL-004提交0e354f4恢复dt_min=0，CPU回归和HCU4 FP64 NumPy/解析势/三CIF十步通过 | 本地移植bug，非上游bug；不代表零下限完整Batch收敛/性能 |
| 完成并集成 | 弛豫移植收口审查 | M1–M5、运行入口、实际参数、HCU/CPU证据边界和未覆盖项已核对；merge `620a485` | 应用参数/H100 停止度量由下游工程负责 |
| 完成 | MPA-0 应用分离 | CIF 应用入口及应用专属测试移入同级独立工程 `alchemi-app`；toolkit 保留原生 API、通用生命周期测试和迁移证据 | 应用参数/H100 停止度量属于下游应用，不加入 toolkit 公共 API；全目录作业另行安排 |
| 后续队列 | 固定晶胞 ASE-compatible BFGS | 对齐固定晶胞的 `3N` 优化状态、收敛和 checkpoint 语义 | 变胞 `3N+9` 与并行 planner |

每个任务都先建立 Torch reference 和最小回归测试，再接 framework；只有接口、支持状态或
验证证据发生变化时才同步能力契约、STATUS 和报告。未确认 owner 前不自动创建任务分支。

MPA-0 应用的后续 CLI/停止度量/性能工作属于 `alchemi-app`，不列入 toolkit 上游移植队列。

MPA-0 应用的后续 CLI/停止度量/性能工作属于 `alchemi-app`，不列入 toolkit 上游移植队列。

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
- 已确认的 HIP M1 当前晶胞完整重建以外的扩展，须按里程碑确认；当前 M2 COO/MACE、M3变胞工作流已收口；M4实验完成但workspace候选已拒绝，M4实现 `95f2ef6` 收口，M5实现及窄验收已收口 `e2d569c`，M6查询协作完成窄验收并收口，M7签名收口 `00637c0`，M8签名收口 `04f80bd`；M9 probe条件修正及两组复验已以 `fa1259b`/`b549558` 收口并合入 `develop`；
  no-PBC、half、skin/rebuild、target/pair、旧 plan 变胞复用、native geometry backward、
  compile/opcheck 或 `auto` 继续暂停；M8/M9测试证据见对应报告。
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
