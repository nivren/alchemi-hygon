# 稳定移植基线与上游跟踪流程

启动日期：2026-10-09。用户已确认按 A–D 分阶段实施；每阶段完成后先总结结果，
提出下一阶段具体目标，等待用户确认再启动。此文件规定长期流程；当前执行状态见
[STATUS.md](STATUS.md)，任务队列见 [PARALLEL_DEVELOPMENT_PLAN.md](PARALLEL_DEVELOPMENT_PLAN.md)。

## 1. 稳定基线与能力边界

- 产品上游基线由 [UPSTREAM_LOCK.yaml](UPSTREAM_LOCK.yaml) 的完整 SHA 锁定。
  本轮保持 Framework `4dfe3723def34df3fadb245981081ccf8c94c257`、
  Ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`，生产 Torch/DTK 环境保持原规格。
- 当前集成源码起点为 `develop` 的 `2e211ab93a4f1ad0c8c720eecb22129d846c7792`；
  HIP Neighbor M1–M9 已集成并推送三处远端；A 的指定 CPU/HCU4 补充验收已完成，
  证据见 [阶段报告](../reports/g2-integration-a-2026-10-09.md)。
  已有合并采用用户授权的 fast-forward；不补造历史 PR 或重写已共享提交。
- 完整保留上游源码、公共 API、测试和来源，区分源码保留、后端准入与实际验证。
  能力范围以 [FEATURE_COMPATIBILITY.yaml](FEATURE_COMPATIBILITY.yaml)、报告和运行证据为准。
- 日常功能移植在固定基线上继续；新上游功能进入待评估队列，不自动成为当前 HCU 支持项。

## 2. 上游版本组合状态

| 状态 | 含义 | 必须记录 |
|---|---|---|
| Baseline | 当前产品采用的上游组合 | 两包完整 SHA、本地集成提交、环境、支持范围与证据 |
| Candidate | 正在审查的上游组合 | 两包完整 SHA、官方 URL、观察到的 ref、日期、依赖与阻塞 |
| Validated | 通过指定 gate 的候选组合 | 上述身份、具体 gate/设备/退出码、未验证能力 |

`main`、`0.5.0-rc` 等 ref 只描述来源；冻结后按完整 SHA 分析，单次运行不得随 ref 漂移。
版本字符串或相关 Issue/PR 不证明兼容性或修复等价。Validated 不代表全上游功能已验证；
经用户确认升级范围和验收后，才能提升为新的 Baseline。

## 3. A–D 阶段及确认点

| 阶段 | 工作 | 交付与验收 |
|---|---|---|
| A：集成基线验收 | 固定当前 develop 源码；CPU reference + HIP 定向 CPU/HCU + 真实 MPA-0 回归 | SHA/环境/夹具/命令/退出码报告；限制明确；完成后停在确认点 |
| B：Upstream Delta V1 | 候选记录、最小只读差异工具、首轮报告 | 两仓库精确 SHA、历史关系、API/依赖/特性/本地修改交集；可重复、可复核 |
| C：UBF 对齐 | UBF-001～016 的官方修复及额外本地边界审查 | 每项有证据和 action，无证据写 unknown；形成回归/backport 清单 |
| D：隔离升级评估 | 独立分支/worktree/环境，验证代表性依赖闭包 | 接受升级、部分 backport 或暂缓的决策及阻塞 |

A–C 完成后恢复正常功能开发；D 单独确认范围，可与日常功能开发并行。
HCU4 不可用时不等待、不占用其他卡，记录未运行项；A 的 HCU 验收保持待完成，
不以 CPU 通过替代它。经用户确认可继续 B/C 中不依赖硬件的工作。

## 4. 差异分析及补丁审查

B 阶段计划交付 `scripts/upstream_delta.py`、`docs/UPSTREAM_CANDIDATES.yaml` 和
`reports/upstream-delta-2026-10.md`，当前尚未实施。
工具以明确的 Git 对象为输入，默认不获取网络更新、不执行上游代码、不修改产品源码、
external 或版本锁。上游对象获取放在独立缓存中，显式 fetch 与分析分开记录。
报告至少比较锁定上游→候选上游、锁定上游→本地产品子树，以及两组变更的交集。

- 记录 ancestry、merge-base 和两侧独有提交；存在历史分叉时同时审查基线独有功能。
- 文件增删改/重命名和静态 API 签名、参数、默认值可自动提取；动态 API、返回值和数值语义
  需要人工复核。文件交集属于潜在冲突，不能直接写成已确认合并冲突。
- 依赖按基础、extras、平台 marker、跨包导入和实际 API 使用分别检查。
- 同时盘点新增、删除、弃用的特性，人工决定保留、backport、延期或整版本升级评估。

[UPSTREAM.md](UPSTREAM.md) 保留全部 LP 适配索引；
[UPSTREAM_BUGFIXES.md](UPSTREAM_BUGFIXES.md) 保留科学原因、本地/fork 来源和验证边界。
C 阶段计划新增 `docs/UPSTREAM_BUGFIX_STATUS.yaml` 作为上游对齐状态来源，原因和本地实现
仍由现有 Markdown 记录解释；当前尚未实施。结构化对照维护 local commit、official issue/PR、fix commit、observed branch、
equivalence、action、核验日期、源码路径和 verification，并注明等价判断的具体范围。
核心公式等价不等于整个实现等价。官方修复后仍需检查额外的边界处理、兼容逻辑和测试价值。
移除补丁须验证边界已被覆盖，回归测试和平台适配仍保留。反馈上游需另获发送授权。

## 5. 隔离升级与正式接受

- 日常分支使用 `<开发者>/<类型>-<主题>`；升级实验使用 `sync/upstream-*`；
  `develop` 用于已验证集成，`release/hygon-*` 留作未来发布基线。
- 升级实验从稳定 develop 建立独立 worktree；保留完整候选源码组合后，选择代表性依赖闭包，
  不把任意新版文件复制到旧版目录并当作完整兼容组合。
- 先验证依赖和 API，再进行 CPU/HCU。不得为候选升级生产 Torch/DTK 或原地重建 `.venv`。
  当前环境的依赖/API 阻塞明确记录；其他环境通过不等于当前环境已通过。
- 历史分叉先评估 subtree 三方合并与补丁重放策略；不盲目执行 subtree pull。
- `sync/*` 不成为长期功能主线；实验允许结论为暂不升级。
- 正式升级同时更新版本锁、LP/UBF、特性契约、必要的能力矩阵和指定回归报告。

## 6. 验证、PR 与持续跟踪

| 级别 | 触发 | 指定验证范围 |
|---|---|---|
| L1 | 功能/工具 PR | 相关 CPU 单元测试、API/导入、静态兼容性；工具用小型 Git 夹具验证分叉与变化识别 |
| L2 | HIP 路径或相关上游变更 | L1 + HCU4 数值、reference parity、FP32/64、代表性 Batch |
| L3 | 上游基线升级或版本发布 | 明确列出的完整 L1/L2 gate、真实 MPA-0 FIRE/FIRE2、状态/补位、特性核验和性能回归 |

既有 CPU 入口为 `scripts/check_cpu_reference.sh`；HIP 和真实模型入口按阶段报告列出。
framework/ops pytest 使用独立进程。共享设备先只读检查，每项限时；原始日志保存在
忽略的 artifacts，报告记录具体 SHA、命令、夹具哈希、设备/环境、退出码和已知限制。
Warp 默认选择/导入边界可在当前环境审查；Warp 数值验证需要实际 NVIDIA/Warp 环境。

后续变更通过 PR 审查后集成 develop；每阶段确认不等于自动授权推送/合并。
用户明确授权外部交付后，将相同 develop SHA 同步到指定远端并逐处核验。
每两周开展一次上游变化观察；正式 release、相关 bugfix 或新功能开发前追加定向审查。
每次记录观察日期、冻结 SHA、报告和决策；定期跟踪不表示自动运行任务或自动升级。

本轮不开展全工程重构，不自动合并 NVIDIA main，不混入 BFGS、DomainParallel 或全部新特性移植。
