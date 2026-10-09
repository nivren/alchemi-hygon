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

B 阶段已交付 [只读工具](../scripts/upstream_delta.py)、
[冻结候选记录](UPSTREAM_CANDIDATES.yaml) 和
[首次报告](../reports/upstream-delta-2026-10.md)，工具 Git/AST 夹具验证 14 项通过。
候选仍为 Candidate；B 当时尚未运行候选源码。D 的三原样模块隔离验证见第5节，
没有取得候选整包组合的硬件兼容性证明。
工具以明确的 Git 对象为输入，默认不获取网络更新、不执行上游代码、不修改产品源码、
external 或版本锁。上游对象获取放在独立缓存中，显式 fetch 与分析分开记录。
报告至少比较锁定上游→候选上游、锁定上游→本地产品子树，以及两组变更的交集。

- 记录 ancestry、merge-base 和两侧独有提交；存在历史分叉时同时审查基线独有功能。
- 文件增删改/重命名和静态 API 签名、参数、默认值可自动提取；动态 API、返回值和数值语义
  需要人工复核。文件交集属于潜在冲突，不能直接写成已确认合并冲突。
- 依赖按基础、extras、平台 marker、跨包导入和实际 API 使用分别检查。
- 同时盘点新增、删除、弃用的特性，人工决定保留、backport、延期或整版本升级评估。

### 4.1 获取对象与重复生成

工具使用 Python >=3.11 的标准库；本机用项目 `.venv/bin/python`。输入 `.yaml` 遵循
本工程的 JSON 兼容 YAML 格式，不需要安装 PyYAML 或运行候选包。
锁提供基线 SHA/URL/子树路径；候选记录提供两个目标完整 SHA、本地产品完整 SHA、观察日期和
人工审阅线索。更换候选时新增一份记录，通过 `--candidate` 指定；不覆盖产品版本锁。

Candidate 一旦用于生成报告，其原始字节即冻结。来源 SHA、本地产品 SHA 或审阅输入需要
变化时，创建带日期或 ID 的新文件，通过已有 `--candidate` 指定；不更新历史 Candidate。
`UPSTREAM_CANDIDATES.yaml` 保存 B 阶段快照，SHA256 为
`9ba1fc4e8804d2b0c15d43ce9d22314f5a6911295ab70cccacd4ee27e9520ef7`。
其中 `not_run` 是 B 当时的观察；D 的后续状态保存在
[独立验证记录](../reports/upstream-sync-2026-10-validation.json)及
[D 报告](../reports/upstream-sync-2026-10.md)，不据历史输入判断当前运行状态。

本轮 Git 对象来自 2026-10-09 官方上游审阅，已复制到忽略的独立缓存。
在没有缓存的新 checkout 中先显式获取对象（完整可达历史，不使用 `--depth`）：

```bash
mkdir -p artifacts/upstream-delta-cache
git init --bare artifacts/upstream-delta-cache/framework.git
git -C artifacts/upstream-delta-cache/framework.git fetch --no-tags \
  https://github.com/NVIDIA/nvalchemi-toolkit.git \
  4dfe3723def34df3fadb245981081ccf8c94c257 f59e28dc6701583a13bb294e536076c2a7379e98
git init --bare artifacts/upstream-delta-cache/ops.git
git -C artifacts/upstream-delta-cache/ops.git fetch --no-tags \
  https://github.com/NVIDIA/nvalchemi-toolkit-ops.git \
  26dbceb61e30cca80e1a5805eebeb51d7dc68fd1 5fffc0726b1e71bb8b68e10b5f87942064759f5b
```

候选输入若登记额外官方 fix SHA，也需显式获取该对象；本轮 PR194 的 fix 已可从 Ops 候选到达。
工具拒绝短 SHA、ref 漂移、缺失对象和浅历史；本地仓库须保留完整 subtree 导入基线提交。
它分析冻结的本地产品 Git 子树，另列未提交的产品差异，不将工作树内容伪装为该提交证据。

从项目根目录执行：

```bash
.venv/bin/python scripts/upstream_delta.py \
  --framework-repo artifacts/upstream-delta-cache/framework.git \
  --ops-repo artifacts/upstream-delta-cache/ops.git \
  --output reports/upstream-delta-2026-10-new.md \
  --json-output artifacts/upstream-delta-v1/analysis-new.json
.venv/bin/python -m unittest discover -s scripts -p test_upstream_delta.py -v
```

可用 `--lock`、`--candidate`、`--local-ref` 指定其他明确输入；输出只准写到本工程的
`reports/` 或 `artifacts/`，不得覆盖输入。保留原报告时先选择新的输出文件名。
Markdown 或 JSON 任一输出已存在时，默认在分析和写入前报错（exit 2）。
明确允许替换输出时使用 `--overwrite`；输入文件保护及输出路径冲突检查始终有效。
同一输入、工具与 Python/Git 版本重复生成的 Markdown/JSON 字节一致；报告保存输入/工具哈希及分析环境，
JSON 保存同签名实现变化和完整依赖配置。文件交集和 AST 条目计数都不是已确认冲突/正式 API 数量。
以上命令使用当前补丁文档和工具生成新的报告；原 B 报告的字节复现见第4.2节。

### 4.2 历史报告复现与 V1 维护收口

原 B 报告依赖完整输入：Candidate、版本锁、`UPSTREAM.md`、`UPSTREAM_BUGFIXES.md`、
本地产品 SHA、上游 Git 对象、工具版本及 Python/Git 版本。工具自身哈希写入报告，
历史字节复现使用原工具；当前补丁文档继续维护后续审查结果。

在项目根目录，用一个尚不存在的 worktree 路径复现 B；复用第4.1节的独立对象缓存：

```bash
git worktree add --detach artifacts/upstream-delta-history-b \
  84b4ba95ac008d4022acfcff6ea076a5d7793a55
.venv/bin/python artifacts/upstream-delta-history-b/scripts/upstream_delta.py \
  --framework-repo "$PWD/artifacts/upstream-delta-cache/framework.git" \
  --ops-repo "$PWD/artifacts/upstream-delta-cache/ops.git" \
  --output "$PWD/artifacts/upstream-delta-history-b/artifacts/reproduced-delta.md" \
  --json-output "$PWD/artifacts/upstream-delta-history-b/artifacts/reproduced-delta.json"
cmp reports/upstream-delta-2026-10.md \
  artifacts/upstream-delta-history-b/artifacts/reproduced-delta.md
```

本轮环境为 Python3.12.13 / Git2.34.1，与原报告记录一致。新路径保留原证据；
只恢复 Candidate 后用当前补丁文档或修改后的工具生成，不能保证与原报告字节相同。

2026-10-09 最小维护收口完成（修复提交 `34dcf9f`，待用户确认集成/推送）：
21项 unittest 通过（原14项保留、新增7项）；原 B Markdown 字节一致，SHA256 均为
`64074a979847a0db07395f93287bc8104ee96e88bfddccd1623eebd47f0ed817`。
当前工具用相同冻结输入生成的 Markdown/JSON 仅工具哈希不同；实际历史报告覆盖被拒绝，exit2。
D 的结构化结果完整迁移，原报告/UBF/产品源码/版本锁/生产环境控制及两套缓存引用保持原值。
原始命令、退出码、哈希和日志位于 `artifacts/upstream-sync-v1-maintenance-2026-10-09/`。
本轮只验收工具和记录；沿用既有 HCU 证据，后续正常功能移植在新 session 单独确认范围。

[UPSTREAM.md](UPSTREAM.md) 保留全部 LP 适配索引；
[UPSTREAM_BUGFIXES.md](UPSTREAM_BUGFIXES.md) 保留科学原因、本地/fork 来源和验证边界。
[UPSTREAM_BUGFIX_STATUS.yaml](UPSTREAM_BUGFIX_STATUS.yaml) 已由 C 建立，作为上游对齐状态来源；
原因和本地实现仍由现有 Markdown 记录解释。结构化对照维护 local commit、official issue/PR、fix commit、observed branch、
equivalence、action、核验日期、源码路径和 verification，并注明等价判断的具体范围。
核心公式等价不等于整个实现等价。官方修复后仍需检查额外的边界处理、兼容逻辑和测试价值。
移除补丁须验证边界已被覆盖，回归测试和平台适配仍保留。反馈上游需另获发送授权。

首轮 16 项结论、官方回归吸收和验证缺口见
[C 报告](../reports/upstream-bugfix-alignment-2026-10.md)。状态文件使用 JSON-compatible YAML，
可用 Python stdlib `json.load` 读取；equivalence 和 action 枚举及 unknown 的含义写在 policy 中。
`upstream_fix_commit` 仅证明明示范围的具体修复；partial 仍需保留额外边界。

每次新候选先生成 Delta，再人工复核该表：核对 candidate SHA，检查已登记官方 fix 的
包含性、源码变化和额外边界，更新日期、作用范围与 verification。旧 SHA 的等价结论
不得直接套到新 SHA。B 工具报告保留生成时的 UBF 线索，C 表独立记录人工等价判断；
不从 Delta 的文件交集或提交主题自动更新 equivalence，不覆盖历史报告。
独立科学案例可运行 `scripts/test_upstream_bugfix_regressions.py`，它验证当前本地
Torch reference，不能据此宣称候选或原生 Warp 已通过。

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

首次 D 已完成，见 [隔离升级决策](../reports/upstream-sync-2026-10.md)。完整候选组合仍为
Candidate：stock import/依赖/API/适配闭包存在阻塞；三原样独立模块的 CPU/HCU 结果
单独登记，不能晋升整包为 Validated。本轮结论暂不整体升级，日常移植可继续。

复用该实验时按以下顺序：

1. 从稳定 develop 建立 `sync/upstream-*` 的独立 worktree；带入已确认的流程/补丁表时
   核对产品子树不变。固定候选 SHA，从官方对象导出完整源码并记录 tree/hash/许可证。
2. 独立评估 venv 不运行 uv/pip sync。首次 D 通过自己的 `.pth` 只读借用项目已安装依赖；
   该环境未满足候选全部声明，实验结果必须注明此范围，不作为整包安装兼容证明。
3. 检查源码组合约束、实际依赖、当前产品到候选 API、跨包符号和逐文件三方文本冲突。
   单文件 merge-file 结果不替代真正 subtree 合并与 rename/跨文件语义审查。
4. 原样运行代表性依赖闭包，显式核对源码 SHA 与执行设备。不能为跑通而禁用版本门槛、
   删除高级能力或隐式混入旧包。若按路径导入和最小夹具验证，须与公开包/真实 Batch 分开记录。
5. 保存 stock import 的失败、成功模块和未运行 gate，再决定接受、独立 backport 或延期。
   完成后总结并等待确认；候选源码、runtime-env 和原始日志均不进入产品 PR。

本轮可复验 [探针](../probes/upstream_sync_v1.py)，三个模块的源码 SHA256 固定于探针。
完整候选、独立环境和命令准备见报告；调用显式传入 `--framework-root`、`--device` 与
`--output`。后续候选发生变化时重新审查探针范围和源哈希，不把旧结果套到新版本。

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

后续变更默认通过 PR 审查后集成 develop；每阶段确认不等于自动授权推送/合并。
2026-10-09 本轮用户明确取消 PR，授权 A–D 与补测直接集成本地 develop，
并同步 github-origin/gitee-origin/local-origin；该次例外按用户新指令执行，
保留成果提交和授权记录，见[集成收口报告](../reports/g2-upstream-sync-v1-closure-2026-10-09.md)。
用户明确授权外部交付后，将相同 develop SHA 同步到指定远端并逐处核验。
每两周开展一次上游变化观察；正式 release、相关 bugfix 或新功能开发前追加定向审查。
每次记录观察日期、冻结 SHA、报告和决策；定期跟踪不表示自动运行任务或自动升级。

本轮不开展全工程重构，不自动合并 NVIDIA main，不混入 BFGS、DomainParallel 或全部新特性移植。
