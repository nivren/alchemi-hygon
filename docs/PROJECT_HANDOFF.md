# 项目背景、兼容性契约与当前交接

更新时间：2026-11-06（本次仅更新集成交付状态）。本文件只保留稳定架构、接口约束和当前交接；完整历史保存在[整理前快照](history/PROJECT_HANDOFF-snapshot-2026-11-06.md)与
[更早交接记录](history/PROJECT_HANDOFF.md)。当前支持范围以
[`STATUS.md`](STATUS.md)、能力矩阵和真实报告为准。

本地 `develop` 已纳入BFGS四个提交；CPU/HCU候选复验通过，本次未推送。
当前集成记录见[报告](../reports/partner-reference-stage1-integration-2026-11-06.md)。

## 1. 产品目标

基于 NVIDIA `nvalchemi-toolkit` 与 `nvalchemi-toolkit-ops`，在海光 DCU 上保留上层 API、
`AtomicData`/`Batch`、模型包装器、Dynamics、Hooks、FusedStage、训练和分布式接口，逐项建立
可验证的 Torch reference、Triton 和 HIP 后端。不得从空白重写成只保留少量功能的相似框架。

## 2. 代码和上游边界

- `packages/framework`、`packages/ops` 是正式产品源码，保持两个可独立构建的包边界。
- 面向具体模型/材料数据的应用脚本归独立下游工程 `alchemi-app`；本仓库只保留通用示例、framework/ops 能力和上游兼容性跟踪。
- `external/nvalchemi-toolkit`、`external/nvalchemi-toolkit-ops` 只读，版本以
  [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml) 为准，不作为产品安装源。
- `probes/` 是设备、模型、梯度和性能验证入口，不是第三份生产实现。
- `reports/` 保存脱敏证据摘要；原始输出在被忽略的 `artifacts/`。
- 修改导入的上游源码时，在 [`UPSTREAM.md`](UPSTREAM.md) 登记补丁、动机、默认路径影响、
  upstream candidate 和回归指针。缺陷原因与历史实现补齐见
  [`UPSTREAM_BUGFIXES.md`](UPSTREAM_BUGFIXES.md)。

## 3. 后端语义

- `backend=None`：保持上游 legacy Warp 语义；不能因 Hygon 设备或 Warp 缺失静默改路。
- `backend="warp"`：显式 legacy Warp 路径。
- `backend="torch_reference"`：正确性优先 reference；neighbor 的 cell-list 通过
  `method="cell_list"` 作为 operation strategy 选择。
- `backend="auto"`：只有 capability、profile 和证据满足时才可选择优化后端；当前 HIP
  neighbor 不进入 `auto`。
- `backend="triton"`：当前没有已登记生产 capability。
- `backend="hip"`：当前登记 `hip.neighbor.cell_list-v1` 的 periodic/Batch/full-list/MATRIX/COO（Torch打包）、
  FP32/FP64、`skin=0`；调用间晶胞可变，每次完整重建。不满足契约必须显式失败。

后端选择必须按 operation、device、dtype、layout、PBC、full/half、梯度等级、stream、容量、
确定性和分布式约束登记。framework 解析一次 `BackendSelection` 并传给 ops；dispatcher 不
重复解析，不允许未记录的 fallback。

## 4. Feature Compatibility Contract

每条特性在 [`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml) 中独立记录：

- A/B/C 分类与用户场景；
- 锁定的上游 SHA、路径和符号；
- 输入输出 shape/dtype/layout、单位、PBC、full/half、空输入、容量和错误；
- mutation/alias、stream、确定性、梯度等级和 Batch/inflight 语义；
- backend、测试、证据、DCU 验证状态、限制和 release gate。

`status` 描述目标契约的实现阶段，`verification.dcu_status` 描述列出的 HCU 证据范围；
`partial` 不表示完整特性已通过。拓扑通常不可微，但 distance/vector/energy/force 连续路径
必须分别说明一阶、二阶和训练力损失路径。

## 5. 当前邻居实现边界

Torch reference 已覆盖 periodic/no-PBC、full/half、MATRIX/COO、mixed/triclinic Batch、
image shift、分层 build/query、容量、selective rebuild 和连续 geometry 梯度。
显式 HIP 邻居已登记 full-PBC/Batch/full-list/MATRIX + 设备内 Torch COO、FP32/64、
`skin=0`；允许调用间晶胞变化但每次完整重建，不进入 `auto`。原 M1–M9 的窄范围
开发与验证记录可在[整理前交接快照](history/PROJECT_HANDOFF-snapshot-2026-11-06.md)
和[报告索引](../reports/README.md)追溯；特别是 M9 已复跑两组受影响 FP32 FIRE2。

HIP no-PBC/half、skin/selective rebuild、target/pair、旧 plan 变胞复用、native geometry
backward、compile/opcheck、DomainParallel 与自动准入尚未覆盖。Torch geometry 仍是已有
连续梯度路径；指定场景的实测性能不等于生产默认性能。

## 6. 当前任务和暂停点

BFGS 四提交已纳入本地 `develop`（文档集成 `42f3f19`），CPU/HCU 窄范围复验与
11体系 ASE 对照见[集成报告](../reports/partner-reference-stage1-integration-2026-11-06.md)。
参考版本正在整理项目入口、当前状态和可运行示例。外部模型及两类 CIF 结构可由用户
提供，输入与模型路径应可配置；具体应用仍以实际代码和验证报告限定。

此前 upstream sync Candidate、HIP 邻居 M1–M9、MPA-0 应用分离和 BFGS M1–M5 的完整
阶段经过保留在[整理前快照](history/PROJECT_HANDOFF-snapshot-2026-11-06.md)，
上游候选未整体升级。NPT/NPH、DomainParallel、生产 Triton/HIP dynamics 和真实 MLIP
完整训练仍需单独规划与验收。

每个阶段和小里程碑开始前先向用户说明目标；完成后总结结果和验证，获得用户明确确认
才开始下一个。不得把历史推送或合并授权用于新的交付；当前 Git 状态须重新检查。

## 7. 验证与交接规则

- 先写 operation contract 和 CPU Torch oracle，再做 dispatcher/framework 接线，再做 HCU。
- CPU 与 framework/ops pytest 分进程运行，避免顶层 `test` 包导入冲突。
- HCU 证据必须注明 DTK、设备、架构、可见卡、dtype、warmup/同步、退出码和共享负载。
- 本机后续单卡验证统一使用 `HIP_VISIBLE_DEVICES=4`；历史 HCU 0 证据保持原样。
- 性能区分 cold/JIT、warm steady、kernel/device time、API wall-clock 和端到端，不把一个窄
  workload 推广为全局排序。
- 提交前运行 `git diff --check`、相关 CPU gate 和可获得的限时 HCU probe；使用短而单一目的
  的 `git commit -s`。

## 8. 相关入口

- 当前状态：[`STATUS.md`](STATUS.md)
- 模块与应用总览：[`IMPLEMENTATION_STATUS_MATRIX.md`](IMPLEMENTATION_STATUS_MATRIX.md)
- 当前任务：[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)
- 系统结构：[`DEVELOPER_ARCHITECTURE.md`](DEVELOPER_ARCHITECTURE.md)
- 开发规则：[`DEVELOPMENT_GUIDE.md`](DEVELOPMENT_GUIDE.md)
- 报告索引：[`../reports/README.md`](../reports/README.md)
- 2026-11-06整理前交接：[快照](history/PROJECT_HANDOFF-snapshot-2026-11-06.md)；更早历史：[`history/PROJECT_HANDOFF.md`](history/PROJECT_HANDOFF.md)
