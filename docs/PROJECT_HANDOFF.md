# 项目背景、兼容性契约与当前交接

更新时间：2026-09-20。本文件只保留稳定架构、接口约束和当前交接；按日期追加的完整历史
保存在 [`docs/history/PROJECT_HANDOFF.md`](history/PROJECT_HANDOFF.md)。当前支持范围以
[`STATUS.md`](STATUS.md)、能力矩阵和真实报告为准。

## 1. 产品目标

基于 NVIDIA `nvalchemi-toolkit` 与 `nvalchemi-toolkit-ops`，在海光 DCU 上保留上层 API、
`AtomicData`/`Batch`、模型包装器、Dynamics、Hooks、FusedStage、训练和分布式接口，逐项建立
可验证的 Torch reference、Triton 和 HIP 后端。不得从空白重写成只保留少量功能的相似框架。

## 2. 代码和上游边界

- `packages/framework`、`packages/ops` 是正式产品源码，保持两个可独立构建的包边界。
- `external/nvalchemi-toolkit`、`external/nvalchemi-toolkit-ops` 只读，版本以
  [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml) 为准，不作为产品安装源。
- `probes/` 是设备、模型、梯度和性能验证入口，不是第三份生产实现。
- `reports/` 保存脱敏证据摘要；原始输出在被忽略的 `artifacts/`。
- 修改导入的上游源码时，在 [`UPSTREAM.md`](UPSTREAM.md) 登记补丁、动机、默认路径影响、
  upstream candidate 和回归指针。

## 3. 后端语义

- `backend=None`：保持上游 legacy Warp 语义；不能因 Hygon 设备或 Warp 缺失静默改路。
- `backend="warp"`：显式 legacy Warp 路径。
- `backend="torch_reference"`：正确性优先 reference；neighbor 的 cell-list 通过
  `method="cell_list"` 作为 operation strategy 选择。
- `backend="auto"`：只有 capability、profile 和证据满足时才可选择优化后端；当前 HIP
  neighbor 不进入 `auto`。
- `backend="triton"`：当前没有已登记生产 capability。
- `backend="hip"`：当前只登记 `hip.neighbor.cell_list-v1` 的 periodic/fixed-cell/Batch/
  full-list/MATRIX、FP32/FP64、`skin=0` 窄 capability；不满足契约必须显式失败。

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

阶段一 Torch reference 已覆盖 periodic/no-PBC、full/half、MATRIX/COO、mixed/triclinic
Batch、image shift、分层 build/query、容量、selective rebuild 和连续 geometry 梯度。

阶段二 HIP 已完成并合入 `develop`（`9f53f80`）：

- shared ABI、cell-list build、query、topology materialization、Torch geometry bridge；
- framework explicit selection/executor 与 compatibility gate；
- gfx936/BW200、`HIP_VISIBLE_DEVICES=4` 的 CPU/HCU parity 和窄 API evidence。

当前不宣称完整上游邻居后端。未覆盖或未准入的能力包括 no-PBC/half/COO HIP、skin/rebuild
lifecycle、target/pair outputs、pair-centric/sorted query、变胞、native geometry backward、
compile/opcheck、DomainParallel、`auto` 和生产默认性能。Torch geometry 继续承担训练/梯度
路径；native HIP geometry 是 forward-only 候选。

## 6. 当前任务和暂停点

当前任务队列和 owner 确认入口见 [`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)。
`TORCH-NVT-LANGEVIN` 的固定晶胞 BAOAB 窄 reference、短统计和最小 integrator continuation
state 已完成；它不等于完整上游 Langevin/NVT，剩余限制已在 STATUS 和 Feature Contract 中列明。
当前建议顺序是固定晶胞 NHC、FIRE2 stress→cell-force/variable-cell、固定晶胞
ASE-compatible BFGS。未确认 owner 前不自动创建新分支。

继续暂停：M2 planner、NPT/NPH、DomainParallel、生产 Triton/HIP 和扩大 HIP neighbor capability。

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
- 当前任务：[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)
- 系统结构：[`DEVELOPER_ARCHITECTURE.md`](DEVELOPER_ARCHITECTURE.md)
- 开发规则：[`DEVELOPMENT_GUIDE.md`](DEVELOPMENT_GUIDE.md)
- 报告索引：[`../reports/README.md`](../reports/README.md)
- 历史交接：[`history/PROJECT_HANDOFF.md`](history/PROJECT_HANDOFF.md)
