# 项目背景、兼容性契约与当前交接

更新时间：2026-10-05。本文件只保留稳定架构、接口约束和当前交接；按日期追加的完整历史
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
固定晶胞 NHC Torch reference 已完成窄范围，实现提交 `b4050ec`；
由 `codex/feat-fixed-cell-nhc` 随本轮交付到本地 `develop`，未推送；
保留异构 Batch、Hooks、FusedStage、scalar refill、最小 continuation；CPU/HCU 证据和
四项上游逻辑修正见 [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md)。
不承诺 vector arrivals、完整 workflow checkpoint、Warp/NVIDIA、DomainParallel 或生产后端。
`TORCH-CELL-STRESS-FORCE` 基础 slice 明确排除晶胞及 FIRE/FIRE2 移植。
该 slice 位于 `codex/feat-cell-stress-force`，起点 `e6dd99c`，实现与测试提交 `a76107e`，
文档提交 `35ea133`；未合并/推送；
保留旧 stress API/import，ops reference 与 generic executor 接线已完成，CPU/HCU4 的转换和
真实周期 MACE 应力证据见 [报告](../reports/g2-torch-cell-stress-force.md) 与
[契约](TORCH_CELL_STRESS_FORCE_CONTRACT.md)。公式修正记录为 UBF-009；Warp 未执行。
用户已确认下一目标为 MPA-0 + 原生 FIRE/FIRE2 弛豫，里程碑1在
`codex/test-mace-mpa0-baseline`（起点 `35ea133`）完成基础收口、来源/契约和 CPU 单点验证。
证据见 [CPU 报告](../reports/g2-mace-mpa0-m1-cpu-baseline.md) 与
[应用契约](TORCH_MACE_MPA0_RELAXATION_CONTRACT.md)。当时 MPA-0 HCU/弛豫未运行；后续证据分里程碑列出。
里程碑2已在 `codex/feat-cell-tools-reference`（起点 `f802d49`，实现 `036827e`）完成通用晶胞工具移植，
CPU/HCU4 窄验证见 [M2 报告](../reports/g2-cell-tools-m2.md) 和
[工具契约](TORCH_CELL_TOOLS_CONTRACT.md)。UBF-010 保护左右手分数坐标/strided 写回；
M2 的 Warp 数值未执行，M2 本身未移植优化步。
里程碑3已在 `codex/feat-variable-cell-fire`（起点 `d5b5f3c`，实现 `29a70f3`）接通原生六项 FIRE2、
指定 fork 修复后的完整矩阵 FIRE 和 AlignCellHook 预处理。ops 单步39项、公共同步18项；
HCU4 合成单步16组/公共解析势4组通过；MPA-0 CPU 三个 CIF、两精度/两优化器各3步通过。
见 [M3 报告](../reports/g2-variable-cell-fire-m3.md) 和
[契约](TORCH_VARIABLE_CELL_FIRE_CONTRACT.md)。UBF-011–014 注释/归因已记录；
实际 Warp、MPA-0 HCU 当时未验收；M3 分支未合并/推送。
里程碑4已在 `codex/feat-mpa0-relaxation-workflow`（起点 `317b988`，实现 `7a8af50`）完成窄工作流验收：
邻居变胞、联合收敛、退出/私有状态保护、scalar refill priming、薄 CIF/JSONL 入口。
CPU 426 passed/3 skipped，HCU4 只运行两精度/两优化器合成工作流四组；真实 MPA-0
CPU FP64 FIRE 三样本收敛、FIRE2 NaLiTl2 收敛，另两项墙钟退出；FP32 只短步预算。
14份实际输出 CIF 独立核验；UBF-015/016 与本地应用 LP-083 区分归因。
见 [M4 报告](../reports/g2-mpa0-relaxation-m4.md)。新分支未合并/推送。
里程碑5已在 `codex/test-mpa0-hcu-validation`（起点 `fa74ca1`）完成真实目标 HCU4 窄验收，
只新增探针/契约/证据，无 framework/ops/应用源码修改。CPU gate426 passed/3 skipped；
两精度/两优化器单体系及3结构异构补位全部收敛。扩大为8个不同真实CIF、共59原子、
容量4体系/64原子后，两精度 FIRE各8项收敛、FIRE2各7项收敛、Li(CN)2各500步预算退出。
48份输出全部独立CPU/HCU重算；46份 converged/2份 max_steps，没有扩大阈值。
Li(CN)2 FIRE FP64/FP32分别88/364步、不同最终E/V；精度敏感性与重复稳定性未验收。
详见 [M5 报告](../reports/g2-mpa0-relaxation-m5.md)。用户确认更大 Batch 测量后，
M5已收口提交 `4656cce`，未合并/推送。当前 `codex/test-mpa0-batch-timing` 起点即该提交：
只测一个FP64/FIRE原生完整矩阵Batch，64个随机真实输入/2287原子，容量2304、refill20；
62项原生raw力阈值.05收敛、2项500步预算退出。应用101.046秒、探针124.420秒，
峰值allocated/reserved为23.486/35.096 GiB。64项NumPy力/ID/哈希/CIF几何核验通过，
没有独立模型重算；此判据与M5双阈值、H100的UnitCellFilter判据不同。
没有运行完整1600或16000原子Batch；按25个同等批次条件估算约41–50分钟，
不能承诺全部收敛或持续补位收益。详见[测量报告](../reports/g2-mpa0-single-batch-timing.md)。
用户追加8000门槛与分项测量后，上一测量已收口 `7410445`，未合并/推送。
8000诊断分支 `codex/test-mpa0-batch8000-timing` 起点为该提交：
同一抽样顺序前231项/7937原子，容量8000，80%/95%/默认100%allocator上限
均在初始e3nn张量积前向OOM，模型求值/优化更新完成数0，没有弛豫输出。
默认上限峰值allocated/reserved为59.115/62.604 GiB；冷邻居2.007秒、失败前部分模型
.694秒，不能当完整模型/优化循环耗时。分项同步包装只在探针中，无framework/ops改动。
见[容量诊断报告](../reports/g2-mpa0-batch8000-timing.md)。作业后无KFD PID。
用户确认4000门槛后，8000诊断收口 `eff753d`，未合并/推送。
当前 `codex/test-mpa0-batch4000-timing` 起点eff753d：同一抽样前缀114结构/3959原子，
FP64/e3nn/原生FIRE，显式Torch reference邻居（变胞/COO未纳入优化HIP邻居支持范围）。
一次完整Batch完成，111项原生raw力阈值收敛、3项500更新预算退出，无OOM/溢出/计算错误。
应用160.145秒、探针182.673秒；循环156.489秒中模型118.819（75.93%）、
邻居30.261（19.34%）、优化器4.404（2.81%）；峰值allocated/reserved为41.131/49.938 GiB。
114项CPU NumPy原生力/ID/哈希/CIF几何/计时核验通过，无独立模型重算或完整产品gate。
见[4000报告](../reports/g2-mpa0-batch4000-timing.md)，作业后无KFD PID。
本阶段尚未提交；下一建议核对并恢复FIRE dt_min=0上游兼容性，CPU回归/HCU短步，待确认。
测量显式采用默认.002；当前未修正该边界，也未扩展性能后端或完整1600作业。
里程碑顺序按用户2026-10-05新要求执行：实施前说明目的与方案并等待确认；完成后
先总结实现、验证和限制，再给下一阶段方案；用户确认下一阶段后，才收口提交当前
阶段并启动下一阶段，不自动推送。M4 的 `7a8af50`/`f909f52` 已在此新要求前提交，
既有提交保留；流程文档已在 M5 启动前以 `fa74ca1` 收口；全目录长作业另行授权。
固定晶胞 ASE-compatible BFGS 保留为后续独立队列。

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
- 模块与应用总览：[`IMPLEMENTATION_STATUS_MATRIX.md`](IMPLEMENTATION_STATUS_MATRIX.md)
- 当前任务：[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)
- 系统结构：[`DEVELOPER_ARCHITECTURE.md`](DEVELOPER_ARCHITECTURE.md)
- 开发规则：[`DEVELOPMENT_GUIDE.md`](DEVELOPMENT_GUIDE.md)
- 报告索引：[`../reports/README.md`](../reports/README.md)
- 历史交接：[`history/PROJECT_HANDOFF.md`](history/PROJECT_HANDOFF.md)
