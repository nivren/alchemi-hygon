# 项目背景、兼容性契约与当前交接

更新时间：2026-10-09。本文件只保留稳定架构、接口约束和当前交接；按日期追加的完整历史
保存在 [`docs/history/PROJECT_HANDOFF.md`](history/PROJECT_HANDOFF.md)。当前支持范围以
[`STATUS.md`](STATUS.md)、能力矩阵和真实报告为准。

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

阶段一 Torch reference 已覆盖 periodic/no-PBC、full/half、MATRIX/COO、mixed/triclinic
Batch、image shift、分层 build/query、容量、selective rebuild 和连续 geometry 梯度。

阶段二 HIP 已完成并合入 `develop`（`9f53f80`）：

- shared ABI、cell-list build、query、topology materialization、Torch geometry bridge；
- framework explicit selection/executor 与 compatibility gate；
- gfx936/BW200、`HIP_VISIBLE_DEVICES=4` 的 CPU/HCU parity 和窄 API evidence。

HIP M1 当前晶胞快照已在 `codex/feat-hip-current-cell-neighbors` 完成 CPU/HCU4 窄验证，
实现收口 `fecbc39`，本次随 M1–M9 提交链合入 `develop`；见[契约](HIP_CURRENT_CELL_NEIGHBOR_CONTRACT.md)与[报告](../reports/g2-hip-current-cell-neighbor-m1.md)。
每次重新派生 inverse/grid/radius/offset 并创建新 plan；低层复用 plan 仍是固定 metadata ABI。
修正 Hook 的容量溢出类型，并以 `auto_enabled=False` 落实 HIP 不进入 auto 的既定边界。

HIP M2 在 `codex/feat-hip-coo-mace-snapshot` 完成设备内 Torch COO 接线与真实 MPA-0
两精度/三快照单点 E/F/stress gate，实现收口 `3e394d7`；见[M2契约](HIP_COO_MACE_SNAPSHOT_CONTRACT.md)
与[报告](../reports/g2-hip-coo-mace-snapshot-m2.md)。Hook 允许将过量分配的 MATRIX
裁去 padding，但有效 counts 超容量明确失败。模型单点不等于 HIP FIRE/FIRE2 工作流验收。
后续 M3 在 `codex/test-hip-variable-cell-workflow` 完成现有 Torch reference 优化器 + HIP
COO 的 CPU/HCU4 短步/生命周期 gate，以及真实 MPA-0 两精度/两优化器各五步和预算补位；
无新增产品 kernel，已收口实现 `6b1680a`，见[契约](HIP_VARIABLE_CELL_WORKFLOW_CONTRACT.md)与
[报告](../reports/g2-hip-variable-cell-workflow-m3.md)。真实预算退出不代表物理收敛。

M4 在 `codex/perf-hip-neighbor-workspace` 完成性能基线/trace与workspace实验，目标
配对median下降1.29%、真实neighbor+model下降0.64%，未达5%门槛，产品缓存已撤回，
原stateless源码恢复。保留显式归档重放benchmark和证据，见
[M4报告](../reports/g2-hip-neighbor-workspace-m4.md)。主热点为逐体系metadata；后续M5结果如下。

M5在 `codex/perf-hip-batch-metadata` 完成当前晶胞批量metadata，每次仍全量重建，
只有一个产品HIP后端。CPU ops45/framework21、HCU4 ops73/framework49通过；
真实两精度FIRE/FIRE2五步及预算补位回归通过。约20,000原子、200/40/10结构
两精度、六变胞/补位快照三路径比较完成；FP64相对Torch邻居约130/29/11倍，
仅所测邻居API，不是完整弛豫吞吐。原小Batch FP32有既有cutoff舍入分歧，失败保留。
见[M5契约](HIP_BATCH_METADATA_CONTRACT.md)与[M5报告](../reports/g2-hip-batch-metadata-m5.md)。
用户确认M6后，M5实现以 `e2d569c` 收口；本次随提交链合入 `develop`，后续查询并行实验见M6。

M6在 `codex/perf-hip-cooperative-query` 完成单一HIP查询协作实现及窄验收，实现收口 fe18147、文档 c2a8339；本次随提交链合入 `develop`。
CPU ops51/framework21、HCU环境ops85/framework49；真实两精度两优化器短轨迹及补位通过。
FP64约20,000原子200/40/10结构公共邻居相对M5下降15.37%/15.46%/11.34%，
两精度八配置无稳定退化；实际18原子邻居+模型约5.3%–5.5%下降，未含整个优化步。
同名扩展别名的首轮时序作废保留，最后用独立扩展名验证；FP32原cutoff边界分歧未扩大。
见[M6契约](HIP_COOPERATIVE_QUERY_CONTRACT.md)与[报告](../reports/g2-hip-cooperative-query-m6.md)。
M7在 `codex/test-hip-real-batch-validation` 完成较大真实Batch实验及M1–M6审阅，签名提交 `00637c0`；本次随提交链合入 `develop`。
FP64真实1029/39及2030/92两优化器三轮三步、47/100 CIF预算补位通过；
完整step相对Torch reference cell-list约2.04–2.37倍，模型耗时几乎相同；
CPU ops51/framework21、HCU环境ops85/framework49定向回归通过。
FP32仅1029 FIRE完整组通过，其余三组forces重复严格状态门失败；
纯Torch/Torch FIRE2也出现3.03e-5失败，逐物理量输出容差全部通过，未改任何容差。
见[M7契约](HIP_REAL_BATCH_VALIDATION_CONTRACT.md)和[报告](../reports/g2-hip-real-batch-validation-m7.md)。
M8 `codex/fix-mpa0-fp32-state-validation` 比较层修正与真实Batch复验完成，签名收口 `04f80bd`；只改probe测试与文档，framework/ops产品源码、HIP kernel和数值容差未改。CPU单测3 passed；HCU八组合中六组完整通过，另两组所有数值/轨迹状态比较通过但被每个晶胞必须移动的附加断言阻断。纯Torch/Torch控制未出现几何/私有状态严格门失败。见[M8契约](HIP_MPA0_FP32_STATE_VALIDATION_M8.md)及[报告](../reports/g2-hip-mpa0-state-validation-m8.md)。
M9定位两组相同的未移动结构为 `mp-1013795_Li3Ge3(MoO6)2.cif`：HIP/Torch reference 的终态晶胞均与初态完全相同，应力约 `7.68e-4 eV/Å³`。移除“每个结构均须移动”的过强probe条件，保留逐结构backend/state parity和批次至少发生一次晶胞变化；两组FP32 FIRE2在HCU4复跑均通过。只改probes和文档，无邻居后端或framework/ops改动。见[M9报告](../reports/g2-hip-fp32-fire2-cell-movement-m9.md)，实现 `fa1259b`、收口文档 `b549558`。
2026-10-09 用户明确要求将已完成更新合入 `develop` 并推送到 `github-origin`、`gitee-origin`、`local-origin`；本次以 fast-forward 集成 M1–M9 提交链及交接文档更新，保留各阶段报告的历史验证范围。

当前不宣称完整上游邻居后端。未覆盖或未准入的能力包括 no-PBC/half HIP、skin/selective rebuild
lifecycle、target/pair outputs、pair-centric/sorted query、旧 plan 变胞复用、native geometry backward、
compile/opcheck、DomainParallel、`auto` 和生产默认性能。Torch geometry 继续承担训练/梯度
路径；native HIP geometry 是 forward-only 候选。

## 6. 当前任务和暂停点

用户已确认 G2 集成收口与 Upstream Sync V1，长期规则见
[UPSTREAM_SYNC.md](UPSTREAM_SYNC.md)。A 已在集成源码 `2e211ab` 完成指定 CPU/HCU4 验收：
CPU reference `425 passed/3 skipped`，HCU ops `101 passed`、framework `77 passed/1 xfailed`；
两精度/两优化器/两规模真实 MPA-0 Batch 共 8/8 通过，状态/预算补位和窄范围限制详见
[A 报告](../reports/g2-integration-a-2026-10-09.md)。实际 Warp/NVIDIA 和完整生产能力未验证。
用户确认后 A 以 `132ff32` 签名收口；B 在 `wangleping/feat-upstream-delta-v1` 完成只读
差异工具/候选记录/[报告](../reports/upstream-delta-2026-10.md)，14 项 CPU Git/AST 夹具验证通过。
B 阶段未运行候选源码/HCU，候选仍为 Candidate；产品源码/锁/环境未改变。
B 已签名收口 `84b4ba9`；C 在 `wangleping/docs-upstream-bugfix-alignment` 完成
[UBF-001～016 官方对照](UPSTREAM_BUGFIX_STATUS.yaml)：核心 equivalent 3、partial 5、
different 1、unknown 7；全部 retain。PR194 本地科学案例 6 passed，既有 ops
111 passed/3 skipped、framework compatibility 87 passed、storage 两节点通过；
checkpoint 隔离断言 1 passed，原 MACE 测试模块收集失败独立记录。
45 份源码快照和 912 份产品/夹具哈希核验通过；没有新增候选/HCU/Warp 运行证据。
[C 报告](../reports/upstream-bugfix-alignment-2026-10.md)已签名收口 `ad18998`。
D 在 `sync/upstream-2026-10` 独立 worktree 完成
[隔离升级评估](../reports/upstream-sync-2026-10.md)：候选三原样模块 CPU/HCU4 各11项检查通过，
stock Batch/L-BFGS 导入因缺 Warp 失败；依赖/API/三方文本冲突尚有阻塞，结论暂不整体升级。
这不是候选整包或 grouped 生命周期验收，现有产品/版本锁/环境/能力契约保持原值。
D 已签名收口 `a0a3763`，完整候选组合保持 Candidate。
A–C 完成后恢复正常功能开发，D 可单独确认并行；不升级当前锁或 Torch/DTK。

当前任务队列和 owner 确认入口见 [`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)。

用户随后确认补充 FIRE2 变胞 + HIP 邻居小规模完整弛豫；已在独立验证 worktree 完成，
见[补测报告](../reports/g2-hip-fire2-convergence-2026-10-09.md)。HCU4两精度/两邻居分支四作业
均退出0，固定三个Li/Na CIF全部双阈值收敛，真实补位、终态同几何和导出CPU/HCU核验通过。
CPU定向59 passed/16设备skip、探针4 passed；产品/锁/环境保持原值。
这是本地稳定基线的三样本证据，与另一个sync worktree中的候选评估分别保存；
不提升上游候选版本状态。补测已签名收口 `8306b7f`。

A–D 与补测已整理到 `wangleping/integration-g2-upstream-sync-v1` 独立分支，
合并提交 `8623517` 保留两个成果提交；CPU 定向工作流59、Delta14、科学案例6、
补测夹具4、候选独立模块11项复验通过，工作流16项既有设备skip单列。
[集成收口报告](../reports/g2-upstream-sync-v1-closure-2026-10-09.md)记录后续用户授权：
本轮取消 PR，直接合并到本地 develop 并推送 github-origin/gitee-origin/local-origin。
本地 develop 已从 `2e211ab` 快进到 `68ceb00`，随后带入本节签名交付记录；
三处 develop 使用同一最终提交，SHA/退出码核验保存于
`artifacts/g2-upstream-sync-delivery-2026-10-09/verification.json`。
复用已有准确 SHA/探针哈希对应的测试证据，本次未新增 CPU/HCU 测试。

`TORCH-NVT-LANGEVIN` 的固定晶胞 BAOAB 窄 reference、短统计和最小 integrator continuation
state 已完成；它不等于完整上游 Langevin/NVT，剩余限制已在 STATUS 和 Feature Contract 中列明。
固定晶胞 NHC Torch reference 已完成窄范围，实现提交 `b4050ec`；
由 `codex/feat-fixed-cell-nhc` 交付到本地 `develop`，并随本次集成同步到三个配置的目标 remote；
保留异构 Batch、Hooks、FusedStage、scalar refill、最小 continuation；CPU/HCU 证据和
四项上游逻辑修正见 [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md)。
不承诺 vector arrivals、完整 workflow checkpoint、Warp/NVIDIA、DomainParallel 或生产后端。
`TORCH-CELL-STRESS-FORCE` 基础 slice 明确排除晶胞及 FIRE/FIRE2 移植。
该 slice 位于 `codex/feat-cell-stress-force`，起点 `e6dd99c`，实现与测试提交 `a76107e`，
文档提交 `35ea133`；其提交已包含在 `develop` 集成 merge `620a485` 中并已同步至三个目标 remote；
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
实际 Warp、MPA-0 HCU 当时未验收；M3 分支提交已随整体移植链合入 `develop`。
里程碑4已在 `codex/feat-mpa0-relaxation-workflow`（起点 `317b988`，实现 `7a8af50`）完成窄工作流验收：
邻居变胞、联合收敛、退出/私有状态保护、scalar refill priming、薄 CIF/JSONL 入口。
CPU 426 passed/3 skipped，HCU4 只运行两精度/两优化器合成工作流四组；真实 MPA-0
CPU FP64 FIRE 三样本收敛、FIRE2 NaLiTl2 收敛，另两项墙钟退出；FP32 只短步预算。
14份实际输出 CIF 独立核验；UBF-015/016 与本地应用 LP-083 区分归因。
见 [M4 报告](../reports/g2-mpa0-relaxation-m4.md)。M4 提交已随整体移植链合入 `develop`。
LP-083 的 MPA-0 CIF 应用入口及其专属测试现已移至同级独立工程 `alchemi-app`；本仓库
保留 M4/M5 的历史验证记录、上游迁移契约和 framework 通用生命周期测试，不再承载该应用 CLI。
里程碑5已在 `codex/test-mpa0-hcu-validation`（起点 `fa74ca1`）完成真实目标 HCU4 窄验收，
只新增探针/契约/证据，无 framework/ops/应用源码修改。CPU gate426 passed/3 skipped；
两精度/两优化器单体系及3结构异构补位全部收敛。扩大为8个不同真实CIF、共59原子、
容量4体系/64原子后，两精度 FIRE各8项收敛、FIRE2各7项收敛、Li(CN)2各500步预算退出。
48份输出全部独立CPU/HCU重算；46份 converged/2份 max_steps，没有扩大阈值。
Li(CN)2 FIRE FP64/FP32分别88/364步、不同最终E/V；精度敏感性与重复稳定性未验收。
详见 [M5 报告](../reports/g2-mpa0-relaxation-m5.md)。用户确认更大 Batch 测量后，
M5已收口提交 `4656cce`，现已随整体移植链合入 `develop`。当前 `codex/test-mpa0-batch-timing` 起点即该提交：
只测一个FP64/FIRE原生完整矩阵Batch，64个随机真实输入/2287原子，容量2304、refill20；
62项原生raw力阈值.05收敛、2项500步预算退出。应用101.046秒、探针124.420秒，
峰值allocated/reserved为23.486/35.096 GiB。64项NumPy力/ID/哈希/CIF几何核验通过，
没有独立模型重算；此判据与M5双阈值、H100的UnitCellFilter判据不同。
没有运行完整1600或16000原子Batch；按25个同等批次条件估算约41–50分钟，
不能承诺全部收敛或持续补位收益。详见[测量报告](../reports/g2-mpa0-single-batch-timing.md)。
用户追加8000门槛与分项测量后，上一测量已收口 `7410445`，现已随整体移植链合入 `develop`。
8000诊断分支 `codex/test-mpa0-batch8000-timing` 起点为该提交：
同一抽样顺序前231项/7937原子，容量8000，80%/95%/默认100%allocator上限
均在初始e3nn张量积前向OOM，模型求值/优化更新完成数0，没有弛豫输出。
默认上限峰值allocated/reserved为59.115/62.604 GiB；冷邻居2.007秒、失败前部分模型
.694秒，不能当完整模型/优化循环耗时。分项同步包装只在探针中，无framework/ops改动。
见[容量诊断报告](../reports/g2-mpa0-batch8000-timing.md)。作业后无KFD PID。
用户确认4000门槛后，8000诊断收口 `eff753d`，现已随整体移植链合入 `develop`。
4000测量 `codex/test-mpa0-batch4000-timing` 起点eff753d，收口1dba058：同一抽样前缀114结构/3959原子，
FP64/e3nn/原生FIRE，显式Torch reference邻居（测量时变胞/COO未纳入优化HIP邻居支持范围）。
一次完整Batch完成，111项原生raw力阈值收敛、3项500更新预算退出，无OOM/溢出/计算错误。
应用160.145秒、探针182.673秒；循环156.489秒中模型118.819（75.93%）、
邻居30.261（19.34%）、优化器4.404（2.81%）；峰值allocated/reserved为41.131/49.938 GiB。
114项CPU NumPy原生力/ID/哈希/CIF几何/计时核验通过，无独立模型重算或完整产品gate。
见[4000报告](../reports/g2-mpa0-batch4000-timing.md)，作业后无KFD PID。
用户确认后先提交4000测量，再在 `codex/fix-fire-zero-dt-min` 修正 LOCAL-004：
只将完整矩阵 FIRE reference 校验由 `dt_min<=0` 改为 `<0`，恢复锁定 NVIDIA/fork 零下限兼容。
CPU ops46/公共19项通过，HCU4 FP64四种功率单步符合NumPy、三种非法边界无mutation，
公共解析势25步CPU/HCU对照及三CIF/124原子MPA-0十步通过；证据见
[零下限报告](../reports/g2-fire-zero-dt-min.md)。不代表零下限完整弛豫收敛或完整HCU gate；
修正已签名提交 `0e354f4`，并与应用分离收口提交一起于本地 merge `620a485` 合入 `develop`。
M1–M5与应用设置收口审查见[审查报告](../reports/g2-relaxation-port-closure-review.md)；
toolkit 收口提交为 `730a848`，下游应用首提交为 `9ae235c`，均已提交到各自本地 `develop`；
三个 remote 的 `develop` refs 已核验一致地包含上述交付提交；GitHub 提示直推绕过了要求经 PR 的分支规则，但按用户明确授权完成。
历史4000测量仍采用.002，无零下限完整Batch复测；没有扩展性能后端或1600作业。
里程碑顺序按用户2026-10-05新要求执行：实施前说明目的与方案并等待确认；完成后
先总结实现、验证和限制，再给下一阶段方案；用户确认下一阶段后，才收口提交当前
阶段并启动下一阶段，不自动推送。M4 的 `7a8af50`/`f909f52` 已在此新要求前提交，
既有提交保留；流程文档已在 M5 启动前以 `fa74ca1` 收口；全目录长作业另行授权。
ASE-compatible BFGS 用户已于2026-10-09确认 M1–M4；这是 NVIDIA 上游没有的
本地新增功能。M1 在 `wangleping/feat-ase-bfgs-m1` 从 develop `cfd22ad` 完成独立
State/prepare/step/restart、FP64 固定晶胞 Torch Ops 与 metadata executor 接线；
未提交/合并/推送。CPU reference 493 passed/3 skipped（BFGS68）；HCU4 三规模
各八步/重启/stream/设备守卫与 GPU profiler 通过，H误差3.55e-14、坐标6.66e-16。
M1 当时 Framework BFGS、MLIP、Filter、Batch/inflight/CSP 尚未接入；来源/命令/边界
见[M1报告](../reports/g2-ase-bfgs-m1.md)和[合同](TORCH_ASE_BFGS_CONTRACT.md)。
用户随后确认 M2，在 `wangleping/feat-ase-bfgs-m2` 延续未提交 M1：公共
ASECompatibleBFGS、模型/Hooks/严格初始收敛/全历史冻结/restart 已实现并完成
CPU/HCU4 窄验证；MPA-0 水分子11次更新收敛。最终 CPU gate、数值及实际退出码见
[M2 报告](../reports/g2-ase-bfgs-m2.md)。未提交/合并/推送，未升级版本锁或环境。
eigh 18组实测：批量候选有收益，小矩阵 B=1 Jacobi 更慢；Torch n=138 主机
LAPACKE 和 D2H/H2D 已证实。随后用户确认E1，在 `wangleping/feat-ase-bfgs-e1`
完成显式 `hip.bfgs-ase-rocsolver-v1`，共享BFGS数学、独立native求解器/current stream/
RAII/info检查，当前HCU构建已测规模无受监测host数学/矩阵回传；CPU550/3skip、
原生HCU31项、解析势/真实MPA-0及产品eigh18组均通过。小B1 Jacobi更慢，仍显式
选择；求解器batch不代表BFGS batch。未提交/集成/推送，见[E1报告](../reports/g2-ase-bfgs-e1.md)
及[合同](HIP_BFGS_EIGH_CONTRACT.md)。随后用户已确认M3。
M3在 `wangleping/feat-ase-bfgs-m3` 完成单体系完整UnitCellFilter 3N+9 Ops与
ASECompatibleBFGSVariableCell：全部ASE Filter参数、joint fmax、原子/晶胞更新及
参考晶胞/配对几何restart。BFGS数学/native solver与E1不变，固定晶胞快照保留。
最终CPU621/3既有skip；HCU变胞8/solver31回归通过，CPU/HCU解析三路径ASE
15/12/16步收敛、含压力FD约2e-10；周期Al4 MPA-0 CPU/HCU显式HIP5步/6次求值
联合收敛及重启；15 DOF完整变胞3步trace显示HIP无受监测host数学/矩阵回传、
Torch有host LAPACK，ID/检查元数据允许同步。M3当时未提交/集成/推送，
M4当时待确认；后续结果见下段。见[M3报告](../reports/g2-ase-bfgs-m3.md)与
[UnitCellFilter合同](TORCH_ASE_UNITCELL_FILTER_CONTRACT.md)。其他构建/多卡/compile未验收。
用户“继续”确认M4后，在 `wangleping/feat-ase-bfgs-m4` 完成异构固定/变胞Batch、
按ID独立状态/冻结/重排/退出/补位、按DOF分组batched数学、共享模型、Batch配对
快照与旧单体系快照提升。最终CPU650/3既有skip，HCU solver39、Batch/变胞14通过。
真实46/92原子MPA-0 FIRE2→BFGS变胞完整收敛，各ID前3步ASE坐标/晶胞约1e-14、
H约2.55e-11；四次inflight弛豫全部退出/补位/收敛，103.37秒、470主模型调用。
CPU固定晶胞配对checkpoint续算通过；HCU固定晶胞完整收敛。该CPU耗时仅续算窗口。
本机strided-batched DC在真实285DOF简并Hessian出现VMFault及info=0错误谱；当前
HIP全部宽度直接Jacobi，并加设备内分解有效性守卫，不改变ASE数学。147/285/147
DOF完整分组三步trace无受监测host数学/数值矩阵往返，D2H仅1/4字节诊断；默认
Torch仍有host路径。E1旧DC性能不代表当前代码。见[M4报告](../reports/g2-ase-bfgs-m4.md)
及[Batch合同](TORCH_ASE_BFGS_BATCH_CONTRACT.md)。M1–M4全部仍未提交/集成/推送；
小样本、当前构建FP64 eager门，不承诺全CSP成功率、大Batch吞吐或通用采样checkpoint。
M5可选；M4当时建议M5-1针对已测eigh瓶颈做同输入预热基准与方案选择，后续结果见下段。
2026-10-10用户要求独立复验整个agy-eigh-test，产品源码未改。Default已走
hipSOLVER但逐矩阵分解；当前DTK n<200 host LAPACK回退。独立无Torch HIP
复现单矩阵/strided-batched UPPER/DC错误谱，原样外部扩展复现VMFault；当前
Default/cuSOLVER在真实输入有界复验超时。Jacobi无host数学/矩阵往返，但库内
有4字节收敛读取；性能不能仅按GPU驻留推断。见[调查报告](../reports/g2-ase-bfgs-eigh-investigation.md)。
当时下一阶段建议聚焦Jacobi关键路径和安全候选，不恢复已知故障DC。
用户追加DTK26.10复验后，独立HIP真实285维单矩阵/B1/B4 DC和LOWER均正确完成；
n199 CPU LAPACK仍存在，285维DC实际命中私有CPU `steqr_host` 子块求解及数值
往返，仍不满足设备内数学合同。当前26.04构建Torch在隔离26.10环境缺
libhipblas.so.2，完整26.10 Torch/BFGS未验收；产品准入/默认DTK/全部Jacobi不变。
见[26.10报告](../reports/g2-ase-bfgs-dtk2610-eigh.md)，该版本仅为候选，不登记支持。
随后用户确认M5-1，在 `wangleping/perf-ase-bfgs-m5-1` 完成两测量探针与合同、
147/285 B1/B4三进程×7样本稳态、异构组、native hipprof及三次MPA-0真实变胞CSP。
11个BFGS/Filter产品源hash未变；没有优化/替换产品solver，版本锁/环境/默认未改。
285/B1 raw92.16ms/checked92.64ms、GPU kernel占trace97%；三次CSP均收敛，
233阶段步/234次模型求值，工作流窗口中位数56.88秒，checked占更新88.8%/整个
窗口47.8%。模型为MACE MPA-0 medium，不是mace_off small；无额外ASE模型对照，
有同步/checkpoint/冷模型首调用，不能据此声明生产吞吐或加速。全部退出0，
本阶段不重跑M4全功能gate。见[M5-1报告](../reports/g2-ase-bfgs-m5-1.md)。
用户随后“继续”确认[M5-2隔离HIP batched Jacobi原型](ASE_BFGS_M5_2_PROPOSAL.md)，
在 `wangleping/perf-ase-bfgs-m5-2` 实现自写round-robin Jacobi，仅probes接入。
FP64 LOWER147/285、B≤4：CPU oracle、HCU19例、10步ASE/重启/失败原子性，
真实MPA-0 medium两CIF变胞ASE对照均通过；233阶段步/234主轨迹模型求值。
三进程×7样本checked solver相对现有rocSOLVER Jacobi耗时下降28.7%–56%，
分组数学136.715→68.556ms；11个产品源hash及版本锁未变。
共享八卡作业随后占用HCU4，旧整段端到端CSP计时仍排除，自身driver已停止（退出143）。
后续用户确认新窗口三轮配对，六次全部退出0并收敛，均233步/234模型求值；
174次资源观察无其他HCU4进程。同步工作流中位数57.308→46.726秒，下降18.466%；
92原子最大配对坐标差异3.386e-4Å，同rocSOLVER重复最大1.472e-3Å，误差与范围
见[新配对验收](../reports/g2-ase-bfgs-m5-2-e2e.md)。原型/11产品源/版本锁hash仍不变。
**M5-2隔离候选验收完成**；没有产品注册、M5整体完成或生产吞吐结论。
每sweep显式标量host读取仍存在；不恢复已知真实简并DC故障路径，不引入clamp。
原型失败/修正、参考eigh-test介绍的适用性和剩余范围见[M5-2报告](../reports/g2-ase-bfgs-m5-2.md)。
用户随后明确要求取消147/285维与B≤4限制，原受限接入方案撤回；下一步
[M5-3动态尺寸与Batch扩展](ASE_BFGS_M5_3_PROPOSAL.md)已由用户确认拆为A/B验收。
M5-3A在 `wangleping/perf-ase-bfgs-m5-3a` 完成隔离原型动态n/B、64位偏移、
完整配对launch、设备稳定排序与可选chunk；无147/285或B≤4入口限制。
CPU oracle、HCU旧19+新32矩阵及B8/16/32、晚块失败/stream、旧原型逐位对照、
新11体系十步ASE/restart通过；CPU专项104+105、HCU Framework14通过。
8个含锂独立CIF（2–160原子）CPU MLIP+ASE短步通过；8个T2独立CIF解析为
184原子/561维B8，共享HCU力的ASE CPU数学与独立HCU短步通过，未完整收敛。
T2独立CPU MLIP三步对照超时124，保留未通过边界。三进程×7样本真实H分解
窗口下降46.84%/47.07%；合成561 B8慢44.80%，不登记普遍加速。
旧46/92完整CSP新配对各一次均收敛（233步/234求值）；513维trace有HCU radix
排序，无数值矩阵D2H，标量同步仍存在。11产品源/锁不变，原型未注册，全部工作
未提交/集成/推送。见[M5-3A报告](../reports/g2-ase-bfgs-m5-3a.md)及
[动态合同](HIP_BFGS_JACOBI_DYNAMIC_CONTRACT.md)。M5-3A不同宽组仍串行。

M5-3B随后确认并完成隔离比选：32矩阵五输出逐位、9真实组数值门，11体系10步ASE，
Framework24组合/CPU104+105通过。三进程×7微基准Li分解2stream少33.24%，混合少
11.50%；4stream未额外收益，原生异构未优于2stream。实际kernel双队列重叠43.910ms。
Li/T2各三轮fresh MPA-0 medium变胞短轨迹，1FIRE+10BFGS/12主模型调用、3步ASE及
暖机后21更新样本：Li217.380→170.806ms（少21.43%）；T2不声明加速，均非完整收敛。
独立CPU MLIP Li门通过，T2共享力CPU数学不算模型CPU/HCU门；末态位置差1.719e-8 Å。
11产品/原M5-3A源/锁hash不变，未产品注册/提交/集成/推送。见
[M5-3B报告](../reports/g2-ase-bfgs-m5-3b.md)和[隔离合同](HIP_BFGS_HETEROGENEOUS_PROBE_CONTRACT.md)。
M5-3B当时建议多宽2stream/单宽batched；随后用户确认[M5-4产品接入](ASE_BFGS_M5_4_PROPOSAL.md)。
新增显式 `hip.bfgs-ase-jacobi-streams-v1`，无经验n/B白名单，默认和`hip`选择保持；
单宽batched、多宽同进程双worker/双stream。最终CPU668 passed/3既有skip，HCU
Framework24/native7及32矩阵/9真实组/11体系十步ASE门通过，既有rocSOLVER39回归通过。
修正零轮对角谱中心化抵消极小非零值的边界，迭代kernel不变。真实CSP与性能
复验14个限时作业均退出0；46/92原子CSP再次全收敛（233阶段步/234主调用），
Li/T2短步及Li独立CPU MLIP通过。三进程同输入微基准Li双stream分解少33.53%/
完整math少23.45%，单宽T2不获并行收益、合成189 B32有退化，不进auto。
最终证据见[M5-4报告](../reports/g2-ase-bfgs-m5-4.md)和[合同](HIP_BFGS_JACOBI_STREAMS_CONTRACT.md)。
本轮依然未提交/集成/推送；支持仅已准入FP64/eager HCU构建，生产全目录/auto/其他设备待另行确定。
随后用户确认最终交付审阅：修正Batch有限方向但步长范数溢出的零步提交遗漏，
补CPU/两个HIP后端的双宽组失败原子性。最终CPU669 passed/5 skipped（3既有Warp、
2CPU环境HCU参数，后二项已实际HCU通过），HCU Ops18/Framework24及11体系十步
ASE通过；46/92原子MPA-0 medium CSP再次联合收敛233阶段步/234主求值，ASE H
2.552e-11。两个包wheel/sdist和独立解包CPU/HIP JIT smoke通过，来源/SPDX核对保持。
见[审阅报告](../reports/g2-ase-bfgs-delivery-review.md)、[使用指南](ASE_BFGS_USAGE.md)、
[提交方案](ASE_BFGS_DELIVERY_PLAN.md)。仍未暂存/提交/合并/推送；待确认四组本地
DCO sign-off提交。历史性能证据和未覆盖范围不变。
2026-10-11用户确认后完成四组本地收口：Ops `d68ea01`、Framework `bf52bca`、
验证记录 `93955b0`，第四组为本段所在的工程文档提交。120个文件按原清单归组，
产品/测试/probe与审阅内容一致，第四组仅更新交付状态。完整SHA及每组文件见
[交付记录](ASE_BFGS_DELIVERY_PLAN.md)；尚未合入develop/推送/发布。支持范围保持，
下一阶段本地develop集成待用户确认；前述“未提交”为各历史阶段当时状态。
本轮阶段编号以[已确认计划](ASE_BFGS_DEVELOPMENT_PLAN.md)为准，覆盖历史 BFGS 队列
编号；不改上游锁、不将本地新增功能记为 UBF、每阶段总结后再确认下一阶段。

已确认 HIP 邻居分阶段任务：M1 已收口实现 `fecbc39`；M2 MATRIX→COO 与 MACE 单点已收口实现 `3e394d7`；
M3 HIP邻居变胞 FIRE/FIRE2 短步/预算补位窄 gate 已完成，已收口实现 `6b1680a`，本次随提交链合入 `develop`。
M4实测和workspace实验完成，候选未达门槛已撤回并收口；M5批量metadata完成窄验收、实现收口 `e2d569c`；M6查询协作实现及窄验收完成并收口；M7签名收口 `00637c0`；M8签名收口 `04f80bd`；M9 probe移动判定修正和两组HCU复验完成，以 `fa1259b`/`b549558` 收口。M1–M9 本次整体合入 `develop`。
继续暂停 M2 planner、NPT/NPH、DomainParallel、
生产 Triton/HIP dynamics 及未经确认的邻居能力扩展。

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
