# 当前开发状态

## 当前权威重构计划

- 后续 backend 架构重构以 [`docs/BACKEND_PLATFORM_PIPELINE_PLAN.md`](BACKEND_PLATFORM_PIPELINE_PLAN.md)
  为准：`PlatformFingerprint → ImplementationRegistry → BackendProfile → PipelinePlanner →
  Frozen BackendPlan`。
- M1 已于 2026-09-09 完成；当前先完成候选 `team/dev-baseline-v0.1` 的团队基础开发版本，
  M2 尚未开始。`backend=None` 的 legacy 语义、显式 `auto` 策略、operation-specific neighbor
  strategy、单次解析和 checkpoint plan hash 是已锁定的设计约束。
- 本文件后面的历史记录仍保留作为证据；若历史“下一步”与上述计划冲突，以该计划和最新
  交接记录为准。

## 交接初始状态（历史，2026-09-05）

- 交接日期：2026-09-05。
- 已有：项目目的、架构建议、功能兼容策略、探针计划和服务器初始化说明。
- 未完成：服务器盘点、上游提交锁定、代码导入、环境构建、任何 DCU 算子与端到端测试。
- 目标 DCU 型号/卡数/显存/互联：未知，待本机核查。
- DTK/PyTorch/Triton/HIP/通信库组合：未知，待本机核查。
- 用户优先模型与数据集：未指定。默认探测 MACE，按真实依赖可用性选首个模型。
- 人力、机器独占条件、发布时间：未确定。历史工期仅供估算。

## 下一步（首轮启动时的历史指令）

从项目根目录启动 Codex，执行 AGENTS.md 的首轮任务，首先检查现有环境与两个 external clone。首轮无需请求用户重新描述项目背景。

## 后续每轮更新格式

```text
日期与执行环境：
当前阶段/任务 ID：
根仓库及两个上游 SHA：
本轮修改与对应特性 ID：
运行命令、设备、退出码：
验证结果及证据相对路径：
尚未验证/阻塞项：
下一个可独立执行的任务：
```

使用追加记录或保留历史摘要。不要把 planned、import 成功、单卡通过、DDP 通过、域分解通过混写成一个“支持”。

## 当前轮执行计划（2026-09-06）

按小步闭环推进，不把短探针证据写成完整特性支持：

1. A2、A1、A3、A6 及 FIRE2+skin 功能契约已完成并保留 CPU/HCU 证据；A7 compile 限制已登记。
2. 上游 reference 行为回归继续按模块解除隐式 Warp 阻断；本轮已覆盖 observer、周期 utility、safety/freeze、bias、StageTimingHook、sink 和 DemoDynamics/FusedStage 编排。
3. HCU 计算允许共享；已通过项目环境脚本在主机权限终端完成 utility/periodic、safety/freeze/bias reference 的 HCU probe 与上游回归。受限沙箱仍可能在 Python 输出前报告 `No HIP GPUs are available`，后续 HCU 证据统一使用设备节点可见的主机权限终端。
4. A4 已完成 periodic reference 的 device-side scatter 及 CPU/HCU 规模回归；下一步单独评估 no-PBC 装配向量化，完整端到端 profile 仍不以共享 HCU 读数决定 cell-list、Triton 或 HIP 方案。
5. NVT/Langevin、Nose-Hoover/NPT、变胞 stress、checkpoint/restart 和生产 Triton/HIP 仍保持独立排期，不在本轮 reference 行为回归中混入。

当前状态：A2 已完成；A1 mixed/single 三路 smoke 和 HCU 收敛对照已通过；A3 窄 slice 契约与快照已刷新；A6、FIRE2+skin 功能契约及 A7 compile 边界探针已完成 CPU/HCU 或 CPU 证据；A4 已完成 Tier 1 前邻居基线，并完成 periodic full-list device-side scatter 的 CPU/HCU 语义回归与 92 原子 batch=32 对照；完整规模重测、no-PBC 向量化和干净端到端性能结论仍未完成。

## 每轮 DoD 清单模板

- [ ] 当前快照与时间线已更新。
- [ ] `FEATURE_COMPATIBILITY.yaml` 已登记本轮窄 slice，未扩大为完整支持。
- [ ] CPU 与 HCU 命令、设备、退出码和结果已记录。
- [ ] 影响上游导入文件的改动已登记 `docs/UPSTREAM.md`。
- [ ] 未验证项、失败模式和静默回退风险已记录。
- [ ] 下一个可独立执行的小任务已明确。

## 当前快照（2026-09-09）

- N0 能力探针已完成：主机单卡 Triton vector-add 通过（BW200/UBB BW1000，冷启动约 0.519 s，预热稳态约 24.6 μs/次）；主机双卡 RCCL/NCCL all-reduce 和双向 P2P 通过。证据见 `reports/g0-capability-probes.md`。这只解除 Triton 基础编译/执行和 RCCL 原语的架构未知，不代表生产 kernel、LJ ownership 或 DomainParallel 已验证。
- 环境加载修正：`scripts/activate_hygon_env.sh` 在 bash 使用 `/opt/dtk-26.04/env.sh`，在 zsh 使用 `/opt/dtk-26.04/env.zsh`；沙箱内 `/dev/kfd` 不可见，GPU 结果均来自主机权限探针。
- 代码基线：Torch reference 的 Batch/邻居/PBC/LJ/skin 纵向切片已在 CPU 和部分 BW200/gfx936 HCU 通过；MACE wrapper 本地 checkpoint 路径修复已实现并有回归测试。
- G2 收敛：ops 现有单一 capability registry，framework 的邻居、LJ、固定晶胞 dynamics、observer 与 periodic 计算均委派给它；`None`/`warp` 保留 legacy 路径，`auto` 只选择已验证 capability 并报告选择。`LevelStorage` 的 Torch backend 现明确为数据层默认，Warp 仅为显式对照。
- no-PBC 邻居 Tier 1：逐原子/逐边 Python 写回已替换为逐 system 的 device-side `nonzero`/`bincount`/row-rank/scatter。CPU 契约、synthetic `[46,92]` probe、BW200 HCU 0 ops 回归和 framework Batch 写回 smoke 均已通过；固定阶梯性能基线和 cell-list 仍未完成。
- MACE 证据：用户缓存的 `MACE-OFF23_small.model` direct model 在探索环境和项目 `.venv` 的 CPU/HCU 通过；两个 `perf_46` CIF 的 framework `MACEWrapper + compute_neighbors(torch_reference)` batching 在探索环境和项目 `.venv` 的 CPU/HCU 通过。项目环境批次为 `batch_ptr=[0,46,92]`、1754 条边、跨体系边 0。Tier 1 邻居装配后重新完成真实 32×92 周期 MACE/FIRE2 固定晶胞 HCU 弛豫（`max_steps=2000`，834 步、32/32 收敛，79.54 s，最终最大 `fmax=0.0099955`），以及异构 `[46,92]` 三步 MACE/FIRE2→HostMemory 轨迹（读回 `batch_ptr=[0,46,138,184,276,322,414]`、无跨体系边）。详细报告见 `reports/g1-mace-wrapper-batch.md`、`reports/g2-mace-fire2-batch32.md`、`reports/g2-mace-fire2-batch32-tier1.md` 和 `reports/g2-mace-fire2-heterogeneous-trajectory.md`。
- 当前未验证：低干扰的统一 no-PBC/periodic full-list 阶梯性能、cell-list、变胞/stress、长轨迹 inflight occupancy、checkpoint/restart、训练 wrapper 混合二阶梯度、生产 cuEquivariance/Triton/HIP kernel、多卡域分解、PhysicsNeMo profiling 和 DomainParallel。periodic Tier 1 与 no-PBC 窄功能证据已存在，但 HCU 绝对时间仍受共享负载影响，不能替代下一轮统一基线。
- 依赖检查：北外镜像 dry-run 解析到 `mace-torch==0.3.15`、`e3nn==0.4.4`、`matscipy==1.1.1`、`ase==3.29.0` 等 45 个包，随后安装到项目 `.venv`；另补齐 framework 基础依赖，Torch `2.9.0+das.opt1.dtk2604`、Triton `3.3.0+das.opt1.dtk2604.torch290` 未被替换。PhysicsNeMo 未安装，单进程路径由可选导入保持可用。
- 环境规格已补齐：项目 `.venv` 现在包含 `pytest==8.4.2`、`pytest-asyncio==1.4.0`；直接输入见 `configs/hygon-reference.in`，当前主机精确冻结见 `configs/hygon-reference-lock.txt`，冻结脚本为 `scripts/freeze_hygon_env.sh`，报告快照见 `reports/probe-environment-freeze.txt`。冻结中的 Torch/Triton URI 是主机本地海光 wheel，换机时必须先提供同版本 wheel；PhysicsNeMo 不属于 Hygon reference 安装集。
- 特性状态已复核：`status` 表示完整目标契约的实现阶段，`verification.dcu_status` 表示已列出的 HCU 证据范围；因此 MACE、LJ、neighbors.topology 的完整条目仍是 `planned`，但其 reference 子路径为 `partial`，`neighbors.skin_rebuild` 为 `implemented/partial`。本轮纠正了 neighbors.topology 的 HCU 状态，并在 `FEATURE_COMPATIBILITY.yaml` 写明两轴语义，避免把窄 reference slice 写成完整特性通过。
- backend registry 收尾审计已完成：`dynamics/_ops` 的 `_select_backend` 为零，所有 reference dispatch 文件使用中央 resolver；Warp-only 模块、COO 写回分支、分布式能力保护和 `StageTimingHook.timer_backend` 已分别归类。审计报告见 `reports/g2-backend-registry-audit.md`。
- 统一 benchmark 已完成 HCU 小范围 smoke：source DTK 26.04、`HIP_VISIBLE_DEVICES=0`、设备 `BW200, UBB BW1000`；periodic/no-PBC full/half 的 46/92 原子和 batch=1 均退出 0，边数、跨体系边、周期 image shift 和 `source_pbc/effective_pbc` 元数据检查通过。periodic 46/92 steady 约 `3.256/3.686 ms`；这只是 harness/语义 smoke，不是完整规模性能基线。报告见 `reports/g2-unified-reference-benchmark-hcu-smoke.md`。
- 统一 benchmark 的 HCU 单体系阶梯已通过：`perf_46/92/184/368` 的 periodic full、no-PBC full/half 全部退出 0；periodic steady 为 `3.243/3.702/5.753/13.969 ms`，no-PBC full 为 `1.853/1.852/1.859/1.880 ms`。首个 periodic cold 样本包含约 `4.209 s` 的 HCU context/kernel 初始化，不与 steady 混比。该证据仍不覆盖 batch 阶梯或端到端 MACE/FIRE2。报告见 `reports/g2-unified-reference-benchmark-hcu-scale.md`。
- 统一 benchmark 的 HCU `perf_92` batch 阶梯已通过：batch `1/4/8/16/32` 的 periodic full steady 为 `3.768/9.789/17.819/33.571/65.577 ms`，no-PBC full 为 `1.856/3.931/6.705/12.232/23.316 ms`；batch=32 的 periodic 边数 `54,760`、steady `0.065577 s` 与既有 Tier-1 结果连续。该证据仍不覆盖 MACE/FIRE2 端到端。报告见 `reports/g2-unified-reference-benchmark-hcu-batch92.md`。
- 统一 benchmark 的 HCU periodic `[46,92]` MACE/FIRE2 固定晶胞 100 步已通过：总耗时 `11.1279105 s`，100 步平均 `0.1112791 s/step`，最后邻居边数 `3,284`；`StageTimingHook` 的 `BEFORE_COMPUTE→AFTER_COMPUTE` total 为 `10.439570 s`，但最大单样本 `6.586 s`、std `0.671 s`，因此只能作为共享 HCU 下的端到端相对基线。报告见 `reports/g2-unified-reference-benchmark-hcu-e2e.md`。
- M1 registry 已完成：`ImplementationRegistry` 记录 implementation ID/family/strategy，cell-list 改为 `backend="torch_reference", method="cell_list"`，旧未发布名称显式失败；`None`/Warp 与 auto dense default 语义不变。CPU ops/framework 为 `25 passed`/`22 passed`；BW200/gfx936 HCU ops `25 passed`、M1 framework strategy `2 passed`。证据见 `reports/g2-backend-registry-m1.md` 与 ADR 0006；真实规模性能、周期 cell-list、BackendProfile 和 planner 仍未完成。
- M1 follow-up 已完成：FIRE/FIRE2 拆为独立 operation ID（`torch_reference.fire-v1` / `torch_reference.fire2-v1`）；LJ、固定晶胞 VV/FIRE/FIRE2、periodic、kinetics、segmented reduction 和 observer 均由 framework 解析一次 `BackendSelection` 后传给 dispatcher。变胞 FIRE/FIRE2 仍冻结 legacy Warp，显式 Torch reference 因缺少 `variable_cell` capability 明确失败。CPU selection propagation `3 passed`，reference/observer/periodic/LJ slice `103 passed, 1 deselected`（合并为 `106 passed, 1 deselected`），state lifecycle `22 passed`，import/reference `15 passed`；报告见 `reports/g2-backend-registry-m1-dispatch-propagation.md`。本轮没有新增 HCU 证据。
- 项目环境 pytest 基线：ops reference `10 passed`、framework optional-import/neighbor Hook `11 passed`，均退出码 `0`；两包测试需分开启动以避开上游都使用顶层 `test` 包名造成的 `ImportPathMismatchError`。详细命令见 `reports/g1-project-reference-pytest.md`。
- 可重建性检查：`uv pip sync --dry-run --python .venv/bin/python ... configs/hygon-reference-lock.txt` 在北外镜像上解析并核对 `77 packages`，退出码 `0`，显示 `Would make no changes`。
- skin/rebuild 当前进展：Torch reference 已对 Batch 中变化的 system 做 eager 局部重建，并保持全局索引、MATRIX/COO 写回和未变化 system 的缓存；两体系 CPU/HCU probe 均通过，报告见 `reports/g1-skin-rebuild-batch-reference.md`。Hook staging 已有自动容量处理，算子层仍保留显式 overflow 防御。
- N1 staging 容量契约已完成窄 reference slice：Hook 支持 16 对齐 grow-and-retry、idle shrink、override floor、异构 Batch 和 6 次交替 skin rebuild；framework Hook 回归 `16 passed`，CPU/HCU probe 均退出码 `0`，报告见 `reports/g1-neighbor-capacity-reference.md`。算子层显式 `NeighborOverflowError` 仍保留。
- 当前未完成：PBC 容量压力、周期 half-list、生产 cell-list、规模化性能和异步 rebuild；这些不因 N1 reference 通过而提前标记为 verified。
- N2a 首个切片已完成：新增顶层 `nvalchemi._dynamics_reference.velocity_verlet`，不触发 `nvalchemi.dynamics` 或 Warp，覆盖原位 position/half-kick/final-kick、异构 per-system `dt`、float32/64 和显式输入检查。项目 `.venv` CPU 测试 `5 passed`；同一项目 `.venv` 在 source DTK 26.04、BW200/gfx936 HCU 探针退出码 `0`，最大位置误差 `6.94e-18`、final velocity 最大绝对值 `0.0`。证据见 `reports/g2-dynamics-reference-velocity-verlet.md`。
- N2a 第二个切片已完成：新增顶层 `nvalchemi._dynamics_reference.kinetics`，用 Torch `index_add_` 实现异构 Batch 的 kinetic energy 和 `3N` temperature。项目 `.venv` CPU 回归 `7 passed`（含 VV 与 kinetics）；同一项目 `.venv` 在 source DTK 26.04、BW200/gfx936 HCU 探针退出码 `0`，动能 `[7.0, 6.0]`、温度 `[27077.2089507397, 46418.07248698234]`，无 Warp 导入。证据见 `reports/g2-dynamics-reference-kinetics.md`。
- N2a 第三个切片已完成：新增顶层 `nvalchemi._dynamics_reference.fire`，实现固定晶胞、异构 `batch_idx` 的 FIRE `fire_step`/`fire_update` 与 FIRE2 `fire2_step_coord`；项目 `.venv` CPU reference 测试 `4 passed`，同一项目 `.venv` 在 source DTK 26.04、BW200/gfx936 HCU 探针退出码 `0`，FIRE/FIRE2 结果与 CPU 一致且无 Warp 导入。证据见 `reports/g2-dynamics-reference-fire.md`。
- N2b 第一小步已完成：`nvalchemi.dynamics`、integrators/optimizers/hooks 命名空间和 `_bridge` 改为惰性/按需 Warp；公共 VV/FIRE/FIRE2 wrapper 接受显式 `backend="torch_reference"`/`"auto"`，默认 `None` 仍走 Warp；NVE 保存 backend 并传递给 VV。项目 `.venv` 导入边界测试 `3 passed`，上游 `test/hooks/test_stage_timing_hook.py` 已可收集 `43 tests`。证据见 `reports/g2-dynamics-public-boundary.md`。
- FIRE2 参考语义已补齐两层证据：锁定上游 NumPy `_fire2_reference_step` 的 AST oracle 对照在异构 `batch_idx`、float32/64 下 `2 passed`；公共 `FIRE2` 驱动 `NeighborListHook(backend="torch_reference", skin=0.5)` 的局部重建契约在 CPU/HCU 各 `1 passed`。后者验证 sub-skin 缓存复用、单体系刷新、Batch 边界和邻居边归属，不含性能结论。报告见 `reports/g2-fire2-upstream-reference.md`、`reports/g2-fire2-skin-functional.md`。
- A7 compile 边界探针 `probes/dynamics_reference_compile.py` 在项目 `.venv` CPU/Torch 2.9.0 运行退出码 `0`：kinetic energy custom op 在 `fullgraph=True` 通过；FIRE2 和 temperature 在严格 fullgraph 下失败，默认 `fullgraph=False` 分别产生 4/1 次 graph break。该结果已登记为 reference 的已知限制，不改变 eager 正确性或当前 HCU 性能暂停。报告见 `reports/g2-dynamics-reference-compile.md`。
- 上游 reference 行为测试继续扩展：`test_single_loop.py` 的 `TestConvergenceHook`、`TestFusedStage`、`TestFusedStageSubstageHooks` 共 `55 passed`，`test_sinks.py` 的 DataSink/Drain/HostMemory 子集共 `17 passed`；两组在项目 `.venv` CPU 和 DTK 26.04 的 BW200/gfx936 HCU 均通过。设备/stream/pipeline 额外子集 HCU `31 passed`，CPU 的 `27 passed, 4 failed` 属无 GPU 默认设备与测试打补丁时序差异，已记录在报告。`test_state_management.py::TestMakeNewState` 中 NVE/FIRE/FIRE2 通过，Langevin/Nose-Hoover/NPT 六项仍在 Warp import boundary 阻断。报告见 `reports/g2-upstream-fusedstage-state-sinks-reference.md`。
- `test_state_management.py` 已为固定晶胞 NVE/FIRE/FIRE2 构造器接入 `NVALCHEMI_TEST_BACKEND`：显式 reference 的 lazy init、state shape、Batch invariant、partial removal 和 `_make_new_state` 子集 CPU `20 passed`、HCU `20 passed`；未设置变量的三项反向检查仍在 Warp custom-op 边界失败。该测试载体补丁登记为 `docs/UPSTREAM.md` 的 LP-010。
- 上游 `test_sampler.py` 的 `SizeAwareSampler` 全部 `51` 项在项目 `.venv` CPU（`0.40 s`）和 DTK 26.04、BW200/gfx936 HCU（`0.13 s`）通过；覆盖异构大小分箱、原子/边/批大小预算、初始批次、replacement 和耗尽语义。报告见 `reports/g2-upstream-sampler-reference.md`。这仍是 sampler 控制流证据，不等于真实 MACE 长轨迹 inflight 或性能通过。
- 上游 observer hook 子集发现并修复了 `_segmented_max`/kinetic 的隐式 Warp 依赖：`scatter_reduce_per_graph`、`LoggingHook`、`EnergyDriftMonitorHook` 现在支持显式 `compute_backend`，并可由 reference dynamics workflow backend 继承。设置 `NVALCHEMI_TEST_BACKEND=torch_reference` 后，CPU `44 passed, 33 skipped`、HCU `77 passed`；公共 `FIRE(backend="torch_reference")` 不显式设置 observer backend 的 CPU/HCU 单步也通过且未加载 Warp。未设置时单测仍在 Warp 边界失败。报告见 `reports/g2-upstream-observer-reference.md`。
- 当前快照已包含：公共固定晶胞 NVE/FIRE/FIRE2 reference 的完整短流程、真实 MACE 单卡组合、32×92 批量 FIRE2 弛豫、异构 `[46,92]` 的最小 HostMemory 轨迹和短 inflight 补位，以及上游 FusedStage/状态/HostMemory/Sampler/inflight/observer/StageTimingHook/GPUBuffer/ZarrData 的 CPU/HCU reference 子集；safety/freeze/bias 的 CPU 与 HCU 参数化回归也已通过。`test_hook_utils.py` 与周期 hook 现在可在无 Warp 环境收集并运行；周期 helper 的 Torch reference 已实现，CPU utility `30 passed, 5 skipped`、periodic `15 passed, 11 skipped`，在正确加载 DTK 26.04 的主机权限 HCU 上 probe 退出码 `0`，utility+periodic 上游回归 `61 passed`（含 CUDA compile smoke）。受限沙箱仍会因隐藏 `/dev/kfd` 报 `No HIP GPUs are available`，不作为 HCU 能力失败。Safety/freeze HCU `57 passed`、BiasedPotentialHook HCU `22 passed`、StageTimingHook `42 passed, 1 skipped`（skip 仅 nvtx）、完整 `test_sinks.py` `56 passed`。G2 registry/no-PBC 收尾和 32×92 原参数复跑已经落盘；下一步转入统一 no-PBC/periodic full-list benchmark，不把这些窄 slice 写成完整生产 dynamics 支持。

## 2026-09-05：G0 初始化审计

- 指令来源已核对：用户指定 SHA 优先；读取根 AGENTS、PROJECT_HANDOFF、STATUS、START_HERE。主仓库初始无提交，交接文件未跟踪，无产品代码。旧探索仓库仅只读参考。
- 锁定 framework `4dfe3723def34df3fadb245981081ccf8c94c257` / ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。版本交集及 97 条显式跨包导入静态审计符合源码基线导入门槛，不代表 DCU runtime 兼容。
- 新增 UPSTREAM/LOCK、FEATURE_COMPATIBILITY（21 组）、ENVIRONMENT、PROBE_PLAN、ADR 0001、源码完整清单和可重跑 probes。未移植任何产品算子。
- 本机 8×BW200（65520 MiB/卡），驱动 6.3.30-V1.4.1a，DTK 路径 26.04，HSW 互联；初次盘点全部满载，设备计算记录为资源阻塞。
- 创建 uv 隔离 .venv，使用指定 Torch/Triton wheel，禁止替换系统环境。镜像 403 后使用 PyPI；实测 NumPy ABI 问题，约束为 1.26.4。
- 命令、退出码及数值见 reports/g0-validation.md；详细重跑命令见 PROBE_PLAN。GPU、MLIP、原测试、NVIDIA fixture 均无本轮通过证据。
- 下一可独立任务 G1：先隔离 Batch/AtomicData 的 import-time Warp/PhysicsNeMo 耦合，再实现 Torch reference 邻居→LJ→NVE 纵向链。对应兼容测试保护异构两体系 ID/索引、half/full/PBC pair 集合、FP64 解析 E/F、容量溢出明确失败、短轨迹能量漂移与轨迹输出。保留上游测试，再按功能回归。
- 有分配卡后依次运行 P01/P02/P03 和双卡 P07；指定可信 MACE checkpoint 并审计依赖后运行 P06。通信探针不能替代两卡 LJ 域分解验收。
- 导入完成：`52ffa2d` framework / `88aa209` ops，不 squash；`probes/verify_import.py` 退出 0，源码 tree 精确相同、祖先历史可达、无嵌套 .git。工作分支 `codex/g0-initialization`，未推送。
- 用户补充确认目标架构 gfx936；更新 HIP 编译命令并重编译，gfx928 不作为适配目标。

### 本轮停止点（用户要求分步推进）

- G0 源码审计和完整历史导入完成；不展开 G1。
- 本轮小任务已完成：从探索项目环境复用 NumPy 1.26.4 到项目 `.venv`，未替换 Torch/Triton；CPU/NumPy 互操作及原有梯度、segment、FFT 检查通过。
- 下一步仍不展开 G1；先等待用户安排，再单独规划 Batch/AtomicData 的 import-time 依赖审计。GPU 探针继续等待空闲卡分配。
- gfx936 HIP 最终编译退出 0，无编译警告；无设备执行证据。

### 2026-09-05：G1 数据导入链审计

- `PYTHONPATH=packages/framework:packages/ops /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -c 'import nvalchemi.data.atomic_data'` 独立进程退出 1：`ModuleNotFoundError: No module named 'warp'`。
- 失败链为 `nvalchemi.data.__init__ → Batch → level_storage → buffer_kernels → import warp/wp.init()`；`AtomicData` 顶层本身未直接导入 Warp，但包初始化使其间接受影响。`nvalchemiops` 及 `nvalchemiops.torch` 也在顶层导入 Warp。
- `import nvalchemi`、`import nvalchemi._optional` 可通过；PhysicsNeMo 可发现但不是本次数据 API 失败原因。详细证据见 `reports/g1-data-import-audit.md`。
- 本步只完成审计，未改产品代码、未安装 Warp、未启动 GPU。下一小步：设计并实现保持公共导入路径的最小 Warp 隔离边界，再补 import smoke 测试。

### 2026-09-05：存储后端协议第一步

- 新增 `packages/framework/nvalchemi/data/storage_backend.py`，只依赖 Torch，冻结现有 uniform/segmented fit mask、masked put、defrag 和 segment expansion 操作签名。
- 新增 ADR 0002，确定 `AtomicData`/`Batch`/LevelStorage 数据模型保留，Warp 仅作为延迟加载执行后端；Torch reference 先定义语义，Triton/HIP 面向 gfx936 逐项替换。
- 协议尚未接入 `LevelStorage`，因此当前 `nvalchemi.data` 的 Warp import-time 阻断仍存在；本步未改变运行时行为。下一小步：实现 Torch reference backend 并补充无 Warp import smoke，再接入 LevelStorage。

### 2026-09-05：Torch reference backend

- 在 `packages/framework/nvalchemi/data/storage_backend.py` 实现 `TorchStorageBackend`，覆盖 uniform/segmented fit mask、masked put、defrag、segment expansion；保持原位 mutation、索引顺序、batch pointer 和容量语义，不导入 Warp。
- `.venv/bin/python probes/storage_backend_probe.py` 退出 0，验证 uniform/segmented copy、空 segment、defrag 尾部清零、pointer 更新和 index expansion；源码编译检查通过。
- 本步没有接入 `LevelStorage`，也没有改变 `nvalchemi.data` 当前导入行为。下一小步：在 LevelStorage 中注入 backend，并将 Warp 实现改为延迟加载；先补无 Warp import smoke，再运行存储契约测试。

### 2026-09-05：LevelStorage 后端注入

- `level_storage.py` 已移除顶层 `buffer_kernels` 导入，LevelStorage 的 fit mask、put、defrag、segment expansion 改为调用 `self._backend`；存储构造器、copy/clone/select 会保留 backend。
- 无 Warp 环境下 `from nvalchemi.data import AtomicData, Batch` 退出 0，构造的 LevelStorage 默认 backend 为 `torch`。这解决了原有 import-time Warp 阻断，但遗留 Warp helper 声明仍待移入专用后端模块。
- 回归：`packages/framework/test/data/test_batch.py` 为 90 passed、4 skipped、1 failed。唯一失败是 `test_pin_memory` 在无 HIP GPU 进程中报 `No HIP GPUs are available`；Batch 构造、异构索引、put/defrag 相关用例通过。
- 本步未实现 Triton/HIP、未运行 GPU；下一小步为补充正式的无 Warp import smoke 和后端契约测试，然后再清理遗留 Warp helper。

### 2026-09-05：Warp backend 隔离

- 旧 `level_storage.py` Warp helper 已移入 `packages/framework/nvalchemi/data/warp_storage_backend.py`，由 `WarpStorageBackend` 包装现有 `buffer_kernels`，并保留 segment expansion 的 Warp kernel。基础 `level_storage.py` 不再包含 Warp import、初始化或临时 stub。
- 无 Warp 环境下基础导入仍通过：`from nvalchemi.data import AtomicData, Batch`，且 `warp` 不出现在 `sys.modules`；显式导入 `WarpStorageBackend` 按预期 fail-fast。
- Batch 回归仍为 90 passed、4 skipped、1 failed（`pin_memory`：当前无 HIP GPU）；清理 Warp helper 未引入新的失败。完整输出：`artifacts/g0/test_batch_after_warp_split.txt`。
- 本步未实现 Triton/HIP 或自动 backend resolver。下一小步：补正式 import smoke/StorageBackend contract tests，并检查多属性 put 的共同复制掩码语义。

### 2026-09-05：探针环境 NumPy 修复

- 从 `/home/wangleping/codes/nvalchemi-toolkit/.venv` 复用 NumPy 1.26.4，项目 `.venv` 当前实际版本为 1.26.4；没有重新安装或替换海光 Torch/Triton。
- `.venv/bin/python probes/torch_probe.py --device cpu` 退出 0；新增 NumPy↔Torch 转换检查通过，FP64 一阶/二阶梯度、segment、FFT 仍通过。证据：`artifacts/g0/torch_cpu_numpy1264.json`、`artifacts/g0/torch_cpu_numpy1264.stderr`。
- stderr 只有环境日志 `Could not open /var/log/hylog/.`，不再有 NumPy ABI 警告。环境冻结已更新为 NumPy 1.26.4。

### 2026-09-05：pin_memory 失败原因复核

- 之前的 Batch 回归在受限探针沙箱中运行；该进程看不到 `/dev/kfd` 和 `/dev/dri`，因此海光 Torch 报告 `torch.cuda.is_available()=False`、`device_count=0`，上游 `test_pin_memory` 随之失败为 `No HIP GPUs are available`。
- 在主机设备可见的 shell 中显式执行 `source /opt/dtk-26.04/env.sh`，项目 `.venv` 识别 8 张 `BW200, UBB BW1000`（HIP 6.3.26093），一个元素的 `pin_memory()` 成功。该失败是探针执行环境阻断，不是 HCU 不能共享，也不是存储后端改动造成。
- 证据与可重跑命令见 `reports/g1-pin-memory-environment.md`；在相同主机环境中重跑 `test_pin_memory` 得到 1 passed。项目 `.venv` 尚未安装 pytest，单测暂用已有探索环境执行。后续 GPU 测试必须在可见 `/dev/kfd`/`/dev/dri` 且已加载 DTK 的作业环境中运行，并记录实际设备占用。

### 2026-09-05：存储后端契约与多属性复制

- `UniformLevelStorage.put` 与 `SegmentedLevelStorage.put` 现在只在第一个共同属性上决定复制集合；其余属性复用相同的目标槽位/segment offset 和复制掩码，避免多属性数据错位或掩码被清零。
- 新增 `packages/framework/test/data/test_storage_backend.py`：无 Warp 公共导入 smoke、uniform 双属性复制、segmented 双属性复制共 3 项全部通过。
- `test_level_storage.py` 96 passed；`test_batch.py` 在受限沙箱中仍为 90 passed、4 skipped、1 failed（pin_memory 环境阻断，已在上一条记录并在主机 DTK 环境单测通过）。证据：`reports/g1-storage-contract.md`。
- 本步仍是 Torch reference/CPU 验证，未宣称 gfx936 算子已验证；下一小步继续补充后端契约边界，再选择邻居→LJ 的最小纵向链。

### 2026-09-05：邻居→LJ Torch reference 纵向切片

- 新增 `probes/neighbor_lj_reference.py`，不导入 Warp，固定两个体系、无 PBC、dense neighbor matrix 的 full/half 语义，并用 Torch autograd 定义 LJ 力参考。
- CPU 和 `source /opt/dtk-26.04/env.sh` 后的 `HIP_VISIBLE_DEVICES=0` 单卡 HCU 均通过：能量 `-0.6236757013081533`，full/half 一致；batch 边界、pair 集合、每体系零总力均通过。
- 这是 reference oracle，不等同于 `nvalchemiops` 生产 API 已移植；PBC、cell-list、容量扩容、switching、virial、NVE 和双卡 ownership 仍未覆盖。证据：`reports/g1-neighbor-lj-reference.md`。

### 2026-09-05：Torch-first 后端策略与 ops reference slice

- 按 ADR 0003，当前阶段不实现 Triton/HIP kernel；先以海光 PyTorch 作为设备内 Torch reference 和正确性基线，后续只在真实 gfx936 基准显示瓶颈且能力/梯度满足时逐算子增加 Triton 或 HIP。选择不能由 `cuda` 字符串或语言偏好决定，必须记录实际后端和原因。
- 新增 `packages/ops/nvalchemiops/torch_reference.py`、`backend.py`、`torch_backend.py` 及 `packages/ops/test/torch/test_torch_reference_backend.py`。reference 模块不导入 Warp，dispatcher 的 `backend="auto"` 当前明确选择并报告 `torch_reference`；覆盖 no-PBC neighbors → LJ energy/force 的 full/half、异构 batch、COO、容量溢出和二阶梯度检查。已有 Warp-facing 子包改为在包边界显式调用 `initialize_warp()`，保留原 Warp 路径。
- `packages/ops` 基础依赖不再强制安装 Warp，Warp 放入显式 `warp`/Warp adapter extras，并增加 `torch-reference` extra；framework 的 uv dependency metadata 已同步。使用 PyPI 构建 wheel 成功，基础元数据只要求 NumPy。
- CPU reference/dispatcher 测试 `6 passed`、退出码 0；加载 `/opt/dtk-26.04/env.sh` 后，`HIP_VISIBLE_DEVICES=0` 的 BW200 HCU dispatcher 检查退出码 0，full/half 均报告 `torch_reference`，能量 `-0.6236757013081533`、力范数 `23.859458411523782`。详细命令和限制见 `reports/g1-torch-reference-backend.md`。
- 该 dispatcher 仍是独立 Warp-free 入口，尚未替换上游 `nvalchemiops.torch.neighbors`/LJ 默认公共 API；PBC、cell-list、NVE、Triton/HIP 和自动性能选择仍未实现。下一小步是审计上游调用方并接入不改变默认返回值的显式 reference backend 入口。

### 2026-09-05：framework.compute_neighbors reference 入口

- `packages/framework/nvalchemi/neighbors.py` 保留原有 `compute_neighbors` 公共入口，新增可选 `backend` 参数。`backend=None` 仍保持上游 Warp 默认路径，并将 Warp 相关导入延迟到该路径；`backend="torch_reference"` 和当前的 `backend="auto"` 通过 ops dispatcher 执行 Torch reference。未注册的 Triton/HIP 请求会明确失败，不会静默转 CPU。
- Torch reference 结果现在可以写回 framework 的 MATRIX 或 COO 邻居存储；Warp 默认路径仍调用上游矩阵到 COO 转换，Torch 路径使用等价的无 Warp 转换并对邻居容量做显式检查。PBC、cell-list、skin、动态重建和邻居 hook 尚未接入该 dispatcher。
- `packages/framework/test/models/test_neighbors_torch_reference.py` 在探索环境通过 `2 passed`；加载 `/opt/dtk-26.04/env.sh`、使用 `HIP_VISIBLE_DEVICES=0` 的 BW200 HCU 运行 MATRIX 路径通过，结果为邻居矩阵 `[[1], [0], [3], [2]]`、计数 `[1, 1, 1, 1]`。项目 `.venv` 运行 framework 测试仍被缺少 `plum` 阻断，未安装额外依赖。
- 证据已补入 `reports/g1-torch-reference-backend.md`，兼容清单的 `neighbors.topology` 已登记 framework 测试和该报告。下一小步：审计 `NeighborListHook` 与 LJ wrapper 的调用契约，确定 reference 入口的最小纵向接入范围。

### 2026-09-05：NeighborListHook 与 LJ wrapper 调用契约审计

- `NeighborListHook` 当前在模块导入时直接依赖 Warp-facing neighbors/rebuild 模块，并在 `_rebuild()` 中使用原地输出矩阵、`rebuild_flags`、`method`、cell-list/cluster-tile scratch 和 Verlet skin。当前 Torch reference dispatcher 返回新张量，且不支持这些参数、PBC 或动态重建，因此不能直接替换该热路径。
- `LennardJonesModelWrapper` 当前固定消费 MATRIX 邻居，顶层导入 `nvalchemi.models._ops.lj`；该 custom op 直接依赖 Warp，提供 analytic force、switching 和可选 virial/stress。Torch reference LJ 要求 `positions.requires_grad`，目前只覆盖 no-PBC、`switch_width=0`，没有 virial 输出。
- 探索环境的无 Warp 导入检查：`nvalchemi.hooks.neighbor_list` 和 `nvalchemi.models.lj` 均以 `ModuleNotFoundError: No module named 'warp'` 失败。这是尚未隔离的依赖边界，不是 HCU 设备运行结论。详细审计见 `reports/g1-neighbor-hook-lj-audit.md`。
- 下一小步：为 Hook 增加显式 `backend="torch_reference"` 的受限分支（先无 PBC、`skin=0`），再为 LJ wrapper 增加同样显式的 reference 分支和 energy/force contract tests；默认 Warp 参数和路径保持不变。

### 2026-09-05：NeighborListHook Torch reference 受限入口

- `NeighborListHook` 新增 `backend` 参数。默认 `None` 仍使用原 Warp 实现；显式 `"torch_reference"`/`"auto"` 绕过 Warp staging、rebuild 和 scratch，调用已有 Torch dispatcher 并写回 MATRIX/COO。`nvalchemi.hooks` 的 `WrapPeriodicHook` 改为按需导入，避免显式 reference Hook 导入时触发 Warp。
- reference 分支明确拒绝 `skin != 0`、PBC 和 `method` 选择；这些能力尚未移植，未做静默忽略或 CPU 回退。新增 `packages/framework/test/hooks/test_neighbor_list_torch_reference.py`。
- 探索环境测试 `6 passed`，覆盖真实 `@torch.compile` 入口、无 Warp 导入、异构 batch、MATRIX/COO 和受限条件失败。静态编译与 diff 检查通过；本轮未新增 HCU Hook 运行证据。
- 下一小步：给 `LennardJonesModelWrapper` 增加同样显式的 Torch reference 分支，先覆盖无 PBC、无 switching、无 stress 的 energy/force contract。

### 2026-09-05：LennardJonesModelWrapper Torch reference 受限入口

- `LennardJonesModelWrapper` 新增 `backend` 参数。默认 `None` 仍延迟加载 Warp custom op；显式 `"torch_reference"`/`"auto"` 使用 Torch dispatcher，并保持全局邻居索引、full/half 归约、per-system energy scatter 和返回 force 的符号约定。
- reference wrapper 明确拒绝 PBC、非零 `switch_width`、virial/stress 和 domain decomposition；LJ 模块导入本身不再强制 Warp。当前前向路径要求 Batch 已有 MATRIX 邻居数据，自动动态 Hook 接线尚未完成。
- 新增 `packages/framework/test/models/test_lj_torch_reference.py`：探索环境 `4 passed`。验证独立 FP64 LJ 公式、full/half 数值一致、总力守恒、energy gradient 与 force 关系，以及 unsupported 条件显式失败。本轮未新增 HCU wrapper 运行证据。
- `BaseModelMixin.make_neighbor_hooks()` 现在会把模型 backend 传给 Hook；默认模型仍使用 Warp。后端字符串集合目前是组件级显式能力边界，用来拒绝未知请求和避免静默回退；后续应收敛到中央 backend registry，避免各组件重复维护选择规则。由于该方法仍导入 Warp-backed dynamics stage，reference 模型的完整无 Warp 动态构造尚未完成。
- 验证策略已明确记录：上游 framework/ops 测试作为主回归来源，新增测试只补 DCU backend 选择、无 Warp 导入边界和上游未覆盖的适配约束；数值校验先用独立 FP64 解析结果与 Torch CPU reference，再在 HCU 上复核，ASE 只作为可选的独立交叉检查。当前 LJ 测试已覆盖独立公式、full/half、一阶梯度和总力守恒。
- 下一小步：隔离 `make_neighbor_hooks()` 所需的 dynamics stage 导入，再跑完整单卡邻居→LJ wrapper 链。

### 2026-09-05：reference 模型动态 Hook 接线与 CPU 纵向链

- 新增无 Warp 的轻量 `nvalchemi._dynamics_stage.DynamicsStage`；`nvalchemi.dynamics.base` 继续重新导出同一枚举，保留上游公共导入身份。`BaseModelMixin.make_neighbor_hooks()` 不再为了取得 stage 而触发 eager dynamics/Warp 包初始化。
- `LennardJonesModelWrapper(backend="torch_reference").make_neighbor_hooks()` 现在可以在无 Warp 环境构造 reference Hook，并由 Hook 写入邻居表后直接调用 LJ wrapper，形成 `model → hook → neighbor matrix → energy/forces` 的 CPU reference 链。
- framework 邻居 Hook 与 LJ reference 测试合计 `13 passed`，退出码 `0`；新增测试覆盖无 eager dynamics 导入、共享 `DynamicsStage` 域识别和完整 Hook→LJ 链。ruff、compileall、diff 检查通过。项目 `.venv` 的 HCU 命令首次因缺少 `plum` 阻断，补齐 `plum-dispatch` 后继续暴露尚未安装的 `jaxtyping`；改用已有探索环境后，在 source DTK 26.04、BW200/gfx936、`HIP_VISIBLE_DEVICES=0` 上同一条链退出码 `0`，能量 `[-0.9833724493736826, -0.8909652875830761]`，总力 `[0, 0, 0]`，设备为 `BW200, UBB BW1000`。原始摘要见 `artifacts/g1/framework_neighbor_lj_reference_hcu0.json`。
- 上游 `test/hooks/test_stage_timing_hook.py` 在当前探索环境收集阶段仍因 `nvalchemi.dynamics` eager 导入 Warp 而阻断；这不是本次 stage-domain 修复的断言失败，reference-specific stage-domain 回归已包含在上述 `13 passed` 中。
- 下一小步：把 G1 的 PBC 邻居集合和短 NVE/轨迹输出拆成独立小步；本轮仍未宣称 PBC、NVE 或生产 Warp API 已支持。

### 2026-09-05：Torch reference PBC 邻居集合与 image shifts

- `packages/ops/nvalchemiops/torch_reference.py` 现在支持 full periodic neighbor list：批量 cell、部分 PBC、正交/可逆三斜胞、MATRIX/COO 和 `neighbor_matrix_shifts`；使用 `r_ij = r_i - r_j - shift @ cell` 的上游约定。reference 路径保留 eager 小输入定位，不把它当作最终性能后端。
- PBC `half_fill=True`、周期 pair geometry、skin/rebuild、method/scratch 仍显式失败；因此尚未改变 LJ wrapper 的 PBC 拒绝契约，也没有伪装支持周期力或 NVE。
- ops PBC 测试 `8 passed`；framework `compute_neighbors`/Hook PBC 测试通过；CPU 三斜胞契约和 source DTK 26.04 后 BW200/gfx936 HCU 探针均通过。HCU 结果为邻居矩阵 `[[1], [0]]`、shift `[[[-1, 0, 0]], [[1, 0, 0]]]`、计数 `[1, 1]`，设备 `BW200, UBB BW1000`。证据：[reports/g1-pbc-neighbor-reference.md](../reports/g1-pbc-neighbor-reference.md)。
- 下一小步：为 LJ reference 接入 PBC shift 后的 energy/force 公式和 FP64 解析对照；通过后再做短 NVE/轨迹。

### 2026-09-05：暂停交接点

- 当前暂停基线为 commit `4a7d0df`（`feat: add periodic torch reference neighbors`）；暂停前工作树干净。相关前置提交包括 `68433be`（隔离 `DynamicsStage`）和 `f5a4621`（记录数值验证策略）。
- 已有真实证据：Torch reference 的 no-PBC 邻居→LJ 链在 CPU 与 BW200/gfx936 HCU 上通过；PBC full-list 邻居集合、MATRIX/COO 写回和 signed image shifts 在 CPU、framework 测试及 HCU 探针上通过。当前回归结果为 ops PBC `8 passed`，framework 邻居/Hook/LJ 组合 `16 passed`。证据见 [PBC 报告](../reports/g1-pbc-neighbor-reference.md) 和 `artifacts/g1/pbc_neighbor_reference_hcu0.json`。
- 当前边界保持明确：PBC half-list、带 image shift 的 LJ energy/force、switching、virial/stress、skin/rebuild、NVE/轨迹、Triton/HIP kernel 和 Warp 生产默认路径仍未移植或验证；不得将邻居 PBC 通过写成完整周期物理链。项目 `.venv` 已记录北外 PyPI 镜像和缓存位置，但仍缺少完整 framework 依赖（当前 HCU framework 探针复用 `/home/wangleping/codes/nvalchemi-toolkit/.venv`）。
- 下次第一条开发任务：实现并测试 `lj_energy_forces` 消费 `neighbor_matrix_shifts` 的 full-list 周期参考公式（先固定无 switching、无 stress、无 half-list），用独立 FP64 解析结果比较 energy/force；通过后再单独增加短 NVE/轨迹探针。开始前先重跑 `probes/pbc_neighbor_reference.py`，确认 DTK 26.04、gfx936 设备和当前环境仍可见。

### 2026-09-06：周期 LJ 与独立短 NVE reference 闭环

- `torch_reference.lj_energy_forces` 与 dispatcher 现在消费批量 `cell`、`batch_idx` 和 `neighbor_matrix_shifts`，沿用 `r_ij = r_i - r_j - shift @ cell`；只支持 full-list、无 switching、无 virial/stress，周期 half-list 和跨体系 active pair 继续显式失败。
- 周期邻居不再静默丢弃重叠 active pair；新增不可逆 cell、multi-image、批量/部分 PBC、容量溢出和周期 LJ 独立 FP64 对照测试。ops `10 passed`，framework 邻居/Hook/LJ `16 passed`。
- 新增 `probes/pbc_lj_nve_reference.py`：使用 Batch、reference NeighborListHook/LJ 与独立 Torch velocity-Verlet，不改完整 Warp-backed `nvalchemi.dynamics.NVE`。CPU 与 source DTK 26.04 后 BW200/gfx936 HCU 均通过，1200 步、`dt=1e-4`、周期 wrap 第 `1079` 步发生，最大总能漂移 `5.373538503050668e-09`，总力 `[0, 0, 0]`。证据见 [reports/g1-pbc-lj-nve-reference.md](../reports/g1-pbc-lj-nve-reference.md)。HCU 运行 stderr 仍有 DTK 环境的 `clang++`/`hipconfig` 提示，但退出码为 0，设备运行结果有效。
- 本轮仍未实现周期 half-list、skin/rebuild、switching、virial/stress、完整 dynamics NVE API、Triton/HIP 或两卡 ownership。下一步应先整理本轮 diff/兼容清单并提交，再决定是否进入 skin/rebuild 或真实 MLIP 链；不把独立 NVE 探针写成完整 NVE API 已支持。

### 2026-09-06：Torch reference cached skin/rebuild

- `NeighborListHook(backend="torch_reference", skin>0)` 现在缓存以 `cutoff + skin` 构建的 full-list 邻居；raw Cartesian 位移超过 `skin/2`、cell 改变或 batch 结构改变时重建整批，否则复用缓存。周期 half-list、method/scratch、生产级 per-system rebuild 和 cell-list 仍显式不支持。
- `lj_energy_forces` 会过滤 cached skin 中距离达到实际 cutoff 的 pair，避免邻居缓存范围被误算为物理 cutoff；负 skin、跨体系 active pair 和重叠 active pair 显式失败。
- framework Hook/LJ reference 回归 `15 passed`，ops reference 回归 `10 passed`；短 NVE `skin=0.5` 在 CPU 与 BW200/gfx936 HCU 通过，1200 步、`dt=1e-4`、第 1079 步 wrap，最大总能漂移仍为 `5.373538503050668e-09`。报告见 [reports/g1-pbc-lj-nve-reference.md](../reports/g1-pbc-lj-nve-reference.md)。
- 这一步是正式 framework/ops reference 实现和测试；探针只负责跨设备证据。下一小步可独立加入环境加载脚本，然后再评估真实 MLIP 依赖和 MACE 前向/力梯度链；不宣称完整 dynamics NVE 或生产 Warp skin/rebuild 已支持。

### 2026-09-06：Torch reference per-system skin rebuild

- 在保持 Warp 默认路径不变的前提下，reference Hook 对 Batch 中超过 `skin/2` 位移阈值或 cell/pbc 变化的 system 做局部 eager 重建；局部矩阵索引恢复为全局 batch 偏移，未变化 system 的缓存和 Batch 边界保持不变。
- 新增 Batch 两体系回归：只移动第一个 system 时，第二个 system 的 reference 坐标和邻居矩阵保持不变，并覆盖 COO 全局偏移；framework Hook 测试 `11 passed`。容量不足仍由 reference dispatcher 显式抛出 `NeighborOverflowError`，动态扩容留到下一小步。
- 该步仍不是 cell-list、异步 rebuild 或生产级容量管理；下一小步是为容量溢出/扩容确定契约和测试，再进入更宽的 G2 路径。

### 2026-09-06：MACE 依赖、wrapper 与 batching 验证

- 使用 `source scripts/activate_hygon_env.sh exploration` 审计了前期探索环境：`mace-torch 0.3.16`、`e3nn 0.4.4`、`ase 3.29.0` 可导入，Torch 为海光 `2.9.0+das.opt1.dtk2604`。用户确认 `~/.cache/mace/MACE-OFF23_small.model` 为可信本地权重；依赖边界见 [审计报告](../reports/g1-mace-dependency-audit.md)。
- 修复 `MACEWrapper.from_checkpoint` 的本地 `Path | str` 处理：已有文件直接加载，命名 foundation checkpoint 仍走 MACE 下载器。上游 MACE 回归 `3 passed, 89 deselected`，退出码 `0`。
- `probes/mace_probe.py` 的 direct model 在探索环境 CPU 和 source DTK 26.04 的 BW200/gfx936 HCU 均退出 `0`，能量 `-2077.7669553149117`、force loss `0.4526716740143`、26 个非零参数梯度。
- `probes/mace_wrapper_reference.py` 的 H₂O wrapper CPU/HCU 均退出 `0`；两个 `perf_46` CIF 的 CPU batching 通过，`batch_ptr=[0,46,92]`、跨体系边 `0`、逐体系总力约 `1e-6`。详细命令和 SHA256 见 [wrapper/batch 报告](../reports/g1-mace-wrapper-batch.md)。
- 向量化周期 reference 邻居后，单个 `perf_46` HCU 阶段从邻居约 `111.32` 秒降至约 `5.33` 秒，完整 wrapper 到同步约 `9.76` 秒；两个 `perf_46` CIF 的 HCU batching 退出 `0`，`batch_ptr=[0,46,92]`、1754 条边、跨体系边 `0`、逐体系总力最大分量约 `9.6e-7`。向量化前两次 180 秒超时和阶段定位仍保留在报告中，用于说明优化原因。
- 北外镜像 dry-run 成功解析 45 个包，安装 `mace-torch==0.3.15`、`e3nn==0.4.4`、`ase==3.29.0`、`matscipy==1.1.1` 及其运行依赖；另补齐 `jaxtyping`、`periodictable`、`tensordict`、`pydantic`、`dm-tree`、`zarr`、`plotext`、`loguru`。项目 `.venv` 的 Torch/Triton 版本保持不变。
- 为避免 HCU 单进程 MACE wrapper 被 NVIDIA 绑定阻断，`hooks`、`training`、`dynamics hooks` 的 PhysicsNeMo profiler 和域并行导入改为按需加载；PhysicsNeMo 未安装，新增 optional-import 回归。该改动保留显式 profiling/DomainParallel 名称，未宣称这些能力已在 HCU 可用。
- 项目 `.venv` CPU H₂O wrapper 退出 `0`；source DTK 26.04、`HIP_VISIBLE_DEVICES=0` 的项目 `.venv` H₂O HCU wrapper 退出 `0`；同环境两个 `/data/csp_data/perf_46` CIF HCU batching 退出 `0`，`batch_ptr=[0,46,92]`、1754 条边、跨体系边 `0`、逐体系总力最大分量约 `1.2e-6`。artifact：`artifacts/g1/mace_wrapper_project_perf46_batch2_hcu0.{json,stderr,exit}`。
- 默认 PyPI 源已从历史 HUST 尝试切换为 `https://mirrors.bfsu.edu.cn/pypi/web/simple`，脚本和当前环境文档已同步；HUST TLS 失败只作为历史 dry-run 记录。

### 2026-09-06：公共 reference dynamics 与 MACE 固定晶胞端到端组合

- 修复固定晶胞 `FIRE`/`FIRE2` 的隐式导入边界：变量晶胞 `npt_nph` 模块改为方法内按需导入，因此 `from nvalchemi.dynamics import FIRE, FIRE2` 在没有 Warp 的项目 `.venv` 中可以完成；变量晶胞路径仍明确保留 Warp 依赖。
- 公共 `FIRE`/`FIRE2` 新增 `backend` 参数并向固定晶胞 reference 算子传递；默认 `None` 仍保持上游 Warp 路径。公共 `NVE`、`FIRE`、`FIRE2` 均可用 `backend="torch_reference"` 进入同一 `BaseDynamics.run()` 流程。
- 新增 `packages/framework/test/compatibility/test_public_dynamics_reference.py`，用 DemoModel 和异构 `[3,5]` Batch 覆盖公共 NVE/FIRE/FIRE2；与 import-boundary 回归共 `6 passed`。
- 新增 `probes/mace_dynamics_reference.py` 和报告 `reports/g2-mace-dynamics-reference.md`。项目 `.venv` CPU 通过；在 source DTK 26.04、`HIP_VISIBLE_DEVICES=0` 的 BW200/gfx936 HCU 上，真实 `MACE-OFF23_small.model` + reference NeighborListHook + 公共 NVE/FIRE/FIRE2 各运行 2 步通过。`batch_ptr=[0,3,8]`、原子数 `[3,5]`、26 条邻居边、跨体系边 0、状态 finite，设备识别为 `BW200, UBB BW1000`。
- 这回答了当前组合问题：VV、固定晶胞 FIRE/FIRE2 reference 已足以组成短的真实 MACE MD/relaxation 纵向链，并已在 HCU 验证异构 batching；还不能宣称完整生产模拟。长轨迹能量漂移、DataSink/重启、inflight 补位、周期 dynamics、变胞 stress、NVT/Langevin 和 Triton/HIP 性能后端仍分别排期。
- 当前工作树包含本轮未提交的实现、测试、探针和文档；下一小步应把上游 `test/dynamics/` 中与 NVE/FIRE/FIRE2 对应的行为测试按 reference backend 解除阻断并运行，再接最小 DataSink/轨迹记录，而不是立即进入 Triton/HIP 优化。

### 2026-09-06：32×92 周期 MACE/FIRE2 批量弛豫

- 在项目 `.venv`、`source scripts/activate_hygon_env.sh project`、单张 BW200/UBB BW1000（gfx936）上运行 `probes/mace_fire2_batch_relaxation.py`：32 个 92 原子周期 CIF 组成一个 Batch（2944 原子、连续 `batch_ptr`），真实本地 `MACE-OFF23_small.model`、公共 `FIRE2(backend="torch_reference")`、固定晶胞、`fmax=0.01`、最大 2000 步、`dt=0.01`、skin 0.5。
- 命令退出码 `0`；`step_count=837`、`elapsed_s=143.8770088129677`，32/32 收敛，结束后在最终坐标重算的 `max_final_fmax=0.00997547060251236`。这证明了当前 Torch reference 的真实 MACE/FIRE2/HCU 批量纵向链，仍不等同于生产 cell-list/Triton/HIP 或变胞流程。详细证据见 `reports/g2-mace-fire2-batch32.md`。
- 监控显示邻居缓存后的 active `fmax` 单调进入阈值附近；此前全局约 10 的陈旧力来自 graduated 图临时 pre-update 后只恢复坐标/速度。`BaseDynamics._mutable_fields` 现同时保存/恢复 `forces`、`energy`、`stress`，并增加 `test_inactive_dynamics_outputs.py`（1 passed）；`test_base.py` 全部 85 项仍通过。
- FIRE2 reference 逐式按锁定上游核对，并与 `/home/wangleping/codes/hyalchemi-ops` 的 5 组异构随机 Torch 实现交叉运行，位置、速度、alpha、dt、计数器最大差均为 0。探索工程只作数值交叉证据，不改变本项目的上游规范、API 或状态布局。
- 当前仍未完成：上游 `test/dynamics/` reference 套件解除阻断、DataSink/轨迹/重启、inflight 补位、变胞 stress、NVT/Langevin、周期 half-list、cell-list 和生产 Triton/HIP；下一小步先接上游 FIRE/FIRE2 行为测试中的 reference 可运行子集，并登记每项未支持参数的失败行为。

### 2026-09-06：上游 dynamics reference 行为测试子集

- `packages/framework/test/dynamics/test_ops.py` 保留锁定上游测试作为主要回归载体，仅增加测试环境变量：`NVALCHEMI_TEST_BACKEND` 显式传入 `backend`，`NVALCHEMI_TEST_DEVICE` 选择设备；未设置时仍为 `backend=None`/CPU，产品默认路径没有改成 reference 或静默回退。
- 显式 `torch_reference` 运行上游 VV/FIRE/FIRE2 子集：项目 `.venv` CPU 为 `23 passed, 74 deselected`，source DTK 26.04 后 BW200/UBB BW1000（gfx936）单卡 HCU 同为 `23 passed, 74 deselected`，退出码均为 `0`。HCU 运行约 1.21 秒。
- 未设置测试覆盖变量的 CPU 基线为 `2 passed, 13 failed, 82 deselected`；失败点均为上游 Warp kernel 或 `nvalchemiops.torch` 的 Warp 初始化，证明默认 Warp 选择仍然显式且没有被隐式替换。详细命令和结果见 `reports/g2-upstream-dynamics-reference.md`。
- 这一小步只解除上游 `test_ops.py` 中 VV/FIRE/FIRE2 固定晶胞行为测试的 reference 阻断；Langevin、Nose-Hoover、NPT、变胞 FIRE2、完整 dynamics 套件、DataSink/轨迹和 inflight 仍分别排期。下一步可接最小 DataSink/轨迹契约，再决定是否扩展更多上游 dynamics 测试。

### 2026-09-06：A2 上游本地补丁清单

- 在 `docs/UPSTREAM.md` 增加“本地补丁清单”，详细登记 `BaseDynamics._mutable_fields` 的输出恢复补丁和上游 `test_ops.py` 的 reference/HCU 测试环境变量，并按 storage、neighbors/LJ/MACE、dynamics 导入隔离、可选 NVIDIA 依赖和 ops dispatcher 分组登记其余已知上游文件修改。
- 清单明确区分上游候选、测试载体约定和产品侧新增 reference 文件；特别标注 `_mutable_fields` 同时影响 Warp 默认路径，后续同步前必须重新审阅。`external/` 未修改。
- 文档自检通过：`git diff --check` 和补丁清单标记校验退出码 `0`；本步未改变运行时代码。下一步进入异构 Batch 的最小 DataSink/轨迹契约，并接入 `[46,92]` MACE/FIRE2 验收。

### 2026-09-06：异构 Batch 最小 DataSink/轨迹契约

- 发现并修复 `HostMemory.write` 的真实快照别名问题：同设备 CPU 输入调用 `data.to("cpu")` 可能保留 live Batch 张量，第二帧的 in-place 更新会覆盖第一帧。现在设备迁移后显式 `AtomicData.clone()`；修改登记为 `docs/UPSTREAM.md` 的 LP-008。
- 新增 `packages/framework/test/compatibility/test_heterogeneous_trajectory_reference.py`。两个不同原子数体系 `[3,5]` 在两个 frame 写入同一 HostMemory，验证 `batch_ptr=[0,3,8,11,16]`、`batch_idx` 有序、体系 ID/status、位置/力/速度/能量/自定义 `trajectory_step` 的独立快照。该测试不把轨迹写出扩展成 checkpoint/restart。
- CPU 项目 `.venv`：轨迹契约 `1 passed, 1 deselected`；HCU 项目 `.venv`、DTK 26.04、BW200/gfx936 单卡：`1 passed, 1 deselected`，18.35 s，退出码均为 `0`。HostMemory/Drain/DataSink 上游子集 CPU `17 passed, 39 deselected`。完整证据见 `reports/g2-heterogeneous-trajectory-reference.md`。
- `FEATURE_COMPATIBILITY.yaml` 的 `data.storage` 保持整体 `planned`，新增的 HostMemory 轨迹 slice 标为 `dcu_status: partial`；GPUBuffer、ZarrData、完整 stream 生命周期和 restart 仍未验证。`test_observer_hooks.py` 的 logging/`_segmented_max` Warp 边界失败仍单独记录，未被隐藏。
- 下一小步：把该最小轨迹 sink 接入真实 `[46,92]` MACE/FIRE2 短弛豫，检查异构体系状态与 frame 输出；若先发现编排问题，优先修正并保持 CPU/HCU 双证据。

### 2026-09-06：真实异构 `[46,92]` MACE/FIRE2 短轨迹

- 新增 `probes/mace_fire2_heterogeneous_trajectory.py`：一个 46 原子周期 CIF 与一个 92 原子周期 CIF 在同一 Batch 中运行公共 `FIRE2(backend="torch_reference")`，通过 reference NeighborListHook 和 `SnapshotHook → HostMemory` 记录 3 个 frame。
- CPU 与 HCU 均退出码 `0`。读回 `stored_batch_ptr=[0,46,138,184,276,322,414]`、`trajectory_step=[1,1,2,2,3,3]`、`cross_system_edges=0`；CPU 耗时约 `18.54 s`，HCU（BW200/gfx936）约 `17.94 s`，最后一帧邻居边数 `3284`。详细证据见 `reports/g2-mace-fire2-heterogeneous-trajectory.md`。
- 这补上了真实 MLIP 异构 Batch 的最小轨迹证据，但只覆盖固定晶胞 3 步 reference；不等价于收敛弛豫、checkpoint/restart、GPUBuffer/ZarrData 或生产后端支持。`FEATURE_COMPATIBILITY.yaml` 的 `data.storage` 和 `dynamics.fire` 已同步切片证据与限制。
- 下一小步：回到 A3 契约欠账，集中刷新 `FEATURE_COMPATIBILITY.yaml` 的已验证 reference slice 和 `STATUS.md` 顶部当前快照；随后再决定是否进入更长异构弛豫或 inflight/容量契约。

### 2026-09-06：A3 契约与快照刷新

- 核对并刷新 `FEATURE_COMPATIBILITY.yaml`：`neighbors.skin_rebuild` 的容量证据、`dynamics.integrators` 的 velocity-Verlet/kinetics/public-boundary reference、`dynamics.fire` 的固定晶胞 FIRE/FIRE2、`models.mace` 的 32×92 与异构轨迹证据、`data.storage` 的 HostMemory 轨迹 slice 均保留 `status` 与 `verification.dcu_status` 两轴；窄 reference 证据没有改写成完整特性 verified。
- `STATUS.md` 顶部当前快照已改为反映真实进展：包括 32×92 HCU FIRE2 弛豫、异构 `[46,92]` 轨迹和当前未验证的变胞/stress、inflight、restart、性能后端等边界。`docs/STATUS.md` 时间线与当前快照以后都作为每步 DoD 的独立项。
- JSON 解析、证据路径核对和 `git diff --check` 均通过。当前工作树仍保留本轮未提交代码、测试和报告，未修改 `external/`。
- 下一小步：优先复核/补齐异构 Batch 的长弛豫状态隔离与 inflight 补位契约；若先做性能轴，则只做 N4 三口径 profile，不直接注册 Triton/HIP kernel。

### 2026-09-06：上游 FusedStage/inflight reference 子集

- 对锁定上游 `test/dynamics/test_inflight.py` 只增加 `NVALCHEMI_TEST_BACKEND` 选择，未改变未设置变量时的默认 Warp 语义；变更登记为 `docs/UPSTREAM.md` 的 LP-009。
- 显式 `torch_reference` 下 `TestFusedStageInflight`：项目 `.venv` CPU `10 passed, 3 skipped, 20 deselected`；source DTK 26.04、BW200/gfx936 HCU `13 passed, 20 deselected`，退出码均为 `0`。通过了毕业/补位、状态清理、send path 收缩、数据集耗尽和 HostMemory 写入。
- 未设置 backend 的 CPU 反向验证为 `10 passed, 1 failed, 3 skipped, 20 deselected`，唯一失败在 `_ops.fire` 的 Warp 导入边界。`test_state_management.py` 中依赖 `NVTLangevin` 的 5 个状态测试仍在 `_ops.langevin` Warp 边界失败，未被伪装成 reference 通过。证据见 `reports/g2-upstream-inflight-reference.md`。
- `FEATURE_COMPATIBILITY.yaml` 已将 `dynamics.fused_stage` 与 `dynamics.inflight` 登记为 `planned/partial`，明确窄 reference 范围和真实 MLIP inflight、NVT/Langevin、restart、多卡 ownership 的剩余限制。
- 下一小步：以真实 `[46,92]` MACE/FIRE2 为载体验证 inflight 补位与逐体系收敛隔离；若需引入 NVT/Langevin，先单列其 reference 算子和 Warp-free 导入边界，不把它混入 FIRE2 验收。

### 2026-09-06：真实 MACE/FIRE2 异构 inflight 补位

- 新增 `probes/mace_fire2_inflight_reference.py`：四个周期结构 `[46,92,46,92]`，`SizeAwareSampler(max_atoms=138, max_batch_size=2)`，公共 FIRE2 reference 与 HostMemory sink。初始 `[46,92]` 同时毕业后，一次预算请求补入 `[92,46]`；四个 `system_id` 均只写入一次。
- CPU 退出码 `0`，耗时 `18.77 s`；HCU 退出码 `0`，BW200/gfx936 耗时 `18.60 s`。两边均得到 `graduated_batch_ptr=[0,46,138,230,276]`、`trajectory_steps=[1,1,2,2]`、`cross_system_edges=0`、邻居边 `6568`。报告见 `reports/g2-mace-fire2-inflight-reference.md`。
- 探针曾先暴露两个明确的接线问题并已修正：FusedStage 初始 compute 必须使用普通 `register_hook` 才能触发邻居 Hook；HCU dataset 必须在目标设备构造 replacement AtomicData，sampler 不会自动迁移任意 dataset。这些没有被静默回退。
- 这一步只证明短 inflight 补位和 sink 顺序，`refill_frequency=1` 的检查开销尚未 profile；真实混合收敛 occupancy、长轨迹、checkpoint/restart、NVT/Langevin 和多卡 ownership 仍未验证。
- 下一小步：对异构真实弛豫做较长但受限的逐体系收敛隔离验证，记录 `refill_frequency` 的语义/开销观测；之后再按 N4 三口径 profile 决定是否推进 cell-list。

### 2026-09-06：轨迹字段副本契约补全与本轮计划

- 最小 `[3,5]` 轨迹测试现在在第一帧保存位置、力、速度、能量，第二帧修改 live Batch 后再写入；读回断言确认四类字段均为独立快照，同时保留体系边界、`batch_ptr`、`system_id`、`status` 和 `trajectory_step` 检查。
- CPU 项目 `.venv` 退出码 `0`（`1 passed, 1 deselected`，3.34 秒）；主机 DTK 26.04/HCU 权限下 BW200/gfx936 退出码 `0`（`1 passed, 1 deselected`，18.35 秒）。沙箱内 HCU 运行仅得到设备不可见的 skip，不计入 HCU 证据。
- 当前执行顺序固定为：A1 的 `[46,92]` 混合 Batch 与两个单体系基线对照 → A3 证据同步 → A4 冷启动/稳态/规模阶梯/混合 Batch 性能基线；不提前注册 Triton/HIP kernel。

### 2026-09-06：A1 mixed/single 三路 smoke 对照

- 新增 `probes/mace_fire2_baseline_comparison.py`，对同一 `[46,92]` 输入分别运行 mixed、46-only、92-only fresh Batch；每个 case 输出逐步 JSON，并检查 `batch_ptr`、轨迹帧、体系 ID、邻居边和最终逐体系 `fmax`。
- CPU 三步 smoke 退出码 `0`：mixed 与 single 的最终 `fmax` 差值为 `9.54e-7`（46 原子）和 `2.44e-6`（92 原子），无跨体系边。
- DTK 26.04/HCU gfx936 三步 smoke 退出码 `0`：差值为 `1.01e-6` 和 `4.17e-7`，无跨体系边。完整数据见 `reports/g2-mace-fire2-baseline-comparison.md`。
- 该证据只覆盖三步 reference smoke；三个 case 均未达到 `fmax=0.01`，A1 的收敛步数、最终状态和长轨迹对照仍未完成。下一小步为受限 HCU 收敛运行，并持续监控逐步 JSON 日志。

### 2026-09-06：A1 HCU mixed/single 收敛对照

- 将 `probes/mace_fire2_baseline_comparison.py` 增加 `--only-case`，把收敛 case 分开运行，避免单个慢 case 阻塞其余基线；三路均使用 `fmax=0.01`、最大 2000 步、内部 90 秒墙钟和逐 20 步 JSON 日志。
- HCU mixed `[46,92]`：425 步，47.22 秒，最终 `fmax=[0.0094126,0.0098724]`，`status=[1,1]`，跨体系边 0。
- HCU single 46：367 步，36.46 秒，最终 `fmax=0.0090830`；single 92：425 步，39.93 秒，最终 `fmax=0.0098400`；两者均 `status=1`、跨体系边 0。三路轨迹的 `batch_ptr`、`system_id` 和 `trajectory_step` 均通过检查。
- CPU mixed 收敛可行性检查设置 45 秒内部墙钟、60 秒外部超时，只完成第 1 步后终止；CPU 三步 smoke 仍通过。该限制记录为当前 reference 性能边界，不伪装成 CPU 收敛通过。
- A1 的 HCU 收敛证据已闭合；报告见 `reports/g2-mace-fire2-baseline-comparison.md`。下一步同步本轮窄 slice 文档，然后开始 A4 三口径性能基线。

### 2026-09-06：A4 第一组 MACE/邻居阶段 profile

- 将 `probes/mace_wrapper_stages.py` 升级为冷启动、预热和稳态分项 profile，HCU 每个阶段显式同步；报告见 `reports/g2-mace-profile-reference.md`。
- `[46,92]` 混合 Batch：CPU 稳态约 `5.226 s/step`（邻居 `0.539 s`、MACE `4.661 s`）；HCU 稳态约 `0.217 s/step`（邻居 `0.191 s`、MACE `0.025 s`）。冷邻居分别约 `0.959 s` 和 `15.607 s`，不能与稳态直接比较。
- HCU 等长规模阶梯（92/276/1012 原子）均完成短 profile：稳态总计约 `0.148/1.369/2.283 s/step`；邻居均值约 `0.125/0.568/1.477 s`。样本包含首个 JIT 波动，不能作为最终吞吐率。
- HCU 混合规模阶梯（约 276/1012 原子）也完成短 profile：稳态总计约 `1.319/2.429 s/step`；邻居均值约 `0.560/1.622 s`。`batch_ptr` 按 `[46,92]` 混合分层保持正确，首个稳态 MACE JIT 波动已单独记录。
- 当前在共享 HCU 负载下观察到 HCU 稳态主要受 Torch reference 邻居构建影响、CPU 主要受 MACE forward 影响，但绝对耗时和波动不能作为干净性能结论，也不能直接决定 cell-list/Triton/HIP 优先级。已观测同步点包括模型/Batch、邻居构建、输入适配和 MACE 输出边界；skin rebuild、FIRE2 端到端分项、完整同步点清单和 cell-list 决策仍未完成。下一小步可做 FIRE2+skin 功能性 profile；干净性能基线等待目标卡空闲或低干扰窗口，不终止其他任务。

### 2026-09-06：A6 FIRE2 上游 NumPy oracle 对照

- 新增 `packages/framework/test/compatibility/test_fire2_upstream_reference.py`。测试从锁定上游 `packages/ops/test/dynamics/test_fire2.py` 的 AST 提取 `_fire2_reference_step`，避免执行其 Warp 顶层导入；被测为 `nvalchemi._dynamics_reference.fire.fire2_step_coord`。
- 异构 `batch_idx=[0,0,1,1,1,2,2]` 的 float32/float64 各连续运行 5 步，positions、velocities、alpha、dt、nsteps_inc 均与上游 NumPy oracle 对齐；项目 `.venv` CPU 结果为 `2 passed`，退出码 `0`。
- 证据见 `reports/g2-fire2-upstream-reference.md`。性能 profile 暂停；下一小步转入不依赖绝对耗时的 FIRE2+skin rebuild 功能契约或 reference compile 限制登记。

### 2026-09-06：FIRE2 + skin/rebuild 功能契约

- 新增 `packages/framework/test/compatibility/test_fire2_skin_functional.py`，用公共 `BaseDynamics.step()` 将 `FIRE2(backend="torch_reference")` 与 `NeighborListHook(backend="torch_reference", skin=0.5)` 接在一起。
- 异构 `[3,5]` Batch 验证首步建缓存、低于 `skin/2` 的位移复用缓存、只移动 system 0 超阈值时只刷新 system 0，以及 `batch_ptr`、`batch_idx` 和邻居边无跨体系变化；CPU 与 DTK 26.04/HCU gfx936 各 `1 passed`，退出码均为 `0`。
- 证据见 `reports/g2-fire2-skin-functional.md`。这只是 reference 组合正确性，不是 cell-list、异步 rebuild、Warp 生产路径或性能基线。由于 HCU 仍由其他任务共享，A4 profile 继续暂停；下一步登记 reference `torch.compile` 限制或继续解除未覆盖的上游 reference 行为测试。

### 2026-09-06：reference dynamics `torch.compile` 边界登记

- 新增 `probes/dynamics_reference_compile.py`，在项目 `.venv` CPU/Torch 2.9.0 中分别检查 FIRE2、kinetic energy 和 temperature 的 `fullgraph=False` 与严格 `fullgraph=True` 行为。
- kinetic energy 的 `torch.library.custom_op` + fake 注册在严格 fullgraph 下通过；FIRE2 默认编译可运行但产生 4 次 graph break，temperature 产生 1 次；两者严格 fullgraph 均失败，原因分别是 Tensor→Python 验证分支和 data-dependent branching。
- 结果登记于 `reports/g2-dynamics-reference-compile.md` 及 `FEATURE_COMPATIBILITY.yaml`。当前 reference 保证 eager 正确性，默认编译只保证可运行，不承诺完整图或优化吞吐；后续如需 compile 支持，再单独增加 custom-op 外壳并做数值回归。

### 2026-09-06：上游 FusedStage、状态和 sink reference 子集

- 直接运行锁定上游 `test_single_loop.py` 的 `TestConvergenceHook`、`TestFusedStage`、`TestFusedStageSubstageHooks`，CPU 项目环境共 `55 passed`，验证共享 forward、逐体系状态迁移、masked update、Hook 时序/频率和收敛生命周期。
- 直接运行锁定上游 `test_sinks.py` 的 `TestDataSinkABC`、`TestDrainMethod`、`TestHostMemory`，CPU 项目环境共 `17 passed`，验证 sink 抽象、drain、容量和 HostMemory mask 写入。
- 运行 `test_state_management.py::TestMakeNewState` 的初始审计得到 `7 passed, 6 failed`；随后只为固定晶胞 NVE/FIRE/FIRE2 测试接入 `NVALCHEMI_TEST_BACKEND`，显式 reference 状态管理子集 CPU `20 passed`。Langevin/Nose-Hoover/NPT 测试仍在未移植的 `_ops.langevin` 或 `_ops.nose_hoover` Warp import boundary 阻断；未把失败改成 skip 或静默回退。
- 证据见 `reports/g2-upstream-fusedstage-state-sinks-reference.md`，兼容清单已加入对应上游测试载体，补丁登记为 `docs/UPSTREAM.md` 的 LP-010。下一步可继续审计其他不依赖热浴/压强积分器的上游行为；Langevin/NPT 仍单独排期。

### 2026-09-06：上游设备/stream/pipeline、Sampler 与完整 inflight 回归

- `test_single_loop.py` 的设备默认、通信 stream、FusedStage stream 传播和 `DistributedPipeline` 组合子集在 DTK 26.04、BW200/gfx936 HCU 上 `31 passed`；同一组在无 GPU CPU 进程为 `27 passed, 4 failed`，失败是上游 CUDA 默认断言和构造后打补丁的环境差异，已在报告中保留，不改生产默认设备语义。
- FusedStage/Hook 55 项与 HostMemory/DataSink 17 项合并在 HCU `72 passed`；固定胞 NVE/FIRE/FIRE2 状态管理 reference 精确子集在 HCU `20 passed`，CPU 同样 `20 passed`。未移植 Langevin/Nose-Hoover/NPT 仍在 Warp import boundary 阻断。
- 完整 `test_sampler.py` 的 `SizeAwareSampler` 51 项在 CPU `51 passed`、HCU `51 passed`；覆盖异构大小分箱、原子/边/批大小预算、replacement 和耗尽语义。完整 `test_inflight.py` 显式 reference 在 CPU `30 passed, 3 skipped`、HCU `33 passed`，补充覆盖多阶段状态、partial replacement、system_id 保留和 convergence return。证据见 `reports/g2-upstream-sampler-reference.md`、`reports/g2-upstream-inflight-reference.md`。
- 本轮仍不启动 A4 性能 profile；上述 HCU 时间只用于测试退出/规模记录，不能作为吞吐结论。下一步保持小步，检查轨迹字段与上游 inflight 状态组合的边界遗漏；NVT/Langevin、变胞 stress、checkpoint/restart 和生产性能后端继续单独排期。

### 2026-09-06：observer hook 的显式 Torch reference

- 上游 observer 回归首次暴露：`LoggingHook` 的逐图 `fmax` 会无条件进入 `_segmented_max` Warp custom op，`EnergyDriftMonitorHook` 的动能路径也会触发 Warp；这与已完成的 kinetics reference 不一致。
- `scatter_reduce_per_graph` 新增 `backend` 选择，`LoggingHook`/`EnergyDriftMonitorHook` 新增 `compute_backend`，未显式设置时跟随 `ctx.workflow.backend`，否则继续使用 Warp。Torch reference 覆盖 sum/amax/amin/mean、temperature 和 energy drift 的异构 Batch 计算。
- 上游 `test_observer_hooks.py` 仅增加 `NVALCHEMI_TEST_BACKEND` 载体；显式 reference CPU `44 passed, 33 skipped`、HCU `77 passed`。新增 compatibility 的公共 dynamics/segment 契约 CPU/HCU 各 `8 passed`。未设置变量的单测仍在 `_segmented_max` Warp 边界失败，默认路径反向验证保持成立。证据见 `reports/g2-upstream-observer-reference.md`，补丁登记为 `docs/UPSTREAM.md` 的 LP-011。
- 该步没有启动 A4 profile；下一步继续清点 observer/trajectory 中可能残留的 Warp helper，并保持 NVT/Langevin、变胞 stress、checkpoint/restart 和生产 Triton/HIP 独立排期。

### 2026-09-06：上游 hook utility 的周期 helper 隔离

- `test_hook_utils.py` 的模块收集原先在导入 `nvalchemi.hooks.periodic` 时直接
  触发 `import warp`，导致 reduction、kinetics、temperature 等与周期包裹无关的
  上游测试在无 Warp 环境也无法收集。现将 Warp/ops 导入延迟到默认 Warp 分支，
  增加 `backend`/`compute_backend` 边界，并新增顶层
  `nvalchemi._dynamics_reference.periodic` Torch custom op，保留原地更新、异构
  Batch cell 选择、triclinic fractional wrapping 和逐维 PBC 语义。
- 上游 `test_hook_utils.py` 与 `test_periodic_hook.py` 仅通过
  `NVALCHEMI_TEST_BACKEND` 载体选择 reference；CPU 分别为 `30 passed, 5 skipped`
  与 `15 passed, 11 skipped`，退出码均为 `0`。`probes/hooks_utils_reference.py`
  的 CPU 四种逐图 reduction、KE/temperature 和两体系 periodic wrapping 通过，
  且 `warp_loaded=false`。未设置变量时 reduction 仍在 `_segmented_max` 的 Warp
  边界失败，默认路径反向验证保持成立。
- 受限沙箱中的同一 HCU probe 仍返回 `No HIP GPUs are available`，因为该环境隐藏
  `/dev/kfd`/`/dev/dri`；这不是 DTK 或 HCU 能力结论。使用
  `source scripts/activate_hygon_env.sh project` 在设备节点可见的主机权限终端，
  仓库内 `probes/torch_probe.py --device cuda` 退出码 `0`，随后
  `probes/hooks_utils_reference.py --device cuda` 退出码 `0`，并完成
  `test_hook_utils.py` + `test_periodic_hook.py` 的 HCU reference 回归（`61 passed`，
  含 CUDA compile smoke）。补丁登记为 `docs/UPSTREAM.md` 的 LP-012，报告见
  `reports/g2-upstream-hook-utils-reference.md`。未启动性能 profile，也未展开
  Langevin/NPT、变胞 stress 或 checkpoint/restart。

### 2026-09-06：上游 safety/freeze hook 回归

- 锁定上游 `test_safety_hooks.py` 与 `test_freeze_hook.py` 在项目环境 CPU
  `45 passed, 14 skipped`，退出码 `0`；覆盖 NaN/Inf 检测、逐图错误信息、最大力
  clamp、冻结位置/速度/力、异构 Batch、原地 mutation、Hook 阶段/频率和 CPU
  compile smoke。
- 在正确加载 DTK 26.04、设备节点可见的主机权限 HCU 上，CPU/CUDA 参数化共
  `57 passed`，退出码 `0`；CUDA `cudagraphs` compile smoke 实际执行。受限沙箱
  仍可能跳过 CUDA 或报告 `No HIP GPUs are available`，不作为 HCU 能力结论。报告
  见 `reports/g2-upstream-safety-freeze-reference.md`，原始输出见
  `artifacts/g2/upstream_safety_freeze_bias_hcu0.*`。未修改默认 Warp 路径，也未
  启动性能 profile 或热浴/变胞实现。

### 2026-09-06：上游 DemoDynamics/FusedStage 集成回归

- 锁定上游 `test_demo_dynamics.py` 全文件在项目环境 CPU `34 passed, 2 skipped`，
  退出码 `0`。覆盖 DemoDynamics 多图独立演化、接口字段、Hook 顺序、
  FusedStage/DistributedPipeline 组合、收敛 hook 和 `n_steps` 契约。在正确加载
  DTK 26.04 的主机权限 HCU 上同一文件 `36 passed`，退出码 `0`，CUDA 参数实际
  执行。该证据是上游纯 Torch 编排替身的行为回归，不等于 MACE、reference 邻居
  或 HCU 性能验证；结果追加至 `reports/g2-upstream-fusedstage-state-sinks-reference.md`。

### 2026-09-06：上游 BiasedPotentialHook 回归

- 锁定上游 `test_bias_hook.py` 在项目环境 CPU `12 passed, 8 skipped`，退出码
  `0`；覆盖能量/力 bias shape、原地 mutation、零 bias、阶段/频率、与
  NaNDetector 的组合和 CPU compile smoke。
- 在正确加载 DTK 26.04、设备节点可见的主机权限 HCU 上，CPU/CUDA 参数化共
  `22 passed`，退出码 `0`；CUDA compile smoke 实际执行。该结果只覆盖纯 Torch
  bias hook，不代表 MACE、邻居或完整 Hook 生产后端。报告见
  `reports/g2-upstream-bias-reference.md`，原始输出与 safety/freeze 合并保存于
  `artifacts/g2/upstream_safety_freeze_bias_hcu0.*`。

### 2026-09-06：上游 GPUBuffer sink 回归

- 锁定上游 `test/dynamics/test_sinks.py` 全文件在设备节点可见的 DTK 26.04、
  BW200/gfx936 HCU 上 `56 passed`，退出码 `0`，约 `1.37 s`。覆盖 GPUBuffer 和
  ZarrData 的基础写读、容量、mask、zero/drain、HostMemory 及错误边界；报告见
  `reports/g2-upstream-gpubuffer-reference.md`，原始输出见
  `artifacts/g2/upstream_sinks_full_hcu0.*`。
- 同一文件的 CPU 对照退出码 `0`，`33 passed, 23 skipped`，跳过项仅为 GPUBuffer
  设备用例；原始输出见 `artifacts/g2/upstream_sinks_full_cpu.*`。
- 受限沙箱中完整 sink 测试曾在 Zarr 首次写入处超时；独立事件循环探针与主机权限
  对照表明这是沙箱载体的异步唤醒问题，不能作为 Zarr 依赖、HCU 能力或性能失败。
  checkpoint/restart、完整 stream 生命周期和长轨迹容量继续保持未验证。

### 2026-09-06：Tier 1 前邻居基线与 batch 扩展

- 新增 `probes/neighbor_baseline.py`，只调用 `compute_neighbors(backend="torch_reference")`，
  每个 case 记录 cold、warmup 和两次 steady，并输出 `batch_ptr`、原子数和边数；不混入
  MACE forward 或 FIRE2，避免把模型成本混入邻居对照。
- CPU/HCU 单体系规模阶梯使用真实 `/data/csp_data/perf_46`、`perf_92`、`perf_184`、
  `perf_368` CIF，覆盖 46、92、184、368 原子。46 和 92 原子等长 batch 均测到
  `1/4/8/16/32`，184 原子测到 `4/8/16`，另测异构 `[46,92,184,368]`。
- 关键 steady 均值：92 原子 batch=32 为 CPU `9.2946 s`、HCU `4.2626 s`（54,760 边）；
  184 原子 batch=16 为 CPU `19.8595 s`、HCU `4.0736 s`（54,448 边）；异构 690 原子
  batch 为 CPU `6.5289 s`、HCU `0.8983 s`（12,598 边）。所有完整 case 退出码为 0；
  HCU 为 DTK 26.04/BW200/gfx936 共享负载，时间仅作相对基线。
- 首次 184 原子 CPU 合并运行在 batch=16 前达到 timeout，`tee` 曾掩盖退出码；已用
  `pipefail` 和单 case（`neighbor-baseline-cpu-batch184-16.log`）重跑并确认通过，截断日志
  不计入证据。完整原始日志见 `artifacts/g2/neighbor-baseline-*.log`，摘要见
  `reports/g2-neighbor-baseline-reference.md`。
- 基线表明当前周期 pair/image 距离计算虽已按体系 Torch 向量化，但逐边 Python 装配仍是
  明显目标；下一步进入 Tier 1 device-side `nonzero`/rank/scatter，保持邻居集合、shift、
  padding、overflow 和 Batch 边界语义不变。Tier 2 cell-list 与后续 Triton/HIP 仍排在
  Tier 1 正确性回归之后。

### 2026-09-06：Periodic 邻居 Tier 1 device-side scatter

- `packages/ops/nvalchemiops/torch_reference.py` 的 periodic full-list 路径已去除逐边
  Python `tolist()`/`append`/写回；每个 system 保留原有几何候选计算，活跃 pair 由设备端
  `nonzero`/`bincount`/行内 rank 和矩阵索引写回装配。shift、自相互作用排除、重合错误、
  padding、overflow、MATRIX/COO 顺序均保持原契约。
- CPU ops reference `10 passed`、framework NeighborListHook `16 passed`；在 source
  DTK 26.04 的 BW200/gfx936 HCU 上 ops reference `10 passed`，均退出码 0。报告见
  `reports/g2-neighbor-tier1-scatter-reference.md`。
- framework `compute_neighbors` 集成回归在同一 HCU 上 `3 passed`（CPU 对照亦为
  `3 passed`），确认 Batch/PBC 接线与直接 ops 结果一致。
- 同口径真实 `/data/csp_data/perf_92`、92 原子 batch=32 邻居构建（统一
  `OMP_NUM_THREADS=1`）：CPU steady `9.2946→7.7101 s`，HCU `4.2626→0.0658 s`，边数
  保持 `54,760`。HCU 仍有共享负载，该数值仅作为相对证据；原始日志见
  `artifacts/g2/neighbor-tier1-*.log`。
- 当前快照和 `FEATURE_COMPATIBILITY.yaml` 已同步；46/92/184 原子阶梯及异构 Batch 的
  scatter 后结果已补齐。下一步单独评估 no-PBC 装配，再决定 torch reference cell-list
  的优先级。

### 2026-09-07：Tier 1 邻居装配后 32×92 MACE/FIRE2 HCU 弛豫

- 在 periodic full-list device-side scatter 修改后，按用户要求重新运行 32 个 92
  原子周期 CIF；所有结构在同一个 Batch 中使用真实 `MACE-OFF23_small.model` 和
  `FIRE2(backend="torch_reference")` 做固定晶胞弛豫。
- 命令显式设置 `--fmax 0.01 --max-steps 2000 --dt 0.01 --skin 0.5`，其余 FIRE2
  参数使用上游默认值；通过 `source scripts/activate_hygon_env.sh project` 加载
  DTK 26.04 和项目 `.venv`，设备为 BW200/UBB BW1000、`gfx936`。
- HCU 退出码 `0`，`status=passed`；`num_nodes=2944`、`batch_ptr=[0,92,...,2944]`、
  最终邻居边 `75010`。32/32 体系收敛，最后一个在第 834 步达到阈值；总耗时
  `79.54159364895895 s`，最终逐体系最大 `fmax=0.009995493106544018`。
- 逐步 JSON 日志保存在 `artifacts/g2/mace-fire2-batch32-tier1.log`，完整结果与
  证据边界见 `reports/g2-mace-fire2-batch32-tier1.md`。该结果只证明单 HCU、固定
  晶胞、周期、等长 32×92 的 reference 组合；不扩大为 cell-list、生产 Triton/HIP、
  变胞 stress、热浴、restart 或无干扰性能支持。

### 2026-09-07：32×92 收敛能量与密度指标补采

- 使用与上项完全相同的 HCU 设置补采最终指标；32/32 仍收敛，`max_steps=2000`，
  最少/最多/平均首次收敛步数分别为 `234/834/483.46875`。
- 平均最终总能量为 `-78382.53393554688 eV/structure`（平均每原子
  `-851.9840812683105 eV/atom`），平均固定晶胞密度为
  `0.5551176927983761 g/cm³`。该次耗时 `79.80547558492981 s`。
- 原始 JSON 日志见 `artifacts/g2/mace-fire2-batch32-metrics-hcu.log`，统计和
  复现实验说明已补入 `reports/g2-mace-fire2-batch32-tier1.md`。这些是
  `torch_reference`/MACE HCU 指标，不作为 NVIDIA Warp 数值等同性结论。

### 2026-09-07：新开发者环境与开发模式文档

- 当前工程状态检查点已提交为 `9fb9d1f`；本次新增文档只覆盖部署和协作方法，不改变运行时代码。
- 新增 `docs/DEVELOPMENT_ENVIRONMENT.md`，说明 Hygon reference 环境的本地 Torch/Triton
  wheel、锁文件、DTK 加载、设备权限、最小验证和常见故障；`docs/ENVIRONMENT.md` 与
  `docs/START_HERE.md` 已增加入口。
- 新增 `docs/DEVELOPMENT_GUIDE.md`，说明 Torch reference、Triton、HIP、dispatcher、
  custom op、算子契约、功能接入、测试/探针和 Git 冲突规约；根 `AGENTS.md` 已增加精简的
  Agent 开工与交接约束。
- 本轮文档提交后，下一位开发者仍应先运行 `git status --short --branch`，再按部署指南确认
  本机 DTK、海光 Torch/Triton 和设备节点；文档不扩大当前已经验证的后端支持范围。

### 2026-09-08：G2 backend 收敛与 no-PBC 邻居 Tier 1

- `nvalchemiops.backend` 已从单一 reference resolver 改为 operation/device/dtype/gradient/features capability registry；`backend=None`/`"warp"` 只生成 legacy Warp 选择记录，`auto` 对每个选择签名首次发出包含实际 backend 与原因的 warning。Triton/HIP 仍无登记 capability，显式请求失败。
- `compute_neighbors`、`NeighborListHook`、LJ、固定晶胞 VV/FIRE、observer 与 periodic helper 已移除局部 backend 字符串集合；`make_neighbor_hooks(backend=...)` 是新公共拼写，`neighbor_backend` 保留为冲突检测的兼容别名。LoggingHook 的 `backend` 仍是 writer，计算参数保持 `compute_backend`。
- `LevelStorage` 的 `TorchStorageBackend` 默认已在测试与 ADR 0005 中明确；无 Warp 环境的显式 Warp 对照为 `WARP-EQUIV-001` strict xfail。基础数据导入不加载 Warp。
- no-PBC reference 邻居现在对每个 system 在设备上构造 pair geometry、`nonzero`、`bincount` 与 row-rank/scatter，保持 full/half、MATRIX/COO row-major、batch 边界、distance/vector、overlap 和 overflow 契约。CPU ops `17 passed`；相关 framework storage/neighbors/Hook/默认路径边界/public dynamics reference 为 `32 passed, 2 xfailed`，LJ `7 passed`，dynamics/observer/periodic 子集均退出 `0`。两个 strict xfail 分别跟踪无 Warp 的等价性与 legacy 默认路径守护；synthetic `[46,92]` CPU probe full/half 为 `734/367` 边。
- 新增 no-PBC 代码已在项目 `.venv`、DTK 26.04、BW200/gfx936 HCU 0 获得窄 slice 证据：ops registry/reference 回归 `17 passed, 1 warning`，异构 `[46,92]` probe 的 full/half 为 `890/445` 边、MATRIX/distance/vector 均为预期 shape；两体系 `AtomicData` 的 framework `compute_neighbors` HCU Batch 写回也通过。HCU 随机序列与 CPU 不同，不以边数对拍；这是功能 smoke，不报告为性能结论。统一 no-PBC/periodic benchmark harness 已完成 CPU neighbor 与 periodic `[46,92]` 2 步端到端 smoke，下一步补 HCU 阶梯和 100 步端到端；NVT/Langevin、PBC half-list、Triton/HIP、compile 和分布式保持未完成。
- 按原参数复跑 32×92 周期 MACE/FIRE2 固定晶胞弛豫：`fmax=0.01`、`max_steps=2000`、`dt=0.01`、`skin=0.5`，HCU `status=passed`，32/32 收敛，`step_count=835`，`max_final_fmax=0.009996769018471241`，耗时 `81.58037368883379 s`。独立日志见 `artifacts/g2/mace-fire2-batch32-rerun-20260908.log`，与此前 834 步、约 79.5--79.8 秒结果一致；仍不作为无干扰性能基线。

### 2026-09-08：neighbor backend selection 单次解析接线

- framework 的 `compute_neighbors` 与 `NeighborListHook` 现在先通过中央
  `nvalchemiops.backend` capability registry 得到 `BackendSelection`，再把同一选择传给
  Torch neighbor dispatcher；dispatcher 不再对同一请求二次解析。Warp 仍由 framework
  legacy 边界执行，默认 `backend=None` 语义未改变。
- `neighbors.py` 及相关 compute-backend docstring 不再手工维护后端枚举，改为引用
  `resolve_backend`/`backend_capabilities` 的 operation-scoped registry；logging writer、
  storage 和 stage-timing 等不同语义的 backend 保持各自边界。
- CPU ops dispatcher 回归 `12 passed`，framework neighbors/reference 回归 `5 passed`；
  HCU BW200/gfx936 ops 选择/neighbor 子集 `5 passed`，framework neighbor + FIRE2 skin
  子集 `6 passed`。新增测试确认预解析 selection 不会触发第二次 resolver 调用。
- 本步尚未把 LJ、动力学和 observer 的所有 dispatcher 改为 selection 传递，也未改变
  `auto` 优先级；下一步继续处理其余 compute backend 调用点并补 operation-scoped 文档
  和回归。

### 2026-09-09：M1 ImplementationRegistry 与 neighbor strategy

- 前置 cell-list 和计划文档已分别封存在本地 `codex/feat-reference-cell-list` 的
  `a33932b`、`afad629`；从该干净基线创建当前
  `codex/refactor-backend-plan-m1`，未推送、未改写任何提交。
- 固定 `BackendName`/capability tuple 已替换为 `ImplementationRegistry`。每个选择记录
  request、implementation ID、family、strategy 和 profile ID 占位；Warp legacy、Torch
  dense reference、no-PBC Torch cell-list 与现有 reference operation 均为 lazy metadata
  登记，resolver 不导入 executor。
- `compute_neighbors` 和 `NeighborListHook` 的 cell-list 用法改为
  `backend="torch_reference", method="cell_list"`；Torch dispatcher 按 implementation ID
  执行并接受 framework 的预解析 selection。旧的未发布全局名称明确失败；`None`/Warp
  default 与 auto dense default 均保持不变，auto strategy 在 M2 profile 前显式失败。
- CPU：ops registry/reference/cell-list `25 passed, 1 warning`，framework neighbor/Hook
  `22 passed, 1 warning`。BW200/gfx936 HCU 0：ops `25 passed, 1 warning`（19.39 s），
  新增 framework one-shot/Hook cell-list strategy `2 passed`（16.84 s），退出码均为 0。
  完整 framework HCU suite 的 compiled-entrypoint 运行没有产生可恢复完成记录，不计为
  通过证据。报告见 `reports/g2-backend-registry-m1.md`。
- M1 不实现 PlatformFingerprint、BackendProfile、Frozen BackendPlan、runtime fallback
  或性能选择；真实规模/周期 cell-list 与 auto profile 仍未验证。下一步仅在用户确认后
  进入 M2，并先建立 fingerprint/profile/plan 的独立契约和测试。

### 2026-09-09：M1 dynamics/LJ/observer selection propagation

- FIRE 与 FIRE2 已在 registry 中使用独立 operation 和 implementation ID，避免两个优化器
  共享一个含义不清的 `fire` contract。
- 固定晶胞 NVE/FIRE/FIRE2 在 workflow 初始化阶段冻结 selection；LJ model、periodic
  hook、Logging/energy-drift/reporting observer 辅助路径也按 operation 单次解析并向
  dispatcher 传递 selection。分布式 FIRE wrapper 保留该传递链路。
- 变胞 FIRE/FIRE2 显式要求 `variable_cell` capability；当前 Torch reference 未登记，
  因此请求会明确失败，默认路径仍是 legacy Warp。
- CPU 回归：ops `23 passed`；selection propagation `3 passed`；framework reference/observer/periodic/LJ
  slice `103 passed, 1 deselected`（与 propagation 合并为 `106 passed, 1 deselected`）；state
  lifecycle `22 passed`；import/reference `15 passed`。
  详细范围、命令与限制见 `reports/g2-backend-registry-m1-dispatch-propagation.md`。
- 本轮未新增 HCU、Triton/HIP 或变胞数值证据；M2 PlatformFingerprint/Profile/Plan 仍未开始。

### 2026-09-09：B0 团队基础开发版本候选

- 从 M1 follow-up `fb11528` 建立候选集成分支 `team/dev-baseline-v0.1`；该分支只可由人工
  审阅后合入 `develop`，没有自动 push 或 merge。
- 默认 registry inventory 按 legacy、neighbors、interactions、dynamics、observability 拆入
  私有 metadata-only catalog。公开 registry API、implementation ID、登记顺序和 executor
  lazy-load 语义不变；catalog 回归额外确认其导入不会加载 Warp 或 Torch reference executor。
- 新增 `scripts/check_cpu_reference.sh` 和显式设备要求的
  `scripts/check_hcu_reference_smoke.sh`，以及协作基线/新增 operation 文档。基础版本不新增
  产品能力，故 `FEATURE_COMPATIBILITY.yaml` 不变。
- CPU gate 已退出 `0`：ops `27 passed, 1 warning`、独立 import boundary `1 passed`、framework
  golden-path `121 passed, 1 deselected`、state `22 passed, 34 deselected`、上游 VV/FIRE/FIRE2
  op 子集 `24 passed, 73 deselected`，且 compileall/diff check 通过。未设置
  `HIP_VISIBLE_DEVICES` 的 HCU script 在运行前以退出码 `1` 明确拒绝；这只验证失败保护，
  HCU 状态仍为 pending，既有 M1 HCU 证据不自动覆盖这次结构改动。详细记录见
  `reports/team-dev-baseline-v0.1.md`。下一任务从周期 cell-list、NVTLangevin、Nose-Hoover
  chain 三项中选择一个独立 operation；M2 保持延期。
