# 上游锁定与导入依据

2026-09-05，按用户指定的两个完整 SHA 锁定，见 UPSTREAM_LOCK.yaml。
指令来源为本次用户指令、根 AGENTS.md、PROJECT_HANDOFF.md 和 START_HERE.md；历史路线文档与外部探索仅供参考。
初始主仓库 master 无提交，仅有未跟踪的交接文档和 .gitignore，无 packages 产品代码。已有真实 Git 身份 Wang, Leping / wangleping@dcu2。

两个 external origin 均为 NVIDIA 官方 GitHub URL，工作树干净，HEAD 正好等于锁定 SHA，is-shallow-repository=false。两仓库 `git fsck --full --no-dangling` 均退出 0；完整保留本地所含的可达历史，不声称核验远端所有 refs。根仓库从本地 clone 获取对象，使用不 squash subtree 独立导入提交；不修改 external。

framework 0.2.0 要求 Python >=3.11,<3.14、torch>=2.8、ops>=0.4.1；ops 0.4.1 要求 Python >=3.11,<3.15，torch extras >=2.8，基础依赖 warp-lang>=1.13.0、numpy。Python 3.12 / 海光 Torch 2.9 满足声明版本交集。AST 审计的 97 条 framework→ops 显式 from-import 均可从指定 ops 的模块/符号静态解析。脚本及完整 API、导入、测试、示例、配置和许可证清单见 probes/audit_upstream.py 与 reports/upstream_inventory.json。

这是**源码导入兼容基线**，不是 DCU 运行验证：静态符号解析不证明调用参数、动态导出、数值或二阶梯度正确。原始 Warp/NVIDIA 依赖尚未适配，未安装两包到探针环境。framework uv 的 dependency-metadata 对 ops 0.4.1 与本次源码依赖一致，但注释中的 Torch >=2.11 不适用于这两个锁定 pyproject 的实际 >=2.8 约束。不运行原包的 CUDA extras / uv sync。

两个根 LICENSE 均为 Apache-2.0；完整导入还保留 .licenses、内嵌第三方来源、SPDX、测试、CI 和文档。许可证清单记录于 lock/inventory，不将根许可证推广为所有第三方文件的许可证。

依赖风险：Hooks 直接使用 physicsnemo.utils.profiling；分布式还含 vendored upstream 与 shard 包装；不能整体删除 PhysicsNeMo 能力。当前 HCU 项目环境不安装该 NVIDIA/Warp 绑定包，单进程 Torch/HCU 通过可选导入路径，域并行与 profiling 保留为显式能力。MACE extra 固定 mace-torch==0.3.15，与 UMA 的 e3nn 版本要求存在冲突；当前项目环境使用 `mace-torch 0.3.15/e3nn 0.4.4`，探索环境使用 `mace-torch 0.3.16/e3nn 0.4.4`，cuEquivariance/UMA/compile 单列 C。保留 upstream tests；未运行的测试均不算通过。

旧探索 /home/wangleping/codes/nvalchemi-toolkit 和 /home/wangleping/codes/hyalchemi-ops 只读查阅了文件清单及后者 PROGRESS.md 的邻居/PBC/LJ 记录。其性能与数值记录未重跑、不纳入当前验证。后续按算子提取思路与测试案例，逐项检查来源、许可证和当前 SHA 语义后才移植。

## 实际导入记录

主分支 `codex/g0-initialization`；初始审计提交 `bd4c612`；framework subtree 提交 `52ffa2d`；ops subtree 提交 `88aa209`。两次导入均不 squash，锁定提交是对应导入提交的第二父提交，全部可达历史保留。

普通 fetch 仅获取 external 本地 main，而 external HEAD 锁定在另一个提交，首次 subtree 因对象不存在而退出 1、未导入目录。随后显式 `git fetch upstream-framework 4dfe3723def34df3fadb245981081ccf8c94c257` 和 `git fetch upstream-ops 26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`，再执行 subtree，均成功。不得将默认 main 当作锁定 SHA。

`probes/verify_import.py` 退出 0：两个 packages 子树 tree hash 与上游完全相同，两个 SHA 均是 HEAD 祖先，无嵌套 .git。证据 reports/import-verification.json。external 保持干净。嵌入的原 CI 文件仅作为来源保留，不会自动作为根 CI 生效。

## 本地补丁清单

本节登记导入到 `packages/framework` 和 `packages/ops` 后对上游文件的修改。
新增的 reference 模块、兼容性测试和探针属于产品侧新增文件，不冒充上游补丁；
它们仍需在对应特性报告中记录来源和限制。当前清单中的“工作树待提交”表示
修改已经存在但尚未形成新的本地提交。

| ID | 文件与位置 | 相对锁定上游的变化 | 动机与影响 | upstream candidate | 回归/证据 |
|---|---|---|---|---|---|
| LP-001 | `packages/framework/nvalchemi/dynamics/base.py`：`BaseDynamics._mutable_fields` | 上游 `("positions", "velocities", "cell")`；本地增加 `"forces"`, `"energy"`, `"stress"` | 混合 status Batch 中，已收敛体系会经历共享前向的临时 pre-update；恢复输出字段可以避免返回的力/能量对应临时坐标。该行为同时作用于 Warp 默认路径，需在后续上游同步和 ADR 0004 中重新审阅。 | **候选**：需要向上游说明 mixed-status 输出一致性语义；当前不改变默认后端。 | `test/compatibility/test_inactive_dynamics_outputs.py`；`test/dynamics/test_base.py` 85 项；工作树待提交 |
| LP-002 | `packages/framework/test/dynamics/test_ops.py`：`device`/`backend` fixtures 及 VV/FIRE/FIRE2 调用 | 上游没有 `NVALCHEMI_TEST_DEVICE`、`NVALCHEMI_TEST_BACKEND`；本地允许测试显式选择设备和 reference backend | 复用锁定上游行为测试作为 Torch reference/HCU 载体，同时保留环境变量未设置时的 `backend=None` Warp 默认语义。仅修改测试，不改变产品 API 默认值。 | **不候选**：测试载体约定，不应进入上游产品接口。 | `reports/g2-upstream-dynamics-reference.md`；CPU/HCU 各 `23 passed`；默认 Warp 反向验证 `13` 项在 Warp 边界失败；工作树待提交 |
| LP-003 | `nvalchemi/dynamics/__init__.py`、`dynamics/_ops/{_bridge,fire,velocity_verlet}.py`、`dynamics/base.py`、`dynamics/hooks/`、`dynamics/integrators/`、`dynamics/optimizers/` | 将 Warp/ops 导入延迟到明确的 Warp 路径；公共固定晶胞 dynamics 接受显式 reference backend；`DynamicsStage` 通过顶层无 Warp 模块复用 | 解除公共导入的 eager Warp 阻断，保持 `None`/`warp` 默认路径，提供 `torch_reference`/`auto` 的显式委派；变量晶胞和 NPT/NPH 仍按需保留 Warp。 | **候选**：导入隔离和可选 backend 边界可向上游提议，但 reference 实现不属于本次上游同步。 | `reports/g2-dynamics-public-boundary.md`、`reports/g2-mace-dynamics-reference.md`；主要实现提交 `6ef6446`，后续工作树修改待提交 |
| LP-004 | `nvalchemi/data/batch.py`、`data/level_storage.py`；新增 `data/storage_backend.py`、`data/warp_storage_backend.py` | storage 操作从模块级 Warp helper 改为 backend 协议，新增 Torch reference 和显式 Warp backend | 保留 Batch/LevelStorage 数据语义，在无 Warp 环境可导入并验证 uniform/segmented copy、defrag、扩容和多属性 mutation；不把 storage 协议误写成邻居 staging K 扩容管线。 | **候选**：backend 边界可回馈上游；Hygon Torch 实现不直接上游化。 | `reports/g1-project-reference-pytest.md`、`reports/g1-neighbor-capacity-reference.md`；提交范围 `917cbfa`–`3f89f6b` |
| LP-005 | `nvalchemi/neighbors.py`、`hooks/neighbor_list.py`、`models/base.py`、`models/lj.py`、`models/mace.py`、`hooks/stage_timing.py` | 增加显式 Torch reference 邻居/LJ/MACE 接线、PBC/skin/cache/capacity 受限路径和按需导入 | 建立 Batch→neighbor→energy/force→MACE 的可验证 reference 链；默认 Warp API 保持不变，half-list、switching、stress、cell-list 和生产性能仍显式限制。 | **部分候选**：后端参数/导入隔离可讨论；Hygon reference 受限语义需单独维护。 | `reports/g1-pbc-lj-nve-reference.md`、`reports/g1-skin-rebuild-batch-reference.md`、`reports/g1-mace-wrapper-batch.md`；提交范围 `43f2e3d`–`cfe8a39`、`78f9a3f`–`a1b6e59` |
| LP-006 | `nvalchemi/hooks/__init__.py`、`training/__init__.py`、`training/hooks/__init__.py` | PhysicsNeMo/profiling/DomainParallel 由顶层强制导入改为按需可选导入 | 避免 HCU 单进程 Torch/reference 路径被 NVIDIA 绑定依赖阻断；显式 profiling 和域并行名称仍保留，未宣称其 HCU 支持。 | **候选**：可选依赖边界可能适合上游；不删除高级能力。 | `test/hooks/test_optional_imports.py`、MACE wrapper 报告；提交 `4be155f` |
| LP-007 | `packages/ops/nvalchemiops/{__init__,backend,torch_backend,torch_reference}.py` 及 `dynamics/`、`interactions/`、`jax/`、`math/`、`neighbors/`、`torch/` 的 `__init__.py` | 新增可审计 backend dispatcher/reference；Warp 初始化移至显式 Warp-facing 边界 | 让 `nvalchemiops` 顶层可导入，避免 reference 路径级联 `initialize_warp()`；未知 Triton/HIP 请求明确失败。 | **候选**：dispatcher 结构可参考上游；Hygon 后端和受限 reference 不是上游默认实现。 | `reports/g1-torch-reference-backend.md`、ops reference tests；提交 `05bd0f0`、`109e453` |
| LP-008 | `packages/framework/nvalchemi/dynamics/sinks.py`：`HostMemory.write` | 上游写入使用 `data.to(self._device)`；本地在设备迁移后追加 `clone()` | DataSink 是轨迹快照所有者；当输入已经在 CPU 时，`Tensor.to("cpu")` 可能保留原张量，后续 Batch 的 in-place 更新会覆盖已写入帧。显式 clone 使标准字段和动态 graph/system 字段均满足 snapshot decoupling；只影响 HostMemory，不改变 GPUBuffer/ZarrData 或默认 dynamics backend。 | **候选**：可向上游提交“HostMemory snapshots must not alias live Batch”语义；需要评估 CPU 写入开销和现有 sink API。 | `test/compatibility/test_heterogeneous_trajectory_reference.py` CPU/HCU；`test/dynamics/test_sinks.py -k 'HostMemory or DrainMethod or DataSinkABC'` 17 passed；工作树待提交 |
| LP-009 | `packages/framework/test/dynamics/test_inflight.py`：`TestFusedStageInflight.test_send_path_realigns_state_and_last_converged` | 上游直接构造 `FIRE(...)` 未提供 backend；本地读取 `NVALCHEMI_TEST_BACKEND` 并传入，未设置时仍为 `None`/Warp 默认 | 让锁定上游的 inflight/FusedStage 状态回归可以在显式 Torch reference 下运行；只改变测试载体，不改变产品默认路径，也不把 NVT/Langevin 的 Warp 依赖伪装成 reference。 | **不候选**：沿用 LP-002 的本地测试环境约定，不应改变上游产品接口。 | `reports/g2-upstream-inflight-reference.md`；CPU `10 passed, 3 skipped`，HCU `13 passed`；工作树待提交 |
| LP-010 | `packages/framework/test/dynamics/test_state_management.py`：固定晶胞 NVE/FIRE/FIRE2 构造器 | 上游固定晶胞状态测试未传 backend；本地读取 `NVALCHEMI_TEST_BACKEND` 并传入，未设置时仍为 `None`/Warp 默认；Langevin/Nose-Hoover/NPT 测试保持原样 | 复用上游状态初始化、state shape、state invariant、partial removal 和 `_make_new_state` 测试验证已完成的 Torch reference；限制改动只作用于固定晶胞组件，避免把未移植热浴/压强积分器伪装成 reference。 | **不候选**：与 LP-002/LP-009 相同，是本地测试载体约定。 | `reports/g2-upstream-fusedstage-state-sinks-reference.md`；显式 reference 固定晶胞子集 CPU `20 passed`；工作树待提交 |
| LP-011 | `packages/framework/nvalchemi/dynamics/hooks/_utils.py`、`hooks/logging.py`、`hooks/monitors.py`；`packages/framework/test/dynamics/test_observer_hooks.py` | 新增 observer 计算 `compute_backend`，reference 使用 Torch segment reduce/kinetics；上游 observer 测试增加 `NVALCHEMI_TEST_BACKEND` 载体 | 解除 Logging/energy-drift 在显式 reference dynamics 中触发 `_segmented_max` 或 kinetic Warp 的隐式阻断；`None`/`warp` 仍保留上游 Warp，`auto`/`torch_reference` 只显式选择 Torch reference。日志输出 backend（csv/tensorboard/custom）与计算 backend 分离。 | **部分候选**：可选 backend 边界和无 Warp observer 辅助函数可向上游提议；Hygon Torch 具体实现与默认选择需单独评审。 | `reports/g2-upstream-observer-reference.md`、`test/compatibility/test_public_dynamics_reference.py`；CPU observer `44 passed, 33 skipped` + compatibility `8 passed`、HCU observer `77 passed` + compatibility `8 passed`；默认反向单测在 Warp 边界失败；工作树待提交 |
| LP-012 | `packages/framework/nvalchemi/hooks/periodic.py`、`packages/framework/test/dynamics/test_hook_utils.py`、`packages/framework/test/dynamics/test_periodic_hook.py` | 将周期包裹 helper 的 Warp/ops 导入延迟到 Warp 分支；新增 `backend`/`compute_backend` 边界及 `_dynamics_reference.periodic` Torch custom op；上游工具/周期 hook 测试增加 `NVALCHEMI_TEST_BACKEND` 载体 | 解除 `test_hook_utils.py` 在模块收集阶段的隐式 Warp 阻断，并为 reference dynamics 的周期坐标维护提供与上游相同的原地语义；`None`/`warp` 保留默认 Warp，`auto`/`torch_reference` 仅显式走 Torch reference。WrapPeriodicHook 可从 workflow backend 继承计算后端。 | **部分候选**：延迟可选导入与 backend 边界可向上游提议；Hygon reference 实现和 Torch custom op 需单独评审。 | `packages/framework/test/dynamics/test_hook_utils.py`、`test_periodic_hook.py`；`probes/hooks_utils_reference.py`；CPU utility `30 passed, 5 skipped`、periodic hook `15 passed, 11 skipped`；项目环境 DTK 26.04、BW200/gfx936 HCU 的 probe 退出码 `0`，上游 utility+periodic reference 回归 `61 passed`（含 CUDA compile smoke）；受限沙箱中的 `No HIP GPUs are available` 仅表示 `/dev/kfd` 不可见；默认反向单测在 `_segmented_max` Warp 边界失败；工作树待提交 |

清单维护规则：今后每次修改导入的上游文件，必须在同一提交或紧邻提交中新增一条
记录，包含文件、符号/范围、动机、默认路径影响、upstream candidate 标记和回归
测试指针。同步上游前先逐条重新审阅这些记录；不得直接在 `external/` 修改来绕过
清单。当前工作树尚未提交的代码、测试和报告仍以真实 Git diff 为准，不能把本表
当成已提交状态的替代品。
