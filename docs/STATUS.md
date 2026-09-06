# 当前开发状态

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

## 当前快照（2026-09-06）

- N0 能力探针已完成：主机单卡 Triton vector-add 通过（BW200/UBB BW1000，冷启动约 0.519 s，预热稳态约 24.6 μs/次）；主机双卡 RCCL/NCCL all-reduce 和双向 P2P 通过。证据见 `reports/g0-capability-probes.md`。这只解除 Triton 基础编译/执行和 RCCL 原语的架构未知，不代表生产 kernel、LJ ownership 或 DomainParallel 已验证。
- 环境加载修正：`scripts/activate_hygon_env.sh` 在 bash 使用 `/opt/dtk-26.04/env.sh`，在 zsh 使用 `/opt/dtk-26.04/env.zsh`；沙箱内 `/dev/kfd` 不可见，GPU 结果均来自主机权限探针。
- 代码基线：Torch reference 的 Batch/邻居/PBC/LJ/skin 纵向切片已在 CPU 和部分 BW200/gfx936 HCU 通过；MACE wrapper 本地 checkpoint 路径修复已实现并有回归测试。
- MACE 证据：用户缓存的 `MACE-OFF23_small.model` direct model 在探索环境和项目 `.venv` 的 CPU/HCU 通过；两个 `perf_46` CIF 的 framework `MACEWrapper + compute_neighbors(torch_reference)` batching 在探索环境和项目 `.venv` 的 CPU/HCU 通过。项目环境批次为 `batch_ptr=[0,46,92]`、1754 条边、跨体系边 0。周期 reference 邻居向量化后，单个 perf_46 邻居约 5.33 秒、完整链约 9.76 秒。详细报告见 `reports/g1-mace-wrapper-batch.md`。
- 当前未验证：更大规模 reference 邻居性能、完整 dynamics/弛豫、训练 wrapper 混合二阶梯度、生产 cuEquivariance/Triton/HIP kernel、多卡域分解、PhysicsNeMo profiling 和 DomainParallel。向量化前双 perf46 超时记录仍保留为优化前证据。
- 依赖检查：北外镜像 dry-run 解析到 `mace-torch==0.3.15`、`e3nn==0.4.4`、`matscipy==1.1.1`、`ase==3.29.0` 等 45 个包，随后安装到项目 `.venv`；另补齐 framework 基础依赖，Torch `2.9.0+das.opt1.dtk2604`、Triton `3.3.0+das.opt1.dtk2604.torch290` 未被替换。PhysicsNeMo 未安装，单进程路径由可选导入保持可用。
- 环境规格已补齐：项目 `.venv` 现在包含 `pytest==8.4.2`、`pytest-asyncio==1.4.0`；直接输入见 `configs/hygon-reference.in`，当前主机精确冻结见 `configs/hygon-reference-lock.txt`，冻结脚本为 `scripts/freeze_hygon_env.sh`，报告快照见 `reports/probe-environment-freeze.txt`。冻结中的 Torch/Triton URI 是主机本地海光 wheel，换机时必须先提供同版本 wheel；PhysicsNeMo 不属于 Hygon reference 安装集。
- 特性状态已复核：`status` 表示完整目标契约的实现阶段，`verification.dcu_status` 表示已列出的 HCU 证据范围；因此 MACE、LJ、neighbors.topology 的完整条目仍是 `planned`，但其 reference 子路径为 `partial`，`neighbors.skin_rebuild` 为 `implemented/partial`。本轮纠正了 neighbors.topology 的 HCU 状态，并在 `FEATURE_COMPATIBILITY.yaml` 写明两轴语义，避免把窄 reference slice 写成完整特性通过。
- 项目环境 pytest 基线：ops reference `10 passed`、framework optional-import/neighbor Hook `11 passed`，均退出码 `0`；两包测试需分开启动以避开上游都使用顶层 `test` 包名造成的 `ImportPathMismatchError`。详细命令见 `reports/g1-project-reference-pytest.md`。
- 可重建性检查：`uv pip sync --dry-run --python .venv/bin/python ... configs/hygon-reference-lock.txt` 在北外镜像上解析并核对 `77 packages`，退出码 `0`，显示 `Would make no changes`。
- skin/rebuild 当前进展：Torch reference 已对 Batch 中变化的 system 做 eager 局部重建，并保持全局索引、MATRIX/COO 写回和未变化 system 的缓存；两体系 CPU/HCU probe 均通过，报告见 `reports/g1-skin-rebuild-batch-reference.md`。Hook staging 已有自动容量处理，算子层仍保留显式 overflow 防御。
- N1 staging 容量契约已完成窄 reference slice：Hook 支持 16 对齐 grow-and-retry、idle shrink、override floor、异构 Batch 和 6 次交替 skin rebuild；framework Hook 回归 `16 passed`，CPU/HCU probe 均退出码 `0`，报告见 `reports/g1-neighbor-capacity-reference.md`。算子层显式 `NeighborOverflowError` 仍保留。
- 当前未完成：PBC 容量压力、周期 half-list、生产 cell-list、规模化性能和异步 rebuild；这些不因 N1 reference 通过而提前标记为 verified。
- N2a 首个切片已完成：新增顶层 `nvalchemi._dynamics_reference.velocity_verlet`，不触发 `nvalchemi.dynamics` 或 Warp，覆盖原位 position/half-kick/final-kick、异构 per-system `dt`、float32/64 和显式输入检查。项目 `.venv` CPU 测试 `5 passed`；同一项目 `.venv` 在 source DTK 26.04、BW200/gfx936 HCU 探针退出码 `0`，最大位置误差 `6.94e-18`、final velocity 最大绝对值 `0.0`。证据见 `reports/g2-dynamics-reference-velocity-verlet.md`。
- N2a 第二个切片已完成：新增顶层 `nvalchemi._dynamics_reference.kinetics`，用 Torch `index_add_` 实现异构 Batch 的 kinetic energy 和 `3N` temperature。项目 `.venv` CPU 回归 `7 passed`（含 VV 与 kinetics）；同一项目 `.venv` 在 source DTK 26.04、BW200/gfx936 HCU 探针退出码 `0`，动能 `[7.0, 6.0]`、温度 `[27077.2089507397, 46418.07248698234]`，无 Warp 导入。证据见 `reports/g2-dynamics-reference-kinetics.md`。
- 本轮仍未改动上游 `nvalchemi.dynamics._ops.velocity_verlet.py`、`hooks/_utils.py` 或完整 `NVE`，默认 Warp 路径保持不变；下一步移植固定晶胞 FIRE/FIRE2 reference，随后由 N2b 接入公共 dynamics 后端委派和 hooks。

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
