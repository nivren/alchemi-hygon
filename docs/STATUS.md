# 当前开发状态

## 交接初始状态

- 交接日期：2026-09-05。
- 已有：项目目的、架构建议、功能兼容策略、探针计划和服务器初始化说明。
- 未完成：服务器盘点、上游提交锁定、代码导入、环境构建、任何 DCU 算子与端到端测试。
- 目标 DCU 型号/卡数/显存/互联：未知，待本机核查。
- DTK/PyTorch/Triton/HIP/通信库组合：未知，待本机核查。
- 用户优先模型与数据集：未指定。默认探测 MACE，按真实依赖可用性选首个模型。
- 人力、机器独占条件、发布时间：未确定。历史工期仅供估算。

## 下一步

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

### 2026-09-05：探针环境 NumPy 修复

- 从 `/home/wangleping/codes/nvalchemi-toolkit/.venv` 复用 NumPy 1.26.4，项目 `.venv` 当前实际版本为 1.26.4；没有重新安装或替换海光 Torch/Triton。
- `.venv/bin/python probes/torch_probe.py --device cpu` 退出 0；新增 NumPy↔Torch 转换检查通过，FP64 一阶/二阶梯度、segment、FFT 仍通过。证据：`artifacts/g0/torch_cpu_numpy1264.json`、`artifacts/g0/torch_cpu_numpy1264.stderr`。
- stderr 只有环境日志 `Could not open /var/log/hylog/.`，不再有 NumPy ABI 警告。环境冻结已更新为 NumPy 1.26.4。
