# HIP 邻居 M1：当前晶胞完整重建

日期：2026-10-05。起点 `fb05b61`，分支 `codex/feat-hip-current-cell-neighbors`。
本阶段已实现并完成下述窄验证；用户确认下一阶段后，收口实现 `fecbc39`。
仅提交到本地阶段分支，未合并或推送。

## 目的与实现

为 MPA-0 + MACE + 变胞 FIRE/FIRE2 建立邻居前置能力：显式 HIP、full-PBC、异构
Batch、full-list、MATRIX、FP32/FP64、`skin=0`，每次求值按当前位置和晶胞完整重建。
详细接口、数值和生命周期边界见[契约](../docs/HIP_CURRENT_CELL_NEIGHBOR_CONTRACT.md)。

来源为锁定 ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` 的
`neighbors/cell_list/kernels.py`、`launchers.py::batch_query_cell_list`。
未修改 `external/`，未更新上游锁定 SHA。

源码审查确认，公共 executor 原本就逐调用派生 inverse、grid dimensions、search radius、
cell offsets，并分配新 CSR buffers 和 trusted plan。本轮将 helper 命名与文档明确为
current-cell，登记 `variable_cell` 快照能力，用真实 HIP gate 验证调用间修改同一晶胞张量。
低层 `NativeHipNeighborPipelineABI.supports_variable_cell=False` 保持：复用旧 plan
时仍要求固定 metadata，公共调用创建新 plan。query kernel 仍为每原子一个线程。

修复两项接线问题：

- 显式容量不足由普通 `RuntimeError` 转为既有 `NeighborOverflowError`，使
  `NeighborListHook` 能在晶胞收缩后发现新容量、扩容并重建；其他错误继续透传。
- 原 registry 会在 `auto` 中考虑 HIP，与既定契约不符。新增 implementation
  `auto_enabled`（默认 True），HIP 设 False；保持 HIP 的显式默认 strategy，
  家族名和 implementation ID 都可显式选择。CPU resolver 回归覆盖该边界。

## 验证证据

环境：项目 `.venv`、DTK26.04、Torch2.9.0、HIP6.3.26093；HCU4
BW1000 64G、`gfx936:sramecc+:xnack-`，`OMP_NUM_THREADS=1`。
作业前只读确认没有 KFD PIDs、HCU4 空闲；结束后无 KFD PIDs。

| Gate | 结果 | 原始日志（`artifacts/hip-current-cell-neighbor-m1/`） |
|---|---|---|
| CPU ops 聚焦回归 | 87 passed、6 skipped、1 warning，退出0 | `cpu-ops.log` |
| CPU framework 独立进程 | 33 passed、4 skipped、1 xfailed、1 warning，退出0 | `cpu-framework.log` |
| HCU4 新增 ops gate | 22 passed，退出0 | `hcu-ops.log` |
| HCU4 新增 framework gate | 4 passed，退出0 | `hcu-framework.log` |
| HCU4 原有 46×32 固定晶胞 gate | passed，退出0 | `hcu-fixed-regression.log` |

CPU skip 为 HCU 专属用例；framework xfail 是既有 `WARP-DEFAULT-001`，需要
NVIDIA/Warp runner 验证默认路径，本阶段未添加 skip/xfail 豁免。warning 是既有
`auto` 选择 Torch reference 提示。以上为聚焦回归，未执行全量 CPU/HCU 套件。
最终 `compileall`、`git diff --check`、特性契约 JSON/唯一 ID/证据路径及新文件空白检查通过。

新增 ops 测试为两体系 3+2 原子，涵盖 initial→shrink→expand→shear→wrap→multi-image→restore。
CPU reference 另以独立周期镜像穷举检验；证明夹具 `[-6,6]^3` 覆盖所有可能邻居，
并避开 cutoff 浮点临界点。HIP 比较含重复项的 `(i,j,shift)` 多重集合与 counts，
验证零镜像自身排除、非零镜像自身和多个 image、全局索引、padding、输入不修改、
旧结果不 alias、默认/非默认 stream，以及显式溢出后的自动增长和恢复。

FP64 独立镜像与 reference 一致的各快照总边数为 `4/10/0/8/8/240/4`。
multi-image 各原子 counts 为 `[56,54,54,38,38]`，包含绝对值至少2的 self-image shift；
收缩前最大 count 为2。FP32 和 HIP 对应集合全部一致。framework 复用同一个 Batch
与 Hook 经收缩、完整三斜剪切、wrap、恢复，实际 count 超过 Hook 初始容量时通过重建。

既有固定晶胞 gate 覆盖 FP32/64、mixed-PBC、空输入、重复 Hook、自动容量 `1→8`
（最大 count6）、显式容量不足，以及 no-PBC/half/COO 请求拒绝。它保持历史契约，
新增变胞 mixed-PBC 不在本轮证据中。

## 复现入口

CPU ops 分进程运行下列六个文件（`HIP_VISIBLE_DEVICES=''`）：

```text
packages/ops/test/torch/test_hip_current_cell_neighbors.py
packages/ops/test/torch/test_hip_neighbor_executor.py
packages/ops/test/torch/test_hip_batch_neighbor_runtime.py
packages/ops/test/torch/test_torch_reference_cell_list.py
packages/ops/test/torch/test_backend_registry.py
packages/ops/test/torch/test_torch_reference_backend.py
```

CPU framework 的独立进程运行 `test/models/test_neighbors_torch_reference.py`、
`test/hooks/test_neighbor_list_torch_reference.py` 和 `test/compatibility/` 下
`test_hip_current_cell_neighbors.py`、`test_backend_default_boundary.py`、
`test_backend_selection_propagation.py`、`test_executor_binding.py`。

HCU4 的两个 pytest 文件与原有探针分别在独立进程中运行，单个 gate 限时180秒。
例如（使用已有 JIT cache 时）：

```bash
source scripts/activate_hygon_env.sh project
export HIP_VISIBLE_DEVICES=4 PYTORCH_ROCM_ARCH=gfx936 MAX_JOBS=1 OMP_NUM_THREADS=1
timeout 180 .venv/bin/python -m pytest -q packages/ops/test/torch/test_hip_current_cell_neighbors.py
timeout 180 .venv/bin/python -m pytest -q packages/framework/test/compatibility/test_hip_current_cell_neighbors.py
timeout 180 .venv/bin/python probes/framework_neighbor_hip_contract.py
```

本次实际使用任务独立缓存；`native_env.sh` 保存五个 `NVALCHEMI_HIP_*_BUILD_DIR`
的配置，`compile_native.py` 在无可见设备的进程加载五个模块，限时420秒。
若复用本任务产物，在每个进程执行前追加 `source artifacts/hip-current-cell-neighbor-m1/native_env.sh`。

首次 HCU 尝试沿用 DTK 多目标架构，冷 JIT 限时180秒退出124，未得到数值结果。
其日志保留为 `hcu-initial-cold-timeout.log`。随后仅 gfx936、MAX_JOBS=1、独立 cache
冷编译成功，五个模块依次约 `64.95/61.709/61.207/61.760/69.541` 秒，见
`cold-build.log`。这些是编译耗时，不能作为 kernel 或端到端性能数据。
环境、资源与夹具统计分别保存在 `environment.json`、`resources-after.log`、
`fixture-counts.json`。

## 限制与下一阶段

本阶段没有 native query 性能优化，也没有性能测量或 auto 准入。host metadata
readback、错误同步、每次 workspace 分配仍保留；未改变原查询线程映射。
COO、HIP MACE 单点及变胞 FIRE/FIRE2、skin/selective rebuild、旧 plan 变胞复用、
no-PBC/half、target/pair、native geometry backward、compile/opcheck、DomainParallel
均未扩展或验收。

下一阶段已获用户确认：在保留 HIP MATRIX 核心的基础上复用设备内 Torch MATRIX→COO
适配，接通 `compute_neighbors`/Hook/MACE；用同 checkpoint、同 dtype、同几何的
Li/Na 小样本比较完整邻居集合与 MPA-0 能量/力/应力。连续几何与模型梯度保持 Torch
路径，之后独立验收变胞 FIRE/FIRE2，再以实测决定 workspace/query/native COO 优化。
