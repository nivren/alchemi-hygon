# 后端能力矩阵

更新时间：2026-10-04。此表是 registry 当前的真实登记，不替代 `FEATURE_COMPATIBILITY.yaml` 的完整特性契约。

| operation | Torch reference 已登记宽度 | 梯度 | `auto` 当前选择 | HCU 证据 |
| --- | --- | --- | --- | --- |
| `neighbor_list` | Torch reference no-PBC/periodic full/half、`cell_list`、MATRIX/COO、距离/向量；HIP periodic/Batch/full/MATRIX，调用间 full-PBC 当前晶胞完整重建，支持 `max_neighbors=None` 自动扩容 | 拓扑不可微；Torch geometry 保留一阶/二阶候选路径 | `torch_reference`（HIP 不进入 `auto`） | Torch periodic/full 与 cell-list reference 已验证；HIP framework selection→native build/query/topology 在 gfx936/BW200 `HIP_VISIBLE_DEVICES=4` 的 46×32、92×64 通过；Point39 API wall-clock、Point40 contract 和 Point41 自动容量增长/上界 gate 通过；M1 HCU4 BW1000 64G 当前晶胞 ops22/framework4项通过，仍不是完整上游 capability 或性能准入 |
| `lj_energy_forces` | no-PBC/PBC full、no-PBC half、energy/force | 至二阶 | `torch_reference` | G1 CPU/HCU 窄 slice |
| `velocity_verlet` | 固定晶胞 reference | 一阶 | `torch_reference` | G2 CPU/HCU 窄 slice |
| `fire` | 固定晶胞 reference | 一阶 | `torch_reference` | G2 CPU/HCU 窄 slice |
| `fire2` | 固定晶胞 reference；M3 原生下三角六项变胞 | 固定晶胞一阶；变胞 forward-only | `torch_reference` | 固定晶胞历史证据；M3 HCU4 单步/解析势，目标 MPA-0 HCU 未运行 |
| `fire_cell` | fork 完整3×3 FIRE pre/post、联合控制、共同位移限制 | forward-only | `torch_reference` | M3 HCU4 单步/解析势；CPU MPA-0 三步；见 [报告](../reports/g2-variable-cell-fire-m3.md) |
| `nose_hoover` | 固定晶胞 NHC，C1–8、Yoshida3/5、FP32/64 | 原地 MD forward-only；functional v输入CPU一二阶已验证 | `torch_reference` | HCU4 12组CPU/HCU对照、短真实MACE；见NHC报告 |
| `stress_to_cell_force` | row-cell `[B,3,3]`、FP32/64、完整/下三角投影、strided/empty | 一阶/二阶；CPU gradcheck/gradgradcheck；HCU 一阶 | `torch_reference` | HCU4 转换/stream、周期非正交真实 MACE；FP64 有限差分；见 [报告](../reports/g2-torch-cell-stress-force.md) |
| `cell_tools` | align、六项 pack/unpack、mass 与 extend；FP32/64、单体系/异构/empty；append 与 interleaved 索引分别保留 | forward-only，requires_grad 拒绝 | `torch_reference`（既有选择规则） | HCU4 六组 primitive CPU parity；未含模型/优化步；见 [报告](../reports/g2-cell-tools-m2.md) |
| `kinetics` / `segmented_reduce` | 按 graph 归约 | 一阶 / forward | `torch_reference` | G2 CPU/HCU 窄 slice |
| `periodic_wrap` | 原地周期坐标包裹 | forward | `torch_reference` | G2 CPU/HCU 窄 slice |

`triton` 没有已登记 capability；`hip` 只有上述显式窄 capability，不进入 `auto`，unsupported request 必须抛出 `BackendUnavailableError`。`backend=None` 与 `backend="warp"` 不属于 registry executor，保持 framework 的 legacy Warp 路径。

Torch `cell_list` 当前通过 `backend="torch_reference", method="cell_list"` 公开选择；HIP
`cell_list` 只通过显式 `backend="hip"` 覆盖 periodic/Batch/full/MATRIX，允许调用间 full-PBC 变胞并完整重建。两者的
`max_neighbors=None` active neighbor 集合已验证，但 HIP 本地 doubling capacity policy 不承诺
上游最小 16/16 对齐 padded shape。当前晶胞快照见 [M1报告](../reports/g2-hip-current-cell-neighbor-m1.md)。
旧 plan 变胞复用、性能准入、skin/rebuild lifecycle、HIP MACE/FIRE2 和更宽 public capability
仍未完成验证。

## M1 单次选择传递

固定晶胞的 NVE、FIRE、FIRE2、LJ model，以及 periodic、kinetics、segmented
reduction、stress-to-cell-force 和 observer 辅助路径，均由 framework 按 operation 解析一次
`BackendSelection`，再传给 dispatcher 的通用 executor binding；binding 按 catalog 声明的
entrypoint lazy load，不会用原始 backend 请求再次解析，也不维护 operation 分派表。
FIRE 与 FIRE2 使用独立的 operation/implementation ID：
`torch_reference.fire-v1` 和 `torch_reference.fire2-v1`。

M1 当时未登记变胞 capability；M3 已新增 `torch_reference.fire2-cell-v1` 和
`torch_reference.fire-cell-v1`，梯度0，分别为六项与完整矩阵原生优化步；默认 None/warp
仍保留。两种优化器冻结 stress selection，FIRE2 同时冻结 cell_tools selection。
M3 单步/解析势 HCU 与 MPA-0 CPU 三步证据单列，不代表目标模型 HCU 或生产实现。

## 数据层默认

`LevelStorage(..., backend=None)` 使用 `TorchStorageBackend`。这是无 Warp 数据模型和 HCU 路径的正式默认；`WarpStorageBackend` 必须显式传入。`WARP-EQUIV-001` 在没有 Warp 的环境为 strict xfail，在 NVIDIA/Warp 环境运行同一 put 合同对照。

## Warp-only 白名单

以下模块可以顶层导入 Warp，但只能通过明确的 Warp/NVIDIA 功能路径到达：

- `data/buffer_kernels.py`、`data/warp_storage_backend.py`
- `dynamics/_ops/{cell_align,langevin,thermostat_utils}.py`
- `models/_ops/lj.py`、`models/_ops/electrostatics/ewald.py`

reference 公共导入和已登记的 reference operation 不得触发上述模块的顶层 Warp 依赖。新的顶层 Warp 依赖必须先加入白名单并补导入边界测试，或下沉至显式路径。

NHC dispatcher 的 Warp 导入已下沉到 legacy 函数，reference 公共导入不加载 Warp；
[本次报告](../reports/g2-torch-nvt-nhc.md)区分 CPU chain ABI 的 eager compile/opcheck
与未验证的完整 workflow/HCU compile、DomainParallel。

`npt_nph.py` 保留旧 stress import 路径，barostat/Warp 依赖已下沉到调用内；基础 reference
转换由 `_ops/stress.py` 接线。该导入隔离不构成 NPT/NPH 或变胞 FIRE/FIRE2 后端支持。
转换的 CPU compile/opcheck 已验证，HCU 二阶/compile 与 legacy Warp 数值未验证。

## MPA-0 CPU 单点基线

MPA-0 M1 使用现有 torch_reference neighbor/geometry 与 stress conversion。
三个真实 CIF 的 CPU FP32/64 energy/forces/stress、FP64 Batch/ASE 与有限差分已验证，
见 [报告](../reports/g2-mace-mpa0-m1-cpu-baseline.md)。这不新增 ops capability：
M1 只提供单点基线；M3 变胞 optimizer capability 与 CPU MPA-0 三步已单列，
MPA-0 HCU/完整收敛和批量应用仍待 M4/M5。
单一精度作业须显式配置默认 dtype、模型权重和输入一致；禁止静默转换输出掩盖提升。
