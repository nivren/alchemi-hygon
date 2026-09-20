# 后端能力矩阵

更新时间：2026-09-20。此表是 registry 当前的真实登记，不替代 `FEATURE_COMPATIBILITY.yaml` 的完整特性契约。

| operation | Torch reference 已登记宽度 | 梯度 | `auto` 当前选择 | HCU 证据 |
| --- | --- | --- | --- | --- |
| `neighbor_list` | Torch reference no-PBC/periodic full/half、`cell_list`、MATRIX/COO、距离/向量；HIP 仅 periodic/fixed-cell/Batch/full/MATRIX，支持 `max_neighbors=None` 自动扩容 | 拓扑不可微；Torch geometry 保留一阶/二阶候选路径 | `torch_reference`（HIP 不进入 `auto`） | Torch periodic/full 与 cell-list reference 已验证；HIP framework selection→native build/query/topology 在 gfx936/BW200 `HIP_VISIBLE_DEVICES=4` 的 46×32、92×64 通过；Point39 API wall-clock、Point40 contract 和 Point41 自动容量增长/上界 gate 通过，仍不是完整上游 capability 或性能准入 |
| `lj_energy_forces` | no-PBC/PBC full、no-PBC half、energy/force | 至二阶 | `torch_reference` | G1 CPU/HCU 窄 slice |
| `velocity_verlet` | 固定晶胞 reference | 一阶 | `torch_reference` | G2 CPU/HCU 窄 slice |
| `fire` | 固定晶胞 reference | 一阶 | `torch_reference` | G2 CPU/HCU 窄 slice |
| `fire2` | 固定晶胞 reference | 一阶 | `torch_reference` | G2 CPU/HCU 窄 slice |
| `kinetics` / `segmented_reduce` | 按 graph 归约 | 一阶 / forward | `torch_reference` | G2 CPU/HCU 窄 slice |
| `periodic_wrap` | 原地周期坐标包裹 | forward | `torch_reference` | G2 CPU/HCU 窄 slice |

`triton` 没有已登记 capability；`hip` 只有上述显式窄 capability，不进入 `auto`，unsupported request 必须抛出 `BackendUnavailableError`。`backend=None` 与 `backend="warp"` 不属于 registry executor，保持 framework 的 legacy Warp 路径。

Torch `cell_list` 当前通过 `backend="torch_reference", method="cell_list"` 公开选择；HIP
`cell_list` 只通过显式 `backend="hip"` 覆盖 periodic/fixed-cell/Batch/full/MATRIX。两者的
`max_neighbors=None` active neighbor 集合已验证，但 HIP 本地 doubling capacity policy 不承诺
上游最小 16/16 对齐 padded shape。真实性能、skin/rebuild lifecycle、MACE/FIRE2 和更宽 public
capability 仍未完成验证。

## M1 单次选择传递

固定晶胞的 NVE、FIRE、FIRE2、LJ model，以及 periodic、kinetics、segmented
reduction 和 observer 辅助路径，均由 framework 按 operation 解析一次
`BackendSelection`，再传给 dispatcher 的通用 executor binding；binding 按 catalog 声明的
entrypoint lazy load，不会用原始 backend 请求再次解析，也不维护 operation 分派表。
FIRE 与 FIRE2 使用独立的 operation/implementation ID：
`torch_reference.fire-v1` 和 `torch_reference.fire2-v1`。

变胞 FIRE/FIRE2 默认继续冻结为 legacy Warp；显式请求当前 Torch reference 会因
缺少 `variable_cell` capability 显式失败。该轮只验证 CPU selection propagation，未新增
HCU、Triton 或 HIP 生产实现证据。

## 数据层默认

`LevelStorage(..., backend=None)` 使用 `TorchStorageBackend`。这是无 Warp 数据模型和 HCU 路径的正式默认；`WarpStorageBackend` 必须显式传入。`WARP-EQUIV-001` 在没有 Warp 的环境为 strict xfail，在 NVIDIA/Warp 环境运行同一 put 合同对照。

## Warp-only 白名单

以下模块可以顶层导入 Warp，但只能通过明确的 Warp/NVIDIA 功能路径到达：

- `data/buffer_kernels.py`、`data/warp_storage_backend.py`
- `dynamics/_ops/{cell_align,langevin,nose_hoover,npt_nph,thermostat_utils}.py`
- `models/_ops/lj.py`、`models/_ops/electrostatics/ewald.py`

reference 公共导入和已登记的 reference operation 不得导入上述模块。新的顶层 Warp 依赖必须先加入白名单并补导入边界测试，或下沉至显式路径。
