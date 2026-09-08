# 后端能力矩阵

更新时间：2026-09-08。此表是 registry 当前的真实登记，不替代 `FEATURE_COMPATIBILITY.yaml` 的完整特性契约。

| operation | Torch reference 已登记宽度 | 梯度 | `auto` 当前选择 | HCU 证据 |
| --- | --- | --- | --- | --- |
| `neighbor_list` | no-PBC full/half、periodic full、MATRIX/COO、距离/向量；periodic+half 明确拒绝 | 拓扑不可微 | `torch_reference` | periodic full 已验证；no-PBC device-side 装配在 BW200 HCU 0 窄 slice 通过 |
| `lj_energy_forces` | no-PBC/PBC full、no-PBC half、energy/force | 至二阶 | `torch_reference` | G1 CPU/HCU 窄 slice |
| `velocity_verlet` / `fire` | 固定晶胞 reference | 一阶 | `torch_reference` | G2 CPU/HCU 窄 slice |
| `kinetics` / `segmented_reduce` | 按 graph 归约 | 一阶 / forward | `torch_reference` | G2 CPU/HCU 窄 slice |
| `periodic_wrap` | 原地周期坐标包裹 | forward | `torch_reference` | G2 CPU/HCU 窄 slice |

`triton`、`hip` 没有任何已登记生产 capability；显式请求必须抛出 `BackendUnavailableError`。`backend=None` 与 `backend="warp"` 不属于 registry executor，保持 framework 的 legacy Warp 路径。

## 数据层默认

`LevelStorage(..., backend=None)` 使用 `TorchStorageBackend`。这是无 Warp 数据模型和 HCU 路径的正式默认；`WarpStorageBackend` 必须显式传入。`WARP-EQUIV-001` 在没有 Warp 的环境为 strict xfail，在 NVIDIA/Warp 环境运行同一 put 合同对照。

## Warp-only 白名单

以下模块可以顶层导入 Warp，但只能通过明确的 Warp/NVIDIA 功能路径到达：

- `data/buffer_kernels.py`、`data/warp_storage_backend.py`
- `dynamics/_ops/{cell_align,langevin,nose_hoover,npt_nph,thermostat_utils}.py`
- `models/_ops/lj.py`、`models/_ops/electrostatics/ewald.py`

reference 公共导入和已登记的 reference operation 不得导入上述模块。新的顶层 Warp 依赖必须先加入白名单并补导入边界测试，或下沉至显式路径。
