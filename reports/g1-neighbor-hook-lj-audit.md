# G1 NeighborListHook 与 LJ wrapper 调用契约审计

日期：2026-09-05  
上游锁定：framework `4dfe3723def34df3fadb245981081ccf8c94c257`，ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`

## 现有调用链

1. `BaseModelMixin.make_neighbor_hooks()` 读取 `model_config.neighbor_config`，把 `cutoff`、`format`、`half_list` 和 `skin` 传给 `NeighborListHook`。
2. `NeighborListHook.__call__()` 在 `BEFORE_COMPUTE` 阶段调用 `_rebuild()`。`_rebuild()` 维护 staging tensors、`rebuild_flags`、邻居容量和算法 scratch，然后调用 `nvalchemiops.torch.neighbors.neighbor_list(...)` 的原地接口。
3. 原地邻居接口接收预分配的 `neighbor_matrix`、`num_neighbors`、可选 `neighbor_matrix_shifts`，同时接收 `rebuild_flags`、`method` 和 cell-list/cluster-tile scratch。完成后由 `_write_neighbor_data_to_batch()` 写入 Batch。
4. `LennardJonesModelWrapper` 的 `NeighborConfig` 固定为 `MATRIX`，`adapt_input()` 需要全局索引矩阵、计数、可选 PBC shifts 和 cell；`forward()` 调用 `nvalchemi.models._ops.lj` 的两个 Torch custom op。该模块和 custom op 当前直接导入 Warp，力由 Warp 解析计算，virial/stress 与 switching 也属于现有契约。

## 与 Torch reference 的差异

| 契约 | 现有 framework/Warp 路径 | 当前 Torch reference | 结论 |
|---|---|---|---|
| 邻居输出 | 预分配矩阵原位写入，可增容 | 返回新分配的 `(matrix, counts)` | 不能直接塞入现有 staging 调用 |
| rebuild/skin | `rebuild_flags`、`cutoff + skin`、缓存参考坐标 | 不接受 rebuild 参数，也没有 skin 状态 | 首个 reference 分支必须显式限制 `skin=0` |
| PBC | cell、pbc、整数 shifts、naive/cell-list/cluster-tile | 明确 `NotImplementedError` | 不得静默丢弃 PBC；先显式失败 |
| 方法选择 | `method` 和预分配 scratch | 不接受额外 kwargs | reference 不参与现有方法选择器 |
| LJ 力 | Warp analytic force，不要求 positions grad | Torch autograd，要求 `positions.requires_grad` | wrapper 需要单独的梯度和输出适配 |
| LJ switching | 支持 `switch_width` | 明确未实现 | reference 分支先限制 `switch_width=0` |
| virial/stress | 支持矩阵 virial 和 per-system 累加 | 没有 virial 输出 | active `stress` 时必须显式失败 |
| 邻居格式 | LJ 当前只消费 MATRIX | reference LJ 当前只消费 MATRIX | COO 不应在本步伪装成 LJ 支持 |

## 无 Warp 导入审计

在探索环境运行：

```bash
PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python - <<'PY'
import importlib
for name in ("nvalchemi.hooks.neighbor_list", "nvalchemi.models.lj"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: {type(exc).__name__}: {exc}")
PY
```

结果：两个模块均为 `ModuleNotFoundError: No module named 'warp'`。这说明当前已完成的 `compute_neighbors(backend="torch_reference")` 隔离还没有覆盖动态 Hook 和 LJ wrapper；本审计没有把该失败写成 DCU 运行失败，也没有安装 Warp 来掩盖边界。

## 下一条最小纵向实现

后续实现应保持默认参数和 Warp 路径不变，并增加显式 backend 入口：

1. `NeighborListHook(..., backend="torch_reference")` 走一个 `@torch.compiler.disable` 的 reference 分支，调用已有 dispatcher 并复用 `_write_neighbor_data_to_batch()`；第一版只接受无 PBC、`skin=0`、无预分配 scratch，其他组合明确抛出 `BackendUnavailableError` 或 `NotImplementedError`。
2. `LennardJonesModelWrapper(..., backend="torch_reference")` 走 Warp-independent LJ dispatcher；第一版只接受无 PBC、`switch_width=0`、不请求 stress，明确处理 `positions` 的 autograd 需求，并保持 energy/force 的 full/half 归约和 per-system scatter 语义。
3. 先补 CPU contract tests，再在 HCU 上运行同一条邻居→LJ 链；确认数值后再考虑把 skin/rebuild、PBC、switching 和 virial 拆成独立特性。

本步没有修改 Hook 或 LJ 生产代码，也没有宣称动态 MD、NVE、PBC 或 stress 已支持。
