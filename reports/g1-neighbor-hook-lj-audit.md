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

## Hook reference slice 实现结果

`NeighborListHook` 现在接受显式 `backend` 参数。默认 `None` 仍初始化并使用原有 Warp 路径；`"torch_reference"`/`"auto"` 不加载 Warp，绕过 staging/rebuild 热路径，调用已有 neighbor dispatcher，再通过共享写回函数写入 MATRIX 或 COO。为保持边界明确，该分支拒绝 `skin != 0`、PBC 和 `method` 选择。`nvalchemi.hooks` 同时把 Warp-backed `WrapPeriodicHook` 改为按需导入，使显式 reference Hook 可以在无 Warp 环境导入。

验证命令：

```bash
PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -m pytest -q \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py
```

结果：`6 passed`，退出码 `0`。覆盖无 Warp 导入、真实 `@torch.compile` 入口、MATRIX/COO、`auto`、method/skin/PBC 的显式失败。该测试在探索环境 CPU 上运行；本轮没有重复受限沙箱 GPU 探针，也没有新增 HCU Hook 运行证据。

## LJ wrapper reference slice 实现结果

`LennardJonesModelWrapper` 现在接受显式 `backend` 参数。默认 `None` 仍延迟加载并使用 Warp custom op；`"torch_reference"`/`"auto"` 使用 ops dispatcher，按 framework 的 per-atom energy 归约到 per-system energy，并返回 Torch autograd force。为避免伪装支持，该分支明确拒绝 PBC、非零 `switch_width`、virial/stress 和分布式 domain decomposition。LJ 模块导入本身不再要求 Warp。

验证命令：

```bash
PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -m pytest -q \
  packages/framework/test/models/test_lj_torch_reference.py
```

结果：`4 passed`，退出码 `0`。测试使用独立 FP64 LJ 公式检查距离 `1.1`/`1.2` 的能量和力，比较 full/half 邻居约定，并验证 energy gradient 与返回 force 的符号关系。单卡 HCU wrapper 运行尚未在本轮新增；此前 ops dispatcher 和 framework `compute_neighbors` 已有 HCU 证据。

当前 reference wrapper 的前向测试使用预先写入的 neighbor matrix；`BaseModelMixin.make_neighbor_hooks()` 现在会传播模型的 backend，但该方法仍属于 dynamics/Warp 导入边界，自动构造 reference Hook 的完整无 Warp 动态路径留待下一步单独处理。

## 后续最小纵向实现

后续实现应保持默认参数和 Warp 路径不变，并增加显式 backend 入口：

1. 隔离 `make_neighbor_hooks()` 所需的 dynamics stage 导入，使 reference 模型可以在无 Warp 环境构造 Hook。
2. 在 HCU 上运行同一条邻居→LJ wrapper 链；确认数值后再考虑把 skin/rebuild、PBC、switching 和 virial 拆成独立特性。

本步没有修改 Hook 或 LJ 生产代码，也没有宣称动态 MD、NVE、PBC 或 stress 已支持。
