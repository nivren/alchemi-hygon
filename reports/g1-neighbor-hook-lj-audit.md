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

结果：`4 passed`，退出码 `0`。测试使用独立 FP64 LJ 公式检查距离 `1.1`/`1.2` 的能量和力，比较 full/half 邻居约定，并验证 energy gradient 与返回 force 的符号关系。

这也固定了本阶段的验证顺序：优先复用上游 framework/ops 测试作为回归基线，仅为 DCU 后端边界和上游未覆盖的约束补充测试；数值方面先以独立 FP64 解析式和 Torch CPU reference 建立基线，再用 HCU 复核，ASE 保留为可选的独立交叉检查工具。组件当前显式接受的 backend 字符串是阶段性能力边界，后续将统一到中央 registry。

## Dynamics stage 隔离与最小纵向链

`DynamicsStage` 已移到不导入 dynamics 包的轻量模块 `nvalchemi._dynamics_stage`，`nvalchemi.dynamics.base` 仍重新导出同一枚举。`BaseModelMixin.make_neighbor_hooks()` 因此能在无 Warp 环境创建 reference Hook，并继续把模型 backend 传给 Hook。

新增的纵向测试先用 CPU reference 运行 `model.make_neighbor_hooks()`，调用 Hook 写入邻居矩阵，再调用 `LennardJonesModelWrapper(backend="torch_reference")`。与独立 FP64 LJ 解析结果比较能量，并检查总力守恒；邻居 Hook 与 LJ 测试合计 `13 passed`，另覆盖共享 `DynamicsStage` 的 stage timing 域识别。同一探针在 source DTK 26.04 的 BW200/gfx936 上使用探索环境运行成功，设备识别为 `BW200, UBB BW1000`，两体系能量为 `[-0.9833724493736826, -0.8909652875830761]`，总力为零，退出码 `0`；摘要见 `artifacts/g1/framework_neighbor_lj_reference_hcu0.json`。项目 `.venv` 首次因缺少 `plum` 阻断，补齐后继续暴露缺少 `jaxtyping`。该链仍不覆盖 PBC、skin/rebuild、switching、virial/stress、NVE 或分布式路径。

上游 `test/hooks/test_stage_timing_hook.py` 在当前探索环境无法收集，因为它直接导入 `nvalchemi.dynamics.base`，而 dynamics 包初始化仍要求 Warp；因此没有把该上游测试记为通过。新增的无 Warp stage-domain 测试只保护本次轻量枚举隔离和域识别边界。

## 后续最小纵向实现

后续实现应保持默认参数和 Warp 路径不变，并增加显式 backend 入口：

1. 以解析/FP64 对照为基线，接入 G1 的 PBC 邻居集合和短 NVE 闭环。
2. 把 skin/rebuild、switching 和 virial 拆成独立特性，分别建立契约和证据。

本步没有修改 Hook 或 LJ 生产代码，也没有宣称动态 MD、NVE、PBC 或 stress 已支持。
