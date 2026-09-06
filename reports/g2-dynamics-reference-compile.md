# Reference dynamics `torch.compile` 边界

日期：2026-09-06  
范围：CPU、Torch 2.9.0、Warp-free reference FIRE2 与 kinetics。此探针只登记编译边界，不测 HCU 性能。

## 可重跑命令

```bash
PYTHONPATH=packages/framework:packages/ops \
  timeout 60 .venv/bin/python probes/dynamics_reference_compile.py --device cpu
```

退出码：`0`。

## 结果

| 函数 | 默认 `fullgraph=False` | `fullgraph=True` |
|---|---|---|
| `fire2_step_coord` | 可运行，5 个图、4 次 graph break | 失败：`Unsupported`，验证逻辑中的 Tensor→Python 分支无法追踪 |
| `kinetic_energy_per_graph` | 可运行，1 个图、0 次 graph break | 通过 |
| `temperature_per_graph` | 可运行，2 个图、1 次 graph break | 失败：`Data-dependent branching` |

FIRE2 的 graph break 来自输入检查中的 Tensor 布尔判断和动态分支；temperature 还包含基于 Tensor 的有效性检查。kinetic energy 已使用 `torch.library.custom_op` 与 fake 注册，因此可以作为 compile 兼容形态的先例。

## 契约登记

当前 reference dynamics 的保证范围是 eager 正确性和 CPU/HCU 设备执行；`torch.compile(fullgraph=True)` 暂不属于 FIRE2 或 temperature 的支持契约。默认 `fullgraph=False` 仍可运行，但会拆成多个图，不能宣称完整编译优化。

后续若需要编译支持，应为 FIRE2 和 temperature 增加保持原位 mutation、shape/dtype 校验和 fake/meta 语义的 custom op 外壳，并重新进行数值回归；这不是本轮性能或后端 kernel 工作。
