# ASE-compatible BFGS 使用与验证范围

本地新增功能，基于 Framework 0.2.0/Ops 0.4.1；没有升级到上游候选版本。
数学对齐 ASE 3.29.0 BFGS。固定晶胞与 UnitCellFilter 的完整 `3N+9` 坐标
共用数值核心；异构 Batch 每个体系有独立 Hessian、历史及参考晶胞。

## Ops：可直接运行的 CPU 示例

在项目环境中执行下列代码。它用解析力演示异构 Batch 与配对重启，不是 MLIP。

```python
import torch
from nvalchemiops.bfgs_ase import (
    ASEBFGSState, bfgs_ase_prepare_state, bfgs_ase_batch_step,
)

backend = "torch_reference"
device = "cpu"
positions = [
    torch.full((n, 3), 0.25, dtype=torch.float64, device=device)
    for n in (2, 3)
]
states = [
    bfgs_ase_prepare_state(q.numel(), device=device, system_id=i, backend=backend)
    for i, q in enumerate(positions)
]
diagnostics = bfgs_ase_batch_step(
    positions, [-q for q in positions], states, backend=backend,
)
assert [state.step_count for state in states] == [1, 1]

# 调用方同时保存几何、优化器状态；模型力在恢复后重新计算。
saved_positions = [q.clone() for q in positions]
saved_states = [state.state_dict() for state in states]
restored_positions = [q.clone() for q in saved_positions]
restored_states = [ASEBFGSState.from_state_dict(s) for s in saved_states]
bfgs_ase_batch_step(
    restored_positions, [-q for q in restored_positions], restored_states,
    backend=backend,
)
```

在已准入 HCU 环境使用同一接口时，设置 `device="cuda"`，并显式设置
`backend="hip.bfgs-ase-jacobi-streams-v1"`。所有输入与状态必须在同一设备、
连续、FP64、无梯度、无共享 storage，并具有独立 system ID；空 Batch 报错。
位置原位更新，力只读。所有尺寸组通过数值检查后才统一提交；失败保留全部状态。

## Framework：连接已经配置的模型

```python
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell

# model、neighbor_hooks、batch 由现有 MLIP/Batch 工作流提供。
optimizer = ASECompatibleBFGSVariableCell(
    model=model,
    hooks=neighbor_hooks,
    backend="hip.bfgs-ase-jacobi-streams-v1",
    fmax=0.05,
    maxstep=0.2,
    mask=None,
    cell_factor=None,
    hydrostatic_strain=False,
    constant_volume=False,
    orig_cell=None,
    scalar_pressure=0.0,
)
relaxed = optimizer.run(batch, n_steps=500)
```

固定晶胞换用 `ASECompatibleBFGS`，去掉 Filter 参数。模型需要返回 energy/forces；
变胞还需要 ASE 正拉应力 `stress`。Batch 预先配置独立 energy/forces/cell/stress
张量，多体系配置唯一整数 `system_id`。需要邻居的模型使用 BEFORE_COMPUTE 的
NeighborListHook，变胞时按当前晶胞重建；不要接入改变 ASE 坐标图的晶胞对齐 Hook。

`cell_factor=None` 为每体系的原子数；mask 使用 ASE 六元 Voigt 顺序或 3×3 矩阵。
参数逐体系应用。`constant_volume` 沿用 ASE 去迹投影，不保证每步体积精确恒定。
`maxstep` 与严格 `< fmax` 判据作用于广义坐标；广义力收敛不等于物理原子力
也低于同一个阈值。完整约定见 [Filter 合同](TORCH_ASE_UNITCELL_FILTER_CONTRACT.md)。

用 `bfgs_states`、`last_steps`、`filter_states` 按 ID 查询。重排保留历史，退出释放，
补位初始化新状态。optimizer `state_dict()` 与配对 Batch 几何一起保存，恢复时
调用 `load_state_dict()`；模型、sampler 游标、pipeline/status 由调用方另存。
不支持直接读取 ASE JSON restart。

## 后端与交付范围

| 选择 | 实现 | 当前边界 |
|---|---|---|
| 默认 / `torch_reference` | Torch 数值基线及 `torch.linalg.eigh` | CPU FP64 逐步 ASE 已验证；HCU 库内可能走主机数学路径 |
| `hip` | rocSOLVER strided-batched Jacobi | 本机已准入构建；不同宽度组串行 |
| `hip.bfgs-ase-jacobi-streams-v1` | 本地动态 HIP Jacobi | 同宽 batched；异宽同进程、同卡双 worker/双 stream |

没有 147/285 维或每组四矩阵的经验性限制；实际受显存、整数表示与分解成本约束。
新实现仍有主机调度及标量收敛同步；未开启 auto，单宽或部分输入可能更慢。
`None`/`warp`/`auto` 对此新增优化器不提供可用实现；未支持的请求明确报错。

HIP 当前准入为 gfx936、DTK26.04、海光 Torch2.9.0+das.opt1.dtk2604 / HIP6.3.26093。
wheel 携带原生源码，首次使用 HIP 后端按目标环境 JIT 编译，需要匹配的 DTK
工具链与已有 Torch 扩展依赖；构建产物放缓存，不随 wheel 发布预编译 `.so`。
产品不依赖外部 `agy-eigh-test`；其源码与介绍仅用于调查和比选。

真实验证使用 MACE MPA-0 medium，非 MACE-OFF small。已完成小样本完整变胞
CSP、Li/T2 异构短步、冻结/重排/补位与重启门；未验收全目录生产吞吐、其他设备/
构建、NVIDIA 硬件、compile、优化器反向梯度、其他 Filter 或 Atoms constraints。
实际结果见 [M5-4 报告](../reports/g2-ase-bfgs-m5-4.md)及
[最终交付审阅](../reports/g2-ase-bfgs-delivery-review.md)。
