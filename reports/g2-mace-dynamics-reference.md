# G2 MACE fixed-cell dynamics reference

日期：2026-09-06  
目标设备：BW200 / UBB BW1000（gfx936），`HIP_VISIBLE_DEVICES=0`  
环境：本项目 `.venv`，通过 `source scripts/activate_hygon_env.sh project` 加载 DTK 26.04；PyPI 配置为北外镜像。  
权重：`/home/wangleping/.cache/mace/MACE-OFF23_small.model`

## 范围

本探针使用公共 `NVE`、`FIRE`、`FIRE2` 类，显式选择 `backend="torch_reference"`，并由 `MACEWrapper.make_neighbor_hooks(neighbor_backend="torch_reference")` 在 `BEFORE_COMPUTE` 构建邻居表。输入是两个不同原子数的非周期体系（H₂O 3 原子、CH₄ 5 原子），因此同时检查了 Batch 边界和模型/邻居/动力学组合。

这一步覆盖固定晶胞、非 PBC、无 switching、无 virial/stress 的短运行；不覆盖变胞、NVT/Langevin、checkpoint/restart、长轨迹守恒或生产 Triton/HIP 后端。

## 可重跑命令

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 240 \
  python probes/mace_dynamics_reference.py \
    --device cuda --mode all --steps 2 \
    --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model
```

CPU 同一探针也在项目 `.venv` 运行通过：

```bash
PYTHONPATH=packages/framework:packages/ops .venv/bin/python \
  probes/mace_dynamics_reference.py --device cpu --mode all --steps 1 \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model
```

## 结果

HCU 命令退出码为 `0`，设备识别为 `BW200, UBB BW1000`。三个模式均报告：

| 项目 | 结果 |
|---|---|
| `batch_ptr` | `[0, 3, 8]` |
| 原子数 | `[3, 5]` |
| 邻居边数 | `26` |
| 跨体系边 | `0` |
| 后端 | `torch_reference` |
| finite | `true` |

末步输出如下：

```text
NVE:   energy=[[-2078.119873046875], [-1103.0587158203125]], force_norm=4.206514358520508
FIRE:  energy=[[-2078.1201171875], [-1103.0587158203125]], force_norm=4.207834720611572
FIRE2: energy=[[-2078.119873046875], [-1103.0587158203125]], force_norm=4.2064948081970215
```

项目 `.venv` 的兼容测试 `test_public_dynamics_reference.py` 与 import-boundary 测试共 `6 passed`；该测试使用 DemoModel 覆盖同一异构 Batch 和三个公共类，证明不依赖 Warp 的 `BaseDynamics.run()` 组合路径。

## 结论与边界

因此，velocity-Verlet、固定晶胞 FIRE/FIRE2 的 reference 实现已经可以和真实 MACE wrapper 组成短的端到端 MD/relaxation 链，并且在 HCU 上验证了异构 batching。它还不是完整的生产验收：缺少长轨迹能量漂移、收敛 Hook/DataSink、inflight 补位、周期 dynamics、变胞 stress、重启和性能后端。下一步应把这条组合接到上游 dynamics 测试载体和最小轨迹输出，再分别排期这些独立特性。
