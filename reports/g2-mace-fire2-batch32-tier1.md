# G2 MACE + FIRE2 32×92 弛豫（Tier 1 邻居装配后）

## 范围

本次运行验证邻居 periodic full-list Tier 1 device-side scatter 修改后的真实
端到端组合：32 个 92 原子周期 CIF 在同一个 `Batch` 中使用
`MACE-OFF23_small.model` 和 `FIRE2(backend="torch_reference")` 做固定晶胞
弛豫。收敛条件为逐体系 `fmax < 0.01 eV/Å`，最大步数明确设置为 2000。

除必填的初始步长 `dt=0.01` 外，FIRE2 使用锁定上游的默认参数；邻居
reference 使用 `skin=0.5`。模型前向、邻居写回和优化器状态更新均在同一
HCU batch 中进行。

## 可重跑命令

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops \
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 230 \
  .venv/bin/python -u probes/mace_fire2_batch_relaxation.py \
  --device cuda \
  --data-root /data/csp_data/perf_v2_sorted/perf_v2_92 \
  --count 32 --start 0 \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --fmax 0.01 --max-steps 2000 \
  --log-every 20 --max-wall-seconds 190 \
  | tee artifacts/g2/mace-fire2-batch32-tier1.log
```

命令通过 `scripts/activate_hygon_env.sh project` 加载 DTK 26.04 和项目
`.venv`；HCU 设备节点在主机权限终端可见。外层 `timeout` 为 230 秒，探针
内部墙钟保护为 190 秒。

## 结果

- 设备：`BW200, UBB BW1000`，HCU `gfx936`，`HIP_VISIBLE_DEVICES=0`。
- 退出码：`0`；状态：`passed`。
- `num_structures=32`，`num_nodes=2944`，`batch_ptr=[0,92,...,2944]`，每个
  体系均为 92 原子；最终邻居边数为 `75010`。Batch 边界由连续的
  `batch_ptr` 保持；该探针未另行输出跨体系边计数。
- 32/32 体系收敛；最后一个体系在第 `834` 步达到阈值。总耗时
  `79.54159364895895 s`，低于 190 秒内部保护和 2000 步上限。
- 重新在最终坐标上评估模型后，逐体系最终 `fmax` 最大值为
  `0.009995493106544018 eV/Å`，全部满足 `0.01 eV/Å` 阈值。
- 模型依赖真实 `mace-torch`/`e3nn` 组合；运行日志提示
  `cuEquivariance` 不可用，因此本次使用 MACE 的 Torch 路径。

逐步 JSON 日志和完整 stdout 保存在
`artifacts/g2/mace-fire2-batch32-tier1.log`，包括每 20 步的 active fmax、
已收敛体系数和 elapsed time。

## 收敛统计与能量/密度指标

为补齐原始探针未输出的能量和密度字段，使用完全相同的输入、模型、FIRE2
参数、`max_steps=2000` 和 HCU 环境进行了指标重跑。该次运行同样为 `32/32`
收敛、最大步数 `834`，耗时 `79.80547558492981 s`。按每个体系首次达到
`fmax<0.01` 的步数统计：

- 最少收敛步数：`234`
- 最多收敛步数：`834`
- 平均收敛步数：`483.46875`
- 平均最终总能量：`-78382.53393554688 eV/structure`
- 平均最终每原子能量：`-851.9840812683105 eV/atom`
- 平均固定晶胞密度：`0.5551176927983761 g/cm³`

指标重跑的完整 JSON 日志为
`artifacts/g2/mace-fire2-batch32-metrics-hcu.log`；密度按
`sum(masses)/cell_volume` 从 CIF 计算，因本次为固定晶胞，收敛前后密度相同。

## 证据边界

该结果是 Tier 1 邻居装配后的单 HCU、固定晶胞、周期、等长 32×92
`torch_reference` 组合证据。它验证了真实 MACE/FIRE2 batching 与 2000 步
上限下的收敛流程，不构成 cell-list、Triton/HIP 生产 kernel、变胞 stress、
NVT/Langevin、checkpoint/restart、长轨迹 occupancy 或无干扰性能结论。

## 2026-09-08 原参数复跑

按上面的同一数据、模型和参数重新运行（`fmax=0.01`、`max_steps=2000`、
`dt=0.01`、`skin=0.5`、单 HCU、`OMP_NUM_THREADS=1`），结果为：

- `status=passed`，32/32 体系收敛，`step_count=835`；
- `num_nodes=2944`，最终邻居边数 `75074`；
- `max_final_fmax=0.009996769018471241 eV/Å`；
- `elapsed_s=81.58037368883379`，设备为 `BW200, UBB BW1000`。

这与此前第 834 步、约 79.5--79.8 秒的结果一致，差异属于 HCU 运行波动；
两次均只作为单卡固定晶胞 reference 功能证据，不作为无干扰性能基线。
本次完整日志见 `artifacts/g2/mace-fire2-batch32-rerun-20260908.log`。
