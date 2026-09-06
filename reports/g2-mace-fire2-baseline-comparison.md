# A1 MACE/FIRE2 异构与单体系基线对照

日期：2026-09-06  
范围：同一 46 原子周期 CIF、同一 92 原子周期 CIF，分别组成 mixed `[46,92]`、`single_46` 和 `single_92` 三个 fresh Batch；使用 MACE OFF23_small、固定晶胞 `FIRE2(backend="torch_reference")`、`skin=0.5`。本报告的首轮只运行 3 步 smoke，不宣称体系已经收敛。

## 验收内容

- 每个 case 的 `batch_ptr` 和读回轨迹 `batch_ptr` 与原子数一致。
- `batch_idx` 非递减，邻居边没有跨体系连接。
- `system_id` 和 `trajectory_step` 能恢复每一帧的体系边界。
- mixed 与两个 single case 的逐体系最终 `fmax` 做数值对照。
- 三个 case 都使用独立 fresh Batch，状态不会由前一个 case 污染。

## CPU smoke

```bash
PYTHONPATH=packages/framework:packages/ops \
  timeout 100 .venv/bin/python probes/mace_fire2_baseline_comparison.py \
  --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --max-steps 3 --log-every 1 --max-wall-seconds 80
```

退出码 `0`。三步结果：

| case | final `fmax` | neighbor edges | elapsed | stored `batch_ptr` | cross edges |
|---|---:|---:|---:|---|---:|
| mixed `[46,92]` | `[0.7451217, 0.8051754]` | 3284 | 21.13 s | `[0,46,138,184,276,322,414]` | 0 |
| single 46 | `[0.7451227]` | 1062 | 13.58 s | `[0,46,92,138]` | 0 |
| single 92 | `[0.8051778]` | 2222 | 16.85 s | `[0,92,184,276]` | 0 |

mixed 与 single 的最终 `fmax` 差值分别为 `9.54e-7`（46 原子）和 `2.44e-6`（92 原子），低于比较阈值 `5e-3`。三个 case 均为 `status=0`，这是预期的三步 smoke 结果，不是未收敛失败。

## HCU smoke

主机 DTK 26.04 环境、`HIP_VISIBLE_DEVICES=0`、BW200/UBB BW1000（gfx936）：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 100 \
  .venv/bin/python probes/mace_fire2_baseline_comparison.py \
  --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --max-steps 3 --log-every 1 --max-wall-seconds 80
```

退出码 `0`。三步结果：

| case | final `fmax` | neighbor edges | elapsed | stored `batch_ptr` | cross edges |
|---|---:|---:|---:|---|---:|
| mixed `[46,92]` | `[0.7451196, 0.8051743]` | 3284 | 16.92 s | `[0,46,138,184,276,322,414]` | 0 |
| single 46 | `[0.7451206]` | 1062 | 1.76 s | `[0,46,92,138]` | 0 |
| single 92 | `[0.8051739]` | 2222 | 2.21 s | `[0,92,184,276]` | 0 |

mixed 与 single 的最终 `fmax` 差值分别为 `1.01e-6`（46 原子）和 `4.17e-7`（92 原子），低于比较阈值 `5e-3`。三个 case 均为 `status=0`。日志显示 mixed case 包含主要的首次设备/模型预热成本；这不是 A4 性能结论。运行输出还报告 `cuequivariance` 未安装，当前仍是 MACE reference 路径。

## HCU 收敛对照

在同一 DTK 26.04/HCU gfx936 环境中，将三个 case 分开运行，使用 `fmax=0.01`、最多 2000 步、`--require-converged`，每个 case 内部墙钟上限 90 秒：

```bash
for case in mixed single_46 single_92; do
  HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
    PYTHONPATH=packages/framework:packages/ops timeout 120 \
    .venv/bin/python probes/mace_fire2_baseline_comparison.py \
    --device cuda --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
    --max-steps 2000 --fmax 0.01 --log-every 20 \
    --max-wall-seconds 90 --only-case "$case" --require-converged
done
```

每个 case 均在主机权限 shell 中单独执行，实际运行前加载 `source scripts/activate_hygon_env.sh project`。

| case | steps | final `fmax` | elapsed | neighbor edges | cross edges |
|---|---:|---:|---:|---:|---:|
| mixed `[46,92]` | 425 | `[0.0094126, 0.0098724]` | 47.22 s | 3290 | 0 |
| single 46 | 367 | `[0.0090830]` | 36.46 s | 1078 | 0 |
| single 92 | 425 | `[0.0098400]` | 39.93 s | 2210 | 0 |

三个 HCU case 均退出码 `0`，所有体系 `status=1`。mixed 与 single 的最终 `fmax` 都低于 `0.01`，收敛状态独立；轨迹中 mixed 的 `system_id`、`trajectory_step` 和 `batch_ptr` 均保持有序。

## CPU 收敛边界

为保持短超时，另运行了 CPU mixed 收敛可行性检查：

```bash
PYTHONPATH=packages/framework:packages/ops timeout 60 \
  .venv/bin/python probes/mace_fire2_baseline_comparison.py \
  --device cpu --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --max-steps 2000 --fmax 0.01 --log-every 20 \
  --max-wall-seconds 45 --only-case mixed
```

该命令只完成并输出了第 1 步（约 7.08 秒），未在预算内完成收敛，随后被外部超时终止；没有残留进程。CPU 三步 smoke 仍通过，因此当前 CPU 证据覆盖结构/数值 sanity，不覆盖完整收敛对照。

## 当前结论与边界

本轮已证明异构 Batch 与两个单体系在 CPU/HCU smoke 中的体系边界、邻居 ownership、快照帧序和短程数值行为一致，并完成 HCU 上 `fmax=0.01` 的 mixed/single 收敛对照。A1 的 HCU 证据已闭合；CPU 完整收敛仍受当前 reference 路径性能限制，暂不进行无监控长等待。下一步是同步窄 slice 契约和 STATUS，再进入 A4 三口径性能基线。
