# A4 MACE/邻居 Torch reference 阶段性能基线

日期：2026-09-06  
设备：项目 `.venv`；HCU 运行使用 DTK 26.04、BW200/UBB BW1000（gfx936）、`HIP_VISIBLE_DEVICES=0`。模型为本地 `MACE-OFF23_small.model`，邻居为 `compute_neighbors(backend="torch_reference")`。运行期间本机 8 张 HCU 有其他任务占用，以下时间是共享设备负载下的阶段定位和相对趋势，不是干净的绝对性能基线。

## 计时口径

`probes/mace_wrapper_stages.py` 显式同步设备后分别记录：

- 模型加载和 Batch 构造；
- 首次邻居构建和首次 MACE forward；
- 预热步；
- 稳态每步的邻居构建、输入适配和 MACE forward。

MACE forward 计时包含 `wrapper.model.forward(compute_force=True)` 和输出适配；邻居计时包含一次完整的 Torch reference 邻居构建。每个 profile 使用同一结构重复构造 Batch，暂未加入 FIRE2 坐标变化或 skin 缓存。

## `[46,92]` 混合 Batch

命令：

```bash
# CPU
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python probes/mace_wrapper_stages.py --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --cif /data/csp_data/perf_46/formal_c1_1_100_z1_46.cif \
       /data/csp_data/perf_v2_sorted/perf_v2_92/formal_c1_2_1000_z1_46.cif \
  --warmup-steps 1 --steady-steps 2

# HCU
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python probes/mace_wrapper_stages.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --cif /data/csp_data/perf_46/formal_c1_1_100_z1_46.cif \
       /data/csp_data/perf_v2_sorted/perf_v2_92/formal_c1_2_1000_z1_46.cif \
  --warmup-steps 2 --steady-steps 5
```

| device | nodes | cold neighbors | cold model | steady neighbors | steady input | steady MACE | steady total |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 138 | 0.959 s | 6.563 s | 0.539 s | 0.026 s | 4.661 s | 5.226 s |
| HCU | 138 | 15.607 s | 0.708 s | 0.191 s | 0.0008 s | 0.025 s | 0.217 s |

HCU 稳态 profile 的 5 个样本均已越过首轮 MACE JIT；此前 2 步试跑的首个稳态 MACE 为约 1.51 秒，随后降至约 0.025 秒，因此首步必须与后续稳态分开看。

## HCU 规模阶梯（等长 92 原子结构）

使用同一个 92 原子 CIF 重复构造 1、3、11 个体系，分别约为 92、276、1012 原子；每次 `warmup-steps=1`、`steady-steps=2`，命令只改变 `--cif` 重复次数。

| nodes / systems | edges | cold neighbors | steady neighbors | steady MACE | steady total |
|---:|---:|---:|---:|---:|---:|
| 92 / 1 | 1702 | 16.073 s | 0.125 s | 0.0226 s | 0.148 s |
| 276 / 3 | 5106 | 15.564 s | 0.568 s | 0.801 s | 1.369 s |
| 1012 / 11 | 18722 | 17.012 s | 1.477 s | 0.805 s | 2.283 s |

276 和 1012 的首个稳态 MACE 分别为约 1.57 秒，后续样本约 0.026/0.033 秒；表中均值保留了两个样本，不能作为最终稳态吞吐率。冷邻居在这三个点约为 15.6–17.0 秒，当前短样本没有显示简单的 `N²` 冷启动增长，但还不足以证明算法复杂度。

## HCU 混合规模阶梯

混合阶梯使用真实 46/92 原子 CIF 重复构造 Batch：约 276 原子为 `[46,92,46,92]`，约 1012 原子为 6 个 46 原子和 8 个 92 原子。

| nodes / systems | edges | cold neighbors | steady neighbors | steady MACE | steady total |
|---:|---:|---:|---:|---:|---:|
| 276 / 4 | 5038 | 14.603 s | 0.560 s | 0.758 s | 1.319 s |
| 1012 / 14 | 18778 | 14.756 s | 1.622 s | 0.806 s | 2.429 s |

两组混合 Batch 的 `batch_ptr` 分别为 `[0,46,138,184,276]` 和 `[0,46,92,138,184,230,276,368,460,552,644,736,828,920,1012]`。首个稳态 MACE 分别为约 1.49 秒和 1.56 秒，后续样本明显下降，仍需与后续稳态分开。

## 已观测同步点

当前 profile 明确同步了以下边界：

1. 模型加载和 Batch 构造计时前后；
2. 每次邻居构建计时前后；
3. `adapt_input` 完成后；
4. MACE forward 和输出适配完成后。

这些是 profile 为取得设备时间所需的同步点，不是完整运行时同步审计。FIRE2 hook、HostMemory 写出、skin/rebuild 触发路径和跨 stream 行为仍需单独登记。

## 当前判断与下一步

- HCU `[46,92]` 的稳态主要由邻居 reference 构建贡献，MACE forward 已低于邻居成本；CPU 则主要由 MACE forward 贡献。
- 冷启动和首个稳态步包含设备/JIT/cache 成本，不能和后续稳态直接比较。
- 当前 HCU 共享负载会影响所有绝对耗时和波动；不能据此比较空闲 CPU/HCU，或直接决定 cell-list/Triton/HIP 优先级。
- 当前 profile 已覆盖等长和混合规模阶梯，但尚未覆盖坐标变化触发的 skin rebuild、FIRE2 端到端分项、完整 host-device 同步清单或 1000 原子长轨迹。
- 这些数据只用于决定是否推进 cell-list；任何 Triton/HIP kernel 仍需通过 ADR 0003/0004 的数值、梯度、编译和回退门槛。
- 干净性能基线需要目标卡空闲或获得独占/低干扰窗口；不终止其他用户任务，也不把当前共享负载数据写成发布性能。
