# Torch reference 邻居基线（Tier 1 前）

日期：2026-09-06  
代码基线：`b5fd0ae580aaec92340c856675cb613b65c8e9b0`；上游锁定
framework `4dfe3723def34df3fadb245981081ccf8c94c257`、ops
`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。  
模型配置：`MACE-OFF23_small.model`，`cutoff=4.5`，`COO`，
`compute_neighbors(backend="torch_reference")`。探针只构建邻居，不调用 MACE
forward 或 FIRE2；因此表中时间是当前参考邻居实现（含周期 pair/image 计算和
Python 装配）的基线，不是端到端 MACE 性能。

## 可重跑命令

项目环境由 `scripts/activate_hygon_env.sh project` 加载，Torch/Triton 未被替换。
每个 case 执行一次 cold、一次 warmup 和两次 steady，设备计时边界显式同步。
CPU 命令如下；HCU 命令在同一命令前加 `HIP_VISIBLE_DEVICES=0`，并在设备节点可见
的主机权限终端运行：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops OMP_NUM_THREADS=1 \
  timeout 90 .venv/bin/python -u probes/neighbor_baseline.py \
  --device cpu --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --scale-root /data/csp_data/perf_46 \
  --scale-root /data/csp_data/perf_92 \
  --scale-root /data/csp_data/perf_184 \
  --scale-root /data/csp_data/perf_368 \
  --batch-sizes 1 2 4 8 --warmup 1 --steady 2
```

小体系 batch 到 32：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops OMP_NUM_THREADS=1 \
  timeout 120 .venv/bin/python -u probes/neighbor_baseline.py \
  --device cpu --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --batch-root /data/csp_data/perf_92 --batch-sizes 1 4 8 16 32 \
  --skip-scales --skip-mixed --warmup 1 --steady 2
```

大体系 batch=4/8/16 使用同一命令，将 `--batch-root` 改为
`/data/csp_data/perf_184`。46 原子 batch=1/4/8/16/32 使用
`/data/csp_data/perf_46`。HCU 运行替换 `--device cuda` 并设置
`HIP_VISIBLE_DEVICES=0`；本轮均退出码 0。逐步 JSON 原始日志位于：

- `artifacts/g2/neighbor-baseline-cpu.log`
- `artifacts/g2/neighbor-baseline-hcu-scales.log`
- `artifacts/g2/neighbor-baseline-cpu-batch46.log`、`neighbor-baseline-hcu-batch46.log`
- `artifacts/g2/neighbor-baseline-cpu-batch92.log`、`neighbor-baseline-hcu-batch92.log`
- `artifacts/g2/neighbor-baseline-cpu-batch184-16.log`
- `artifacts/g2/neighbor-baseline-cpu-batch184.log`、`neighbor-baseline-hcu-batch184.log`
- `artifacts/g2/neighbor-baseline-hcu-mixed.log`

`neighbor-baseline-cpu-batch184.log` 是首次合并运行的部分日志，batch=4/8 已完整，
batch=16 在 timeout 前未完成；最终 batch=16 数据来自独立的
`neighbor-baseline-cpu-batch184-16.log`。第一次使用 `tee` 的管道掩盖了 timeout
退出码，后续独立命令已用 `pipefail`/单 case 校正，未将截断运行记为通过。

## 单体系规模阶梯

steady 为两次 steady 样本均值，边数为完整 periodic COO 边数。

| 体系 | 原子数 | 边数 | CPU cold (s) | CPU steady (s) | HCU cold (s) | HCU steady (s) |
|---|---:|---:|---:|---:|---:|---:|
| `perf_46` | 46 | 824 | 0.0919 | 0.0886 | 4.7514 | 0.0600 |
| `perf_92` | 92 | 1,702 | 0.3256 | 0.3128 | 0.1229 | 0.1212 |
| `perf_184` | 184 | 3,200 | 1.2160 | 1.2076 | 0.2327 | 0.2287 |
| `perf_368` | 368 | 6,872 | 5.0313 | 4.8142 | 0.7394 | 0.4967 |

CPU 从 46 到 368 原子时 steady 约增长 54 倍，接近当前逐体系 dense pair/image
路径的二次增长；边/原子约保持在 17.4–18.7。HCU cold 样本包含首次设备运行开销，
不能与 steady 混比。

## 等长 batch 阶梯

### 46 原子结构

| batch | 总原子 | 边数 | CPU steady (s) | HCU steady (s) |
|---:|---:|---:|---:|---:|
| 1 | 46 | 824 | 0.0898 | 0.0606 |
| 4 | 184 | 3,396 | 0.3294 | 0.2447 |
| 8 | 368 | 6,828 | 0.6322 | 0.4962 |
| 16 | 736 | 13,810 | 1.2849 | 1.0976 |
| 32 | 1,472 | 27,478 | 2.4234 | 2.0947 |

### 92 原子结构

| batch | 总原子 | 边数 | CPU steady (s) | HCU steady (s) |
|---:|---:|---:|---:|---:|
| 1 | 92 | 1,702 | 0.3139 | 0.1239 |
| 4 | 368 | 6,590 | 1.2411 | 0.4701 |
| 8 | 736 | 13,422 | 2.6075 | 1.0977 |
| 16 | 1,472 | 27,484 | 5.1349 | 2.0976 |
| 32 | 2,944 | 54,760 | 9.2946 | 4.2626 |

### 184 原子结构

| batch | 总原子 | 边数 | CPU steady (s) | HCU steady (s) |
|---:|---:|---:|---:|---:|
| 4 | 736 | 12,704 | 4.9089 | 0.8971 |
| 8 | 1,472 | 27,600 | 9.9217 | 2.0442 |
| 16 | 2,944 | 54,448 | 19.8595 | 4.0736 |

### 异构 batch

`[46,92,184,368]` 共 690 原子、12,598 条边，`batch_ptr=[0,46,138,322,690]`。
CPU steady 为 6.5289 s，HCU steady 为 0.8983 s。所有输出的 `batch_ptr` 非递减，
边数和体系分层保持稳定；探针没有跨体系边断言，因为它只测邻居构建流程，已有
异构功能测试覆盖该语义。

## 判断与下一步

这些数据证明在真实 periodic CIF、异构/等长 Batch 和 batch=32 小体系上，当前
reference 邻居构建可在 CPU/HCU 完成，并提供 Tier 1 的可比较数值。HCU 运行时本机
存在共享负载，绝对值和波动不作为发布性能承诺。

当前代码已经把 pair/image 距离计算移到每体系 Torch 张量，但仍在
`_periodic_neighbor_rows` 和矩阵写回处保留 Python 逐边装配。下一步按已确认策略只
做 Tier 1：先改周期分支为 device-side `nonzero`/行内 rank/scatter，保持 shift、
自相互作用、重叠错误、行内顺序、padding 和 overflow 语义；随后用本报告全部 CPU/HCU
case 做集合/边数回归。Tier 2 cell-list 和后续 Triton/HIP 注册仍待 Tier 1 正确性与
基准结果之后决定。
