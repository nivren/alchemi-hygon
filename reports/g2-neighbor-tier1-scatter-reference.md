# G2 periodic 邻居 Tier 1 scatter 回归

日期：2026-09-06  
范围：`nvalchemiops.torch_reference.neighbor_list` 的 periodic full-list MATRIX/COO 装配

本轮只替换周期邻居的装配阶段。pair/image 距离和向量的几何计算保持不变；原先
`active_indices.detach().cpu().tolist()` 加逐边 Python `append`/写回，改为设备端
`nonzero`、`bincount`、行内 rank 和二维索引写回。仍保留每个 system 的几何候选枚举，
因此这不是 cell-list，也没有改变 half-list、skin、switching、virial 或动态扩容范围。

## 语义验证

CPU ops reference 回归：

```text
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_reference_backend.py
10 passed in 5.34s
```

CPU framework Hook 回归：

```text
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py
16 passed in 3.37s
```

主机权限 HCU 回归使用 `source scripts/activate_hygon_env.sh project`、DTK 26.04、
`HIP_VISIBLE_DEVICES=0`，同一 ops 测试文件为 `10 passed in 18.91s`，退出码 0，设备为
BW200/UBB BW1000（gfx936）。

framework 的 `compute_neighbors` 接线在同一环境下运行：

```text
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/models/test_neighbors_torch_reference.py
3 passed in 1.51s  # CPU
3 passed in 15.53s # HCU
```

两次均退出码 0，覆盖异构 Batch 边界、PBC shift 写回和未注册 backend 的显式失败。

已有 periodic contract 覆盖并保持通过：

- 正交与三斜胞的 signed shift 约定 `r_ij = r_i - r_j - shift @ cell`；
- 自相互作用排除、重合 active pair 报错和容量溢出报错；
- 多体系 batch offset、空行、MATRIX/COO 输出和距离/向量输出；
- 不导入 Warp 的 reference 入口。

## 同口径前后时间

CPU/HCU 的正式比较均在项目环境、`OMP_NUM_THREADS=1` 和相同 warmup/steady 口径下
完成。CPU 命令的完整形式为：

```text
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops OMP_NUM_THREADS=1 timeout 120 \
  .venv/bin/python -u probes/neighbor_baseline.py --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --batch-root /data/csp_data/perf_92 --batch-sizes 32 \
  --skip-scales --skip-mixed --warmup 1 --steady 2
```

HCU 命令为：

```text
source scripts/activate_hygon_env.sh project && \
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
timeout 120 .venv/bin/python -u probes/neighbor_baseline.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --batch-root /data/csp_data/perf_92 --batch-sizes 32 \
  --skip-scales --skip-mixed --warmup 1 --steady 2
```

此前有一次未设置 `OMP_NUM_THREADS=1` 的 CPU 快速试跑得到 `1.0508 s`，由于与 Tier 1
前基线口径不同，不计入比较；正式结果使用上面的统一环境变量。

| 92 原子、batch=32 | Tier 1 前 steady | Tier 1 scatter steady | 变化 |
|---|---:|---:|---:|
| CPU，54,760 边 | 9.2946 s | 7.7101 s | 约 1.2× 加速 |
| HCU，54,760 边 | 4.2626 s | 0.0658 s | 约 64.7× 加速 |

Tier 1 后 cold 时间为 CPU `9.0479 s`、HCU `0.0685 s`；HCU 首个小体系 case 的 cold
样本包含设备首次运行开销。HCU 机器存在共享任务，steady 绝对值只作为本轮相对比较，
不能作为发布性能承诺。

## Tier 1 后覆盖

所有完整 case 退出码均为 0，边数和 `batch_ptr` 与 Tier 1 前保持一致。steady 均值如下：

| case | 边数 | CPU (s) | HCU (s) |
|---|---:|---:|---:|
| `perf_46` | 824 | 0.0655 | 0.0032 |
| `perf_92` | 1,702 | 0.2799 | 0.0038 |
| `perf_184` | 3,200 | 1.1712 | 0.0057 |
| `perf_368` | 6,872 | 4.6541 | 0.0140 |
| 46 原子 batch=4 | 3,396 | 0.2534 | 0.0076 |
| 46 原子 batch=8 | 6,828 | 0.4580 | 0.0137 |
| 46 原子 batch=16 | 13,810 | 0.8760 | 0.0258 |
| 46 原子 batch=32 | 27,478 | 1.7693 | 0.0492 |
| 92 原子 batch=4 | 6,590 | 1.1308 | 0.0096 |
| 92 原子 batch=8 | 13,422 | 2.2677 | 0.0176 |
| 92 原子 batch=16 | 27,484 | 4.1307 | 0.0338 |
| 92 原子 batch=32 | 54,760 | 7.7101 | 0.0658 |
| 184 原子 batch=4 | 12,704 | 4.7218 | 0.0179 |
| 184 原子 batch=8 | 27,600 | 9.3460 | 0.0338 |
| 184 原子 batch=16 | 54,448 | 18.1132 | 0.0660 |
| 异构 `[46,92,184,368]` | 12,598 | 6.1197 | 0.0214 |

原始输出保存在：

- `artifacts/g2/neighbor-tier1-cpu-scales-batch46.log`
- `artifacts/g2/neighbor-tier1-cpu-batch92.log`
- `artifacts/g2/neighbor-tier1-cpu-batch184.log`
- `artifacts/g2/neighbor-tier1-cpu-batch184-16.log`
- `artifacts/g2/neighbor-tier1-hcu-scales-batch46.log`
- `artifacts/g2/neighbor-tier1-hcu-batch92.log`
- `artifacts/g2/neighbor-tier1-hcu-batch184.log`

`neighbor-tier1-cpu-batch184.log` 中 batch=4/8 完整，batch=16 为提前停止的合并运行；
表中 CPU batch=16 使用独立的 `neighbor-tier1-cpu-batch184-16.log`，其退出码为 0。

Tier 1 前的完整规模和 batch 基线仍保留在
`reports/g2-neighbor-baseline-reference.md`，没有被新结果覆盖。

## 当前边界与下一步

本轮证明 periodic full-list reference 的装配替换保持语义；HCU 上逐边 Python 开销显著
下降，而 CPU 大体系仍受 dense pair/image 几何计算限制。`neighbor_list` 的 no-PBC
per-atom 循环、cell-list 算法、半邻居、PBC 容量压力、skin 异步重建以及 Triton/HIP
registry 仍未因此完成。下一步应分别评估 no-PBC 装配向量化和 torch reference 内的
cell-list，不能把本轮时间直接外推为完整生产性能。
