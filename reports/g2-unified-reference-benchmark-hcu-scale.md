# G2 unified reference benchmark：HCU 单体系规模阶梯

日期：2026-09-08。此次里程碑只测邻居构建，不包含 Batch 阶梯或 MACE/FIRE2 端到端；
目标是用同一计时边界比较 periodic full、no-PBC full 和 no-PBC half 随单体系规模的变化。

## 命令与环境

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 300 \
  .venv/bin/python -u probes/unified_reference_benchmark.py \
  --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --scale-sizes 46 92 184 368 \
  --skip-batches --skip-e2e --warmup 1 --steady 2
```

退出码 `0`。设备为 `BW200, UBB BW1000`，source DTK 26.04，
`HIP_VISIBLE_DEVICES=0`。所有 case 的 `cross_system_edges=0`，periodic case 的
image shift shape 与边数契约通过。

## steady 结果

steady 是一次 warmup 后两次 steady 样本的均值；时间单位为秒。

| 原子数 | periodic full 边数 | periodic full | no-PBC full 边数 | no-PBC full | no-PBC half 边数 | no-PBC half |
|---:|---:|---:|---:|---:|---:|---:|
| 46  | 824  | 0.003243 | 626  | 0.001853 | 313  | 0.001887 |
| 92  | 1,702 | 0.003702 | 1,536 | 0.001852 | 768  | 0.001875 |
| 184 | 3,200 | 0.005753 | 2,524 | 0.001859 | 1,262 | 0.001894 |
| 368 | 6,872 | 0.013969 | 5,412 | 0.001880 | 2,706 | 0.001900 |

周期 full 从 46 到 368 原子约增加 `4.31×`；此范围内 HCU 并行性掩盖了部分
dense pair/image 路径的二次增长。no-PBC 两条路径基本处于约 `1.85–1.90 ms`
的平台，说明当前规模下固定启动/分配开销已经占主导；这不是 cell-list 已经
解决复杂度的证据。

## cold 样本与限制

46 原子 periodic 的 cold 样本为 `4.208951 s`，明显高于其 steady `3.243 ms`，
这是本进程首次 HCU context/kernel 初始化开销，不与后续规模的 cold 值混作算法
性能。其余 periodic cold 值为 92/184/368 原子的 `5.225/8.127/18.581 ms`。

本机 HCU 仍可能存在共享负载；本报告用于建立统一相对基线，不是发布吞吐承诺。
完整 cell-list 决策仍必须结合后续 batch 阶梯和 periodic `[46,92]` MACE/FIRE2
100 步端到端成本。

