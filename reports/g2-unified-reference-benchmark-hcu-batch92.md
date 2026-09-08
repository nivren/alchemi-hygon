# G2 unified reference benchmark：HCU `perf_92` batch 阶梯

日期：2026-09-08。此次里程碑固定 `/data/csp_data/perf_92`，测等长 92 原子结构的
batch=1/4/8/16/32；每个 batch 同时运行 periodic full 和 no-PBC full。no-PBC half
已在单体系阶梯及前一轮 smoke 覆盖，本轮不重复以缩短设备占用。

## 命令与环境

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 360 \
  .venv/bin/python -u probes/unified_reference_benchmark.py \
  --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --skip-scales --batch-root /data/csp_data/perf_92 \
  --batch-sizes 1 4 8 16 32 --skip-e2e --warmup 1 --steady 2
```

退出码 `0`。设备为 `BW200, UBB BW1000`，source DTK 26.04，
`HIP_VISIBLE_DEVICES=0`。所有 case 的 `cross_system_edges=0`，periodic shift shape、
边数和 `batch_ptr` 均通过。

## steady 结果

steady 是一次 warmup 后两次 steady 样本的均值；时间单位为秒。

| batch | 总原子数 | periodic 边数 | periodic full | no-PBC 边数 | no-PBC full |
|---:|---:|---:|---:|---:|---:|
| 1  | 92   | 1,702  | 0.003768 | 1,536  | 0.001856 |
| 4  | 368  | 6,590  | 0.009789 | 4,742  | 0.003931 |
| 8  | 736  | 13,422 | 0.017819 | 10,434 | 0.006705 |
| 16 | 1,472 | 27,484 | 0.033571 | 20,642 | 0.012232 |
| 32 | 2,944 | 54,760 | 0.065577 | 39,024 | 0.023316 |

periodic batch=32 的 `0.065577 s` 与已有 Tier-1 scatter 基线 HCU steady `0.0658 s`
一致，边数 `54,760` 也一致，说明统一 harness 的计时/数据构造与既有证据连续。
periodic 从 batch=1 到 32 约增加 `17.4×`，no-PBC 约增加 `12.6×`；这组数据仍是
邻居单独成本，不等于 MACE/FIRE2 端到端吞吐。

## cold 样本与限制

batch=1 periodic cold 为 `4.1465 s`，包含本进程首次 HCU context/kernel 初始化；后续
batch 的 cold 样本不用于 steady 比较。HCU 可能存在共享负载，绝对时间只作当前设备
相对证据。下一步仍需运行 periodic `[46,92]` MACE/FIRE2 固定晶胞 100 步，才能用
端到端成本作为 cell-list 决策输入。

