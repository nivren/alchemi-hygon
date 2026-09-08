# G2 unified reference benchmark：CPU harness smoke

日期：2026-09-08。此次里程碑只验证 benchmark harness 的数据构造、计时边界
和结果 schema；不把 CPU smoke 当作 HCU 性能结论。

## Harness

新增 `probes/unified_reference_benchmark.py`，统一覆盖：

- periodic full-list；
- no-PBC full/half；
- cold、warmup、steady neighbor-only 计时；
- periodic `[46,92]` MACE/FIRE2 fixed-cell 端到端模式；
- Batch 边界、跨体系边、最大邻居数和 periodic image shift 检查。

no-PBC 使用与 periodic case 相同的 CIF 坐标但关闭 PBC，作为算法边界测量；
当前真实 MACE 生产链仍以 periodic case 为主。端到端模式默认运行 100 步，
CPU smoke 使用 2 步验证载体，HCU 正式基线必须使用 100 步。

## 输入确认

“periodic 46/92”指的是原子数加上 CIF 中的三维周期边界条件，不是把一个
非周期分子强行命名为 periodic。当前按排序后的第一个 CIF 选取：

| case | CIF | 原子数 | source `pbc` | 晶胞体积 (Å³) |
|---|---|---:|---|---:|
| periodic 46 | `/data/csp_data/perf_46/formal_c1_1_100_z1_46.cif` | 46 | `[true,true,true]` | 1405.25 |
| periodic 92 | `/data/csp_data/perf_92/formal_c1_2_1000_z1_46.cif` | 92 | `[true,true,true]` | 2623.32 |

no-PBC 行仍使用这两个 CIF 的坐标和晶胞，但在传入邻居算子前将 effective
`pbc` 设为全 `false`；新 harness 同时记录 `source_pbc` 和 `effective_pbc`，避免
把这两种口径混淆。

## CPU neighbor smoke

命令：

```bash
PYTHONPATH=packages/framework:packages/ops OMP_NUM_THREADS=1 \
  .venv/bin/python -u probes/unified_reference_benchmark.py \
  --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --scale-sizes 46 --batch-sizes 1 --warmup 1 --steady 2 --skip-e2e
```

退出码 `0`。同一次进程内 periodic/no-PBC 以及 no-PBC half 均通过：

- periodic 46：824 edges，steady mean 约 `55.31 ms`，shift shape `[824,3]`；
- no-PBC 46 full：626 edges，steady mean 约 `0.790 ms`；
- no-PBC 46 half：313 edges，steady mean 约 `0.801 ms`；
- periodic batch 1（92 原子）：1702 edges，steady mean 约 `278.89 ms`；
- no-PBC batch 1（92 原子）：1536 edges，steady mean 约 `1.153 ms`。

所有 case 的 `cross_system_edges=0`，结果 schema 和边界检查通过。

92 原子 periodic case 显著慢于 46 原子并不矛盾：当前 reference 周期实现按体系
构造 `delta[N,N,3]`，再对候选周期 image 计算距离并执行 `nonzero`。这不是按最终
edge 数线性工作；在两个输入上候选 image 数都为 512，因此主要的 dense pair/image
张量工作从 `46²×512` 增长到 `92²×512`，约为 4 倍。当前 CPU smoke 的 steady
均值为约 `55.31 ms → 278.89 ms`（约 5.0 倍），额外差异来自张量分配、归约和
缓存/调度开销。92 原子 case 的边数约为 2.1 倍，不能用边数预测该实现的耗时。
这解释的是当前 reference 邻居 kernel 的 CPU 成本；HCU 统一阶梯和周期 `[46,92]`
端到端 100 步仍需单独实测，不能从此 CPU smoke 外推。

## CPU end-to-end smoke

命令将同一 harness 限制为 periodic `[46,92]`、2 步：

```bash
PYTHONPATH=packages/framework:packages/ops OMP_NUM_THREADS=1 \
  .venv/bin/python -u probes/unified_reference_benchmark.py \
  --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --skip-scales --skip-batches --e2e-steps 2 --e2e-skin 0.5
```

退出码 `0`；2 步端到端耗时约 `8.30 s`，最后一步邻居边 `3284`，
`stage_timing` 成功记录 `BEFORE_COMPUTE->AFTER_COMPUTE` 等阶段。该阶段计时
包含邻居 Hook 和模型 compute，是总成本分解的辅助数据；邻居单独计时仍来自
neighbor-only cases。

## 尚未完成

- HCU no-PBC/periodic 规模×batch 阶梯；
- HCU periodic `[46,92]` MACE/FIRE2 100 步；
- 低干扰窗口的绝对性能基线；
- cell-list 实现或 `auto` capability 注册。
