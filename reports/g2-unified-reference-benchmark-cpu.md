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
