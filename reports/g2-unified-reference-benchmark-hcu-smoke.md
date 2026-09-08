# G2 unified reference benchmark：HCU smoke

日期：2026-09-08。此次里程碑只验证统一 harness 在 HCU 上的设备路径、输入元数据、
周期 image shift 和邻居契约；不把小规模 smoke 当作完整性能基线。

## 命令与环境

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 180 \
  .venv/bin/python -u probes/unified_reference_benchmark.py \
  --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --scale-sizes 46 92 --batch-sizes 1 --warmup 1 --steady 2 --skip-e2e
```

退出码 `0`。设备识别为 `BW200, UBB BW1000`；运行使用 source DTK 26.04 和
`HIP_VISIBLE_DEVICES=0`。

## 结果

| case | 边数 | steady mean (s) | cold (s) | shift/契约 |
|---|---:|---:|---:|---|
| periodic 46 | 824 | 0.003256 | 3.969064 | `[824,3]`，通过 |
| no-PBC 46 full | 626 | 0.001850 | 0.015706 | 无 shift，通过 |
| no-PBC 46 half | 313 | 0.001890 | 0.006130 | 无 shift，通过 |
| periodic 92 | 1702 | 0.003686 | 0.004992 | `[1702,3]`，通过 |
| no-PBC 92 full | 1536 | 0.001854 | 0.002181 | 无 shift，通过 |
| no-PBC 92 half | 768 | 0.003602 | 0.002237 | 无 shift，通过 |

所有 case 的 `cross_system_edges=0`，`source_pbc`、`effective_pbc`、CIF 路径和晶胞
体积均已写入 JSON 输出。92 原子 no-PBC half 的第二次 steady 样本出现一次短时抖动
（`1.866 ms`、`5.338 ms`），因此该 case 的均值/标准差不作为稳定性能结论；完整阶梯
将保持相同的 warmup/steady 口径并记录波动。

## 未完成

- `perf_46/92/184/368` 的 HCU 规模阶梯和等长 batch 阶梯；
- periodic `[46,92]` MACE/FIRE2 固定晶胞 100 步端到端；
- 低干扰窗口下的绝对性能判断和 cell-list 决策。

