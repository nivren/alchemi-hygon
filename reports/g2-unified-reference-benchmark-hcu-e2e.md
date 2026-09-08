# G2 unified reference benchmark：HCU periodic MACE/FIRE2 端到端

日期：2026-09-08。此次里程碑运行真实 `MACE-OFF23_small.model`、periodic
`NeighborListHook(skin=0.5)` 和固定晶胞 `FIRE2(backend="torch_reference")`，输入为
`[46,92]` 两个异构周期体系，严格执行 100 步。`ConvergenceHook` 使用负阈值只为
保证不提前停止，因此这不是收敛率或最终结构质量测试，而是端到端成本基线。

## 命令与环境

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 600 \
  .venv/bin/python -u probes/unified_reference_benchmark.py \
  --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --skip-scales --skip-batches --e2e-steps 100 --e2e-skin 0.5
```

退出码 `0`。设备为 `BW200, UBB BW1000`，source DTK 26.04，
`HIP_VISIBLE_DEVICES=0`。模型加载耗时 `1.5378 s`，不计入下方端到端 100 步区间。

## 端到端结果

| 项目 | 结果 |
|---|---:|
| 输入原子数 / batch pointer | `[46,92]` / `[0,46,138]` |
| effective/source PBC | 两个体系均 `[true,true,true]` |
| 步数 / dt / skin | `100` / `0.01` / `0.5` |
| 总耗时 | `11.1279105 s` |
| 100 步平均耗时 | `0.1112791 s/step` |
| 最后一步邻居边数 | `3,284` |
| 状态 | `passed` |

## StageTimingHook 摘要

以下是 100 个样本的 host wall-clock 阶段统计；它用于成本分解，不等同于单个 HCU
kernel 的 profiler 时间。

| 阶段 | total (s) | mean (ms) | min–max (s) | std (s) |
|---|---:|---:|---:|---:|
| `BEFORE_COMPUTE→AFTER_COMPUTE` | 10.439570 | 104.396 | 0.019–6.586 | 0.671 |
| `BEFORE_PRE_UPDATE→AFTER_PRE_UPDATE` | 0.433535 | 4.335 | 0.001–0.281 | 0.028 |
| `BEFORE_STEP→BEFORE_PRE_UPDATE` | 0.132860 | 1.329 | 0.001–0.054 | 0.005 |
| `AFTER_POST_UPDATE→AFTER_STEP` | 0.080387 | 0.804 | 0.001–0.001 | 0.000061 |

计算阶段占主要成本，但其最大值和标准差明显高于均值，说明当前共享 HCU/首次编译
或同步抖动仍然存在。因而 `0.1112791 s/step` 是本次 100 步运行的算术平均，不是
低干扰稳态吞吐承诺。cell-list 决策应同时参考邻居单独阶梯和这一端到端成本，后续
需要独占或低干扰窗口复测冷启动、warmup、steady 三种口径。

