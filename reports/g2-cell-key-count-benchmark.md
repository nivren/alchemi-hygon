# G2 cell-key count microbenchmark

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 6。

## 测量契约

本次比较同一 HCU 上两个显式 count API：Torch
`build_cell_counts_reference_into` 与 HIP `build_cell_counts_hip_into`。计时范围是单个 API 的
HIP-event device 时间：包含 ABI 输入检查、完整 output 覆写/清零和 count；不包含 JIT/模块加载、
key 生成、输出比较、host wall-clock、scan、sort/fill、query 或完整 cell-list build。

两种实现都接受相同 `(32768,) int32` key tensor、写同一大小的 `(4096,) int32` count buffer。固定
generator seed 为 `20260919`，覆盖两个 workload：

- `uniform`：keys 在 4096 cells 上均匀随机；
- `single_cell`：全部 key 为 0，刻意构造原子加竞争。

每个 workload 在计时前以相同输入分别运行 Torch/HIP 并逐元素比较；20 次交替 warm-up 后，以每
sample 100 次调用、20 个 sample 记录 HIP events。计时后再次逐元素比较最终 output。原始样本不
入库，保留在 `artifacts/cell-key-count-benchmark/{torch,hip}-{uniform,single_cell}-ms.txt`，每文件
20 个正数毫秒值。

## 执行环境与结果

运行前主机 `hy-smi` 显示 HCU 0 利用率和显存占用均为 0%；同一权限上下文的 `/dev/kfd`、`/dev/dri`
可见。环境为 DTK 26.04、PyTorch HIP `6.3.26093`、`HIP_VISIBLE_DEVICES=0`、
`OMP_NUM_THREADS=1`、BW200/gfx936。JIT extension 已在计时前通过正确性调用加载；JIT 不在 HIP
event 区间内。

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python probes/cell_key_count_benchmark.py \
  --output-dir artifacts/cell-key-count-benchmark
```

| workload | Torch median | HIP median | Torch / HIP | HIP 降低 | Torch/HIP 相对总体标准差 |
|---|---:|---:|---:|---:|---:|
| uniform | 0.26104 ms | 0.20087 ms | 1.300x | 23.05% | 0.17% / 0.31% |
| single_cell | 0.26266 ms | 0.20278 ms | 1.295x | 22.80% | 2.49% / 0.19% |

`hygon-dcu-performance/scripts/compare_timings.py` 已从上述原始样本计算中位数、均值、范围和总体
标准差。single-cell Torch 样本有一个 0.29243 ms 高值，故其相对标准差较高；即使如此，HIP
中位数仍较低。所有 count 输出在计时前后均逐元素相等。

## 解释与边界

HIP candidate 在这两个固定的 count-only workload 上有约 1.30x 的局部 device-time 优势，且高
atomic contention 没有在此规模下消除该优势。它支持保留 count candidate，但不能推断为完整
cell-list、邻居构建或 MACE/FIRE2 的同等收益：scan、sort/fill、query、Python/dispatcher 及容量
管理仍不在测量范围。

现有 stage-one hipprof trace 已显示完整 Torch cell-list 的成本分散在 indexing、sort、scan/reduce
与大量 launch；本小点没有调整 kernel 参数，也没有运行 profiler，因此没有提出新的 kernel 调优
结论。该 23% 局部结果不满足 neighbor 2x 或端到端 20% 的 `auto` 准入门槛，不改变默认路径、
`auto` 或 `hip` capability。

## 下一步

下一小点建议独立实现/验证 native HIP exclusive-scan：输入 active `int32` counts、输出同形的
exclusive `int32` starts 加 global atom offset，counts 不得被破坏。先建立与现有 Torch starts oracle
一致的空输入、offset、overflow、stream 与数值测试；不接入完整 build 或 dispatcher。之后再以
相同 benchmark discipline 判断 count+scan 的组合价值。
