# G2 native HIP query enumeration plus Torch materialization benchmark

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 18。

## 测量合同

本小点在小点 16 的高 Batch 矩阵上拆分 query device work：

- `enumeration_native`：native HIP Batch query enumeration，输出 unordered row candidates；
- `query_native_torch_materialize`：native enumeration 加 Torch stable sort/scatter，生成现有
  public `neighbor_matrix`、`neighbor_matrix_shifts` 和 `num_neighbors`；
- `query_torch`：现有 Torch reference Batch query，使用相同 CSR input、cutoff 和 public surface。

三类均使用 HIP events 测预热后的 device timeline；JIT、extension 初始化、分配、Python wall time
和 runtime selection 不在 event scope 内。candidate 的 host-side overflow check 不等于 event
elapsed，因此结果不是 Python API wall-clock。每组 3 次 warm-up、5 个交替 samples。

设备为 HCU 0（BW200/gfx936），DTK 26.04，PyTorch HIP `6.3.26093`，FP32，cutoff `0.6`，
capacity 256，固定两套 cell geometry template（`32^3` 与 `24^3` cells），Batch 使用
`46/92 atoms × 32/64 systems`，同时覆盖 uniform/clustered positions 和 mixed PBC。

正确性 oracle 是小点 17 的 Torch stable materialization：计时前后 candidate canonicalization
的 matrix/counts/shifts 与 Torch reference 一致。当前 benchmark 不请求 optional distance/vector
outputs；native kernel 只计算 cutoff 所需几何，不输出可微距离/向量，也不覆盖 autograd。

## 结果

median，单位 ms；`candidate factor` 定义为 `Torch query / native enumeration + Torch
materialization`，大于 1 表示 candidate device work 更快。

| systems | distribution | native enumeration | native + Torch materialize | Torch query | candidate factor | candidate full relative pstdev |
|---|---|---:|---:|---:|---:|---:|
| 46 × 32 | uniform | 0.91536 | 1.97792 | 89.87483 | 45.44× | 1.18% |
| 46 × 32 | clustered | 1.00480 | 2.13712 | 110.57818 | 51.74× | 1.32% |
| 46 × 64 | uniform | 0.89728 | 1.99056 | 173.96167 | 87.39× | 1.67% |
| 46 × 64 | clustered | 0.99488 | 2.15520 | 218.01826 | 101.16× | 3.24% |
| 92 × 32 | uniform | 0.91136 | 2.02864 | 102.80618 | 50.68× | 2.24% |
| 92 × 32 | clustered | 1.04432 | 2.16880 | 111.19785 | 51.27× | 0.82% |
| 92 × 64 | uniform | 0.94864 | 2.11920 | 206.04083 | 97.23× | 1.84% |
| 92 × 64 | clustered | 1.07792 | 2.22432 | 223.97537 | 100.69× | 3.11% |

主要观察：

- 8 个 workload 中，native enumeration + Torch matrix/count/shift materialization 均低于
  Torch query，median factor 为 `45.44×--101.16×`，描述性降低 `97.80%--99.01%`；candidate
  full 的相对总体标准差最高 `3.24%`，仍显著低于差距本身。
- Torch materialization 相对 enumeration 增加约 `1.07--1.22 ms`，总成本约为 enumeration 的
  `2.06--2.23×`。这说明当前最小 HIP enumeration 已足以覆盖 Torch query 的主要 device-work
  瓶颈，但 stable sort/scatter 仍是 candidate pipeline 的主要剩余部分。
- 该结果只证明固定 FP32、高 Batch、固定 grid、matrix/count/shift surface 的 device-time
  优势；不能推广为完整 neighbor、MACE/FIRE2 或端到端加速，也不能与带 distance/vector 输出
  或 autograd 的 workload 等同。

## 结论与边界

当前 native query candidate 值得保留为后续优化方向；没有必要在没有 profile/测量依据时继续只
微调 cell build。下一步优先补齐 candidate 的 optional distance/vector materialization 和连续
路径梯度合同，再重新测相同矩阵；若完整 public surface 仍保持收益，才讨论 dispatcher 接线和
HIP/Triton canonicalization 的实现选择。

本小点没有修改 runtime、dispatcher、AOT、`auto` 或 `hip` capability。candidate 仍为隔离的
forward-only slice，未覆盖 FP64 性能、rebuild、target indices、pair callbacks、pair-centric
query、compile、分布式、端到端模型和生产 API wall-clock。

原始样本目录：

- `artifacts/native-batch-cell-query-materialization-benchmark-46x32/`
- `artifacts/native-batch-cell-query-materialization-benchmark-46x64/`
- `artifacts/native-batch-cell-query-materialization-benchmark-92x32/`
- `artifacts/native-batch-cell-query-materialization-benchmark-92x64/`

比较脚本：`hygon-dcu-performance/scripts/compare_timings.py`。
