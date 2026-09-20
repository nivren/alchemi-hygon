# G2 Batch query topology materialization HIP benchmark

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 22。

## 目标与范围

小点 21 的 native HIP topology materialization candidate 已通过语义边界，但其实现包含五次
rocPRIM radix sort、临时字段/workspace 分配和 host-side pair-count synchronization。本小点在
固定 native unordered candidate 上，对比：

- `materialize_topology`：point20 的 Torch stable canonicalization/scatter stage；
- `materialize_topology_hip`：point21 native HIP sort/scan/scatter candidate。

两者都只覆盖 public matrix/count/shift topology，不包含 distance/vector、neighbor enumeration、
cell build 或完整上游 query。

## 测量合同

HCU 0/BW200/gfx936，DTK 26.04，PyTorch HIP `6.3.26093`，FP32，cutoff `0.6`，capacity `256`，
每个 workload 3 warm-up、5 samples，覆盖 `46/92 atoms × 32/64 systems` 和 uniform/clustered
positions。固定 grid/CSR metadata；JIT、Python wall time 和 allocation setup 不计入 event scope。

native HIP stage 前显式执行 `torch.cuda.synchronize()`，用于隔离 Torch 前序异步 work 的 stream
边界；该同步位于 event 计时之外。candidate 输出在每组计时前后都与 Torch materialization
逐元素比较，所有 8 个 workload parity 通过。native extension 本身仍包含 C++ 临时分配和
`candidate_counts.sum().item()` 的 host synchronization，后者不是 API wall-clock 的完整测量。

raw samples：

- `artifacts/native-batch-query-topology-hip-46x32/`
- `artifacts/native-batch-query-topology-hip-46x64/`
- `artifacts/native-batch-query-topology-hip-92x32/`
- `artifacts/native-batch-query-topology-hip-92x64/`

## 结果

单位为 warm-up 后 HIP-event device time median；factor 为 HIP/Torch，低于 `1.0x` 表示 HIP
较快。

| atoms × systems | workload | Torch topology (ms) | HIP topology (ms) | HIP/Torch |
|---|---|---:|---:|---:|
| 46 × 32 | uniform | 1.22112 | 1.05040 | 0.860x |
| 46 × 32 | clustered | 1.27616 | 1.05920 | 0.830x |
| 46 × 64 | uniform | 1.23840 | 1.57712 | 1.274x |
| 46 × 64 | clustered | 1.34496 | 1.58080 | 1.175x |
| 92 × 32 | uniform | 1.18512 | 1.61792 | 1.365x |
| 92 × 32 | clustered | 1.23520 | 1.53120 | 1.240x |
| 92 × 64 | uniform | 1.24752 | 2.11552 | 1.696x |
| 92 × 64 | clustered | 1.29472 | 2.11872 | 1.636x |

## 结论

当前 native HIP candidate 不是稳定的 topology optimization：仅在 `46×32` 的两个 workload
上出现 `14%--17%` 的 device-time 降低；其余六个 workload 慢 `17%--70%`。Batch 增大后，五次
radix sort、字段 gather、临时 workspace 和 scatter 的成本增长明显，不能把低 Batch 的局部收益
外推到高 Batch 或完整 neighbor pipeline。

因此本 candidate 继续保留为 HIP correctness/profile 实验，不登记 `hip` capability，不接入
dispatcher 或 `auto`。下一步不应直接扩大该实现的生产范围；应先分析排序/临时分配热点，评估
单次复合 key、专用 compaction/scatter、workspace reuse 或 Triton 规则路径，再以同一矩阵重新
验证。未覆盖 FP64 performance、API wall-clock、rebuild、target/pair outputs、pair-centric query、
compile/opcheck、分布式和端到端 MACE/FIRE2。
