# G2 Batch query topology composite-key benchmark

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 25。

## 目标与测量合同

本小点在相同 native unordered query candidate 上比较：

- `materialize_topology`：Torch stable canonicalization/scatter；
- `materialize_topology_hip`：现有五次 int32 stable rocPRIM sort candidate；
- `materialize_topology_composite_hip`：一次 signed-int64 composite-key rocPRIM sort candidate。

设备为 HCU 0/BW200、目标架构 `gfx936`、DTK 26.04、PyTorch HIP `6.3.26093`。输入为
FP32、capacity `256`、cutoff `0.6`，覆盖 `46/92 atoms × 32/64 systems` 的
uniform/clustered 两类位置分布。每个 workload 使用 3 次 warm-up、5 次交替顺序 samples。

计时范围是预热后的 HIP-event topology-stage device timeline；固定 grid/CSR metadata，JIT、
Python wall time、调用外部的 setup 不计入。两个 HIP candidate 都保留 C++ 临时分配和
`candidate_counts.sum().item()` host-side synchronization，因此结果不是 custom-op API
wall-clock，也不是完整 neighbor 或端到端 MACE/FIRE2 时间。每次计时前后均将 public
matrix/count/shift 与 Torch candidate 对照，8 个 workload 全部通过 parity。

## 结果

单位为 median milliseconds；`factor` 为 candidate/Torch topology，低于 `1.0x` 表示
candidate device timeline 更短；`composite/five` 为 single-key candidate 相对五次
int32 HIP candidate 的 factor。

| atoms × systems | workload | Torch topology | HIP five-sort | HIP composite int64 | composite/Torch | composite/five |
|---|---|---:|---:|---:|---:|---:|
| 46 × 32 | uniform | 1.20384 | 1.04880 | 0.35616 | 0.296x | 0.340x |
| 46 × 32 | clustered | 1.27072 | 1.05344 | 0.36720 | 0.289x | 0.349x |
| 46 × 64 | uniform | 1.24736 | 1.53056 | 0.62096 | 0.498x | 0.406x |
| 46 × 64 | clustered | 1.34640 | 1.57776 | 0.63776 | 0.474x | 0.404x |
| 92 × 32 | uniform | 1.20640 | 1.52928 | 0.58016 | 0.481x | 0.379x |
| 92 × 32 | clustered | 1.24768 | 1.61584 | 0.62864 | 0.504x | 0.389x |
| 92 × 64 | uniform | 1.24720 | 2.11520 | 0.65664 | 0.526x | 0.310x |
| 92 × 64 | clustered | 1.26976 | 2.09680 | 0.65280 | 0.514x | 0.311x |

Composite-key 的 sample relative population standard deviation 为约 `0.5%--12.3%`；
`46×32` 的分布相对更抖，`92×64` 约 `0.5%--1.4%`。因此表中 factor 是本次固定
设备和 workload 的描述性结果，不外推到所有邻居规模。

## 结论

在本次 `46/92 × 32/64` 矩阵中，single composite-key candidate 的 topology-stage
device timeline 全部短于现有五次 int32 stable-sort candidate，factor 为 `0.310x--0.406x`；
相对 Torch topology 为 `0.289x--0.526x`。这说明减少全量排序次数是有效优化方向，但还
不能直接称为完整邻居算子加速或生产 backend 支持。

需要特别保留的限制：

- int64 key 仍是 packed scalar storage；本结果不能推广为 gfx936 的通用 int64 高吞吐结论；
- 未测 FP64 性能、不同 shift 位宽/更大原子数、API wall-clock、workspace reuse、host-sync
  removal、rebuild、target/pair outputs、pair-centric query、compile/opcheck 或端到端；
- 未修改 dispatcher、AOT、`auto` 或 `hip` capability，当前 candidate 仍是隔离实验。

原始 samples 保存在：

- `artifacts/g2-point25-46x32/`；
- `artifacts/g2-point25-46x64/`；
- `artifacts/g2-point25-92x32/`；
- `artifacts/g2-point25-92x64/`。

下一小点应先把 composite candidate 复用到现有完整 topology correctness boundary，补齐
FP32/FP64、full/half、empty、mixed-PBC、capacity 和 stream 合同；通过后再独立评估
workspace reuse 或 host-sync removal，不立即接入 runtime。
