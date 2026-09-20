# G2 Batch-scale HIP build plus Torch-query performance matrix

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 16。

## 测量合同

本报告补充小点 15 的固定大体系基线，专门覆盖中小体系的高 Batch 场景：
`46/92 atoms per system × 32/64 systems`。每个 workload 同时测 uniform 和 clustered
两种位置分布。

比较三类预热后的 HIP-event device scope：

- `build`：native HIP fused key/count + rocPRIM scan + atomic fill，对 Torch key/count + scan + stable fill；
- `query`：两套 CSR buffer 均调用同一现有 Torch `batch_query_cell_list`；
- `full`：对应 build 紧接 query。

计时不包含 JIT、extension 初始化、buffer 分配、Python wall-clock 或 runtime selection。每个
case 使用 3 次 warm-up、5 个交替 samples；计时前后检查 build metadata、cell membership 和
public neighbor matrix/counts/pair-shifts。设备为 HCU 0（BW200/gfx936），DTK 26.04，PyTorch
HIP `6.3.26093`，FP32，mixed PBC，capacity 256。

当前 benchmark 使用两套固定 system geometry template，因此 global cell 数为：

| workload | total atoms | global cells |
|---|---:|---:|
| 46 × 32 | 1472 | 745472 |
| 46 × 64 | 2944 | 1490944 |
| 92 × 32 | 2944 | 745472 |
| 92 × 64 | 5888 | 1490944 |

## 结果

下表为 median，单位 ms；格式为 `Torch/native`。full delta 定义为
`(Torch - native) / Torch`，负值表示 native 更慢，不能解释为 native 加速。

| systems | distribution | build Torch/native | query Torch/native | full Torch/native | full delta |
|---|---|---:|---:|---:|---:|
| 46 × 32 | uniform | 1.77760 / 2.32944 | 89.89092 / 89.83684 | 91.90500 / 92.22868 | -0.35% |
| 46 × 32 | clustered | 1.77664 / 2.33872 | 110.63090 / 110.67666 | 112.45666 / 112.97361 | -0.46% |
| 46 × 64 | uniform | 2.18864 / 3.02672 | 173.94728 / 175.02824 | 175.16792 / 176.25368 | -0.62% |
| 46 × 64 | clustered | 2.02736 / 2.92128 | 216.83315 / 216.76819 | 218.77795 / 219.73235 | -0.44% |
| 92 × 32 | uniform | 1.73008 / 2.30496 | 98.87859 / 98.91683 | 100.56290 / 101.19762 | -0.63% |
| 92 × 32 | clustered | 1.73248 / 2.31200 | 107.34241 / 107.39410 | 109.01218 / 109.62321 | -0.56% |
| 92 × 64 | uniform | 2.05968 / 2.85680 | 198.57990 / 198.91316 | 207.03188 / 201.66454 | +2.59%* |
| 92 × 64 | clustered | 2.01856 / 2.92544 | 216.64148 / 216.58339 | 218.70723 / 219.45572 | -0.34% |

主要观察：

- native build 在 8 个 workload 中均慢于 Torch，按 median 约慢 `31%--45%`；不能据此继续
  把 build candidate 宣称为性能收益路径。
- query 在 8 个 workload 中约为 Torch full 的 `96%--99%`，native/Torch query median 差异
  约在 `-0.62%--+0.06%` 之间，属于接近持平的窄证据。
- 除 `92×64 uniform` 外，full native median 慢约 `0.34%--0.63%`。`92×64 uniform` 的 Torch
  full 样本相对总体标准差为 `1.70%`，而 native 为 `0.08%`，该 `+2.59%` 不能作为稳定 native
  加速结论，需增加样本和复测确认。
- 46/92 原子规模与 32/64 Batch 均未触发 capacity 256 overflow；这只证明本 benchmark fixture
  的容量足够，不代表任意密度或 cutoff 下的容量策略已完成。

此前一次 `2×16384` 运行使用 capacity 512 时出现 `row 16385 count 540` 的显式
`NeighborOverflowError`，随后以 capacity 1024 成功完成。该事件是容量合同按 workload sizing
生效的证据，不是数值失败，也说明后续性能矩阵必须单独记录 effective capacity。

## 结论与边界

这组测试支持用户提出的“中小体系、多 Batch”作为正式性能矩阵维度：在小体系下，global cell
metadata/query 的成本远大于当前 build candidate，Batch 增大后该趋势更加明显。下一优化重点应
转向现有 query 的 candidate/materialization，而不是继续只优化 native build 的 atomic fill。

本小点没有修改 kernel、query、dispatcher、AOT、`auto` 或 `hip` capability，也没有将任何候选
接入默认路径。输入仍只覆盖 FP32、单 HCU、两套固定 grid、当前 atom-centric direct query、
固定 capacity 和 forward public outputs；未覆盖 FP64 性能、rebuild、梯度、pair-centric、
`target_indices`、pair outputs、compile、分布式或端到端 MACE/FIRE2。

下一小点应冻结 query/materialization 的独立 reference contract，明确 HIP 不规则枚举与 Triton
规则分块的候选边界，然后只实现一个最小 query candidate correctness probe；在 correctness 和
代表性 benchmark 通过前不接入 runtime。

原始样本目录：

- `artifacts/native-batch-cell-build-query-benchmark-46x32/`
- `artifacts/native-batch-cell-build-query-benchmark-46x64/`
- `artifacts/native-batch-cell-build-query-benchmark-92x32/`
- `artifacts/native-batch-cell-build-query-benchmark-92x64/`
- `artifacts/native-batch-cell-build-query-benchmark-2x16384-cap1024/`
