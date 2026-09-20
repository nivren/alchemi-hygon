# G2 native HIP Batch cell build plus Torch-query benchmark

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 15。

## 测量合同

比较固定 grid metadata、相同 FP32 Batch 输入下的三类预热后 HIP-event device scope：

- `build`：native HIP fused key/count + rocPRIM scan + atomic fill，对 Torch key/count + scan + stable fill；
- `query`：两套 CSR buffer 都调用同一现有 Torch `batch_query_cell_list`；
- `full`：对应的 build 紧接 query。

不包含 JIT、extension 初始化、buffer 分配、Python wall-clock 或 dispatcher/runtime 选择。每侧 5 次
warm-up，8 个交替顺序 samples；计时前后检查 build metadata、cell membership 和 public
neighbor matrix/counts/pair-shifts。设备为主机 HCU 0（BW200/gfx936），DTK 26.04，PyTorch HIP
`6.3.26093`，FP32，2 systems，`[32,32,32] + [24,24,24]` global cells 共 `46592`，每体系
2048 atoms，capacity 256。

## 结果

median，单位 ms；factor 定义为 `Torch / native`，大于 1 才表示 native 更快：

| workload | build Torch/native | build factor | query Torch/native | query factor | full Torch/native | full factor |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 1.60848 / 1.90568 | 0.844x | 7.52934 / 7.54790 | 0.998x | 9.12286 / 9.42998 | 0.967x |
| clustered | 1.61168 / 1.90768 | 0.845x | 9.79998 / 9.79542 | 1.000x | 11.37270 / 11.69894 | 0.972x |

因此在此窄输入上：native build 比 Torch build 慢约 `18.4%`；两种 query 基本持平；完整 build+query
native 慢约 `2.9%--3.4%`。样本相对总体标准差为 `0.26%--1.23%`。这是性能结果，不是
`auto` 准入结果；native HIP build candidate 当前没有收益证据。

query 占 Torch full median 约 `82.5%`（uniform）和 `86.2%`（clustered），所以后续优化重点应
转向 query/materialization，而不是继续微调当前 build kernel。

## Profile 证据与边界

使用同一 device-visible HCU 上的受限 `hipprof --hip-trace --hiptx-kernel`，`1024 atoms/system`、
1 warm-up、1 sample，仅作热点方向探查。profile 文件为
`artifacts/native-batch-cell-build-query-hipprof.json`，原始 timing 为
`artifacts/native-batch-cell-build-query-profile-benchmark/`。

该 trace 包含新进程启动和 extension JIT/module 初始化：HIP API 统计中
`hipModuleLoadData` 占 `67.69%`、`hipModuleGetFunction` 占 `9.74%`、`hipLaunchKernel` 占
`7.88%`，不能把这些比例当作 steady-state kernel 性能。剔除该解释风险后，HIP OPS 列表显示
Torch query 有大量 index/elementwise/reduction/radix-sort/rocPRIM fragment；native
`batch_cell_key_count_kernel` 与 `cell_atom_list_fill_kernel` 只占该小 profile 的很小部分。
这支持“下一步研究 query/materialization”的方向，但不构成 kernel 优化或通用性能结论。

## 边界与下一步

本小点没有修改 native kernel、Torch query、dispatcher 或 `auto`。输入只覆盖 FP32、两个固定
grid、单设备和当前 atom-centric direct query；未覆盖 FP64、其他规模、rebuild、梯度、pair-centric、
pair outputs、分布式或端到端 MACE/FIRE2。

下一小点应先冻结 query/materialization 的独立 reference contract 和候选边界：明确哪些 pair
enumeration、distance/vector、scatter/capacity 工作可迁移到 HIP，哪些规则分块适合 Triton，再针对
一个最小 query candidate 做 correctness probe；在此之前不接入 runtime。
