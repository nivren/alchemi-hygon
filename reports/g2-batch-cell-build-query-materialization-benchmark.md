# G2 Batch cell-list build-inclusive neighbor benchmark

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 31。

## 目标与 measured scope

本小点把前面固定 CSR metadata 的 query/materialization benchmark 扩展为一次调用内的
完整隔离流程：

```text
cell-list build
    -> query candidate
    -> public topology materialization
    -> distance/vector geometry materialization
```

比较三条路径：

1. `full_torch`：Torch cell-list build + Torch query/topology/geometry reference；
2. `full_native_torch_geometry`：native HIP build/query/composite topology + Torch geometry，
   这是保留训练和梯度路径的 hybrid；
3. `full_native_native_geometry`：native HIP build/query/composite topology + native HIP
   forward-only geometry candidate。

本 probe 是隔离性能候选，不改变 dispatcher、`auto`、`hip` capability 或默认 runtime 路径。
Torch geometry 仍是训练/reference 路径；native HIP geometry 不提供 autograd 或二阶梯度。

计时区间预热后开始，固定 grid metadata、输入、输出 buffer、cell-list scan workspace、topology
workspace 和 JIT 均不计入。每次 `full_*` 调用都会重新执行 build，并覆盖 query、topology 和
geometry 输出；因此不是复用上一轮 CSR 的窄 benchmark。另记录 build-only device time，帮助
区分 build 的单独代价。HIP-event 与 API wall-clock 分开记录：后者在调用前后显式执行
`torch.cuda.synchronize()`。

## 输入与验证

- 设备：HCU 0，BW200 / `gfx936`，DTK 26.04，PyTorch HIP `6.3.26093`；
- dtype：FP32，capacity `256`，cutoff `0.6`；
- 输入：`46/92 atoms × 32/64 systems`，uniform/clustered，共 8 个 workload；
- batch：混合 PBC 模板、非正交 cell、固定 cell grid dimensions，seed `20260919`；
- 计时：每个 workload warm-up `3` 次、samples `5` 次，逐样本保存原始 ms；
- 正确性：native build 的 key/count/start 与 Torch reference 对照，atomic atom-list 按
  cell membership 对照；完整 matrix/count/shift/distance/vector 与 Torch reference 做
  parity。native FP32 geometry 的舍入按 `1e-6` 检查，拓扑字段严格检查。

smoke（46×32，warm-up 1，samples 2）和正式四个规模矩阵均通过上述末尾复核；没有把
benchmark 中的 parity 误写成 autograd、compile 或 runtime integration 证据。

## 结果

factor 定义为 `candidate / full_torch`，低于 `1.0x` 表示 candidate 时间更低。下表的
full 时间是 HIP-event median，括号内为同一 workload 的 API wall-clock median；API factor
同样以 candidate / Torch API wall-clock 计算。

| atoms × systems | workload | Torch full ms | native + Torch geometry ms | hybrid device/API factor | native + native geometry ms | forward device/API factor | build native/Torch |
|---|---|---:|---:|---:|---:|---:|---:|
| 46 × 32 | uniform | 89.968 / 89.865 | 4.592 / 4.500 | 0.0510x / 0.0501x | 3.453 / 3.425 | 0.0384x / 0.0381x | 1.324x |
| 46 × 32 | clustered | 110.712 / 110.652 | 4.624 / 4.641 | 0.0418x / 0.0419x | 3.521 / 3.523 | 0.0318x / 0.0318x | 1.318x |
| 46 × 64 | uniform | 175.764 / 175.785 | 5.291 / 5.307 | 0.0301x / 0.0302x | 4.218 / 4.167 | 0.0240x / 0.0237x | 1.372x |
| 46 × 64 | clustered | 220.722 / 220.635 | 5.435 / 5.468 | 0.0246x / 0.0248x | 4.261 / 4.294 | 0.0193x / 0.0195x | 1.385x |
| 92 × 32 | uniform | 103.823 / 103.935 | 4.776 / 4.790 | 0.0460x / 0.0461x | 3.656 / 3.693 | 0.0352x / 0.0355x | 1.324x |
| 92 × 32 | clustered | 112.581 / 112.342 | 4.917 / 4.968 | 0.0437x / 0.0442x | 3.804 / 3.816 | 0.0338x / 0.0340x | 1.311x |
| 92 × 64 | uniform | 207.631 / 207.413 | 5.527 / 5.511 | 0.0266x / 0.0266x | 4.321 / 4.349 | 0.0208x / 0.0210x | 1.385x |
| 92 × 64 | clustered | 225.852 / 225.971 | 5.646 / 5.645 | 0.0250x / 0.0250x | 4.422 / 4.454 | 0.0196x / 0.0197x | 1.399x |

跨 8 个 workload：

- native build-only / Torch build factor：`1.311x--1.399x`，说明当前 native build 单独仍慢
  约 `31%--40%`；
- native build/query/topology + Torch geometry / Torch full factor：device
  `0.0246x--0.0510x`，API `0.0248x--0.0501x`；
- native build/query/topology + native geometry / Torch full factor：device
  `0.0193x--0.0384x`，API `0.0195x--0.0381x`。

因此 build 的劣化没有抵消 query/topology/geometry 的收益，但这个结论只适用于本 probe
定义的固定容量、中小体系高 Batch、FP32、预热后隔离流程；不能外推到大体系、FP64、动态
rebuild、分配/JIT、模型/MD 端到端或默认 neighbor API。

## 结论与边界

Point 31 已完成 build-inclusive 的 isolated end-to-end neighbor benchmark，并确认：

- 保留 Torch geometry 的 native discrete hybrid 在训练语义上仍可沿用已有 Torch geometry
  梯度路径；
- native HIP geometry 在 forward-only scope 进一步降低 geometry 阶段成本，但不能进入训练
  路径；
- 当前 cell-list build 是明确的优化热点，不能因为完整链路快就忽略其单独约 `31%--40%`
  的劣化；后续应针对 build 做融合、workspace/launch 和稀疏/密集负载分析；
- 三条路径仍是 isolated candidates，未接 ops dispatcher、framework runtime、`auto`、
  `hip` capability、AOT/compile、rebuild/skin 或默认生产路径；未覆盖 `target_indices`、
  `pair_fn`/pair outputs 和 DomainParallel。

原始逐样本结果：

- `artifacts/native-batch-cell-build-query-materialization-benchmark/46x32/`；
- `artifacts/native-batch-cell-build-query-materialization-benchmark/46x64/`；
- `artifacts/native-batch-cell-build-query-materialization-benchmark/92x32/`；
- `artifacts/native-batch-cell-build-query-materialization-benchmark/92x64/`。

下一步建议：先不接默认 runtime，针对 build pipeline 做独立的阶段 breakdown 和低/中/高
occupancy 负载分析，确认是 key/count、scan、atomic fill 还是固定 global-cell grid 的代价，
再决定是否设计 build/query/geometry 融合候选。Torch geometry 的训练/梯度路径继续保留。
