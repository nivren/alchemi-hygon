# G2 native HIP Batch cell-list trusted build fast path

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 33。

## 目标与边界

Point 32 的分阶段 profiling 表明，当前 native Batch build 的 public wrapper 和
mutation-only custom-op 会重复执行 ABI validation，其中包含 `torch.any`、
`sum().item()`、`torch.equal` 等 device work。本小点只验证一个窄假设：对已经由上层
验证过的、结构稳定的精确 buffer 集合，绕过第二次 Python/custom-op validation，直接
串接已有 HIP key/count、rocPRIM scan 和 atomic fill extension，是否能稳定降低 build
时间。

新增私有 `_build_batch_cell_csr_hip_into_trusted`。公开的
`build_batch_cell_csr_hip_into` 保持原有 validation/error contract；trusted 函数不加入
`__all__`，不接 dispatcher、AOT、`auto`、`hip` capability，也不允许作为未知输入的
无条件替代。它没有改变 key、count、scan、fill 的 kernel、CSR layout、PBC、dtype 或
cell membership 语义。

## 验证合同

- 设备：HCU 0，BW200，`gfx936`，DTK 26.04，PyTorch HIP `6.3.26093`。
- dtype：FP32；mixed PBC、非正交 cell；固定 metadata、output buffer 和 workspace。
- workload：`46/92 atoms × 32/64 systems`，uniform/clustered，共 8 个 workload。
- 每组 3 warm-up、5 samples；HIP event 和显式同步 API wall-clock 分开测量；JIT/module
  load 与分配不计入稳态计时。
- trusted 输出 buffer 在计时前各通过一次 public validation；每次正式调用仍要求
  public、trusted 和 Torch CSR 输出 parity，包括 shifts/mapping/keys/counts/starts、
  每 cell atom membership、cursor final state 和 capacity tail。

## 结果

下表为 8 个 workload 中 uniform/clustered 两种分布的范围。单位为 HIP-event median
ms；factor 小于 `1.0x` 表示 trusted 更快。

| atoms × systems | public build | trusted build | trusted/public | Torch build | trusted/Torch |
|---|---:|---:|---:|---:|---:|
| 46 × 32 | 2.255--2.269 | 0.0706--0.0722 | 0.0313--0.0318 | 1.705--1.722 | 0.0410--0.0423 |
| 46 × 64 | 2.781--2.794 | 0.0794--0.0803 | 0.0285--0.0287 | 1.979--1.980 | 0.0401--0.0406 |
| 92 × 32 | 2.267--2.267 | 0.0707--0.0707 | 0.0312--0.0312 | 1.719--1.719 | 0.0411--0.0411 |
| 92 × 64 | 2.781--2.791 | 0.0797--0.0803 | 0.0286--0.0288 | 1.963--2.002 | 0.0398--0.0409 |

跨全部 workload，trusted/public device factor 为 `0.0285x--0.0318x`，即相对 public
composition 降低约 `96.8%--97.2%`；trusted/Torch factor 为 `0.0398x--0.0423x`。
API wall-clock 也保持同方向，trusted/public factor 为约 `0.0338x--0.0348x`，但这些
数字只描述 isolated cell-list build，不是完整 neighbor、MD 或 MACE 加速。

## 正确性与回归

- 四个正式规模、两种分布全部通过 trusted/public/Torch CSR parity。
- 46×32 HCU smoke（1 warm-up/2 samples）通过，验证了 direct extension 调用签名和
  非默认结构的实际 device 执行。
- CPU focused ABI/boundary suite：`26 passed`。
- `py_compile` 和 `git diff --check` 通过。

## 结论与限制

在当前固定 Batch、FP32、forward build 合同下，重复 validation 是足够显著且稳定的
开销来源，trusted direct-extension composition 候选应保留。它仍然是私有、显式选择的
性能候选，不改变公开 API，也不意味着可以把所有调用都标记为已验证结构。

尚未验证或不在本小点范围内：默认 runtime/dispatcher 接线、`auto`/`hip` capability、
FP64 性能、JIT 与分配成本、rebuild/skin、autograd/compile、`target_indices`、
`pair_fn`/pair outputs 和 DomainParallel。下一步应评估如何在不复制完整 validation 的
前提下建立可复用的 prevalidated plan/生命周期，并重新测包含 query/materialization/
geometry 的完整端到端流程；若该 plan 无法安全表达，应保留 public path 并转向 clear/
scan/fill 的算法优化。

原始 samples：

- `artifacts/native-batch-cell-build-trusted-fastpath-benchmark/46x32/`；
- `artifacts/native-batch-cell-build-trusted-fastpath-benchmark/46x64/`；
- `artifacts/native-batch-cell-build-trusted-fastpath-benchmark/92x32/`；
- `artifacts/native-batch-cell-build-trusted-fastpath-benchmark/92x64/`。
