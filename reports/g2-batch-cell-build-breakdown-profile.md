# G2 native HIP Batch cell-list build breakdown profile

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 32。

## 目标与测量合同

本小点不修改 HIP kernel。目标是定位 Point 31 中 native HIP cell-list build 比 Torch build
慢约 `31%--40%` 的来源，再决定下一次只优化哪一类问题。

新增 `probes/native_batch_cell_build_breakdown_benchmark.py`，在相同预分配输出和 workspace
下分开测量：

```text
key/count: fused geometry + PBC + key + count
scan:      rocPRIM exclusive scan(count -> CSR start)
fill:      cursor clear + atomic CSR atom-list fill
full:      上述三段串联
```

Torch build 使用同一输入和既有 reference ABI。每个 isolated scan 前先完成 key/count、每个
isolated fill 前先完成 key/count+scan，但这些准备工作位于 HIP-event/API wall-clock 起点之前，
不计入该 stage。每次 full build 都重新执行三段。计时前后比较 shifts/mapping/keys/counts/starts，
atomic atom-list 以每 cell membership 比较，并要求 cursor 等于 final counts。

设备为 HCU 0（BW200 / `gfx936`）、DTK 26.04、PyTorch HIP `6.3.26093`。FP32、mixed PBC、
非正交 cell、seed `20260919`，`46/92 atoms × 32/64 systems` 的 uniform/clustered 8 workload，
每组 3 warm-up、5 samples。固定 grid metadata、buffer/workspace 分配、JIT/module load 不在
稳定计时区间；HIP-event device time 与显式 `torch.cuda.synchronize()` 包围的 API wall-clock
分开保存。

## 分阶段结果

下表为 HIP-event median，单位 ms。`native/Torch` 是完整 build factor，超过 `1.0x` 表示 native
build 更慢；stage sum 与 full 的约 `1.5%--2.4%` 差异来自独立采样而非额外 work。

| atoms × systems | workload | key/count | scan | fill | native full | Torch full | native/Torch |
|---|---|---:|---:|---:|---:|---:|---:|
| 46 × 32 | uniform | 0.810 | 0.704 | 0.806 | 2.265 | 1.718 | 1.319x |
| 46 × 32 | clustered | 0.803 | 0.700 | 0.804 | 2.258 | 1.723 | 1.311x |
| 46 × 64 | uniform | 0.844 | 0.992 | 1.051 | 2.838 | 2.016 | 1.407x |
| 46 × 64 | clustered | 0.837 | 0.990 | 1.052 | 2.823 | 2.002 | 1.410x |
| 92 × 32 | uniform | 0.816 | 0.709 | 0.815 | 2.287 | 1.739 | 1.316x |
| 92 × 32 | clustered | 0.816 | 0.707 | 0.810 | 2.283 | 1.724 | 1.324x |
| 92 × 64 | uniform | 0.795 | 0.983 | 1.037 | 2.773 | 1.950 | 1.422x |
| 92 × 64 | clustered | 0.800 | 0.984 | 1.037 | 2.771 | 1.948 | 1.423x |

API wall-clock 与 device time 接近：例如 92×64 uniform 的 key/count、scan、fill、full native
分别为 `0.789/0.979/1.035/2.753 ms`，因此当前稳态问题不是明显的 host API 等待。随着 global
cell 数从 `745472` 增至 `1490944`：

- scan 由约 `0.700--0.709 ms` 增至 `0.982--0.992 ms`；
- fill 由约 `0.804--0.815 ms` 增至 `1.037--1.052 ms`；
- key/count 保持约 `0.795--0.844 ms`，也不是可忽略项。

因此不能只针对 atomic fill 调 block size：在 64 batch case，fill 约占 full build `37%`，scan
约占 `35%`，key/count 仍约占 `29%`。三段均与 global-cell buffer 的清零、scan 或重复校验有关。

## 受限 profiler 证据

在同一 device-visible 主机确认 `/dev/kfd`、`/dev/dri` 可见，并以
`hipprof --hip-trace --stats --output-type 0` 对 92×64、1 warm-up、1 sample 做受限组成核验。
原始数据库为 `artifacts/point32-hipprof/hip-prof-578460.db`。

该 profile 不用于稳态时间结论：HIP API 统计中 module load/get-function/unload 合计占绝大部分
（`hipModuleLoadData` 约 `68.7%`），受 JIT/module instrumentation 污染。它只支持以下结构判断：

- HIP OPS 统计中 `at::native::any_kernel_continuous_big` 约占 `40.5%`，并有大量 Torch
  elementwise/reduce kernel；
- 代码审查显示 `build_batch_cell_csr_hip_into` 每段调用 Python public wrapper，而 wrapper 和
  mutation-only custom-op 都再次运行 ABI validation；这些 validation 包含 `torch.any`、
  `sum().item()`、`torch.equal` 等 device work；
- 在这个混合 trace 内，`batch_cell_key_count_kernel<float>` 22 次共约 `0.126 ms`、
  `cell_atom_list_fill_kernel` 14 次共约 `0.055 ms`。profile 执行次数和 profiler 开销不同于
  steady benchmark，不能用这些绝对值替换上表，却否定了“核心 atomic kernel 单独占满 build”的
  假设。

## 结论与下一步

本小点验证了 Point 33 的可证伪假设：**重复 public ABI validation 是 native composition 的
显著 device-work 来源；若对结构稳定、已验证的 Batch 使用私有 trusted composition fast path，
则可在不改数学、dtype、PBC、CSR membership 或对外错误合同的前提下，降低完整 native build
时间。**

下一小点应只实现和比较该一类优化：

1. 对外 `build_batch_cell_csr_hip_into` 仍保留完整 validation；
2. 新增仅供已验证 composition 使用的私有 direct-extension path，避免同一次调用中 Python
   validator 和 custom-op validator 的重复 GPU checks；
3. 先以同一 8 workload 测 direct/public、再测 native/Torch，并要求完整 CSR parity；
4. 只有在稳定收益超过样本噪声时才保留；否则拒绝该实验，再分别研究 global-cell clear、scan
   或 fill 的算法/launch 优化。

这不等于允许静默绕过输入检查：trusted fast path 必须由同一次上层结构验证显式选择，不能成为
未知输入、默认 dispatcher 或公开 API 的无条件替代。Torch build/reference、Torch geometry
训练路径和当前 runtime 选择均不改变。

原始 samples：

- `artifacts/native-batch-cell-build-breakdown-benchmark/46x32/`；
- `artifacts/native-batch-cell-build-breakdown-benchmark/46x64/`；
- `artifacts/native-batch-cell-build-breakdown-benchmark/92x32/`；
- `artifacts/native-batch-cell-build-breakdown-benchmark/92x64/`。
