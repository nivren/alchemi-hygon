# G2 Batch query topology composite-key workspace reuse

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 27。

## 目标与实现

本小点把 composite-key topology candidate 的临时生命周期拆开：

- direct path：每次 native HIP 调用内部分配 key/order/row-start/sort/scan buffers；
- reused path：调用方预先创建 `CompositeTopologyWorkspace`，重复调用时复用 caller-owned
  `int64 key`、`int32 order/row_starts` 和 `uint8 sort/scan workspace`。

新增 workspace-size query 和 native binding。workspace 只约束 scratch 的 shape、dtype、device
和 key layout；public outputs 仍由调用方提供。`candidate_counts.sum().item()` 的 host-side
pair-count synchronization 没有在本点移除，作为独立后续实验。

## 正确性

CPU focused boundary 为 `6 passed`，包括 CPU workspace 创建显式失败。HCU 0/BW200/gfx936
完整 query/materialization boundary 通过：FP32/FP64、mixed-PBC、full/half、non-default stream、
empty、native query capacity overflow、composite public-capacity rejection，以及同一
workspace 连续复用两次的 public matrix/count/shift parity 均通过。

## 性能合同

设备为 HCU 0/BW200、`gfx936`、DTK 26.04、PyTorch HIP `6.3.26093`。FP32、capacity `256`、
固定 grid/CSR metadata，覆盖 `46/92 atoms × 32/64 systems` 的 uniform/clustered workload，
每个 workload 3 次 warm-up、5 次 samples。

- HIP-event device time：只包围 native topology call，JIT、初始化和 Python wall time 不计入；
- API wall-clock：以 `torch.cuda.synchronize()` 包围完整 native call，包含 host API、内部
  allocation 和现有 host synchronization；
- 两条路径使用相同 unordered candidate 和 public parity gate；8 个 workload 均通过。

## 结果

下表为 reused/direct 的 median factor，低于 `1.0x` 表示 reused 更快。

| atoms × systems | workload | device-time factor | API wall-clock factor |
|---|---|---:|---:|
| 46 × 32 | uniform | 0.904x | 0.945x |
| 46 × 32 | clustered | 0.910x | 1.129x |
| 46 × 64 | uniform | 0.958x | 1.049x |
| 46 × 64 | clustered | 0.956x | 0.902x |
| 92 × 32 | uniform | 0.900x | 0.975x |
| 92 × 32 | clustered | 1.026x | 1.047x |
| 92 × 64 | uniform | 0.925x | 0.962x |
| 92 × 64 | clustered | 0.935x | 0.983x |

API wall-clock 的 factor 覆盖 `0.902x--1.129x`，没有在全部 workload 上稳定低于 `1.0x`；
device-time 也出现 `92×32 clustered` 的轻微回退。当前结果只能说明 caller-owned workspace
路径可工作，并在部分 workload 降低生命周期成本，不能宣称稳定性能收益。

## 结论与下一步

workspace reuse 作为可复用基础模块已完成 correctness 和受限性能验证，但不进入 runtime，
也不替换 direct path。它没有解决当前 candidate 的 host-side pair-count synchronization。

下一小点独立评估去除 `candidate_counts.sum().item()`：改为 device-side pair count 或已知
capacity-safe scatter contract，并重新同时测 device timeline 与 API wall-clock；不得把 host
sync 变化和 composite-key 算法再次混合。

原始 samples 保存在：

- `artifacts/g2-point27-46x32/`；
- `artifacts/g2-point27-46x64/`；
- `artifacts/g2-point27-92x32/`；
- `artifacts/g2-point27-92x64/`。
