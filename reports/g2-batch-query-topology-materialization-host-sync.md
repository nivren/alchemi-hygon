# G2 Batch query topology host-sync removal

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 28。

## 目标与实现

本小点处理 topology materialization 在 scatter 前的 host synchronization：

- 原实现用 `candidate_counts.sum().item()` 把有效 pair 总数同步到 host，再决定 scatter 的
  launch size；
- 新实现按 `atoms × candidate_capacity` 发射 scatter，kernel 从 sorted `order` 恢复原始
  row/slot，并在 device 侧以 `candidate_counts[row]` 跳过 padding entry；
- 五次 int32 stable-sort path 和 single int64 composite-key path 均采用同一 contract；
- public matrix/count/shift 的排序、填充、容量和 stream 语义不变，没有改变 dtype、顺序或
  public API。

该方案以固定 candidate capacity 换取无 host pair-count 查询，可能增加低 occupancy/低实际
pair-count场景的 scatter 线程数，因此只作为隔离 candidate，不自动接入 runtime。

## 正确性

CPU focused boundary 为 `6 passed`。HCU 0/BW200/gfx936 boundary 通过 FP32/FP64、mixed-PBC、
full/half、non-default stream、empty、capacity overflow、composite public-capacity rejection、
workspace reuse；新增全零 `candidate_counts` case 也验证 public matrix 填充为 atom sentinel、
shift/count 清零。

## 性能合同

设备为 HCU 0/BW200、`gfx936`、DTK 26.04、PyTorch HIP `6.3.26093`。FP32、capacity `256`、
固定 grid/CSR metadata，覆盖 `46/92 atoms × 32/64 systems` 的 uniform/clustered workload，
每个 workload 3 次 warm-up、5 次 samples。

- HIP-event device time：只包围 pre-warmed native topology call；
- API wall-clock：用 `torch.cuda.synchronize()` 包围 native call；
- 所有 workload 在计时前后检查 public matrix/count/shift parity；
- point27 的 direct median 作为历史对照，但两次运行不是交替配对采样，因此前后 factor 只作
  描述性证据，不作为独立 speedup 结论。

## 结果

### host-sync removal：point28 direct / point27 direct

低于 `1.0x` 表示 point28 direct 较低。point27 原始样本位于
`artifacts/g2-point27-*`，point28 原始样本位于 `artifacts/g2-point28-*`。

| atoms × systems | workload | device-time factor | API wall-clock factor |
|---|---|---:|---:|
| 46 × 32 | uniform | 0.914x | 1.042x |
| 46 × 32 | clustered | 0.817x | 1.036x |
| 46 × 64 | uniform | 0.860x | 0.858x |
| 46 × 64 | clustered | 0.883x | 0.861x |
| 92 × 32 | uniform | 0.828x | 0.935x |
| 92 × 32 | clustered | 0.884x | 0.929x |
| 92 × 64 | uniform | 0.885x | 0.942x |
| 92 × 64 | clustered | 0.884x | 0.943x |

device median 在这组前后样本中均低于 point27；API wall-clock 在 8 个 workload 中 6 个低于
point27，但 `46×32` 两个 workload 略高，说明 allocator、同步边界和运行噪声仍然存在。

### point28 removal 后 workspace reused / direct

| atoms × systems | workload | device-time factor | API wall-clock factor |
|---|---|---:|---:|
| 46 × 32 | uniform | 0.828x | 0.776x |
| 46 × 32 | clustered | 0.948x | 0.800x |
| 46 × 64 | uniform | 0.943x | 1.052x |
| 46 × 64 | clustered | 0.952x | 1.140x |
| 92 × 32 | uniform | 0.882x | 0.970x |
| 92 × 32 | clustered | 0.951x | 1.123x |
| 92 × 64 | uniform | 0.943x | 0.976x |
| 92 × 64 | clustered | 0.953x | 0.970x |

这张表只是 removal 后 workspace 生命周期对照，不是 Torch 对照。workspace 复用仍没有在
全部 workload 上形成稳定 API wall-clock 收益。

## 结论与下一步

`candidate_counts.sum().item()` 已从两条 native topology scatter 路径移除，且 correctness
边界通过；这证明了 device-side capacity-safe scatter contract 可行。由于固定容量 launch
可能在稀疏 candidate 上增加无效线程，当前只保留为 candidate，不接 dispatcher、AOT、`auto`
或 `hip` capability。

下一步可进入一次完整 hybrid pipeline 对照：HIP build/query/topology 加 Torch geometry，与
完整 Torch reference 比较 matrix/count/shift/distance/vector 和梯度；随后再决定 geometry
是否适合 Triton/HIP。

原始 samples 保存在：

- `artifacts/g2-point28-46x32/`；
- `artifacts/g2-point28-46x64/`；
- `artifacts/g2-point28-92x32/`；
- `artifacts/g2-point28-92x64/`。
