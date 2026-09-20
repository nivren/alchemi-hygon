# G2 Batch query topology composite-key correctness boundary

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 26。

## 目标与范围

将 single composite-key HIP topology candidate 接入已有 native Batch query/materialization
boundary probe，复用同一 Torch reference、native query candidate 和 public topology parity
合同。五次 int32 HIP materializer 与 composite int64 HIP materializer 均对同一 unordered
candidate 执行，并比较 `matrix/count/shift`。

本小点只扩展 correctness，不测性能、不接 dispatcher/AOT/`auto`/`hip` capability，也不把
composite topology 输出接入 geometry/autograd。已有 probe 中的 geometry distance/vector
及一/二阶梯度检查仍保留，用于确认本次没有破坏相邻的 Torch continuous path，但不构成
composite candidate 的 geometry gradient 证据。

## 覆盖范围

- FP32 与 FP64 positions/cells；
- mixed-PBC Batch：system 0 为三维 PBC，system 1 为 no-PBC；
- full 与 half query candidate；
- 默认 stream 与 non-default stream；
- empty candidate/public buffers；
- native query capacity overflow；
- composite materializer 的 public capacity overflow rejection；
- composite public topology 与 Torch reference、五次 HIP stable-sort candidate 的 parity。

## HCU 结果

环境为 DTK 26.04、HCU 0/BW200、目标架构 `gfx936`、PyTorch HIP `6.3.26093`。实际 probe
输出为：

```json
{
  "atoms_per_system": [16, 11],
  "batch_systems": 2,
  "capacity_overflow_explicit": true,
  "composite_capacity_rejected": true,
  "dtype_parity": {"torch.float32": true, "torch.float64": true},
  "empty_input": true,
  "half_and_full": true,
  "mixed_pbc": [[true, true, true], [false, false, false]],
  "native_composite_topology_materialization": "rocprim_int64_composite_key_sort_scatter",
  "native_topology_materialization": "rocprim_stable_radix_sort_scatter",
  "non_default_stream": true,
  "performance_measured": false
}
```

`composite_capacity_rejected` 验证的是 materializer 的 caller-owned public capacity 合同；
`capacity_overflow_explicit` 是 native query candidate 的既有容量错误边界，两者分别保留。

## 结论与限制

composite-key candidate 已从 5 原子 synthetic fixture 扩展到现有 Batch query/materialization
boundary，并在所有列出的 dtype、PBC、full/half、stream、empty 和容量场景通过 topology
parity。它仍是隔离 correctness candidate；未覆盖 FP64 performance、API wall-clock、
workspace reuse、host-sync removal、target/pair outputs、pair-centric query、compile/opcheck
或端到端 MACE/FIRE2。

下一小点可独立评估 composite candidate 的 workspace 生命周期和 host-side pair-count sync，
但应先保持现有 parity gate；不因 correctness boundary 通过而自动替换五次排序或进入 runtime。
