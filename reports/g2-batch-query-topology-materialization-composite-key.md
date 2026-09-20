# G2 Batch query topology composite-key candidate

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 24。

## 目标与范围

本小点验证 single composite key 是否能在不改变上游 public order 的前提下，替代当前
五字段稳定 radix sort。候选只覆盖 topology materialization：将
`row -> column -> shift_x -> shift_y -> shift_z` 编码到一个非负 signed `int64` key，
执行一次 rocPRIM radix sort，再恢复 public matrix、count 和 shift。

本小点没有修改 dispatcher、AOT wheel、`auto` 或 `hip` capability，也没有替换现有五次
稳定排序 candidate 的运行路径。workspace 生命周期、allocation 和 host-side pair-count
sync 仍保持原状；没有把“少一次排序”直接当作性能收益。

## 编码合同

- `row_bits = max(1, atoms.bit_length())`，因为 padded candidate 还要表示 `fill_value=atoms`；
  `column` 使用同样的位宽；
- 每个有符号 PBC shift 使用 `shift + shift_bias` 编码，范围为
  `[-shift_bias, (1 << shift_bits) - 1 - shift_bias]`；
- 总位宽为 `2 * row_bits + 3 * shift_bits`，必须不超过 signed `int64` 的 63 个安全位；
- key 按 `row -> column -> shift_x -> shift_y -> shift_z` 的高位到低位排列，因而非零 key
  的 radix order 与上游 `_sort_pairs` 的 public lexicographic order 相同；
- 编码范围在 device 输入上显式检查，超出范围报告错误，不截断 shift 或静默改 dtype。

这里的 `int64` 是 packed key 的存储/标量索引容器，不是 FP64，也不是要求 gfx936 的矩阵
核心以 int64 做高吞吐计算。gfx936 可以进行标量 int64 load/store 和必要的整数运算，但其
吞吐和 rocPRIM int64 radix-sort 成本仍需独立实测；int64 还会带来相对于 int32 的更大 key
带宽和 workspace 占用。

## 验证结果

### CPU

```text
compileall + git diff --check: passed
pytest packages/ops/test/torch/test_hip_batch_query_materialize.py \\
       packages/ops/test/torch/test_hip_batch_query_materialize_composite.py
5 passed
```

CPU 测试覆盖 signed-int64 位宽溢出、shift bias 范围和 CPU 调用显式失败；没有 CPU fallback。

### HCU

环境为 DTK 26.04、HCU 0、BW200、`gfx936`、PyTorch HIP `6.3.26093`。固定的 unordered
5 原子/5 capacity mixed-sign PBC fixture 在默认 stream 和非默认 stream 上，对照现有
五次稳定排序 HIP candidate：

```json
{"atoms": 5, "capacity": 5, "default_parity": true,
 "device": "BW200, UBB BW1000", "non_default_stream_parity": true,
 "row_bits": 3, "shift_overflow_rejected": true,
 "torch_hip": "6.3.26093", "total_key_bits": 18}
```

因此本小点的 correctness slice 通过；这不是高 Batch 性能结果，也不验证 geometry、
autograd、target/pair outputs、compile/opcheck 或端到端 MACE/FIRE2。

## 结论与下一步

single composite key 的字段布局、负/正 PBC shift 编码、signed-int64 安全位宽和溢出失败
行为已通过窄 HCU 边界验证。它可以作为后续性能实验的候选，但尚未证明比五次 int32
稳定排序更快；特别需要关注 gfx936 上 int64 radix sort 的吞吐、key 带宽和临时 workspace。

下一小点在相同 `46/92 atoms × 32/64 systems`、uniform/clustered、相同 warm-up/sample
合同下，比较 int64 single-key 与现有五次 int32 stable-sort topology stage 的 HIP-event
device time，并保留 parity gate。workspace reuse 和 host-sync removal 继续作为独立实验。
