# G2 Batch query topology materialization profile

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 23。

## 目标与范围

本小点使用 DTK 26.04 `hipprof` 对小点 21 的 native HIP topology
materialization correctness candidate 做受限 kernel profile，拆解：

- 五轮稳定 radix sort 相关 kernel；
- candidate field gather；
- `prepare_fields`、`initialize_order` 和 public scatter；
- scan 与临时分配/host synchronization 的可观测边界。

profile 只用于定位热点，不改变实现、dispatcher、AOT、`auto` 或 `hip`
capability，也不把 profiler 运行时间当作正式性能 benchmark。

## 测量合同

- 设备：HCU 0，BW200，目标架构 `gfx936`；DTK 26.04；PyTorch HIP
  `6.3.26093`；FP32；capacity `256`。
- workload：`46 atoms × 32 systems` 与 `92 atoms × 64 systems`，各覆盖
  uniform/clustered positions。
- 每个 workload 使用 1 次 warm-up、1 次 measurement sample；该 sample
  只用于和 kernel trace 对齐，不用于性能准入。
- 正确性仍由既有 benchmark 在 profile 前后检查 public
  matrix/count/shift parity；四个 workload 均通过。
- profile 使用项目环境入口
  `source scripts/activate_hygon_env.sh project`。直接 source
  `/opt/dtk-26.04/env.sh` 会把 `DTKROOT` 指向错误的旧路径，导致 JIT
  编译失败；该失败不计入本次有效 profile。

profile command 的核心形式为：

```sh
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 240 \
  /opt/dtk-26.04/bin/hipprof --stats --hip-trace \
  -o artifacts/hipprof-topology-46x32-close \
  /usr/bin/bash -c 'source scripts/activate_hygon_env.sh project && \
    HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
    .venv/bin/python probes/native_batch_cell_query_materialization_benchmark.py \
    --atoms-per-system 46 --batch-systems 32 --warmup 1 --samples 1 \
    --geometry --breakdown --hip-topology'
```

`92×64` 使用相同命令，仅替换 workload 参数和输出目录。原始数据库和
`hipkernel.csv` 保留在：

- `artifacts/hipprof-topology-46x32-close.db`；
- `artifacts/hipprof-topology-46x32-close.hipkernel.csv`；
- `artifacts/hipprof-topology-92x64-close.db`；
- `artifacts/hipprof-topology-92x64-close.hipkernel.csv`。

## kernel 证据

profile 中 native topology candidate 共出现 8 次（两个 workload、每个
workload 两种分布和 profile 内的固定调用序列）。以下是按 kernel 调用
次数归属于 candidate 的累计 kernel duration，再除以 8 次调用的结果；单位
为单次调用的近似 device kernel time，不包含 Python、JIT、allocator host
时间，也不是完整 custom-op API wall time。

| scope | `46×32` | `92×64` | 说明 |
|---|---:|---:|---|
| 五轮 sort 相关 kernel | `0.755 ms` | `1.638 ms` | 小规模为 radix block/merge-path；大规模为 rocPRIM onesweep 相关 kernel |
| `gather_key_kernel` | `0.034 ms` | `0.113 ms` | 每轮一次，五轮共五次 |
| `prepare_fields_kernel` | `0.011 ms` | `0.030 ms` | 从 padded candidate 拆出 row/column/shift 字段 |
| `initialize_order_kernel` | `0.006 ms` | `0.021 ms` | 初始化 identity order |
| `scatter_public_kernel` | `0.007 ms` | `0.009 ms` | 写 public matrix/shift |

在已能明确归属于该 candidate 的 native kernel duration 中，sort 相关部分
约占九成以上；gather、prepare 和 order 初始化是次要成本，public scatter
不是当前主要热点。`92×64` 下 rocPRIM 的 sort 实现形态发生变化，不能把
小规模的单一 kernel 名称或 block 配置直接外推到所有输入。

`exclusive_scan`、public output 初始化、临时 Tensor 分配和
`candidate_counts.sum().item()` 的 host synchronization 仍不能从这份混合
进程 kernel CSV 中无歧义地单独归属。它们保留为需要后续专用 API/trace 或
workspace 实验验证的边界，不能据此声称已经量化了 host sync 或 allocation
成本。

## 结论与下一步

1. 当前 HIP topology candidate 的第一热点是五字段稳定排序，而不是
   `scatter_public`；单纯优化 scatter 不太可能解决高 Batch 下的变慢。
2. 下一候选应优先减少五次全量排序的工作量，首选验证保持同一 lexicographic
   public order 的 single composite key；编码必须处理负 PBC shift、字段范围、
   `int32` 溢出和 stable tie order，不能仅以减少 sort 次数为理由改变语义。
3. workspace reuse 和去除 host-side pair-count sync 仍应作为独立实验，不能
   与 composite-key 算法变化混在一个性能结论中。
4. 在新候选通过相同 parity/gradient/capacity contract 并重新覆盖
   `46/92 × 32/64` 矩阵前，不登记 `hip` capability、不接 dispatcher 或
   `auto`。

完整 candidate benchmark 仍见
`reports/g2-batch-query-topology-materialization-benchmark.md`；本报告只补充
kernel hotspot evidence，不改变小点 22 的“不具备稳定收益”结论。
