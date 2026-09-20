# G2 native Batch query candidate geometry materialization

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 19。

## 目标与实现边界

本小点把小点 17/18 的 query candidate 继续拆成两个语义层：

1. native HIP 只生成离散候选拓扑：`neighbor_matrix`、image shifts 和 per-row counts；
2. Torch materialization 负责与上游 reference 相同的 stable public order、matrix/count/shift
   scatter，并按需从原始 `positions`/`cells` 计算可微 vectors/distances。

新增私有 helper：
`packages/ops/nvalchemiops/_torch_batch_query_materialization.py`。
它复用 `torch_reference_cell_list._sort_pairs`，因此候选行内无序不会改变公共 pair order；
geometry 计算采用与 reference 相同的
`positions[row] - positions[column] - shift @ cell[batch[row]]`，没有 detach、CPU 回退或
静默降精度。拓扑仍是 forward-only，连续 geometry 路径保留一阶/二阶 autograd。

该 helper 没有接入 dispatcher、`auto`、`hip` capability、AOT 或生产 neighbor API。

## 正确性验证

CPU focused command：

```text
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_batch_query_materialization.py \
  packages/ops/test/torch/test_hip_batch_cell_query_boundary.py
```

结果：`6 passed`。新增测试覆盖：

- mixed/triclinic PBC Batch、full/half、候选 row 内乱序与 reference public output parity；
- distances/vectors 输出；
- positions/cells 的一阶和二阶梯度；
- candidate capacity 越界显式失败。

HCU boundary command：

```text
source scripts/activate_hygon_env.sh project && \
test -e /dev/kfd && test -d /dev/dri && \
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
PYTHONPATH=packages/framework:packages/ops timeout 300 \
  .venv/bin/python probes/native_batch_cell_query_candidate_hip_boundary.py
```

HCU 0（BW200/gfx936，DTK 26.04，PyTorch HIP `6.3.26093`）通过：
FP32/FP64、mixed PBC、full/half、空输入、非默认 stream、显式 capacity overflow，以及
candidate 经 Torch helper materialization 后的 matrix/count/shift/distance/vector parity。
该探针还在 HCU 上检查了 geometry 路径的一阶/二阶梯度有限。

## 高 Batch geometry device-time benchmark

benchmark 使用 `probes/native_batch_cell_query_materialization_benchmark.py --geometry`，
保持小点 18 的 workload、cell metadata、capacity、warm-up 和 HIP-event 口径：HCU 0、FP32、
cutoff `0.6`、capacity `256`、3 warm-up、5 交替 samples，固定两套 mixed-PBC geometry，
覆盖 `46/92 atoms × 32/64 systems` 与 uniform/clustered positions。

测量 scope：

- `enumeration_native`：native HIP enumeration；
- `query_native_torch_materialize`：native enumeration + Torch stable topology scatter +
  distance/vector materialization；
- `query_torch`：相同 CSR input 上的完整 Torch reference query，含 distance/vector outputs。

计时是预热后的 HIP-event device timeline，不包含 JIT、分配、Python API wall time 和 runtime
selection；geometry benchmark 没有在计时中启用 autograd，梯度合同由上面的 CPU/HCU correctness
probe 单独验证。

| atoms × systems | workload | native enumeration (ms) | native + Torch geometry (ms) | Torch geometry query (ms) | factor |
|---|---|---:|---:|---:|---:|
| 46 × 32 | uniform | 0.91088 | 2.59296 | 91.26470 | 35.20× |
| 46 × 32 | clustered | 1.00464 | 2.70352 | 111.99812 | 41.43× |
| 46 × 64 | uniform | 0.91040 | 2.55312 | 173.68333 | 68.03× |
| 46 × 64 | clustered | 0.98464 | 2.75248 | 218.78584 | 79.49× |
| 92 × 32 | uniform | 0.91616 | 2.55776 | 101.96580 | 39.87× |
| 92 × 32 | clustered | 1.03984 | 2.70256 | 110.27476 | 40.80× |
| 92 × 64 | uniform | 0.94080 | 2.64384 | 200.70393 | 75.91× |
| 92 × 64 | clustered | 1.06160 | 2.77280 | 219.11336 | 79.02× |

factor 定义为 `Torch geometry query / native enumeration + Torch geometry`。8 个 workload
均通过计时前后的 public output parity，factor 为 `35.20×--79.49×`；candidate full 的相对
总体标准差为 `0.66%--4.70%`，显著小于与 Torch reference 的差距。原始一值一行 samples 位于：

- `artifacts/native-batch-cell-query-geometry-benchmark-46x32/`
- `artifacts/native-batch-cell-query-geometry-benchmark-46x64/`
- `artifacts/native-batch-cell-query-geometry-benchmark-92x32/`
- `artifacts/native-batch-cell-query-geometry-benchmark-92x64/`

`compare_timings.py` 已对 8 组 raw samples 独立复核，candidate factor 与表中一致。

## 结论与未验证边界

在当前 workload 上，即使保留 Torch 的排序/scatter 和可微 geometry materialization，native
enumeration candidate 仍明显低于完整 Torch query 的 device-time。这支持下一小点评估将
canonicalization/materialization 中适合规则分块的部分交给 Triton，或将不规则/显式线程路径交给
HIP；不能据此固定 HIP 一定优于 Triton。

本小点仍不是完整上游等价或生产 backend 支持。未覆盖：FP64 性能、API wall-clock、allocation/
validation 成本、rebuild、`target_indices`、pair callbacks/outputs、pair-centric/sorted query、
compile/opcheck、分布式 ownership、端到端 MACE/FIRE2，以及 runtime/`auto`/`hip` 接线。
