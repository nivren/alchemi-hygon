# G2 Batch query native HIP forward geometry candidate

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 30。

## 目标与实现

本小点在 Point 29 的 public canonical topology 上增加隔离的 native HIP geometry
candidate：

```text
native HIP query
    -> native HIP composite-key topology
    -> native HIP forward distance/vector geometry
```

新增：

- `packages/ops/nvalchemiops/_hip_batch_query_geometry.py`：输入/输出 ABI、lazy HIP
  extension、mutation-only custom-op 和显式 CPU/HIP 边界；
- `packages/ops/nvalchemiops/_native/batch_query_geometry.cpp/.cu`：一个 `(row, slot)`
  对应一个线程，读取 public matrix/shift/count，写 distance/vector；
- `probes/native_batch_cell_query_candidate_hip_boundary.py`：将 native geometry 加入已有
  Batch query correctness boundary；
- `probes/native_batch_cell_query_materialization_benchmark.py`：增加完整 native geometry
  hybrid 以及固定 topology 的 geometry-only stage 对照。

该 candidate 只负责 forward distance/vector，不提供 autograd/二阶梯度，不接 dispatcher、
AOT、`auto` 或 `hip` capability。Torch geometry 仍是训练和梯度路径的 reference。

## 正确性

CPU focused suite：`8 passed`，包含 native HIP geometry import/CPU/autograd rejection boundary 和
既有 topology/geometry reference、梯度回归。

HCU 0/BW200/gfx936 boundary 通过：

- FP32/FP64；
- mixed PBC；
- full/half；
- empty、capacity、zero-count、non-default stream；
- native composite topology 后接 native HIP geometry；
- public matrix/count/shift/distance/vector 与 Torch reference parity。

native FP32 geometry 与 Torch reference 的少量舍入差异按 `1e-6` tolerance 验证；拓扑
matrix/count/shift 仍保持逐元素严格 parity。native geometry 不参与梯度，不能把本报告的
forward parity 扩大为训练路径支持。

## 性能合同

设备为 HCU 0/BW200、`gfx936`、DTK 26.04、PyTorch HIP `6.3.26093`。FP32、capacity `256`、
固定 grid/CSR metadata，覆盖 `46/92 atoms × 32/64 systems` 的 uniform/clustered workload，
每个 workload 3 次 warm-up、5 次 samples。

两个 scope 分开记录：

1. `materialize_geometry_native_hip / materialize_geometry`：相同 public topology 上的
   geometry-only device-time；API 对照显式用 `torch.cuda.synchronize()` 包围调用。
2. `query_native_hybrid_native_geometry / query_torch`：native query + composite topology
   + native geometry 与完整 Torch query/materialization 的窄 scope 对照。

两者均不包含每次调用重新 build cell-list CSR metadata、JIT、首次分配或 runtime dispatch。
每个样本前后检查完整 public output parity。

## 结果

factor 定义为 candidate / Torch，低于 `1.0x` 表示 candidate 时间更低。

### Geometry-only

| atoms × systems | workload | device-time factor | API wall-clock factor |
|---|---|---:|---:|
| 46 × 32 | uniform/clustered | 0.0803x/0.0784x | 0.1114x/0.1092x |
| 46 × 64 | uniform/clustered | 0.0821x/0.0873x | 0.1230x/0.1180x |
| 92 × 32 | uniform/clustered | 0.0848x/0.0876x | 0.1223x/0.1231x |
| 92 × 64 | uniform/clustered | 0.1165x/0.1220x | 0.1441x/0.1537x |

跨 8 个 workload，geometry-only device factor 为 `0.0784x--0.1220x`，API factor 为
`0.1092x--0.1537x`。说明 kernel 本身有稳定收益，但小 geometry workload 的 launch/API
成本不可忽略。

### 完整窄 hybrid query/materialization

| atoms × systems | workload | device-time factor | API wall-clock factor |
|---|---|---:|---:|
| 46 × 32 | uniform/clustered | 0.0132x/0.0116x | 0.0134x/0.0126x |
| 46 × 64 | uniform/clustered | 0.0077x/0.0069x | 0.0082x/0.0075x |
| 92 × 32 | uniform/clustered | 0.0135x/0.0146x | 0.0147x/0.0155x |
| 92 × 64 | uniform/clustered | 0.0075x/0.0074x | 0.0083x/0.0079x |

该结果只说明固定 CSR metadata 下的 query/materialization candidate 明显快于当前 Torch
reference，不代表完整 build-inclusive neighbor、MD/MACE 或整个上游 neighbor API 的同等
加速。

## 结论与下一步

Point 30 证明 native HIP forward geometry 在 gfx936 上可复现 Torch public geometry，并在
固定 topology 上取得稳定 device/API 收益；因此它可以作为后续 hybrid backend 的性能候选。

当前仍有边界：

- native geometry 没有 autograd/二阶梯度；
- 未接 dispatcher、AOT、`auto` 或 `hip` capability；
- 未覆盖 FP64 performance、build-inclusive end-to-end、rebuild、`target_indices`、
  `pair_fn`/pair outputs、compile/opcheck；
- 没有实现 Triton geometry，当前只保留 native HIP candidate；
- 没有宣称完整 upstream neighbor API 或生产默认路径已切换。

下一小点建议先做 native geometry 的训练路径决策：若继续保留 Torch geometry 以获得一/二阶
梯度，进入 build-inclusive end-to-end benchmark；若要替换 geometry，则先设计并验证 native
autograd contract，不能直接把当前 forward-only candidate 接入训练。

原始 samples 保存在：

- `artifacts/g2-point30-stage-46x32/`；
- `artifacts/g2-point30-stage-46x64/`；
- `artifacts/g2-point30-stage-92x32/`；
- `artifacts/g2-point30-stage-92x64/`。
