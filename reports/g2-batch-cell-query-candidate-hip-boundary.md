# G2 native HIP Batch cell-query candidate boundary

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 17。

## 独立 query/materialization contract

本小点将 query 拆成两个边界：

1. **pair enumeration**：从 caller-owned Batch CSR cell storage 枚举满足 cutoff 的
   `(source_atom, target_atom, image_shift)` 候选；拓扑是离散的、forward-only。
2. **public materialization**：对候选做稳定排序、neighbor matrix scatter、distance/vector
   计算和后续 autograd。当前仍由 Torch reference 负责。

enumeration candidate 的输入约定为：

- `positions (N, 3)`、`cells (B, 3, 3)` 为 FP32 或 FP64，`pbc (B, 3)` 为 bool；
- `batch_idx (N,)`、`cells_per_dimension (B, 3)`、`neighbor_search_radius (B, 3)`、
  `cell_offsets (B,)` 和所有 CSR metadata/list 为 contiguous int32；
- `atom_periodic_shifts`、`atom_to_cell_mapping` 与 CSR counts/starts/list 由已验证 build
  pipeline 提供；
- caller-owned outputs 为 `neighbor_matrix (N, K)`、`neighbor_matrix_shifts (N, K, 3)`、
  `num_neighbors (N,)`，均为 contiguous int32；
- 所有 tensor 在同一 HIP device，使用当前 stream；CPU、非 HIP 和隐式 fallback 均明确失败。

输出语义为：每个 row 对应一个 source atom；候选 row 内顺序不稳定，不得直接当成 public
neighbor order。candidate 必须执行：邻近 cell offset 遍历、mixed PBC wrapping/image shift、
atom periodic shift 合成、严格 cutoff、self zero-image 排除和 full/half 过滤。capacity overflow
和 periodic overlapping active pair 必须显式报错，不能截断。

public materialization 继续遵循 Torch reference 的稳定排序（row、column、shift）、capacity
scatter、距离/向量定义与一阶/二阶连续梯度路径。`target_indices`、pair callbacks、pair-centric
query、compile 和 candidate 本身的 autograd 不在本小点内。

## 实现

新增：

- `packages/ops/nvalchemiops/_hip_batch_cell_query.py`：lazy HIP custom-op boundary；
- `packages/ops/nvalchemiops/_native/batch_cell_query.cpp/.cu`：每个 source atom 遍历邻近
  cell 的 forward HIP kernel；
- `probes/native_batch_cell_query_candidate_hip_boundary.py`：与 Torch reference 的独立 parity
  probe；
- `packages/ops/test/torch/test_hip_batch_cell_query_boundary.py`：CPU/no-fallback boundary。

kernel 逻辑与上游/本地 Torch `_query_pairs` 的 enumeration 语义一致，但暂不复制其 public
sort/materialization：native 输出先由 probe canonicalize，再与 stable reference 的 pair、shift、
distance 和 vector 对照。

## 验证证据

CPU boundary：`2 passed`，包括 import 不触发编译、CPU 输入显式拒绝且不静默 fallback。

HCU 0（BW200/gfx936，DTK 26.04，PyTorch HIP `6.3.26093`）probe 最终输出：

```json
{
  "atoms_per_system": [16, 11],
  "batch_systems": 2,
  "capacity_overflow_explicit": true,
  "dtype_parity": {"torch.float32": true, "torch.float64": true},
  "half_and_full": true,
  "mixed_pbc": [[true, true, true], [false, false, false]],
  "non_default_stream": true,
  "public_order_not_exposed": true,
  "performance_measured": false
}
```

该 probe 覆盖了 native build CSR → native query enumeration → Torch canonicalization/materialization
的组合；pair、shift、distance、vector 与 reference 一致。它是 correctness boundary，不是性能
结论，也不代表 native query 已支持完整上游 neighbor API。

## 下一步

先在同一高 Batch 矩阵上测“native enumeration + Torch materialization”与完整 Torch query，区分
enumeration、canonicalization 和 distance/vector 的成本；随后再决定将 public canonicalization
放入 HIP 还是 Triton。通过代表性性能门槛和更宽数值/梯度验证前，不接入 dispatcher、`auto` 或
默认 `hip` capability。
