# G2 native HIP Batch CSR build plus Torch-query order boundary

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 14。

## 结论与实现

当前 Torch direct query 已对最终 `(row, column, shift)` 调用 stable `_sort_pairs`，随后再 scatter 到
每 row 的 neighbor matrix。因此，native HIP atomic fill 所造成的 **同 cell 内 atom-list 无序** 不会
泄露到此范围的公开 neighbor order：不需要为该 direct query 添加 fill-side canonicalization。

本小点没有修改 runtime 或新增 native query。新增 CPU 回归将每个非空 CSR cell 内的 atom-list 反转，
再用现有 `query_cell_list` 比较完整输出；新增 HCU probe 以
`build_batch_cell_csr_hip_into` 的实际 atomic CSR outputs 调用现有 `batch_query_cell_list`，并与
stable Torch build composition 的 query outputs 对照。

结论仅适用于当前 Torch reference 的 atom-centric direct query、matrix/count/shift 和已实现的
distance/vector outputs。`target_indices`、`pair_fn`/pair outputs、pair-centric/sorted query、compile
与其他尚未支持的 public paths 不从本结论获益。

## 验证证据

CPU focused suite：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_cell_list_abi.py \
  packages/ops/test/torch/test_hip_cell_key_boundary.py \
  packages/ops/test/torch/test_hip_cell_count_boundary.py \
  packages/ops/test/torch/test_hip_cell_scan_boundary.py \
  packages/ops/test/torch/test_hip_batch_cell_key_count_boundary.py \
  packages/ops/test/torch/test_hip_cell_fill_boundary.py \
  packages/ops/test/torch/test_hip_batch_cell_build_boundary.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py
```

退出码 0，`43 passed`。新增四个 query regression 覆盖 PBC/no-PBC 与 full/half；对每种组合，稳定
CSR list 和按 cell 反转后的 list 得到完全相同的 matrix、num-neighbors、pair shifts、distances 与
vectors。`compileall` 与 `git diff --check` 通过。

在 device-visible 主机上下文，`/dev/kfd`、`/dev/dri`、DTK 26.04 hipcc（dcc `25.10.0` / clang
`17.0.0`）及 BW200 HCU 均可见。HCU 0、PyTorch HIP `6.3.26093` 限时运行
`probes/native_batch_cell_build_query_hip_boundary.py` 退出码 0：已验证 HIP fusion/scan/fill extension
实际加载和执行后，交由 Torch Batch query。16/11 atoms 的连续双体系（第一体系全 PBC、第二体系无
PBC）、triclinic cells、FP32/FP64、full/half 和 non-default stream 下，public neighbor matrix、
counts、pair shifts、distances、vectors 及其 row order 与 stable Torch CSR composition 一致。

没有性能采样，也没有将 HIP build path 接入 dispatcher 或 runtime。

## 下一步

现在可以在固定相同 contract 上测量完整 `HIP build + Torch query` 与 `Torch build + Torch query`，
分离 JIT/预热后的 device time，并先 profile 决定下一轮优先 native HIP irregular query 还是 Triton
geometry/materialization candidate。该 benchmark 仍不能直接修改 `auto`；必须满足完整 numerical
contract 和既定 neighbor `2x` 或目标端到端 `20%` 准入门槛。
