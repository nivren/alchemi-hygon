# G2 native HIP CSR atom-list fill boundary

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 12。

## 实现

新增显式、forward-only 的 `build_cell_atom_list_hip_into` native HIP candidate。它复用小点 11
的 keys/counts/starts/list/cursor ABI：先在 current stream 上以 `hipMemsetAsync` 清零独立
cursor，再由一个 HIP kernel 对每个 atom 的 cell cursor 做 `atomicAdd`，以
`cell_starts[key] - global_atom_offset + slot` 写入 global atom ID。public counts 与 starts
不被当作 transient cursor 覆写。

`cell_atom_list` 的同 cell 内 atomic insertion order 是未指定的。因此该 candidate 的正确性
合同是每个 CSR cell 的 atom **集合**、final cursor 和 public CSR metadata，而不是 stable list
order。它没有注册 dispatcher/capability、没有接入 `torch_reference_cell_list`、`auto` 或 `hip`
runtime 路径，也没有测量性能；不能将其视为完整 build 或 neighbor 加速。

上游对应 `bin_atoms` 的 atomic cursor fill 生命周期。本地 Torch oracle 的 stable key sort 仍是
公开顺序的基线；在验证现有 query 是否会 canonicalize 最终 pair output 前，atomic candidate 不得
替换该路径。

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
  packages/ops/test/torch/test_torch_reference_cell_list.py
```

退出码 0，`37 passed`。其中 native fill 的 CPU/非 HIP 调用明确失败，导入不会触发 JIT。`compileall`
与 `git diff --check` 通过。

主机权限 DTK 26.04、HCU 0（BW200/gfx936，PyTorch HIP `6.3.26093`）上限时运行
`probes/native_cell_atom_list_fill_hip_boundary.py` 退出码 0。实际 JIT 编译、链接、加载并执行 native
kernel；257 atoms、2 个异构 triclinic system、mixed PBC、FP32/FP64、strided read-only
keys/counts/starts、empty B=1 与 non-default stream 都通过。比较为每 cell sorted atom set；final
cursor 等于 counts，counts/starts 与 capacity tail 未变。未采集性能数据。

## 边界与下一步

尚未将 fused key/count、scan 与 atomic fill 串为一个 build candidate，尚未验证 atom-list 的未指定
内部顺序对现有 query/public pair order 的影响，也未实现 runtime、AOT、autograd、compile 或性能选择。

下一小点应先在隔离 probe 中组合已验证的 fused key/count、native scan 和 atomic fill，验证完整
Batch CSR metadata 与每 cell atom 集合；仍不接入 runtime。随后才可检验 query 对内部顺序的
canonicalization 与完整 neighbor parity。
