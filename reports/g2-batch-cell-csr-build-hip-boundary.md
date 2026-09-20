# G2 native HIP Batch CSR build composition boundary

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 13。

## 实现

新增 `build_batch_cell_csr_hip_into`，这是 Python 层的显式 forward-only composition，不是新 native
kernel 或 runtime backend。它按固定顺序组合三个已经独立验证的 candidate：

1. fused geometry/PBC/key/count 写 Batch-global keys/counts；
2. rocPRIM exclusive scan 从 counts 写 starts；
3. atomic fill 使用独立 cursor 写 CSR atom-list。

全部 outputs 和 scan workspace 仍由 caller 所有。workspace 应在调用前通过 scan 的显式 size query
分配；因为该 query 会校验 count 值，尚未执行 fusion 的 caller-owned count buffer 可以先置零。fusion
随后完整覆写它。`global_atom_offset` 同时传给 scan 与 fill，因此 starts 和 stored global atom IDs
使用同一 slice contract。

组合不注册 dispatcher/capability，不改变 `backend=None`/Warp、`auto` 或 `hip` runtime 选择。atomic
fill 的同 cell 内 list order 仍未指定，只能作为后续 query canonicalization 前的内部候选。

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

退出码 0，`39 passed`。组合 facade 导入不触发 JIT；CPU 调用在第一个 native phase 明确报告无可见
HIP Torch device，没有 Torch/CPU fallback。`compileall` 与 `git diff --check` 通过。

主机权限同一上下文确认 `/dev/kfd`、`/dev/dri`、`/opt/dtk-26.04/bin/hipcc`（dcc `25.10.0` / clang
`17.0.0`）及 BW200 HCU 可见。DTK 26.04、HCU 0、PyTorch HIP `6.3.26093` 上限时运行
`probes/native_batch_cell_build_hip_boundary.py` 退出码 0：fusion、scan 与 fill 三份 extension 均实际
JIT 编译/加载/执行。257 atoms、两个异构 triclinic system、mixed PBC、FP32/FP64、
`global_atom_offset=17`、strided read-only inputs、empty B=1 和 non-default stream 均通过。

对照独立 Torch composition：shifts/mapping/keys/counts/starts 逐元素相等；atomic list 按每 cell sorted
atom set 相等；cursor 等于 counts，capacity tail 未变。没有性能采样。

## 边界与下一步

这证明了 isolated build-half data flow 的 CSR metadata 与 atom membership，不证明现有 query 对未指定
atom-list order 的行为，也不证明 public neighbor matrix/pair shifts、half/full、梯度、compile、AOT
或端到端性能。没有 runtime 接线。

下一小点应以该 isolated composition 产生的 CSR 输入调用现有 Torch query，先对照 stable reference 的
public neighbor pair/shift **集合和顺序**，覆盖 PBC/no-PBC、half/full 与 Batch；只有明确结果后才能决定
是否需要 canonicalization，仍不改变 dispatcher。
