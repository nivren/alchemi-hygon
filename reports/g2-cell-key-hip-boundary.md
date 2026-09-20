# G2 native HIP cell-key custom-op boundary

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 2。

## 实现范围

在 shared cell-key ABI 之上增加一个不改变现有 dispatcher 的可选 native HIP 边界：

- `nvalchemiops._hip_cell_key` 注册私有
  `nvalchemiops::_cell_key_build_hip` mutation-only `torch.library.custom_op`；三个
  `int32` 输出 buffer 是显式 `mutates_args`，并提供 fake 注册。
- `_native/cell_key_build.cpp` 负责 shape、dtype、device、输出 contiguous 和同设备检查，
  `_native/cell_key_build.cu` 负责当前 stream 上的一原子一线程 fractional/wrap/cell-key
  kernel；只读输入允许 strided，并在取 raw pointer 前复制为 contiguous。输出布局与 Torch
  shared ABI 一致。
- loader 延迟到显式 HIP 调用时才执行，并把源文件复制到指定的
  `NVALCHEMI_HIP_CELL_KEY_BUILD_DIR` 或 Torch 默认 cache；不会在源码树生成 HIP 转换副本。
- 当前是 JIT-loaded native source boundary；wheel 携带 `.cpp/.cu` 源码，但没有把预编译
  `.so` 塞入 wheel，也没有修改 Hatchling 为正式 AOT 编译后端。没有 native HIP 时不会静默
  调用 Torch/CPU fallback。

## CPU 与静态验证

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_hip_cell_key_boundary.py \
  packages/ops/test/torch/test_cell_list_abi.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py -k 'cell_key or cell_list'
```

退出码为 0，`17 passed`。导入边界不编译、不初始化 Warp；CPU 调用显式失败。`git diff
--check`、Python `py_compile` 和已有 cell-list 回归均通过。

DTK 26.04 工具检查：`hipcc=/opt/dtk-26.04/bin/hipcc`，`dcc 25.10.0-0`、clang 17，
`hipcc.version_status=ok`。以 `PYTORCH_ROCM_ARCH=gfx936`、`MAX_JOBS=4` 调用 loader，
native C++/HIP 源码编译、链接并加载成功；随后复用同一 cache 再加载成功。第二次检查确认
`packages/ops/nvalchemiops/_native/cell_key_build.hip` 不存在，转换副本只在 cache 内。

重新构建 ops wheel 成功：`nvalchemi_toolkit_ops-0.4.1-py3-none-any.whl`。wheel 内容包含
`nvalchemiops/_native/cell_key_build.cpp`、`.cu` 与 loader/reference Python 文件，不包含
`.so` 或生成的 `.hip`。

## HCU / 数值证据

在主机权限执行上下文中，`/dev/kfd`、`/dev/dri` 可见，`hy-smi` 正常；项目 `.venv` 的
PyTorch 2.9.0 / HIP 6.3.26093 以 `HIP_VISIBLE_DEVICES=0` 识别一张
`BW200, UBB BW1000`。以下限时单卡 probe 退出码为 0：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTORCH_ROCM_ARCH=gfx936 MAX_JOBS=4 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python probes/native_cell_key_hip_boundary.py
```

257 原子、三斜晶胞、mixed-PBC `[true, true, false]` 的 native 输出与
`build_cell_keys_reference_into` 在 FP32、FP64 下逐元素一致。空输入保持输出语义一致；
在非默认 Torch stream 上执行后同步也一致；strided `positions` 和 `inverse_cell` 作为只读
输入可由 native boundary 连续化后得到同一输出。probe 明确没有测量性能。

该证据仅覆盖 cell-key build 子模块及其 current-stream ABI，不覆盖 sort/CSR/query/fill、
完整 cell-list dispatcher、梯度、compile/opcheck、预编译 AOT wheel 或性能门槛；因此不登记
`hip` capability，也不改变 `auto`。

## 后续

下一小点建议仅建立正式 AOT extension 构建入口，使 wheel 能携带目标 gfx936 的 native
artifact；它必须复用本报告的 output-parity probe，且仍不登记 `hip` capability、不改变
`auto`、不接入完整 cell-list。AOT 边界确认后再选择 query/count/fill 模块。
