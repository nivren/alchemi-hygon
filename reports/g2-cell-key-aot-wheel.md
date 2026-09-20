# G2 native HIP cell-key AOT wheel

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 3。

## 实现范围

正式 AOT 入口保留现有 Hatchling 基线 wheel，不把 Hygon Torch 编译依赖加入通用 build
isolation：

- `scripts/build_hip_cell_key_aot_wheel.py` 先构建一个基础 ops wheel，再以活跃 Hygon PyTorch
  和 `PYTORCH_ROCM_ARCH=gfx936` 编译 native cell-key extension。
- 脚本将 `.so` 注入 `nvalchemiops/_native/`，把 wheel 改为 CPython/platform tag，更新
  `Root-Is-Purelib: false` 与全部 `RECORD` 哈希，并写入 artifact/arch/HIP provenance JSON。
- `_hip_cell_key` 在 AOT artifact 存在时优先加载它；加载失败显式报错，不回退到 JIT。源码
  checkout 中没有该 artifact 时才保留此前的显式 JIT 路径。
- `packages/ops/Makefile` 新增 `build-hip-cell-key-aot` 入口。它只构建这个可选子模块，
  不登记 `hip` capability、不改 dispatcher 或 `auto`。

## 构建与 wheel 证据

主机权限 DTK 26.04、项目 `.venv`、HIP 6.3.26093 下执行：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python scripts/build_hip_cell_key_aot_wheel.py \
  --source-dir packages/ops \
  --out-dir /tmp/alchemi-cell-key-aot-wheel-v1 \
  --build-dir /tmp/alchemi-cell-key-aot-build-v1 \
  --arch gfx936
```

构建成功，输出：

- 基础：`nvalchemi_toolkit_ops-0.4.1-py3-none-any.whl`；
- AOT：`nvalchemi_toolkit_ops-0.4.1-cp312-cp312-linux_x86_64.whl`；
- artifact：`nvalchemiops/_native/nvalchemi_cell_key_hip.so`。

`unzip -t` 通过。wheel 元数据为 `Root-Is-Purelib: false`、
`Tag: cp312-cp312-linux_x86_64`；独立校验 182 条 `RECORD` 记录均匹配 SHA-256 与大小。
provenance 记录 `gfx936` 和 HIP `6.3.26093`。

文档化入口 `make -C packages/ops build-hip-cell-key-aot` 也在同一主机环境退出码为 0，
在 `packages/ops/dist/hip-cell-key-aot/` 产出相同 tag 的 wheel。

## HCU 装载与数值证据

从临时目录解包 AOT wheel、只将该目录置入 `PYTHONPATH` 后，在主机 HCU 0
`BW200, UBB BW1000` 执行 `probes/native_cell_key_hip_boundary.py`。退出码为 0，probe 回报的
`extension_path` 为临时解包 wheel 内的
`nvalchemiops/_native/nvalchemi_cell_key_hip.so`，证明本次未走源码/JIT loader。

与小点 2 相同，257 原子三斜 mixed-PBC 的 FP32/FP64 输出与 Torch reference 逐元素一致；
空输入、strided read-only inputs 和非默认 stream 也通过。probe 不测性能。

## 分发与功能边界

该 artifact 针对本机 CPython 3.12、Linux x86_64、DTK 26.04、HIP PyTorch 2.9.0/6.3.26093 与
gfx936 构建；它不是 manylinux 或跨 DTK/PyTorch/架构兼容声明。完整 wheel 安装、依赖解析和
发布策略仍需独立验收。

当前只验证 cell-key build；sort/CSR/query/fill、完整 cell-list dispatcher、梯度、compile/
opcheck、性能门槛均未完成。下一小点应先冻结并实现 `cell_key -> cell_count/start` 的 CSR count
ABI 和 Torch oracle，再评估 native HIP atomic count kernel。
