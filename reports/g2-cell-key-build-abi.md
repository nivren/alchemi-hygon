# G2 cell-key build shared ABI

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 1。

## 范围

新增私有模块 `nvalchemiops._cell_list_abi`，以 Torch reference 固化将来 HIP/Triton
共用的 build/binning 子模块 ABI。它不登记新 backend、不改变 dispatcher，也不改变
`backend=None`/Warp 或 `auto` 的选择语义。

- 输入：`positions (N, 3)`、同设备同 dtype 的 `inverse_cell (3, 3)`、`int32`
  `cells_per_dimension (3,)` 和 `bool pbc (3,)`；位置与晶胞逆只支持 FP32/FP64。
- 输出：调用方分配的 `int32` `atom_periodic_shifts (N, 3)`、
  `atom_to_cell_mapping (N, 3)` 与 `cell_keys (N,)`。`*_into` 只写这三个显式输出
  buffer；便利函数返回同一布局的分配结果。
- 语义：以 fractional coordinate 计算 periodic image shift、wrap 后 cell coordinate 与
  row-major linear key；非周期维 clamp，超出 `int32` key 范围显式失败。

`torch_reference_cell_list._build_into` 已改为通过该 ABI 构建 key，再继续其既有
sort/CSR/query 逻辑。因此本小点是未来原生输出 kernel 的可替换边界，而不是新的性能实现。

## CPU 合同与数值结果

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_cell_list_abi.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py \
  packages/ops/test/torch/test_backend_registry.py -k 'cell_list or cell_key'
```

退出码为 0，`15 passed, 17 deselected`。新增的三斜 mixed-PBC 夹具给出 image shift、cell
coordinate 和 linear key 的逐元素预期值；另验证 in-place 输出、dtype、正维度和 PBC dtype
错误。现有周期性 full/half、Batch selective rebuild、capacity 与 vector/distance 二阶梯度
回归仍通过。`git diff --check` 和两处修改模块的 `py_compile` 也通过。

## HCU 验证边界

执行前 `hy-smi` 显示八张 HCU 均 Normal、空闲。但项目 `.venv` 的运行时指纹为
`torch=2.9.0`、`hip=6.3.26093`、`torch.cuda.is_available()=False`、`device_count=0`。
因此限时 `probes/pbc_cell_list_reference.py --device cuda` 在设备可用性门处退出，未进入
算子；本小点没有新增 HCU 正确性或性能证据，也不将其记为 HCU 功能失败。

## 后续

确认此 ABI 后，下一小点应仅建立与其一一对应的 native HIP custom-op/AOT 装载边界和
`cell_key_build_into` 的 capability 状态；先验证编译、stream、输出与本 ABI 的 parity，仍不
接入 query/CSR/dispatcher，也不作性能门槛结论。
