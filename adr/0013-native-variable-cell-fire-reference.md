# ADR-0013：原生变胞 FIRE/FIRE2 的独立 reference 接线

日期：2026-10-04。状态：接受，限 M3 update/reference 验收。

## 背景

NVIDIA 原生 FIRE2 是下三角六项，指定 fork 修复后的 FIRE 是完整 H/Hdot、
pre/post half-kick。已有 `fire` fixed-cell executor 和 `velocity_verlet` capability
不能代表完整变胞 FIRE，也不能用 ASE deformation-gradient filters 替代来源语义。

## 决定

- 六项更新以 operation `fire2` 的独立 implementation `torch_reference.fire2-cell-v1`
  声明 `variable_cell/six_cell_dofs`。完整矩阵以 operation `fire_cell` 的
  `torch_reference.fire-cell-v1` 声明 `variable_cell/full_cell`，pre/post 两 entrypoint。
- ops-owned `cell_fire_reference` 实现原生更新，framework 只冻结 selection 并桥接，
  使用既有 generic executor，无中心 operation 分派表。旧私有 reference import 转发保留。
- 两种控制顺序、质量/力缩放、位移限制分别保留；更新梯度0，stress 转换既有梯度不变。
- None/warp 继续选择 legacy。fork 修复同时摘入相应默认路径，注释 UBF-011–013；
  本机实际 Warp 未运行。AlignCellHook 的 fork 掩码/本地同帧修复单列 UBF-014。
- 不引入 ASE filter、新 cell mass、压力参数或工作流重写；Hook 只作对齐预处理。

## 后果与证据

reference 可在无 Warp 环境运行两种公共优化器，原 fixed-cell 与高级接口保留。
这是平台 backend 边界和 fork bug 修复，完整材料弛豫应用仍待 M4/M5。
详见 [契约](../docs/TORCH_VARIABLE_CELL_FIRE_CONTRACT.md)、
[报告](../reports/g2-variable-cell-fire-m3.md)。Compile/轨迹梯度/DD/checkpoint/生产性能
未验收；新增 entrypoint 是私有执行层 ABI，不是新增科学用户参数化。
