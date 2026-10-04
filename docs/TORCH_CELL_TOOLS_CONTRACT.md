# M2 通用晶胞工具契约

来源为锁定 ops `26dbceb` 的 `dynamics/utils/cell_filter.py` 和 framework
`4dfe372` 的 `dynamics/_ops/cell_align.py`。指定 fork `eadc3ee` 的同文件已核对；
其 lower-triangular 文案修复采用，未整体同步新版本。

本次仅移植已有 align、pack/unpack、extend 工具，不实现优化步、ASE filters 或 NPT。
operation 为 `cell_tools`，executor 为 `nvalchemiops.cell_reference`；framework
保留 `align_cell` 入口，增加 keyword-only `backend/selection`。默认 None/warp 仍执行
lazy Warp。其他工具采用 ops executor 的上游同名 ABI（Torch 张量替代 Warp 数组），
可通过已解析 selection 和 generic executor 调用，不改变原 Warp 入口。

实现与测试提交 `036827e`；CPU/HCU4 的窄验证见 [M2 报告](../reports/g2-cell-tools-m2.md)。

## 数据与布局

- 原子向量 `[N,3]`、cell/velocity/force/transform `[B,3,3]`，标量 mass `[N]`
  和 cell mass `[B]`。FP32/64、同设备同 dtype；整数 idx/ptr 为 int32 或 int64。
  数值单位沿用调用者，pack 不缩放任何量；无邻居/PBC 拓扑操作。
- 对齐为正对角下三角 L，保存行晶格 metric，`r=s@H`。scratch transform 为
  `T=inverse(L)@H`，原子行坐标更新为 `r@T.T`。右手输入是旋转，左手输入为反射；
  两者保持分数坐标和周期距离。拒绝非有限/奇异或数值不可对齐晶胞。
- 六项依照上游实际顺序，两个向量为 `[H00,H10,H20]`、`[H11,H21,H22]`。
  unpack 将上三角置零；对全矩阵 pack 只提取六项，故 round-trip 保证仅适用于下三角。
- 单体系追加两个向量；batched pack 为逐体系原子后接两个向量的交错布局。
  `ext_ptr=ptr+2*arange(B+1)`，原子数允许不同，允许零原子体系和整个空 Batch。
  batched 必须同时提供 atom_ptr/ext_atom_ptr；batch_idx 必须与连续 CSR 块一致。
- **extend_batch_idx 保留上游的另一种布局**：所有原子索引后追加每体系两个索引。
  它不能与 batched pack 的交错数组配对使用。交错索引应由 ext_ptr 的块长度
  repeat_interleave 得到；M3 adapter 按已有上游 torch/fire2.py 的此方式构造。

## Mutation、错误和验收

输出均由调用者预分配，原地写入并返回相同对象；framework align 返回 None，
ops align 返回 (positions,cell)。其他输入不改写。支持非连续但非重叠的张量；
写入缓冲与其他张量共享 storage（含不相交视图）明确拒绝；内部重叠视图不支持。
align 的 positions/cell/transform 必须独立。整数溢出、错位/非单调 ptr、容量、
dtype/device、非有限输入、requires_grad 输入和未知/未支持后端均明确失败。
forward-only：不支持优化路径梯度或 torch.compile，注册梯度等级为 0；不通过 detach
掩盖需求。使用当前 Torch stream，检查可能同步主机；无 CPU 回退或 dtype 转换，
不保证跨设备逐位一致；性能未验收。

CPU 验收 FP32/64：非正交左右手晶胞对 ASE lower standard_form/metric/分数坐标/
周期位移与解析径向能量不变；异构和空体系 pack/unpack 精确 round-trip、索引归属、
质量复制、输入错误、非连续写入、selection 只解析一次、第二 executor 和无 Warp 导入。
几何门 FP64 rtol/atol=2e-12，FP32=2e-5；pack 与整数门为精确相等。
资源允许时 HCU4 限时执行同一测试和 CPU parity（FP64 2e-12、FP32 2e-5）。
Warp/NVIDIA 实际运行若不可用单独列出；stub 仅验证 bridge 写回，不作为 Warp 数值证据。

M2 primitive 证据保留不变。M3 的 AlignCellHook 后端接线、周期/active 子集和 cached
F/v/stress 协变、MPA-0 CPU 旋转对照单列在
[优化器契约](TORCH_VARIABLE_CELL_FIRE_CONTRACT.md) 与 [M3 报告](../reports/g2-variable-cell-fire-m3.md)。
Hook 是预处理，不准入任意运行中晶胞重参数化。

UBF-010：原 Warp 负 determinant 分支把 H 取负后用于 transform，破坏分数坐标；
应以原始 H 计算 transform。framework bridge 对 positions.contiguous() 缺少写回；
修正非连续坐标丢失。来源、测试和验证边界见 UPSTREAM_BUGFIXES.md。
