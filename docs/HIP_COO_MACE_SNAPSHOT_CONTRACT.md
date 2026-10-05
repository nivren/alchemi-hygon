# HIP COO 与 MACE 单点契约（里程碑 2）

起点：HIP M1 收口 `8120282`，实现 `fecbc39`。本阶段已获用户确认。

## 目的与实现边界

复用 HIP full-list MATRIX 核心与设备内 Torch 格式转换，准入显式 periodic/full/COO，
接通 `compute_neighbors`、`NeighborListHook(skin=0)` 和消费外部 COO 的模型接口。
COO 是通用邻居输出能力；MPA-0 能量/力/应力对照是独立模型验收，不推定其他 MLIP 已验证。

锁定上游 ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` 的
`torch/neighbors/neighbor_utils.py::get_neighbor_list_from_neighbor_matrix` 采用 MATRIX
转 COO；framework `4dfe3723def34df3fadb245981081ccf8c94c257` 的 `neighbors.py`、
`hooks/neighbor_list.py` 与 `models/mace.py::MACEWrapper.adapt_input` 为接线来源。
产品直接 ops 输出复用 reference 的 `_result`，framework 复用 `_neighbor_matrix_to_coo`。
native query 与 materialization 仍输出 MATRIX，不新增 native COO kernel。

## 数据与行为

- 沿用 M1 的 HIP、FP32/64、Batch/global indices、full-PBC 当前晶胞完整重建；
  fixed-cell mixed-PBC 延续旧 gate，新增变胞 mixed-PBC 另列未验证。
- ops `return_neighbor_list=True` 输出 `edges[2,E]` int32、每原子 `row_ptr[N+1]`
  int64（沿用 reference cumsum ABI）、`shifts[E,3]` int32；framework Batch 存储
  `neighbor_list[E,2]`、`neighbor_list_shifts[E,3]` int32，按源原子的体系分段。
- 只打包每行前 counts 项；保留所有不同 image 和合法重复项，排除 padding。
  顺序保留实际 MATRIX 行顺序和行内顺序，不承诺不同实现间排序一致。
- 空原子输出 ops `[2,0]/[1]/[0,3]`；非空零边输出 `[2,0]/[N+1]/[0,3]`。
  显式容量不足仍 `NeighborOverflowError`，自动增长不截断；COO 不绕过容量检查。
- 转换使用输入设备/current Torch stream；不修改输入，不静默 CPU 回退/改 dtype。
  topology/shift 不可微；MACE 从实时 positions/cell 重算连续位移并计算力/应力。
- HIP 保持 `auto_enabled=False`；`None`/Warp 路径保持。no-PBC/half、skin/selective
  rebuild、target/pair、native geometry backward、compile/opcheck、DD 不扩大。

## 验证与状态

先以 CPU 构造的 MATRIX 验证 COO/shift/空输入/分段及 registry；HCU4 FP32/64
比较 ops/framework/Hook 与 reference 的 `(i,j,shift)` 多重集合、counts/ptr、容量、stream。
对 MPA-0 SHA256 `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`，
采用既有三个 Li/Na CIF 小样本，同 checkpoint/dtype/default dtype/几何/6Å cutoff，
普通 e3nn eager、冻结权重、连续 positions/strain 梯度启用；比较 HIP 与设备内
Torch reference 的完整邻居、E(eV)/F(eV/Å)/stress(eV/Å³)。

模型容差沿用既有 MPA-0 单点基线：FP64 rtol2e-8，E/F atol2e-7、stress2e-9；
FP32 rtol2e-4，E atol2e-3、F2e-4、stress2e-6。不放宽全局容差；集合要求精确一致。
本阶段不运行弛豫；FIRE/FIRE2 工作流、性能测量和优化分别属于后续里程碑。

实施状态：已实现并通过窄 CPU/HCU4 topology 与真实 MPA-0 单点 gate。
CPU ops `88 passed, 10 skipped`，framework `38 passed, 12 skipped, 1 xfailed`；
HCU ops `26 passed`、framework `17 passed`（含5项CPU adapter/Hook oracle）。
MPA-0 两精度各三种快照和 Hook 对照、FP64 原子/应变有限差分通过。证据记录于
`reports/g2-hip-coo-mace-snapshot-m2.md` 与 `artifacts/hip-coo-mace-snapshot-m2/`。
