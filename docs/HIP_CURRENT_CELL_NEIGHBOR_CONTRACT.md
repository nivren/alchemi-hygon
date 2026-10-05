# HIP 当前晶胞邻居重建契约（里程碑 1）

## 目的与来源

本阶段准入显式 `backend="hip", method="cell_list"` 的逐次当前晶胞快照：每次调用均按
当前 `positions` 与 `cell` 完整重建邻居。晶胞可在调用之间改变；不是对旧 CSR plan 的
变胞复用，也不是 skin/selective rebuild 生命周期。

上游 ops 锁定 SHA `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`：
`nvalchemiops/neighbors/cell_list/kernels.py` 的 atom-centric query factory 与
`nvalchemiops/neighbors/cell_list/launchers.py::batch_query_cell_list`。正确性基线为
本工程 `nvalchemiops.torch_reference_cell_list.neighbor_list`，另以 CPU 周期镜像穷举
校验小夹具，避免仅比较共享 grid metadata 的两条实现。

## 输入、输出和语义

- 目标设备：HIP Torch；FP32/FP64，位置、晶胞同 dtype/device；shape 为 `[N,3]`、
  `[B,3,3]`，Batch 使用连续体系分段和全局原子索引。
- 本阶段变胞证据范围为全周期 PBC `[B,3]`、full-list、MATRIX、eager、`skin=0`。
  既有固定晶胞 mixed-PBC 支持保持原契约，变胞 mixed-PBC 不在新增证据中。
- 行晶胞约定 `r=s@H`；边向量为 `r_j-r_i+image_shift@H`。shift 为整数 `[N,K,3]`。
  保留同一原子对的多个不同镜像，排除零镜像自身；cutoff 使用严格 `<`，单位跟随输入。
- 输出为 MATRIX `[N,K]`、每行 counts `[N]`、整数 shifts `[N,K,3]`。
  每行有效邻居连续存于前 counts 项；padding 使用不与原子索引冲突的 fill value。
  判定正确性使用含重复项的 `(i,j,shift)` 多重集合，不能只比较 `(i,j)`。
- 每次调用重算逆晶胞、网格尺寸、搜索半径、体系 cell offsets，并新建 checked build
  plan 和 workspace。位置/晶胞/PBC/Batch 输入不被修改；返回结果独立拥有存储。
- `max_neighbors=None` 延续有限上界的 doubling capacity；显式容量不足明确报错，
  不截断。空原子输入和非法参数延续既有公共 executor 行为。
- native build/query/materialization 使用当前 PyTorch stream；不允许隐式设备切换、
  CPU 拓扑回退或 dtype 改变。现有 metadata host readback 与错误同步尚未优化。
- 拓扑/整数 shift 不可微。此阶段仅登记梯度 0 的拓扑，MACE 连续几何及力/应力另验收。

## 能力边界

registry 的 `variable_cell` feature 仅表示**公共 stateless executor 在调用之间允许
晶胞改变并完整重建**。`NativeHipNeighborPipelineABI.supports_variable_cell=False`
继续限制低层复用 plan：其 inverse/grid/offset metadata 必须保持一致，结构变化要重新
初始化。公共 executor 对每个快照创建新 plan，因此不依赖旧 plan 更新晶胞。

`None`/Warp 默认语义保持；HIP 不进入 `auto`。COO、half、no-PBC、skin/selective
rebuild、target/pair、compile/opcheck、DomainParallel 与 native geometry backward
均不在本阶段。MACE 及 FIRE/FIRE2 工作流将在后续里程碑分别准入。

## 验证与状态

最小回归：FP32/64 的异构 Batch 收缩、膨胀、剪切、周期 wrap、多镜像；同一张量的
连续 in-place 更新；输出不 alias 输入/先前结果；显式溢出、自动扩容和恢复；默认及
非默认 stream。CPU 以独立镜像穷举检验 reference；HIP 在 HCU4 逐快照对照 reference。

实施状态：已实现并通过窄 CPU/HCU4 gate（FP32/64）。CPU ops `87 passed, 6 skipped`，
framework `33 passed, 4 skipped, 1 xfailed`；HCU ops `22 passed`、framework `4 passed`，
既有 46×32 固定晶胞 gate 通过。不能扩大为 MACE/FIRE 工作流支持。结果记录于
`reports/g2-hip-current-cell-neighbor-m1.md`，原始日志放在
`artifacts/hip-current-cell-neighbor-m1/`。
