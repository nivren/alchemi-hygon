# HIP 邻居变胞 FIRE/FIRE2 工作流契约（M3）

M2 收口实现 `3e394d7`、文档 `4136f0f`；M3 已获用户确认，分支
`codex/test-hip-variable-cell-workflow`。本阶段验证现有公共 API 的组合，按实际发现修正接线。

## 目的与来源

将显式 HIP COO 邻居与 Torch reference 的 `FIRE2VariableCell`、`FIREVariableCell`
组合，验证当前晶胞求值、短步数值与 Batch 生命周期；不将优化器后端设为 HIP。
沿用 M1/M2 的 HIP native MATRIX + 设备内 Torch COO 打包、full-PBC/full-list/skin=0，
每次模型求值完整重建 cell metadata/plan；不复用旧 plan 的变胞 metadata。

framework 锁定 `4dfe3723def34df3fadb245981081ccf8c94c257`，ops 锁定
`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`；原生 FIRE2 六项下三角 `3N+6` 与
指定 fork 修正的 FIRE 全矩阵语义按 `TORCH_MACE_MPA0_RELAXATION_CONTRACT.md`、
`UPSTREAM_BUGFIXES.md` 及 MPA-0 M3–M5 报告保持，不改为 ASE filter。

## 行为边界

- FP32/64、普通异构 Batch、全周期晶胞，位置和晶胞同 dtype/device；GPU 验证只用 HCU4。
  邻居为显式 `backend=hip, method=cell_list`，优化器/对齐/stress conversion 为
  显式 Torch reference；None/Warp 和 auto 不改变。
- `NeighborListHook` 在 BEFORE_COMPUTE 重建，模型使用新位置/晶胞；FIRE2 首次计算前
  对齐晶胞，随后保持六项下三角更新；FIRE 使用全 3×3，参数沿用公共类契约。
- 保护有效邻居 `(i,j,shift)` 多重集合与体系归属，不因压缩、cutoff、容量或 N/B/ptr
  改变丢边、截断或沿用旧行。连续力/应力为模型实时 autograd；整数拓扑不可微。
- 退出的结构保留位置/晶胞、输出、速度和私有优化状态；移除/补位后幸存者状态按体系
  重排，arrival 控制参数/速度/归约状态重新初始化，首个 kick 前有当前 E/F/stress。
  拓扑/edges 在实际 BEFORE_COMPUTE 与 refill priming 中按新 Batch 重建。
- 联合收敛使用力范数和 stress 绝对值；测试中的预算退出单列，不能报告为物理收敛。
  拓扑数组的分配/stream/容量错误和未知后端失败沿用 M1/M2。

## 验证

CPU 验证可控解析势的公共短步、inactive、survivor/arrival 和 FusedStage lifecycle；
HCU 将同一流程的邻居换为 HIP，逐次对照设备内 Torch reference 的完整邻居集合。
解析势精确表达 cell stress，验证 E/F/stress priming 和私有状态保持；不把解析势模拟为 MPA-0。

真实 MPA-0 使用既有三个 Li/Na CIF、锁定 checkpoint SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`；
两优化器×两精度，各有限短步，与同设备 Torch reference 邻居分支对照。另在每个当前
HIP 几何上独立重算 reference 邻居和完整 E/F/stress，避免仅比较两条漂移轨迹。
模型/默认 dtype 同精度、普通 e3nn eager、冻结权重但启用位置/应变梯度。

集合/索引/status/ID/整数计数要求精确一致；同几何 E/F/stress 沿用 M2 逐物理量容差。
解析势浮点状态 FP64 rtol/atol2e-12、FP32 3e-5；真实短步位置/晶胞/速度/控制状态
FP64 rtol2e-8/atol2e-9，FP32 rtol2e-4/atol2e-5，功率分支的整数控制状态精确比较。
不通过放宽容差掩盖失败，也不要求两种优化器彼此轨迹相同。

不包含全目录、大 Batch/长弛豫、完整物理收敛、restart、训练力损失、高阶梯度、
compile/opcheck、DD、native dynamics kernel、skin/selective rebuild 或性能准入。
先完成正确性，性能测量与 workspace/query 优化留给后续里程碑。

实施状态：窄 gate 通过，已收口实现 `6b1680a`、文档 `47cf259`，已随 M1–M9 提交链集成到 `develop`。CPU 新测试16 passed/16 skipped，
相关回归48 passed/12 skipped；HCU4 新测试16 passed/16 deselected；真实模型两优化器
×两精度各五步及三结构容量2预算补位通过。真实预算退出不等于物理收敛。
SizeAwareSampler 返回 host 数据，FusedStage.run 显式传入设备 Batch；共享邻居 Hook
登记在 FusedStage BEFORE_COMPUTE。没有新增产品 kernel 或修改优化器。
证据目录 `artifacts/hip-variable-cell-workflow-m3/`，摘要
`reports/g2-hip-variable-cell-workflow-m3.md`。
