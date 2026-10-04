# 上游缺陷、实现补齐与逻辑修正记录

更新日期：2026-10-04。

本文件记录移植中发现的上游计算逻辑、状态管理和接口行为缺陷，以及本项目如何修正。
每项记录须能回答：原来哪里错、为什么判定有问题、修改了什么、如何验证、同步上游时如何处理。
问题判断以记录中的上游 SHA 为准，不能据此断言官方最新版本仍有同一问题。

## 1. 文档与记录规则

- [UPSTREAM.md](UPSTREAM.md)：维护来源、锁定和全部本地补丁的索引，包含后端适配。
- 本文件：维护上游缺陷的原因、修正决策和跟踪状态。
- 源码注释：在修正位置解释原因，使用可检索的问题标记，与本文件对应。
- `reports/` 与回归测试：保存运行证据、具体输入、数值结果和未验证范围。

新增 Torch/HIP 后端、依赖可选导入等平台适配，按实际来源登记到 UPSTREAM.md。
若错误产生于本项目新增代码，标为本地实现问题；若上游语义尚不明确，标为待确认。
只有锁定源码和可复核依据支持时，才登记为上游缺陷。已复核 LP-001–068 索引、
截至 `7a90fcf` 的产品源码相对锁定上游的差异，以及相关行为修正的源码与报告。
本次审查覆盖已有改动的归因，不是全库缺陷搜索，也未核验官方当前分支。
历史运行结果引用对应报告，本次文档审查没有重跑测试。

记录使用稳定编号 `UBF-xxx`；编号不随合并或上游修复重排。
“本地已修正”“本地验证范围”“已提交上游”“官方已修复”分别记录。
issue/PR、官方修复 SHA 和核验日期缺失时写明，不由本地测试通过推断官方状态。
同步上游时按问题逐项判断补丁可否移除；移除后保留记录和回归测试。

## 2. 当前索引

| 编号 | 问题 | 包 | 源码标记 | LP 记录 | 当前状态 |
|---|---|---|---|---|---|
| [UBF-001](#ubf-001yoshida-参数未传递) | Yoshida 参数被接受却未使用 | framework | `NHC-YOSHIDA` | LP-065 | 已提交 `b4050ec`；Torch 验证通过，Warp 未执行 |
| [UBF-002](#ubf-002补位体系的-qndof-计数错误) | 异构补位体系 Q/ndof 计算错误 | framework | `NHC-REFILL` | LP-066 | 已提交 `b4050ec`；scalar refill 验证通过 |
| [UBF-003](#ubf-003非活动体系的私有温控链仍被推进) | mask 未保护私有温控链 | framework | `NHC-MASK` | LP-067 | 已提交 `b4050ec`；Torch 生命周期回归通过 |
| [UBF-004](#ubf-004链坐标-eta-的时间推进多一倍) | eta drift 累计多推进一倍时间 | ops | `NHC-ETA` | LP-068 | 已提交 `b4050ec`；Torch 数值验证通过，Warp 未执行 |
| [UBF-005](#ubf-005混合状态-batch-的输出未随坐标恢复) | 已退出体系输出与返回坐标不一致 | framework | 现有原因注释 | LP-001 | 已提交 `b5fd0ae`；专用回归存在 |
| [UBF-006](#ubf-006hostmemory-快照与活跃数据共享存储) | CPU 轨迹帧被后续更新改写 | framework | 现有原因注释 | LP-008 | 已提交 `b5fd0ae`；历史 CPU/HCU 验证 |
| [UBF-007](#ubf-007mace-本地-checkpoint-被交给下载器) | 声明支持本地路径却按 URL 处理 | framework | 现有原因注释 | LP-005/069 | 已提交 `3f5b544`；历史回归和模型执行证据 |
| [UBF-008](#ubf-008uniform-storage-多属性复制使用不同目标槽位) | 多属性 put 复用已变更的占用 mask | framework | 现有原因注释 | LP-004/070 | 已提交 `3f89f6b`；Torch 双属性回归 |
| [UBF-009](#ubf-009行晶格的-stress-to-cell-force-乘法顺序错误) | 非正交行晶格的晶胞力公式错误 | ops/framework | `CELL-STRESS-ROW` | LP-072/073 | 实现提交 `a76107e`；CPU/HCU reference 通过，Warp 未执行 |
| [UBF-010](#ubf-010晶胞对齐的左手坐标与非连续写回错误) | 左手 transform 改变分数坐标；非连续坐标未写回 | ops/framework | `UBF-010` | LP-074/075 | CPU/HCU4 reference 通过，bridge stub 通过，Warp 数值未执行 |
| [UBF-011](#ubf-011fire2-仿射运动使用了列晶格公式) | 非正交行晶格的仿射运动不保持分数坐标 | ops | `UBF-011` | LP-076 | fork 修复移植；CPU/HCU4 reference 通过，Warp 未执行 |
| [UBF-012](#ubf-012完整晶胞-fire-缺少应力驱动与联合控制) | 零初始化晶胞速度未受应力驱动；缺少九项联合控制 | framework | `UBF-012` | LP-078 | fork 修复移植；CPU/HCU4 解析势与 CPU MPA-0 短步通过 |
| [UBF-013](#ubf-013fire-联合控制更新与混合存在竞态) | 参数写入与速度混合在同一并行 kernel 中竞态 | ops | `UBF-013` | LP-077 | fork 的分离 kernel 移植；reference 通过，Warp 未执行 |
| [UBF-014](#ubf-014aligncellhook-未保护体系子集及坐标系一致性) | 忽略周期体系掩码；cached 力/速度/应力未旋转 | framework | `UBF-014` | LP-080 | 本地修正；CPU 公共 Hook/MPA-0 旋转协变通过 |

NHC 条目的共同来源：

- framework：`4dfe3723def34df3fadb245981081ccf8c94c257`。
- ops：`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。
- 本地分支：`codex/feat-fixed-cell-nhc`，起点 `7a90fcf`；修正提交 `b4050ec`，随本轮交付到本地 `develop`，未推送。
- 证据报告：[g2-torch-nvt-nhc.md](../reports/g2-torch-nvt-nhc.md)。
- 上游反馈：本次未提交 issue/PR；此处没有已关联的官方修复记录，未重新核验当前官方分支。

## UBF-001：Yoshida 参数未传递

**位置**：`NVTNoseHoover.pre_update/post_update` 与 framework 的
`nhc_chain_update` wrapper。

**原问题**：构造参数 `yoshida_order=3/5` 存入积分器，但调用链没有把它传给 ops
的 `nloops`；ops 因而使用默认 `nloops=1`。公开参数与实际执行不一致。

**修正**：pre/post 两处显式传 `nloops=self.yoshida_order`；Torch 与 Warp dispatcher
保留同一 ABI。DomainParallel adapter 同步接收/转发参数，但其 Torch 路径仍明确拒绝。

**影响**：`backend=None` 的 Warp 选择不变；默认类原先实际执行的一段子步变为公开参数
指定的三段或五段，轨迹可能改变。低层 wrapper 的默认 `nloops=1` 保留。

**依据与验证**：`test_yoshida_order_forwarded_to_actual_executor` 检查实际 Torch
executor 收到的参数；链可逆性测试覆盖 nloops=1/3/5。没有 Warp/NVIDIA 执行证据，
也没有 DomainParallel 数值验收。

**上游处理建议**：候选修复为参数传递和对应回归；本地后端 registry、DD capability
限制须单独审查，不能把整个适配 diff 视作同一上游缺陷修复。

## UBF-002：补位体系的 Q/ndof 计数错误

**位置**：`NVTNoseHoover._make_new_state`。

**原问题**：用 dummy masses/index 构造 Q，并用平均体系大小构造 ndof。
新到达体系原子数不同时，这与上游采用的 `ndof=3N`、
`Q[:,0]=ndof*kT*tau²` 不一致。

**修正**：保留 BaseDynamics 选择旧状态和追加新状态的机制；从 updated Batch 尾部
新体系的实际 `batch_ptr/batch_idx` 计算 ndof 和 Q。普通 Batch 保留逐体系参数。
scalar 构造参数可用于补位；vector 构造参数没有到达体系来源映射时明确报错。

**影响**：保留体系的链状态不重置；新体系取得正确的质量和自由度。
vector arrivals 的参数映射尚未实现，这是当前接口边界，不能算作已经解决的完整补位能力。

**依据与验证**：`test_initial_and_refill_chain_masses_use_actual_counts` 覆盖初始
2/4 原子、保留4原子体系后追加7/3原子体系；
`test_actual_sampler_refill_preserves_ids_and_chain_state` 使用真实 SizeAwareSampler，
检查体系 ID、保留链状态和新 ndof/Q。

**上游处理建议**：实际计数修正及异构 refill 回归可提交；逐体系到达参数的来源与映射
应与上游接口设计单独讨论，不能用复用第一个旧体系参数掩盖缺口。

## UBF-003：非活动体系的私有温控链仍被推进

**位置**：`NVTNoseHoover` 的 masked update、pre/post 和 step 状态管理；关联
BaseDynamics 的 Batch 字段恢复机制。

**原问题**：masked 操作恢复非活动体系的原子字段，却没有恢复 NHC 私有链状态。
当前阶段不拥有的体系或已退出体系，仍会改变 eta/eta_dot 等状态；后续进入该阶段时
使用了不应产生的链历史。

**修正**：在既有 masked 机制外保护非活动体系的六个被修改的链/工作张量。
普通 step 在 pre/post 内完成恢复，让 AFTER_* hooks 观察到正确状态。
FusedStage 继续使用其显式阶段 mask，不按普通积分器 exit_status 代替阶段归属。

**影响**：活动体系照常积分；非活动温控链保持不变。未重写 BaseDynamics 的 Hook、
Batch 或 FusedStage 管线。该修正接入两种后端，但运行证据来自 Torch 路径。

**依据与验证**：`test_inactive_graph_chain_does_not_advance` 覆盖四种入口；
`test_fusedstage_matches_separate_graph_runs_and_migrates` 对照独立轨迹和迁移；
`test_after_step_hook_observes_restored_inactive_chain` 检查观察者时序。
HCU 短 MACE 第4步还检查非活动体系 eta 不变。

**上游处理建议**：NHC 私有状态保护是修复候选；其他有私有状态的积分器是否存在相同
问题须分别审查，本条不据此宣称已经修复所有 masked dynamics。

## UBF-004：链坐标 eta 的时间推进多一倍

**位置**：ops `nose_hoover.py` 中单体系和批量链传播函数；本地 Torch functional
reference 使用相同修正。

**原问题**：单次 thermostat operator 推进的是 `dt/2`，但两次对称 eta drift
各使用 `dt/2`，累计推进 `dt`。按连续方程 `d eta/dt=eta_dot`，小步极限应为
`delta eta/dt → eta_dot/2`，原实现得到一倍 eta_dot。

**修正**：两次 eta drift 各用 `dt/4`。velocity/eta_dot 更新公式保留。
源码同时修正锁定 Warp 的两个传播函数；Torch reference 使用同样的 dt/4。

**影响**：eta 及其扩展能量改变；单独修改 eta drift 不改变 v/eta_dot 演化，
因为该链传播公式不通过 eta 反馈速度。公开 timestep 和后端默认选择不变。

**依据与验证**：独立单链标量公式、小步导数测试
`test_thermostat_half_step_eta_derivative`、谐势扩展能量收敛测试。
同一物理时长，dt 从0.02减半为0.01时，两个体系扩展能量峰峰误差比约0.253/0.250，
符合二阶时间步误差；修正前约7.37/11.47 eV且减半步长不改善。输入与数值见证据报告。

**验证边界与上游建议**：Torch/CPU 与短 HCU 已验证；本机没有 Warp，三项 Warp CPU
对照测试跳过。可提交公式修正及独立物理回归作为候选，但需补齐 Warp/NVIDIA 的
执行回归后再声明该后端验证完成。

## UBF-005：混合状态 Batch 的输出未随坐标恢复

**来源与位置**：锁定 framework SHA，`BaseDynamics._mutable_fields/step`。
已提交修正 `b5fd0ae`，对应 LP-001。

**原问题**：混合 status Batch 在共享 model forward 中仍经过临时 pre-update。
step 恢复已退出体系的 positions/velocities/cell，却留下临时坐标对应的 forces/energy/stress。
返回 Batch 的输入与输出因此可能不一致。

**修正与影响**：在恢复字段中增加 forces、energy、stress；已退出体系保留原有输出。
这是状态/输出一致性修正，同时影响 Warp 默认路径，没有改势函数计算公式。
恢复前输出必须已与被冻结坐标一致；此补丁不承担重新计算任意陈旧输入输出的功能。

**证据**：源码相对锁定上游的 diff，
`test_inactive_dynamics_outputs.py::test_step_preserves_outputs_for_graduated_graph`；
历史行为验证见 [g2-upstream-dynamics-reference.md](../reports/g2-upstream-dynamics-reference.md)。
新增字段包含 stress，但专用测试直接断言的是 positions/forces/energy，不能扩大为 stress 数值验收。

**上游状态**：适合作为 mixed-status 一致性候选；本次未提交或核验官方 issue/PR。

## UBF-006：HostMemory 快照与活跃数据共享存储

**来源与位置**：锁定 framework SHA，`HostMemory.write`；修正提交 `b5fd0ae`，LP-008。

**原问题**：写入只做 `data.to("cpu")`。输入已在 CPU 时，设备迁移可以保留原张量；
live Batch 后续原地更新会改写 sink 内此前保存的帧，违背轨迹快照语义。

**修正与影响**：设备迁移后 `clone()`，确保标准字段和动态字段均由 sink 独立持有。
影响 CPU 快照存储和复制成本，不修改积分公式；GPUBuffer/ZarrData 未在此修正范围。

**证据**：[g2-heterogeneous-trajectory-reference.md](../reports/g2-heterogeneous-trajectory-reference.md)：
异构3/5原子、两帧、体系 ID、动态 trajectory_step 的 CPU/HCU 各1项通过；
历史上游 HostMemory/Drain/DataSink 子集17项通过。测试为
`test_heterogeneous_trajectory_reference.py::test_snapshot_hook_preserves_heterogeneous_frames`。
本次未重跑，也不据此宣称 checkpoint/restart 已完成。

**上游状态**：独立快照所有权修复候选；本次未提交或核验官方 issue/PR。

## UBF-007：MACE 本地 checkpoint 被交给下载器

**来源与位置**：锁定 framework SHA，`MACEWrapper.from_checkpoint`；提交 `3f5b544`，
原 LP-005 中未明确列出这一行为修正，本次补登记 LP-069。

**原问题**：公共签名/docstring 支持本地 `Path | str`，实现却无条件调用 MACE foundation
下载器；该依赖将任意字符串按 URL 处理，Path 可出现 decode 错误，字符串路径出现
unknown url type。本地文件支持的承诺没有兑现，属于接口实现缺陷。

**修正与影响**：先 `Path(...).expanduser().is_file()`；文件存在则本地加载，其余名称
仍交给原下载器。保持模型、dtype、device 放置路径；没有修改 MACE 能量/力公式。

**证据**：[g1-mace-wrapper-batch.md](../reports/g1-mace-wrapper-batch.md)：
本地文件及 load-device 相关回归历史3项通过，正式 wrapper 后续有真实缓存模型 CPU/HCU
执行证据。专用测试为 `test_mace.py::TestFromCheckpointErrors.test_from_checkpoint_uses_existing_local_path`；本次未重跑。

**上游状态**：可提交本地文件分支及下载器不应被调用的回归；本次未核验官方当前实现。

## UBF-008：Uniform storage 多属性复制使用不同目标槽位

**来源与位置**：锁定 framework SHA，`UniformLevelStorage.put`；提交 `3f89f6b`，
原 LP-004 同时包括后端隔离，本次单列 LP-070 的行为修正。

**原问题**：上游循环多个共同属性时向每次 `put_masked_per_system` 传同一 dest_mask。
底层复制 kernel 会把写入槽位置为占用；下一属性于是选择不同槽位，或无容量可复制。
同一体系的多个字段无法保持同一目标行，这不是海光后端特有问题。

**修正与影响**：保存初始目标 mask；首属性提交占用结果，后续属性用初始快照和实际
copied_mask 重放同一复制集合。该逻辑位于共同 LevelStorage 层，也作用于显式 Warp backend。

**证据**：[g1-storage-contract.md](../reports/g1-storage-contract.md)：历史 storage backend
3项、level storage96项通过；专用 uniform 双属性回归为
`test_storage_backend.py::test_uniform_multi_attribute_put_reuses_slots_and_mask`。
另有 segmented 多属性 offset/mask 保持及显式 Warp 合同对照测试。

**边界**：同一提交也让 segmented put 使用原 pointer 快照及实际复制集合，这一项记录为
多属性合同加强；本次没有证明锁定 Warp segmented kernel 在相同输入上必然错位。
不能把 uniform 的已确认缺陷直接推广到全部 segmented 操作。

**上游状态**：uniform 多属性映射修复为上游候选；Warp 执行证据与 Torch 对照须分别核验。

## UBF-009：行晶格的 stress-to-cell-force 乘法顺序错误

**来源**：锁定 ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` 的
`dynamics/utils/cell_filter.py` stress kernel/接口，以及锁定 framework
`4dfe3723def34df3fadb245981081ccf8c94c257` 的 `dynamics/_ops/npt_nph.py` 转换说明。

**问题与依据**：原 kernel 使用 `-V sigma H^-T`；行晶格 `r=s@H`、
`H'=H@(I+epsilon)` 要求固定分数坐标的负能量梯度为 `-V H^-T sigma`。
独立非正交周期谐键 FP64 有限差分复现旧公式偏差。部分说明的 compression-positive/
upper-triangular 与公共拉伸为正、实际下三角投影不一致。

**修正与影响**：Torch reference 和本地 legacy Warp kernel 均更正公式；转换说明同步
更正，源码标记 `CELL-STRESS-ROW / UBF-009`。保留原位置参数、keep_aligned 默认值和
None/warp 默认选择；legacy 变胞晶胞力/轨迹可能变化。未移植 fork 的晶胞更新、
FIRE/FIRE2、归一化、仿射映射或 DomainParallel 修复。reference 后端、fake/autograd
和依赖隔离是平台适配，与这个上游缺陷分开归因。

**参考修复**：用户指定 fork
[ops c4c3bb56b6c6](http://10.208.16.64:3000/wangleping/nvalchemi-toolkit-ops/commit/c4c3bb56b6c6ae0d5e85b4e1ed14bd4bd4748a2a)
及 [framework 4bf8bc8084a2](http://10.208.16.64:3000/wangleping/nvalchemi-toolkit/commit/4bf8bc8084a2f588c332e0bf090978ba4ba76c4b)。
不更改上游锁定，不整体同步 develop；fork 已修正不代表 NVIDIA 官方修复状态。

**本地状态**：`codex/feat-cell-stress-force` 实现与测试提交 `a76107e`，未合并、未推送。
CPU 一二阶/有限差分、公共接线、小模型 stress loss 参数反向、真实周期 MACE 和
HCU4 窄 smoke 见 [报告](../reports/g2-torch-cell-stress-force.md)。Warp 缺失，legacy
kernel 未执行；不能把 Torch 通过当作 Warp/NVIDIA parity。本轮未向 NVIDIA 提交
issue/PR，未核验其当前分支。补丁索引 LP-072/073。

## UBF-010：晶胞对齐的左手坐标与非连续写回错误

**来源**：锁定 ops `26dbceb` 的 `_align_cell_kernel` 与锁定 framework `4dfe372`
的 `_align_cell_op`。指定 fork ops `eadc3ee` 已修订 lower-triangular 文案，但仍存在
负 determinant 分支的 H 取负；未将本次修正误记为 fork 已修复。

**问题与依据**：原 kernel 以取负后的 H 计算 T，而原子坐标未取负，因此左手输入
从 `r=s@H` 变为 `s'=-s`。解析 CPU 测试复现分数坐标翻转，与正确变换独立对照。
framework 使用 `positions.contiguous()` 临时副本却仅写回 cell，导致 strided 原子
未旋转；bridge stub 的回归验证真实 custom-op 调用的 buffer 写回路径。

**修正**：保持正对角下三角 L，以原始 H 计算 `T=inverse(L)@H`，原子行向量
`r'=r@T.T`；左手输入是反射，保持分数坐标和周期距离。bridge 同时写回 positions
和 cell。源码均加 UBF-010 注释。文案采用 fork 下三角修订，不更改布局/默认后端。

**证据与边界**：CPU FP32/64 ASE 标准晶胞、metric/周期距离、旧公式复现；HCU4
六组 CPU parity。详见 [M2 报告](../reports/g2-cell-tools-m2.md)。Warp 不可用，
kernel 本身未执行，stub 不是 Warp 数值对照；未对 NVIDIA 当前分支或外部 PR 作声明。
非法输入检查和 alias/梯度失败是 reference 契约加强，不当作上游已确认缺陷。

## UBF-011：FIRE2 仿射运动使用了列晶格公式

**来源**：锁定 ops `26dbceb` 的 `_fire2_coord_cell_*` 两个 kernel；指定 fork
ops `c4c3bb5` 已修复。行晶格 `r=s@H` 要求 `r_new=r@H^-1@H_new`；原代码对
列坐标使用 `H_new@H^-1`，非正交晶胞不能保持分数坐标。

**修正**：原 Warp kernel 使用 `H_new.T@H^-T`；reference 使用增量
`dr=r@H^-1@dH+dt*v`。分支、cell-force 原子数缩放、位移限制及 dt 缩放保持来源语义。
CPU 独立 NumPy、旧公式反例与纯晶胞分数坐标守恒通过；HCU4 FP32/64 单步 parity
通过。实际 Warp/NVIDIA 未运行；不宣称其余 NPT/NPH 仿射实现已修正。

## UBF-012：完整晶胞 FIRE 缺少应力驱动与联合控制

**来源**：锁定 framework `4dfe372` 的 `FIREVariableCell` 声明需要 stress，
但晶胞速度零初始化后没有完整的应力 half-kick，原子 FIRE 控制也未包含晶胞。
采用指定 fork framework `b12b54c` 的修复，而非新增 ASE filter 算法。

**修正**：pre/post 各半步施加 `F_H`，晶胞优化质量为1；保留原子质量；所有原子
和三行完整 Hdot/FH 参与同一次功率/控制/混合，位移限制同时考虑实际原子和 cell-row。
默认原子 kick/mixing 仍走 Warp，reference 独立可用。CPU 全九项独立单步 oracle、
解析势160步能量下降、CPU MPA-0 三步及 HCU4 解析势25步通过。没有实际 Warp 或
目标模型 HCU、完整收敛与补位验证；保留 DD 元数据不代表 DD 已适配。

## UBF-013：FIRE 联合控制更新与混合存在竞态

**来源**：锁定 ops 的 batch_idx update-only kernel 一部分线程写 alpha/dt，
其他线程同一 kernel 读这些控制值混合速度；没有跨线程同步保证读取新值。
采用指定 fork ops `eadc3ee` 的先按体系更新参数、再按 DOF 混合的分离 launch。

**修正范围**：`fire_update` 的 single/batch_idx 且不启用 downhill-check 路径，
正是完整晶胞 FIRE 的联合控制入口。`fire_step`、PTR、downhill-check 路径保留，
未据此宣称整个 Warp FIRE 已无竞态。Torch full-cell reference 同样先更新控制再混合。
CPU/HCU reference 与阈值 oracle 通过；本机 Warp 不可用，kernel 并行运行未验证。

## UBF-014：AlignCellHook 未保护体系子集及坐标系一致性

**来源**：锁定 framework 的 `AlignCellHook`；指定 fork `b12b54c` 已修复周期/active
子集写回，但尚未修复 cached F/v/sigma 的坐标系一致性。
原 Hook 只以 periodic_mask 判断是否返回，随后对整个 Batch 对齐；且在模型已评估后
只旋转 r/H，不旋转 cached F/v/sigma，可能用不同坐标系的力和应力推进第一步。

**修正**：采用 fork 的周期/active 子集保护，只对需对齐的体系处理；本地补充同一正交/反射 T 对
列向量 F/v 和 tensor stress 协变旋转。已对齐体系跳过；已有晶胞运动时需要再次
对齐则明确报错，避免对六项状态作未经定义的动态重参数化。新增 keyword-only
backend/冻结 selection 是平台适配，错误检查为合同加强，分别区别于 bug 修复。
CPU Hook 子集/协变/重复调用和 MPA-0 FP32/64 旋转前后 E/F/stress 对照通过；
不能扩展为任意运行中重对齐、full-cell FIRE 每步 Hook 或 Warp 数值已验证。

以上 M3 证据汇总见 [报告](../reports/g2-variable-cell-fire-m3.md) 和
[契约](TORCH_VARIABLE_CELL_FIRE_CONTRACT.md)；来源 SHA 不变，未验证 NVIDIA 当前分支。

## 3. 上游没有的实现与本地扩展

### EXT-001：Langevin 最小积分器续算

- **锁定上游**：`NVTLangevin` 和 BaseDynamics 没有 state_dict/load_state_dict 续算实现。
- **本地新增**：提交 `267bba2`；保存版本、step_count、random_seed、dt/kT/friction，
  用 `random_seed + step_count` 延续随机序列。新增两项公开方法，补登记 LP-071。
- **归因**：这是本地新增能力，不是修复已存在的上游 checkpoint 实现，也没有证据表明
  锁定上游承诺了这套接口。Batch/model/sampler/sink 仍须调用者保存。
- **历史证据**：[g2-torch-nvt-langevin-restart.md](../reports/g2-torch-nvt-langevin-restart.md)：
  3+2对照连续5步，CPU/HCU0各2项通过；普通固定晶胞 Batch 范围。
- **限制**：不是通用 workflow checkpoint、inflight 或跨设备随机逐位一致。

### EXT-002：无 Warp 执行和可替换后端

LP-003–007、011–012、014、017–020 等增加 lazy import、Torch storage/reference、
capability registry、单次 selection 与 executor binding。上游已有相应 Warp 功能，
缺少的是本项目目标平台执行路径和后端边界，应归为适配/扩展。
尤其 B1 的 `Implementation.executor` 真正接上执行器，补齐的是**本地 registry**，
该 registry 并非锁定 NVIDIA 上游拥有的未完成实现。

NVE/FIRE/FIRE2、Langevin BAOAB、邻居 full/half/PBC/cell-list、capacity/rebuild、
Torch geometry 梯度和 HIP 邻居组合，均是在保留上游契约下新增后端。
已提交产品 ops 的原 dynamics/interaction/math/neighbor 算法文件没有因这些任务改公式；
本轮 NHC 修改原 ops eta 公式是另列的 UBF-004。测试/benchmark/AOT/planner 也分别归为
验证、分发或规划，不计成上游 bug 修复。

## 4. 既往修正中属于本地实现的问题

| 条目 | 已修正内容 | 归因与证据 |
|---|---|---|
| LOCAL-001 / LP-059 | composite workspace size-query 把 int32 candidate storage 当作 int64 key storage，改用正确 dtype/长度临时 buffers | 本地新增 HIP materializer；[runtime wrapper 报告](../reports/g2-native-hip-neighbor-runtime-wrapper.md) |
| LOCAL-002 / LP-061 | Torch cell-list allocator 与 build 的 max_nbins 不一致，统一8192 | `torch_reference_cell_list.py` 为本地新增文件；[wall-clock 报告](../reports/g2-framework-hip-neighbor-wallclock.md) |
| LOCAL-003 / LP-061 | HIP executor 在 query 前用未初始化 candidate_counts 校验，先 zero_ | `_hip_neighbor_executor.py` 为本地新增文件；同上报告 |

以上均须保留回归，但不能据其报告中的“修复”二字认定 NVIDIA 上游存在这些 bug。
LP-061 原候选说明将 Torch reference 修复泛称上游候选，本次按文件来源纠正归因。

### 本次审查发现、尚未修改的本地风险

`LennardJonesModelWrapper.__init__` 将 `backend=None` 作为 legacy Warp，但
`distribution_spec` 使用 `self.backend != "warp"` 拒绝条件，所以 None 也会被拒绝。
这是本地后端接线新增的条件，与默认语义不一致；不是此前修好的上游 bug。
本次仅静态确认，未执行 DomainParallel；后续应在独立适配任务中修正并补回归。
当前 DomainParallel 仍未验收，本次不改变其支持状态。

## 5. 审阅范围与追踪

- 复核 LP-001–068（NHC 前LP-001–063），并补登记历史行为变化 LP-069–071。
- 比较锁定 framework `nvalchemi` 与 HEAD `packages/framework/nvalchemi`：43个变更/新增文件。
- 比较锁定 ops `nvalchemiops` 与 HEAD `packages/ops/nvalchemiops`：原有文件变更集中在
  各包 __init__ 的导入隔离；Torch/HIP/registry 为新增文件。NHC 实现提交另列。
- 针对 storage、BaseDynamics、HostMemory、MACE、Langevin、模型和邻居接线的实质变化
  核对来源、历史提交、当前源码与报告。原源码没有的功能与原源码错误分开记录。
- 本次只修改文档，没有新增/执行测试、跑 HCU、提交 issue/PR 或同步官方上游。
- 官方修复状态未在线复核；历史结果不冒充本次新实测。

## 6. 后续条目模板

新增条目至少填写：

1. 稳定编号、发现日期、问题分类（上游缺陷／待确认／本地实现问题）。
2. 仓库、上游 SHA、文件和符号；LP 编号、源码注释标记。
3. 原行为、触发输入、预期行为及判定依据。
4. 修正方式、默认路径/API/数值/状态影响、可替代方案及选择原因。
5. 回归测试、报告、CPU/HCU/NVIDIA 分别的实际结果与缺口。
6. 本地提交或工作树状态、上游 issue/PR、修复 SHA、核验日期。
7. 同步上游时保留／移除补丁的条件和尚待解决的问题。
