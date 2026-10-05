# M3 原生变胞 FIRE/FIRE2 契约

来源：NVIDIA framework `4dfe372`、ops `26dbceb`；指定 fork framework `b12b54c`、
ops `c4c3bb5` / `eadc3ee`。只摘取原生算法与已确认修复，不同步新版 schema/依赖。
沿用 M1/M2 契约和显式 backend；默认 None/warp 保留 Warp 选择。

## 数据、后端与边界

原子 r/v/F `[N,3]`，行晶胞 H/Hdot/FH `[B,3,3]`，质量 `[N]` 或 `[N,1]`；
FP32/64 同设备同 dtype，idx/计数 int32/64，idx 连续且按体系非递减。
DT/alpha/计数每体系保存；不静默转换输出、回退 CPU 或开启优化轨迹梯度。
优化步 forward-only eager，梯度等级0，输入 requires_grad 明确拒绝；framework
沿用上游 dynamics 的显式状态 detach 边界，训练与 stress 转换的原有梯度不改。
未提供的 scratch 在设备内分配，提供者须正确 shape/dtype/device；输入/输出缓冲
不允许共享 storage，原地更新保持对象身份；非连续非重叠 tensor 可用。
当前 stream，检查可能同步主机，归约不承诺跨设备逐位一致。
奇异/非有限 H、非法索引/控制参数/容量/alias、未知后端均报错。

operation fire2 的新实现只声明 variable_cell、entrypoint fire2_step_coord_cell、
梯度0；现有 fixed_cell 实现仍独立。完整 FIRE 以独立 fire_cell operation 声明
pre/post 两阶段 executor，避免把 NPH/VV 变胞能力当作已移植。framework 生命周期中
冻结 optimizer/stress/align selection，调用 generic executor 不再次解析。

## FIRE2：六项下三角

上游 cell_force 先除以 `atoms_per_system * cell_force_scale`（默认1），原子力不除质量。
六项 pack `[00,10,20]` / `[11,21,22]`，扩展布局为每体系原子后接两个 vec3；
扩展 idx 从 ext_ptr 块长构造，不能用 append 布局 extend_batch_idx。
保持低层 ops 的 metadata/scratch/compute_reductions/cell_force_scale ABI；framework
保持原位置参数和默认控制值，不新增用户参数化。

`v_kick=v+dt_old*F` 的联合 vf/vv/ff 决定分支。P>0 增计数，计数>delaystep 才
增长 dt/衰减 alpha；P<=0 重置计数、alpha0，dt 缩至 tmin。使用新 alpha 与旧 dt
做融合 mixing；运动使用更新后的 dt，downhill factor=1、uphill=-0.5。
`dH=factor*dt*Hdot`，`dr=r@inverse(H)@dH+factor*dt*v`；每体系真实原子位移
最大范数决定共同缩放，r/H/dt 乘同一 scale，上坡后原子与晶胞速度均归零。
此行晶格仿射公式采用 fork 修复；上游错误旧公式作为回归反例记录。
拒绝未对齐 cell；对齐通过已有 AlignCellHook/align_cell 完成，不能悄悄丢掉上三角。
上游 N=0 整个调用不推进，仅将提供的归约 scratch 清零；非空 Batch 的零原子体系
按来源明确拒绝，不将 M2 packing 支持扩张为优化器支持。

## FIRE：完整3×3

采用指定 fork：pre 原子 v+=dt/2*F/m，Hdot+=dt/2*FH（晶胞优化质量1）。
`dH=dt*Hdot`，`dr=r@inverse(H)@dH+dt*v`，原子位移与 cell 每行位移最大值
共同决定缩放，仅缩放 r/H 的 drift，不缩 dt 或速度。
post 用新 F/stress 完成半 kick，然后所有原子与三行晶胞打包 `[N+3B,3]`，
联合 vf/vv/ff 更新控制且只每体系更新一次，再同步混合原子与晶胞速度。
P>0 计数>=n_min 时 dt 增长/alpha 衰减；P<=0 dt 缩至 dt_min、alpha_reset，
原子与全部晶胞速度归零。保留 dt_min/max 的来源语义，无自定义 cell mass/filter。
空 Batch 无操作；全矩阵不要求对齐。DomainParallel 只保留原有接口，不准入 reference。

## 验收与尚未完成

AlignCellHook 沿用原 stage/frequency，只处理需要对齐的周期/active 体系，已对齐跳过。
缓存 F/v 按列向量 T、stress 按 T sigma T.T 转换；fork 已保护子集，本地补齐缓存旋转。
新 keyword-only backend 和继承冻结 selection 是平台接线。已有非零晶胞速度而需要
再次对齐时拒绝，不能把预处理当作 full-cell FIRE 每步重对齐工具。

CPU FP64 独立 NumPy 逐体系单步 oracle；FP32 容差独立；覆盖纯晶胞应力、零/正/负
联合功率、延迟阈值、异构体系、上坡回退、maxstep、分数坐标守恒、完整九项混合、
质量差异、预分配/外部归约、错误/空输入、公共 pre/model/post 与 selection/import。
FP64 单步 rtol/atol=2e-12，FP32=3e-5；计数/索引精确。短解析周期势能收敛作为
优化器证据；MACE-MPA-0 仅作小步 CPU 接线，真实收敛与 HCU模型属于 M4/M5。
限时 HCU4 用同一合成异构状态比较 CPU，不可用则标 pending。
不验收训练梯度、compile、NPT/NPH、ASE filters、checkpoint、退出补位、skin 缓存、
多卡或生产性能；后续 M4 处理工作流生命周期，M5 处理真实目标 HCU。

实际证据见 [M3 报告](../reports/g2-variable-cell-fire-m3.md)：CPU ops39项/公共18项、
HCU4单步16组/公共25步4组、MPA-0 CPU三步4组。保留上述未验证范围。

M4 的生命周期追加证据独立见 [M4 报告](../reports/g2-mpa0-relaxation-m4.md) 与
[应用契约](TORCH_MACE_MPA0_RELAXATION_CONTRACT.md)：普通单优化器 FusedStage 的联合
收敛、退出/补位、变胞邻居、薄 CIF 入口已验收窄 CPU/合成 HCU 范围；MPA-0 CPU
FP64 有真实收敛，FP32 仅 budget 短步。M3 的数值门不变；真实目标 HCU 仍待 M5。
