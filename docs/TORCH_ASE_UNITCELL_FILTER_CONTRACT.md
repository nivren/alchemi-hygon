# TORCH-ASE-UNITCELL：M3 合同

M4 扩展另见[Batch合同](TORCH_ASE_BFGS_BATCH_CONTRACT.md)：沿用本合同的每体系
3N+9数学、Filter参数与快照，增加异构Batch、ID历史、冻结/重排/退出/补位及Batch
restart。下文的单体系边界是M3验收边界，不是当前完整功能上限。

2026-10-09，用户已确认 M3。新增功能，不是 NVIDIA 上游 BFGS 移植。
Oracle 为锁定环境 ASE 3.29.0 `UnitCellFilter`，来源及哈希见
[BFGS 合同](TORCH_ASE_BFGS_CONTRACT.md)。上游 Ops 0.5.0-rc 的变胞
L-BFGS 使用对齐晶胞的六个三角分量（`N+2` 个 vec3 行）；本合同使用完整
形变梯度（`N+3` 行）。9 是矩阵分量数，不是九个独立应变，两个表示不数值等价。

## 数值和接口

Ops `ASEUnitCellFilter.prepare(cell, num_atoms, *, mask=None, cell_factor=None,
hydrostatic_strain=False, constant_volume=False, orig_cell=None,
scalar_pressure=0.0)` 返回独立参考晶胞及参数。输入为 FP64、CPU/CUDA-HIP、
contiguous strided、forward-only；cell/orig_cell 为 `[3,3]`，体系非空。
mask 接受 ASE Voigt 顺序 `[xx,yy,zz,yz,xz,xy]` 六元数组或 `[3,3]`，
有限数值掩码语义也保留。None 为全一。cell_factor 默认 float(N)，须非零且有限；
负缩放参数也按 ASE 保留。
orig_cell 默认初始晶胞；当前和参考晶胞须有限、非奇异，不要求右手性。
压力单位 eV/Å³。无隐式迁移或转换输入张量 dtype；Python 参数按 FP64 在设备上创建。

行晶胞 `r=s@H`，参考晶胞 H0，`F=solve(H0,H).T`：

- `pack(r,H)`：`[solve(F,r.T).T; cell_factor*F]`。
- `unpack(q)`：`T=I+(q_cell/cell_factor-I).T*mask`，返回独立
  `r_new=q_atom@T, H_new=H0@T`，校验后才允许调用者原位提交。
- `generalized_forces(f,stress,H)`：stress 为 `[3,3]`、ASE 正拉应力。
  `W=-abs(det(H))*(stress+pI)`；原子行 `f@F`；晶胞行先
  `solve(F,W.T).T`，再依次静水投影、mask、减去 trace/3 定体积投影，
  最后除 cell_factor。返回广义力和投影后的 ASE Voigt 应力（非对角取对称平均）。
- `enthalpy(E,H)=E+p*abs(det(H))`；不覆盖 Framework Batch 的物理势能。

无 clamp、阻尼、对齐/三角化、精确体积约束或额外曲率规则。
constant_volume 与 ASE 一样只是去迹投影，**不保证每步体积精确恒定**；
投影选项不都构成全坐标上的保守梯度。有限差分门只对未投影的完整广义坐标
（含外压）执行，投影路径逐项和 ASE 对照。不支持 Atoms constraints、
StrainFilter、FrechetCellFilter、梯度和 compile。异常不回退 CPU 或更改精度。

## State、Framework、调度

共用 `bfgs_ase_compat` 的 Torch/HIP prepare/step 数学，State 增加明确
`coordinate_mode='cartesian'|'ase_unitcell'`。固定晶胞快照 schema 1 原样保留；
UnitCell 快照 schema 2 带模式且至少 12 个标量自由度。缓存 selection 按模式
校验 `single_system/fixed_cell` 或 `single_system/variable_cell/ase_unitcell`。
两个模式不能混用快照或 selection。HIP eigensolver 算法和运行时准入不变。

新增公共 `ASECompatibleBFGSVariableCell`，继承现有 BFGS 的 BaseDynamics/
Hooks 生命周期，参数与 ASE Filter 对齐，接受单体系 FP64 Batch。
forces、energy、cell `[1,3,3]`、stress `[1,3,3]` 必须具有匹配的独立存储；
每次模型计算前通过已有 BEFORE_COMPUTE 邻居 hook 使用当前晶胞。
严格 `max(norm(generalized_force_rows)) < fmax`，maxstep 缩放全部 N+3 行；
这并非单独限制物理原子 Cartesian 位移或原始应力。
退出/False mask 冻结物理几何及 Hessian/history/Filter。补位、异构 Batch 留给 M4。

为保持无效候选晶胞时原子/晶胞/历史同时不变，单体系 M3 在 scratch State 中
计算候选，unpack 校验成功后提交；增加 O(d²) 设备内复制开销，性能单独记录。
模型计算失败发生在已提交的更新之后，沿用现有 Dynamics 异常行为。
检测普通位置/力/晶胞/应力张量原位修改以失效力缓存，不改变参考晶胞。

Filter snapshot 独立克隆参考晶胞、mask 和参数；Framework 专属带 kind 的版本 1
snapshot 保存物理坐标、当前晶胞、原子/PBC 身份、Filter、BFGS 和调度计数。
load 全量校验后原子提交；显式 device 可迁移、不可改 dtype。重启需配套 Batch
坐标/晶胞/身份一致，首次重新计算模型力/应力；自定义 hook/模型单独重建。

## 验收

先 CPU：非三角形/左手晶胞，参数及组合逐项 ASE 对照；完整 3N+9 广义力中心差分；
逐步 Hessian/方向/广义坐标/物理坐标/晶胞/力/应力，继续运行和 restart；
初始联合收敛、冻结、失效缓存、拒绝别名和无效快照，固定晶胞回归及 CPU gate。
再限时单张空闲 HCU：Torch/HIP 同一解析路径、设备 guard，周期 MPA-0 小体系与
ASE CPU oracle 联合收敛和 restart。实测命令、退出码、误差及性能写入独立报告。
没有 NVIDIA 环境，不宣称 NVIDIA GPU 运行验收；M4 未完成前完整 BFGS 特性仍 planned。
