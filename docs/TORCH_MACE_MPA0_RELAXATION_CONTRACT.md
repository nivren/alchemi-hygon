# MACE-MPA-0 与原生 FIRE/FIRE2 弛豫契约

任务：`TORCH-MACE-MPA0-RELAX`。M1 分支 `codex/test-mace-mpa0-baseline`，起点 `35ea133`，
CPU 探针提交 `61484db`。本契约是分里程碑移植与验收约定，不代表下列功能均已实现。
每个里程碑开始前由用户确认，完成后报告实现、CPU/HCU 证据与剩余限制，再提出下一步。

## 1. 来源与范围

- NVIDIA framework 基线：`4dfe3723def34df3fadb245981081ccf8c94c257`。
- NVIDIA ops 基线：`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。
- 指定 fork framework：
  [b12b54c48fc8001adb84654c3b42f2bef21db710](http://10.208.16.64:3000/wangleping/nvalchemi-toolkit/commit/b12b54c48fc8001adb84654c3b42f2bef21db710)。
  补齐 `FIREVariableCell` 应力驱动晶胞更新、联合功率/混合/重置和共同位移限制。
- 指定 fork ops：
  [c4c3bb56b6c6ae0d5e85b4e1ed14bd4bd4748a2a](http://10.208.16.64:3000/wangleping/nvalchemi-toolkit-ops/commit/c4c3bb56b6c6ae0d5e85b4e1ed14bd4bd4748a2a)
  修正行晶格公式与耦合运动；
  [eadc3eeebaeceb832f7341eb59303dc588f857bd](http://10.208.16.64:3000/wangleping/nvalchemi-toolkit-ops/commit/eadc3eeebaeceb832f7341eb59303dc588f857bd)
  保证每体系先更新 FIRE 控制参数，再混合原子/晶胞速度。
- fork 的两份 develop 在 2026-10-04 分别指向上述 b12b54c 和 eadc3ee；后续只按 SHA
  摘取相关修改。fork 含较新的数据、训练、编译与依赖改动，不整体覆盖本地 packages，
  不改变 `UPSTREAM_LOCK.yaml`，不把 fork 状态当作 NVIDIA 官方修复状态。

公共入口保持 `MACEWrapper`、`FIRE`、`FIRE2`、`FIREVariableCell`、
`FIRE2VariableCell`、`AtomicData`、`Batch`、`ConvergenceHook` 和 `NeighborListHook`。
保留原参数与默认值；新增后端参数沿用项目已有显式选择惯例。
晶胞工具来源包括 ops `dynamics/utils/cell_filter.py`、framework
`dynamics/_ops/cell_align.py::align_cell`；保持其导入入口、单体系/Batch 约定和原地更新语义。

目标是单 HCU 普通异构 Batch 的周期晶体弛豫：固定晶胞或零外压原子/晶胞联合优化。
ASE UnitCellFilter/FrechetCellFilter、BFGS、有限外压新接口、NPT/NPH、DomainParallel、
训练/二阶模型梯度、完整 workflow checkpoint 和生产 HIP/Triton 优化不在此任务验收范围。
已有高级接口继续保留，未验证能力不得登记为 verified。

## 2. 两条晶胞优化路径

| 路径 | 来源与语义 | 需保护的行为 |
|---|---|---|
| FIRE2 | NVIDIA 原生对齐下三角晶胞；六项 `[00,10,11,20,21,22]`，`3N+6` | pack/unpack、每体系原子数归一化、联合归约/混合、上坡回退、仿射运动及真实 Cartesian 位移限制 |
| FIRE | NVIDIA 原版已有全 3×3 状态但缺少完整驱动；采用指定 fork 修复后的九项晶胞更新 | 原子 kick 使用原子质量，晶胞优化质量为 1；晶胞速度是直接 H 导数；九项参与同一 FIRE 控制；原子与 cell-row 共同位移限制 |

全矩阵 FIRE 不是新增 ASE deformation-gradient filter。九个矩阵分量也不代表九个独立
物理形变自由度。两条路径不统一为新的自定义参数化，不要求轨迹或局部极小值完全相同。

晶格向量为 H 的行，`r=s@H`。E 为 eV，r/H 为 Å，力为 eV/Å，stress 为
拉伸为正的 Cauchy stress，单位 eV/Å³；V=`abs(det(H))`。
原子与晶胞同时施加对称应变时 `sigma=(dE/d epsilon)/V`；固定分数坐标晶胞力
`F_H=-V H^-T sigma`。细节与 UBF-009 见 [基础契约](TORCH_CELL_STRESS_FORCE_CONTRACT.md)。
FIRE2 的力缩放和 FIRE 的优化质量分别按来源实现，不能用一个统一经验系数代替。

## 3. 算子、后端与数据约定

- atoms：positions/velocities/forces `[N,3]`；cell/stress/cell velocity `[B,3,3]`。
  原子质量按来源 ABI 接受 `[N]` 或 `[N,1]`；dt/alpha/计数等按体系保存。
- `batch_idx` 非递减并与 `batch_ptr`/`atom_ptr` 对应。FIRE2 扩展布局为
  `[N+2B,3]`；fork FIRE 为每体系原子后接三行晶胞，布局 `[N+3B,3]`。
  收缩、退出和补位时同步全部优化器状态，不混用原体系索引。
- FP32/FP64 同设备、同 dtype；reference 在输入设备内运行，无静默 CPU 回退/降精度。
  每个实际新增 operation 通过 ops catalog 和 generic executor 登记，framework 解析
  selection 一次。`None/warp` 保留 legacy 选择；未注册、能力不满足的请求显式失败。
- 晶胞对齐同步旋转原子坐标；能量与几何不变，力/应力按同一变换对应。
  纯晶胞仿射运动保持分数坐标，不人为锁定分子内部结构或相对于晶格的取向。
- pack/unpack、alignment 和优化更新以 forward-only 为首轮验收；保留来源 mutation 和
  alias 行为。stress conversion 已有的可微路径保持；不宣称优化轨迹可微。
- 对每个 operation 明确输出/原地写入对象、stride/alias 限制、当前 stream 行为、空 Batch
  和错误测试。奇异/非有限晶胞、非正物理体积、索引错位、非法 dtype/device/控制参数
  明确失败；单体系零原子的支持按来源单列，不用空整个 Batch 测试替代。
- 不承诺跨设备逐位一致、完整 compile 或 NVIDIA/Warp parity；分别报告证据。

## 4. 模型与输入

目标 checkpoint：`mace-mpa-0-medium.model`（官方标识 `medium-mpa-0`）。
SHA256：`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
使用已有可信本地文件，无自动下载/E0 改写；ordinary e3nn eager，
`enable_cueq=False`、`compile_model=False`，显式激活 energy/forces/stress。
MACE 0.3.15 的 ZBL 排斥项中整数原子序数的浮点幂受 PyTorch 默认 dtype 影响。
单一精度作业须显式匹配默认 dtype、权重和输入；probe 每个 case 配置且最终恢复默认值。
ASE calculator 会设置进程默认 dtype，其对照须隔离；不通过事后转换输出掩盖提升。
同进程混合 dtype 并发模型推理未验收，不修改共享模型包装器的全局 dtype 行为。

输入为类似指定 Li/Na 目录的 CIF，保留原子、晶胞、PBC 和来源 ID。校验模型元素覆盖、
三维周期性、晶胞可逆/体积、有限坐标；部分占位/混合占位必须明确处理或报错，
不能静默选择元素。ASE 只承担 CIF I/O 和单点对照，不用 ASE optimizer 替代原生优化器。
原子数与 cell shape 不固定为旧 92-atom 探针的夹具。

M1 固定三个真实 CIF：Li2MgGe（4 原子）、LiSnCl3（10 原子）、NaLiTl2（4 原子），
对应输入内容 SHA 和每次评估的 Batch 元数据写入原始证据。
这是有限样本，不代表完整目录或材料预测精度已验收。

## 5. 邻居、收敛与应用输出

复用 periodic full-list reference、image shifts、容量扩展与连续 geometry。
首轮 `skin=0` 每步用当前晶胞重建；压缩/剪切、cutoff 附近、多周期镜像和增长容量
需要独立回归。已有 skin>0 接口保留，其晶胞变化缓存失效行为单列验证。
不将当前 fixed-cell HIP neighbor 能力用于变胞任务。

复用现有多条件 ConvergenceHook：原子最大力范数与最大绝对应力分量同时达标。
建议初值 `fmax<=0.01 eV/Å`、`max(abs(sigma))<=0.1 GPa`
（约 `6.2415e-4 eV/Å³`），均可配置；固定晶胞只用其适用的力条件。
步数/墙钟耗尽、非法结构或计算异常分别记录，不标为成功。
已退出体系的 positions/cell/forces/energy/stress 保持一致；补位初始化 cell velocity、
dt/alpha/计数和临时缓冲。最终在输出坐标与晶胞上重算模型并核验阈值。

提供薄应用入口读取目录、按可控批次处理，输出最终 CIF 和逐体系 JSONL/表格；记录
来源 ID、模型 SHA、后端、dtype、优化器及参数、状态、步数、最终 E/体积/fmax/stress。
不以自行实现的工作流替代上游 Dynamics/Hooks。全目录长作业待小样本验收后另行安排。

## 6. 里程碑与验收

1. M1：stress 基础收口、来源与契约、真实 MPA-0 CPU 单点/有限差分。
   使用 FP64 做数值导数，检查三个体系 Batch 对逐体系及 ASE calculator 的 E/F/stress；
   原子位移、六种对称应变和两项固定分数坐标 cell 扰动分别检查，差分期间邻居集合不变。
   FP32 单独做有限前向与 Batch 对照，不用 FP32 相减的总能量作导数 oracle。
2. M2：通用晶胞工具 Torch reference，解析/round-trip/异构 Batch CPU 与限时 HCU gate。
   已完成窄 primitive 验证，详见 [工具契约](TORCH_CELL_TOOLS_CONTRACT.md) 与
   [M2 报告](../reports/g2-cell-tools-m2.md)；应用优化器和目标模型旋转验证不由此替代。
3. M3：先 FIRE2 后 fork FIRE，分别验收单步、功率分支、耦合运动、位移限制和公共接线。
   已完成上述窄 reference、Hook 对齐协变、CPU MPA-0 三步与合成 HCU4 对照；
   [更新契约](TORCH_VARIABLE_CELL_FIRE_CONTRACT.md) / [M3 报告](../reports/g2-variable-cell-fire-m3.md)。
   不代替后续真实收敛、退出补位及目标模型 HCU 验收。
4. M4：邻居生命周期、联合收敛、退出/补位、薄应用入口与结果导出。
5. M5：真实小样本 HCU4，单体系→异构小 Batch→扩大 batch；FP64 再 FP32。

M1 数值门：FP64 Batch/ASE 对照 `rtol=2e-8`，能量 `atol=2e-7 eV`、力
`atol=2e-7 eV/Å`、stress `atol=2e-9 eV/Å³`。
中心差分步长 `1e-5`（原子/cell 为 Å，应变无量纲）；力与 cell-force
`rtol=3e-4, atol=3e-6 eV/Å`，stress `rtol=3e-4, atol=3e-7 eV/Å³`。
FP32 Batch 对照 `rtol=2e-4`，能量 `atol=2e-3 eV`、力 `atol=2e-4 eV/Å`、
stress `atol=2e-6 eV/Å³`。容差按物理量定义，失败先查原因，不放宽全局容差。
这些门只约束 M1 小样本；优化器、HCU parity 与真实收敛另按对应里程碑定义并记录。

每阶段运行相关 CPU gate，HCU 先只读确认资源、单卡限时；无资源记录 pending。
缺少 NVIDIA fixture 时记录缺口，fork 的归档测试说明不能充当本工程运行证据。
