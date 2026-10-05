# MACE-MPA-0 与原生 FIRE/FIRE2 弛豫契约

任务：`TORCH-MACE-MPA0-RELAX`。M1 分支 `codex/test-mace-mpa0-baseline`，起点 `35ea133`，
CPU 探针提交 `61484db`。本契约是分里程碑移植与验收约定，不代表下列功能均已实现。
每个里程碑开始前由用户确认，完成后报告实现、CPU/HCU 证据与剩余限制，再提出下一步。
按用户2026-10-05确定的顺序，下一阶段确认后才收口提交当前里程碑、启动下一阶段。

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

### M4 工作流验收约定（2026-10-05）

复用现有 BaseDynamics/FusedStage、SizeAwareSampler、HostMemory 和多条件
ConvergenceHook，不引入新晶胞参数化或独立 filters。修正掩码更新遗漏私有状态和
补位后未 priming 的来源缺陷，分别登记 UBF-015/016。每次模型评估仍调用 compute
Hooks；补位 priming 不推进步数。退出字段在 observer Hooks 前恢复。

CPU 门覆盖两种优化器的四种掩码入口、异构 scalar refill 的 survivor/fresh-state
一致性、真实 sampler/sink ID 唯一性、力与绝对应力 AND 收敛、解析势联合收敛；
邻居门独立覆盖当前晶胞压缩/剪切、cutoff、多周期镜像、容量增长及 skin 缓存失效。
向量构造参数继续支持普通 Batch，但动态 arrivals 缺少来源参数映射时明确拒绝。

应用为薄 CIF 入口，按体系计步且分别记录 converged/max_steps/wall_timeout/
invalid_input/calculation_error/not_started；budget 退出不冒充收敛。
写出后重新读取 CIF，在实际交付的 r/H 上重算 E/F/stress 并检查阈值；CIF 规范化
晶胞朝向，分量应力用导出坐标系测量。final_check_failed 与计算异常分别记录。
模型、输入/输出哈希及实际 backend/dtype/参数写入 JSONL；部分占位拒绝。
限时实际 CPU 小样本同时验证
成功与 budget 路径，真实模型 HCU 仍属 M5；M4 HCU 只做合成工作流 smoke。

实现入口为 `examples/relax_mace_mpa0.py`。单个 FusedStage 承载一种原生变胞优化器，
scalar 参数补位；HostMemory/预读输入的内存随输入数增长，不是通用流式服务。
墙钟在步间检查，输出核验可略超预算。完整重启、同进程混合 dtype、多种变胞算法
混合阶段、全目录成功率及生产性能不在 M4 验收范围。证据见
[M4 报告](../reports/g2-mpa0-relaxation-m4.md)。

M1 数值门：FP64 Batch/ASE 对照 `rtol=2e-8`，能量 `atol=2e-7 eV`、力
`atol=2e-7 eV/Å`、stress `atol=2e-9 eV/Å³`。
中心差分步长 `1e-5`（原子/cell 为 Å，应变无量纲）；力与 cell-force
`rtol=3e-4, atol=3e-6 eV/Å`，stress `rtol=3e-4, atol=3e-7 eV/Å³`。
FP32 Batch 对照 `rtol=2e-4`，能量 `atol=2e-3 eV`、力 `atol=2e-4 eV/Å`、
stress `atol=2e-6 eV/Å³`。容差按物理量定义，失败先查原因，不放宽全局容差。
这些门只约束 M1 小样本；优化器、HCU parity 与真实收敛另按对应里程碑定义并记录。

### M5 真实 HCU 验收约定（2026-10-05，用户已确认）

复用 M4 薄应用，不新增框架/ops API 或晶胞算法。HCU4 单卡依次验证 FP64 单体系、
三结构异构 Batch（容量2，必须发生补位）、FP32 重复及受限扩大 Batch。
每体系最多500步，单次应用步间墙钟预算180秒；外层探针另限时，最终核验不计入步间预算。
扩大样本先记录真实 CIF 的原子数/哈希和显式容量，不运行全目录长作业。
固定 `dt=0.02`、零外压、`skin=0`，真收敛继续要求
`fmax <= 0.01 eV/Å AND max(abs(stress)) <= 0.1 GPa`，不将 budget 状态算作成功。

同一几何的 CPU/HCU E/F/stress 对照沿用上列 M1 各物理量 rtol/atol，比较完整力和应力
张量及排序后的邻居/周期镜像；检验输出 shape、dtype、device、有限性和无跨体系边。
FP64 再 FP32，分别设置模型与进程默认 dtype。导出 CIF 独立重建邻居、用 CPU 和 HCU
重新评估，并核验逐体系来源/ID/结果哈希与真收敛判定；不要求非线性优化轨迹逐步相同。
Torch reference 的模型/算子数据保留在指定设备；仅对照证据和 CIF I/O 显式转主机。

记录启动/模型装载与端到端墙钟、同步的 warm 单点评估耗时、峰值 allocated/reserved
显存，二者不混作生产吞吐。真实模型工作流通过不等于完整 HCU gate、生产 HIP/Triton、
NVIDIA 数值对照、全目录成功率、完整 checkpoint 或流式输入已验证。

每阶段运行相关 CPU gate，HCU 先只读确认资源、单卡限时；无资源记录 pending。
缺少 NVIDIA fixture 时记录缺口，fork 的归档测试说明不能充当本工程运行证据。

### 补充：单个较大 Batch 的耗时测量（2026-10-05，用户明确授权）

只测 FP64/FIREVariableCell，不执行全部1600结构。当前目录CIF按文件名排序，
Python random.Random(24).sample(...,1600) 保留返回顺序，取前64个；
本地候选21755个，和用户提供的H100约34570候选不同，不称为同一输入集合。
预检64结构共2287原子、单结构3–132原子；最大活跃64、原子容量2304，refill_frequency=20。
全部输入可一次放入，因此可观察退出/清空而没有新到达结构，不作为持续补位吞吐证据。

复用原生优化器参数 dt=.1/dt_max=1/dt_min=.002/maxstep=.2/n_min=5/
f_inc=1.1/f_dec=.5/alpha_start=.1/f_alpha=.99，真实原子质量、晶胞质量1、零外压。
每结构最多500次实际更新，模型eval且冻结权重，仍计算位置/应变梯度；每次重建6Å邻居。
普通e3nn eager/Torch reference，无CuEq。COO邻居容量动态分配，逐原子超过512则明确报错，
不截断。单卡HCU4、单worker/OMP=1，单次步间墙钟600秒、外层限时720秒。
本进程Torch allocator限总显存80%，超限明确失败；不是整卡或非Torch内存的绝对上限。

用户追加建议改用FIRE自带晶胞工具：停止观测器调用现有stress_to_cell_force的
完整矩阵路径 `F_H=-V*inv(H).T@stress`，与原子物理力一起取行二范数的最大值。
<=.05 eV/Å收敛，>100 eV/Å溢出退出，没有独立应力阈值。该原生raw力判据
与H100提供的参考形变/cell_factor=N的UnitCellFilter判据不同，不称为等价对照。
仅在探针内配置现有ConvergenceHook，不新增产品filter或改变优化更新方程。
NumPy三解析案例检验原生晶胞力及异构体系最大值；CPU/HCU分别运行。
原始r/H用于最终判定并保存lossless状态，不用CIF规范化后的朝向替代它。
初轮在完整优化步完成前被本地校验 `dt_min<=0` 拒绝，不计入耗时外推；
最终测量采用当前原生默认下限.002，与H100给定的0不同，未在本任务修改算子校验。

端到端包括导入/设备初始化、输入/模型、优化/邻居/退出、导出；分项及实际峰值显存另记。
1600估算以当前实测规模/步骤分布作条件外推，区分固定启动成本，不能声称已跑1600或
已验证16000原子Batch、CuEq/H100性能、持续补位收益或所有结构都在500步收敛。

### 补充：8000原子门槛与分项计时（2026-10-05，用户追加授权）

用户要求仍只跑一个完整Batch，max_atoms提高到8000，并估计模型/优化器等各部分耗时。
复用相同Random(24)抽样返回顺序，最长不超过容量的前缀为231个结构、7937原子，
每项3–176原子；下一项80原子会超过门槛，故不纳入、不拆分结构。
沿用上一测量的FP64、普通e3nn、原生完整矩阵FIRE与raw力阈值、dt_min=.002、
refill20和单卡单线程条件；初始模型求值即为显存预检，成功才继续优化。
本进程allocator仍限80%；OOM明确记录失败阶段，不降精度、CPU回退、拆模型微批或多卡。
首轮初始求值被80%限额拒绝时卡仍有13.4 GiB空闲，不能据此判定整卡容量不足；
资源重新确认空闲后，对同一批输入以95%限额再预检，保留约3 GiB余量。
95%仍在初始e3nn前向中OOM；为区分人工限额和实际可用显存，确认整卡仍空闲后，
对同一输入最后以默认100%allocator上限试一次。仍失败则结束，不再重试。
所有尝试保留；失败的部分前向计时不当作完整模型或弛豫耗时。
步间墙钟上限1800秒、外层限时1920秒，不扩展到全部1600项。

仅在隔离探针给现有调用加同步墙钟观测：邻居重建及容量检查、模型compute
（包含输入适配、能量前向、力/应力autograd及输出回写）、优化器masked pre/post update
（含晶胞力、更新和私有状态/退出mask保护）、停止指标观测器。
其余工作流耗时为优化循环总墙钟减上述互不嵌套的分项，含sampler/退出清空/Hook等。
准备、导入/oracle、导出另列。同步测量可能改变耗时，不能将这些墙钟称为纯kernel时间
或与上一未分项同步的测量作严格吞吐比较；记录调用数和失败调用，未执行项不估造耗时。
不细分MACE内部tensor product/反向kernel，也不据未完成的初始求值估算1600项耗时。
验收仍区分CPU观测器/oracle与HCU实际Batch结果；若首求值失败，只记录输入和诊断，
不导出未求值的结构作为弛豫结果。

### 补充：4000原子门槛单Batch完整分项（2026-10-05，用户确认）

用户确认将门槛降至4000，仍只尝试一个完整Batch。复用同一21755候选、Random(24)
无放回1600清单的最长容量前缀：114结构/3959原子，每项3–176原子，下一项44原子
会超过4000，未纳入。直接读取原CIF，不预弛豫、标准化或拆分结构。
复用上一阶段已接入的同步分项探针，不修改framework/ops/产品应用或算法。
FP64、普通e3nn、原生完整9项FIRE、raw力.05收敛/>100溢出、500实际更新预算、
dt_min=.002、6Å邻居每次重建及逐原子512容量检查、单卡HCU4/OMP=1/refill20不变。
初批全部装入，后续只退出/清空，不构成持续到达补位吞吐证据。

初始allocator限80%；显存初始求值通过后继续完整优化。若只因人工限额失败而卡仍空闲，
可对同一输入用95%或默认上限预检，明确保留所有尝试；不改变精度/模型/算法或拆微批。
步间墙钟1800秒、外层1920秒。分项包括compute的E/F/stress求值、邻居、原生masked
优化器pre/post、停止观测器和工作流残差；准备/导入/oracle/导出另列。
分项同步有测量开销，不称为纯kernel时间；不得从首次求值推断完整峰值显存或每步吞吐。
完成后核验全部ID/哈希、lossless FP64原始状态/原生力/步数分类、CIF几何及计时分项总和。
独立模型重算、完整产品gate、1600全量或H100同口径对照仍非本次验收。
