# ASE-compatible BFGS M5-1：Jacobi 性能定位

2026-10-10，用户确认后在 `wangleping/perf-ase-bfgs-m5-1` 完成。
延续未提交M1–M4；本阶段只新增合同、探针和记录，**没有优化或替换产品求解器**。
11个BFGS/Filter产品源文件的阶段前后SHA256相同，版本锁及默认环境未改。
未提交、推送或集成；M5-2尚未启动。

## 结论与适用范围

- 已测真实147/285维Hessian中，原生Jacobi与完整checked eigh的差距约0.4–0.5ms；
  独立残差/正交有效性守卫约0.5ms。删除检查或优化Python封装的收益空间有限。
- 285维B1/B4的native trace窗口分别有97.0%/97.6%用于GPU kernel；主要热点为
  `syevj_offd_kernel` 和 `syevj_offd_rotate`。收敛标量回传确实存在，但API等待与
  kernel重叠，不能把API总时间当成PCIe搬运时间，也不能相加作耗时拆账。
- 三次真实MPA-0 FIRE2→UnitCellFilter BFGS两结构均收敛。checked eigh约占
  BFGS更新88.8%、整个工作流窗口47.8%，支持继续针对Jacobi kernel做隔离试验。
- 当前已经使用rocSOLVER strided-batched，B4对应grid.z=4；再次绕过Torch的
  逐矩阵循环不能解释或解决当前核心瓶颈。这里没有新后端或加速倍数声明。

这是当前构建、单卡、小样本FP64 eager证据；不是全CSP成功率、生产吞吐或
其他DTK/硬件验证。使用[性能合同](../docs/ASE_BFGS_M5_1_PROFILE_CONTRACT.md)。

## 环境、输入与可重复性

当前项目Torch `2.9.0+das.opt1.dtk2604`，HIP `6.3.26093`，
rocSOLVER `3.20.0.80bb870c`；DTK `/opt/dtk-26.04`，dcc25.10/clang17。
物理HCU4：BW1000 64G/gfx936、80 CU，映射逻辑cuda:0。前后hy-smi无其他KFD进程；
全部测试顺序、限时运行，只占一张卡，没有更换库或占用其他设备。

真实Hessian来自M4捕获的 `last-147.pt` / `last-285.pt`；SHA256分别为
`796be649c9d4ce7e82de840de53242d6acd9c0c4f727eabd91772cc5b5d8d5b9`、
`b6292e2269fb80370da1ffe4729d8a9c85ab928d8842ecd9621bc4ad2c0e2ded`。
CPU FP64 LOWER oracle单独生成并冻结，profiler进程不现场运行CPU eigensolver。
B4为相同捕获矩阵的四份副本，不能代替不同谱的batch覆盖。

原始材料：`artifacts/ase-bfgs-m5-1-2026-10-10/`，含
`commands.txt`、`exits.tsv`、环境/资源记录、raw JSON/trace/log、输入/产品hash、
可重复分析脚本与 `final-sha256.json`。正常计时与profiler材料分别保存。
本阶段新增 [Jacobi探针](../probes/ase_bfgs_jacobi_profile.py)、
[CSP分项探针](../probes/ase_bfgs_csp_profile.py)；后者只在独立进程内包裹既有
函数，不改产品文件。原生扩展的构建仍使用现有loader/flags，不引入新HIP算子。

## 同输入稳态微基准与数值结果

三个独立进程，每项预热2次、7样本，共21样本；raw和checked均以完成GPU执行
为计时边界，wall与stream-event原始值同时保留。stream-event包含流空隙，
不能当成GPU active kernel累计时间。下面为pooled wall中位数，单位ms：

| n | B | raw native | checked eigh | 输入检查 | 输出检查 | clone | 布局copy | 有效性守卫 |
|---|---|---|---|---|---|---|---|---|
| 147 | 1 | 40.072 | 40.507 | 0.150 | 0.307 | 0.048 | 0.057 | 0.495 |
| 147 | 4 | 46.027 | 46.475 | 0.151 | 0.301 | 0.049 | 0.058 | 0.489 |
| 285 | 1 | 92.161 | 92.641 | 0.152 | 0.307 | 0.050 | 0.059 | 0.495 |
| 285 | 4 | 115.404 | 115.789 | 0.152 | 0.308 | 0.045 | 0.053 | 0.533 |

raw范围分别40.022–40.278、45.968–46.175、92.032–92.257、115.355–115.514ms；
相对标准差约0.04–0.17%。详细标准差和逐进程中位数见 `micro-summary.json`。
分项是独立窗口，不能直接相加作为一次checked调用的成本。布局copy为同shape
transpose/contiguous的代表测量。raw不包含产品的检查，不能作为可部署候选。

冷加载与首调用已分开：首次扩展加载/JIT约30.51秒，后续进程cache加载约
0.58–0.98秒；微基准每进程首个147维solver调用约2.99–3.04秒，均不进入上述
稳态样本。CSP工作流的首模型求值冷初始化仍计入其工作流窗口。

所有case info=0、输入不变；谱atol=2e-10/rtol=2e-11，方程/正交误差<1e-10：

| n | 最大谱绝对误差 | 相对方程误差 | 相对正交误差 | Jacobi sweeps |
|---|---|---|---|---|
| 147 | 1.302e-11 | 5.659e-14 | 5.596e-14 | 19 |
| 285 | 2.561e-11 | 1.134e-13 | 1.143e-13 | 20 |

147/285/147异构核心step按[2,147,147]/[1,285,285]分组，7样本wall中位数
138.488ms、范围138.318–138.547ms、标准差0.091ms；坐标相对CPU独立谱方向
oracle误差分别1.241e-14/7.216e-16/1.241e-14。这是捕获Hessian加合成q/f、
每次恢复未初始化历史的核心窗口，不是三条真实CSP轨迹或一般H更新基准。

## hipprof：kernel、复制与API等待

每个case预热后采一分解，按探针输出的UNIX纳秒窗口筛选事件；数值oracle检查
及其D2H在窗口外。对GPU kernel区间求union，避免与CPU API等待重复累计：

| case | 窗口ms | kernel union ms | 窗口占比 | kernel数 | 4-byte D2H次数/物理时长ms |
|---|---|---|---|---|---|
| 147/B1 | 47.550 | 39.192 | 82.4% | 577 | 20 / 0.649 |
| 285/B1 | 93.966 | 91.143 | 97.0% | 1007 | 21 / 0.610 |
| 285/B4 | 117.131 | 114.375 | 97.6% | 1007 | 21 / 0.733 |

147 trace含额外等待，明显高于40ms稳态，不能用单次profile替代重复基准。
数值矩阵复制均为D2D（各1次，172872/649800/2599200 bytes）；B4依然每次只回传
4-byte批收敛标量。本轮trace没有重新部署LAPACK数学observer，设备内数学的既有
准入证据沿用M4；仅根据上述复制大小不能证明所有私有CPU函数都不存在。

285/B1主要kernel累计：offd 52.153ms/180次，offd_rotate(false)18.253ms/360次，
offd_rotate(true)8.898ms/180次，finalize6.135ms/1次；B4分别52.292、33.375、
14.399、6.232ms。offd每launch是5个1024-thread block，B4的grid.z=4。
该并行粒度是下一候选的依据，不是已证明的寄存器/LDS/指令根因。

`hipMemcpyAsync`的CPU duration包含等待之前GPU kernel；不能把它归为纯复制。
例如147/B1的相关API累计约38.5ms，同时GPU kernel已占39.2ms，两者重叠。
Torch分组profile另有ROCTracer duplicate-flow-start warning；因此不据该flow图
推断依赖根因。hipprof的独立窗口及上述稳态/数值门均正常完成，退出0。

## 真实MLIP变胞CSP分项

使用 **MACE MPA-0 medium，不是mace_off small**；FP64、energy/forces/stress，
cuEquivariance/compile关闭。checkpoint为
`/home/wangleping/.cache/mace/mace-mpa-0-medium.model`，SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
46/92原子周期CIF与M4相同，其完整路径/hash在各 `csp-RUN.json`。
FIRE2/Torch reference预弛豫20步；BFGS显式HIP、默认ASE UnitCellFilter、
generalized fmax<0.05；neighbor仍为Torch reference/skin0。

三进程分别限时210秒、工作流wall上限180秒，禁用CPU/独立ASE额外模型求值
(`--check-steps 0`)、捕获和profiler。本阶段未重新做真实MLIP逐步ASE验收；
JSON零误差字段不代表本轮做了对照，原有M4数值证据保持独立。

| run | 工作流窗口s | 模型s | BFGS更新s | checked eigh s | 有效性守卫s |
|---|---|---|---|---|---|
| 1 | 56.877 | 13.750 | 30.607 | 27.187 | 0.141 |
| 2 | 56.553 | 14.077 | 30.509 | 27.089 | 0.147 |
| 3 | 56.949 | 14.157 | 30.605 | 27.188 | 0.149 |

三次均status passed，2/2结构收敛，stage233步/234次主模型求值，独立更新88/213次，
全部solver info=0。最终广义fmax范围0.043552及0.049079–0.049162；小的跨进程
数值变化存在，未声明轨迹逐量一致。原子物理fmax与广义fmax不同，本阶段采用既有
UnitCellFilter的联合收敛判据，不把physical fmax单独解释为0.05门。

工作流时间中位数56.877秒、标准差0.172秒；checked占整个窗口中位数47.80%、
BFGS更新88.83%，独立guard约0.15秒，不属于checked内部。两种solver分项是
嵌套计时：native_cpu_return在CPU返回处结束，可能尚有finalize/布局kernel，
checked包括既有同步检查；两者之差不能直接解释成封装开销。

此处总时间从stage.run前计至返回并同步，包含首模型求值冷初始化、同步计时、
原有每10步checkpoint和框架调度；不含模型加载、预先扩展加载或最终写盘。
这些是仪表化完整工作流窗口，共3次，不能当成7次稳态吞吐采样或无I/O生产性能。
其他工作流残差中位数约12.19秒没有继续细分，不能全部归因于调度。

## 验收、交付与下一步

三micro、group、三hipprof、group-profile、三完整CSP均实际退出0；探针语法、
分析脚本与源文件不变检查通过，相关命令/退出码在原始目录。本阶段未重跑
M4全CPU650/HCU53项功能gate，因为产品源未变；不把历史gate报为本轮测试。

下一步建议：[M5-2隔离HIP batched Jacobi提案](../docs/ASE_BFGS_M5_2_PROPOSAL.md)。
先做FP64 full w/V、真实147/285 B1/B4、简并谱/ASE/stream/失败语义验证；过门后
才比较当前rocSOLVER的同输入稳态和同配置CSP。建议保留条件为完整checked
solver至少10%下降且CSP有重复可见收益。候选未过数值或性能门则记录并停止接入。
本阶段没有实现自研求解器、没有承诺收益，也没有扩大现有产品支持范围。
