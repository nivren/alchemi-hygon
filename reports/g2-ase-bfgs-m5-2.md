# ASE-compatible BFGS M5-2：隔离 HIP Jacobi 原型与验收

2026-10-10；分支 `wangleping/perf-ase-bfgs-m5-2`，基于
`cfd22ad11fe0e1704dff48f53abee69135f8db99`，承接未提交的 M1–M4/M5-1 工作。
用户“继续”确认[M5-2方案](../docs/ASE_BFGS_M5_2_PROPOSAL.md)。

**隔离原型已实现，CPU oracle、HCU 数值/BFGS/真实变胞 ASE 对照及微基准已通过；
后续新窗口三轮真实CSP配对亦通过，完整端到端性能门已补齐。**
新窗口结果见[配对验收报告](g2-ase-bfgs-m5-2-e2e.md)：中位数57.308→46.726秒，
耗时下降18.466%。下文保留初次受资源影响的排除记录，该序列仍不用于性能结论。
没有注册或替换产品后端，没有提交、集成或推送。上游没有 BFGS，本阶段为本地
新增求解器实验，不记为上游 bugfix 或上游 BFGS 移植。

## 实现与环境

新增[合同](../docs/HIP_BFGS_JACOBI_PROTOTYPE_CONTRACT.md)、
[原生 HIP 源码](../probes/native/ase_jacobi_candidate.cu)、
[Torch 扩展入口](../probes/ase_bfgs_jacobi_candidate.py)、
[数值探针](../probes/ase_bfgs_jacobi_candidate_test.py)、
[配对微基准](../probes/ase_bfgs_jacobi_candidate_benchmark.py)及
[CSP 注入探针](../probes/ase_bfgs_jacobi_candidate_csp.py)。均位于 probes，产品源不改。

- 当前匹配环境：DTK26.04、Torch `2.9.0+das.opt1.dtk2604`、
  HIP `6.3.26093`、rocSOLVER `3.20.0.80bb870c`；HCU4 BW1000 64G/gfx936。
- `HIP_VISIBLE_DEVICES=4`，OMP/OPENBLAS 各1线程，MAX_JOBS=2；编译 O2，
  gfx936，不启用 fast-math。构建命令和编译产物留在原始 artifacts/build。
- 原型只接收连续实 FP64 LOWER `[n,n]` / `[B,n,n]`，n=147/285、B=1..4。
  不支持 CPU、其他规模/dtype/UPPER、grad、compile；明确报错，不回退。
- 自写 round-robin Jacobi：每轮不相交旋转，旧 A/V 只读，新 A/V 分别写入
  ping-pong 工作张量；不同矩阵使用 grid.y 批量执行，不逐矩阵调用求解器。
  奇数维度的虚拟索引仅用于配对，不增加自由度或特征值。
- 设备内展开 LOWER 为对称列主序工作矩阵，设备内中心化
  `sigma=trace(A)/n`，分解 `A-sigma*I`，排序后特征值加回 sigma。
  返回连续 Torch 行主序 V，列为特征向量。中心化不改变外部 Hessian 或 ASE
  的 abs 谱、更新、maxstep、Filter、Batch 状态提交语义。
- 收敛条件为 off-diagonal Frobenius norm / **原始未平移矩阵 norm** ≤1e-15；
  max_sweeps=100。H更新、旋转、排序、向量及诊断规约均在 HCU；主机调度轮次，
  每个 sweep 读取一个未收敛标量。允许显式标量同步，不能声称没有任何传输或全异步。
- 原型没有调用 rocSOLVER；CSP 在隔离进程临时注入原生入口，沿用既有产品
  info/finite/residual/orthogonality 守卫与 Framework。结束后恢复入口。

11个 BFGS/Filter 产品源的 SHA256 前后相同，版本锁不变；见原始
`product-source-before.json` / `product-source-after.json` / `source-check.json`。
当前注册的 HIP 实现仍是 M4 验收的 rocSOLVER 全 Jacobi，实现范围没有扩大。

## 测试和数值结果

原始目录：`artifacts/ase-bfgs-m5-2-2026-10-10/`；已记录命令/复现参数、退出码、失败
输出、原始样本与 hash 清单分别见 `commands.txt`、`exits.tsv`、各日志/JSON、
`manifest.json`。早期一次性诊断及旧版本profiler的完整shell选项未全部写入命令文件；
保留对应源码/输入/日志及原始trace，不能将复现参数当作所有历史shell调用的完整记录。

| 检查 | 实际结果 | 证据 |
|---|---|---|
| CPU 配对/旋转与独立 oracle | 退出0；7/147/285配对覆盖，7维密集 RᵀAR/V R；CPU FP64 生成16例及17例回归集 | `oracles*.json`、`cpu-prepare.log`、`cpu-regression.log` |
| 最终 HCU eigensystem | 退出0；19例：捕获/随机/解析谱/简并/近简并/负及零谱/UPPER污染/真实失败回归/两组不同矩阵B4；输入位模式不变 | `correctness-coalesced.json` |
| 失败与 stream | 13个非法输入明确拒绝；真实 max_sweeps=1 得 info=1 并报错；非默认stream通过；B4含早收敛零矩阵，冻结通过 | 同上 |
| ASE BFGS 逐步 | 退出0；147/285/147，10步，与 ASE CPU 独立 H/谱/方向/步长/坐标对照；实际分组 [2,147,147]/[1,285,285] | `bfgs-coalesced.json` |
| 重启/单与 Batch/失败原子性 | 第4步保存恢复，后续位相同；单体系与Batch一致；后组零谱、真实非收敛、错误有限V均拒绝，所有状态/坐标不提交 | 同上 |
| 访存优化不改变计算结果 | 5例的 w/V/info/residual/sweeps，共25张量，与优化前位相同 | `bitwise-coalescing.json` |
| 真实变胞 MLIP/ASE | 退出0；两CIF均收敛，各ID前3次更新通过ASE及独立Framework对照 | `csp-candidate-ase.json` |

最终19例最大谱绝对误差 **3.695e-13**，最大相对方程残差 **4.935e-15**，
最大相对正交误差 **2.381e-13**，低于既定谱 atol2e-10/rtol2e-11 与残差/正交1e-10。
误差没有用放宽门槛、降精度、jitter、clamp 或 BFGS 算法变化获取。

10步 BFGS 最大绝对误差：坐标 **9.603e-15**、H **5.684e-14**、谱 **3.524e-12**、
raw step **2.040e-14**、applied step **3.497e-15**。

真实 CSP 使用 **MACE MPA-0 medium**，不是 mace_off small；同 M4 的46/92原子
CIF与 checkpoint SHA，FP64，energy/forces/stress，cueq/compile关闭。
Torch reference neighbor、skin=0，固定晶胞 FIRE2 20步→UnitCellFilter 3N+9
BFGS；fmax=.05，233阶段步/234主轨迹模型求值，两个ID更新88/213次。
最终广义fmax **.04355185/.04890700**；ASE坐标/晶胞最大误差 **1.421e-14**，
H **2.553e-11**；独立同后端坐标 **5.329e-15**、cell **2.220e-16**、H **2.486e-12**。
该资格检查包含额外 ASE/独立模型对照，其总时间和组件次数不作为性能对照。

本阶段是探针门，不是 pytest 全门；产品未改，**未重跑**历史 M4 CPU650/3既有skip
与HCU53项，不能将历史结果报为本阶段执行。info=2非有限内部诊断定义已实现，但
本轮没有独立触发该分支；无 NVIDIA、多卡/多线程、其他构建/规模/生产验证。

## 性能证据

同卡/输入/FP64/检查范围，rocSOLVER 基线和原型交替运行。三进程，每项预热2次、
各7样本，共21样本；首调用、构建、profiler、CPU oracle不计入稳态。
完整 checked solver wall 中位数，单位ms：

| 维度 | B | 输入 | rocSOLVER Jacobi | 自写 HIP | 耗时下降 |
|---|---|---|---:|---:|---:|
| 147 | 1 | 真实捕获 H | 40.508 | 17.815 | 56.02% |
| 147 | 4 | 捕获 H 的4份副本 | 46.481 | 20.814 | 55.22% |
| 147 | 4 | 捕获/随机/简并/对角 | 44.202 | 19.795 | 55.22% |
| 285 | 1 | 真实捕获 H | 92.621 | 45.989 | 50.35% |
| 285 | 4 | 捕获 H 的4份副本 | 115.890 | 82.658 | 28.68% |
| 285 | 4 | 捕获/随机/简并/对角 | 101.270 | 64.775 | 36.04% |

包含相同产品 residual/orthogonality 守卫的窗口下降28.57%–55.45%，全部超过10%
微基准门，跨三次运行成立。不同输入与同副本分列，不把随机输入伪装成真实轨迹。
原始样本、范围与标准差见 `benchmark-coalesced-{1,2,3}.json` / `benchmark-summary.json`。

147/285/147直接分组数学窗口中位数 **136.715→68.556ms**，下降49.86%；
基线范围136.560–137.112、σ=.170ms，原型68.478–69.252、σ=.160ms。
此窗口包含残差/正交检查，**排除公共 dispatcher 入参检查**，且使用捕获H与
合成q/f、每次恢复历史，不能与M5-1的不同边界138.488ms绝对值直接比较。
stream event记录的是带调度/等待的窗口，不等于GPU kernel实际占用时间。

最初中心化版本285/B4同副本仅下降6.64%，未过门。对应hipprof窗口112.159ms，
`apply_rotations` 累计85.165ms/3420次，旋转准备18.156ms/3420次。
据此仅调整对称矩阵的等价三角读取，使相邻lane更多连续访存，保留规范的
算术次序与精度；25张量位相同，最终重复基准改善为28.68%。
没有采集最终版本的无竞争hipprof，不能声称已用counter证明具体硬件瓶颈消失。

### 初次端到端计时：资源竞争，保留排除记录

原计划同配置三轮配对，轮次顺序 baseline/candidate、candidate/baseline、
baseline/candidate。第1轮baseline耗时异常后检查资源，发现新的八卡作业已同时
占用HCU4；停止自身driver及限时子进程，driver实际退出143，没有停止其他任务。
见 `resources-during-csp.txt` / `resources-after-interruption.txt`。

已完成 `csp-rocsolver-1`、`csp-candidate-1`、`csp-candidate-2`，均退出0且233步/234
主轨迹求值；`csp-rocsolver-2`中断，第三轮未启动。**这整个计时序列排除出性能验收**，
不计算或引用其端到端加速比，不与M5-1清洁窗口拼接。
`performance-exclusions.json` 明确排除；分析程序输出 `pending_resource`、0个接受
配对、空aggregate。数值输出/日志仍保留。

微基准完成于该序列之前，初始资源检查没有其他HCU进程，重复基线接近M5-1且
方差较低；本轮没有持续采集每个样本的共享负载，不作全程独占保证。
当时已通过的微基准与数值资格不能代替尚缺的完整CSP收益门。
按项目AGENTS第7节，共享资源不可用时跳过并报告，不原地等待或占用其他任务资源。

## 开发失败记录与参考方案适用性

1. 初版285维排序错误：缺失 `__launch_bounds__(512)`，编译器警告512线程超默认
   256约束，最大谱误差80.458、退出1。修正本项目原型遗漏后通过；不是DTK缺陷
   证据。原始源码/日志保留在 `initial-rejected-source/` 与 `correctness.log`。
2. 未中心化 Jacobi 在 BFGS 第3次更新的285维真实H上100 sweeps仍未收敛，退出1。
   保存失败矩阵并加入CPU/HCU回归；中心化后原阈值下26 sweeps收敛，方程残差
   1.653e-15。保留 `unshifted-rejected-source/` / `bfgs-debug.failed-matrix.pt`。
3. 初始不同矩阵B4含零矩阵，rocSOLVER基线147/B4返回info=[0,0,0,1]，停止该轮。
   最终双方比较同一捕获/随机/简并/对角组；原型零谱/冻结仍在资格检查内。
   保留 `benchmark-baseline-zero-rejected.log`，不隐去失败。

用户补充的 `agy-eigh-test/torch_rocsolver_eigh` 介绍已对照当前源码：

- 可沿用直接 strided-batched、连续行主序的 UPLO 反转、列特征向量恢复、按线程/
  设备管理handle和当前stream的方向；产品已有受检device guard/stream/生命周期。
  原型自行展开LOWER到列主序，不能在已经展开的路径重复使用UPLO反转。
- “N>200全DCU DC”不适用于当前BFGS准入：26.04真实285维高重数H有VMFault及
  info=0错误谱，纯HIP也复现；26.10已测错误谱消失，但真实285维仍有CPU
  `steqr_host` 子块求解。见[独立调查](g2-ase-bfgs-eigh-investigation.md)、
  [26.10报告](g2-ase-bfgs-dtk2610-eigh.md)。不恢复按200维硬切DC的规则。
- 实测小矩阵LAPACK边界为199/200/201，其中n199命中，n200/201未命中；
  `N≤200`只能作参考工程阈值说明，不能推断所有版本、输入、内部子块路径。
- strided-batched是一次批量**库API调用**，内部有多个kernel/迭代；Jacobi还有
  标量收敛读取。`transpose`可为view，`.contiguous()`通常需要显存内复制；
  “一次kernel”“没有任何D2H/H2D”“零开销布局恢复”均不作产品承诺。
- 参考扩展只验API status，未读取每矩阵info；`compute_v=False`的
  `A.contiguous()`还可能alias输入，非连续布局/错误UPLO/自动微分/fake/compile
  也不能因TORCH_LIBRARY注册便认定兼容。thread_local不能代替多卡、多线程测试。
- 参考 benchmark.py 输入是随机对称矩阵，README性能表标注BW200；原子数只是
  3N维度标签，138/276/552与UnitCellFilter147/285/561不同。15–27倍是该工程
  相对Torch的历史结果；本轮下降28.7%–56%相对的是现有rocSOLVER Jacobi。
  不将这些速度/精度推广到BW1000真实BFGS全轨迹。参考demo的clamp不用于ASE算法。
- 单项残差接近1e-15不等于任意输入的特征值或方向均达到机器精度；需要分别检查
  谱误差、方程残差、正交性和ASE逐步轨迹，简并空间内也不要求逐列V符号/基相同。

## 下一步目标

用户随后确认，在新目录保留三轮同配置CSP配对：六次均退出0、收敛且233阶段步/
234模型求值；174次资源观察无竞争，中位数57.308→46.726秒。坐标/晶胞/能量
差异与同后端重复波动单独记录，见[新验收报告](g2-ase-bfgs-m5-2-e2e.md)。
原型/11个产品源及版本锁hash均不变，没有重跑历史M4全功能门。

M5-2隔离原型验收完成，建议保留候选。下一步是[M5-3尺寸与Batch扩展提案](../docs/ASE_BFGS_M5_3_PROPOSAL.md)，
等待用户确认；不能宣布M5整体完成、通用eigh替换或生产CSP吞吐提升。
