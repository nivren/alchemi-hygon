# ASE-compatible BFGS M2：Framework 接入与 HCU eigh 实测

日期：2026-10-09。用户已确认 M2；分支 `wangleping/feat-ase-bfgs-m2`，从 M1
分支延续未提交工作，Git HEAD 仍为 develop 起点
`cfd22ad11fe0e1704dff48f53abee69135f8db99`。M1/M2 尚未提交、合并或推送。
M2 开始前14份相关文件备份于 `artifacts/ase-bfgs-m2-2026-10-09/m1-snapshot/`，
M1 的实现身份和原始验收仍保存在原 acceptance-summary/source hashes 中。

## 范围、来源与实现

这是 NVIDIA 审阅基线没有的**本地新增功能**，不是 BFGS 移植或 UBF。产品锁不变：
Framework 0.2.0 `4dfe3723def34df3fadb245981081ccf8c94c257`；Ops 0.4.1
`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。架构参考冻结 rc 的 L-BFGS 分层，
数值独立对齐 ASE3.29.0。来源和 ASE 文件哈希继承
[合同](../docs/TORCH_ASE_BFGS_CONTRACT.md)，本地补丁登记 LP-088；不升级环境。

- 新增 `nvalchemi.dynamics.ASECompatibleBFGS` 和 optimizers lazy export，默认
  alpha70/maxstep.2/fmax.05/显式 torch_reference；无 Warp 或 ASE 运行依赖。
- 使用已有 BaseDynamics 模型、实际更新的 Hooks 和 run 生命周期。首次活动入口
  经 compute Hooks 求力，严格检查 `< fmax`，已收敛初始结构不更新。更新在
  pre_update，随后新坐标模型求值；每个普通成功更新增加一次求值。
- Framework 只解析一次 BackendSelection，prepare/step 委托 M1 Ops。M1 executor
  只新增 hessian_update/eigh profiler scopes，数学、失败检查和 commit 顺序不改。
  未修改公共 BaseDynamics、FIRE/FIRE2、通用 registry 或 default/None 选择语义。
- 稠密 Ops State 独立于 Base 的 system-only Batch State；False mask/退出体系
  在入口跳过数学，冻结 H、历史、initialized 和 Ops step_count。Base step_count
  计调度访问，Ops 计实际更新，两个计数独立。测试覆盖四个入口。
- 本地 tensor restart 验证配套 Batch 坐标、system_id/原子身份/cell/pbc 与 dtype/device，
  恢复后重新求当前模型力。默认/custom 收敛模式必须一致，custom hook/模型/Batch
  由调用方另行保存重建。不是 ASE JSON；拒绝 alias，避免初始模型输出覆盖坐标。
- 单体系固定晶胞 FP64；多体系、sampler/refill、跨 rank 明确失败。未声称完整
  FusedStage、多体系状态重排、Filter、CSP、训练或 compile 已支持。

## 环境、输入和计时方法

Python3.12，ASE3.29.0，Torch2.9.0（HCU版 `2.9.0+das.opt1.dtk2604`，构建
`4e3c1e79303a0774d61d4d7efa36c8f27dac7dc6`），HIP6.3.26093，DTK26.04，
rocSOLVER3.20.0。HCU 资源只读检查无 KFD 作业，单卡选物理 HCU4/BW1000 64G，
`HIP_VISIBLE_DEVICES=4`；没有占用其他卡、升级软件或修改外部工程。
CPU 使用 `HIP_VISIBLE_DEVICES=''`，OMP/OPENBLAS 线程数1。所有原始证据位于
`artifacts/ase-bfgs-m2-2026-10-09/`；退出码存 `.exit`，失败尝试日志另名保留。

真实模型为已验证 MACE-MPA-0 medium checkpoint，SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
FP64/default dtype FP64，eval，active_outputs={energy,forces}，无 cueq/compile。
三原子无 PBC 水分子，初始 `[0,0,0],[1.06,0,0],[-.29,.88,0]`，固定10Å盒。
Torch reference neighbors、skin0、BEFORE_COMPUTE 重建；独立 ASE MACECalculator
在 CPU 构建自己的邻居并使用相同 checkpoint，逐帧比较，两条轨迹独立演进。

eigh 使用 CPU seed24+n+B 生成对称随机 FP64 **不定矩阵**，完整特征向量；维度
138/199/200/201/276/552，各 B=1/4/16。首次调用单列，预热1次，steady3次，
每次同步后计 wall clock，取中位数；包括分配、输入复制、输出布局转换及 info 检查。
18组均做 CPU eigval oracle、重构/正交/输入不变检查。

参考工程为用户提供的 `/home/wangleping/codes/agy-eigh-test/torch_rocsolver_eigh`。
其现存扩展是 CPython3.10，当前3.12不直接导入；M2 通过独立 ctypes/现有 DTK C ABI
评估同一 Jacobi/DC 策略。没有复制、修改或安装该工程，也没有构建/注册产品后端。
原始 C++ SHA256 由 eigensolver JSON 记录。探针额外检查 API status/solver info，
原工程的默认路径不检查 info；原 demo 的 clamp 不是 ASE 数学，未使用。

## CPU gate 与数值验收

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 bash scripts/check_cpu_reference.sh
HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_ase_bfgs.py
HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python \
  probes/ase_bfgs_framework_analytic.py --device cpu \
  --output artifacts/ase-bfgs-m2-2026-10-09/cpu-framework-analytic.json
```

完整 gate `cpu-reference-final.log/.exit` 退出0：

| 进程 | 结果 |
|---|---|
| Ops（含M1 BFGS68） | 224 passed，3 skipped |
| fresh-process LJ import | 1 passed |
| Framework compatibility 等（含M2 BFGS39） | 261 passed，1 deselected |
| 原 state management 选定项 | 22 passed |
| 原 dynamics Ops 选定项 | 24 passed |
| 合计 | **532 passed，3 skipped** |

3个 skip 是既有 Warp NHC 对照，不是新增 BFGS skip。之后审阅补强初始 output
alias 拒绝和2项测试，最终 `cpu-bfgs-reviewed.log/.exit` **41 passed**、退出0。
不能将这2项计入此前532；最终文件 Ruff、compileall、C probe语法、diff check通过。
完整包 CI、Sphinx build/interrogate、NVIDIA 原测试全集未运行。

专项包括三规模二次/四次势8步 ASE H/坐标/实际方向，初始力范数0/.049/.05/.051
相对阈值fmax=.05的边界，
整体停止步数、compute-before-update、每更新求值次数、Hooks 时序/enter/exit，
四入口历史冻结、原子/晶胞/身份变化、缓存选择、重启/错误原子性、公开 lazy import，
外部坐标原位/替换修改后重求力、custom stopping mode 和模型输出 alias 错误。

独立解析 Framework 探针 CPU/HCU4：3/9/36 DOF 各8步、主轨迹9次求值、
重启后4步**位相等**；False mask 和 DeviceGuard 通过。CPU 最大 H/坐标/原始方向
差为1.56e-13/3.89e-16/3.16e-15；HCU为1.56e-13/7.22e-16/1.89e-15。
guard只观察 Torch 张量输出，不能证明库内无主机数学。

## 真实 MLIP CPU/HCU 结果

```bash
HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ase-bfgs-m2-mpl timeout 120 .venv/bin/python \
  probes/ase_bfgs_framework.py --device cpu \
  --checkpoint /home/wangleping/.cache/mace/mace-mpa-0-medium.model \
  --require-convergence --profile \
  --output artifacts/ase-bfgs-m2-2026-10-09/cpu-mace-final.json
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ase-bfgs-m2-mpl timeout 180 .venv/bin/python \
  probes/ase_bfgs_framework.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/mace-mpa-0-medium.model \
  --require-convergence \
  --output artifacts/ase-bfgs-m2-2026-10-09/hcu4-mace-final.json
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 120 \
  .venv/bin/python probes/ase_bfgs_framework_analytic.py --device cuda \
  --output artifacts/ase-bfgs-m2-2026-10-09/hcu4-framework-analytic.json
```

两设备均退出0，**11次更新/12次主轨迹模型求值**，与 ASE 停止一致，cell 不变。
重启从第2步开始，另比较9次续跑，不计入主轨迹12次求值。

| 全轨迹最大绝对差（独立CPU ASE oracle） | CPU | HCU4最终重验 |
|---|---|---|
| 坐标 Å | 1.33e-15 | 2.10e-15 |
| 力 eV/Å | 4.55e-13 | 4.69e-13 |
| 能量 eV | 6.39e-14 | 5.68e-14 |
| Hessian | 2.88e-12 | 2.23e-12 |

CPU真实模型 restart 位相等。最初 HCU 位相等断言失败（坐标2.22e-16）；随后
同几何同模型重求力实测差2.22e-15、能量差0，表明模型求值有舍入变化。最终HCU
restart最大坐标6.66e-16、Hessian2.85e-12；按坐标/力/能量2e-9、Hessian2e-7的
独立轨迹容差记录，非全局放宽容差。解析势HCU restart仍要求位相等并通过。

普通执行排除首个样本的同步 wall 中位数：CPU 模型200.38ms、update3.00ms、
完整step205.41ms；最终HCU模型28.86ms、update4.85ms、step37.35ms、模型末端
同步0.125ms（在模型时间内，不另加）。各阶段中位数不可相加推导整体中位数。

另一次预热后3步 profiler 的 **user_annotation wall 平均**：CPU H更新.854ms、
eigh.337ms、模型214.69ms、完整step221.94ms；HCU H更新1.392ms、eigh.784ms、
模型49.23ms、完整step62.30ms。profiler 扰动明显；user范围重叠且包含检查/同步，
不是纯GPU kernel耗时，不能相加。vendor key_averages 对user范围的CPU/GPU分类
不用于数学驻留结论；实际 Chrome trace为准。当前水分子主要耗时在模型，不能外推CSP。

## eigh 性能与低层事实

```bash
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 180 \
  .venv/bin/python probes/ase_bfgs_eigh.py --sizes 138 199 200 201 \
  --output artifacts/ase-bfgs-m2-2026-10-09/hcu4-eigh-small.json
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 180 \
  .venv/bin/python probes/ase_bfgs_eigh.py --sizes 276 552 \
  --output artifacts/ase-bfgs-m2-2026-10-09/hcu4-eigh-large.json
```

两窗口退出0。API成功，全部info=0；Jacobi最多10 sweeps，残差和sweeps原值保存在
JSON。最大eigval绝对差1.07e-12、重构相对残差7.37e-14、正交绝对差5.64e-14。

| n | B | Torch ms | native策略 ms | 分解单项比值 |
|---|---|---|---|---|
| 138 | 1 | 4.855 | 19.641 | 0.25x |
| 138 | 4 | 19.001 | 22.064 | 0.86x |
| 138 | 16 | 75.618 | 27.254 | 2.77x |
| 199 | 16 | 135.485 | 69.326 | 1.95x |
| 200 | 1 | 7.317 | 30.217 | 0.24x |
| 200 | 16 | 115.344 | 70.963 | 1.63x |
| 201 | 1 | 7.034 | 7.110 | 0.99x |
| 201 | 16 | 109.479 | 45.247 | 2.42x |
| 276 | 1 | 8.995 | 9.010 | 1.00x |
| 276 | 16 | 141.613 | 54.670 | 2.59x |
| 552 | 1 | 17.363 | 17.443 | 1.00x |
| 552 | 4 | 68.766 | 31.128 | 2.21x |
| 552 | 16 | 274.027 | 88.450 | 3.10x |

其余行完整记录在JSON。原工程旧表是另一环境的历史测量，不能作为当前收益；当前
小矩阵 B=1 的 Jacobi明显更慢，DC单矩阵基本相等，收益主要来自批量减少逐矩阵调度。
不代表本仓库已支持 batched BFGS 或实际CSP端到端加速。

追踪使用现有 hipprof `--hip-trace --trace-args --output-type 0`，配合独立 C
LD_PRELOAD observer 只记录原 LAPACKE_dstedc/_work 的实际调用与wall时间，不
修改输入/输出。C observer以 `cc -shared -fPIC -O2 ... -ldl` 生成到 artifacts，
无环境安装。执行90s窗口：Torch n138/201/552 B4；Torch n199/200/201 B1；
native hybrid n138/199/200/201/552 B4。3个 target JSON status=passed、profiler
exit0；同时检查 target日志，不仅根据 profiler退出码。

- Torch n138 B4：4次 LAPACKE_dstedc，实际host调用约1.93–2.37ms/体系。
  HIP有每体系1104/1096 bytes三对角数据D2H，1104 bytes特征值及152352 bytes
  (`138²*8`)特征向量H2D。不是纯info同步，证实库内主机数学。
- Torch边界：n199有host调用；n200/201没有受监测调用，HIP也没有相应矩阵回传。
  所以不能照搬参考工程 `n<=200`；200本身不应据旧注释判为host路径。
- native hybrid：在所测窗口未观察到上述主机调用，D2H只有info/布尔标量（4/1byte）；
  大块H2D是探针CPU输入初始化，不能当作solver回退。Jacobi/DC具体kernel活动和
  D2D布局转换在HIP trace中。未证明所有规模、线程/stream和其他平台完整路径。

原始 `hip-*.json/.db/.hiptrace.csv/.hipkernel.csv`、`lapack-*.log`、
`trace-summary.json` 保存证据。Torch tensor guard通过与host LAPACK发现并不矛盾。
M2 当前 reference 只能验收数值集成，**不能验收全部数学在HCU运行**。

## 剩余限制与下一阶段提案

首批失败为探针cell/pbc缺失、C ABI枚举常量和真实GPU模型位相等假设，已分别修正
输入、按当前头文件核实常量、记录同几何求值误差。全部失败日志保留；没有修改生产
邻居/MLIP数学、放宽全局容差、忽略info或修改ASE公式。CPU profiler有设备不可见
诊断，HCU Kineto有duplicate flow警告，均与target成功记录分开保存。

未验证：Filter/应力联合优化、异构Batch/完整Fused共享求值/动态补位、真实分子晶体
CSP、NVIDIA硬件、DomainParallel、compile/autograd-through-optimizer、生产性能。
外部原扩展的layout/alias/handle/stream/Autograd注册/solver-info等问题仍须E1审查，
隔离probe不证明该原包可生产使用。完整feature保持planned，M1/M2窄范围单列。

**建议下一阶段E1，待用户确认**：在BFGS自身边界内实现显式HCU rocSOLVER求解器，
保留Torch reference，只替换分解，不改Hessian/abs谱/maxstep。合同包含FP64完整
eig系统、单矩阵与等维batch、UPLO/layout/input不变、device guard/current stream/
handle生命周期、API/solver info及非收敛错误；选小矩阵Jacobi或DC时以当前低层
证据为准。验收ASE逐步H/方向/位置/restart、正交/重构/特征值、边界/退化谱、
非默认stream及目标HCU无主机数学/矩阵往返；逐尺寸/B记录稳态成本，不承诺单矩阵
加速、不启用自动默认选择。E1完成后再提出M3 UnitCellFilter目标，最终仍交付M4。
这项原生求解适配与可选M5端到端性能优化区别记录，不因此扩大为通用线性代数框架。
