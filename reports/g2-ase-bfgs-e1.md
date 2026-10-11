# ASE-compatible BFGS E1：显式 HCU rocSOLVER 求解器

2026-10-09；用户已确认 E1 后实施，阶段已完成当前构建/单卡窄验收。
分支 `wangleping/feat-ase-bfgs-e1`，HEAD仍为
`cfd22ad11fe0e1704dff48f53abee69135f8db99`，延续未提交 M1/M2；未提交、合并或推送。
原始输出在 `artifacts/ase-bfgs-e1-2026-10-09/`，源码/doc SHA256、实际退出码及
验收汇总在 acceptance-summary；M2源码快照和数学AST对照另存。

## 来源与实施

本地新增功能，非上游 NVIDIA BFGS 移植或 UBF。产品锁不变：Framework0.2.0
`4dfe3723def34df3fadb245981081ccf8c94c257`，Ops0.4.1
`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`；只以冻结rc L-BFGS作架构参考，
ASE3.29.0作CPU逐步数学Oracle，来源SHA与独立实现声明见[ASE合同](../docs/TORCH_ASE_BFGS_CONTRACT.md)。
E1记为本地补丁LP-089，与可选M5 CSP端到端优化分开。

外部参考路径 `/home/wangleping/codes/agy-eigh-test/torch_rocsolver_eigh` 只读，
审阅未找到许可证声明；未复制、修改、安装或加载其Python3.10二进制。
本轮C++/HIP独立表达，使用当前安装rocSOLVER头文件的公开API。

- 新增BFGS私有 `_bfgs_eigh_hip.py`、`_native/bfgs_eigh.cpp/.cu` 和
  `bfgs_ase_hip.py`；显式 `hip.bfgs-ase-rocsolver-v1` 在既有metadata catalog登记。
  Ops dispatcher、generic executor和Framework优化器源码无需增加ID分支。
- 原Torch step body移入私有 `_step_with_eigh`，两个executor共用。将M2 callee
  `torch.linalg.eigh`归一成注入的`eigh`后，AST数学body完全相同。H更新、abs谱、
  maxstep、坐标/历史commit、奇异/非有限错误不变；Torch永久reference保留为默认。
- FP64 `[n,n]`/等维`[B,n,n]`完整谱，严格L/U、连续独立输出、指定三角检查、输入
  不变；自定义op声明无输入mutation，fake/schema通过，requires_grad明确拒绝。
- C++device guard、checked current stream、每线程/设备独立RAII rocBLAS handle；
  workspace由库管理，输出/临时数组依Torch当前stream管理。GPU info检查保证同步
  完成后消费输出。正常API每次检查，进程/线程退出的RAII cleanup不抛异常。
- Jacobi n<200（absTol1e-15/maxSweeps100），DC n>=200；这是本机追踪后的规则，
  不是外部n<=200规则。API失败、info非零或非有限输出直接失败，不重试/换算法。
- 初期严格guard HCU Torch2.9.0+das.opt1.dtk2604、HIP6.3.26093、rocSOLVER
  3.20.0.80bb870c；其他构建必须重新验收，cuda标签不足以准入NVIDIA。
- 既有lazy HIP loader只增加可选link flags连接rocSOLVER/rocBLAS；其他caller
  空默认不增加链接项。导入/default Torch不编译；显式HIP首次调用JIT，可指定
  `NVALCHEMI_HIP_BFGS_EIGH_BUILD_DIR`，缺库/编译失败不fallback。

公共BaseDynamics、Backend Registry通用逻辑、FIRE/FIRE2、上游版本锁与系统
Torch/DTK均未改动。合同见[HIP-BFGS-EIGH](../docs/HIP_BFGS_EIGH_CONTRACT.md)。

## 环境与命令

Python3.12.13，Torch2.9.0，HCU版本/库版本见guard；DTK26.04。
运行前只读hy-smi确认无KFD进程、各卡显存0%；限时单卡选物理HCU4/BW1000 64G。
OMP/OPENBLAS各1线程，MAX_JOBS=2、gfx936编译；各HCU作业顺序执行。
结束后资源检查同样无KFD进程、显存0%，不占多卡、不安装/升级环境。

以下从repo根运行，启动环境采用zsh项目模式：

```sh
source scripts/activate_hygon_env.sh project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export PYTORCH_ROCM_ARCH=gfx936
export NVALCHEMI_HIP_BFGS_EIGH_BUILD_DIR="$PWD/artifacts/ase-bfgs-e1-2026-10-09/build"
HIP_VISIBLE_DEVICES='' bash scripts/check_cpu_reference.sh
HIP_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q packages/ops/test/torch/test_backend_registry.py
HIP_VISIBLE_DEVICES=4 timeout 90 .venv/bin/python -m pytest -q packages/ops/test/torch/test_bfgs_eigh_hip_hcu.py
HIP_VISIBLE_DEVICES=4 timeout 90 .venv/bin/python probes/ase_bfgs_framework_analytic.py \
  --device cuda --backend hip --output artifacts/ase-bfgs-e1-2026-10-09/hcu-analytic.json
HIP_VISIBLE_DEVICES=4 timeout 120 .venv/bin/python probes/ase_bfgs_framework.py \
  --device cuda --backend hip --checkpoint /home/wangleping/.cache/mace/mace-mpa-0-medium.model \
  --steps 50 --require-convergence --profile --output artifacts/ase-bfgs-e1-2026-10-09/hcu-mpa0.json
HIP_VISIBLE_DEVICES=4 timeout 120 .venv/bin/python probes/ase_bfgs_eigh_native.py \
  --output artifacts/ase-bfgs-e1-2026-10-09/hcu-eigh-performance.json
```

首次native测试限时240s且verbose=1，以覆盖编译；后续native测试90s。
实际命令日志/exit文件保留，最终以上各target均退出0。Focused Ruff check/format、
compileall、git diff --check通过；全包CI/Sphinx/interrogate/NVIDIA未运行。

## 测试与数值结果

| 验收 | 结果/退出码 |
|---|---|
| CPU reference gate | 550 passed、3既有Warp NHC skips，退出0 |
| CPU分组 | Ops240/3skip、fresh LJ1、Framework263、state22、ops24 |
| E1 CPU合同/既有BFGS/registry专项 | 102 passed，退出0；其中新合同16项 |
| 最终catalog lazy import断言补验收 | registry18 passed，退出0 |
| 原生HCU专项 | 31 passed，退出0 |
| HIP Framework解析势/MPA-0/solver18组 | 均退出0 |
| native/Torch/完整BFGS trace | target打印passed、hipprof退出0 |

HCU专项覆盖single/batch、n1/3/9/36/138/199/200/201/552，L/U下非指定三角NaN，
输入不变、负/重复/精确零/1e-18谱、非默认stream、当前逻辑device恢复、schema/fake
opcheck、非有限输入、零谱BFGS拒绝、API错误传播与实际非收敛状态原子性。
只有单可见设备：不同物理设备guard压力和多线程未验收。API错误路径通过注入
测试，不伪称制造真实库API故障；非收敛则确实调用native Jacobi并得到info=1。

n138/maxSweeps1的原始诊断：info=[1]、sweeps=[1]、residual=[60.03387209786284]。
checked入口抛`BFGSNumericalError`，positions/H/previous q/f/initialized/count冻结。
精确零谱明确失败；近零谱保留，不夹紧。成功18组info全部0，Jacobi sweep<=10。

解析势DOF3/9/36各8次更新/9次主轨迹求值，与ASE3.29.0逐步最大误差：q3.47e-16、
H6.40e-14、raw方向2.23e-15；四次restart continuation逐位相同，False mask冻结
和TorchDispatch设备/FP64检查通过。该guard与底层native trace结合使用。

真实FP64 MPA-0水分子N3/DOF9，与独立CPU ASE MACECalculator邻居/求值对照：
11次更新/12次主轨迹模型求值达fmax=.05，cell不变。checkpoint SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
最大误差q1.56e-15Å、F1.28e-13eV/Å、E1.43e-14eV、H1.99e-12eV/Å²。
重启9次continuation：q5.56e-16Å、H1.48e-12；同几何重算力3.56e-15eV/Å、E0。
GPU MACE atomic reduction有舍入差，使用M2既定逐量容差，不声明真实MLIP位相等。

## 库内执行证据

使用本项目纯observer `ase_bfgs_lapack_trace.c` 拦截并转调
`LAPACKE_dstedc`/`_work`，结合hipprof HIP参数trace；不改数学。
工作目录为E1 artifacts，避免在repo根散落profiler输出：

```sh
cc -shared -fPIC -O2 probes/ase_bfgs_lapack_trace.c -ldl \
  -o artifacts/ase-bfgs-e1-2026-10-09/lapack-observer.so
# 先设上述build环境，再cd artifacts/ase-bfgs-e1-2026-10-09；路径均使用绝对路径。
# 以下ROOT表示repo绝对路径，仅用于复现，不改HOME/CODEX_HOME。
export BFGS_TRACE_ROOT=/home/wangleping/codes/alchemi-hygon
HIP_VISIBLE_DEVICES=4 LD_PRELOAD="$BFGS_TRACE_ROOT/artifacts/ase-bfgs-e1-2026-10-09/lapack-observer.so" \
 BFGS_LAPACK_LOG="$BFGS_TRACE_ROOT/artifacts/ase-bfgs-e1-2026-10-09/lapack-native.log" \
 timeout 90 /opt/dtk-26.04/bin/hipprof --hip-trace --trace-args --output-type 0 -o hip-native \
 "$BFGS_TRACE_ROOT/.venv/bin/python" "$BFGS_TRACE_ROOT/probes/ase_bfgs_eigh_native.py" \
 --sizes 138 199 200 201 552 --batches 4 --trace-only --solver hip --output native-trace-result.json
```

同样命令分别改prefix/log/result为torch且`--solver torch`，构成observer阳性对照；
完整BFGS改为prefix/log/result=bfgs，`--solver hip --sizes 9 138 201 552 --batches 1
--trace-only --bfgs-steps 3`。三者target passed与profiler退出码分开确认，原始JSON/db/
HIP kernel/copy表保留。trace-summary.json按HIP参数的kind/sizeBytes统计。

- Torch B4：n138与199各4次实际host `LAPACKE_dstedc`；工作函数/查询不重复计数。
  n138 D2H D1104/E1096字节，特征向量H2D152352字节；n199 D1592/E1584、
  向量316808字节。n200/201/552未观察受监测host调用。
- 产品native B4同五规模：受监测LAPACK0，D2H仅15次1字节与44次4字节。
  609408/1267232/1280000/1292832/9750528字节H2D各1次为CPU随机输入准备，
  没有对应矩阵D2H/向量回传；32/64/128字节H2D为DC库索引元数据。
- 完整HIP BFGS DOF9/138/201/552各3步：受监测LAPACK0，D2H204次1字节和30次
  4字节；H2D64/128字节各4次仅库索引元数据，无q/H/向量回传。H更新、dot/GEMM/
  谱/方向/坐标路径对应GPU kernel。不是无host调度/无同步实现。

结论仅限当前构建与所列输入/符号，不能从有限trace证明所有规模或其他版本库内
行为。显式HIP当前范围实现了HCU数学执行；Torch reference的host限制仍保留。

## 性能

产品接口计时包含clone/分配/layout、有限性及info检查/同步；预热1次、同步wall
3次取median，JIT/load/首调用另外记录。没有使用M2 ctypes时间冒充产品时间，
没有采用外部旧环境10–27倍结论。缓存extension load本次930.53ms，首次native
编译包含在首次HCU测试42.51s中，未单独隔离，不当作纯编译时间。

| n | B | Torch ms | 原生产品 ms | Torch/原生 |
|---|---|---|---|---|
| 138 | 1 | 4.882 | 19.976 | 0.24 |
| 138 | 4 | 19.094 | 22.452 | 0.85 |
| 138 | 16 | 75.881 | 27.688 | 2.74 |
| 199 | 1 | 8.637 | 28.190 | 0.31 |
| 199 | 4 | 34.033 | 35.716 | 0.95 |
| 199 | 16 | 135.514 | 69.783 | 1.94 |
| 200 | 1 | 7.444 | 7.919 | 0.94 |
| 200 | 4 | 29.377 | 16.763 | 1.75 |
| 200 | 16 | 117.119 | 53.152 | 2.20 |
| 201 | 1 | 7.194 | 7.647 | 0.94 |
| 201 | 4 | 28.170 | 15.889 | 1.77 |
| 201 | 16 | 111.836 | 48.437 | 2.31 |
| 276 | 1 | 9.105 | 9.503 | 0.96 |
| 276 | 4 | 36.080 | 19.381 | 1.86 |
| 276 | 16 | 143.929 | 57.815 | 2.49 |
| 552 | 1 | 17.542 | 17.952 | 0.98 |
| 552 | 4 | 69.392 | 32.329 | 2.15 |
| 552 | 16 | 276.629 | 92.120 | 3.00 |

18组最大谱差9.99e-13、相对重构7.37e-14、正交范数4.89e-13，全部在合同界限内。
B1 Jacobi更慢；DC B1接近Torch但略有接口检查开销；B16收益1.94–3.00倍。
仅为等维eigh，不代表异构BFGS或CSP加速，默认不变/auto不准入。

真实水分子排除首样本的同步median：MLIP28.06ms、update5.06ms、完整step36.87ms、
terminal sync.139ms。M2同工作负载step37.35ms仅作历史参考，无配对性能实验，不据此
声称端到端加速。独立3步Kineto trace的user_annotation wall范围每步Hupdate1.442ms、
eigh1.153ms、model50.019ms、step63.914ms；这是受profiler影响的inclusive范围，
不是纯GPU kernel时间，不累加或与未profile计时直接比较。

## 失败记录、边界和下一阶段

首次CPU fake CUDA夹具在无可见设备时因view/Autograd触发真实HIP初始化，退出134；
随后view夹具退出1。改为直接构造fake元数据、CPU检查requires_grad，最终102项
通过，未放宽数值容差或跳过测试。首次native31项中小9x9一次sweep未制造失败，
30通过/1失败退出1；改用n138，实际info=1后最终31通过。全部日志保留。
Kineto仍有duplicate flow诊断，与成功target/退出码分别记录。

已完成的是E1当前构建/单HCU窄范围；完整BFGS功能仍planned。未交付Filter/应力、
异构BFGS/Fused共享求值/refill/CSP；未验收NVIDIA/其他HCU-ROCm构建、多卡/线程
压力、compile/Autograd-through-optimizer/训练、生产性能。Native邻居核未修改，
本轮未重复其专用HCU性能/全套回归，CPU现有相关gate通过。

下一阶段提议M3：单体系ASE UnitCellFilter 3N+9，mask/cell_factor/hydrostatic_strain/
constant_volume/orig_cell/scalar_pressure与ASE3.29签名/默认/组合语义一致；独立
pack/unpack与广义力层接同一BFGS，保存参考晶胞/参数及广义历史，复用现有模型
stress/Hooks/neighbor重建。验收ASE逐步q/H/方向/原子位置/cell/广义力/restart，
保守情形有限差分、投影情形ASE行为对照，CPUgate+限时HCU+真实周期小体系联合
收敛。等待用户确认后实施M3；M4异构Batch/refill/CSP仍必交付，M5可选。
