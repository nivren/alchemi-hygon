# ASE-compatible BFGS M4：异构 Batch、生命周期与真实 CSP 验收

2026-10-09；用户“继续”确认后实施。分支 `wangleping/feat-ase-bfgs-m4`，HEAD
`cfd22ad11fe0e1704dff48f53abee69135f8db99`，延续未提交 M1/M2/E1/M3。
**M1–M4 授权功能已完成本报告范围内的 CPU/HCU 验收；未提交、集成或推送。**
BFGS 是本地新增功能，非 NVIDIA 已有算子移植或 UBF。产品仍锁定
Framework 0.2.0/Ops 0.4.1，不升级环境或修改 external 工程。

## 实现与接口

- Ops 新增 `bfgs_ase_batch_step`：独立 FP64 State/坐标/力序列，按标量 DOF 分组，
  组内 vectorized Hessian 更新、batched eigh、方向与逐体系 maxstep；全部组通过
  数值检查后才提交。没有零填充全局 Hessian，也不改变 ASE 公式或 abs 谱。
- 既有公共 `ASECompatibleBFGS` / `ASECompatibleBFGSVariableCell` 接受异构 Batch。
  私有 ID 字典维护 Hessian、历史、Filter、冻结与计数；重排保留历史，退出释放，
  新体系从 alpha*I 开始。Filter 默认 cell_factor 为各自 N；参数沿用 ASE。
- 模型共享既有 Batch 求值。FusedStage masked entrance 使用已经共享计算的力，
  不再重复调用模型。沿用 BaseDynamics/Hooks/sampler/sinks 的迁移、预算与补位；
  单体系进入 FusedStage 也验证了共享求值次数。未改通用引擎、FIRE/FIRE2 或默认选择。
- 新 Batch 快照保存全部 ID 的独立 State、Filter、配对几何、冻结标记与参数；
  完整验证后恢复，显式支持整套状态移动设备。载入旧单体系快照后进入 Batch
  也保留配对 ID/坐标/晶胞检查，同时允许新到达体系。旧单体系快照继续兼容。
- 同一数学核心用于固定晶胞 3N 与 ASE UnitCellFilter 3N+9。
  默认 `torch_reference`；显式 `hip` 只替换分解实现，数值张量留在输入设备。
  `None`/warp/auto/未知后端明确失败。Host 只调度 ID、边界与诊断标量。

合同：[Batch](../docs/TORCH_ASE_BFGS_BATCH_CONTRACT.md)、
[数值核心](../docs/TORCH_ASE_BFGS_CONTRACT.md)、
[UnitCellFilter](../docs/TORCH_ASE_UNITCELL_FILTER_CONTRACT.md)、
[HIP eigh](../docs/HIP_BFGS_EIGH_CONTRACT.md)。ASE 3.29.0 的来源 hash 沿用 M1/M3
冻结记录；最新 L-BFGS 仅作架构参考，不代表 BFGS 上游来源。

## 真实 Hessian 暴露的 DC 故障与修正

E1 使用 n<200 Jacobi、n>=200 strided-batched divide-and-conquer（DC）。M4
真实变胞 CSP 的 285-DOF Hessian 在该 DC 路径两次出现 `deflate2_2_kernel` VMFault，
实际退出 **134**。保存矩阵 SHA256：
`b6292e2269fb80370da1ffe4729d8a9c85ab928d8842ecd9621bc4ad2c0e2ded`；
267 个特征值距初始 alpha=70 小于 1e-8。隔离同一矩阵还复现了 **info=0 但
15/285 特征值错误**，最大绝对误差 **79.85904001800401**，退出 **1**。
info 与 finite 检查不足以证明该库调用结果正确。

当前私有 solver 全部宽度直接使用 native Jacobi，absTol=1e-15、maxSweeps=100，
不在失败后切换求解器；没有谱夹紧、阻尼、曲率跳过或降精度。新增设备内
`HV−Vw` 相对 Frobenius 残差及相对 `||I||_F` 的正交检查（各 <=1e-10），
错误分解在坐标/H/历史提交前报错。单体系与 Batch 共用守卫。

| 保存矩阵对照 | 实际退出码 | 结果 |
|---|---|---|
| 旧产品 strided-batched DC | 1 | info=0，谱误差79.859；`fault-native.log` |
| 诊断 Jacobi | 0 | 谱误差2.56e-11，相对重构2.28e-13，绝对正交1.93e-12；`fault-jacobi.json` |
| HCU Torch eigh | 0 | 谱误差1.14e-13，重构1.99e-15，正交2.91e-14；`fault-torch.json` |
| 当前产品 Jacobi | 0 | 与诊断 Jacobi 相同精度；`captured-product-accepted.json` |

故障已定位到本机构建/该 strided-batched DC 路径及输入，未归因为所有 Torch/
rocSOLVER 分解故障。未修改系统库。旧 DC 源码保存在 `frozen-dc-source/`，
诊断输入/日志原样保留；故障探针新增显式 `--method dc --source-root` 用于重放
归档源码，该新重放入口未再次执行。旧 `--method native` 日志在修正前产生；
现在 native 指当前产品 Jacobi，不能用同名参数误指旧实现。

原生测试新增 n200/276/285/552、L/U、alpha 简并+低秩更新共8项。
初次 n552 测试因绝对正交误差1.00276e-11 超过1e-11失败两项；大简并谱专项改用
有尺度定义的相对正交 <=1e-12，原随机谱绝对1e-11保持不变，并保留独立谱/重构
检查。没有放宽全局数值容差。E1/M3 旧 DC 性能只能作为历史证据。

## 环境、命令与测试

Python3.12.13，ASE3.29.0，Torch2.9.0，HCU build `2.9.0+das.opt1.dtk2604`，
HIP6.3.26093，DTK26.04，rocSOLVER3.20.0.80bb870c；physical HCU4 BW1000 64G。
只读资源检查后在空闲单卡顺序运行有界测试；没有多卡占用、系统安装或干预他人任务。
CPU gate 及 HCU OMP/OPENBLAS 各1线程；真实 CPU CSP 各4线程。

| 最终验证 | 实际退出码 | 结果 |
|---|---|---|
| `bash scripts/check_cpu_reference.sh` | 0 | **650 passed / 3 existing skips**：Ops276、fresh LJ1、Framework327、state22、dynamics Ops24；`cpu-gate-final-accepted.log` |
| Ops Batch 专项，纳入 CPU gate | 0（门） | 8项：逐步 ASE、实际等维组、后组失败与错误有限谱原子回滚、alias/ID/输入准入 |
| Framework Batch 专项，纳入 CPU gate | 0（门） | 21项：固定/变胞参数轨迹、冻结、重排/退出/到达、快照、共享求值、原生补位、FIRE2迁移、单体系快照提升 |
| HCU native solver 专项 | 0 | **39 passed**；`hcu-native-accepted.log` |
| HCU Batch + M3 变胞回归 | 0 | **14 passed**，Batch6+单体系8；`hcu-framework-final-accepted.log` |
| 完整 Ops HIP/Torch trace、保存矩阵产品复验、真实 CSP 最终四次运行 | 均0 | 见下文及 acceptance manifest |

3 skip 为既有 Warp 专属项，不新增 skip/xfail。HCU53项是有针对性的 BFGS/solver
门，不能写成完整项目 HCU gate。此前648/3与HCU12次门也通过；最后配对重启
提升修正后重新运行 CPU650/3和HCU14为最终证据。

从项目根激活 `source scripts/activate_hygon_env.sh project`，HCU设置：

```bash
export HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MAX_JOBS=2 PYTORCH_ROCM_ARCH=gfx936
export NVALCHEMI_HIP_BFGS_EIGH_BUILD_DIR="$PWD/artifacts/ase-bfgs-m4-2026-10-09/build"
ulimit -c 0
.venv/bin/python -m pytest -q packages/ops/test/torch/test_bfgs_eigh_hip_hcu.py
.venv/bin/python -m pytest -q packages/framework/test/compatibility/test_ase_bfgs_batch_hcu.py packages/framework/test/compatibility/test_ase_bfgs_cell_hcu.py
.venv/bin/python probes/ase_bfgs_csp.py --device cuda --backend hip --variable --check-steps 3 --max-wall-seconds 300 --output artifacts/ase-bfgs-m4-2026-10-09/csp-hcu-cell-accepted.json
.venv/bin/python probes/ase_bfgs_csp.py --device cuda --backend hip --variable --inflight --check-steps 0 --max-steps 500 --max-wall-seconds 300 --output artifacts/ase-bfgs-m4-2026-10-09/csp-hcu-inflight.json
.venv/bin/python probes/ase_bfgs_csp.py --device cuda --backend hip --check-steps 0 --profile-bfgs-steps 3 --max-wall-seconds 300 --output artifacts/ase-bfgs-m4-2026-10-09/csp-hcu-fixed-profile.json
```

以上 CSP 为最终运行的可重放命令；部分默认参数在原调用中显式传入，执行界限可
更严格。记录文件的实际退出码、命令语义、输入/输出及最终源码 hash 见
`artifacts/ase-bfgs-m4-2026-10-09/acceptance-summary.json`。
最终源码包括随后探针格式整理、profile 参数校验及旧单体系重启提升检查；
该提升分支不用于下面已有 Batch CSP，最终 CPU/HCU门再次覆盖修正后的产品代码。

## 真实分子晶体与数值结果

两个独立 CIF：46/92 原子，真实 FP64 MACE-MPA-0 medium checkpoint，全部 PBC；
输入及模型 SHA256 在每个结果 JSON/manifest。邻居显式 Torch reference、skin=0、
当前晶胞重建；不将 BFGS HIP 的证据扩大为 HIP 邻居/模型加速。
FIRE2 固定晶胞预算20步后进入 BFGS（CPU链预算2步）。UnitCellFilter 使用全部
ASE 默认参数；严格所有广义行 `fmax < .05 eV/Å`，退出 status=2，不是 BFGS预算退出。

| 实际运行 | 退出码 | 收敛与计算次数 | 最大逐步绝对差 |
|---|---|---|---|
| HCU HIP 变胞两结构，`csp-hcu-cell-accepted.json` | 0 | BFGS88/213更新，233工作流步、234主轨迹模型调用；广义fmax .04355185/.04902421 | 各ID前3次更新对独立ASE CPU：r1.24e-14Å、cell1.42e-14Å、H2.55e-11；独立同后端Framework：r5.33e-15、cell2.17e-17、H2.88e-12 |
| HCU HIP 变胞 inflight，`csp-hcu-inflight.json` | 0 | 两个输入各重复一次，4次弛豫；BFGS88/213/88/213，466步/470调用；退出顺序0/2/1/3，sampler耗尽、状态字典释放 | 此次 `--check-steps 0`，无独立数值Oracle；JSON初始零误差字段不代表已比较 |
| CPU fixed配对续算，`csp-cpu-fixed-accepted.json` | 0 | 总工作流152步，BFGS90/150；续算窗口62主轨迹调用；fmax .04889487/.04533296 | 续算ID0检查1更新、ID1检查3更新：ASE r1.78e-14Å、H2.73e-12；独立Framework r2.49e-14、H2.13e-12 |
| HCU HIP fixed及预热profile，`csp-hcu-fixed-profile.json` | 0 | BFGS88/149，169步/170调用；fmax .04217439/.04725471 | 无逐步Oracle，本次验证完整固定晶胞流程并采集3步预热profile |

变胞 inflight 四次广义 fmax 为 .0435518447/.0435518447/.0492018624/.0492164588。
其后两个 raw Cartesian fmax .05313513/.05362665，不应误用 raw-force阈值判为失败：
ASE UnitCellFilter 坐标下原子广义力随参考形变变换，停止语义与 raw-force不同。
晶胞应力最大绝对值约4.69e-4/1.12e-3 eV/Å³；不是另加独立raw stress停止条件。

CPU第一轮1线程超过600秒主动壁钟界限，退出1。第二轮4线程在工作流91步保存
配对checkpoint后主动停止自己的进程，退出143；修正比较夹具后从该checkpoint
续算并严格收敛（退出0）。所以 CPU证据是实际 FIRE2→BFGS→配对重启→收敛链，
不是一次连续完整计时，CPU/HCU FIRE预算也不相同，不能据此计算设备加速比。

一个中间 HCU CSP 已物理收敛但独立比较断言失败：夹具继承 Fused状态1，而
独立单优化器退出状态也为1，导致它没有更新。重置独立状态并在续算时恢复完整
H/历史后，重新运行获得上表的有效对照。保留 `csp-hcu-cell-final.log` 失败记录，
不把该旧运行列为通过；没有为夹具问题修改通用 FusedStage 或放宽数值容差。

## 设备执行与性能证据

完整分组 Ops trace（三步，无 CPU Oracle/模型/Filter）在147/285/147 DOF，实际每步
分解 `[2,147,147]` 与 `[1,285,285]`。DeviceGuard 检查全部数值浮点张量为 HCU FP64。
HIP trace有1542个 syevj kernel、0个deflate kernel，受监测LAPACK调用0；所有
copy API的 D2H 仅1/4字节检查标量，H2D仅1/2/8/16字节参数/索引元数据，大矩阵
copy均kind3设备内拷贝。Torch阳性对照在6/9/6 DOF仍有 host `LAPACKE_dstedc`9次、
`_work`18次，以及40/48/64/72字节D2H、对应H2D数值拷贝。
observer源/marker/trace在manifest；这是被追踪规模的底层数学证据，不是整个
模型工作流无host操作的承诺。HCU Torchreference保留已证实的库内host数学路径。

完整 Ops trace 的重放配置如下（已激活上节环境），marker 使用单独文件名，避免
被 hipprof 的 `<prefix>.json` 覆盖。HIP零调用时observer不生成日志；Torch阳性
对照使用同一个二进制。下列为等价重放命令，不重复占用卡执行：

```bash
export LD_PRELOAD="$PWD/artifacts/ase-bfgs-m3-2026-10-09/unitcell-lapack-trace.so"
export BFGS_LAPACK_LOG="$PWD/artifacts/ase-bfgs-m4-2026-10-09/batch-hip.lapack.log"
hipprof -t -o artifacts/ase-bfgs-m4-2026-10-09/batch-hip .venv/bin/python probes/ase_bfgs_batch_trace.py --backend hip --output artifacts/ase-bfgs-m4-2026-10-09/batch-hip-result.json
export BFGS_LAPACK_LOG="$PWD/artifacts/ase-bfgs-m4-2026-10-09/batch-torch.lapack.log"
hipprof -t -o artifacts/ase-bfgs-m4-2026-10-09/batch-torch .venv/bin/python probes/ase_bfgs_batch_trace.py --backend torch_reference --small --output artifacts/ase-bfgs-m4-2026-10-09/batch-torch-result.json
unset LD_PRELOAD BFGS_LAPACK_LOG
```

| 计时范围 | 实际结果 | 解释 |
|---|---|---|
| HCU 变胞inflight，4次成功弛豫 | 总103.371s；模型21.319s；BFGS更新60.495s/426窗口，中位.132889s | 无Oracle/profiler；同步分项计时，含冷初始化和调度，不是纯无仪器benchmark；约25.84s/成功结构仅适用此输入/配置 |
| HCU 变胞两结构及独立比较 | 总91.416s；主模型14.068s；更新30.688s | 总时间包含ASE/独立运行，不能作为生产吞吐或与inflight直接比较 |
| CPU fixed续算窗口 | 总239.930s；模型199.216s；更新.847s | 只覆盖62次调用的续算窗口，不是完整CPU工作流耗时 |
| HCU fixed有界预热profile | 3步/6分解：eigh device累计308.218ms；native API CPU308.970ms；H更新device7.786ms | 实际模型工作流预热后窗口，两个不同DOF各B1；设备累计与host inclusive不能相加当wall |
| 同一profile全窗口事件 | `_local_scalar_dense`527次/CPU47.603ms；clone285次/CPU8.248ms | 包含模型/neighbor/Framework，不能全部归因于BFGS；需更细trace确认 |

独立 Ops profiler 的首次 bmm包含约3.7秒库冷初始化，不能作为稳态瓶颈证据。
真实预热窗口已另存 `csp-hcu-fixed-profile.profile.json` 与摘要。当前结果显示BFGS
成本值得后续优化，但未建立匹配Torch/其他设备或旧DC端到端对照，不宣称加速收益。

## 剩余范围与下一阶段提案

本报告只验证两个独立真实CIF、四次inflight弛豫及小型解析异构Batch；不是全CSP
数据集成功率、大Batch规模/吞吐保证。FP64 eager/forward-only；无constraints、
其他ASE Filters、DomainParallel、NVIDIA硬件、其他HCU构建、多卡/线程压力或
optimizer autograd/compile门。完整模型/数据源/采样游标/Fused调度checkpoint仍由
调用方保存；本地配对State续算不等同于通用服务checkpoint或ASE JSON互读。
稠密H和scratch成本为O(d²)，eigh为主要瓶颈；尚无M5优化或auto准入。

建议下一阶段 **M5-1：eigh稳态性能试验与方案选择**，等待用户确认。用真实保存的
Hessian与147/285 DOF等维组做相同输入/精度、显式同步、独立冷/热计时；按组规模
采集solver、guard、copy/标量同步与模型调度成本，评估安全的库调用候选。候选先过
简并谱/残差/正交/ASE逐步/原子失败/stream门，再做相同CSP端到端比较；没有收益
则保留当前实现。Torchreference和ASE数学保留，不自动重新启用已知故障DC、修改
算法、整体升级依赖、推送或合并。M5为可选阶段，当前工作在M4收口后暂停。
