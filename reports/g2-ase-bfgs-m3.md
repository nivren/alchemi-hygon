# ASE-compatible BFGS M3：单体系 UnitCellFilter 验收

2026-10-09；用户确认后实施。分支 `wangleping/feat-ase-bfgs-m3`，HEAD
`cfd22ad11fe0e1704dff48f53abee69135f8db99`，延续未提交 M1/M2/E1。
**M3 单体系 CPU/HCU 窄范围通过；完整特性仍 planned，M4 未开发。**
未提交/合并/推送，不升级 Torch/DTK/上游产品锁；本地新增功能，非 NVIDIA BFGS 移植或 UBF。

## 实现与来源

- 新 Ops `ASEUnitCellFilter`：完整广义坐标 pack/unpack、应力转广义力、投影应力、
  E+pV、参考晶胞/参数快照。独立编写，ASE 仅作为测试 Oracle，不增加产品 ASE 依赖。
- 新公共 `ASECompatibleBFGSVariableCell`：继承现有单体系 BFGS 的 BaseDynamics、
  模型/Hooks/状态生命周期；全部 ASE Filter 参数与默认值、N+3 广义行严格 fmax、
  当前晶胞邻居重建、配套原子/晶胞/Filter restart、退出及 False mask 全历史冻结。
- 同一 BFGS 数值 body 和 Torch/HIP executor。Cartesian 快照 schema1 原样保留，
  UnitCell schema2 增加 coordinate_mode；selection 按模式检查能力，不静默换后端/精度。
  M3 scratch State 校验候选晶胞后才提交，新增 O(d²) 设备内复制成本；数学不变。
- `math-preservation.json`：`_step_with_eigh` AST 与 E1 相同，三个原生 solver 源文件
  SHA256 与 E1 相同。冻结产品 Framework0.2.0/Ops0.4.1 来源锁未修改。

ASE3.29.0 `filters.py`、`_4/optimize/cellutil.py` 和 `stress.py` 是数值 Oracle；
来源/hash 见[主合同](../docs/TORCH_ASE_BFGS_CONTRACT.md)及 M3 acceptance manifest。
架构比较使用本地冻结 Ops0.5.0-rc `5fffc0726b1e71bb8b68e10b5f87942064759f5b`，
三个 L-BFGS/cell_filter 源文件与 candidate-manifest 的 SHA256 逐项吻合，证据
`upstream-lbfgs-chart-source.json`。external clone 当前无该 Git object，使用已冻结
candidate 文件，未 fetch/修改 external 或把候选包安装成产品。Framework candidate
是其他 SHA，未把其细节错误归因于历史 0.3.0-rc SHA。

| 表示 | 晶胞坐标 | 广义行数 | 标量坐标数 |
|---|---|---|---|
| 冻结上游变胞 L-BFGS | 对齐后的形变矩阵六个三角分量 | N+2 | 3N+6 |
| ASE UnitCellFilter | 相对参考晶胞的完整 3×3 形变梯度 | N+3 | 3N+9 |

九个矩阵项不是九个独立应变。坐标、力和 maxstep 范数不同，不能补零或复用 native
对齐路径声称数值等价。M3 不加入 AlignCellHook，保留非三角形和左手晶胞语义。
完整接口、单位、异常、mutation/stream/梯度边界见
[UnitCellFilter 合同](../docs/TORCH_ASE_UNITCELL_FILTER_CONTRACT.md)。

## 环境与实际测试结果

ASE3.29.0、Python3.12.13、Torch2.9.0/HCU `2.9.0+das.opt1.dtk2604`、
HIP6.3.26093、DTK26.04、rocSOLVER3.20.0.80bb870c，OMP/OPENBLAS各1线程。
HCU 前只读 hy-smi 确认所有卡无 KFD PID，physical4 BW1000 64G 空闲，单卡限时运行；
无多卡、系统安装或其他任务干预。复用 E1 JIT build，不更改 native 源码。

| 实际命令/门 | 实际退出码 | 结果/证据 |
|---|---|---|
| `bash scripts/check_cpu_reference.sh` 最终执行 | 0 | **621 passed / 3 skipped**，Ops268/freshLJ1/Framework306/state22/ops24；`cpu-gate-final.log` |
| M3 CPU Filter 专项，纳入最终门 | 0（门） | 28项；非三角/左手晶胞、6/3×3及有限数值mask、正/负cell_factor、压力、hydrostatic/constant-volume投影及组合、FD/快照/输入错误 |
| M3 CPU Framework 专项，纳入最终门 | 0（门） | 43项；ASE逐步H/历史/坐标/力/晶胞、联合收敛、freeze/restart、缓存失效、alias/admission及singular候选rollback |
| HCU Framework 专项 | 0 | **8 passed**；CPU→HCU显式恢复、非默认stream、退出/False-mask冻结；`framework-hcu.log` |
| 原生 solver HCU 回归 | 0 | **31 passed**；`native-hcu-regression.log`，L/U/非默认stream/info/输入保护/原子失败等 E1 门继续通过 |
| 解析势 CPU、HCU Torch、HCU HIP，三个独立进程 | 均0 | 各3条轨迹，与ASE一样15/12/16步联合收敛，reference/压力与自定义缩放/mask；15个广义自由度中心差分 |
| 周期 MPA-0 Al4 CPU、HCU HIP 最终复验 | 均0 | 均 **5次更新/6次主轨迹模型求值** 联合收敛，3次重启继续更新；`mpa0-*-final.json/log` |
| isolated full variable-cell HIP/Torch trace、CPU LU阳性对照、observer编译 | 均0 | 15DOF/3步，结果见下表及 `trace-summary.json` |
| Ruff check / format check、compileall、`git diff --check` | 0 | 最终检查通过；compileall由最终CPU门执行 |

3个 skip 是既有 Warp 专属测试，不新增 skip/xfail。没有 NVIDIA 环境，不伪造比较。
较早 CPU门615/3与最终621/3的差别是新增6个输入准入检查；两次均通过，最终为准。
最早测试夹具漏 `free_energy` 及未激活 DTK 的子进程已修正；HCU guard 修正了
cuda/cuda:0 标签、Framework bool 状态标量和既有 neighbor metadata 边界。
数值容差没有因此放宽，也未为通过探针改动 BaseDynamics 或邻居实现。

## 数值结果

以下为逐步最大绝对差，不仅比较最终能量/结构。Oracle 是独立 ASE CPU FP64；
解析势 NumPy calculator 与 Torch model 独立表达，MPA-0 Oracle 使用 ASE CPU calculator。

| 核验 | 最大误差 |
|---|---|
| CPU 三条解析轨迹 H / 原子位置 / 晶胞 | 8.98e-12 / 8.19e-15 Å / 1.38e-14 Å |
| HCU HIP 三条解析轨迹 H / 原子位置 / 晶胞 | 7.24e-12 / 8.57e-15 Å / 1.31e-14 Å |
| HCU HIP 解析广义力 / raw direction | 2.95e-14 / 1.31e-14 |
| 全15DOF、p=0.037 eV/Å³、cell_factor=2.7 FD，CPU / HCU | 1.96e-10 / 1.45e-10 |
| Al4 CPU最终复验，H / 原子位置 / 晶胞 / 力 / stress | 1.06e-11 / 1.86e-13 Å / 1.37e-13 Å / 2.39e-13 eV/Å / 2.13e-14 eV/Å³ |
| Al4 HCU HIP最终复验，同上 | 1.49e-11 / 6.00e-14 Å / 5.33e-14 Å / 1.22e-13 eV/Å / 3.36e-15 eV/Å³ |
| Al4 HCU HIP restart，H / 原子位置 / 晶胞 | 5.26e-12 / 1.34e-14 Å / 1.43e-14 Å |

解析势在 CPU、HCU Torch/HIP 的本机 restart 均位相等；CPU→HCU迁移按逐量容差验收。
真实 HCU MACE 重算有归约舍入，不能要求任意 MLIP restart 位相等。
Al4最终广义 fmax约0.02591814、物理 fmax约0.02637236 eV/Å、原始应力最大
8.15087e-4 eV/Å³，能量约-14.95946503 eV；晶胞实际变化约0.10656708 Å。
仅是一个周期无机小晶胞，尚非分子晶体 CSP 工作流。

投影情形逐步对 ASE，未将非保守投影假称全空间梯度。`constant_volume` 与 ASE
相同只是去迹，不严格控制每步体积；在压力/mask投影路径最终 raw force/stress
可能超过未经投影的阈值，默认停条件始终是全部 N+3 **广义行**的严格 fmax。

## HCU 主机工作与传输检查

独立 trace probe 不运行 CPU Oracle；observed LAPACK 不修改参数/结果。
新增 observer 覆盖 E1 `LAPACKE_dstedc[_work]`，以及 LAPACKE dgesv/getrf/getrs
与 LP64 Fortran dgesv_/dgetrf_。真实 CPU3×3 solve/det 命中两次 dgetrf_，
验证 LU observer；Torch同维 BFGS 分解命中 host eigen symbols，验证阴性判定工具。

| 15 DOF完整变胞3步 | 受监测host LAPACK | D2H | 大矩阵 H2D |
|---|---|---|---|
| 显式 HIP | 0 | 1字节检查、8字节system_id scalar；无72字节cell/坐标或1800字节H回传 | 仅初始化几何/常量，没有H eigvec回传 |
| Torch reference | 3次dstedc + 6次dstedc_work（含work查询） | 同样元数据，另有3次120字节/3次112字节三对角数据 | 3次1800字节eigenvector H2D（另有3次120字节） |

HIP trace能看到原生 syevj GPU kernel。M3增量3×3 solve/det的数值张量保持设备内，
在这个已验证案例未看到受监测host LU或数值矩阵回传。证据限定当前构建/符号/维数，
不扩大全规模、任意库内部函数或整个MLIP workflow。MLIP仍使用既有 Torch reference
neighbor，其体系 ptr/范围元数据是host可见；数值 BFGS pre_update用更严格device/dtype
guard，解析完整生命周期仅额外允许host零维bool/int状态元数据。

hipprof写 `<prefix>.json`，会覆盖同名的probe小JSON。本次完整trace保留在该路径；
通过log中实际passed marker提取 `trace-*-result.json`，未重建丢失的guard计数。
再次运行应给probe `--output <prefix>-result.json`，避免与profiler output重名。

## 诊断性能记录

最终 Al4 主轨迹含校验/DeviceGuard/同步，模型6次、更新/step各5次；含初次模型冷调用，
JIT初始化不在该表。不同采样集的中位数不能直接相加，也不能视为无探针生产benchmark。

| 中位数 ms | CPU Torch | HCU HIP |
|---|---|---|
| 模型（6次，首个含冷调用） | 322.02 | 43.16 |
| BFGS update（5次，含Filter与scratch clone/检查） | 11.37 | 22.07 |
| Framework step（5次） | 327.49 | 62.81 |

保留完整逐次timings，暂不推断整体加速率或新kernel需求。M4需要按成功CSP结构
测端到端profiling，再决定是否进入M5；不在本阶段改变默认backend或曲率语义。

## 复现命令

从仓库根目录，先 `source scripts/activate_hygon_env.sh project`。CPU用
`HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`；HCU先确认空闲，
设 `HIP_VISIBLE_DEVICES=4`、相同线程数，以及 E1现成cache：

```sh
export NVALCHEMI_HIP_BFGS_EIGH_BUILD_DIR="$PWD/artifacts/ase-bfgs-e1-2026-10-09/build"
export MAX_JOBS=2 PYTORCH_ROCM_ARCH=gfx936
bash scripts/check_cpu_reference.sh
.venv/bin/python -m pytest -q -o addopts='' packages/framework/test/compatibility/test_ase_bfgs_cell_hcu.py
.venv/bin/python -m pytest -q -o addopts='' packages/ops/test/torch/test_bfgs_eigh_hip_hcu.py
timeout 240 .venv/bin/python probes/ase_bfgs_unitcell.py --device cuda --backend hip --output artifacts/ase-bfgs-m3-2026-10-09/analytic-hcu-hip.json
timeout 240 .venv/bin/python probes/ase_bfgs_unitcell.py --device cuda --backend torch_reference --output artifacts/ase-bfgs-m3-2026-10-09/analytic-hcu-torch.json
timeout 240 .venv/bin/python probes/ase_bfgs_unitcell.py --device cuda --backend hip --real --checkpoint /home/wangleping/.cache/mace/mace-mpa-0-medium.model --output artifacts/ase-bfgs-m3-2026-10-09/mpa0-hcu-hip-final.json
```

CPU用相同probe `--device cpu --backend torch_reference` 分别跑解析/`--real`。
MLIP checkpoint SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`；
结构由probe确定性生成（Al fcc4原子，轻微晶胞/原子扰动），不依赖网络下载。
原生测试/trace均限时180秒；完整实际argv/退出码及sha在 `acceptance-summary.json`。

```sh
cc -shared -fPIC -O2 probes/ase_bfgs_unitcell_lapack_trace.c -ldl -o artifacts/ase-bfgs-m3-2026-10-09/unitcell-lapack-trace.so
LD_PRELOAD="$PWD/artifacts/ase-bfgs-m3-2026-10-09/unitcell-lapack-trace.so" \
BFGS_LAPACK_LOG="$PWD/artifacts/ase-bfgs-m3-2026-10-09/trace-hip.lapack.log" \
timeout 180 /opt/dtk-26.04/bin/hipprof --hip-trace --trace-args --output-type 0 \
  -o "$PWD/artifacts/ase-bfgs-m3-2026-10-09/trace-hip" \
  .venv/bin/python probes/ase_bfgs_unitcell_trace.py --backend hip \
  --output artifacts/ase-bfgs-m3-2026-10-09/trace-hip-result.json
```

换 `--backend torch_reference` 和独立prefix得到Torch阳性对照；本次执行时probe output
与prefix.json同名，按上节方式保留实际trace/marker。CPU LU阳性对照具体代码存manifest。

## 限制与下一阶段

单体系 FP64 eager/forward-only；无 constraints、Strain/FrechetFilter、异构Batch/refill、
DomainParallel、NVIDIA硬件/其他HCU构建、多卡/线程压力、compile或训练验收。
HCU参数轨迹覆盖默认、压力+缩放及一个mask，其他参数/组合目前是CPU Oracle证据。
真实MLIP变胞的HCU门是显式HIP一个Al4结构，不是全后端全模型gate。全部运行数据、
资源/代码哈希与命令记录在 `artifacts/ase-bfgs-m3-2026-10-09/`，不入库；本报告可提交。

下一阶段目标是 M4：异构 Batch 模型共享计算、独立ID状态与按自由度分组的batched
eigh；逐体系严格收敛冻结、重排/退出/动态补位及完整快照。先 CPU/独立ASE逐步一致，
再限时单卡HCU生命周期和真实分子晶体 FIRE2→BFGS 变胞CSP/重启，记录成功结构耗时、
模型求值次数与profiling。范围见[计划](../docs/ASE_BFGS_DEVELOPMENT_PLAN.md)。
等待用户确认M4再修改对应代码，不自动推进M5或推送/集成。
