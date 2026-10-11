# ASE-compatible BFGS：DTK 26.10 eigensolver 独立复验

2026-10-10，用户要求测试 `/opt/dtk-26.10`，针对此前海光 Torch/DTK 的
BFGS eigh 问题复验。**本轮没有修改产品代码、默认 DTK、版本锁或 solver 准入，
没有安装软件、提交、合并或推送。**沿用 `wangleping/feat-ase-bfgs-m4`，
HEAD `cfd22ad11fe0e1704dff48f53abee69135f8db99`，保留未提交 M1–M4 工作。
旧版本完整调查见 [DTK 26.04 调查](g2-ase-bfgs-eigh-investigation.md)。

结论：26.10 的独立 DC 对已知真实285维 Hessian 恢复正确输出，LOWER 分支也完成；
但小矩阵 CPU LAPACK 路径仍存在，**285维 DC 还实际调用私有 CPU `steqr_host`
计算子块**，不满足当前产品禁止主机 eigensolver 数学的合同。现有26.04构建的
Torch 在隔离26.10库环境中无法导入，完整 Torch/BFGS 兼容性未验收。

## 环境与证据范围

- 安装标识：`DTK-26.10-release`，发行文件时间2026-09-21。
  `rocsolver_get_version_string` 实际返回 **3.20.0.0a4420f8**；旧版为
  3.20.0.80bb870c，不能仅比较3.20.0主版本。
- HIP头文件7.2.26365；hipSOLVER头文件1.6.0。26.10安装内的 hipcc 输出
  `dcc 26.04.0-0 / clang17`，InstalledDir为26.10；如实记录，不据此推断库版本。
- 不链接Torch的独立HIP程序；ldd及18份运行时maps核实本轮26.10进程使用26.10
  rocSOLVER/rocBLAS/HIP，两个26.04 Jacobi对照进程只加载26.04 DTK库。
  每个子进程单独 source 对应 env.sh，不改变项目激活脚本或系统配置。
- 实测前确认无KFD PID，只使用 physical HCU4，`HIP_VISIBLE_DEVICES=4`，
  OMP/OPENBLAS各1线程，关闭core dump；单进程有界、顺序执行。
  solver进程最多25秒，profiler/debugger最多35秒。CPU oracle另进程禁用GPU。
  结束时再次检查无KFD PID；资源状态不沿用到后续阶段。
- 原始材料：`artifacts/eigh-dtk2610-2026-10-10/`，包括源码、二进制、输入、
  oracle、info/谱/向量、实际库路径/hash、命令参数、trace、debugger调用栈、
  计时和退出码。`final-sha256.json` 冻结最终材料，初始hash文件保持原样。
  原 `agy-eigh-test` 工程及26.04调查材料没有被修改。

26.10库SHA256：

| 库 | SHA256 |
|---|---|
| rocSOLVER | `15840e3d81a438cb1d488b24f53b074766cb43a450ebc512fa8fe35f753917aa` |
| hipSOLVER | `bf1e335270691c79e2bbfabd47b8087fc811809e24bf204d20260cbbe7e06915` |
| rocBLAS | `e70f5c9e4a0fcf2e7b4a8c18757b2be3e1fd68282c20d45d5d17d91559e166b9` |

## 数值结果

使用M4捕获的真实 `[1,285,285]` Hessian，267个特征值近70，未加jitter、
阻尼或谱夹紧。原pt文件SHA256为
`b6292e2269fb80370da1ffe4729d8a9c85ab928d8842ecd9621bc4ad2c0e2ded`；
连续FP64输入bytes的SHA256为
`6e34cb2ebf7e8ec4a4d920f4dcb86002c1bf90431e6721778441be54d4b7307c`。
B4为该真实矩阵的**四份相同副本**，测试batch调用，不代表四条独立CSP轨迹。

CPU FP64 oracle按LOWER三角构造完整对称数学矩阵；row-major LOWER输入映射
column-major UPPER，输出原地column-major特征向量转置恢复。另测试完全对称
数学矩阵的column-major LOWER分支。只比较谱、方程残差和正交，不逐列比较
简并空间的特征向量。

| 独立HIP路径 | 26.04旧调查 | 26.10本轮最终输出 |
|---|---|---|
| 真实285/B1、单矩阵DC、UPPER | info=0，但15个错误特征值，最大误差79.859 | 退出0、info=0，最大谱误差1.42e-13；方程9.03e-16、正交1.61e-15 |
| 真实285/B1、strided-batched DC、UPPER | 同样错误谱，正交0.632 | 退出0、info=0，谱1.42e-13；方程9.04e-16、正交1.57e-15 |
| 真实285/B4、strided-batched DC、UPPER | 本轮未重跑旧版本故障DC | 退出0、四个info=0，谱1.42e-13；方程9.04e-16、正交1.56e-15 |
| 对称真实285/B1、DC、LOWER | 20秒超时、退出124 | 退出0、info=0，谱1.28e-13；方程8.13e-16、正交1.71e-15 |
| 真实285/B1、B4、Jacobi | 既有正确基线 | 本轮两版本均通过，谱2.56e-11；方程/正交约1.14e-13，20 sweeps |
| 随机199/200/201、B1、DC | 旧边界调查正确 | 三组退出0、info=0，谱误差不超过1.96e-14 |

主要13组最终输出全部通过CPU检查，其中11组26.10、2组26.04 Jacobi对照；
另3组trace和2组debugger输出也通过CPU检查。所有18个诊断进程退出0。
不是18项pytest，也不是完整ASE BFGS功能门。重复计时的case执行一次冷调用+
3次预热调用，API status每次检查，保存并数值校验的是最终次输出。

诊断界为谱绝对误差<2e-8、方程/正交相对误差<1e-10，仅用于筛查，
不修改产品逐步ASE及谱atol/rtol门。方程公式为
`||HV−Vw||F / max(||H||F,1)`，正交为 `||VᵀV−I||F / sqrt(n)`。
实测误差如表，无需放宽容差。

这些结果说明**此前可在纯HIP复现的错误谱及LOWER超时，在本次26.10输入/调用
范围没有再出现**。未运行原样Torch外部扩展，不能直接宣称其VMFault已修复，
也没有厂商正式修复条目、其他输入或全BFGS稳定性证据。

## CPU计算路径仍存在，而且不限于整体n<200

1. 随机n199/B1 DC仍实际调用一次 `LAPACKE_dstedc`，库内D2H为1592+1584
   字节三对角数据，H2D为1592+316808字节，后者含CPU计算的n²特征向量。
   n200/201未见受监测LAPACK调用，**这个边界只描述该LAPACK路径**。
2. 真实n285/B4 DC没有受监测的 `LAPACKE_dstedc`，但求解内部有532次async
   copy：D2H共23376字节、H2D共23472字节，以及D2D8464字节。多次160/152
   字节及逐列160字节copy含三对角子块及其向量，不能归为整数metadata。
3. 对同一真实n285/B1的限时hipgdb验证抓到：

   ```text
   HOST_STEQR_HIT n=3
   steqr_host<double,double>
     ← steqr_template<double,double,double*>
     ← rocsolver_steqr_template<double,double,double*>
     ← stedc_template<double,double,double>
     ← rocsolver_stedc_template<false,true,...>
     ← rocsolver_syevd_heevd_template<false,true,...>
     ← rocsolver_syevd_heevd_strided_batched_impl
   ```

   同时抓到160字节D2H从同一个 `steqr_template` 调用；静态反汇编确认
   D/E/向量拷回、stream同步、调用 `steqr_host`、结果写回设备。该host函数
   有CPU浮点乘除/开方及迭代路径，**这是CPU数值求解，不是诊断同步**。
   见 `host-copy-gdb.stdout`、`steqr-leaf-disassembly.txt` 和
   `steqr-host-disassembly.txt`。最初仅对LAPACK/另一hybrid函数设断点未命中，
   那份 `host-math-gdb.stdout` 不能用于推断没有主机数学，保留失败的观察范围。
4. Jacobi n138/B4 trace仍为10次4字节D2H收敛标志、9 sweeps，无数值矩阵
   往返或受监测LAPACK；设备数学不等于完全异步。本轮没有把285维Jacobi
   trace扩大为新的全尺寸库内执行验收。

**仅监测LAPACK或按整体n>=200选择DC，不能保证26.10的设备内数学合同。**
新版本准入须同时检查私有主机数学、数值数据copy与同步，而非只检查API/info
和输出设备。本轮未确认可关闭 `steqr_host` 的公开选项，不猜测控制开关。

## 微基准性能

同一探针源码、同一真实矩阵、同一单卡，FP64 full vectors；冷调用后3次同步
host wall求解计时的中位数。包括库内临时分配和同步，排除探针输入恢复D2D、
初始化、输出D2H、CPU校验；计时进程无profiler/observer/debugger注入。
不等同于产品wrapper或BFGS端到端性能。

| 真实285维，单位ms | 26.04 Jacobi | 26.10 Jacobi | 26.10 DC |
|---|---:|---:|---:|
| B1预热中位数 | 92.189 | 80.004 | 9.123 |
| B4预热中位数 | 115.332 | 122.389 | 18.634 |

26.10 DC对同版Jacobi约8.77倍/B1、6.57倍/B4，但包含上面的CPU子块求解。
不能把这个收益作为当前设备内后端加速承诺。26.10 Jacobi相对26.04的B1约
13.2%下降，B4约6.1%上升，不能笼统声称Jacobi升级加速。
26.10 DC首调用B1/B4为187.104/200.016ms，Jacobi为85.517/129.230ms；
扩展编译不在表内。LOWER DC预热中位数9.682ms。

## 当前Torch兼容性与建议

当前项目 `.venv` 的海光Torch为2.9.0+das.opt1.dtk2604，其ELF需要
`libhipblas.so.2`、`librocblas.so.4`，26.10安装提供SONAME3/5。
隔离到26.10库路径后，import退出1：

```text
ImportError: libhipblas.so.2: cannot open shared object file: No such file or directory
```

未创建兼容软链、混入旧DTK库或安装替代Torch。这样的问题不能靠设置Torch
cuSOLVER preference解决。由于未能进入Torch，**26.10下的Torch默认策略、
batched循环、原外部扩展和Framework完整弛豫未测**；不能继承26.04的结论。

当前产品仍保留26.04已验收准入、全部Jacobi、设备内有效性守卫与一套ASE
BFGS数学，默认 `torch_reference` 不变。本轮不把26.10登记为支持构建。

下一阶段建议保留 **M5-1：Jacobi关键路径定位与安全候选评估**。先在现有匹配
Torch/DTK环境对真实147/285 DOF、B1/B4/实际分组测kernel、收敛同步、分配和
端到端占比；26.10候选的完整集成须先取得匹配Torch，再做隔离环境准入测试。
在当前设备内数学要求下，不恢复DC；如将来考虑显式混合CPU/HCU求解器，需
另定合同并由用户确认，不能作为静默fallback。本轮没有启动该开发阶段。

## 复验入口与未覆盖范围

辅助源码和输入在上述artifact目录，后续复验应另建输出目录，保留本次材料。
典型命令（bash子进程）：

```bash
source /opt/dtk-26.10/env.sh
export HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -c 0
hipcc -O2 -std=c++17 --offload-arch=gfx936 artifacts/eigh-dtk2610-2026-10-10/pure_solver.cpp -o /tmp/eigh-dtk2610-solver -lrocsolver -lrocblas
timeout 25s /tmp/eigh-dtk2610-solver 285 1 dc artifacts/eigh-dtk2610-2026-10-10/real-n285-b1.bin /tmp/eigh-dtk2610-real-b1
```

`pure_benchmark.cpp` 同样编译；`EIGH_REPEATS=3` 执行冷+3次预热。
hipprof使用 `--hip-trace --trace-args --output-type 0`，三次trace均退出0；
`host-copy.gdb` 由限时hipgdb执行。CPU检查另进程激活26.04，禁用GPU后运行
`check_pure.py`、`finalize.py`，二者退出0，最终汇总为 `manifest.json`。
`git diff --check` 退出0；产品代码未改，因此没有重复运行功能回归。

未验证：匹配26.10的Torch、产品准入、逐步ASE/Restart/真实CSP端到端、
其他输入/尺寸/硬件、FP32、多卡/线程压力、Autograd/compile及生产吞吐。
产品源码未改，未重复M4 CPU/HCU功能门；本轮是版本候选调查，不是新功能验收。
