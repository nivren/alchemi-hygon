# ASE-compatible BFGS：agy-eigh-test 独立复验

2026-10-10；用户要求重新审阅整个 `/home/wangleping/codes/agy-eigh-test`，
独立核实海光 Torch/DTK eigensolver 问题，并据此提出 BFGS 实现建议。
本轮为调查，**没有修改 BFGS 产品代码、默认后端或系统库，没有启动 M5 实现**。
沿用 `wangleping/feat-ase-bfgs-m4`，HEAD 为
`cfd22ad11fe0e1704dff48f53abee69135f8db99`；保留未提交的 M1–M4 改动。

## 环境与证据

项目 `.venv`：Python3.12.13，HCU Torch2.9.0，
`__hcu_version__=2.9.0+das.opt1.dtk2604`，HIP6.3.26093，DTK26.04，
rocSOLVER3.20.0.80bb870c，hipSOLVER header1.6.0，MAGMA2.8.0。
构建 commit `4e3c1e79303a0774d61d4d7efa36c8f27dac7dc6`。
空闲 physical HCU4/BW1000 64G，映射为 `cuda:0`；OMP/OPENBLAS 各1线程，
扩展编译 MAX_JOBS=2。顺序运行有界单卡探针，关闭 core dump，未安装/重置/升级系统。
探针完成后一次 `hy-smi --showpids` 检查无 KFD PID；最终文档检查时已出现新的
其他任务，本轮探针进程为0，未再执行 HCU 测试或干预任务。资源空闲判断不沿用到下一阶段。

原始记录：`artifacts/eigh-investigation-2026-10-10/`，包括环境、外部源文件 hash、
实际加载库及 hash、同输入 CPU oracle、独立 HIP 源码/二进制、调用栈、同步计时、
HIP trace、错误日志与 `manifest.json`。外部源码前后 hash 相同，未写入外部工程。
这些 artifacts 是调查材料，不作为产品安装源或可发布依赖。

## 对四个问题的核实

| 原判断 | 当前本机构建的证据与修订 |
|---|---|
| Torch 默认走 MAGMA，须强制 cuSOLVER | 初始 preference 为 `Default`，环境 `TORCH_LINALG_PREFER_CUSOLVER` 未设置。但真实调用栈显示默认已经调用 `hipsolverDnDsyevd → hipsolverDsyevd → rocsolver_dsyevd`。显式 cuSOLVER 调用链相同；仅设置偏好不解决当前瓶颈。不能把旧版默认策略视为当前事实。 |
| Torch batched eigh 循环单矩阵调用 | 已确认。对 n138/B4 和 n285/B4，各一次 Torch eigh 产生4次成功的单矩阵 `hipsolverDnDsyevd`。默认与显式 cuSOLVER 均如此，没有调用 strided-batched DC。 |
| DTK DC 在 n≤200 回退 CPU | 主机 LAPACK 调用已确认，但本机边界为 **n<200**：独立 strided-batched DC 的 n199 有 `LAPACKE_dstedc`，n200/201 没有受监测调用。这里只覆盖 FP64、完整特征向量和当前库版本。rocSOLVER 属于数学库层，不是内核驱动层。 |
| Row-major 与 column-major 不一致 | 正确，但需同时处理所选三角与向量输出。本地连续 row-major 输入的 L 对应 column-major U，lda=n、strideA=n²；原地输出的向量须转置恢复列向量。对称性不能免除输出布局和非连续输入检查。 |

调用栈记录为 `default-v2-api.log`、`cusolver-v2-api.log`。
栈中 `rocsolver_dsyevd status=9` 对应 hipSOLVER **buffer-size 查询**，不能误计为
实际求解失败；实际求解与 hipSOLVER 调用 status=0。
最初 GNU audit 没有收集到调用事件，不用它推断路径。第一版 API observer 的
RTLD_NEXT 未找到局部作用域符号而 abort，退出134也不是 DTK VMFault 证据；
修订版按已加载库句柄解析符号后退出0，记录真实调用栈。

MAGMA 仍存在且可以显式选择，但不宜把“任何规模都直接切到 CPU dstedc”视为
本机二进制的准确描述。对已安装 `libmagma.so` 的反汇编显示，
`magma_dsyevd_gpu` 的小规模分支调用 CPU `dsyevd_`，较大分支调用
`magma_dstedx`，后者含 CPU `dsteqr_` 等辅助例程及混合实现。
这是静态路径证据，不是对所有输入的逐函数运行跟踪。本轮 MAGMA 计时通过，
但未做其内部主机数学逐调用验收，不能把它认定为满足设备内数学合同的替代后端。

## 全工程审阅的适用边界

- `src/verify_eigh.cpp` 验证单矩阵随机谱的 API、info、残差/正交/重构，
  有正确性价值；不覆盖真实 BFGS 高重数谱的 strided-batched 故障。
- `src/benchmark_batched.cpp` 主要计时，检查 API status，未验收输出谱或 info；
  `benchmark_eigh.cpp` 也不是完整 BFGS 逐步数值测试。
- `src/rocsolver_eigh.py` 是直接 DC 的 ctypes 路径，检查 info；
  `torch_rocsolver_eigh/csrc/rocsolver_batched.cpp` 则 n≤200 Jacobi、n>200 DC，
  只检查 API status，**未读取 info/收敛诊断**。两者不能混为同一实现。
- 子包 benchmark 的随机矩阵、晶体 demo 的随机 SPD 矩阵不能代表真实优化历史。
  demo/Skill 中的特征值 clamp 与 ASE 的 abs 谱语义不同，不复用其优化数学。
- 根 `scripts/benchmark_batched_bfgs.py` 在 import Torch 前设置 cuSOLVER 偏好，
  从70I作5次模拟 rank-2 更新，是比随机谱更相关的压力输入，本轮已纳入。
  它的随机力增量、分母 mask/偏移不是 ASE 实际逐步 oracle。
- 原扩展对 full-vector 连续单卡路径的布局处理有参考价值；产品还需补齐
  Device guard、handle 生命周期、checked stream、info/数值守卫与明确输入约束。
  eigenvalues-only 的 `A.contiguous()` 可能 alias 输入，也不能原样当无 mutation API。
- 外部一键脚本会 make clean、使用全 CPU 编译并运行较大固定批量，未执行该脚本。
  原 Python3.10 二进制未装入项目；原样 C++ 源码在项目 artifacts 独立编译。
  历史15–27倍加速不作为当前 BFGS 性能承诺。

## 真实 Hessian：DTK 数值缺陷可独立于 Torch 复现

输入沿用 M4 保存的 `[1,285,285]` Hessian，原 `.pt` SHA256：
`b6292e2269fb80370da1ffe4729d8a9c85ab928d8842ecd9621bc4ad2c0e2ded`。
267 个特征值距70小于1e-8。CPU FP64 按 LOWER 三角构造 oracle；
独立 HIP 程序显式分配全部 buffers，lda/stride 按安装头文件设置，
绑定独立 stream，所有 buffers 保持到同步及诊断结束，**不链接 PyTorch**。

| 该输入的独立测试 | 进程退出码 | info | 数值结果 |
|---|---|---|---|
| rocSOLVER strided-batched DC、UPPER | 0 | 0 | 15个错误特征值，最大绝对误差79.85904001800401；相对正交误差0.6324555，未通过 |
| rocSOLVER 单矩阵 DC、UPPER | 0 | 0 | 同样15个错误特征值和误差，未通过 |
| rocSOLVER strided-batched Jacobi | 0 | 0，20 sweeps | 最大谱误差2.5608e-11，方程/正交相对误差约1.14e-13，通过诊断 |
| 完全对称化同一 LOWER 数学矩阵，DC UPPER | 0 | 0 | 仍是同样错误谱，不能归因于原输入极小的非对称舍入 |
| 完全相同对称矩阵，DC LOWER | 124 | 无完成结果 | 20秒进程超时，不算通过，也不推断具体停滞代码行 |
| 本机 Torch Default / 显式 cuSOLVER | 各124 | 无完成结果 | 其余11组完成；真实输入未在各60秒进程预算内完成完整计时/诊断，不能报告其性能或通过 |
| Torch 显式 MAGMA | 0 | Torch 正常返回 | 谱误差7.11e-14，方程1.18e-15、正交2.05e-15 |

误差公式：`||HV−Vw||F / max(||H||F,1)`；
相对正交为 `||VᵀV−I||F / sqrt(n)`。比较谱而不逐列比较简并特征向量。
明细 `pure-results.json`；完全对称化 UPPER 结果同样由 CPU checker 复验。

**原样外部 C++ 扩展**编译成功，random n138/B4、random n285/B4、外部模拟
Hessian n285/B1 通过；进入真实矩阵后 `deflate2_2_kernel` VMFault，退出134。
见 `exact-external.stdout` 和失败前的部分 `exact-external.json`，不标记为全项通过。

因此现在可将 UPPER/DC 的错误谱按**当前 DTK rocSOLVER 数值缺陷**提报，
无需依赖 alchemi 或 Torch 包装层解释。VMFault 在原外部扩展也复现，但其
确切底层实现根因仍需海光分析。不能据此断言所有 DTK/ROCm 版本有同一缺陷，
也没有官方修复版本/时间证据。不是 n200 以下 CPU 回退本身导致的 n285 故障。
外部模拟 n285/B1 有275个近70特征值，反而在DC上通过；因此“重数很多”本身
也不是已经证明的根因，必须保留真实输入排查具体求解/deflation路径。

## 性能与库内传输

同一 CPU 生成输入、同一单卡，每个case先记录一次首调用，随后3次同步 host wall
计时取中位数。生成、H2D、CPU校验不在稳态计时内；native Jacobi 包括输入
clone、输出转置、info 同步。无 profiler/observer 注入的计时如下，单位 ms。
每方法首case含进程初次库/JIT成本，后续case可复用已加载库；扩展编译不在表内。
这是诊断微基准，不是 BFGS 端到端加速或完整 ASE 轨迹验收。

| 输入 | Default | cuSOLVER | MAGMA | 直接 Jacobi |
|---|---:|---:|---:|---:|
| random n138/B1 | 4.902 | 4.889 | 10.033 | 19.680 |
| random n138/B4 | 18.953 | 18.994 | 38.531 | 22.076 |
| random n285/B1 | 9.287 | 9.279 | 22.151 | 49.466 |
| random n285/B4 | 36.511 | 36.414 | 87.344 | 61.057 |
| 外部模拟 n138/B4 | 12.190 | 12.207 | 32.303 | 48.291 |
| 外部模拟 n285/B1 | 7.631 | 7.612 | 14.125 | 96.652 |
| 外部模拟 n285/B4 | 29.938 | 29.961 | 53.957 | 114.332 |
| 真实 n285/B1 | 未完成 | 未完成 | 13.521 | 92.355 |

Jacobi 12组完成，保留谱/方程/正交/输入不变诊断；模拟 n285/B1 最大绝对谱
误差3.27e-10，真实输入2.56e-11。调查脚本诊断界为绝对谱2e-8、方程/正交
相对1e-10；这些界只用于压力输入筛查，**不替代或修改产品合同的逐步 ASE
容差及谱 atol/rtol 验收**。MAGMA12组完成，Default/cuSOLVER各11组部分完成。

另对两个已通过的小案例做 HIP trace，目标及 profiler 均退出0：

- DC n199/B1：实际一次 `LAPACKE_dstedc`；求解内部 D2H 1592+1584字节的
  三对角数据，再 H2D 1592+316808字节，包含 CPU 计算的 n² 特征向量。
  是主机数学/数值数据往返，不只是诊断同步。
- Jacobi n138/B4：无受监测 LAPACK；求解内部10次4字节 D2H收敛标志读取，
  输出9 sweeps，GPU syevj kernel 可见，没有数值矩阵/向量往返。
  **数值计算在设备内不等于无 host 同步或完全异步**。
- trace 中探针最初 H2D 输入、最后 D2H 全量输出用于 CPU校验，明确另计；
  不把这些探针操作归为库内回退。`trace-summary.json` 保存分类。
- `hipMemcpyAsync` 的 host 时长可包含等待前序 GPU kernel，不能把它简单
  当作4字节 PCIe搬运成本或与全部 kernel 时长相加。下一阶段需分析关键路径。

## 实现建议与下一阶段

1. 保留当前一套 ASE BFGS 数学、完整 FP64 Hessian/历史、abs 谱方向和 maxstep；
   固定晶胞3N和UnitCellFilter 3N+9 共用核心。真实高重数谱是正常输入，不用
   jitter、谱夹紧、阻尼或曲率跳过绕开求解器故障。
2. 异构 Batch 按实际 DOF 分组；组内 Hessian更新、eigh、投影和步长限制留在
   HCU，不改为 Python 逐体系分解，不用补零扩维改变谱，保留独立ID状态/冻结/补位。
3. 当前 HCU 显式 HIP 后端继续全部宽度 Jacobi，API/info/有限性/残差/正交
   多层检查，失败发生在状态提交前。既有 CPU reference 永久保留；不通过全局
   修改 Torch preference 或 monkey patch 实现产品选择。暂不恢复尺寸驱动的 DC。
4. 设备内数学目标明确区分允许的 metadata/诊断标量同步；Jacobi 当前符合
   不进行 CPU 数学/矩阵回传的窄合同，性能仍未达生产承诺。M4 实现及已通过
   功能门保持原结论；新增故障说明限制 GPU Torch reference 的构建/策略范围。
5. 面向海光准备独立 HIP 源码、输入 hash/raw data、编译命令、实际库 hash、
   UPPER/DC错误谱及 info、LOWER超时、原扩展VMFault日志、CPU oracle和
   Jacobi阳性对照。实际提交工单/发送材料需另行授权；本轮没有外发。
6. **下一阶段建议 M5-1：Jacobi性能定位与安全候选方案**。先对真实147/285
   DOF分组、B1/B4/实际CSP分布采样，区分 kernel、收敛检查等待、布局/分配、
   Framework/模型成本；评估现有库可用的安全参数/API和减少重复临时开销，
   先形成有数值门约束的候选对照与端到端潜在收益，再决定后续优化实现。
   不直接开发新HIP/Triton求解器，不修改BFGS语义，不启用已知故障DC。
   新阶段目标/验收须用户确认后实施。

## 复验入口

所有辅助源码及精确命令参数保留在调查 artifact 目录。激活环境、设置
`HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，关闭core dump。
CPU oracle准备使用 `HIP_VISIBLE_DEVICES=''`。
以下列出本轮命令；后续复验另设日期目录，保留本次冻结材料。

```bash
source scripts/activate_hygon_env.sh project
export HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -c 0
HIP_VISIBLE_DEVICES='' .venv/bin/python artifacts/eigh-investigation-2026-10-10/prepare.py
hipcc -O2 -std=c++17 --offload-arch=gfx936 artifacts/eigh-investigation-2026-10-10/pure_solver.cpp -o artifacts/eigh-investigation-2026-10-10/pure_solver -lrocsolver -lrocblas
timeout 25s artifacts/eigh-investigation-2026-10-10/pure_solver 285 1 jacobi artifacts/eigh-investigation-2026-10-10/real-n285-b1.bin artifacts/eigh-investigation-2026-10-10/pure-real-n285-jacobi
HIP_VISIBLE_DEVICES='' .venv/bin/python artifacts/eigh-investigation-2026-10-10/check_pure.py
```

Default/cuSOLVER/MAGMA/Jacobi微基准用 `runner.py --method METHOD --repeats 3
--output FILE`，每进程 timeout60s。原外部扩展 `--method external --repeats 0`
独立 build到 `external-build/`，本轮退出134；不将它用于生产。
调用观察器只用于独立 trace，计时进程没有注入观察器。
HIP profiler 用 `--hip-trace --trace-args --output-type 0`，每进程 timeout30s。
实际结果汇总于 `manifest.json`，有失败/部分结果的步骤未视为验收通过。

未验证：其他 Torch/DTK/硬件、NVIDIA、FP32、跨设备/多线程压力、Autograd/
compile、DTK修复构建、生产规模/成功率。未重跑 M4 CPU全门，因产品源码未改；
本轮新增的是求解器调查证据，不是新的功能阶段验收。
