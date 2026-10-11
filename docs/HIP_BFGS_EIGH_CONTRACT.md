# HIP-BFGS-EIGH：E1 求解器合同

2026-10-10 DTK26.10候选复验：真实285维独立DC错误谱/LOWER超时在已测范围
未再出现，但285维DC实际调用私有CPU `steqr_host` 计算子块并搬运数值数据，
n199 CPU LAPACK也仍存在。仅监测LAPACK或整体尺寸阈值不足以证明设备内数学。
现有Torch在隔离26.10环境import失败；未修改产品构建准入/全部Jacobi选择，
26.10未登记支持。见[26.10报告](../reports/g2-ase-bfgs-dtk2610-eigh.md)。

2026-10-10独立调查：无Torch链接的HIP程序也复现真实285-DOF的单矩阵/
strided-batched UPPER/DC错误谱，原外部扩展复现VMFault。当前库CPU LAPACK
回退边界实测n<200；Torch Default本机已走hipSOLVER，仍逐矩阵调用，同真实
输入Default/cuSOLVER有界复验超时。当前Jacobi保留不变；库内有4字节收敛
标志host读取，设备内数学合同不表示完全异步。见[调查报告](../reports/g2-ase-bfgs-eigh-investigation.md)。
以上仅当前构建/已测输入，不将历史Torch通过结果扩展为任意GPU求解策略可用。

M4 修订（2026-10-09）：本机构建的 strided-batched DC 在真实 CSP 285-DOF
近初始 alpha 简并 Hessian 上出现 deflate2_2_kernel VMFault；保存矩阵的隔离
调用还出现 info=0 但最大特征值误差79.859。私有 solver 现统一使用 native Jacobi，
不再采用下文 E1 的200阈值。该修订保持数学/FP64/stream/API 不变，也不是失败后
切换的 fallback。另增设备内分解残差/正交有效性守卫，拒绝有限但错误的谱；不修改谱。
新 Batch 接口与生命周期见[合同](TORCH_ASE_BFGS_BATCH_CONTRACT.md)。E1/M3
旧 trace 和性能只能作为旧代码证据，不能直接用于当前实现的支持/性能结论。

M3 扩展（2026-10-09）：BFGS prepare/step 现准入单体系 `ase_unitcell` 模式，
`single_system/variable_cell/ase_unitcell` capabilities，共用下述私有 solver，
原生代码/数值 body 不变。ASE `3N+9` pack/unpack、应力及 Filter/restart 的具体
合同另见 [UnitCellFilter](TORCH_ASE_UNITCELL_FILTER_CONTRACT.md)，实测另见
[M3 报告](../reports/g2-ase-bfgs-m3.md)。E1 阶段记录保持历史范围，不视为当时已支持 Filter。

2026-10-09，用户已确认 E1；本地独立实现，非上游 BFGS 移植。外部
`agy-eigh-test/torch_rocsolver_eigh` 仅只读参考，不复制源码或安装其扩展。
数学继续遵守 [ASE BFGS 合同](TORCH_ASE_BFGS_CONTRACT.md)。

## 边界与选择

- 显式 `backend="hip"` 登记 `hip.bfgs-ase-rocsolver-v1`，仅替换 BFGS 的
  特征分解；H 更新、abs 谱方向、maxstep、状态及 restart 与 Torch 共用。
  默认仍为 `torch_reference`；`auto`、None/warp 无准入，不静默回退。
- 初期准入本机已验证 HCU Torch `2.9.0+das.opt1.dtk2604`、HIP `6.3.26093`、
  rocSOLVER `3.20.0.80bb870c`。版本不匹配明确报错，其他构建须重做库内执行验收。
- BFGS 局部求解器接受 FP64、连续 strided、无梯度、HIP 设备的 `[n,n]` 或
  等维 `[B,n,n]`，n/B 正且可由 int32 表达；不支持空矩阵/空 batch、非连续、
  FP32、复数、更高 rank、Autograd。PBC、单位、neighbor 不适用。
  M4 另提供按DOF分组的异构 BFGS Batch 调度；此私有 solver 仍只接受等维组。
- `UPLO` 严格为 L/U，只读取指定三角和对角；非指定三角可含 NaN。
  指定三角非有限报错。输出升序 W 和列特征向量 V，均 FP64 同设备连续、
  独立 storage；输入不变。谱简并时按重构/正交与 BFGS 方向验收，不比较向量符号。
- E1 历史实现按 M2 低层事实使用 n<200 Jacobi，n>=200 divide-and-conquer；
  当前 M4 实现全部宽度使用 Jacobi，以规避已复现的 strided-batched DC 故障。
  不对谱夹紧、不添加阻尼/线搜索/曲率跳过。Jacobi absTol=1e-15、maxSweeps=100；
  不因失败切换求解器。边界 n=199/200/201 必须追踪复核。
  大简并谱另检查相对重构<=1e-11、相对||I||_F的正交误差<=1e-12；
  原随机谱的绝对正交界1e-11保留。BFGS生产有效性守卫使用相对残差及相对
  正交误差1e-10，仅作为提交前错误检测，不替代CPU oracle与专项精度验收。

## 原生实现与失败

- 独立 C++/HIP 扩展，复用项目 lazy JIT 机制，只在显式调用时编译/加载。
  `torch.library.custom_op` 声明无输入 mutation，提供 fake/meta；显式拒绝
  requires_grad。Compile 支持须单列验收，不由 fake 或 opcheck 推导。
- Device guard 保存/恢复当前设备；线程内每设备独立 RAII rocBLAS handle，
  每次 checked set_stream 绑定当前 Torch stream，不使用默认 stream 假设。
  所有临时张量、库 workspace 和 handle 保持到求解诊断同步完成。
- Row-major clone 独立输入；以 column-major API 翻转 UPLO，返回转置后的 V。
  API status 与 GPU solver info 分开检查；info 非零/非收敛/非有限输出明确报错。
  BFGS 在 commit 前失败，positions/H/历史/计数均不变。
- 允许 host 元数据与诊断标量同步；禁止主机特征分解、矩阵 D2H/H2D、隐藏
  dtype 变化、CPU fallback。主机数学包括库私有求解函数，不限于LAPACK。
  库内行为结合当前构建的调用观察与HIP trace验收；只有LAPACK observer未命中
  或输出驻留设备，不能据此推断没有主机数学。新构建出现数值copy须追踪其调用栈。

## 验收

先写 CPU 准入/错误/fake/meta/替换分解不改数学测试，再实现原生后端。
CPU reference gate、既有相关回归通过；空闲单 HCU 限时验证：

1. 单矩阵/等维 batch，L/U、输入不变、重复/负/近零谱，API/非收敛失败，
   非默认 stream；opcheck schema/fake，不支持梯度明确失败。
2. 特征值对 CPU FP64：atol=2e-10、rtol=2e-11；随机谱重构/正交各<=1e-11。
   特殊谱按范数及谱间隔记录对应误差，不放宽全局容差。
3. ASE 逐步 H/方向/q 及 restart；Framework 解析势和真实 MPA-0 固定晶胞。
4. n=138/199/200/201/552 的原生库 trace：监测主机 LAPACK 和矩阵往返；
   标量诊断拷贝与输入准备拷贝分别记录。
5. n=138/199/200/201/276/552，B=1/4/16；冷启动/JIT 与预热稳态分开，
   按实际输出布局和诊断同步计时；不承诺 B=1 加速，不默认启用自动选择。

E1 当时未验证范围包含 Filter/应力和异构 BFGS/refill/CSP，后续已由 M3/M4 窄验收
覆盖，见[M4报告](../reports/g2-ase-bfgs-m4.md)。当前仍未验证 NVIDIA/其他构建、
多卡/多线程压力、compile/Autograd、全CSP数据集成功率与生产性能。

## E1 已运行结果

E1 当时个人分支已完成CPU gate550 passed/3既有skip、原生HCU31项、解析势ASE
逐步/位相等restart、真实MPA-0固定晶胞11步收敛及产品求解器18组验收。
原生求解器n138/199/200/201/552 B4、完整BFGS自由度9/138/201/552各3步trace
无受监测host LAPACKE或矩阵回传；Torch阳性对照有8次host调用。D2H仅1/4字节
诊断；DC的32/64/128字节H2D为库索引元数据，随机输入准备H2D另计。该结论只
覆盖本机构建与已测案例，不推导所有规模、硬件或库版本。详情见
[E1报告](../reports/g2-ase-bfgs-e1.md)。

启用时激活既有DTK/Python环境，显式选择 `backend="hip"`。第一次调用lazy JIT
会编译本项目 `_native/bfgs_eigh.cpp/.cu`，需要已安装的 C++/HIP compiler/ninja
及rocSOLVER/rocBLAS；导入/default Torch不编译。可用
`NVALCHEMI_HIP_BFGS_EIGH_BUILD_DIR` 指定独立cache（默认Torch extension cache），
`NVALCHEMI_HIP_BFGS_EIGH_VERBOSE=1`记录编译。库或编译不可用时直接失败；
不自动安装、升级、加载外部工程二进制或切换后端。
