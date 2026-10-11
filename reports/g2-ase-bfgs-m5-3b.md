# ASE-compatible BFGS M5-3B：异构 Hessian 并行比选

2026-10-10，用户确认的 probes 隔离阶段。本地新增算法/求解器开发，不是
NVIDIA 上游 BFGS 移植。分支 `wangleping/perf-ase-bfgs-m5-3b`，基准 HEAD
`cfd22ad11fe0e1704dff48f53abee69135f8db99`；继承此前未提交的 M1–M5-3A 工作。
无提交、推送、合并、产品注册或默认变更。

本阶段结果与实际命令/退出码见 `artifacts/ase-bfgs-m5-3b-2026-10-10/`；
历史 M5-3A 证据保留，三份原型源复制到本阶段 `frozen-m5-3a/` 并记录 hash。

## 实现与合同

[隔离合同](../docs/HIP_BFGS_HETEROGENEOUS_PROBE_CONTRACT.md) 先于后端编写。

- `ase_jacobi_workers.cpp` 为同一 M5-3A 单宽数学增加释放 GIL 的入口；原始
  cpp/cu/wrapper 不变。预加载后用固定 2/4 worker 和各自 HIP stream，下发
  不同宽度组；有限 worker 不是逻辑 Batch 容量上限。按元素量做主机任务分配。
- producer/完成 event、device/current stream、record_stream 和全部 future
  排空后报错保护输入/输出生命周期。每 sweep 的主机诊断等待保留。
- `ase_jacobi_heterogeneous.cu` 用临时实际 n/int64 元素与向量偏移、tile 列表
  将不同矩阵纳入同一次旋转/更新 launch。矩阵不做谱 padding，配对算术、
  256线程范数规约、中心化、1e-15阈值、冻结与稳定设备排序保持原语义。
  不同本地 round 的空转尾长为偶数，CPU模拟证明双缓冲选择；排序暂仍逐组。
- `ase_bfgs_heterogeneous_math.py` 精确保留产品数学表达式，仅将准备所有 H、
  一起分解、方向检查、原子提交分开。只在 process-only injection 中接 Framework；
  产品 State/Filter、模型共享求值、Hooks、restart schema 未改。

## 正确性证据

- CPU：23种奇偶维度配对/全局空转/冻结缓冲；probe数学adapter与产品12步逐位一致。
  CPU Ops104、Framework105 passed，分别运行；没有重跑全项目门。
- 四个候选：32个冻结矩阵案例加9个真实分组，负谱、简并谱、零谱、UPPER污染、
  混合奇偶和达到1113维。32矩阵的五项原生输出与串行逐位一致，9真实组通过CPU谱/方程/正交门；输入保护和元数据报错通过。
  最大谱绝对差1.052e-12、方程4.935e-15、正交2.956e-13，原有门未放宽。
- 四候选各11体系/10步 ASE：坐标2.673e-14、Hessian7.816e-14、谱1.740e-11、
  raw3.517e-14、applied5.191e-15 最大差；restart逐位、独立single对照通过。
  真实最后一组1-sweep未收敛及末组坏特征向量均失败且全状态不提交。
- Framework24个组合通过：固定和UnitCellFilter（默认/pressure+scale/mask）
  各7步CPU oracle、非默认stream、bitwise restart、完整冻结、重排/退出/新ID；
  两种坐标的single restart promotion和原生sampler/FusedStage budget refill。
  七步模型求值8次；阶段shared调用没有拆成每体系调用。
- 首次Framework门退出1：测试Dataset仍是CPU，而显式HIP拒绝CPU。把测试数据
  显式置于HCU后重跑退出0；保留首次日志，不修改产品或数值门。

## 性能与 profiler

三独立进程，每进程cold分开、预热2次、七次交替/轮换等范围采样，共21个稳态样本。
下表为wall中位数ms，原始每进程的样本/HIP event/范围/标准差全部保留。

| 真实H组合与窗口 | rocSOLVER | 自写串行 | 2 stream | 4 stream | 原生异构 |
|---|---:|---:|---:|---:|---:|
| Li八种宽度，checked solver | 212.732 | 112.183 | 74.888 | 74.811 | 86.378 |
| T2 561 B8，checked solver | 985.575 | 520.959 | 521.510 | 521.654 | 518.906 |
| Li+T2混合，checked solver | 1198.470 | 633.613 | 560.758 | 560.903 | 563.221 |
| Li完整math/guard/commit | 233.898 | 194.085 | 148.990 | 148.971 | 158.402 |
| T2完整math/guard/commit | 904.650 | 505.378 | 505.882 | 505.974 | 503.563 |
| 混合完整math/guard/commit | 1139.385 | 701.250 | 591.197 | 591.274 | 681.377 |

solver是冻结的真实H；完整math窗口用真实H加合成q/f/history，包含真实Hessian更新，
**不算真实MLIP工作流**。rocSOLVER solver控制先通过通用输入检查，再进入既有产品
checked函数，故有重复selected-triangle检查；其完整math行直接调用原产品函数。
并行收益以相同检查的自写串行对照为主，不能把rocSOLVER控制的这一额外检查归因算法。

2 stream相对自写串行：Li分解约少33.24%、完整math少23.23%；混合分解少11.50%、
math少15.69%。T2不加速，约多0.1%；4 stream相对2 stream的差异很小，不选为产品建议。
原生异构虽然减少launch，却在Li/mixed完整math慢于2 stream；T2仅约0.36%微小差异。

首进程Torch allocator峰值额外空间（MiB，包含结果；不含不可见库内部分配）：

| 组合 | rocSOLVER | 串行 | 2 stream | 4 stream | 原生异构 |
|---|---:|---:|---:|---:|---:|
| Li | 6.51 | 9.50 | 9.59 | 9.59 | 13.57 |
| T2 | 67.26 | 100.19 | 100.19 | 100.19 | 120.34 |
| 混合 | 69.50 | 102.47 | 102.47 | 102.47 | 132.65 |

hipprof Li checked窗口（有profiler开销，不作为上述性能样本）：

| 候选 | 窗口ms | kernel数 | 同时执行不同队列数最大值 | 至少两队列同时执行ms |
|---|---:|---:|---:|---:|
| 自写串行 | 127.705 | 13816 | 1 | 0 |
| 2 stream | 81.493 | 13816 | 2 | 43.910 |
| 4 stream | 81.447 | 13816 | 4 | 24.858 |
| 原生异构 | 94.694 | 8088 | 1 | 0 |

重叠统计来自HIPOPS实际Begin/End/queue，不来自stream对象计数；去除窗口外上传和oracle。
异构是在同一kernel网格内分派不同矩阵，单queue不表示组间Python串行。
2/4stream D2H均为1字节诊断共87次；原生异构D2H1字节33次，H2D三次共18800字节
是临时n/偏移/工作列表；没有矩阵D2H/H2D求解路径。所有候选保留CPU调度和标量等待，
不宣称零主机/零同步/全片上/LDS或一次launch解决全部eigh。

真实MLIP为同一 MACE MPA-0 medium checkpoint（SHA75428afe…fb638），邻居后端
`torch_reference`。只读使用M5-3A记录的16个独立CIF：Li原子数2/4/7/12/16/28/64/160；
T2文件名虽然含46，ASE实际每个184原子，561维B8。输入及扰动种子/处理与M5-3A一致。

每组做三次fresh serial/streams2配对，第二轮反转作业顺序。各作业1 FIRE2+10 BFGS
更新，主轨迹12次模型求值，各ID前3步checked；暖机前三次更新后保留七个完整更新
窗口，各候选/数据集共21个样本。框架pack/unpack与状态提交在窗口，MLIP/oracle在外。
这是沿真实短轨迹的七个不同H，不能当作七次重复同一输入；冻结H微基准才是相同输入。

| 真实变胞更新窗口 | 串行中位数ms | 2 stream中位数ms | 判断 |
|---|---:|---:|---|
| Li八个不同维度 | 217.380 | 170.806 | 耗时下降21.43% |
| T2同宽561 B8 | 762.693 | 755.980 | 约0.88%差异，轨迹内变化很大，不声称加速 |

12个工作流的共享HCU力/应力ASE CPU数学和独立HCU比较均通过。
最大ASE坐标2.132e-14、cell1.777e-14、Hessian9.312e-11；独立HCU坐标1.022e-14、
cell2.088e-14、Hessian2.860e-11。六个最终配对ID/status一致，末态最大位置
1.719e-8 Å、cell8.225e-9 Å、能量1.218e-11 eV；沿用2e-8/2e-10数量级门，没有
放宽原ASE门。末态差大于前三步是实测现象；不据此声称长期逐位确定或完整收敛。

另有一个fresh streams2 Li八体系独立CPU MACECalculator+ASE前三步门退出0（Hessian6.753e-11、坐标1.777e-14 Å），
与上述共享力oracle分开记录。T2独立CPU MLIP在M5-3A超时124，本阶段未重试，
不记作已通过。全部新增真实数据作业均为步级qualification，没有完成全弛豫收敛。

最终命令、退出码、资源观察和hash见本阶段acceptance manifest；每个device作业限时，
只用HCU4，采样未见同卡第二个已分配计算进程。JIT和冷启动、checkpoint保存、模型/
独立oracle耗时不计入稳态BFGS更新收益。没有测每成功结构端到端吞吐，不推断生产收益。


## 选择、支持边界与下一阶段

建议产品接入采用已验证的动态自写Jacobi；有多个宽度组时优先2stream，
单组保持原生batched入口。4stream的额外复杂度没有足够收益；原生异构暂留实验。
不实现性能失败时静默回退，用户仍能显式选当前rocSOLVER。M5-3A合成大Batch
退化证据保留，不能把真实低秩H的收益推广到所有谱。

本阶段没有固定147/285或B≤4门；实测子集不等于全尺寸性能保证。
原生异构正确不保证优于多stream；4stream不保证比2stream快。
FP64/eager/forward-only、HCU4 BW1000/gfx936、海光Torch2.9.0+das.opt1.dtk2604 /
DTK26.04；不是26.10准入、NVIDIA/多卡/compile/autograd验收。
主机控制、诊断标量传输、H2D元数据和设备内临时pack成本均存在。

当前产品仍为M4接口与显式rocSOLVER Jacobi，优化实现只在探针。
[下一阶段M5-4提案](../docs/ASE_BFGS_M5_4_PROPOSAL.md) 待用户确认后再产品接入。
