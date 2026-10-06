# HIP 邻居 M6：查询线程协作实验

2026-10-06；用户确认M6后收口M5实现 `e2d569c`、文档 `3073f63`。
当前分支 `codex/perf-hip-cooperative-query`，M6尚未提交/合并/推送。
契约：`docs/HIP_COOPERATIVE_QUERY_CONTRACT.md`；最终窄gate通过，所测性能门槛满足，候选保留；非全库/全输入准入。

## 上游审阅与实施

锁定ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`：
`neighbors/cell_list/kernels.py` atom-centric / pair-centric factory，
`launchers.py` `batch_query_cell_list`。
默认atom_centric/direct每原子一线程，串行cell/候选原子；sorted先gather连续位置。
pair_centric以源cell/offset为logical block，分担原子对并原子更新输出。
锁定SHA原文保存在artifacts的upstream-kernels.py/upstream-launchers.py。
本轮未运行Warp/NVIDIA，源码审阅不等同NVIDIA性能比较。

本地候选仅修改 `_native/batch_cell_query.cu`：一原子一64线程block，各线程
stride分担平展候选cell offset，保留cell内原子循环、距离算术和周期镜像语义。
共享计数atomic预留候选slot；query仍无序，既有排序保持公共确定性。
row padding并行初始化，block barrier保证初始化完成后写输出和最终读取count。
平展offset乘积/步进的int64溢出明确失败，不允许负值漏算；
容量/重叠仍明确失败，所有线程须完成同步，不用局部return造成barrier死锁。
接口、注册选择、默认None/Warp/auto、metadata、CSR、排序、Torch COO打包及
模型梯度保持。正式产品只有一个HIP后端，候选待性能门槛判断保留或撤回。

## 测量与探针诊断

HCU4 BW1000 64G/gfx936，DTK26.04/Torch2.9.0/HIP6.3.26093，
OMP_NUM_THREADS=1；读设备节点/hy-smi空闲后单卡顺序执行，子任务限时。
源与编译flags同M5，初始编译独立排除；分项/hipprof与最终wall计时分开。

三路径：Torch reference cell-list、M5 HIP、M6候选，同几何/dtype/容量。
复用M5六变胞/补位快照：200/40/10异构结构约20,000原子及小Batch，FP32/64；
真实18原子MPA-0两优化器两精度五步的同快照邻居/模型/邻居+模型。
各路径每快照3轮×3样本，轮转顺序，先严格拓扑及逐物理量E/F/stress核对。
预先指定few（10结构均值2000）FP64公共API下降至少5%，多轮一致；其他配置
无稳定超过5%退化且所有适用正确性gate通过才保留。未达则撤回产品候选。

首次探针尝试用相同扩展名从另一binary路径加载M5：Python扩展缓存返回同一个模块
对象，候选/基线 `__file__` 均指向候选。该首轮计时已作废，完整日志/JSON保留在
`few-float64-invalid-module-alias*` 与 `module-diagnostic.log`；未作为优化准入证据。
修复为从归档M5 native源码用不同扩展名 `nvalchemi_m6_serial_query` 重新编译，
并断言两个模块对象及文件路径不同，切换 `_EXTENSION` 在计时外。
保存的旧executor只用于probe，正式registry不变，无silent fallback。

已知M5小Batch seed20261006 FP32 cutoff极近边界与Torch两条边舍入分歧保留；
小Batch性能fixture明确seed20261007，原seed失败不能计作三路径通过。

v2部分测量后审查补充offset乘积的int64溢出防护，停止本次自有runner后重新测最终版本；
部分日志在 `v2-partial/`，不混入最终样本。

## 最终结果

### 变胞合成邻居 API

单位ms；六快照每路径54样本合并中位数。修改前HIP=M5、修改后HIP=M6，正式产品一个后端。

| 配置 | 精度 | Torch reference | 修改前 HIP | 修改后 HIP | HIP耗时下降 | 相对Torch倍数 |
|---|---|---:|---:|---:|---:|---:|
| many | float32 | 1418.384 | 11.094 | 9.124 | 17.75% | 155.46× |
| many | float64 | 1444.432 | 11.138 | 9.426 | 15.37% | 153.24× |
| medium | float32 | 320.512 | 10.657 | 8.985 | 15.69% | 35.67× |
| medium | float64 | 331.958 | 11.321 | 9.571 | 15.46% | 34.69× |
| few | float32 | 112.135 | 10.575 | 9.269 | 12.34% | 12.10× |
| few | float64 | 117.629 | 10.912 | 9.674 | 11.34% | 12.16× |
| small | float32 | 232.948 | 9.566 | 6.069 | 36.56% | 38.39× |
| small | float64 | 235.780 | 9.905 | 6.165 | 37.76% | 38.24× |

预先目标few FP64三个轮次分别下降11.34%、11.43%、11.26%，超过5%门槛且方向一致。
八配置均下降，没有其他所测输入稳定超过5%的退化。48快照三路径拓扑严格一致。
合成sizes/density/cutoff/capacity/seed同M5；补位后N为20070/20350/21400，B为201/41/11；
小Batch初始1456/32，补位后1488/33。初始20,000原子不是始终固定N。

每快照JSON另保存首次API、峰值allocated及incremental peak（相对调用前Tensor基线），
不等同全设备显存；优化不作为显存降低结论。源码、编译flags和数据精度保持。

### 实际 MPA-0 变胞快照

锁定checkpoint SHA256 `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`，
三Li/Na CIF共18原子，e3nn eager、冻结参数、保留位置/应变autograd。
每配置5个实际优化步，六同几何快照，每测量模式每路径54样本。单位ms，中位数。

| 优化器/精度 | 测量 | Torch reference | 修改前 HIP | 修改后 HIP |
|---|---|---:|---:|---:|
| fire/float32 | 邻居 | 24.070 | 7.627 | 5.603 |
| fire/float32 | 模型 | 29.689 | 29.687 | 29.672 |
| fire/float32 | 邻居+模型 | 54.546 | 38.000 | 35.959 |
| fire/float64 | 邻居 | 24.264 | 7.752 | 5.646 |
| fire/float64 | 模型 | 30.449 | 30.427 | 30.444 |
| fire/float64 | 邻居+模型 | 55.141 | 38.605 | 36.527 |
| fire2/float32 | 邻居 | 23.754 | 7.558 | 5.534 |
| fire2/float32 | 模型 | 29.817 | 29.845 | 29.845 |
| fire2/float32 | 邻居+模型 | 53.942 | 37.688 | 35.702 |
| fire2/float64 | 邻居 | 24.199 | 7.744 | 5.636 |
| fire2/float64 | 模型 | 29.877 | 29.894 | 29.886 |
| fire2/float64 | 邻居+模型 | 54.872 | 38.331 | 36.218 |

24实际模型快照拓扑/E/F/stress对照通过，四组r/H均实际改变；不同精度不作为轨迹可互换结论。
模型单独耗时基本一致。邻居+模型不包含优化器、Hooks、sink/IO，不是完整弛豫吞吐。
18原子小真实模型不代表20,000原子模型验收，也不代表物理收敛或所有分子晶体。

### 正确性回归

- CPU ops：51 passed、34设备项 skipped；CPU framework：21 passed、28设备项 skipped。
- HCU4 ops：85 passed（含CPU oracle和实际HIP项），非全库gate。
- 新增非整齐N=67/129、独立27镜像oracle、重复公共输出位级确定性、容量溢出和两精度
  int64 offset-count超限显式失败；原当前cell/多镜像/重叠/非默认stream回归保留。
- HCU framework：49 passed；低层mixed-PBC/full/half、空输入、非默认stream、容量失败及
  Torch连续geometry一阶/二阶回归通过。这些低层回归不新增公共HIP no-PBC/half准入。
- 原M3实际模型回归：两优化器/两精度五步状态及四组预算退出/补位/arrival priming通过。
  同几何E/F/stress最大误差（eV、eV/Å、eV/Å³）：FP64为3.55e-15/1.96e-14/1.84e-16，
  FP32为3.81e-6/7.79e-6/1.15e-7，沿用M3容差；这是独立回归，不是性能样本的误差聚合。
- cutoff诊断：Torch51660边、M5/M6均51658边，M5/M6集合完全相同；保留原两条对称边差异。
  不宣称任意FP32 cutoff边界位级等价。性能fixture未通过放宽容差取得通过。
- 最终20个限时子任务exit_code=0；hy-smi结束检查无KFD PID，设备已释放。

### 分项与 profiler

初始几何，每配置每模式3次独立同步分项（不是公共API最终样本）：

| FP64配置 | 查询 修改前→后 ms | 排序 修改前→后 ms | COO 修改前→后 ms |
|---|---:|---:|---:|
| many | 3.276→1.925 | 2.552→2.563 | 0.704→0.715 |
| medium | 3.630→2.097 | 2.551→2.564 | 0.706→0.722 |
| few | 3.550→2.343 | 2.548→2.570 | 0.706→0.718 |
| small | 4.381→0.727 | 0.868→0.886 | 0.528→0.544 |

其余分项和FP32见stages-comparison.json。诊断插入额外同步，不将这些数值直接相加
作为无插桩API耗时，也不把少量样本当作新的性能准入统计。
查询下降、其他阶段基本保持，支持本次修改的收益归因。排序和checked初始化仍有成本，
COO约0.5–0.7ms；尚不足以承诺任何下一算子的优化收益。

独立 `hipprof --hip-trace --stats` few FP64候选查询kernel一次1.983ms；
trace包含启动/fixture/首次API，kernel百分比不当作稳态端到端占比；没有PMC/occupancy结论。
不将此FP64候选与M5 FP32 trace直接计算优化倍数。原始db/hipkernel/hiptrace CSV保留。

## 准入与限制

候选达到预先5%门槛并通过适用回归，保留单一HIP查询实现；仍显式backend=hip，auto未启用。
M6没有native COO、workspace、skin/selective rebuild、其他MLIP族、长步收敛、NVIDIA
实际比较、compile/opcheck、训练力损失或DomainParallel准入。实际MLIP较大Batch整体收益
仍未测；约20,000原子数字是邻居API，不能外推整个弛豫吞吐。

## 下一里程碑 M7 方案（待确认）

目的：验证较大真实分子/晶体Batch在MPA-0变胞流程中的整体受益，完成综合收口审查。

- FP64优先，先约1000原子再约2000原子的多结构Batch；读取共享资源和显存余量后
  限时单卡执行，FP32补充比较。维持相同checkpoint、模型实现、cutoff、容量和优化参数。
- 对Torch reference邻居与M6 HIP做同快照邻居/模型/组合计时，以及实际变胞FIRE/FIRE2
  短步与补位核对；另外记录完整step耗时，明确优化器/Hook/IO边界，不承诺长步物理收敛。
- 审查M1–M6源码、契约、失败边界、报告与当前支持文档一致性；根据真实Batch证据
  判断是否需要后续排序/初始化/COO优化，不预先增加kernel。

用户确认后先提交收口M6，再开展M7；本阶段不自动合并或推送。


## 重现入口

`source scripts/activate_hygon_env.sh project`、
`source artifacts/hip-current-cell-neighbor-m1/native_env.sh`，
再将 `NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR` 指向
`artifacts/hip-cooperative-query-m6/native-candidate`，
`NVALCHEMI_M6_SERIAL_QUERY_BUILD_DIR` 指向
`artifacts/hip-cooperative-query-m6/native-baseline`。
`HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1`。
`run_remaining.py`/`run-status.json` 记录每项命令、超时、退出码。
归档baseline源码与executor来自M5；未经冷JIT采样，首次加载也不作为稳态收益。


```bash
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 420 .venv/bin/python \
  probes/hip_cooperative_query_benchmark.py \
  --baseline artifacts/hip-cooperative-query-m6/baseline-executor.py \
  --case few --dtype float64 \
  --output-dir artifacts/hip-cooperative-query-m6/few-float64
```

M5 baseline源码可从 `e2d569c` 的相同路径提取（cu/cpp及executor），放上述baseline路径。
small另加 `--seed 20261007`，其他默认20261006；真实模型入口加 `--real` 与M5相同model/data参数。
源码hash在source-hashes.json，原始统计在synthetic/real/stages-comparison.json及各快照ms.txt。
