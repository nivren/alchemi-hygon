# HIP 邻居 M5：当前晶胞批量 metadata 与变胞三路径比较

日期2026-10-06；M5及补充的20,000原子/变胞/三路径方案已获用户确认。
M4收口 `95f2ef6`、`88b6b2f`；分支 `codex/perf-hip-batch-metadata`，本阶段实现与性能比较完成，
实现收口 `e2d569c`，未合并或推送；最终回归/profile记录见下文。

## 实现与来源

只修改ops `_hip_neighbor_executor._current_cell_metadata`；框架API、registry选择和
native build/query/sort kernel保持。CPU/HIP Tensor路径用批量inverse/norm/floor/ceil，
每次当前cell完整重算grid/radius/offset。原先每体系 `_grid_spec` 求inverse并通过
`.item()`逐轴取值，之后再次批量求inverse；新路径只求一次批量inverse。

grid超8192时，每体系最大轴（首轴tie）ceil-halving，逐轮保持原规则；使用float64
乘积判定8192边界，避免大dims int64乘积溢出，减半使用整除+余数避免加1溢出。
半径extra cell+1、int32 offset安全检查、奇异cell失败保留。浮点metadata运算在原dtype，
float64仅用于整数网格乘积边界判定，不改变位置、晶胞或模型dtype。

来源为锁定ops `26dbceb...`、framework `4dfe372...` 的当前移植：Torch reference
`_grid_spec` 是逐字段oracle；M1–M3原邻居/模型/生命周期契约保持。没有缓存metadata、
旧plan或workspace；None/Warp/auto不变，显式HIP full-PBC/full/COO/skin=0。
原 trusted plan checked初始化和完整CSR/query保留，不新增native COO/query并行。

新增契约 `docs/HIP_BATCH_METADATA_CONTRACT.md`、测试
`packages/ops/test/torch/test_hip_batch_metadata.py`，和两个公共验证探针：
`probes/hip_batch_metadata_benchmark.py`、`probes/hip_batch_metadata_mace.py`。

## 公平测量边界

HCU4 BW1000 64G/gfx936，DTK26.04/Torch2.9.0/HIP6.3.26093，OMP_NUM_THREADS=1。
host设备节点、hy-smi空闲检查；单卡任务顺序执行，每子任务120–420秒限时，记录退出码。
现有M1 build cache，原native编译flags相同，无系统/时钟/绑核修改。

三列为设备内Torch reference **cell-list**、修改前HIP（M4）、修改后HIP（M5）。
正式产品仍只有一个注册HIP后端：后两列是同一后端修改前后的代码，不是两个产品实现。
保存的旧executor仅用于probe进程内前后比较。原executor源码保存到
artifacts，通过probe进程内显式函数选择，entrypoint cache切换在计时外，无永久registry改动。
每快照同r/H/ptr/cutoff/capacity/dtype，先精确核对有效 `(i,j,shift)` 与体系归属，
再计同步wall neighbor API（含selection/dispatch、metadata、build/query/sort/COO/Batch
writeback）；不把几何变换、fixture、正确性核对算入API时间。

1次warmup、3轮×3样本/路径/快照，路径顺序轮转，保留所有ms.txt和median/mean/std/range。
首次API独立记录，包含加载/allocator初始化；已有build cache不等于冷JIT。
每路径记录PyTorch peak allocated和相对调用前的incremental peak，不等同全设备显存。
分项/profile单独运行，不混入最终无profiler计时；不是长弛豫吞吐基准。

## 变胞与Batch输入

seed20261006、密度0.04 atom/Å³、cutoff6Å、capacity256、full-PBC/full/COO：

| 配置 | 原子数/结构 | 初始N/B | 补位后N/B |
|---|---|---|---|
| many | 80/120交替 | 20,000/200 | 20,070/201 |
| medium | 400/600交替 | 20,000/40 | 20,350/41 |
| few | 1,600/2,400交替 | 20,000/10 | 21,400/11 |

每配置FP32/64，连续initial→0.94压缩→1.08膨胀→剪切→exit/refill→补位后压缩剪切，
共六快照。exit/refill几何夹具移除source0/1，保留幸存者当前r/H，加入三个不同大小
且新source ID的结构，改变N/B/ptr，再继续变胞。此为邻居布局夹具，不冒称完整动态
优化器退出证据；真正FusedStage状态/预算退出/arrival priming由M3公共回归另验收。
另保留32个小结构两精度，同六快照，用于检查小Batch退化。

小Batch均值46，实际36/55交替（初始1456原子），性能夹具使用seed20261007。
原seed20261006的FP32初始夹具出现Torch reference多两条对称边，原HIP与M5完全
一致；以原FP32 r/H提升到FP64计算，边距为6.000000088Å、d²−36为1.05649e-6，
属于cutoff机器精度邻域的既有舍入分歧。失败日志和small-diagnostic.json保留。
没有修改cutoff或集合容差，也没有把原夹具计为三路径通过；新seed各路径共享，
每快照仍严格核对。该已知边界不由metadata优化解决，不能宣称任意FP32边界位级等价。

真实MPA-0：三个既有Li/Na CIF共18原子，锁定checkpoint SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
两优化器×FP32/64，Torch reference dynamics+优化HIP邻居实际驱动5步，r/H实际改变；
三路径在相同六个轨迹快照核对拓扑/E/F/stress，再分别测neighbor、model-only、
neighbor+model。模型普通e3nn eager、冻结参数、保留位置/应变autograd，default dtype
与模型一致。端到端数字是邻居+模型，未包含optimizer/hook/sink/文件IO。
没有将18原子模型结果外推成20,000原子真实MLIP验收，预算退出不是物理收敛。

## 正确性与最终数值

CPU精确整数metadata oracle、inverse逐dtype容差；HCU实际pipeline拓扑与M1独立
有限镜像oracle；涵盖旋转、压缩、剪切、max_nbins缩减、超大dims、cutoff相邻浮点、
空B与精确零pivot奇异cell。首次奇异夹具用重复浮点行，FP32/LU舍入未稳定触发错误；
改用精确零行并保留失败日志，不放宽容差、不更改产品错误逻辑。

### 变胞邻居 API：FP32 和 FP64 独立结果

六快照每路径54样本合并中位数；单位ms。before/after只是同一HIP后端修改前后。

| 配置 | 精度 | Torch reference | 修改前 HIP | 修改后 HIP | 相对Torch倍数 | HIP耗时下降 |
|---|---|---:|---:|---:|---:|---:|
| many | float32 | 1450.214 | 85.978 | 10.617 | 136.60× | 87.65% |
| many | float64 | 1432.148 | 84.560 | 11.011 | 130.07× | 86.98% |
| medium | float32 | 317.542 | 25.290 | 10.539 | 30.13× | 58.33% |
| medium | float64 | 323.633 | 25.610 | 10.988 | 29.45× | 57.10% |
| few | float32 | 111.241 | 14.048 | 10.472 | 10.62× | 25.45% |
| few | float64 | 117.454 | 14.466 | 10.894 | 10.78× | 24.69% |
| small | float32 | 235.048 | 21.379 | 9.543 | 24.63× | 55.36% |
| small | float64 | 232.536 | 21.398 | 9.781 | 23.78× | 54.29% |

目标many FP32三个轮次分别下降87.56%、87.54%、87.83%，超过5%门槛；
其余配置修改后HIP均未退化。FP64三个20,000原子配置独立通过。
这组结果支持本次metadata优化在所测输入的准入，不是整个弛豫吞吐或auto准入。

FP32 incremental peak MiB（六快照范围；Torch/修改前HIP/修改后HIP）：

| 配置 | Torch | 修改前 HIP | 修改后 HIP |
|---|---:|---:|---:|
| many | 113.7–135.4 | 458.1–458.8 | 458.1–459.2 |
| medium | 119.1–240.2 | 457.6–463.1 | 458.0–463.1 |
| few | 433.2–446.5 | 456.6–483.8 | 456.2–483.8 |

本次优化减少时延，没有形成显存下降结论；两精度逐快照peak/首次API见原始JSON。

### 实际 MPA-0 变胞快照

每配置、每测量模式、每路径54样本。单位ms，中位数；不含优化器更新。

| 优化器/精度 | 测量 | Torch reference | 修改前 HIP | 修改后 HIP |
|---|---|---:|---:|---:|
| fire-float32 | 邻居 | 24.103 | 8.564 | 7.593 |
| fire-float32 | 模型 | 27.875 | 27.866 | 27.864 |
| fire-float32 | 邻居+模型 | 52.484 | 36.917 | 35.945 |
| fire-float64 | 邻居 | 24.106 | 8.671 | 7.696 |
| fire-float64 | 模型 | 27.753 | 27.742 | 27.765 |
| fire-float64 | 邻居+模型 | 52.419 | 36.962 | 35.980 |
| fire2-float32 | 邻居 | 24.645 | 8.685 | 7.688 |
| fire2-float32 | 模型 | 28.356 | 28.370 | 28.378 |
| fire2-float32 | 邻居+模型 | 53.524 | 37.534 | 36.551 |
| fire2-float64 | 邻居 | 23.977 | 8.640 | 7.686 |
| fire2-float64 | 模型 | 27.704 | 27.699 | 27.706 |
| fire2-float64 | 邻居+模型 | 52.296 | 36.922 | 35.967 |

四组实际r/H均改变，拓扑/E/F/stress同快照对照通过。模型耗时基本不变；
修改前到修改后的邻居+模型耗时下降约2.6%，小真实Batch的收益小于大量结构metadata情形。
FP32/64分开测量，不将两者轨迹等价或精度可互换作为结论。

### 回归验收

- CPU ops：45 passed、28 skipped（HIP设备项）。
- CPU framework：21 passed、28 skipped（HIP设备项）。
- HCU4 ops：73 passed；framework：49 passed。
- 合成8配置×6快照：48组，三路径有效拓扑严格一致（小Batch新seed，旧失败单独保留）。
- 实际模型4配置×6快照：24组，拓扑/E/F/stress逐物理量容差通过。
- 原M3真实生命周期：两优化器/两精度五步状态比较通过；预算退出/补位、arrival priming通过。
  这是独立回归，不是三路径性能样本的误差聚合。FP64同几何最大E/F/stress误差分别
  1.42e-14 eV、1.46e-14 eV/Å、1.70e-16 eV/Å³；FP32分别
  3.81e-6、7.85e-6、9.13e-8（单位同前），沿用M3逐物理量容差。

### 独立 metadata 诊断与 trace

200结构FP32当前cell，metadata单独warmup3/9样本，同步wall；修改前78.375ms，
修改后0.667ms；整数metadata精确核对、inverse按既有容差通过。
该诊断与三轮公共API比较分开执行，支持逐体系同步/重复inverse是本次改动收益来源。

hipprof `--hip-trace --stats` 独立many初始快照：现有
`batch_cell_query_kernel<float>` 一次2.641ms。trace含fixture与进程初始化，
CSV总kernel百分比不能当作稳态API成本占比；没有PMC/occupancy结论，
也没有宣称查询已经是唯一瓶颈。原始db/hipkernel/hiptrace CSV保留。
所有限时runner最终任务exit_code=0。没有额外系统设置或常驻GPU任务。

## 下一里程碑方案（M6，已确认）

目的：缩短metadata优化后剩余查询成本，优先覆盖FP64分子晶体变胞弛豫。

1. 在M5基线分项测query、sort、COO与host调度，核对200/40/10结构和小Batch的FP32/64。
2. 结合锁定Warp查询实现审阅，在现有HIP接口内实验线程协作遍历候选cell/原子；
   保留full-PBC、周期镜像、容量溢出、确定排序、stream和每步当前cell重建语义。
3. 同几何三路径复测：Torch reference、M5 HIP、候选HIP；重复变胞/补位和真实
   MPA-0 FIRE/FIRE2短轨迹。单独保留FP32 cutoff边界失败，不能通过放宽检查隐藏。
4. 预先确定目标配置；公共邻居API至少下降5%、多轮一致、其他所测输入不出现稳定
   超5%退化且回归通过才保留。达不到则撤回候选并记录结果。

用户已确认：先收口M5提交，再开始M6；不自动合并或推送。


## 重现

启动环境：`source scripts/activate_hygon_env.sh project`，另source
`artifacts/hip-current-cell-neighbor-m1/native_env.sh`；
`HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1`，所有命令有独立timeout。

```bash
timeout 420 .venv/bin/python probes/hip_batch_metadata_benchmark.py \
  --baseline artifacts/hip-batch-metadata-m5/baseline-executor.py \
  --case many --dtype float32 \
  --output-dir artifacts/hip-batch-metadata-m5/many-float32
```

case另取medium/few/small，dtype另取float64；每次六变胞快照。small显式加
`--seed 20261007`，其余默认20261006。

```bash
timeout 300 .venv/bin/python probes/hip_batch_metadata_mace.py \
  --baseline artifacts/hip-batch-metadata-m5/baseline-executor.py \
  --checkpoint /home/wangleping/mace-mpa-0-medium.model \
  --data-root /data/csp_data/catl_data/li_mp_data \
  --optimizer fire2 --dtype float64 \
  --output-dir artifacts/hip-batch-metadata-m5/mpa0-fire2-float64
```

optimizer另取fire、dtype另取float32。完整顺序runner与每任务确切命令、timeout、
退出码在 `artifacts/hip-batch-metadata-m5/run_remaining.py`、`run-status-first.json`、`run-status.json`；
首次many-float32命令见本节。fixture生成/每样本数据均在各case JSON和ms.txt。

候选门槛：many FP32相对原HIP变胞median下降>=5%、三个轮次一致，其余HIP项无>5%
稳定退化、所有适用正确性gate通过。尚不作为auto或全部输入规模性能准入。
query并行、workspace复用、skin/selective rebuild、native COO、其他MLIP、长收敛、
训练力损失、compile/opcheck、DomainParallel和NPT/NPH均未新增。
