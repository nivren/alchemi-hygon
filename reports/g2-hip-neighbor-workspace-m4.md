# HIP 邻居 M4：性能基线与 workspace 复用实验

日期：2026-10-06；用户已确认。M3 已收口实现 `6b1680a`、文档 `47cf259`，未合并或推送。
M4 分支 `codex/perf-hip-neighbor-workspace`，已收口实现 `95f2ef6`，未合并或推送。

## 结论

完成分项测量、hipprof trace、workspace候选实现及正确性/性能验证。候选未达到
预先规定的5%目标，已撤回产品代码。保留 benchmark、契约和可重放实验，不将
缓存登记为已支持能力。HIP仍显式选择、skin=0，每步当前晶胞完整重建；无auto准入。

主要热点为逐体系 metadata 推导，下一阶段建议批量推导当前晶胞metadata，待用户确认。
本阶段不改变query线程、native COO、容量策略、模型/优化器或数值容差。

## 输入、环境与测量

HCU4 BW1000 64G/gfx936，DTK26.04，Torch2.9.0/HIP6.3.26093，OMP_NUM_THREADS=1。
设备节点和hy-smi在host检查，无其他KFD进程；所有GPU任务限时120/180/240秒、单卡顺序执行。
采用现有M1 native build cache，编译选项/算法不变。CPU与framework/ops测试分进程。

合成输入：seed20260920，46×32/92×64，FP32/64，full-PBC，12/16Å带倾斜晶胞，
cutoff3Å、capacity128、COO。真实模型：既有三个Li/Na CIF共18原子，FP32 MPA-0
checkpoint SHA256 `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`，
cutoff6Å、capacity128，普通e3nn eager，冻结权重但保留力/应变autograd。

每项先精确比较reference `(i,j,shift)`，真实E/F/stress沿用M2容差；然后5次warmup、
3轮×15样本。每个同步wall sample包括framework选择/ops dispatch、完整rebuild与Batch
writeback；模型项为forward能量/力/应力，组合项为邻居+模型，不是完整FIRE step或长作业吞吐。
保存全部样本、median/mean/std/min/max，不只取最好值。

初始独立进程的baseline/candidate有时段差异，补测原baseline，再增加同一进程AB/BA
交替配对。executor切换和entrypoint cache清理在计时外，首次调用的解析开销两侧相同。
配对结果作为准入依据；保留独立进程原始结果，不用其约10%的表观退化作因果结论。
配对workspace常驻与PyTorch allocator cache存在于同一进程，两侧相同设备状态。

首次API（已有build cache）原baseline-repeat 46×32 FP32约777.7ms、candidate约1073.4ms，
包含加载/初始化，既不是冷JIT编译基准，也不是稳态比较；其余shape首次调用亦单列于JSON。
最初baseline JSON中的cold_api_ms为验证后首次计时，不能解释为冷启动，未用于冷启动结论。

## 热点证据

额外逐阶段同步的wall诊断用于定位，不能与无profiler总耗时直接相加或当作kernel时间：

| FP32 输入 | 总warm API ms | metadata ms | checked plan初始化 ms | query ms | 排序/materialize ms | scan workspace查询 ms | topology workspace查询/分配 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 46×32 | 18.006 | 12.345 | 1.879 | 1.027 | 0.766 | 0.148 | 0.182 |
| 92×64 | 30.548 | 24.195 | 1.888 | 1.278 | 1.109 | 0.152 | 0.182 |

hipprof针对两次46×32 FP32 rebuild，query kernel平均约0.605ms；trace另记录427次
hipMemcpyWithStream和大量Torch小kernel。源码 `_current_cell_metadata` 对每个体系调用
`_grid_spec`，后者每体系求inverse并通过 `.item()` 读取dims/argmax，之后又批量求inverse。
这共同支持“metadata有大量细粒度host/device往返”的判断；trace包含输入构建/启动，
不能将427次全部归为单次metadata或当作纯稳态比例。没有PMC/occupancy调优结论。

## 候选与拒绝依据

候选复用内部CSR/candidate/scan/sort scratch；线程局部单device/stream/dtype槽，
高水位容量view、128MiB保留上限。public MATRIX/COO/geometry输出始终新分配；输入、
metadata、plan不缓存，checked初始化和每次完整CSR/query保留。改变N/B/ptr/cell/capacity
仍重建，异常重试重置counts，stream/key改变替换缓存。

以下为同进程配对，每侧45样本，三个轮次目标收益分别1.23%、1.33%、1.33%：

| 项目 | baseline median ms | candidate median ms | 耗时下降 |
|---|---:|---:|---:|
| 46×32 FP32 COO | 17.903 | 17.672 | 1.29% |
| 92×64 FP32 COO | 30.427 | 30.184 | 0.80% |
| 46×32 FP64 COO | 17.970 | 17.728 | 1.35% |
| 92×64 FP64 COO | 30.335 | 30.104 | 0.76% |
| MPA-0 neighbor | 8.447 | 8.210 | 2.81% |
| MPA-0 model-only | 27.665 | 27.661 | 0.01%（近似不变） |
| MPA-0 neighbor+model | 36.662 | 36.428 | 0.64% |

目标样本std baseline0.039ms/candidate0.088ms，median比1.0131；使用skill
`compare_timings.py`核对45样本。收益可测但不达预设5%，因此拒绝保留产品缓存。
不事后降低门槛，也不把微小邻居收益扩大成弛豫应用性能结论。

候选 `test_hip_neighbor_workspace.py + test_hip_current_cell_neighbors.py` **30 passed**，
含CPU规则/oracle和真实HIP分支：容量增长/缩小、内存上限、线程/stream/dtype隔离、
旧输出独立、压缩/多镜像/恢复、同N/B不同ptr、不同B、overflow后重试。
仅正确性通过不代表性能准入。源码、测试和patch归档后从产品撤回。

## 重现命令与原始证据

公共探针 `probes/hip_neighbor_workspace_benchmark.py`；启动前：

```bash
source scripts/activate_hygon_env.sh project
source artifacts/hip-current-cell-neighbor-m1/native_env.sh
export HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1
```

基线命令主体（每项240秒限时）：

```bash
timeout 240 .venv/bin/python probes/hip_neighbor_workspace_benchmark.py \
  --output-dir artifacts/hip-neighbor-workspace-m4/baseline \
  --checkpoint /home/wangleping/mace-mpa-0-medium.model \
  --data-root /data/csp_data/catl_data/li_mp_data
```

产品撤回后仍可通过显式probe参数重放，不修改永久registry或产品文件：

```bash
timeout 240 .venv/bin/python probes/hip_neighbor_workspace_benchmark.py \
  --paired-baseline artifacts/hip-neighbor-workspace-m4/baseline-executor.py \
  --candidate-executor-source artifacts/hip-neighbor-workspace-m4/candidate-executor.py \
  --workspace-source artifacts/hip-neighbor-workspace-m4/candidate-workspace.py \
  --output-dir artifacts/hip-neighbor-workspace-m4/paired-replay \
  --checkpoint /home/wangleping/mace-mpa-0-medium.model \
  --data-root /data/csp_data/catl_data/li_mp_data
```

hipprof仅定位（120秒限时，统计和trace不与无profiler样本混用）：

```bash
timeout 120 hipprof --hip-trace --stats \
  -o artifacts/hip-neighbor-workspace-m4/profile-baseline \
  .venv/bin/python probes/hip_neighbor_workspace_benchmark.py \
  --profile-only --output-dir artifacts/hip-neighbor-workspace-m4/profile-input
```

证据目录 `artifacts/hip-neighbor-workspace-m4/`：baseline、candidate、baseline-repeat、
paired的JSON/逐轮ms.txt；profile.log及hiptrace/hipkernel CSV/DB；candidate-correctness.log；
candidate-executor.py/candidate-workspace.py/candidate-tests.py/candidate-executor.patch；
target-comparison.txt；replay-smoke.log。trace原始数据不入库，摘要不暴露设备网络信息。
最终归档重放1轮×3样本/四合成case通过，仅作为重放正确性smoke，不作准入性能证据。

产品恢复后CPU ops **27 passed/10 skipped**、CPU framework **21 passed/24 skipped**；
HCU4 framework M3 **16 passed/16 deselected**。skips为未分配HIP参数。
产品executor与保存baseline逐字节相同；workspace候选模块/测试未留在产品。
compileall、契约解析和diff检查通过。没有新完整CPU/HCU gate、长轨迹或模型族准入。

## 下一阶段提案（待确认）

M5：批量推导当前晶胞inverse/grid/radius/offset，减少逐体系标量同步及重复inverse。
保留max_nbins缩减规则、严格cutoff、安全容量和每步metadata全重算；与原逐体系实现
逐字段比较，并回归大网格缩减、奇异cell、压缩/剪切/多镜像/ptr变化和真实M3流程。
使用本阶段相同输入/配对计时门槛验收；query并行依据后续剩余热点另阶段确认。
