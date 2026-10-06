# HIP 邻居 M7：较大真实 Batch 综合验收

2026-10-06。用户确认后先收口 M6：实现 `fe18147`、文档 `c2a8339`；
本阶段分支 `codex/test-hip-real-batch-validation`，仅验证探针、契约及证据文档，
未修改产品 kernel/API，未提交 M7、合并或推送。

契约：[HIP_REAL_BATCH_VALIDATION_CONTRACT.md](../docs/HIP_REAL_BATCH_VALIDATION_CONTRACT.md)。
原始记录：`artifacts/hip-real-batch-m7/`；探针：`probes/hip_real_batch_validation.py`。

## 输入和运行方法

HCU4 BW1000 64G/gfx936，DTK26.04、Torch2.9.0/HIP6.3.26093，OMP_NUM_THREADS=1。
执行前 `/dev/kfd`、render nodes 可见，hy-smi 无 KFD PID，各卡显存使用2MiB。
单卡顺序执行，每子任务 timeout900秒；不调整频率、系统环境或其他任务。

真实 CIF 来自 `/data/csp_data/catl_data/li_mp_data`，按文件名排序取前缀。
约1000档实为1029原子/39结构；约2000档实为2030原子/92结构。
不复制单个结构或替换成随机原子；`sources.json` 保存文件名/原子数/SHA256。
每档再加入8个真实 CIF，构成47/100结构的预算退出/补位数据集；
SizeAwareSampler决定初始与到达次序，实际ID/ptr见events；两后端数据完全相同。
checkpoint `/home/wangleping/mace-mpa-0-medium.model`，SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
普通 e3nn eager、enable_cueq=False、compile_model=False；参数冻结，力/应变 autograd 保留。
模型、数据、Torch default dtype 同为 FP64 或 FP32。

两条路径为 **Torch reference cell-list 与当前 M6 HIP**。
邻居 full-PBC/full-list/COO/skin=0，每次当前晶胞重建；不涉及旧 HIP 实现重放。
动力学、stress conversion、FIRE2 AlignCellHook 均 Torch reference。
dt=0.02，FIRE dt_min=0；其他优化器设置沿用现有 M3 公共 API。

```bash
source scripts/activate_hygon_env.sh project
source artifacts/hip-current-cell-neighbor-m1/native_env.sh
export NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR="$HYGON_PROJECT_ROOT/artifacts/hip-cooperative-query-m6/native-candidate"
export HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1
timeout 900 python probes/hip_real_batch_validation.py \
  --checkpoint /home/wangleping/mace-mpa-0-medium.model \
  --data-root /data/csp_data/catl_data/li_mp_data \
  --manifest artifacts/hip-real-batch-m7/sources.json \
  --output-dir artifacts/hip-real-batch-m7/1000-float64-fire2 \
  --atoms 1000 --dtype float64 --optimizer fire2 --lifecycle
```

分别替换atoms=1000/2000、dtype=float64/float32、optimizer=fire/fire2，八组独立进程。
原始 runner/commands/exit status 在 `run_cases.py`、`run-status.json`、`run-env.sh`；
产品/探针 SHA256 在 `source-hashes-v2.json`、`source-hashes-v3.json`；
FP64使用精确归档的`probe-v2.py`，v3只增加失败字段/快照/JSON诊断，计时和容差不变；
query binary在 `native-binary-hashes.json`。

## 计时与正确性协议

同几何初始及第三步后：严格比较有重数的 `(i,j,sx,sy,sz)` 与体系边界，
E/F/stress 沿用 M3 逐物理量容差。先校验，再测 neighbor/model/neighbor+model。
每快照每路径一次预热，3轮×2样本，路径顺序轮换；两快照合计12样本/路径/分项。
完整 step 是3轮各3步，从同初始结构独立 priming，9样本/路径；
含邻居 Hook、模型及优化器更新，不含模型加载、Batch构造、priming、校验、IO或sink。
每步同步墙钟；记录所有样本、median/mean/std/min/max，未开 profiler。
同快照分项与独立轨迹 step 是不同测量，不要求分项加总等于 step。

第一版仅预热一完整步，Torch step2出现 Hook 首次重编译（1771ms）而其他step约550ms。
该版正确性和生命周期通过，但排除稳态比较，保存在 `initial-partial/`。
最终版预热三完整步覆盖所测分支，重新运行全部组合；不删除单个离群样本。
NeighborListHook 源码保留上游 `torch.compile` 装饰，模型本身 eager，
本轮不建立 compile/opcheck 支持声明。预算补位中的shape重编译不混入稳态step计时。

三轮每步核对 mutable fields、优化器私有状态、整数计数/步数、ptr/source IDs；
检查位置/晶胞实际改变。FusedStage 的预算1/2/3退出分别在两后端运行，
比较每次到达 priming 的拓扑/E/F/stress、source/system IDs、ptr和最终逐体系状态。
预算不是物理收敛。私有状态退出冻结/补位重排还由既有定向 CPU/HCU gate 保护。

## M1–M6 一致性审阅

| 阶段 | 当前实现与证据边界 | 审阅结论 |
|---|---|---|
| M1 | stateless 当前r/H metadata与CSR全重建；旧低层plan只对单快照有效 | 保留；历史MATRIX-only契约明确指向后续M2 |
| M2 | registry允许显式periodic/full MATRIX+COO；COO设备内Torch打包 | 无native COO，默认None/Warp及auto过滤保持 |
| M3 | Torch变胞优化器+显式HIP Hook，FusedStage共享compute/补位 | M7扩大验证数据，未增加native dynamics或长收敛 |
| M4 | workspace候选未达到5%门槛并撤回 | 产品无持久缓存，历史失败保留 |
| M5 | inverse/grid/radius/offset设备内批量推导，逐体系缩减规则保持 | 只减少metadata成本，容量有限doubling和错误策略保持 |
| M6 | 每原子64线程block查询，native无序后公共排序；int64 offset guard | 单一HIP产品实现；历史串行binary只用于M6探针 |

锁定上游ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` 的
cell-list kernels/launchers和framework `4dfe3723def34df3fadb245981081ccf8c94c257` 的
邻居/优化器接口来源保持；没有同步external或改变锁定SHA。
历史报告的门槛和范围不覆盖当前较大真实Batch结果，二者独立保留。
M5/M6已知FP32极近cutoff两条边的Torch/HIP舍入分歧保留，不改cutoff/容差。

## 结果

最终八组：**5通过、3失败**；四组FP64全部通过，FP32仅1029原子FIRE通过。
失败的FP32组不作为完整step性能验收；其初始同几何拓扑与逐物理量输出检查通过。

### 通过的组合：稳态墙钟

单位ms，中位数；分项12样本/路径，step9样本/路径。模型耗时几乎相同。

| 原子/结构 | 精度 | 优化器 | 邻居 Torch/HIP | 模型 Torch/HIP | 邻居+模型 Torch/HIP | 完整step Torch/HIP | step倍数 |
|---|---|---|---:|---:|---:|---:|---:|
| 1029/39 | float64 | FIRE2VariableCell | 288.517/7.567 | 257.316/257.329 | 546.802/265.260 | 554.916/272.487 | 2.04× |
| 1029/39 | float64 | FIREVariableCell | 290.382/7.485 | 257.215/257.292 | 548.018/265.339 | 556.413/271.774 | 2.05× |
| 2030/92 | float64 | FIRE2VariableCell | 679.736/8.021 | 477.287/477.286 | 1157.368/485.595 | 1165.159/492.503 | 2.37× |
| 2030/92 | float64 | FIREVariableCell | 669.581/8.044 | 477.301/477.284 | 1148.314/485.825 | 1157.782/492.640 | 2.35× |
| 1029/39 | float32 | FIREVariableCell | 288.858/7.580 | 144.004/144.078 | 433.219/152.040 | 441.354/158.497 | 2.78× |

FP64完整step下降约50.9%–57.7%；三个轮次方向一致，完整step相对std最高0.24%。
该结论只对应39/92个真实晶体的异构Batch、三步短轨迹，不是单个大结构或长收敛吞吐。
通过组的`comparison.json`/`comparison.txt`及每分项`*-comparison.txt`保留mean/std/range、
全部轮次样本；使用performance技能compare_timings.py，不使用首轮或最好样本。

### 显存与数值

| 规模/精度 | 测量点最大 Torch allocated | 最大 Torch reserved |
|---|---:|---:|
| 1029 FP64 | 10.873 GiB | 15.969 GiB |
| 2030 FP64 | 21.031 GiB | 30.742 GiB |
| 1029 FP32 FIRE通过组 | 5.476 GiB | 8.033 GiB |

仅模型/分项/step测量点，不是完整进程或全设备显存；未发生OOM/降精度/容量截断。
四组FP64同几何最大E/F/stress误差：5.68e-13 eV /3.41e-14 eVÅ⁻¹ /2.15e-15 eVÅ⁻³。
三轮的mutable fields、内部优化状态/计数、ptr/IDs、有限值和实际位置/晶胞变化通过。
通过组的预算补位均4步，47/100个真实CIF全部按预算退出；两路径events/最终状态一致。
1029档active atoms为1029→985→604→92；2030档为2029→1878→945→220，
来自SizeAwareSampler的真实初始/补位顺序，非固定N或物理收敛。

### FP32失败及纯Torch对照

原逐物理量比较：FP32 rtol=2e-4，E/F/stress绝对容差分别2e-3 eV、2e-4 eVÅ⁻¹、2e-6 eVÅ⁻³。
随后继承M3的mutable fields检查又对forces施加通用状态atol=2e-5/rtol=2e-4，更严格。
本阶段没有删掉该检查或放宽容差；因此以下组合验收失败：

| 组合 | 严格forces失败数量 | 失败元素最大差异 eVÅ⁻¹ |
|---|---:|---:|
| 1029 FP32 FIRE2 | 1/3087 | 6.55055e-5 |
| 2030 FP32 FIRE2 | 2/6090 | 6.80089e-5 |
| 2030 FP32 FIRE | 1/6090 | 2.17911e-5 |

v2首个1029 FP32 FIRE2失败差异2.16952e-5，保存在`failed-initial/`；v3只补失败诊断后复现。
失败原子/差异会变动。v3 1029 FIRE2失败处ordered edges及shift完全相同，
位置最大差9.54e-7Å、cell5.75e-10Å、速度3.90e-7，均通过状态阈值；失败只在forces重复门。
该处源ID35，force actual0.0705391、reference0.0706046 eVÅ⁻¹。

额外`probes/mace_fp32_state_control.py`：相同几何12次纯Torch模型输出，
E/F/stress最大差3.052e-4/1.641e-5/9.919e-7，全部通过逐物理量容差。
两条**纯Torch reference邻居**FIRE2轨迹，3轮各3步，round0 step3也触发相同严格forces门：
一个元素差3.02911e-5（其他8步通过），逐物理量E/F/stress检查全部通过。
该对照无需HIP即可出现失败，说明不能把这个失败直接归因于HIP查询；
输入完全相同的模型输出存在FP32浮点波动是实测事实，具体底层归约非确定性的归因仍是推断，未做kernel级因果证明。
证据`torch-fp32-control.json/.pt/.log`、`failure-snapshot.pt`、`failed-force-diagnostic.json`。
未测失败组后续补位/末步快照，不把初始同几何通过扩大为全组通过。

### 回归门

- CPU独立进程：ops **51 passed/34 HIP skipped**，framework **21 passed/28 HIP skipped**。
- HCU4独立进程：ops **85 passed**，framework **49 passed**；包括CPU oracle和真实HIP项，非全库gate。
- Gate路径清单、完整日志、退出码：`m6-gate-paths.json`、`cpu-*.log`、`hcu-*.log`、`hcu-gates-exit.json`。
- 两探针compileall、特性契约JSON唯一键/唯一feature ID、git diff --check通过；Ruff未安装，未安装新工具。

## 阶段结论与下一步

M7实验与审阅完成；**FP64较大真实Batch窄验收通过，FP32综合状态验收未全通过**。
维持单一HIP后端和既有未准入边界。M7未提交/合并/推送。
建议下一M8先澄清验收分层：模型E/F/stress使用既有逐物理量容差，几何/速度/私有状态
继续原状态容差，纳入Torch/Torch对照并重跑FP32与FP64回归；不修改HIP、模型或全局容差。
这是对forces重复门的契约修订建议，须等待用户确认；本阶段不自动实施。

## 剩余范围

不代表长步物理收敛、20k真实模型吞吐、完整上游/HCU gate、其他MLIP、训练、
compile/opcheck、checkpoint、DomainParallel、no-PBC/half、skin/selective rebuild、
旧plan变胞复用、native COO/geometry backward或auto准入。
