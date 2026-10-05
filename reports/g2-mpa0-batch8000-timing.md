# FP64 MPA-0 原生FIRE：8000原子门槛与分项计时诊断

日期：2026-10-05。用户要求max_atoms提高到8000，仍只执行一个完整Batch，
并估计模型、邻居、优化器等各部分耗时。
上一阶段2304容量测量已按用户追加任务确认收口为 `7410445`；
分支 `codex/test-mpa0-batch8000-timing` 起点为该提交；用户随后确认4000门槛测量，
本阶段已收口 `eff753d`，未合并或推送。

**结果：231结构/7937原子的初始模型求值OOM，8000门槛下完整弛豫未完成。**
80%、95%和默认100%三个显存上限尝试均在第一次e3nn张量积前向中失败；
均没有完成模型求值、没有执行优化器更新、没有产生弛豫结果。
这些失败不能用来估算完整优化循环各部分占比或1600项耗时。

## 输入、模型和算法边界

当前 `/data/csp_data/catl_data/li_mp_data` 共21755候选，文件名排序后
Random(24).sample(...,1600)保留返回顺序，和上一阶段使用同一抽样清单。
取不超过8000原子的最长前缀：**231个结构、7937个原子、每项3–176原子**；
下一项80原子会使总数超过8000，故未纳入。没有拆分结构、预弛豫或标准化。
当前候选数、抽样前缀和全部231项输入SHA256核验通过。
1600项仅为选样清单，未运行全部1600；仍与用户H100候选集合不同。

HCU4 BW1000 64G/gfx936，设备实际可报告63.98 GiB；
Torch2.9.0/HIP6.3.26093、DTK26.04、单进程/worker/OMP=1、无MPS。
作业前、各次失败后和最终均只读检查资源；无其他KFD PID，退出后显存回收、卡空闲。

MACE0.3.15/e3nn0.4.4，MACE MPA-0-medium模型SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
模型/结构/默认dtype均FP64，eval且冻结权重，力/应力仍需位置和应变autograd。
enable_cueq=False、compile_model=False；运行时实际子模块包含
`e3nn.o3._tensor_product._tensor_product.TensorProduct`、
`e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct` 和
`mace.modules.symmetric_contraction.SymmetricContraction`，没有cuequivariance子模块。
因此本次确实采用普通e3nn/MACE实现，没有接入HCU等变GNN加速算子库。
e3nn张量积内部有TorchScript/FX生成代码；关闭compile_model不等于禁止这些上游内部机制。

配置仍为原生完整9项FIREVariableCell、真实质量/晶胞质量1、零外压；
dt=.1、dt_max=1、dt_min=.002、maxstep=.2、n_min=5、f_inc=1.1、
f_dec=.5、alpha_start=.1、f_alpha=.99。
原生raw原子力与`F_H=-V*inv(H).T@stress`的全部行最大二范数<=.05收敛，>100溢出，
没有独立应力阈值。每项500更新预算，refill20，最大结构数231，邻居6Å每次重建，
COO容量动态分配、逐原子超过512报错，不截断。步间墙钟1800秒、外层1920秒；
实际均约32秒即失败退出，没有长时间占用。
本地dt_min=0拒绝问题尚未修正；UnitCellFilter、CuEq和H100判据均未用于此次测量。

## 仅探针的分项观测

改动 `probes/mace_mpa0_batch_timing.py`：支持清单里的任意单批结构数、
`--components`和显式`--memory-fraction`。
没有修改framework/ops/产品应用入口，没有新增科学算法、filter或上游bug修复。
给当前实例的现有compute及masked更新调用加计时包装，邻居和停止观测器保持原stage/frequency。
返回值、异常和原有Hook顺序保持；计时未对张量做detach、降精度或更换backend。

边界前后同步设备，记录互不嵌套的墙钟及成功/失败调用数：

| 分项 | 范围 |
|---|---|
| model_evaluation | FusedStage.compute：模型输入适配、能量前向、力/应力autograd、输出回写和释放梯度图 |
| neighbor_list / neighbor_capacity_check | Torch reference邻居重建、当前每原子512上限检查 |
| optimizer_pre_update / optimizer_post_update | 原生FIRE晶胞力及更新，含初始化、masked私有状态保护和恢复 |
| native_force_observer | 本探针原生广义力停止指标和有限性/溢出检查 |
| workflow_other_seconds | 循环墙钟减上述分项，含初始Batch构建、Hook、退出/清空及同步边界等剩余开销 |

准备、启动/oracle、导出另列。同步和包装有测量开销，不能称为纯kernel/device时间，
不能与上一阶段未经分项同步的101秒做严格吞吐比较。
此次先在初始priming求值失败，后三类的更新/停止计算并未执行。

## 三次显存预检结果

为避免把人为限额当成设备物理容量结论，先用上一阶段80%allocator限额，
失败且卡仍空闲后以95%重试，最后同一批在默认100%上限下试一次。
未修改模型、dtype、输入、邻居或算法，也未拆成模型微批/多卡/CPU计算。
三次仅是同一批初始求值预检，均不计为完成一批弛豫。

| Torch allocator上限 | 首次邻居重建秒 | 失败前部分模型秒 | 峰值allocated GiB | 峰值reserved GiB | 探针总秒 |
|---|---:|---:|---:|---:|---:|
| 80%（约51.19 GiB） | 2.736 | 0.639 | 47.128 | 50.795 | 31.874 |
| 95%（约60.79 GiB） | 2.016 | 0.679 | 56.293 | 59.779 | 31.249 |
| 100%（默认上限约63.98 GiB） | 2.007 | 0.694 | 59.115 | 62.604 | 31.939 |

默认上限失败现场：e3nn生成的张量积前向，
`torch.einsum('dbc,dca->dba', ...)`，尝试再申请3.95 GiB；
报错时设备空闲3.25 GiB、Torch当前allocated56.86 GiB、reserved但未allocated2.92 GiB。
这里报错瞬时值与阶段峰值不同，不混用。测试未改变allocator策略，
不能排除碎片影响，也没有测出该模型理论最小显存或整个目录通用容量上限。
实测结论仅为：**这批7937原子在当前默认allocator/FP64/e3nn配置中OOM**。
从OOM发生位置能确认失败落在模型张量积计算，不能据此推导完整循环的性能占比，
也不能量化未来HCU等变算子库的速度或显存收益。

默认上限这一轮已执行部分的墙钟：

| 部分 | 秒 | 解释 |
|---|---:|---|
| 导入等启动 | 19.540 | 第三方导入/哈希等固定准备 |
| CPU/HCU oracle及设备初始化 | 3.604 | 正确性检查，非应用吞吐 |
| 输入/模型/工作流准备 | 4.986 | 231份CIF与模型等准备 |
| 初始邻居重建 | 2.007 | 第一次冷调用，不是稳态每步均值 |
| 邻居容量检查 | 0.000322 | 第一次调用通过 |
| 部分模型前向至OOM | 0.694 | 未完成E/F/stress，不能当完整模型推理时间 |
| 循环中其余开销 | 1.107 | 初批构建/Hook/包装边界等残差 |
| 失败后诊断导出 | 0.000410 | 空结果及诊断，未导出弛豫CIF |

上述启动/oracle/导出数字以原始timing.json为准；合计约31.94秒，
其中初始求值阶段约3.808秒。没有完整应用端到端时间、收敛统计或优化器分项均值。
不把模型未完成时的0.694秒作为完整每步时间，也不据此更新1600项的粗估。
此前约41–50分钟估算仍仅针对2304容量、2287原子实测批次的条件外推。

## 验证与证据

CPU：原生晶胞力NumPy oracle通过；mock设备同步验证计时包装的返回值、异常透传、
成功/失败计数和Hook stage/frequency保持。Ruff/格式/py_compile/git diff --check通过。
HCU：三次oracle各3案例通过；三个初始Batch均邻居检查通过、模型OOM、求值/更新完成数0。
CPU事后核验输入SHA256、清单/容量、三个失败诊断和分项加残差等于循环总时间；
没有弛豫CIF、JSONL为空，不输出未求值的几何作为“优化完成”。
未运行完整CPU/HCU产品gate；已有M5数值/工作流证据保持，本次是窄容量失败诊断。

原始 `artifacts/mpa0-batch8000-timing/`：selection.json、资源检查、
fp64-native-batch8000{,-cap95,-cap100}.log及同名目录timing.json，
verification.json聚合各尝试并记录全部231项输入哈希核验。
模型和私人结构不入库。

复现默认上限尝试（卡须空闲，输出目录须不存在）：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 1920 .venv/bin/python \
  probes/mace_mpa0_batch_timing.py \
  --selection artifacts/mpa0-batch8000-timing/selection.json \
  --checkpoint "$MACE_MPA0_CHECKPOINT" \
  --output artifacts/mpa0-batch8000-timing/reproduce-cap100 \
  --max-atoms 8000 --wall-seconds 1800 --components --memory-fraction 1
```

下一阶段建议（待确认）：将单批门槛降低到约4000原子，仍取同一抽样前缀、FP64/e3nn/
原生FIRE，先做显存预检，能容纳后只跑一批完整弛豫，拿到模型/邻居/优化器等完整分项。
用户确认后先提交本阶段，再实施；FIRE dt_min=0本地兼容修正仍列后续独立任务。

后续记录：用户确认4000门槛单Batch测量后，已取得完整弛豫和分项，见
[4000门槛报告](g2-mpa0-batch4000-timing.md)。8000失败结论仍保留。
