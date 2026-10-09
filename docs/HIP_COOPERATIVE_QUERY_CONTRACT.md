# HIP 查询并行实验契约（M6）

2026-10-06 用户确认；M5实现 `e2d569c`、文档 `3073f63`，
分支 `codex/perf-hip-cooperative-query`。实施前建立本契约。

## 来源与假设

锁定ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`，
`neighbors/cell_list/kernels.py`、`launchers.py`：默认atom_centric/direct
每原子一线程遍历cell/原子；sorted路径先gather；pair_centric按源cell/offset块
并行遍历原子对，原子更新输出。不是所有Warp查询都同一并行方式。

本工程M5基线 `_native/batch_cell_query.cu` 当时每原子一线程。M5独立trace中
200结构FP32查询约2.641ms，公共API约10.617ms。
假设：每原子一个64线程block，线程分担候选cell，共享计数原子预留输出位置，
可减少串行枚举成本；同时可能因线程/原子开销、分支和共享原子退化，必须实测。
不直接移植整个Warp pair-centric/sorted pipeline。

## 不变的语义

接口、shape/dtype/layout、row-cell Å、严格d²<cutoff²、full-PBC/full邻居、周期
自镜像保留；只排除零shift自对，活动重叠报错。FP32/64算术次序和编译flags不变。
平展offset数量和loop增量超int64时明确失败，不能wrap漏算。
query输出无序，公共materialize维持确定排序。候选不越界写；容量不足明确报错，
上层既有扩容/重建继续。空N、mutation/alias检查、当前stream和错误同步保持。
拓扑forward-only；连续geometry/模型autograd不改变。每调用当前cell全重建。
不新增native COO、skin、workspace、auto、no-PBC/half公共准入、DD或compile。

## 验收

先CPU reference及现有边界测试，再实际HCU ops/framework/模型生命周期回归。
补充并行计数、非整块原子数、重复公共输出确定性及溢出测试。
基线native源码存artifacts，核对缓存源hash；用不同扩展名重新编译并断言模块独立，
只在探针中切换baseline/candidate，
切换不计时，正式产品仍一个HIP后端。

三路径同几何：Torch reference cell-list、M5 HIP、M6候选；200/40/10结构
约20,000原子及小Batch，FP32/64，六变胞/补位快照，三轮×三样本/路径/快照。
真实MPA-0两精度FIRE/FIRE2五步同快照邻居/模型/邻居+模型；另验实际预算补位。
已知FP32 cutoff极近边界失败保留并诊断，不放宽检查。

预先目标：few（10结构，平均2000原子）FP64公共邻居API相对M5至少下降5%，
三轮方向一致；其他配置不出现稳定超过5%退化且正确性通过才保留。
无profiler最终wall计时，分项/trace独立；未达则撤回产品候选并保留实验。
当前：实现及窄验收完成，目标few FP64三轮均超过5%，八配置无稳定退化；
CPU ops51/framework21、HCU环境ops85/framework49及实际模型/生命周期/低层回归通过。
保留单一HIP实现，实现收口 fe18147、文档 c2a8339；已随 M1–M9 提交链集成到 `develop`；数据、失败和剩余边界以M6报告为准。
