# 验证报告索引

`reports/` 保存可提交、脱敏的验证摘要；原始日志、轨迹和性能样本在被忽略的
`artifacts/` 中。报告是证据，不是当前支持状态的唯一入口。

## 当前先读

| 主题 | 当前摘要 | 主要限制 |
|---|---|---|
| A–D 与补测集成收口 | [`g2-upstream-sync-v1-closure-2026-10-09.md`](g2-upstream-sync-v1-closure-2026-10-09.md) | 签名成果与CPU复验已集成本地develop；用户授权直接三处同步，精确SHA/退出码见交付核验；候选保持Candidate |
| 隔离升级 D（2026-10） | [`upstream-sync-2026-10.md`](upstream-sync-2026-10.md) | 三候选原样模块 CPU/HCU4 各11项检查通过；stock导入及依赖/API/24个文本冲突阻塞；暂不整体升级，仍Candidate |
| FIRE2 + HIP 邻居完整弛豫补测 | [`g2-hip-fire2-convergence-2026-10-09.md`](g2-hip-fire2-convergence-2026-10-09.md) | 固定三Li/Na CIF，HCU4两精度/两邻居分支四作业，全部双阈值收敛及真实补位/终态/导出核验；非大Batch或普遍收敛保证 |
| UBF 对齐 C（2026-10） | [`upstream-bugfix-alignment-2026-10.md`](upstream-bugfix-alignment-2026-10.md) | 16 项官方证据/额外边界；全部 retain；PR194 本地科学案例 6 passed；候选/HCU/Warp 未运行，原 MACE 模块收集失败单列 |
| Upstream Delta V1（2026-10） | [`upstream-delta-2026-10.md`](upstream-delta-2026-10.md) | 冻结两包SHA、分叉、API/依赖/特性和本地交集；14项工具夹具通过，候选源码/HCU未运行；生成时的 UBF 线索，当前对齐见 C |
| G2 集成基线验收 A（2026-10-09） | [`g2-integration-a-2026-10-09.md`](g2-integration-a-2026-10-09.md) | `2e211ab` 指定 CPU/HCU4 与真实MPA-0八组通过；非全上游/Warp/长收敛；长期跟踪流程见 docs/UPSTREAM_SYNC.md |
| 项目环境 | [`probe-environment-freeze.txt`](probe-environment-freeze.txt) | 机器事实和历史资源快照；新机器需重新确认 |
| Torch reference 基线 | [`g2-torch-reference-pbc-cell-list-core.md`](g2-torch-reference-pbc-cell-list-core.md) | 不代表生产性能后端 |
| HIP 邻居阶段二收口 | [`g2-framework-hip-neighbor-compatibility-gate.md`](g2-framework-hip-neighbor-compatibility-gate.md) | 当时覆盖 periodic/fixed-cell/Batch/full-list/MATRIX 窄范围 |
| HIP 邻居当前晶胞 M1 | [`g2-hip-current-cell-neighbor-m1.md`](g2-hip-current-cell-neighbor-m1.md) | FP32/64 当前晶胞完整 MATRIX 重建、CPU/HCU4；COO/MACE/FIRE 工作流与性能待后续验收 |
| HIP 邻居 COO/MACE M2 | [`g2-hip-coo-mace-snapshot-m2.md`](g2-hip-coo-mace-snapshot-m2.md) | 通用设备内 Torch COO 适配、真实MPA-0两精度单点及FP64差分；FIRE/FIRE2与性能未验收 |
| HIP 邻居变胞工作流 M3 | [`g2-hip-variable-cell-workflow-m3.md`](g2-hip-variable-cell-workflow-m3.md) | CPU/HCU4短步/状态生命周期；真实MPA-0两精度/两优化器五步及预算补位；无长步收敛或性能结论 |
| HIP 邻居性能/workspace M4 | [`g2-hip-neighbor-workspace-m4.md`](g2-hip-neighbor-workspace-m4.md) | 分项/hipprof/45样本配对；workspace候选未达5%门槛已撤回，产品仍stateless；批量metadata结果见后续M5 |
| HIP 邻居批量metadata M5 | [`g2-hip-batch-metadata-m5.md`](g2-hip-batch-metadata-m5.md) | FP32/64约20,000原子变胞/补位三路径、真实MPA-0短轨迹；所测门槛通过，FP32 cutoff边界限制单列；实现已收口e2d569c |
| HIP 邻居查询协作 M6 | [`g2-hip-cooperative-query-m6.md`](g2-hip-cooperative-query-m6.md) | 每原子64线程block；两精度约20,000原子变胞三路径，few FP64下降11.34%；真实短轨迹/生命周期通过，fe18147/c2a8339 收口 |
| HIP 邻居真实Batch M7 | [`g2-hip-real-batch-validation-m7.md`](g2-hip-real-batch-validation-m7.md) | FP64真实1029/2030原子两优化器完整step约2.04–2.37倍；FP32严格forces门未全通过、纯Torch也失败；提交 `00637c0` |
| MPA-0 FP32 状态验收分层 M8 | [`g2-hip-mpa0-state-validation-m8.md`](g2-hip-mpa0-state-validation-m8.md) | 修正E/F/stress重复状态比较；真实HCU八组6组完整通过，另两组FP32 FIRE2仅未满足最终46/47、99/100晶胞移动断言；未改产品代码或容差 |
| FP32 FIRE2 晶胞移动断言 M9 | [`g2-hip-fp32-fire2-cell-movement-m9.md`](g2-hip-fp32-fire2-cell-movement-m9.md) | 两组未移动项均为 Li3Ge3(MoO6)2，HIP/Torch reference 晶胞完全相同；改为批次至少一个晶胞移动，受影响两组复跑通过 |
| HIP framework wall-clock | [`g2-framework-hip-neighbor-wallclock.md`](g2-framework-hip-neighbor-wallclock.md) | warm public API 描述性 evidence，不是 `auto` 准入 |
| MACE/FIRE2 reference | [`g2-mace-fire2-batch32-tier1.md`](g2-mace-fire2-batch32-tier1.md) | 固定晶胞 reference；不代表变胞或生产 HIP/MACE |
| MPA-0 弛豫 M1 | [`g2-mace-mpa0-m1-cpu-baseline.md`](g2-mace-mpa0-m1-cpu-baseline.md) | 三个真实 CIF 的 CPU 单点与有限差分；目标 HCU/变胞弛豫未验证 |
| MPA-0 弛豫 M2 | [`g2-cell-tools-m2.md`](g2-cell-tools-m2.md) | 通用晶胞工具 CPU/HCU4 primitive；未含优化步 |
| MPA-0 弛豫 M3 | [`g2-variable-cell-fire-m3.md`](g2-variable-cell-fire-m3.md) | 原生变胞 FIRE2/fork FIRE、CPU/HCU解析势及MPA-0 CPU三步；完整应用/目标HCU待M4/M5 |
| MPA-0 弛豫 M4/M5 | [`g2-mpa0-relaxation-m4.md`](g2-mpa0-relaxation-m4.md)、[`g2-mpa0-relaxation-m5.md`](g2-mpa0-relaxation-m5.md) | 原生变胞批量生命周期与目标模型单卡窄验收；预算/精度敏感性单列，非完整HCU gate |
| MPA-0 FP64单Batch耗时 | [`g2-mpa0-single-batch-timing.md`](g2-mpa0-single-batch-timing.md) | 64结构/2287原子应用101秒；1600是条件外推，原生raw力判据与M5/H100不同 |
| MPA-0 8000门槛分项诊断 | [`g2-mpa0-batch8000-timing.md`](g2-mpa0-batch8000-timing.md) | 231结构/7937原子初始e3nn模型前向OOM，未完成弛豫；分项仅失败前部分，非稳态耗时 |
| MPA-0 4000门槛完整分项 | [`g2-mpa0-batch4000-timing.md`](g2-mpa0-batch4000-timing.md) | 114项/3959原子、111收敛/3预算、应用160秒；模型76%/Torch reference邻居19%/优化器3%，非满载稳态 |
| FIRE 零步长下限兼容修正 | [`g2-fire-zero-dt-min.md`](g2-fire-zero-dt-min.md) | LOCAL-004，本地校验恢复上游 `dt_min=0` 语义；CPU回归/HCU4 FP64 单步、解析势及三CIF十步 |
| MPA-0 弛豫移植收口审查 | [`g2-relaxation-port-closure-review.md`](g2-relaxation-port-closure-review.md) | 对照 M1–M5/LOCAL-004 证据、应用实际参数、可复现入口与 H100 配置差异 |
| Langevin reference | [`g2-torch-nvt-langevin.md`](g2-torch-nvt-langevin.md)、[`g2-torch-nvt-langevin-stat.md`](g2-torch-nvt-langevin-stat.md)、[`g2-torch-nvt-langevin-restart.md`](g2-torch-nvt-langevin-restart.md) | 固定晶胞 BAOAB 窄 reference slice 已完成；不等于完整上游 Langevin/NVT 或生产后端 |
| NHC reference | [`g2-torch-nvt-nhc.md`](g2-torch-nvt-nhc.md) | 固定晶胞 Torch reference、批处理生命周期和单 HCU 短 MACE；Warp/NVIDIA、分布式、完整 checkpoint、生产后端未验收 |
| Stress/cell-force 基础 | [`g2-torch-cell-stress-force.md`](g2-torch-cell-stress-force.md) | CPU/HCU reference、周期 MACE 应力与 FP64 有限差分；不包含晶胞更新、变胞 FIRE/FIRE2、NPT/NPH 或完整 MLIP 训练 |
| Hygon distributed preflight | [`dcu-distributed-preflight-2026-09-21.md`](dcu-distributed-preflight-2026-09-21.md) | P0/P1 单卡、P2/P3 backend、P4/2-rank、P5/P6/P7/P8/P9 correctness、P10 baseline、P11 async/stream correctness、P12 operation-count stability、P13 memory behavior、P14 topology record 通过；P11 未观察到稳定 overlap 收益，DomainParallel 未验证 |

## 邻居 cell-list 报告分组

- Torch reference：`g1-*neighbor*`、`g2-torch-reference-cell-list-*`、
  `g2-neighbor-*`。
- HIP build：`g2-cell-*`、`g2-batch-cell-*`、`g2-native-batch-cell-build-*`。
- HIP query/materialization/geometry：`g2-batch-query-*`。
- runtime/framework 接线：`g2-fixed-cell-*`、`g2-native-hip-neighbor-runtime-*`、
  `g2-framework-hip-neighbor-*`。
- 阶段二完整收口的读取顺序：先读 compatibility gate，再按需要追溯上述分组中的单点报告。

## 其它证据

- backend/registry：`g2-backend-registry-*`、`b1-executor-binding.md`；
- dynamics/reference：`g2-dynamics-*`、`g2-fire2-*`、`g2-upstream-*`；
- MACE/轨迹/端到端：`g1-mace-*`、`g2-mace-*`、`g2-heterogeneous-*`；
- 统一基线：`g2-unified-reference-benchmark-*`；
- 初始审计与导入：`g0-*`、`g1-*`、`upstream_inventory.json`。

## 维护规则

1. 当前支持范围写入 `docs/STATUS.md`、`docs/BACKEND_CAPABILITY_MATRIX.md` 和
   `docs/FEATURE_COMPATIBILITY.yaml`，不要只在报告中宣布支持。
2. 一个独立可审查里程碑原则上只保留一个 summary report；过程拆分报告只有在具有独立
   correctness、性能或故障定位价值时才新增。
3. 报告中的设备号、版本和历史耗时是事实证据，不为匹配当前设备而改写；新的本机单卡
   验证统一使用 `HIP_VISIBLE_DEVICES=4`。
4. 新报告必须写清实现、测试、数值结果、性能口径和未验证范围，并链接对应源码/探针。
