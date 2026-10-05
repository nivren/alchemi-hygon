# 验证报告索引

`reports/` 保存可提交、脱敏的验证摘要；原始日志、轨迹和性能样本在被忽略的
`artifacts/` 中。报告是证据，不是当前支持状态的唯一入口。

## 当前先读

| 主题 | 当前摘要 | 主要限制 |
|---|---|---|
| 项目环境 | [`probe-environment-freeze.txt`](probe-environment-freeze.txt) | 机器事实和历史资源快照；新机器需重新确认 |
| Torch reference 基线 | [`g2-torch-reference-pbc-cell-list-core.md`](g2-torch-reference-pbc-cell-list-core.md) | 不代表生产性能后端 |
| HIP 邻居阶段二收口 | [`g2-framework-hip-neighbor-compatibility-gate.md`](g2-framework-hip-neighbor-compatibility-gate.md) | 当时覆盖 periodic/fixed-cell/Batch/full-list/MATRIX 窄范围 |
| HIP 邻居当前晶胞 M1 | [`g2-hip-current-cell-neighbor-m1.md`](g2-hip-current-cell-neighbor-m1.md) | FP32/64 当前晶胞完整 MATRIX 重建、CPU/HCU4；COO/MACE/FIRE 工作流与性能待后续验收 |
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
