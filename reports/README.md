# 验证报告索引

`reports/` 保存可提交、脱敏的验证摘要；原始日志、轨迹和性能样本在被忽略的
`artifacts/` 中。报告是证据，不是当前支持状态的唯一入口。

## 当前先读

| 主题 | 当前摘要 | 主要限制 |
|---|---|---|
| 项目环境 | [`probe-environment-freeze.txt`](probe-environment-freeze.txt) | 机器事实和历史资源快照；新机器需重新确认 |
| Torch reference 基线 | [`g2-torch-reference-pbc-cell-list-core.md`](g2-torch-reference-pbc-cell-list-core.md) | 不代表生产性能后端 |
| HIP 邻居阶段二收口 | [`g2-framework-hip-neighbor-compatibility-gate.md`](g2-framework-hip-neighbor-compatibility-gate.md) | 只覆盖当前 periodic/fixed-cell/Batch/full-list/MATRIX 窄范围 |
| HIP framework wall-clock | [`g2-framework-hip-neighbor-wallclock.md`](g2-framework-hip-neighbor-wallclock.md) | warm public API 描述性 evidence，不是 `auto` 准入 |
| MACE/FIRE2 reference | [`g2-mace-fire2-batch32-tier1.md`](g2-mace-fire2-batch32-tier1.md) | 固定晶胞 reference；不代表变胞或生产 HIP/MACE |
| Langevin reference | [`g2-torch-nvt-langevin-restart.md`](g2-torch-nvt-langevin-restart.md) | 仍是窄 reference lifecycle slice |

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
