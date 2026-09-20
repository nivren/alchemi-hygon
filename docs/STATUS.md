# 当前开发状态

更新时间：2026-09-20。本文只维护当前状态、已验证范围、下一步和限制；完整历史交接保留在
[`docs/history/STATUS.md`](history/STATUS.md)。

## 当前基线

- 当前分支：`develop`。
- 阶段二实现提交：`9f53f80`；上一轮文档收口提交：`9e295ea`；当前文档重构提交：`97325a4`。
- 两个产品远端 `local-origin/develop`、`github-origin/develop` 均已同步到 `97325a4`。
- 上游 framework/ops 锁定 SHA 见 [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml)；产品源码位于
  `packages/framework` 和 `packages/ops`，`external/` 只作只读参考。

## 当前能力摘要

| 功能 | 当前状态 | 已验证范围 | 明确限制 |
|---|---|---|---|
| Torch reference neighbor/cell-list | 已实现窄 reference | periodic/no-PBC、full/half、MATRIX/COO、mixed/triclinic Batch、image shift、容量、selective rebuild、连续 distance/vector 一二阶路径 | 不等于完整上游生产 cell-list；完整 target/pair/compile/opcheck/DomainParallel 仍未完成 |
| 显式 HIP neighbor/cell-list | 已实现并验证窄 capability | periodic、fixed-cell、Batch、full-list、MATRIX、FP32/FP64、`skin=0`、`max_neighbors=None` 本地 doubling capacity | 不进入 `auto`；不支持 no-PBC、half、COO、skin/rebuild、target/pair、变胞、native geometry backward、compile/opcheck、DomainParallel |
| Torch geometry | 当前训练/梯度路径 | distance/vector 连续坐标的一阶/二阶路径 | native HIP geometry 仍是 forward-only 性能候选 |
| Langevin | 已完成窄 reference slice | 固定晶胞 BAOAB、float32/64、异构普通 Batch、空输入/错误、短统计 oracle、最小 integrator continuation state；当前 CPU 复核 `15 passed` | 不等于完整上游 Langevin/NVT；通用 checkpoint、`atom_ptr`/`_out`、inflight、分布式、完整行为套件、torch.compile、生产 Triton/HIP 未完成 |
| fixed-cell dynamics | reference slice | VV、FIRE、FIRE2、kinetics 及部分公共 wrapper | NHC、NPT/NPH、变胞 stress/FIRE2、生产后端未完成 |

能力宽度以 [`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md) 和
[`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml) 为准；本表不把窄 slice 扩大为完整
上游支持。

## 阶段二收口证据

- Ops focused suite：`102 passed`。
- Framework 邻居/LJ focused suite：`19 passed, 1 warning`。
- HCU：gfx936/BW200，`HIP_VISIBLE_DEVICES=4`，46×32 compatibility gate 通过；FP32/FP64、
  empty、capacity overflow、unsupported request、重复 Hook 和自动容量增长均有证据。
- 自动容量 fixture：初始容量 `1` 增长到 `8`，最大实际邻居数 `6`，保守上界 `343`；active
  neighbor 集合与 Torch reference 一致。HIP padded capacity 形状不承诺与上游最小 16/16
  对齐策略一致。
- 当前阶段二 summary：[`reports/g2-framework-hip-neighbor-compatibility-gate.md`](../reports/g2-framework-hip-neighbor-compatibility-gate.md)。
  相关报告按主题见 [`reports/README.md`](../reports/README.md)。

## Langevin 窄 slice 收口证据

- 公共固定晶胞 BAOAB reference、registry/executor 接线和 contract：
  [`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)。
- 独立谐势短统计 oracle：[`g2-torch-nvt-langevin-stat.md`](../reports/g2-torch-nvt-langevin-stat.md)。
- 普通 Batch 的最小 integrator continuation state：
  [`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md)。
- 当前工作树 CPU 复核：三个兼容性文件合计 `15 passed`；历史 HCU 证据保留原报告中的真实卡号。

## 运行时排障记录

此前“加载/设备同步在当前卡上异常变慢”已定位为：超时终止 JIT 编译后遗留 PyTorch C++
extension `FileBaton` stale lock，后续进程在加载阶段等待。确认没有属于本任务的
`ninja`/`hipcc`/应用进程后清理该缓存 lock，query extension 恢复正常加载，完整 46×32 HCU
gate 重跑通过。长时间验证和发布路径优先使用 AOT 或受控 JIT cache。

## 当前下一步

具体 owner 和分支需用户确认后再启动。当前计划见
[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)：

1. 固定晶胞 Nose-Hoover chain reference；
2. `TORCH-CELL-STRESS-FORCE` 与 coupled FIRE2 variable-cell；
3. 固定晶胞 ASE 3.29-compatible BFGS，变胞 ASE 语义另列子里程碑。

暂不启动 M2 planner、NPT/NPH、DomainParallel、生产 Triton/HIP kernel 或扩大当前 HIP
neighbor capability。

## 验证和设备约定

- CPU gate：`scripts/check_cpu_reference.sh`。
- 本机所有后续单卡 HCU 验证统一使用 `HIP_VISIBLE_DEVICES=4`；历史报告中真实使用的 HCU 0
  结果保留，不改写。
- 受限沙箱中 `/dev/kfd`、`/dev/dri` 不可见只表示设备节点隔离，不能作为 HCU 能力结论。
- 实现、测试、数值结果、性能口径和未验证假设必须分别记录；没有 HCU 运行证据不能写成
  DCU verified。

## 文档权威关系

- 架构与稳定交接：[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)。
- 当前任务队列：[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)。
- 后端能力：[`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md)。
- 逐特性契约：[`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)。
- 上游来源/补丁：[`UPSTREAM.md`](UPSTREAM.md) 与 [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml)。
- 历史日志：[`docs/history/STATUS.md`](history/STATUS.md)、
  [`docs/history/PROJECT_HANDOFF.md`](history/PROJECT_HANDOFF.md)、
  [`docs/history/PARALLEL_DEVELOPMENT_PLAN.md`](history/PARALLEL_DEVELOPMENT_PLAN.md)。
