# Team development baseline v0.1

日期：2026-09-09
候选分支：`team/dev-baseline-v0.1`（从 M1 follow-up `fb11528` 建立；B0 gate 后已按用户授权合入并同步）

## 实现范围

- 默认 `ImplementationRegistry` inventory 已迁入私有、metadata-only 的
  `nvalchemiops._backend_catalog`，按 legacy、neighbors、interactions、dynamics、observability
  分组；登记 ID、顺序、能力与 executor 字符串保持原语义。
- 新增 `scripts/check_cpu_reference.sh`：以独立 pytest 进程运行 ops 与 framework 的
  reference golden-path 回归，并执行 compileall 和 `git diff --check`。
- 新增 `scripts/check_hcu_reference_smoke.sh`：要求调用者显式设置 `HIP_VISIBLE_DEVICES`，对
  neighbor/LJ、PBC neighbor、VV、kinetics、FIRE/FIRE2 依次运行已有 probe；不会选择设备或
  回退到 CPU。
- 新增基础协作说明与新 Torch operation 模板。产品功能、能力宽度和
  `FEATURE_COMPATIBILITY.yaml` 状态均未扩大。

## 验证记录

在项目 `.venv`、DTK 26.04、`OMP_NUM_THREADS=1` 下运行
`scripts/check_cpu_reference.sh`，退出码为 `0`：

- ops registry/reference/cell-list：`27 passed, 1 warning`；warning 是已登记的 explicit
  `backend="auto"` 选择记录；
- 无 eager dynamics 导入边界：`1 passed`；framework main golden-path suite：
  `121 passed, 1 deselected`（该导入边界项在独立进程运行）；
- state lifecycle 的 NVE/FIRE/FIRE2 子集：`22 passed, 34 deselected`；上游
  VV/FIRE/FIRE2 op 子集：`24 passed, 73 deselected`；
- `compileall` 与 `git diff --check` 均退出 `0`。

以未设置 `HIP_VISIBLE_DEVICES` 运行 HCU script 的退出码为 `1`，在加载环境或运行 probe 前
明确报错要求分配设备。这验证了脚本的失败保护，但不是 HCU 测试。

随后在主机权限终端、DTK 26.04、项目 `.venv`、`OMP_NUM_THREADS=1` 和空闲 HCU 0 上运行：

~~~bash
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 scripts/check_hcu_reference_smoke.sh
~~~

退出码为 `0`。五个 probe 均通过：neighbor→LJ 的 full/half 能量和力、periodic
neighbor 的有效矩阵/周期 shift、Velocity Verlet、kinetics、FIRE/FIRE2，设备为
`BW200, UBB BW1000`。periodic probe 的矩阵容量为 `[2, 16]`，有效邻居数为 `[1, 1]`；
该结果也修正了原 probe 对“容量列数必须等于有效邻居数”的过严断言。此次 HCU smoke
只验证 B0 golden paths 的设备回归，不产生生产性能或完整 feature support 结论。

本次 B0 HCU gate 的可重跑命令为：

~~~bash
HIP_VISIBLE_DEVICES=<assigned> scripts/check_hcu_reference_smoke.sh
~~~

## 后续动作

CPU/HCU gate 通过后，`team/dev-baseline-v0.1` 已 fast-forward 合入 `develop`（集成提交
`da394b6`）。下一步优先从 `TORCH-NEIGHBOR-PBC-CELL`、`TORCH-NVT-LANGEVIN`、`TORCH-NVT-NHC` 中选择一个独立任务；
M2 profile/planner 继续延期。
