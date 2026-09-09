# B1 executor binding

日期：2026-09-09
开发分支：`codex/refactor-executor-binding`
状态：CPU verified；B0 式候选 HCU golden-path smoke verified；未合入、未推送共享分支

## 目标与实现

B1 将 `Implementation.executor` 从只写 metadata 变为可执行绑定，解决 catalog 声明与
framework/ops dispatcher 分散 `if-elif` 之间的漂移和多人并行冲突。公共 operation API、
`backend=None` 的 legacy Warp 语义和当前 capability 宽度没有改变。

- `Implementation` 非 legacy 条目声明 dotted executor module、entrypoint 元组和
  `executor_owner`（`ops`/`framework`）；legacy 条目为 framework-owned、`executor=None`、
  空 entrypoints。
- `load_entrypoint` 是独立的 lazy loader，校验 selection/metadata、owner、entrypoint 声明和
  callable，并缓存已加载 callable。导入失败现在标注 `ops-owned package` 或
  `framework-owned package`。
- `execute_selected` 是 operation-neutral adapter：只做一次 legacy implementation 边界，
  其余情况加载并调用声明的 entrypoint。operation-specific legacy handler 仍在各 dispatcher
  内以局部闭包提供；adapter 没有 operation 分派表。
- VV、FIRE/FIRE2、LJ、neighbors、periodic、kinetics/segmented 和 observer 已完成声明式
  接线；`torch_backend.py` 保留为兼容导出 shim。
- `test_executor_binding.py` 检查 framework-owned entrypoint importability 和 dispatcher
  静态 guard；ops 测试通过一个临时登记的第二 neighbor implementation 实际调用 unchanged
  dispatcher，验证了 R1 的 ABI 调用约束，而非只验证 import/callable。

对应提交（按实施顺序）：

`90c43fb` → `98da458` → `0b1698c` → `0f8b9dd` → `f5eb064` → `efedb58`

## CPU 验证

环境：项目 `.venv`、Python 3.12.13、DTK 26.04 环境脚本、`NVALCHEMI_TEST_BACKEND=torch_reference`、
`OMP_NUM_THREADS=1`。正式命令：

```bash
scripts/check_cpu_reference.sh
```

退出码：`0`。

- ops registry/reference/cell-list：`32 passed, 1 warning`；warning 是已登记的
  `backend="auto"` 选择提示。
- framework golden path（含 executor-binding import/ABI static guard）：`123 passed,
  1 deselected`。
- state lifecycle NVE/FIRE/FIRE2 子集：`22 passed, 34 deselected`。
- 上游 VV/FIRE/FIRE2 operation 子集：`24 passed, 73 deselected`。
- `compileall` 和 `git diff --check`：退出码 `0`。

R5 的缺包归因回归另行确认：framework-owned executor 的故意导入失败包含
`framework-owned package`。同一 generic loader 对 ops-owned 条目使用对应 owner 标签。

## 数值结果

本轮没有新增物理算法或 capability 宽度，因此没有新增数值结论。现有 CPU reference golden
path 回归保持通过，覆盖 neighbors/LJ、固定晶胞 VV/FIRE/FIRE2、periodic、kinetics/observer
的既有契约；这些结果证明绑定迁移未改变已保护的 CPU reference 行为，不等于新增 HCU、Triton
或 HIP 生产实现。

## HCU 验证

按 B0 候选集成模式，在本地 `team/b1-executor-binding-candidate` 指针对应提交、主机权限、
DTK 26.04、`HIP_VISIBLE_DEVICES=0`、空闲 HCU 0 上运行：

```bash
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 scripts/check_hcu_reference_smoke.sh
```

退出码：`0`。五个 probe 均通过，设备为 `BW200, UBB BW1000`；neighbor/LJ、periodic、VV、
kinetics、FIRE/FIRE2 均报告 `warp_imported=false`。代表性结果包括：LJ full/half energy
`-0.6236757013081533`，periodic neighbor matrix shape `[2, 16]` 且有效邻居数 `[1, 1]`，VV
最大 position error `6.938893903907228e-18`。这只是当前 golden-path 窄 slice 的设备回归。

受限 sandbox 首次运行同一脚本以退出码 `1` 报 `HIP device required; no silent CPU fallback`；
该载体隐藏 HIP 设备节点，不能作为 HCU 能力失败。主机权限重跑才是本次 HCU 证据。

## 未验证、限制与下一步

- HCU 证据不等于完整 DCU production support、Triton/HIP 实现、性能结论、变胞、分布式或
  全部 framework compile 支持；HCU 失败仍不得切换 CPU。
- 本轮没有扩大 `FEATURE_COMPATIBILITY.yaml` 的功能宽度；它继续描述既有能力和限制。
- compile 回归覆盖当前 reference 载体；这不是完整生产后端 compile、Triton/HIP、变胞、
  分布式或性能证据。
- 人工 review 完成后，下一项是 `TORCH-NVT-LANGEVIN`。周期 cell-list 和
  `TORCH-NVT-NHC` 保持独立 T1 任务；M2 PlatformFingerprint/Profile/Planner 继续延期。

设计决策见 [`adr/0007-declarative-executor-binding.md`](../adr/0007-declarative-executor-binding.md)。
