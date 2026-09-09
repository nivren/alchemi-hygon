# B1 executor binding

日期：2026-09-09  
开发分支：`codex/refactor-executor-binding`  
状态：CPU verified；B0 式候选 HCU gate pending；未合入、未推送共享分支

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

## 未验证、限制与下一步

- B1 HCU gate 尚未在 B0 式候选集成指针上运行，当前不能写成 HCU verified 或 DCU production
  support。候选 gate 命令为：

  ```bash
  HIP_VISIBLE_DEVICES=<assigned> scripts/check_hcu_reference_smoke.sh
  ```

- HCU 失败不得切换 CPU；应记录设备、环境、退出码和具体 probe 失败。没有新的 HCU 记录前，
  `FEATURE_COMPATIBILITY.yaml` 保持不变。
- compile 回归覆盖当前 reference 载体；这不是完整生产后端 compile、Triton/HIP、变胞、
  分布式或性能证据。
- B1 候选 gate 和人工 review 完成后，下一项是 `TORCH-NVT-LANGEVIN`。周期 cell-list 和
  `TORCH-NVT-NHC` 保持独立 T1 任务；M2 PlatformFingerprint/Profile/Planner 继续延期。

设计决策见 [`adr/0007-declarative-executor-binding.md`](../adr/0007-declarative-executor-binding.md)。
