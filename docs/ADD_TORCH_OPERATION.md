# 新增 Torch operation：最小闭环

本指南服务于基础开发版本 v0.1。它刻意只规定能让不同开发者安全并行的最小步骤；详细的
物理与后端约束以 [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) 和 `AGENTS.md` 为准。

1. 在锁定上游源码中定位符号、测试与公共行为，写出小型契约：输入/输出、shape、dtype、
   layout、单位、PBC/full-half、空输入、容量/错误、mutation、stream、确定性与 forward/一阶/
   二阶梯度。
2. 先实现设备内 Torch reference，并写 CPU 解析或上游 oracle 对照。连续量不要 `detach`；
   拓扑不可微时也要明确切断位置。
3. 在 `nvalchemiops._backend_catalog` 对应操作族添加唯一、稳定的 `Implementation` metadata。
   catalog 只能写 metadata 和 executor 路径字符串，不能导入 Torch/Warp/HIP executor；新 family
   或新的 capability 宽度必须有失败测试。
4. 由 framework 在组件/工作流生命周期中解析一次 `BackendSelection`，传给 dispatcher；
   dispatcher 按精确 implementation ID 执行，不重新解析原始 backend request。
5. 保留 `backend=None` 的 legacy Warp 行为。显式 `torch_reference` 不满足契约时抛出
   `BackendUnavailableError` 或 operation 定义的明确错误，绝不能改成静默 CPU 回退。
6. 将 operation 测试接入 `scripts/check_cpu_reference.sh`（或在同一脚本中可见地增加单独进程）。
   在分配的设备窗口运行 `scripts/check_hcu_reference_smoke.sh` 或该 operation 的专用 HCU
   probe；记录命令、设备、退出码和数值范围。
7. 更新 `FEATURE_COMPATIBILITY.yaml`、`STATUS.md` 和 report。没有 HCU 实测就写 CPU verified /
   HCU pending；接口、选择语义或支持范围改变时补 ADR。

提交应只覆盖一个操作或一个可审查的接线步骤。若实现需要 planner、全局 policy、周期
cell-list、跨 rank 或生产性能选择，则先停止并在交接中提出独立设计任务，不把它塞进 reference
小改动。
