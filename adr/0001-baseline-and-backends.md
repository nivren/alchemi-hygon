# ADR 0001：完整上游基线与隔离探针

日期：2026-09-05。状态：采用。

按用户 SHA 不 squash subtree 导入完整两包，导入与适配分开提交。静态依赖交集和显式跨包符号解析满足导入门槛；目标设备运行是独立验收，不能据此声明支持。

生产路线为海光 Torch、Torch reference、按能力/基准选择 Triton/HIP；Warp 保留上游来源但不作为 DCU 可用前提。不安装 CUDA extras，不用 stub 冒充 PhysicsNeMo，不替换系统 Torch。探针 .venv 使用本地海光 wheel 与精确 constraints；后续产品安装须再次审计依赖解析。

当前 8 卡存在满载作业，无资源分配，不启动计算。HIP 可进行无设备的编译检查，CPU 探针仅作为显式 CPU 证据。双卡通信通过也不等于域分解通过。
