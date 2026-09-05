# ADR 0002：保留数据模型、替换 Warp 存储后端

日期：2026-09-05。状态：采用。

## 决策

`AtomicData`、`Batch`、`LevelSchema`、`MultiLevelStorage`、`SegmentedLevelStorage` 和 `UniformLevelStorage` 继续作为上层公共数据模型与存储语义。它们不改为 HCU 专用类，也不向用户暴露 `warp.Array` 或 `wp.launch`。

Warp 代码只作为一种执行后端。masked copy、fit mask、defrag 和 segment expansion 通过无硬件绑定的 `StorageBackend` 协议调用；Torch reference 先完整定义可验证语义，Triton/HIP 按 gfx936 上的实际基准逐项替换，Warp 后端保留为 NVIDIA 参考实现。

后端必须保持以下契约：atoms/edges/system 分层、segment lengths 与 batch pointers、neighbor list 索引偏移、空输入、容量边界、dtype/shape/layout、原位 mutation、alias、device 和 stream 语义。未支持的显式后端必须 fail-fast；不得静默搬到 CPU、截断数据或改变精度。

`nvalchemi.data` 和 `nvalchemiops` 不得在基础导入阶段初始化 Warp。Warp 模块只在显式选择 Warp 后端时延迟加载。`backend="auto"` 的实际选择必须可报告，并依据构建元数据、运行时设备和能力探针，而不是仅检查设备字符串是否包含 `cuda`。

## 分步实施

1. 提取无 Warp import 的 `StorageBackend` 协议，冻结现有 buffer kernel 的操作签名。
2. 将现有 Warp 函数包入 Warp backend，并为协议补充 Torch reference 实现。
3. 让 LevelStorage 通过组合调用 backend，移除顶层 Warp 初始化，加入无 Warp import smoke 和 Batch 语义测试。
4. 在 gfx936 上先验证 Torch reference，再按算子和基准实现 Triton/HIP；最后接入自动选择和性能策略。

## 后果

这会增加一个内部后端边界，但保留上游 Python API、数据布局和 inflight 生命周期，允许逐算子迁移和回归。Torch reference 可能暂时慢于 Warp；这是可观测的设备内基线，不是静默 CPU 回退。Warp 仍可在独立 NVIDIA 环境使用，但不再成为 HCU 基础数据 API 的导入前提。
