# ADR 0005：集中能力 registry 与存储默认值

日期：2026-09-08。状态：采用。

## 决策

`nvalchemiops.backend` 是所有计算后端选择的唯一权威。它以 operation、device、dtype、梯度等级和特性集合解析已验证实现，并返回可记录的 `BackendSelection`。framework 只传递请求和执行已选择的 legacy Warp 路径；不得再维护独立字符串集合或 `_select_backend`。

`backend=None` 与 `backend="warp"` 保持上游 Warp 默认，并且 resolver 不导入、不探测 Warp。`backend="auto"` 仅在显式请求时，从已注册且能力匹配的实现选择；首次选择每个 operation/设备/dtype/特性签名会发出 `BackendAutoSelectionWarning`，其中包含实际后端与原因。当前只有 Torch reference 可被 auto 选择；Triton/HIP 名称已知但没有已登记能力时明确失败。

公共计算入口优先使用 `backend`。`BaseModelMixin.make_neighbor_hooks(neighbor_backend=...)` 保留为兼容别名；与 `backend` 同时给出且不一致时失败。`LoggingHook.backend` 仍是日志输出后端，因此 `compute_backend` 保留；periodic/observer 的 `compute_backend` 同样委派给 registry。

`LevelStorage` 的默认 `TorchStorageBackend` 是有意的跨平台数据层契约，而非临时无 Warp 回退。它保证 `AtomicData`/`Batch` 基础导入不初始化 Warp，并保持 target-device Torch 语义。`WarpStorageBackend` 仅作为显式 NVIDIA 对照实现；两者等价性在可用 Warp 环境以同一存储契约测试。

## 后果

新增 Triton、HIP 或 DTK 实现必须先登记精确 capability，再可参与 `auto`；不能因设备名或存在一个 kernel 自动启用。registry 的矩阵和证据边界见 `docs/BACKEND_CAPABILITY_MATRIX.md`。

storage 的默认选择与计算组件不同：前者是数据模型的无加速器基础实现，后者的 `None` 仍遵从上游 Warp。该差异必须在 API、测试和发布说明中明确，不能用环境探测偷偷改变。
