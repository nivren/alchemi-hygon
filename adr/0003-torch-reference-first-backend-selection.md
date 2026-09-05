# ADR 0003：Torch reference 优先与按算子选择优化后端

日期：2026-09-05。状态：采用。

## 决策

每个需要移植的算子先写一个只依赖海光 PyTorch 的 reference 实现。它必须在 CPU 和 HCU 上使用同一套 shape、dtype、布局、空输入、容量、PBC/邻居和梯度契约，并通过解析结果或 CPU FP64 结果校验。reference 不是把 HCU 计算静默搬回 CPU 的回退：输入在哪个设备，计算就在哪个设备；不支持的参数明确抛出异常。

Triton 和 HIP 暂不作为本阶段的实现前提。只有当 reference 语义、梯度和端到端链路稳定，并且有代表性基准显示某个算子需要优化时，才为该算子增加候选后端。后端选择按能力过滤后再按可复现基准决定，不能因为设备字符串包含 `cuda` 就固定选择某一种语言，也不能假设 HIP 总是快于 Triton。

| 场景 | 首选优化候选 | 选择依据 |
| --- | --- | --- |
| 规则的 tile、逐元素融合、连续归约 | Triton | gfx936 编译能力、dtype/梯度覆盖、稳态吞吐和启动成本 |
| 不规则邻居、动态容量、显式原子/线程控制、迁移和通信打包 | HIP C++ | 原子与同步语义、扫描/排序接口、DTK 库互操作和长尾输入 |
| FFT、扫描、排序、通信 | DTK 可用库 | 复用已验证实现，比较端到端和通信成本 |
| 小输入、冷启动、未覆盖梯度或失败能力探针 | Torch reference | 正确性和可观测回退；报告实际设备与后端 |

`backend="torch_reference"` 是当前唯一实现的显式选择；未来的 `triton`、`hip` 和 `auto` 必须在同一算子契约下接入。`auto` 需要返回或记录实际后端、设备、输入特征、能力探针版本和选择原因。优化后端不满足能力或梯度条件时，只有明确配置允许的 reference 回退可以发生；不得静默改变设备、精度、邻居集合或梯度等级。

## 分步实施

1. 为一个纵向链路建立 Torch reference（当前为 no-PBC neighbors → LJ energy/force），覆盖 full/half、异构 batch、容量溢出和一/二阶梯度检查。
2. 将 reference 接入稳定的公共 dispatcher，并在结果或诊断接口中暴露后端信息；Warp 仍保持独立的 NVIDIA 参考路径。
3. 在真实 gfx936 输入上测量冷启动、预热稳态、端到端时间和显存/通信成本，再为已证明的瓶颈逐算子实现 Triton 或 HIP。
4. 每个优化后端通过 reference 对照、gradcheck/gradgradcheck（适用时）、空输入/容量/PBC 测试后，才进入 `auto` 候选。

## 后果

初期吞吐可能低于 Warp 或未来的 Triton/HIP，但每一步都有可运行的数值基线，能在 HCU 上验证真实设备路径，也避免为迁移而重写上层 API。reference 的 `O(N²)` 邻居实现只适合第一条正确性链路；规模化前必须用基准决定 cell-list、Triton tile、HIP 不规则 kernel 或 DTK 库的边界。
