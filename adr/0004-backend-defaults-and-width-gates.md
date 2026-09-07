# ADR 0004：Warp 默认路径、reference 旁路与特性宽度门槛

日期：2026-09-06

## 决策

上游默认路径继续保留 Warp。`backend=None` 不改变原有公共 API 的默认行为；在没有明确可用的 Warp 环境时，调用应显式失败或由用户选择适配后端，不能因为设备字符串包含 `cuda` 就静默切换。

`backend="torch_reference"` 是旁路入口。它用于定义可验证的 Torch 语义、CPU/HCU 正确性回归和中小输入 fallback，不替换上游 Warp 默认，也不代表完整生产性能。当前 reference 已覆盖的范围必须逐条写入兼容清单；未覆盖的参数继续显式报错。

`backend="auto"` 只对明确选择它的组件生效。当前适配组件在没有已注册优化后端时选择并报告 `torch_reference`，这是临时 resolver 行为，不是全局默认。后续建立中央 backend registry 后，`auto` 必须根据构建元数据、运行时能力、输入规模、dtype、梯度等级、PBC/Batch 特征和实测基准选择，并返回实际后端与选择原因。

Triton、HIP 和 DTK 库后端只能在逐算子注册后参与 `auto`。一个后端进入 registry 前必须同时满足：

1. shape、dtype、layout、单位、PBC、full/half、排序、空输入、容量、mutation/alias、stream 和确定性契约与 reference 对齐；
2. forward、一阶梯度以及需要时的二阶/混合二阶梯度通过 fake/meta、opcheck、gradcheck/gradgradcheck 和物理量对照；
3. 在 gfx936/BW200 上有可重跑正确性证据，并记录冷启动、预热和端到端成本；
4. 对不支持的输入明确拒绝，不能截断邻居、降精度、漏算相互作用或静默 CPU 回退；
5. 该算子所有已登记宽度特性均有覆盖，不能只因窄 smoke 通过就宣称宽路径完成。

## 特性宽度排期

邻居和 LJ 的窄 reference 路径已形成 G1 正确性基线，但以下能力各自是独立特性，不能合并成一个“LJ 已支持”：

- `neighbors.skin_rebuild`：Torch reference 已支持 Batch 缓存、变化 system 的 eager 局部重建，以及 staging K 维的 16 对齐 grow-and-retry、idle shrink 和 override floor；G1 仍记录为 partial。生产 cell-list、异步调度、PBC 容量压力和规模化长轨迹性能另行排期。
- `neighbors.periodic_reference_assembly`：periodic full-list reference 的 pair/image 几何候选保持逐 system 计算，MATRIX/COO 装配使用 device-side `nonzero`、行内 rank 和 scatter；该窄 slice 的 CPU/HCU 语义与批量回归已通过，但不改变 `neighbors.topology` 的完整状态，也不覆盖 no-PBC、half-list 或 cell-list。
- `interactions.lj_switching`：当前 reference 明确拒绝非零 `switch_width`；按解析能量、力和 cutoff 连续性单独实现、测试并登记。
- `interactions.lj_virial_stress`：当前 reference 明确拒绝 virial/stress；按 cell/position 梯度、符号、归一化、单位和 batch 归约单独实现、测试并登记。

每一项完成后分别更新 `docs/FEATURE_COMPATIBILITY.yaml` 的状态、证据和 release gate；G1 只接受已明确范围内的 partial/implemented 结果，G2 才评估生产 cell-list、优化后端和完整 dynamics 链。

## PhysicsNeMo 边界

`nvidia-physicsnemo` 与 Warp/NVIDIA runtime 绑定，不作为 HCU 单进程 Torch/MACE 的默认安装依赖。其 profiling、域并行和 vendored ShardTensor 能力保留公共名称与显式导入路径；项目基础导入和单进程 reference 路径通过可选/延迟导入运行。PhysicsNeMo 替代方案分别按 Torch profiler、RCCL/torch.distributed 和 HCU-specific domain ownership 任务推进，不能用移除导入来宣称这些能力已移植。

## 后果

上层用户体验和 Warp 兼容性得到保护，reference 可以持续提供 DCU 正确性基线；代价是后端选择和特性状态需要显式报告，且窄路径不会自动扩大为完整支持。任何新增 Triton/HIP kernel 都必须在该边界下提交独立契约、测试和基准。
