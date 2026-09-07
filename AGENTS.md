# 海光 DCU 原子模拟框架：项目指令

本文件应放在海光服务器项目根目录。它是本项目的长期约束与启动入口，不是已经完成的开发记录。交接日期：2026-09-05。

## 1. 项目目的与工作方式

基于 NVIDIA `nvalchemi-toolkit` 和 `nvalchemi-toolkit-ops`，在海光 DCU 上建立易用、可维护、可扩展的化学与材料原子模拟框架。科学用户应能够完成大量分子动力学、结构弛豫、单个大体系多卡计算，以及 MLIP 训练、微调和评估。

项目的核心要求是尽可能保留上游框架的先进特性、公共 API 和用户体验。完整继承代码基线后，在清楚的后端边界内做适配。不要从空白重写一个只保留少量功能的相似框架，不要逐文件随意复制，不要为让示例通过而删除高级功能或原测试。

与用户用中文沟通。实现、测试、数值结果和未验证假设必须分别报告。读取本文件后，再读 `docs/PROJECT_HANDOFF.md`、`docs/STATUS.md`。涉及首次部署时读 `docs/START_HERE.md`。需要背景细节时读 `docs/海光DCU移植开发路线与工程规范.md`。

后者是历史调研建议，其中版本、工期、人员数量和平台支持不是已确认事实。与历史建议有差异时，按本文件和 PROJECT_HANDOFF 中明确的修订执行；实际源码和实测证据用于更新判断。用户的新指令优先。

## 2. 已确定的路线

- 上层以海光提供的 PyTorch 为基座，保留 AtomicData、Batch、模型包装器、Dynamics、FusedStage、Hooks、训练与分布式接口。
- PyTorch reference 定义可验证的算子语义。适配版 Triton 处理适合分块、融合和归约的计算；HIP C++ 处理需要显式线程控制的不规则关键路径。FFT、扫描排序、通信优先复用目标 DTK 可用库。
- 不以 Warp 在 DCU 可用作为生产开发前提。Warp 可保留为隔离的 NVIDIA 参考后端。ROCm Warp 预研可选、限时，不阻塞 Torch/HIP 路线。
- 后端按输入规模、dtype、梯度等级、目标设备与基准选择。不要固定写成 HIP 总是优于 Triton，也不要把所有 Warp kernel 机械翻译为 Triton。
- 保留 `toolkit` 与 `ops` 两个包边界，默认一个 Git 主仓库、两个可独立构建发布的 Python 包。前期保持上游 Python 导入命名空间，实际名称以锁定源码为准，发行包品牌稍后确定。
- PyTorch 前端优先。JAX、多极长程、高级编译路径可延期，但必须在功能契约中记录，不能悄悄删除。

## 3. 目录和 Git 边界

```text
项目根目录/                  # 主 Git 仓库；从这里启动 Codex
  AGENTS.md
  .gitignore
  external/
    nvalchemi-toolkit/        # NVIDIA 完整 clone，参考用途，主仓库忽略
    nvalchemi-toolkit-ops/    # NVIDIA 完整 clone，参考用途，主仓库忽略
  packages/
    framework/               # 完整上游基线导入后的正式开发代码
    ops/                     # 完整上游基线导入后的正式开发代码
  probes/                    # 隔离的环境、kernel、梯度和双卡探针
  tests/compatibility/       # 跨包特性兼容测试；原测试仍在各包中
  tests/numerics/
  tests/integration/
  tests/distributed/
  benchmarks/
  examples/
  configs/
  containers/
  docs/
  adr/
  reports/                   # 可提交的脱敏验证摘要
  artifacts/                 # 不入库的原始输出、轨迹和性能数据
```

`packages/framework`、`packages/ops` 在导入前不要创建占位文件。默认使用不 squash 的 Git subtree 完整导入，保留来源历史；具体步骤见 START_HERE。不要在 `packages/` 内再次创建嵌套 `.git`，不要把 external 当作产品安装源。不得对 external 做产品补丁、全局格式化或覆盖文件。上游版本不随 main 自动漂移。

导入前记录 URL、SHA、许可证、两个包之间的依赖约束；核实选定组合可兼容。导入与功能修改分成独立提交。保留 LICENSE、NOTICE、SPDX 等，遵循实际文件而非猜测许可证。

## 4. Feature Compatibility Contract

先盘点上游 API、示例、配置、测试与高级特性，再按三类记录到 `docs/FEATURE_COMPATIBILITY.yaml`：

- A：上层逻辑优先原样保留，例如数据语义、模型接口、Hooks、工作流状态和序列化。A 不代表已在 DCU 验证。
- B：公共接口与语义保持，通过替换底层实现适配，例如邻居表、积分更新、可微算子、设备通信和存储预取。
- C：明确延期或实验性支持，例如某些模型加速依赖、JAX、高阶长程与编译优化。必须保留来源追踪、支持状态、失败行为与后续计划。

每条特性至少记录：稳定 ID、用户场景、上游 SHA/符号/路径、A/B/C、接口和数据约定、依赖、后端、梯度等级、测试、DCU 验证状态、证据路径、限制与发布阶段。分类与测试状态独立；`planned / implemented / verified / blocked / deferred` 不能混用。没有真实运行证据时不得写 verified。

重点保护：Batch 体系边界和索引、FusedStage 阶段迁移、inflight 补位、Hooks 时序、模型组合、轨迹与 checkpoint、训练力损失、DomainParallel 原子归属与通信语义。

## 5. 后端和数值约束

- 不根据设备名含 `cuda` 就认定 NVIDIA。HIP 版 PyTorch 也可能使用 `torch.cuda`。综合构建元数据、运行时探针和实际设备识别；不要全局替换 CUDA 字符串。
- 不向上层新增 `warp.Array` 或直接 `wp.launch` 依赖。逐步隔离已有耦合，采用可选导入。审计 PhysicsNeMo、cuEquivariance 等的具体用途后再适配，避免整块删除相关框架能力。
- 算子契约包含 shape、dtype、layout、单位、PBC、half/full neighbor、排序、空输入、mutation/alias、stream、确定性和梯度。
- 通过 PyTorch 自定义算子机制接入 Triton/HIP。按实际 PyTorch 版本验证 fake/meta、autograd、opcheck 等支持；eager 正确后再验证 compile。
- 区分 forward-only、一阶、二阶梯度。邻居索引/拓扑通常不可微，位移、距离、能量等连续路径保持梯度。力损失常涉及混合二阶导数，不得靠 detach 或零梯度掩盖未实现路径。
- 不允许静默 CPU 回退、降精度、截断邻居、漏算相互作用或关闭高级特性。显式启用的回退须记录实际后端、设备与性能影响。
- 容差按算子、dtype 和物理量定义；不要通过放宽全局容差让测试变绿。邻居容量溢出须扩容/重建或明确报错。

## 6. 第一次执行的具体任务

用户说“开始初始化/按文档开始”时，默认执行以下首轮任务，不仅输出计划：

1. 读取交接文件并检查 Git 状态、现有源码和已有环境。已完成步骤不重复，不覆盖用户工作。
2. 做只读服务器盘点：DCU 型号和数量、系统、驱动/DTK、Python、PyTorch、Triton、HIP 编译器、通信库及可用 GPU 资源。按本机已安装工具探测，不假定厂商命令名和目录。
3. 验证两个 external clone 的来源、完整历史和依赖兼容性。记录选定 SHA，创建 `docs/UPSTREAM.md` 与 `docs/UPSTREAM_LOCK.yaml`。未能锁定时先完成不依赖版本选择的任务。
4. 完成源码依赖/特性/测试盘点，写 `FEATURE_COMPATIBILITY.yaml`、`ENVIRONMENT.md` 和 `PROBE_PLAN.md`。不能只生成空模板，至少从真实源码填充 Batch、FusedStage、Hooks、neighbors、LJ、训练、域分解条目。
5. 在项目隔离环境编写并运行最小探针：Torch 设备张量及梯度、HIP 简单 kernel、Triton vector add（可用时）、一个邻居或 segment 样例、一个真实 MLIP 的前向/梯度；有两卡资源时验证 all-reduce 与点对点交换。安装前先确认不会替换海光 Torch。缺环境或资源时记录具体阻塞，继续其余任务。
6. 基线兼容组合确认且主仓库干净后，按 START_HERE 完整导入两个包。导入前已有产品代码则评估合并路径，不重复导入或覆盖。
7. 在 `STATUS.md` 写入真实结果、可重跑命令、退出码、证据位置、下一任务。提出第一条可实现的纵向功能链路及对应兼容测试。

此首轮任务的完成标准：能明确判断哪些平台能力可用，锁定或说明为何不能锁定上游组合，产生可重跑探针和真实功能清单。首轮不要求完成全部移植。继续开发时按 PROJECT_HANDOFF 的 G1/G2 等验收逐步推进。

## 7. 测试、资源和持续交接

保留原测试，区分纯框架、数值、NVIDIA 专属和 DCU 新增测试。不能无理由全局 skip/xfail；xfail 应 strict 并关联原因、跟踪项和解除条件。尚无 DCU 通过结果的原测试不算已保护特性。

数值证据依次来自解析案例、Torch/CPU FP64、可获得的 NVIDIA 基线、端到端物理验证。没有 NVIDIA 环境时记录 fixture 缺口，不能伪造比较。优先验证邻居集合/PBC、能量与力、能量漂移、FIRE 收敛、重启、体系 ID 和跨卡守恒。

性能必须实测，区分冷启动/JIT、预热后的执行、端到端成本和通信。共享服务器先确认空闲资源/作业分配；探针限时限量，不升级系统驱动/DTK，不杀他人任务，不自行占满所有 GPU。长作业、系统级安装和外部发布按用户授权执行。

每轮更新 STATUS；影响接口/后端/支持范围时更新特性契约和 ADR；同步上游时更新 UPSTREAM。一次修改聚焦一个功能或算子，保留可审查 diff。新分支默认 `codex/` 前缀，不自动推送。

任何“已实现/已支持”的交付说明都要说明验证范围和剩余限制。当前实测范围以
`docs/STATUS.md`、`reports/` 和对应原始 `artifacts/` 为准；不能把某个窄 slice 的结果扩大
成完整 DCU 后端支持。

## 8. 开发者与 Agent 的工作模式

根目录 `AGENTS.md` 是唯一的项目级 Agent 入口；不要另建内容重复的 `Agent.md`。
完整的环境操作见 [`docs/DEVELOPMENT_ENVIRONMENT.md`](docs/DEVELOPMENT_ENVIRONMENT.md)，
算子、后端、功能和 Git 协作规则见 [`docs/DEVELOPMENT_GUIDE.md`](docs/DEVELOPMENT_GUIDE.md)。

每次开始工作时按以下顺序执行：

1. 读取本文件、`docs/PROJECT_HANDOFF.md`、`docs/STATUS.md`；涉及环境时再读部署指南和
   `docs/ENVIRONMENT.md`。
2. 运行 `git status --short --branch`、查看最近提交和目标文件 diff。工作树已有改动时，
   不使用 `reset --hard`、`checkout --`、全库格式化或覆盖式复制；先避开无关文件，无法避开
   时向用户报告。
3. 从锁定 SHA 对应的 `packages/` 源码、测试和 docstring 建立事实；`external/` 只读，不能
   直接修改，也不能把探索环境结果写成当前产品证据。
4. 先写算子/功能契约和最小 reference 测试，再实现后端，再接 framework；每完成一轮同步
   `FEATURE_COMPATIBILITY.yaml`、`STATUS.md` 和必要的 `UPSTREAM.md`/ADR。

后端修改必须遵守：

- `torch_reference` 是当前正确性基线，优先保持设备内 Torch 张量路径；不把 `.cpu()`、
  Python 逐元素循环或 detach 当作默认生产方案，不允许静默 CPU 回退或静默改 dtype。
- Triton 适合经实测确认的规则分块、融合和归约；HIP 适合需要显式线程、原子、复杂不规则
  访问或通信打包的路径。按输入规模、dtype、梯度等级、设备和 benchmark 选后端，不能用
  固定的 `hip > triton > torch` 排序。
- 新后端先在 ops dispatcher/backend registry 中登记，再由 framework 显式传递；`None`/默认
  仍表示上游默认路径，`auto` 只有在能力过滤和证据满足后才能选择优化后端。未知或未注册
  后端必须报错，不能回退到另一个实现而不记录。
- 使用 `torch.library.custom_op` 时同时考虑 `mutates_args`、fake/meta、autograd 和 compile；
  eager 正确不等于一阶、二阶梯度或 compile 已正确。力损失必须实际反向到模型参数。

新算子或功能至少要有：上游来源和符号、输入输出/shape/dtype/layout/单位、PBC 与
full/half neighbor 约定、空输入/容量/错误、alias/mutation/stream/确定性、梯度等级、CPU
reference、HCU smoke、回归测试、探针报告和兼容性条目。不能只补一个“能跑”的示例。

Git 协作默认使用 `codex/<topic>` 分支、短而单一目的的提交和 `git commit -s`。正常工作按
路径显式 `git add`，不使用 `git add -A` 吞入他人改动；导入、实现、测试、文档尽量分开提交。
提交前运行 `git diff --check`、相关 CPU 测试和可获得的 HCU 探针，并在交接中分别报告实现、
测试、数值结果与未验证假设。不要自动推送、不要改写共享提交历史；同步上游前先建立单独
分支并更新 `UPSTREAM_LOCK/UPSTREAM.md`。
