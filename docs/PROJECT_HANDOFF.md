# 项目背景、兼容性契约与实施路线

这份文档汇总此前讨论中与开发有关的需求和约定。演示稿排版及代码展示要求不属于产品需求。
当前已完成的移植、实测证据和暂停点以本文末尾的本轮交接及 `docs/STATUS.md` 为准；
早期章节中的“未完成”描述属于历史路线背景，不覆盖后续状态记录。

## 1. 用户真正需要的产品

用户希望以 NVIDIA 开源项目为基础，建立海光 DCU 原生的、组织良好的化学材料原子模拟计算框架。核心场景如下。

| 场景 | 用户希望得到的能力 | 验证重点 |
|---|---|---|
| 大量独立小/中体系 | 批量 MD、批量弛豫、不同原子数混合批次、完成后补位 | 吞吐、体系 ID、收敛状态、结果与输入正确对应 |
| MD 与弛豫工作流 | NVE/NVT、FIRE，逐步扩展到 NPT/晶胞优化等 | 数值稳定、流程组合、轨迹、异常检测、重启 |
| 一个超过单卡容量的大体系 | 空间域分解、halo、原子迁移与通信 | 跨域相互作用、全局量、粒子守恒、扩展效率 |
| MLIP 推理和训练 | 接入现有模型，配置能量/力/应力目标，微调和评估 | 可微路径、模型依赖、数据管线、checkpoint、DDP |
| 长期研发维护 | 稳定接口、独立模块、可回归测试、持续跟随上游 | 来源可追踪、补丁可审查、功能状态透明 |

用户明确关心：移植不能丢失 nvalchemi 的先进性、特性和易用性。因此“运行一个最小 LJ 示例”只是开发验证链路，不能作为整个产品的完成标准。

## 2. 原项目的可继承部分

下列名称基于聊天期间的公开文档调研，实际模块路径、导出符号和支持状态必须从锁定提交核实。

- `AtomicData` / `Batch`：以图结构统一原子属性、体系属性、晶胞、边与索引，使不同大小的结构共享批量推理/模拟接口。不能简化成等长堆叠而丢失异构批次语义。
- 模型包装器与配置：输入输出适配、能量/力/应力约定、能力协商、cutoff、邻居要求、物理模型组合。保留科学用户接入自有 MLIP 的能力。
- `BaseDynamics` / `FusedStage`：组织积分和弛豫阶段，管理各体系的状态与迁移。`FusedStage` 是单设备流程组织，不能自动等同于 GPU kernel fusion。
- inflight batching 与容量感知采样：已完成体系退出，新体系填入，维护 system ID、原子/边容量和生命周期，避免空转或错误重用状态。
- Hooks：邻居重建、收敛检测、轨迹、安全检查、统计与性能监测。调用阶段、频率、异常传播和资源释放均属兼容行为。
- 数据管线与 Zarr：训练数据、轨迹存储、预取、变换和重放。设备预取中的 stream、生命周期和同步需要平台验证。
- 训练：energy/force/stress loss、验证、EMA、checkpoint 和分布式训练。具体功能按源码确认，不能将历史建议中的 FSDP 当成已验证上游能力。
- 分布式：批量流水线、训练数据并行和单体系域分解服务不同需求，分别登记和测试。

`nvalchemi-toolkit-ops` 是底层算子库。主要家族包括 neighbors、dynamics、interactions、math 与 Torch/JAX 绑定。算法和测试应继承，Warp 类型和 launcher 等硬件绑定通过后端边界替换。

## 3. 已讨论的技术取舍与修订

### 3.1 PyTorch、Triton、HIP 与 Warp

PyTorch 承载模型、自动微分、数据和训练。Torch reference 是完整数值语义的基线，也可以作为明确启用的设备内功能回退，不等于默认把计算搬到 CPU。

Warp 支持 Python kernel、JIT、线程索引、不规则访问及反向代码生成；Triton 偏重 program/tile 的张量分块表达。两者能力有重叠，Warp 也有 tile 能力，不能以“一个高一级、一个低一级”概括全部特征。更易表达不规则逻辑不代表自动高效。

HIP 提供显式线程、共享内存和原子操作控制，适用于复杂邻居表、迁移和通信打包等路径。Triton 适合规则张量融合、归约以及部分邻居消费算子，但目标海光 Triton 编译链必须实测。算法、数据布局和输入分布决定后端选择，语言本身不能保证性能上限。

替换 Warp 的主要成本包括反向实现、零拷贝互操作、编译缓存、stream 管理、测试和调优。这些成本需要计入计划，不能只统计 kernel 行数。

### 3.2 对旧路线文档的明确修订

1. 删除固定 `hip > triton > torch_reference` 的调度优先假设，改为能力过滤、场景基准和可复现策略。
2. “继承框架”具体落实为完整代码/测试基线、最小补丁和兼容契约。分布式模块也优先保持上游控制流程，仅在有证据时重构，不一开始整体重写。
3. 小型单卡 MVP 与首批训练版本可以分阶段交付。DDP 是明确产品目标，但不要因旧文档中“首发同时提供 DDP”的措辞，阻止先验收可用的单卡闭环。
4. 两卡 LJ 域分解要早做架构探针，产品化可后续完成，不必等所有单卡特性移植后再发现分布式问题。
5. ROCm Warp 的版本和显卡支持随时间变化。历史调研中的具体版本不可作为当前事实；即便原型可用，也不自动获得生产后端地位。
6. Python 版本首先服从海光 Torch 和模型依赖支持矩阵。旧文档的 Python 3.11+ 以及工期、人力配置均非已确认约束。

## 4. 源码组织与上游跟踪

### 4.1 推荐流程

根目录是移植主仓库。`external/` 的两份完整 clone 用于源码查阅、版本选择与同步来源，受主仓库 `.gitignore` 排除。确认 SHA 与兼容组合后，采用 Git subtree 不 squash 导入 `packages/framework` 和 `packages/ops`。

这样产品代码由一个仓库管理，两个上游的历史仍可追踪。不能将外部 clone 直接 `git add` 成未配置 submodule 的 gitlink，也不要删除上游 `.git` 后随手复制。若现场已有团队双 fork/submodule 方案，先记录现状及差异，再做非破坏性整合，不强行重建。

包内部暂保留原来的源码、测试、示例、许可证和构建布局，不立刻重命名全部目录。编译代码优先放在 `packages/ops` 的现有或约定子目录，避免根目录与包内出现两份 C++/HIP 真源。原 CI 作为可追踪来源保留；根 CI 逐步调用两包检查，不能假定嵌入子目录的 GitHub workflow 会自动生效。

分别维护：

- `docs/UPSTREAM_LOCK.yaml`：两个 URL、完整 SHA、tag（可选）、依赖兼容依据、导入位置、历史导入方式、日期。
- `docs/UPSTREAM.md`：导入与同步记录、主要补丁、已吸收的 bugfix、版本耦合、许可证变化。
- `adr/`：后端、数据布局、梯度、分布式、兼容范围等重要决策的依据和后果。

### 4.2 同步流程

定期 fetch 并评估差异，不对产品主线自动 pull 最新 main。对候选组合创建同步分支，分别检查公共 API、依赖、数值修复和测试变化，再整合上游。更新后执行兼容回归，更新 lock 与记录。

导入时不用 squash，后续也保持同一历史策略。首个同步周期应做一次可回滚的演练，证明并非只“留下远程地址”而实际无法合并。任何 API 不兼容、字段变更或数值约定变化都应先反映到兼容契约。

## 5. Feature Compatibility Contract

### 5.1 分类与验证是两条轴

| 类别 | 处理原则 | 示例 |
|---|---|---|
| A | 保留上层语义与使用方式，最小化适配 | Batch、模型接口、Hooks、FusedStage、checkpoint 配置 |
| B | 保留 API 与语义，替换内部硬件实现 | neighbors、积分更新、segment、可微绑定、通信 |
| C | 保留来源记录和接口意图，标注延期/实验边界 | JAX、部分高级长程、某些模型/编译特性 |

同一特性可拆子项：FusedStage 状态迁移属于 A，底层按 mask 更新和数据搬运可能属于 B。不可将完整高级特性一律归入 C 来规避工作量。

建议第一版 YAML 使用以下结构。示例是 schema 示意，路径和 SHA 必须由源码审计填入，不得直接当成完成项：

```yaml
schema_version: 1
upstream_lock: docs/UPSTREAM_LOCK.yaml
features:
  - id: dynamics.fused_stage
    category: A
    status: planned
    scenario: multi_stage_batched_simulation
    source:
      repository: framework
      commit: null
      symbols: [FusedStage]
      paths: []
    contract:
      api: preserve_locked_upstream
      state: per_system_stage_and_convergence
      system_identity: preserved
      fallback: explicit_only
    dependencies: []
    backends: [torch_reference]
    gradient_level: not_required_for_default_md
    verification:
      upstream_tests: []
      dcu_tests: []
      environments: []
      evidence: []
    limitations: [not_yet_validated_on_target_dcu]
    release_gate: G2
```

先保留上游用户脚本、导入路径、参数、默认值、返回字段和语义。环境选择或 backend 配置可增加，但应提供清晰迁移说明。生产支持由具体环境与用例证据决定，不靠 API 名字存在与否。

### 5.2 必须保护的高级行为

- Batch：异构大小、空输入约定、原子/体系字段、索引偏移、batch/unbatch、设备搬运与序列化。
- FusedStage：不同体系以不同速度收敛、逐体系阶段迁移、阶段预算、共享/独立模型的上游语义、Hook 时序。
- inflight：退出结果只记录一次、补位后状态正确初始化、ID 可追踪、容量约束和采样耗尽。
- 模型组合：总能量与力/应力一致、单位与符号正确，声明其 gradient 和 distribution 能力。
- checkpoint：模型/优化器/模拟状态、随机数状态、体系 ID、阶段和采样器状态按契约恢复。轨迹写出本身不等于完整重启支持。
- 多卡：DistributedPipeline、DDP、DomainParallel 分开测试。DDP 成功不能作为大体系域分解通过的证据。

## 6. 模块与算子迁移顺序

| 层/算子族 | 初始策略 | 后续优化与扩展 |
|---|---|---|
| 数据、模型、Hooks、流程 | 完整导入，检查 import 时 NVIDIA 依赖，保留上层实现 | 按实际瓶颈优化预取和 batch 更新 |
| neighbor naive / PBC / matrix-COO | Torch FP64 参考，小体系精确集合与位移验证 | Triton/HIP tile，按场景选择 |
| cell list / skin / rebuild | 保留语义，HIP 加 DTK 扫描排序库 | 高密度、长尾、三斜晶胞、容量管理 |
| segment sum 等 | Torch 可微参考 | Triton/HIP，以及实际需要的一阶/二阶反向 |
| LJ 与 switching | 解析或 Torch reference，力和 virial 一致 | 用作所有后端与多卡最小物理基准 |
| NVE / Langevin / FIRE | 保留算法和参数，Torch 起步 | 规则更新用 Triton，必要时 HIP |
| 首个真实 MLIP | 首先探测 MACE，依据可用依赖选择 | AIMNet2/UMA 等独立环境逐个适配 |
| 训练 | 保留循环、损失、数据与 checkpoint，先 eager | DDP、AMP、EMA、compile 按能力分层开启 |
| D3 / DSF / Ewald / PME | 保留算法参数和模型 API，逐家族验收 | FFT 复用库，spread/gather 与梯度专项验证 |
| NPT / 晶胞优化 / FIRE2 | 单卡语义和 stress 正确后实现 | 多卡下温压、优化器统计量全局归约 |
| domain decomposition | 早期两卡 LJ 探针，保留原 spec/adapter 设计 | 迁移、MLIP 消息传递、长程与通信重叠 |
| 球谐 / GTO / spline | 按模型依赖引入，保留数学参考与测试 | 再做硬件优化 |
| JAX / multipole / 特殊编译路径 | 记录 C 类、来源和退出条件 | 资源和用户需求允许后交付 |

## 7. 探针工程

探针独立放 `probes/`，复用结果报告格式，避免污染正式 API。建议按依赖顺序实施，时间以实际资源为准；2–4 周只是预研窗口建议。

| ID | 探针 | 至少输出的证据 |
|---|---|---|
| P00 | 环境与硬件只读盘点 | 设备/软件构建信息、可用资源、未知项 |
| P01 | 海光 Torch 的张量、同步、autograd、FFT | 小数据正确性、设备真实使用、退出码 |
| P02 | HIP 编译、运行、gather/scatter/atomic | 编译命令、目标架构、结果和耗时 |
| P03 | Triton vector add、归约与索引 | 可编译性、结果、启动与稳态耗时；失败可独立记录 |
| P04 | 自定义算子与导数 | opcheck、gradcheck、gradgradcheck 或版本限制 |
| P05 | neighbors / LJ / NVE 小闭环 | PBC 和 pair 计数、能量/力、短轨迹稳定性 |
| P06 | 首个真实 MLIP | 依赖树、前向、力导出、力损失对参数的梯度 |
| P07 | 双卡通信及两卡 LJ 原型 | collective/P2P、跨边界 pair、ownership 和单卡对照 |
| PX | 可选 Warp 预研 | 维护成本、目标支持和可微能力；不作为其它探针依赖 |

报告包含源代码 SHA、命令、设备、dtype、输入、随机种子、同步/预热方法、结果、错误与资源限制。原始大文件放 artifacts；脱敏摘要放 reports。不能扫描打印全部环境变量或访问令牌。

没有空闲第二卡时，P07 标为资源阻塞，同时编写可重跑测试和继续单卡工作。一个 Triton 探针失败不代表项目不可移植，可先推进 Torch 与 HIP。未具备 NVIDIA 基线时，用解析和 CPU/Torch 参考验证并注明基线来源缺失。

## 8. 分阶段交付与验收

| 阶段 | 交付 | 验收边界 |
|---|---|---|
| G0 | 环境、兼容组合、功能/测试清单、探针 | 真实证据与可重跑步骤；明确后端和依赖阻塞 |
| G1 | 两包可安装，Torch reference 的 Batch-neighbor-LJ-NVE 闭环 | 解析/FP64 对照、PBC、短 MD 和最小轨迹输出 |
| G2 | 单卡批量 MD/弛豫、真实 MLIP、FusedStage/inflight | NVE/NVT/FIRE、异构批次、ID、收敛、记录/重启 |
| G3 | MLIP 训练/微调与 DDP | 能量/力目标、二阶相关路径、验证集、恢复与多卡梯度 |
| G4 | 长程相互作用、高级系综与晶胞路径 | 分组件数值、energy-force-stress 一致、按需求梯度 |
| G5 | 单个大体系多卡 | 两卡到更多卡，原子与 pair 守恒、MLIP 适配、扩展效率 |
| G6 | 生产稳定性与生态 | 干净安装、长运行、资源泄漏、文档、兼容矩阵、上游同步 |

G5 架构探针从 G0 穿插推进。第一目标是 G1 的完整纵向链路，之后 G2 面向科学用户试用。实验特性可保留在源码中，但不能作为通过验收的生产承诺。

## 9. 软件工程与物理验证规范

- 保留上游测试；增量补充 DCU、特性兼容、数值和跨包集成测试。构建检查和 package import 不能替代物理验证。
- Python/HIP 风格尽量跟随上游，逐步引入 lint、类型检查、clang-format。不要首轮全库格式化或统一重命名，避免冲淡移植 diff。
- 后端/schema 的每次修改明确输入输出、单位、梯度、aliasing、错误、stream 与设备语义。对用户可见变更提供示例和迁移说明。
- 邻居测试覆盖 half/full、self、PBC shift、三斜晶胞、空批次、邻居长尾和容量溢出。先比较集合，排序仅在契约承诺时要求一致。
- 能量导出力采用负梯度约定；stress/virial 的正负、归一化与单位以原契约和适配目标明确记录。
- MD 验证能量漂移、温控统计和重启；长轨迹的混沌敏感性使逐步 bitwise 等价通常不适合作为跨平台标准。随机过程采用统计和固定算法适用范围内的复现测试。
- `gradcheck` 不替代高阶检查，`opcheck` 不证明梯度数学正确。力损失路径要实际反向到模型参数，防止包装器中间断图。
- 分布式检查 ghost 梯度、归约、不丢不重、rank-count 一致。MPNN 的多层消息依赖不能只用一次 cutoff halo 想当然处理。
- 性能回归分 cold/JIT 与 steady-state、单 kernel 与端到端、同形与异构 batch。阈值根据测量噪声制定，不先写死未经测定的百分比目标。
- CI 逐层增加：CPU/静态检查、单 DCU smoke、nightly 数值/梯度、周期多卡和长轨迹。没有 GPU runner 时保留可在服务器执行的同等命令并标明覆盖缺口。
- 每轮以 STATUS 交接，重要决策进 ADR。源码、模型和数据来源、依赖与测试状态可追踪。文档说“支持”的范围必须与实际验收一致。

## 10. 参考资料

- 上游框架：https://github.com/NVIDIA/nvalchemi-toolkit
- 上游算子：https://github.com/NVIDIA/nvalchemi-toolkit-ops
- FusedStage：https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/01_multistage_pipeline.html
- inflight batching：https://nvidia.github.io/nvalchemi-toolkit/examples/intermediate/04_inflight_batching.html
- 分布式：https://nvidia.github.io/nvalchemi-toolkit/modules/distributed.html
- 分布式模型设计：https://nvidia.github.io/nvalchemi-toolkit/userguide/distributed_design.html
- Warp 自动微分：https://nvidia.github.io/warp/latest/user_guide/differentiability.html
- Triton 编程模型示例：https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- PyTorch HIP：https://docs.pytorch.org/docs/main/notes/hip.html
- 自定义算子：https://docs.pytorch.org/docs/main/library.html
- 海光文档：https://developer.sourcefind.cn/document

在线文档仅辅助理解。特定版本的行为由 UPSTREAM_LOCK 中的代码、对应版本文档与服务器实测共同确定。

## 11. 历史暂停交接（2026-09-06）

### 已完成

- 当时根仓库分支为 `codex/g0-initialization`；上游锁定不变：framework
  `4dfe3723def34df3fadb245981081ccf8c94c257`、ops
  `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。未修改 `external/`；周期邻居 Tier 1
  当前状态检查点已提交为 `9fb9d1f`，后续文档改动另行提交，工作树状态以
  `git status --short` 为准。
- 完成 periodic full-list Torch reference 的 Tier 1 装配替换：
  `packages/ops/nvalchemiops/torch_reference.py` 保留原有逐 system pair/image 几何
  候选计算，改用设备端 `nonzero`、`bincount`、行内 rank 和矩阵索引写回，去除逐边
  Python `tolist()`/`append`/写回。shift 约定、自相互作用、重合错误、padding、
  overflow、MATRIX/COO 顺序未改变。
- 项目 `.venv` CPU ops reference `10 passed`、framework compute_neighbors `3 passed`；
  通过 `source scripts/activate_hygon_env.sh project` 加载 DTK 26.04 后，BW200/UBB
  BW1000（gfx936）HCU 同样为 ops `10 passed`、framework `3 passed`。
- 使用真实 `/data/csp_data/perf_46`、`perf_92`、`perf_184`、`perf_368` CIF 完成
  Tier 1 后 CPU/HCU 规模与 batch 回归：46/92 原子 batch=1/4/8/16/32，184 原子
  batch=4/8/16，及异构 `[46,92,184,368]`。所有完整日志中的 case 均为 passed，边数和
  `batch_ptr` 与前基线一致。

### 统一计时结论

正式前后比较统一使用项目环境、`OMP_NUM_THREADS=1`、一次 warmup 和两次 steady。92 原子
batch=32、54,760 条边：CPU `9.2946→7.7101 s`，HCU `4.2626→0.0658 s`。HCU 仍在共享
任务下，绝对时间只作相对证据。此前一次未设置 `OMP_NUM_THREADS=1` 的 CPU `1.0508 s`
试跑不计入结果。

CPU 大体系的 steady 仍接近原值，说明 dense pair/image 几何计算是主要成本；HCU 上装配
的逐边 Python 开销被显著削减。该优化没有改变算法复杂度，也不能外推为完整生产邻居
性能。

### 证据与文档

- 优化报告：[reports/g2-neighbor-tier1-scatter-reference.md](../reports/g2-neighbor-tier1-scatter-reference.md)
- Tier 1 前基线：[reports/g2-neighbor-baseline-reference.md](../reports/g2-neighbor-baseline-reference.md)
- 可重跑探针：[probes/neighbor_baseline.py](../probes/neighbor_baseline.py)
- 原始输出：`artifacts/g2/neighbor-tier1-*.log`（忽略文件，不入库）
- 本轮已同步：`docs/STATUS.md`、`docs/PROBE_PLAN.md`、`docs/FEATURE_COMPATIBILITY.yaml`、
  `docs/UPSTREAM.md`（补丁 LP-013）、`adr/0004-backend-defaults-and-width-gates.md`。

### 下次从这里开始

1. no-PBC `neighbor_list` 的设备端装配已完成 CPU 与 BW200 HCU 0 的 full/half、批边界、
   距离/向量和 overlap/overflow 合同；统一 no-PBC/periodic harness 已完成 CPU smoke，
   下一步在主机权限低干扰窗口补 HCU 固定结构阶梯和 periodic `[46,92]` mixed
   MACE/FIRE2 100 步端到端测量，不把 shared-HCU 数字当作发布性能门槛。
2. registry 已集中 operation/dtype/device/gradient/features 选择；任何继续的 framework
   接线或优化后端必须先登记 capability，不能重新增加局部字符串分支。
3. 统一 baseline 应同时覆盖 no-PBC full/half 和 periodic full；以 periodic MACE/FIRE2
   端到端成本作为 cell-list 的主要决策输入。基线完成后先评估 Torch reference
   cell-list，再排期周期 half-list、PBC 容量压力、长 skin/rebuild、Triton/HIP。
4. 每次修改导入的上游文件，继续在 `docs/UPSTREAM.md` 登记文件、位置、动机、upstream
   candidate 和回归指针；每轮独立更新 STATUS 当前快照、时间线和 DoD。

暂停时没有遗留运行中的 pytest 或 benchmark 进程。

## 12. 统一 benchmark 进展（历史记录，2026-09-08）

- 已补齐 unified reference benchmark 的输入元数据：每个 case 记录 CIF 路径、原始
  `source_pbc`、实际 `effective_pbc`、原子数和晶胞体积，避免将同一 CIF 的 periodic
  与 no-PBC 算法边界测量混淆。详见 `reports/g2-unified-reference-benchmark-cpu.md`。
- CPU neighbor/E2E smoke 已通过；HCU 小范围 smoke 也已通过：periodic/no-PBC full/half
  的 46/92 原子及 batch=1 契约检查退出码 `0`，设备为 `BW200, UBB BW1000`。详见
  `reports/g2-unified-reference-benchmark-hcu-smoke.md`。
- HCU 单体系规模、`perf_92` batch 阶梯和 periodic `[46,92]` MACE/FIRE2 固定晶胞
  100 步均已完成；stage timing 抖动仍需低干扰窗口复测，当前不把 shared-HCU 数字
  当作发布性能门槛。
- HCU 单体系规模阶梯现已完成：periodic full、no-PBC full/half 的 46/92/184/368
  全部通过；periodic steady 为 `3.243/3.702/5.753/13.969 ms`。
- `perf_92` 的 HCU batch=1/4/8/16/32 阶梯现已完成；periodic steady 为
  `3.768/9.789/17.819/33.571/65.577 ms`，batch=32 与 Tier-1 `54,760` 边基线连续。
  periodic `[46,92]` MACE/FIRE2 固定晶胞 100 步端到端也已通过；总耗时
  `11.1279105 s`，100 步平均 `0.1112791 s/step`。由于 stage timing 存在明显抖动，
  仍需低干扰窗口复测后再决定 cell-list 的切入位置。
- no-PBC 的历史过渡名称 `torch_reference_cell_list` 已完成 registry/dispatcher/framework
  接线和 CPU/HCU synthetic contract 回归；后续 M1 已将公开选择迁移为
  `backend="torch_reference", method="cell_list"`，旧全局请求明确失败。该历史记录中的
  “下一小步”已由后续 unified benchmark 和第 14--17 节覆盖。

## 13. 后续重构权威计划（2026-09-08）

后续不要把 `backend` 字符串继续扩展成全局枚举。已确认的“能力 registry → 平台
profile → 全 pipeline planner → 冻结 BackendPlan”重构计划保存在
[`docs/BACKEND_PLATFORM_PIPELINE_PLAN.md`](BACKEND_PLATFORM_PIPELINE_PLAN.md)。

该计划的 M1 已于 2026-09-09 完成。M1 保持 `backend=None` legacy 语义、将 cell-list
迁移为 neighbor operation strategy，并使 framework→dispatcher 传递同一 selection；未实现
profile、planner 或冻结计划。下一阶段先是 B0 团队基础开发版本，再按独立 operation 开始 T1；
M2 继续延期至出现多实现策略或冻结 plan 的实际需求。

## 14. M1 Registry 语义收敛（历史记录，2026-09-09）

- 已从干净前置分支 `codex/feat-reference-cell-list` 创建
  `codex/refactor-backend-plan-m1`。前置 cell-list 与计划文档分别封存为
  `a33932b`、`afad629`；未推送、未改写历史。
- `ImplementationRegistry` 取代固定 backend literal。选择记录包含 request、stable
  implementation ID、family、strategy 与 profile ID 占位；legacy Warp、dense Torch
  reference 与 no-PBC cell-list 均已登记，executor 仍为 lazy metadata。
- cell-list 公开用法为 `backend="torch_reference", method="cell_list"`；旧的未发布
  `torch_reference_cell_list` 请求明确失败。`None`/`warp` 仍是上游 legacy，`auto`
  仍只选 dense default strategy，未在 M2 前选择任何 strategy。
- CPU：ops `25 passed`、framework neighbor/Hook `22 passed`。BW200/gfx936 HCU：ops
  `25 passed`，新增 one-shot/Hook cell-list strategy `2 passed`；完整命令、退出码和
  载体限制见 `reports/g2-backend-registry-m1.md`。这不是 cell-list 性能、周期
  cell-list、BackendProfile 或 PipelinePlanner 的通过证据。
- 当时的下一步判断是等待用户确认；后续已由第 16、17 节修订为先完成 B0 审核和 HCU gate，
  再进入 T1，M2 继续延期。不得在 runtime benchmark 或改变 `backend=None` 的前提下推进。

## 15. M1 operation selection propagation（历史记录，2026-09-09）

- FIRE 与 FIRE2 已拆为独立 operation 和 implementation ID：
  `torch_reference.fire-v1` 与 `torch_reference.fire2-v1`。
- `LennardJonesModelWrapper`、NVE、FIRE/FIRE2、`WrapPeriodicHook`、Logging、
  energy-drift monitor、reporting scalar 和相关分布式 FIRE wrapper 已改为由 framework
  按 operation 解析一次 `BackendSelection`，再传给 dispatcher/辅助函数。dispatcher
  当时按精确 implementation ID 执行，不用原始 backend request 二次解析；B1 已将运行时
  绑定升级为 catalog entrypoint + generic adapter，历史条目不代表当前实现方式。
- 固定晶胞的 selection 在相应 workflow/model 生命周期中缓存；observer 中不同 operation
  各自只解析一次。变胞 FIRE/FIRE2 在初始化时声明 `variable_cell` 要求；当前 Torch
  reference 没有该 capability，显式请求会明确失败，默认仍保持 legacy Warp。
- 新增回归 `packages/framework/test/compatibility/test_backend_selection_propagation.py`。
  CPU 结果：ops `23 passed`；selection propagation `3 passed`；reference/observer/periodic/LJ
  slice `103 passed, 1 deselected`（与 propagation 合并为 `106 passed, 1 deselected`）；
  state lifecycle `22 passed`；import/reference `15 passed`。
  详细命令和限制见 `reports/g2-backend-registry-m1-dispatch-propagation.md`。
- 本轮只新增 CPU selection/dispatch 接线证据，没有新增 HCU、Triton/HIP、变胞或 M2
  planner 证据。后续已由第 16、17 节修订为先完成 B0 审核和 HCU gate，再进入 T1。

## 16. B0 团队基础开发版本（2026-09-09）

- 候选分支为 `team/dev-baseline-v0.1`，由 M1 follow-up `fb11528` 建立。CPU/HCU 门槛通过后，
  已按用户授权 fast-forward 合入 `develop`，没有改写共享历史。
- 默认 registry inventory 已按 operation family 迁入私有 metadata-only catalog，仍由
  `nvalchemiops.backend` 提供相同 public API、稳定 ID、注册顺序与 lazy executor 语义。
- 基线将 neighbors→LJ、fixed-cell VV/FIRE/FIRE2/kinetics、periodic/observer 定为三个
  reference golden paths，提供 `scripts/check_cpu_reference.sh` 作为每次合入 gate，并以
  `HIP_VISIBLE_DEVICES=<assigned> scripts/check_hcu_reference_smoke.sh` 作为批 HCU 证据入口。
- 细化范围、角色边界、DoD 与 T1 队列见
  [TEAM_DEVELOPMENT_BASELINE.md](TEAM_DEVELOPMENT_BASELINE.md)；新 operation 的最小接线步骤见
  [ADD_TORCH_OPERATION.md](ADD_TORCH_OPERATION.md)。B0 不扩大 feature contract；CPU/HCU gate
  均已通过并已合入 `develop`。

## 17. 当前恢复入口（2026-09-09）

- 当前共享开发基线是 `develop`；`team/dev-baseline-v0.1` 保留为本轮集成指针。开始工作前仍须运行
  `git status --short --branch` 并确认工作树状态。
- M1 registry、operation selection propagation、B0 catalog 拆分、CPU gate、架构图、开发者指南和
  Torch operation 教学文档均已落盘。B0 没有扩大 feature contract；HCU 0 上的 B0 smoke gate 已退出
  `0`，五个 reference probe 均通过。
- 当前最短恢复路径是：阅读 `docs/STATUS.md` 和 `docs/TEAM_DEVELOPMENT_BASELINE.md`，运行
  `scripts/check_cpu_reference.sh`，再由分配到 HCU 的开发者运行
  `HIP_VISIBLE_DEVICES=<assigned> scripts/check_hcu_reference_smoke.sh`。当前两道 gate 均已通过，
  B0 已合入 `develop`；当前 B1 候选 gate 也已通过，但仍须按第 18 节完成人工 review 后再
  进入 T1 的常用积分器移植，M2 仍不启动。

## 18. B1 Executor Binding 交接（2026-09-09）

- 用户已确认按 B1 计划实施；B1 已从 `codex/refactor-executor-binding` 快进合入本地
  `develop`，个人分支仍保留。B1 目的为兑现 catalog 的 `executor` 字段，消除实现声明与 22 处局部 dispatcher
  分支之间的人工绑定地雷，不改变公共 API、`backend=None` legacy 语义或现有 capability 宽度。
- 已完成提交顺序：`90c43fb`（schema/loader/catalog/segmented-reduce）、`98da458`（ops
  generic dispatcher）、`0b1698c`（VV/periodic/kinetics/segmented）、`0f8b9dd`（FIRE/FIRE2）、
  `f5eb064`（高层 neighbors/LJ/Hook 与静态 guard）、`efedb58`（owner diagnostics）、
  `52663f2`（cache isolation follow-up）。
- 实现约束：非 legacy metadata 声明 module、entrypoint 元组和 owner；loader 独立于 registry
  并 lazy import；adapter 只做一次 legacy 比较和声明式调用；legacy 逻辑由各 dispatcher 的
  局部闭包保留；entrypoint ABI 必须与 operation dispatcher 签名兼容。第二实现测试实际调用
  未修改的 dispatcher，不能只测 import/callable。
- CPU gate 现已把 `test_executor_binding.py` 纳入 framework 分进程测试；完整命令、退出码和
  分层结果见 `reports/b1-executor-binding.md`。代码与文档均已通过 `git diff --check`，
  `FEATURE_COMPATIBILITY.yaml` 不变，因为 B1 只改变绑定机制。
- HCU 已按 B0 式候选集成模式完成：本地候选指针为 `team/b1-executor-binding-candidate`，在主机
  权限、DTK 26.04、显式 `HIP_VISIBLE_DEVICES=0` 的 HCU 0 上运行
  `scripts/check_hcu_reference_smoke.sh`，五个 probe 退出码均为 `0`，且均报告
  `warp_imported=false`。该结果只覆盖当前 golden-path 窄 slice，不是完整 DCU production 或
  性能结论；本次用户授权的同步目标仅为 `local-origin/develop` 和 `github-origin/develop`。
- 本次同步完成后暂停开发，不启动 `TORCH-NVT-LANGEVIN`；周期 cell-list 和 NHC 不与其合并，
  M2 继续延期。
- cache 约束：生产 dispatch 使用稳定 callable cache，不自动检测模块变化；测试或受支持的
  module reload 工具必须显式调用 `clear_entrypoint_cache()`。不要在每个 dynamics step 增加
  `sys.modules`/module identity 检查。

## 19. 并行开发启动计划（2026-09-09）

- 团队执行入口为 `docs/PARALLEL_DEVELOPMENT_PLAN.md`。所有开发者从远端最新 `develop`
  创建 `<开发者>/<类型>-<主题>` 分支；每个分支只拥有一个 operation 的 executor、测试、probe
  和 report，共享 catalog/CPU gate/STATUS/兼容矩阵由集成负责人串行收口。
- 三个核心 T1 为 `TORCH-NVT-LANGEVIN`、`TORCH-NEIGHBOR-PBC-CELL` 和 `TORCH-NVT-NHC`；
  `TORCH-THERMOSTAT-UTILS` 与 `TORCH-LJ-SWITCHING` 是额外人力下的低耦合扩展。
- 用户追加批准 `TORCH-FIRE2-VARIABLE-CELL` 作为 P1 扩展。一个 owner 先交付
  `TORCH-CELL-STRESS-FORCE`（stress 符号、volume、cell inverse、`keep_aligned` 与 FP64 oracle），
  再交付原子/晶胞 coupled FIRE2 step 和公共 `FIRE2VariableCell` 窄纵向验证。
- 该扩展不包含普通 `FIREVariableCell`、NPT/NPH barostat、DomainParallel replicated cell
  state、完整 LJ virial/stress、生产 Triton/HIP 或 M2。正确性验证可使用现有 periodic dense
  reference，不等待周期 cell-list 性能任务。
- `FEATURE_COMPATIBILITY.yaml` 的 `dynamics.fire` 仍保持 `planned/partial`：本轮只是批准排期，
  没有新增实现、CPU 数值或 HCU 证据，不得提前改成 implemented/verified。

## 20. ASE-compatible BFGS 并行任务登记（2026-09-10）

- 用户要求框架提供与此前分子晶体弛豫工作对齐的无 line-search BFGS。项目 `.venv` 的
  `ase==3.29.0` 已核实：公开 `ase.optimize.BFGS` 使用
  `ase._4.optimize.bfgs.BFGSMethod`，每步更新 Hessian、执行 `eigh(H)`、对特征值取绝对值
  后计算方向，并以全局最大原子步长缩放；因此 `eigh` 是该兼容目标的一部分，不可用另一种
  标准 BFGS 实现替代后仍称数值对齐。
- `TORCH-BFGS-ASE-COMPAT` 已登记为 P1：先以单体系固定晶胞 Torch/CPU FP64 oracle 实现，
  对照 Hessian、方向、`maxstep` 与 restart；异构 Batch、inflight、DomainParallel 明确不在
  首版范围。HCU 的 `torch.linalg.eigh` 先作为正确性和基线；只有目标尺寸实测确定其为稳定瓶颈
  时，才评估 `HIP-BFGS-EIGH`，不得预设 hipSOLVER/Triton/HIP 的性能结论。
- 严格变胞 ASE 对齐单列为后续 `TORCH-BFGS-ASE-UNITCELL`：必须采用 `UnitCellFilter` 的
  `3N+9` deformation-gradient、`cell_factor`、mask 和压力约定；现有 `3N+6` 上三角
  cell-filter/FIRE2 路径是不同的框架原生语义，不能标注为 ASE-compatible。此前实际使用的
  ASE filter 尚待从原工作记录确认后，才可声明完整变胞轨迹对齐。

## 21. T1 Torch NVTLangevin reference 收口（2026-09-19）

- `codex/feature-torch-nvt-langevin` 已完成固定晶胞 BAOAB Langevin 的 Torch reference、
  registry/catalog、generic executor binding、公共 `NVTLangevin(backend="torch_reference")`
  接线及 focused compatibility tests；`backend=None` 的 legacy Warp 语义保持不变。
- 当前 contract 支持普通多体系 `Batch`、`batch_idx`、异构 per-system `dt/kT/friction`、
  float32/float64、空输入和显式错误。CPU contract `12 passed`，项目 CPU gate 及已分配
  HCU 0 smoke 均通过；完整命令和范围见 `reports/g2-torch-nvt-langevin.md`。
- 本收口不扩大为完整 Langevin/NVT 支持：上游完整行为/统计套件、checkpoint/restart、
  `atom_ptr`、`_out`、inflight refill、分布式 ownership、torch.compile 和 Triton/HIP
  生产实现仍是独立后续任务。受控统计测试的短谐势 CPU/HCU 复核已完成；后续 restart
  采用独立的最小 integrator continuation slice，不扩大为通用 checkpoint。

## 22. T1 Torch NVTLangevin controlled statistics（2026-09-19）

- 新增 `test_dynamics_reference_langevin_statistics.py`：使用独立解析谐势
  `F=-k*x`，两个 Batch system、300/600 K 和不同 friction，检查 canonical 分布的
  `mean(x^2)=kT/k` 与 `mean(v^2)=kT/m`。
- 原有 Langevin contract 与统计测试合计 CPU `13 passed`，短统计 oracle 在 HCU 0 上
  `1 passed`；报告见
  `reports/g2-torch-nvt-langevin-stat.md`。
- 本步只增加短统计 oracle，不实现新的公共 API、MACE 统计、长轨迹框架或 restart。
  统计门只覆盖独立谐势的二阶矩，不等价于完整上游 Langevin/NVT 统计套件。

## 23. T1 Torch NVTLangevin minimal restart（2026-09-19）

- 新增 `test_dynamics_reference_langevin_restart.py` 和 `NVTLangevin.state_dict()` /
  `load_state_dict()`：保存 `step_count`、`random_seed` 及已初始化的 per-system
  `dt/temperature/friction`，支持普通固定晶胞 Batch 的新实例续跑。
- 第 3 步保存后恢复 2 步，与连续运行 5 步的 positions、velocities、forces、energy
  在 CPU 和 HCU 0 均一致；restart 测试各 `2 passed`，报告见
  `reports/g2-torch-nvt-langevin-restart.md`。
- 位置/速度/力、模型参数和 Batch 元数据仍由调用方负责保存；完整 checkpoint、inflight、
  `atom_ptr`、`_out`、分布式 ownership 和跨设备逐位一致仍未实现。

## 24. T1 Torch PBC cell-list 阶段一（2026-09-19）

- 分支 `codex/feature-torch-neighbor-pbc-cell-core`（已由 `88fe538` 合入本地 `develop`）将既有
  no-PBC Torch cell-list 扩展为
  periodic/no-PBC、full/half、mixed/triclinic Batch 的 correctness reference，并补齐分层
  build/query、预分配 scratch、容量错误、selective rebuild 与连续 vector/distance 梯度。
  `backend=None`/Warp、显式未知后端失败和 `auto` 不选 cell-list 的约束保持不变。
- CPU gate 全部通过；主机权限 DTK 26.04、BW200 HCU 0 上，独立 PBC probe 的 pair+shift
  dense parity、half、分层 counts 和二阶梯度通过，真实结构 benchmark、20 步
  MACE/FIRE2 Hook 对照及标准 HCU smoke 也完成。focused pytest 虽在 device-visible 进程运行，
  但其 tensor 为 CPU 构造，只计 CPU contract。完整证据见
  `reports/g2-torch-reference-pbc-cell-list-core.md`。
- 固定真实输入基线显示 Torch cell-list 比 dense 慢 `2.38--2.84x`；hipprof 记录大量细粒度
  indexing/sort/scan 与 launch 碎片。因此当前实现只作为阶段一 oracle，不能进入 `auto`，也
  不能写成 production neighbor backend。
- 阶段二先冻结共享 cell metadata/CSR/scratch/output/capacity/stream ABI，再允许 build、query、
  pair materialization 和 Batch/rebuild 模块并行评估 HIP/Triton。进入 `auto` 前必须保持阶段一
  数值契约，并在代表性 workload 达到 neighbor 2x 或目标端到端 20%。`target_indices`、
  `pair_fn`/pair outputs、pair-centric/compile 和 DomainParallel 继续按独立 capability 排期。
- 当前阶段二工作分支为 `codex/feature-torch-neighbor-pbc-cell-stage2`，从上述 `develop`
  基线创建，尚未加入正式 AOT/custom-op 实现。
- `probes/hip_cell_list_jit_probe.py` 已完成第一个隔离 native HIP build/binning 可行性检查：
  DTK hipcc 编译和 BW200 执行通过，FP32/FP64 mixed-PBC 与 Torch oracle 一致；368/32768
  原子子模块重复计时显示约 `11.51x/11.52x`，但范围仅为 wrap+cell key，不包含 sort/query/
  fill、dispatcher 或 wheel。该 probe 不进入常规 HCU smoke，避免每次门禁触发 JIT 编译。
