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

## 25. T1 阶段二 cell-key shared ABI（2026-09-19）

- 阶段二小点 1 在 `codex/feature-torch-neighbor-pbc-cell-stage2` 增加私有
  `nvalchemiops._cell_list_abi`：以 `positions`、`inverse_cell`、`int32` cells-per-dimension
  和 `bool pbc` 作为输入，以调用方提供的 `int32` periodic shift、cell coordinate 和 linear key
  buffer 作为唯一可变输出。它是后续 HIP/Triton build/binning 模块的共同替换边界；不登记
  custom-op/backend，不改 runtime 选择。
- `torch_reference_cell_list._build_into` 已复用该 reference ABI，原有 sort/CSR/query 仍在原
  模块。CPU 夹具明确覆盖 triclinic mixed-PBC 的逐元素结果、in-place 输出和错误输入；现有
  periodic full/half、capacity、Batch selective rebuild、二阶梯度回归均通过（focused
  `15 passed, 17 deselected`）。完整记录见 `reports/g2-cell-key-build-abi.md`。
- 本轮 HCU smoke 未取得算子证据：`hy-smi` 正常且空闲，但项目 `.venv` 的
  `torch.cuda.is_available()` 为 false、device count 为 0，专用 probe 在可用性门处退出。
  这不是 ABI 数值失败，也不改变已有 BW200 JIT 可行性证据；待设备重新暴露后重跑。
- 确认后只进入下一个小点：以本 ABI 建立 native HIP `cell_key_build_into` 的 custom-op/AOT
  装载边界和 output parity；不提前接入 CSR/query/dispatcher，不作性能门槛结论。

## 26. T1 阶段二 native HIP cell-key boundary（2026-09-19）

- 小点 2 在 `nvalchemiops._hip_cell_key` 增加私有 mutation-only
  `torch.library.custom_op`，三个预分配 `int32` 输出 buffer 通过 `mutates_args` 固定；
  `register_fake` 已补齐。`_native/cell_key_build.cpp/.cu` 将 shared ABI 映射到 native HIP
  kernel，使用当前 Torch stream，不修改 neighbor dispatcher 或 capability catalog。
- loader 只在显式 native HIP 调用时触发，并把 `.cu` 转换/编译副本放入指定 build cache；
  gfx936、DTK 26.04 下 C++/HIP 编译、链接和 Python extension load 已通过。wheel 已确认携带
  native 源码和 Python loader，不携带预编译 `.so`，所以正式 AOT wheel 构建尚未完成。
- CPU focused boundary/reference 回归为 `17 passed`。CPU 调用明确拒绝，不回退 Torch；已有
  cell-list reference 仍经 shared ABI 工作。完整记录见 `reports/g2-cell-key-hip-boundary.md`。
- 主机权限 HCU 0（BW200/gfx936）已实际运行专用 probe：FP32/FP64、257 原子三斜
  mixed-PBC 的 shift/coordinate/key 与 Torch reference 逐元素一致；空输入、非默认 stream、
  strided read-only inputs 也通过。设备节点可见性必须与运行命令同上下文检查，先前沙箱
  `cuda_available=False` 不构成 HCU 失败。
- 该 HCU 证据只覆盖 build 子模块 current-stream output ABI，未测性能，未覆盖 sort/CSR/query/
  fill、完整 dispatcher、梯度、compile/opcheck 或预编译 AOT wheel；当前不登记 `hip`，不改变
  `auto`。下一小点建议是正式 AOT extension 构建入口，之后再选择 query/count/fill 模块。

## 27. T1 阶段二 cell-key AOT wheel（2026-09-19）

- 小点 3 新增 `scripts/build_hip_cell_key_aot_wheel.py` 和 ops Makefile 入口。它保留 Hatchling
  基础 wheel，再以当前 Hygon Torch 编译 gfx936 artifact、注入 platform wheel 并重写 WHEEL/
  RECORD；不把 Torch 加为基础 build dependency，也不替换整个上游打包后端。
- `_hip_cell_key` 发现 package 内 `.so` 时优先加载；artifact 存在却无法加载会显式失败，
  而非回退 JIT。源码 checkout 没有 artifact 时维持小点 2 的 lazy JIT 行为。
- 实测 AOT wheel 为 `cp312-cp312-linux_x86_64`、`Root-Is-Purelib: false`，`unzip -t` 与 182
  条 RECORD 均通过。临时解包、只置入该 wheel 的 `PYTHONPATH` 后，BW200/gfx936 HCU probe
  实际从 wheel 内 `.so` 加载并复现 FP32/FP64、空输入、strided input、stream parity。
- 文档化的 `make -C packages/ops build-hip-cell-key-aot` 在同一主机环境也已退出码 0，产出
  相同 tag 的 wheel。
- 这只声明当前 CPython 3.12、Linux x86_64、DTK 26.04/HIP PyTorch 2.9.0/6.3.26093、gfx936
  的本机 artifact 证据；不是 manylinux、跨环境或发布就绪结论。也不登记 `hip`，不改变
  `auto`。完整证据见 `reports/g2-cell-key-aot-wheel.md`。
- 下一小点应先冻结 `cell_key -> cell_count/start` 的 CSR count ABI 和 Torch oracle，再评估
  native HIP atomic count kernel；不提前进入 query/fill 或完整 dispatcher。

## 28. T1 阶段二 cell-key CSR count/start ABI（2026-09-19）

- 小点 4 在 `_cell_list_abi` 增加 count/start phase：`int32` key 输入映射到 `(M,) int32`
  counts 与 exclusive starts；global offset 只作用于 starts。shape/dtype/device、key range、
  output alias、offset 与 int32 overflow 全部显式验证，空输入仍写出确定的 zero counts/offset
  starts。
- `torch_reference_cell_list._build_into` 现委派该 phase 写 active CSR slice，原有预分配
  capacity、tail zero、stable sort、global offsets 与 query 保持在原路径。CPU focused 为
  `20 passed`；BW200/gfx936 专用 CSR probe 和 PBC cell-list end-to-end probe 均退出码 0。
- 当前是 HCU Torch reference 证据，不是 native HIP count 或 scan；没有性能数据、AOT artifact
  变更、dispatcher/`auto`/`hip` capability 扩展。完整记录见 `reports/g2-cell-key-csr-abi.md`。
- 下一小点只增加 native HIP atomic count custom-op，复用已冻结 count output ABI；starts 保持
  Torch scan，直到独立 scan 模块通过数值和性能验收。

## 29. T1 阶段二 native HIP cell-key atomic count（2026-09-19）

- 小点 5 从 CSR helper 中分离 `build_cell_counts_reference_into` 作为 count-only Torch oracle，
  并在 `nvalchemiops._hip_cell_count` 增加私有 mutation-only custom-op/fake 注册。`int32` keys 和
  caller-owned `(M,) int32` count buffer 的 shape、dtype、同设备、范围和 int32 上界仍以 shared ABI
  为准；CPU 或非 HIP 调用明确失败，不作 fallback。
- 原生 `.cpp/.cu` 路径在 current stream 以 `hipMemsetAsync` 重置完整 count output，随后按 key
  `atomicAdd`。strided read-only key input 允许连续化；output 必须 contiguous。它没有接入
  `torch_reference_cell_list`，该 reference 仍走 Torch `bincount`，故 scan、sort/fill、query、
  dispatcher、`auto` 和 AOT artifact 均不变。
- CPU focused 为 `25 passed`。主机权限 BW200/gfx936 实际 JIT 编译、链接、加载并运行 4099 keys/
  137 cells 的普通、空、strided key 与非默认 stream probe，均与 Torch count oracle 逐元素一致；
  完整 PBC reference probe 也通过。没有 performance 结果，不能宣称比 HCU Torch `bincount` 更快。
  完整记录见 `reports/g2-cell-key-hip-count-boundary.md`。
- 下一小点应先用固定 workload、warm-up 和重复样本对 Torch count 与 native count 作 device timing，
  并覆盖均匀与 dense-cell atomic contention；依据结果再决定 count candidate 的后续接线或 scan 模块
  优先级。仍不得改变 `auto` 或提前进入 scan/sort/fill/query/dispatcher。

## 30. T1 阶段二 cell-key atomic count microbenchmark（2026-09-19）

- 小点 6 新增 `probes/cell_key_count_benchmark.py`，以 HIP events 比较显式 Torch/HIP count API。
  范围包含 ABI validation、output overwrite/zero 和 count，不含 JIT、key 生成、host 时间、scan、
  sort/fill/query 或完整 build。每个 workload 在计时前后逐元素比较，raw samples 作为不入库 artifacts
  保存，一文件一行一个毫秒值。
- BW200/gfx936、32768 keys/4096 cells、20 warm-up、100 repeats/sample、20 samples 的 uniform 与
  all-keys-in-cell-0 case 分别得到 Torch/HIP median `0.26104/0.20087 ms` 和
  `0.26266/0.20278 ms`，即 `1.300x/1.295x` 的 count-only factor。single-cell Torch 有一个高值，
  但 HIP median 仍较低；这些是共享 HCU 上的窄 microbenchmark，不能外推为完整邻居性能。
- 该局部结果支持保留 count candidate，不满足 neighbor 2x 或端到端 20% 的 `auto` 门槛，也不改
  AOT、dispatcher、默认路径或 `hip` capability。完整报告见
  `reports/g2-cell-key-count-benchmark.md`。
- 下一小点建议为独立 native HIP exclusive scan：保持 active counts，不修改 count output，写出带
  global atom offset 的 starts，并先过 Torch oracle 的空输入/offset/overflow/stream 契约；不得提前
  接线 count+scan 到完整 build 或 dispatcher。

## 31. T1 阶段二 HIP cell-list build pipeline ADR（2026-09-19）

- 用户确认阶段二应按完整数据流设计，而非持续孤立替换 Torch primitive。ADR 0009 对照锁定上游
  `count_atoms -> array_scan -> count zero -> bin_atoms`：其二次 per-atom pass 和 count cursor
  复用是算法参考，但 Warp/CUDA 实现不能直接当作 DCU 最优结论。
- 已决定 future HIP build 的阶段为 `grid metadata -> fused geometry/PBC/key-count -> global scan ->
  CSR fill -> query`。count 与 scan 不在一般规模上融合；geometry/PBC/key 与 count 可作为一个新
  candidate 融合；fill 仅在内部生命周期中可复用 count 作为 cursor，完成后必须恢复 public final
  counts。stable Torch sort 是 oracle，内部 atom order 放宽前需保持公共 neighbor 顺序/集合契约。
- DTK 26.04 只读发现 `rocprim::exclusive_scan` 与 hipCUB scan 头文件，`hipcc` 为 dcc 25.10/clang
  17；这还不是本项目 compile/HCU scan 证据。下一小点是 standalone rocPRIM scan boundary 的
  workspace、数值和 stream 验证，仍不接入 build/dispatcher。详见
  `adr/0009-hip-cell-list-build-pipeline.md`。

## 32. T1 阶段二 native HIP cell-start scan boundary（2026-09-19）

- 小点 8 在 `_cell_list_abi` 分离 `build_cell_starts_reference_into`：它以 flattened active-cell
  slice 执行 exclusive starts，单体系与应用 cell offsets 的 Batch slice 共用同一语义；counts 不得
  修改。CSR helper 保持先 count 后 scan，避免将未初始化 count output 当作输入校验。
- `_hip_cell_scan` 以 private mutation-only custom-op 包装 current-stream
  `rocprim::exclusive_scan`。rocPRIM workspace 经 two-call query 由调用方以 `uint8` buffer 显式
  提供；CPU/非 HIP 显式失败，无回退。CPU focused `29 passed`，包括 starts/offset、负 count 与
  overflow、既有 PBC/Batch/梯度回归。
- 主机 BW200/gfx936、PyTorch HIP 6.3.26093 实际 JIT compile/load/run probe 对 output parity、
  counts unchanged、empty、strided read-only counts 和 non-default stream 均通过；随后 FP64 PBC
  cell-list reference probe 也通过（full/half `8/4`、pair/shift parity、二阶梯度）。未测 scan
  性能，未接入 key/count/fill/query、AOT、dispatcher、`auto` 或 `hip` capability。完整记录见
  `reports/g2-cell-start-scan-hip-boundary.md`。
- 下一小点只定义/验证 batch-aware fused geometry/PBC/count candidate，对齐既有 key/count oracle
  后再决定是否基准；不进入 fill/query 或 runtime 接线。

## 33. T1 阶段二 native HIP Batch geometry/PBC/key/count fusion（2026-09-19）

- 小点 9 新增 batch-aware shared oracle 与 explicit HIP candidate。它以 flat atoms、per-system
  inverse cells/dimensions/PBC、任意顺序 `batch_idx` 和连续 `cell_offsets` 写 global keys/counts；
  B=1 是同一 ABI 特例，cell counts 必须精确覆盖 concatenated system cell slices。
- native candidate 以 `hipMemsetAsync` 覆写 complete count output，再在一个 current-stream
  per-atom kernel 同时计算 geometry/PBC、shift/coordinate/key 并 `atomicAdd` count。CPU/非 HIP
  显式失败，strided read-only inputs 可连续化，outputs 必须 contiguous；没有接入 reference 或
  runtime policy。
- CPU focused `33 passed`。主机 BW200/gfx936、HIP 6.3.26093 JIT compile/load/run probe 对 257
  atoms、两个异构 dimensions/mixed-PBC 的 FP32/FP64 outputs 均与 Torch oracle 逐元素一致；empty
  B=1、strided input 和 non-default stream 同样通过。未测性能，未覆盖 scan/fill/query、AOT、
  public neighbor output、compile/opcheck 或 selective rebuild，不改 `auto`/`hip` capability。完整
  记录见 `reports/g2-batch-cell-key-count-hip-boundary.md`。
- 下一小点只建立相同输入与输出合同下的融合-vs-独立 native key+count HIP-event benchmark；先获得
  重复样本再决定是否保留融合 candidate 或进入 CSR fill，不接线 dispatcher。

## 34. T1 阶段二 Batch fusion microbenchmark（2026-09-19）

- 小点 10 在 2x16384 ordered-Batch、4096 global cells、FP32、mixed-PBC 下以同一 global
  shifts/mapping/keys/counts 合同比较 fusion candidate 与 per-system native key+count+offset-add
  composition。范围是 pre-warmed HIP-event GPU work（包括 memset/kernels/add），不含 JIT、Python
  validation、scan/fill/query/dispatcher 或 complete neighbor。
- BW200/gfx936、20 warm-up、100 repeats/sample、20 samples 的 uniform/single-cell median 为
  composed/fused `1.17562/0.79358 ms`、`1.17380/0.79173 ms`，factor `1.481x/1.483x`，融合降低
  `32.50%/32.55%`；四侧 raw samples 的相对总体标准差为 `0.82%/1.57%/1.35%/1.22%`。输出计时前后
  逐元素一致，满足预先定义的两个 workload 均至少 5% 降低门槛。raw artifacts 不入库。
- 该 narrow result 支持保留融合 candidate，不能推广为 Torch comparison、不同 Batch/dtype、完整
  cell-list/neighbor 或 MACE/FIRE2 performance；不改 AOT、dispatcher、`auto` 或 `hip` capability。
  完整记录见 `reports/g2-batch-cell-key-count-fusion-benchmark.md`。
- 下一小点先冻结 CSR fill ABI/Torch oracle 的 cursor、capacity、global atom offset 和 public order
  合同；之后才做 native HIP fill correctness candidate，不接线 runtime。

## 35. T1 阶段二 CSR atom-list fill ABI/Torch oracle（2026-09-19）

- 小点 11 新增 `build_cell_atom_list_reference_into`：输入 active `cell_keys`、final counts 和
  exclusive starts，输出 local `cell_atom_list` 与 caller-owned `cell_cursor`。写入 atom IDs 加
  `global_atom_offset`，支持 Batch slice；capacity tail 保持不变。
- `cell_cursor` 完成后等于 final counts，public counts/starts 只读不变。这对应上游 `bin_atoms`
  的 atomic cursor 生命周期，但 Torch oracle 使用 stable key sort 保持现有公共 atom order；未来
  HIP atomic fill 如果顺序不同，需要 canonicalization 或明确新的内部/公共 order 契约。
- `torch_reference_cell_list._build_into` 已复用该 ABI。CPU focused `27 passed`；BW200/gfx936
  HCU PBC reference probe 通过，full/half `8/4`、pair/shift parity、layered counts `[1,1]`、二阶
  梯度有限。没有 native HIP fill、性能、AOT、dispatcher、`auto` 或 `hip` capability 变更。完整
  记录见 `reports/g2-cell-atom-list-fill-abi.md`。
- 下一小点在该 ABI 上实现 native HIP fill candidate，先验证 empty/single/Batch/triclinic/
  capacity/stream，不接入 runtime。

## 36. T1 阶段二 native HIP CSR atom-list fill candidate（2026-09-19）

- 小点 12 新增 `build_cell_atom_list_hip_into`，它在 current stream 上清零 caller-owned cursor，
  再以 HIP `atomicAdd` 填入 `cell_starts[key] - global_atom_offset + slot`。keys/counts/starts 是
  read-only 输入，cell atom-list 写 global atom ID，public CSR metadata 不被 cursor 覆写。
- atomic insertion 的同 cell atom order 未指定；因此 probe 显式按每 cell atom set 对照 stable
  Torch oracle，并验证 cursor final counts、counts/starts 不变与 capacity tail。它不能替换当前
  stable public path，除非后续验证 query/public pair order 已 canonicalize 或新增该步骤。
- CPU focused `37 passed`；BW200/gfx936 真实 JIT 编译/链接/加载和运行 257 atoms、异构 triclinic
  Batch/mixed-PBC、FP32/FP64、strided inputs、empty B=1、non-default stream 均通过。未测 fill
  性能，未串接 key/count/scan，未接入 dispatcher、AOT、`auto` 或 `hip` capability；完整记录见
  `reports/g2-cell-atom-list-fill-hip-boundary.md`。
- 下一小点在隔离 probe 组合 fused key/count、native scan 与 atomic fill，先固定完整 Batch CSR
  metadata 与 per-cell atom-set contract；仍不接入 runtime，之后再评估 query/order parity。

## 37. T1 阶段二 isolated native HIP Batch CSR build composition（2026-09-19）

- 小点 13 新增 explicit `build_batch_cell_csr_hip_into`，按 `fused key/count -> rocPRIM scan ->
  atomic fill` 组合既有 candidate，不新建 kernel 或 dispatcher backend。所有 output/cursor/workspace
  仍为 caller-owned；workspace 在 fusion 前由 scan query 分配时可将 count buffer 置零，fusion 将完整
  覆写它。
- `global_atom_offset` 同时进入 starts 与 fill，确保 active atom-list slice 的索引/ID 合同一致。probe
  对 Torch composition 验证 shifts/mapping/keys/counts/starts 逐元素一致，atomic list 逐 cell atom set
  一致，cursor final count 与 capacity tail 合同成立；atomic 同 cell order 仍未指定。
- CPU focused `39 passed`。主机 BW200/gfx936 同一 device-visible context 中 `/dev/kfd`、`/dev/dri`、
  DTK hipcc（dcc 25.10/clang 17）可见；HCU 0 实际 JIT/load/run fusion、scan、fill 三 extension，并
  通过 FP32/FP64、257 atoms、异构 triclinic Batch/mixed-PBC、offset 17、strided inputs、empty B=1 与
  non-default stream probe。没有性能、query/public pair、AOT、dispatcher、`auto` 或 `hip` capability
  结论；完整记录见 `reports/g2-batch-cell-csr-build-hip-boundary.md`。
- 下一小点将该 isolated CSR output 传入现有 Torch query，先验证 PBC/no-PBC、half/full 与 Batch 下
  public neighbor pair/shift 的集合和顺序，决定是否需要 canonicalization；仍不接 runtime。

## 38. T1 阶段二 HIP build + Torch direct-query public order（2026-09-19）

- 小点 14 证明当前 Torch atom-centric direct query 的 `_sort_pairs` 是 atomic fill 内部 order 的
  canonicalization boundary：per-cell atom-list permutation 不改变最终 matrix/counts/pair shifts，
  也不改变已实现 distance/vector output 的 row order。因此在这一已实现 query 路径上，无需新增
  fill-side sort/canonicalization。
- CPU focused `43 passed`，新增 PBC/no-PBC、full/half 四种 CSR cell-list permutation regression。
  BW200/gfx936 device-visible HCU 0 上，HIP fusion/scan/fill output 传入现有 Batch Torch query；16/11
  atom mixed PBC/no-PBC triclinic Batch 的 FP32/FP64、full/half 与 non-default stream，所有 public
  outputs 和 order 都与 stable Torch composition 一致。
- 这只覆盖 direct atom-centric query 的已有 matrix/count/shift/distance/vector surface；不覆盖
  target indices、pair_fn/pair outputs、pair-centric/sorted query、compile、gradient 新证据、native query
  或 performance。未改 dispatcher、AOT、`auto` 或 `hip` capability；完整记录见
  `reports/g2-batch-cell-build-query-order-boundary.md`。
- 下一小点以完整 `HIP build + Torch query` 对 `Torch build + Torch query` 做固定 contract 的预热后
  device-time benchmark/profile，再依据热点选择 native irregular query 或 Triton materialization 候选；
  不提前变更 runtime。

## 39. T1 阶段二 HIP build + Torch query benchmark/profile（2026-09-19）

- 小点 15 固定 grid metadata、FP32、2x2048 atoms、46592 global cells、mixed-PBC 和 capacity 256，
  以 HIP events 分别测 build-only、shared Torch query-only 和 full build+query；JIT、初始化、分配、
  Python wall time 与 runtime selection 排除在计时外。5 warm-up、8 交替 samples，计时前后 metadata、
  cell membership 和 public matrix/counts/shifts 均通过。
- uniform/clustered 的 native/Torch build median 为 `1.90568/1.60848` 与 `1.90768/1.61168 ms`，
  即 native 慢 `18.48%/18.37%`；query median 基本持平（native 慢 `0.25%`、快 `0.05%`）；完整
  native/Torch median 为 `9.42998/9.12286` 与 `11.69894/11.37270 ms`，即 native 慢
  `3.37%/2.87%`。因此当前 native build 没有收益，full scope 的主要时间在现有 Torch query。
- 受限 `hipprof` trace 另含进程 JIT/module load，`hipModuleLoadData` 约占 API trace `67.69%`，不能
  当作稳态比例；HIP OPS 仅支持识别 query 的 index/elementwise/reduction/sort 碎片方向。没有修改
  kernel、query、dispatcher、AOT 或 `auto`；完整记录见 `reports/g2-native-batch-cell-build-query-benchmark.md`。
- 下一小点先冻结 query/materialization reference contract 和 HIP/Triton 候选边界，再实现一个最小
  query candidate 的 correctness probe；不提前接 runtime。

## 40. T1 阶段二高 Batch 小体系性能矩阵（2026-09-19）

- 小点 16 按新增测试要求补充 `46/92 atoms per system × 32/64 systems`，同时覆盖 uniform 和
  clustered positions。HCU 0、BW200/gfx936、FP32、3 warm-up、5 交替 samples 下，global cells
  为 `745472/1490944`，capacity 256 未溢出，build metadata、cell membership 和 public query
  outputs 均在计时前后通过 parity。
- native build 在 8 个 workload 中均比 Torch 慢约 `31%--45%`；native/Torch query median
  差异约 `-0.62%--+0.06%`，说明两者接近持平；full 除 `92×64 uniform` 的高方差样本外，native
  慢约 `0.34%--0.63%`。因此当前 build candidate 没有性能收益，优化重点应转向 query/materialization。
- `2×16384` 的 capacity 512 fixture 曾明确报 `row 16385 count 540` overflow，改用 capacity 1024
  后完成；这是容量边界证据，不是正确性失败。后续 benchmark 必须把 active neighbor count、
  preallocated capacity 和 overflow 行为分别记录。
- 这仍是窄的 FP32、单卡、固定 grid、atom-centric direct query 证据；不覆盖 FP64 性能、rebuild、
  梯度、pair-centric、target/pair outputs、compile、分布式或端到端 MACE/FIRE2。没有修改 kernel、
  runtime、`auto` 或 `hip` capability；完整记录见 `reports/g2-batch-scale-performance-matrix.md`。
- 下一小点先冻结 query/materialization contract：保留 Torch reference 作为独立 oracle，明确 HIP
  不规则枚举和 Triton 规则分块的职责边界，再实现一个最小 query candidate correctness probe；
  在 correctness 和代表性 benchmark 通过前不接入 runtime。

## 41. T1 阶段二 native HIP Batch query enumeration candidate（2026-09-19）

- 小点 17 将 query 拆为 native pair enumeration 与 Torch public materialization 两层。native
  candidate 按 source atom 遍历邻近 cell，执行 PBC wrapping/image shift、严格 cutoff、self
  zero-image 排除和 full/half 过滤；输出 row-major 但 row 内 unordered 的 atom/shift 候选。
- Torch reference 仍负责稳定 public order、neighbor matrix scatter、distance/vector 以及连续
  路径梯度。candidate 本身 forward-only，不支持 `target_indices`、pair callbacks、pair-centric
  query 或 compile，不接 dispatcher、`auto` 或 `hip` capability。
- CPU boundary `2 passed`。HCU 0/BW200/gfx936 上实际 compile/load/run，FP32/FP64、2-system
  mixed PBC、full/half、non-default stream 的 pair/shift parity 与 Torch reference 通过；candidate
  加 Torch materialization 的 distance/vector 也一致；capacity 过小明确抛出 overflow。
- 该证据只证明最小 enumeration candidate 的语义边界，不包含性能收益结论。下一小点在
  `46/92 × 32/64` 高 Batch 矩阵上比较 native enumeration + Torch materialization 与完整 Torch
  query，再决定 canonicalization/materialization 适合 HIP 还是 Triton；通过 benchmark 和更宽
  数值/梯度验证前不接入 runtime。完整记录见 `reports/g2-batch-cell-query-candidate-hip-boundary.md`。

## 42. T1 阶段二 native query + Torch materialization 高 Batch benchmark（2026-09-20）

- 小点 18 在 HCU 0/BW200/gfx936、FP32、cutoff `0.6`、capacity 256、3 warm-up/5 samples 下，
  对 `46/92 atoms × 32/64 systems` 的 uniform/clustered workload 分别测 native enumeration、
  native enumeration + Torch stable matrix/count/shift materialization，以及完整 Torch query。
  计时为预热后 HIP-event device timeline，不包含 JIT、分配、Python wall time 或 runtime selection。
- 8 个 workload 均通过 candidate public matrix/count/shift parity；native enumeration + Torch
  materialization median 为 `1.98--2.22 ms`，Torch query 为 `89.87--223.98 ms`，candidate factor
  为 `45.44x--101.16x`。candidate full 相对总体标准差最高 `3.24%`，但该结果只覆盖 matrix/
  count/shift surface。
- 当前 benchmark 不请求 optional distance/vector outputs；native candidate 是 forward-only，
  不提供 autograd，也不覆盖 API wall-clock。因此不能将该 factor 写成完整上游邻居算子或端到端
  MACE/FIRE2 加速；没有修改 dispatcher、`auto` 或 `hip` capability。完整记录见
  `reports/g2-batch-query-materialization-benchmark.md`。
- 下一小点先补 native candidate 的 distance/vector materialization 与连续路径梯度验证，在同一
  高 Batch 矩阵复测后，再决定 canonicalization/materialization 是否需要 HIP/Triton 实现以及何时
  进入 runtime。

## 43. T1 阶段二 native query + Torch geometry materialization（2026-09-20）

- 小点 19 新增私有 `materialize_batch_query_candidate_into`，将 native HIP 的离散 candidate
  与公共连续几何路径明确分层：它复用 reference `_sort_pairs` 恢复 stable matrix/count/shift
  order，再用原始 `positions`/`cells` 计算 vectors/distances。candidate topology 不可微，
  geometry 保留 positions/cells 的一阶和二阶 autograd；没有 CPU fallback、detach 或 dtype 改写。
- CPU focused `6 passed`；HCU 0/BW200/gfx936 的边界 probe 通过 FP32/FP64、mixed PBC、full/half、
  empty、non-default stream、显式 overflow、public parity 和一二阶 geometry gradient。该 helper
  仍是私有隔离桥，不接 dispatcher、AOT、`auto` 或 `hip` capability。
- `--geometry` benchmark 在 `46/92 atoms × 32/64 systems`、uniform/clustered、FP32、3 warm-up/
  5 samples 的 8 个 workload 中均通过计时前后 parity；native enumeration + Torch geometry
  materialization 相对完整 Torch geometry query 为 `35.20x--79.49x`。该结果只代表预热 HIP-event
  device timeline，API wall-clock、FP64 performance、rebuild、target/pair outputs、compile 和
  end-to-end 仍未验证。完整记录见 `reports/g2-batch-query-geometry-materialization.md`。
- 下一小点基于该完整 public geometry contract，分别评估 Triton 规则分块与 HIP 不规则路径能否
  替代排序/scatter/geometry 的子阶段；仍先隔离 correctness + representative benchmark，不改
  runtime backend policy。

## 44. T1 阶段二 materialization topology/geometry breakdown（2026-09-20）

- 小点 20 将私有 materialization helper 拆成固定 candidate 上的三个可测 scope：stable topology
  canonicalization/scatter、canonical geometry distance/vector、完整 helper。公共 helper 接口、
  reference `_sort_pairs`、distance/vector 公式和 autograd 合同保持不变。
- 在 HCU 0/BW200/gfx936、FP32、3 warm-up/5 samples、`46/92 atoms × 32/64 systems`、
  uniform/clustered 的 8 个 workload 中，topology median 占完整 helper `72%--75%`，geometry
  median 占 `17%--19%`；计时前后 topology、geometry 和完整 public outputs 均通过 parity。
  原始 samples 和完整表见 `reports/g2-batch-query-materialization-breakdown.md`。
- HCU correctness probe 重跑通过 FP32/FP64、full/half、mixed PBC、empty、non-default stream、
  overflow 和一/二阶 geometry gradient。该小点只产生 profile evidence，没有新增 HIP/Triton
  kernel，也没有接入 dispatcher、AOT、`auto` 或 `hip` capability。
- 下一小点优先验证 HIP 不规则 topology compaction/scatter/sort candidate；Triton geometry
  candidate 延后作为独立实验，是否继续由 correctness、梯度和代表性 benchmark 决定。

## 45. T1 阶段二 native HIP topology materialization candidate（2026-09-20）

- 小点 21 新增隔离 `_hip_batch_query_materialize.py` 与 native C++/HIP 实现，将 unordered
  Batch query candidate 规范化为 public matrix/count/shift。排序逻辑对照上游 `_sort_pairs`，以
  `shift_z -> shift_y -> shift_x -> column -> row` 五次稳定 rocPRIM radix sort 复现最终
  `(row, column, shift_x, shift_y, shift_z)` 顺序，再用 exclusive scan 的 row starts 做 scatter。
- CPU focused `21 passed`；HCU 0/BW200/gfx936 上 FP32/FP64、mixed PBC、full/half、2-system
  Batch、empty、non-default stream 的 topology parity 通过；原有 candidate 的 distance/vector
  public parity、一二阶 geometry gradient 和 overflow 也重跑通过。DTK 26.04 下 `.cu -> .hip`
  的 hipcc 编译/链接成功，当前本机目标架构为 gfx936。
- 当前实现只用于 correctness：C++ 内部分配临时 workspace/字段 buffer，并通过
  `candidate_counts.sum().item()` 得到 scatter 长度，存在 host synchronization；尚未做性能
  benchmark，不得将其称为比 Torch 快，也不接 dispatcher、AOT、`auto` 或 `hip` capability。
  不覆盖 geometry/autograd、target/pair outputs、pair-centric query、compile/opcheck、rebuild、
  FP64 performance、API wall-clock 或端到端 MACE/FIRE2。下一小点在 `46/92 × 32/64` 高 Batch
  矩阵上测 native topology candidate 对 Torch topology stage 的稳态 device time，并继续保留
  parity gate。

## 46. T1 阶段二 native HIP topology materialization benchmark（2026-09-20）

- 小点 22 在固定 native unordered candidate 上对比 point20 Torch topology stage 与 point21
  HIP radix-sort/scan/scatter candidate；测试为 HCU 0/BW200/gfx936、FP32、`46/92 × 32/64`、
  uniform/clustered、3 warm-up/5 samples。所有 8 个 workload 的计时前后 matrix/count/shift
  parity 通过。
- HIP/Torch median factor：`46×32` 为 `0.830x--0.860x`；`46×64` 为 `1.175x--1.274x`；
  `92×32` 为 `1.240x--1.365x`；`92×64` 为 `1.636x--1.696x`。仅 `46×32` 出现局部收益，
  高 Batch 下临时字段/workspace、五次 radix sort 和 gather/scatter 成本使 HIP candidate 变慢。
- native scope 前显式 `torch.cuda.synchronize()`，用于隔离前序 Torch 异步 work 的 stream 边界，
  不计入 HIP event；candidate 内 C++ allocation 与 host-side pair-count sync 仍在实现中。该
  benchmark 是 topology stage device timeline，不是 API wall-clock、完整 neighbor 或端到端结果。
- 结论：保留 candidate 作为 correctness/profile slice，不登记 `hip` capability、不接 dispatcher
  或 `auto`。下一小点分析排序/字段 gather/workspace 的成本，优先评估 single composite key、
  workspace reuse、专用 compaction/scatter 或 Triton 规则分块；重新通过同一矩阵后才考虑更宽
  后端边界。完整结果见 `reports/g2-batch-query-topology-materialization-benchmark.md`。

## 47. T1 阶段二 topology materialization kernel profile（2026-09-20）

- 小点 23 使用正确的项目 DTK 环境入口，在 HCU 0/BW200/gfx936 上对 `46×32` 和 `92×64`
  uniform/clustered workload 做受限 `hipprof --stats --hip-trace`。profile 前后四个 workload
  的 public matrix/count/shift parity 均通过；该证据只用于 kernel hotspot 定位，不作为稳态
  benchmark 或 `auto` 准入。
- 按 native candidate 的调用次数归并，`46×32` 五轮 sort 相关 kernel 每次约 `0.755 ms`，
  `92×64` 约 `1.638 ms`；同一归并下 gather 为 `0.034/0.113 ms`，prepare fields 为
  `0.011/0.030 ms`，public scatter 为 `0.007/0.009 ms`。高输入下 rocPRIM 从 merge-path 类
  sort kernel 切换到 onesweep 类 kernel，说明不能固定假设一种 sort 实现。
- 混合进程 kernel CSV 不能无歧义地分离 exclusive scan、public output initialization、allocator
  或 `candidate_counts.sum().item()` host synchronization；这些仍是后续独立实验边界。当前
  candidate 仍不接 dispatcher、AOT、`auto` 或 `hip` capability，完整记录见
  `reports/g2-batch-query-topology-materialization-profile.md`。
- 下一小点为保持相同 lexicographic public order 的 single composite key correctness/overflow
  experiment；workspace reuse 和 host-sync removal 单独评估，避免把算法变化与生命周期变化混成
  一个性能结论。

## 48. T1 阶段二 single composite-key topology candidate（2026-09-20）

- 小点 24 新增隔离的 single composite-key topology candidate：将
  `row -> column -> shift_x -> shift_y -> shift_z` 与 signed PBC shift bias 编码为一个
  非负 signed `int64` key，使用一次 rocPRIM radix sort 后恢复 public matrix/count/shift。
  位宽上限固定为 `2 * row_bits + 3 * shift_bits <= 63`，输入 shift 超出编码范围显式失败，
  不允许截断或静默改 dtype。
- 对照上游 `_sort_pairs` 语义和现有五次 int32 stable-sort HIP candidate，HCU 0/BW200/gfx936
  的 unordered `5 atoms × 5 capacity` mixed-sign PBC fixture 在默认 stream 与非默认 stream
  均通过 public topology parity；故意构造的 shift overflow 被拒绝。CPU focused `5 passed`。
- 这里的 int64 是 packed key 的标量存储容器，不是 FP64，也不表示 gfx936 matrix core 具备
  int64 高吞吐路径。int64 radix sort 的吞吐、带宽和 workspace 成本尚未测量，因此不能由
  “排序次数减少”推出性能收益。候选仍不接 dispatcher、AOT、`auto` 或 `hip` capability，
  也不覆盖 geometry/autograd、target/pair outputs、compile/opcheck 或端到端。
- 完整记录见 `reports/g2-batch-query-topology-materialization-composite-key.md`。下一小点
  在同一 `46/92 × 32/64` uniform/clustered 矩阵上做 int64 single-key 与五次 int32 sort 的
  parity-gated device-time 对照；workspace reuse 和 host-sync removal 继续拆开。

## 49. T1 阶段二 composite-key topology benchmark（2026-09-20）

- 小点 25 扩展统一 benchmark probe，同时测 Torch topology、五次 int32 HIP stable-sort 和
  single-int64-key HIP candidate；所有候选使用同一个 native unordered query candidate、固定
  CSR/grid metadata、3 warm-up/5 samples 和 HIP-event device-time scope。
- 在 HCU 0/BW200/gfx936、FP32、`46/92 atoms × 32/64 systems`、uniform/clustered 的 8 个
  workload 上，计时前后 public matrix/count/shift parity 全部通过。composite 相对五次
  int32 sort 的 median factor 为 `0.310x--0.406x`，相对 Torch topology 为 `0.289x--0.526x`；
  `46×32` 的 composite samples 波动较大，最大 relative population std 约 `12.3%`。
- 结论仅支持将 single composite key 保留为高优先级 HIP topology candidate；不等于完整
  neighbor/API wall-clock/端到端加速，也不能把该结果推广成 gfx936 通用 int64 性能结论。
  两种 HIP candidate 的 C++ 临时分配和 host pair-count sync 均仍在 scope 外/实现内保留，
  尚未接入 dispatcher、AOT、`auto` 或 `hip` capability。
- 完整结果见 `reports/g2-batch-query-topology-materialization-composite-key-benchmark.md`。
  下一小点先复用现有 correctness boundary 覆盖 FP32/FP64、full/half、empty、mixed-PBC、
  capacity 和 stream，再独立测试 workspace reuse/host-sync removal。

## 50. T1 阶段二 composite-key topology correctness boundary（2026-09-20）

- 小点 26 复用完整 native Batch query/materialization boundary，将 composite int64 topology
  materializer 与 Torch reference、五次 int32 HIP stable-sort candidate 并列执行。HCU 0/BW200/
  gfx936 上 FP32/FP64、mixed-PBC、full/half、non-default stream、empty、native query capacity
  overflow 和 composite public-capacity rejection 均通过。
- probe 输出的 public matrix/count/shift 在所有 case 与 reference/五次 HIP candidate 一致。
  既有 geometry distance/vector 及一/二阶梯度检查继续验证 Torch continuous path，但没有把
  composite topology candidate 宣称为 geometry/autograd 实现。
- 该点仍未测性能、FP64 timing、API wall-clock、workspace reuse 或 host-sync removal，不接
  dispatcher、AOT、`auto` 或 `hip` capability。完整记录见
  `reports/g2-batch-query-topology-materialization-composite-key-boundary.md`。
- 下一小点可独立选择 workspace 生命周期或 host-side pair-count synchronization 实验；两者
  不与 composite 算法变化混合，并继续保留本点 parity gate。

## 51. T1 阶段二 composite-key workspace reuse（2026-09-20）

- 小点 27 新增 caller-owned `CompositeTopologyWorkspace`：预分配并复用 int64 key、int32
  order/row starts、uint8 sort/scan scratch；native binding 增加 workspace-size query 和
  workspace execution path。原 direct path 保留，未切换默认实现。
- CPU focused `6 passed`；HCU 0/BW200/gfx936 完整 boundary 通过 FP32/FP64、mixed-PBC、
  full/half、empty、stream、capacity，并验证同一 workspace 连续调用两次的 public parity。
- 在 `46/92 × 32/64`、uniform/clustered、3 warm-up/5 samples 矩阵上，8 个 workload parity
  均通过。reused/direct device-time factor 为 `0.900x--1.026x`；显式同步 API wall-clock
  factor 为 `0.902x--1.129x`，没有稳定全矩阵收益。该点保留为可复用基础模块，不接
  dispatcher、AOT、`auto` 或 `hip` capability；`candidate_counts.sum().item()` host sync
  仍未移除。
- 完整记录见 `reports/g2-batch-query-topology-materialization-workspace.md`。下一小点独立
  评估 device-side pair count/容量安全 scatter，单独测 host-sync removal。

## 52. T1 阶段二 topology materialization host-sync removal（2026-09-20）

- 小点 28 移除了五次 int32 stable-sort 和 single-int64 composite-key topology scatter 路径中
  的 `candidate_counts.sum().item()` host synchronization。scatter 按
  `atoms × candidate_capacity` 发射，kernel 从 sorted order 恢复原始 row/slot，并在 device
  侧根据 `candidate_counts` 跳过 padding；排序字段、public matrix/count/shift、容量和 stream
  语义不变。
- CPU focused `6 passed`；HCU 0/BW200/gfx936 boundary 通过 FP32/FP64、mixed-PBC、full/half、
  empty、non-default stream、capacity、workspace reuse 和全零 candidate-count case。相同
  `46/92 × 32/64`、uniform/clustered、3 warm-up/5 samples 的 8 个 workload 全部 parity 通过。
- point28 direct 相对 point27 direct 的描述性 device median factor 为 `0.817x--0.914x`，API
  wall-clock factor 为 `0.858x--1.042x`；两轮不是交替配对样本，不能直接写成稳定 speedup。
  固定容量 launch 在稀疏 candidate 上可能增加无效线程，因此该改动仍不接 dispatcher、AOT、
  `auto` 或 `hip` capability。
- 完整记录见 `reports/g2-batch-query-topology-materialization-host-sync.md`。下一步可做 HIP
  build/query/topology + Torch geometry 的完整 hybrid pipeline 对照，再独立评估 geometry 的
  HIP/Triton 实现。

## 53. T1 阶段二 hybrid HIP topology plus Torch geometry（2026-09-20）

- 小点 29 新增 `materialize_batch_query_topology_geometry_into`，将 native HIP query +
  composite-key topology 的 public matrix/count/shift 接入 Torch distance/vector geometry。
  topology 作为离散输入，positions/cells 上的 geometry 仍保持普通 Torch 表达式和梯度路径。
- CPU focused `11 passed`，含 topology-to-geometry reference parity 和一/二阶 geometry
  gradient；HCU 0/BW200/gfx936 上 FP32/FP64、mixed-PBC、full/half、empty、capacity、stream
  和完整 public output parity 通过。
- 在固定 CSR metadata、不重复计 cell-list build 的 query/materialization scope，`46/92 × 32/64`
  uniform/clustered 8 workload 的 hybrid/Torch device factor 为 `0.0120x--0.0257x`，API
  wall factor 为 `0.0124x--0.0256x`。该结果不能扩大为完整 neighbor/MD/MACE 加速；native
  topology 仍不接 dispatcher、AOT、`auto`、`hip` capability 或 autograd。
- 完整记录见 `reports/g2-batch-query-hybrid-torch-geometry.md`。下一步独立评估 forward-only
  HIP/Triton geometry candidate，再决定是否值得补齐训练路径的 native autograd。

## 54. T1 阶段二 native HIP forward geometry candidate（2026-09-20）

- 小点 30 新增隔离 `materialize_batch_query_geometry_hip_into` 和
  `_native/batch_query_geometry.cpp/.cu`：在 public canonical matrix/count/shift topology
  上按 `(row, slot)` 线程计算 forward distance/vector，并接入 native query + composite-key
  topology 的 hybrid probe。它不提供 autograd/二阶梯度，不接 dispatcher、AOT、`auto` 或
  `hip` capability；Torch geometry 仍是训练/reference 路径。
- CPU focused `8 passed`；HCU 0/BW200/gfx936 的 FP32/FP64、mixed-PBC、full/half、empty、
  capacity、zero-count、non-default stream 和 public output parity 通过。native FP32 与
  Torch 的小量 `sqrt` 舍入差异按 `1e-6` 检查，拓扑输出仍严格逐元素一致。
- 在固定 public topology 的 geometry-only scope，`46/92 × 32/64` uniform/clustered 8
  workload 的 native/Torch device factor 为 `0.0784x--0.1220x`，API factor 为
  `0.1092x--0.1537x`；窄完整 hybrid factor 为 device `0.0069x--0.0146x`、API
  `0.0075x--0.0155x`。不包含 cell-list build，不能扩大为完整 neighbor/MD/MACE 加速。
- 完整记录见 `reports/g2-batch-query-native-hip-geometry.md`。下一小点先决定保留 Torch
  geometry 的梯度路径还是设计 native autograd contract，再做 build-inclusive end-to-end
  benchmark；不要直接把当前 forward-only candidate 接入训练。

## 55. T1 阶段二 build-inclusive neighbor benchmark（2026-09-20）

- 小点 31 新增 `probes/native_batch_cell_build_query_materialization_benchmark.py`，把每次
  调用的测量范围扩展为 `cell-list build -> query candidate -> topology materialization ->
  geometry`。三条隔离路径分别是 Torch reference、native build/query/topology + Torch
  geometry、native build/query/topology + native HIP forward-only geometry；不改变默认
  dispatcher/runtime 选择。
- HCU 0/BW200/gfx936、DTK 26.04、PyTorch HIP `6.3.26093`，FP32、capacity 256，混合
  PBC/non-orthogonal cell，`46/92 × 32/64`、uniform/clustered 共 8 workload，3 warm-up/
  5 samples。每次 full call 都重新 build 并覆盖所有输出；JIT、首次分配、固定 metadata、
  buffer 和 workspace 排除。完整 output parity 通过，原始 samples 在
  `artifacts/native-batch-cell-build-query-materialization-benchmark/`。
- native build-only / Torch build factor 为 `1.311x--1.399x`。保留 Torch geometry 的
  full hybrid / Torch factor 为 device/API `0.0246x--0.0510x`/`0.0248x--0.0501x`；native
  forward geometry full factor 为 `0.0193x--0.0384x`/`0.0195x--0.0381x`。这些是本 probe
  的预热 HIP-event 和显式同步 API wall-clock 结果，不能扩大为完整 MD/MACE 或默认 neighbor
  API 加速。
- Torch geometry 继续作为训练/梯度路径；native geometry 明确 forward-only，不提供
  autograd/二阶梯度。三条路径均未接 dispatcher、AOT、`auto`、`hip` capability，未覆盖
  rebuild/skin、FP64 performance、JIT/分配成本、`target_indices`、`pair_fn`/pair outputs、
  compile 或 DomainParallel。
- 下一步建议不直接接 runtime，先独立拆分 native build 的 key/count、scan、atomic fill 和
  launch/occupancy 成本，并按稀疏/密集负载确认优化方向；保留 Torch geometry 的梯度路径。
  完整记录见 `reports/g2-batch-cell-build-query-materialization-benchmark.md`。

## 56. T1 阶段二 native HIP build breakdown profiling（2026-09-20）

- 小点 32 新增 `probes/native_batch_cell_build_breakdown_benchmark.py`，分别测 native fused
  key/count、rocPRIM scan、atomic fill、full build 与 Torch build。isolated scan/fill 的前序
  准备工作位于计时起点之前；full build 每次仍执行全部三段。所有 8 个 workload 均在计时前后
  通过 shifts/mapping/keys/counts/starts 与 cell-membership parity。
- HCU 0/BW200/gfx936、FP32、mixed PBC/non-orthogonal cell，`46/92 × 32/64`、uniform/
  clustered、3 warm-up/5 samples。74.5 万 global cells 的 key/count/scan/fill 各约
  `0.80/0.70/0.81 ms`；149.1 万 cells 为 `0.80/0.98/1.04 ms`。native/Torch full build
  factor 为 `1.311x--1.423x`，API wall-clock 与 event time 接近；不只存在一个 atomic-fill
  热点。
- 受限 `hipprof --hip-trace --stats` 仅做组成核验，module-load 占 API trace 约 68.7%，不可
  作稳态比例。GPU OPS 中有大量 `any/reduce`；结合代码审查，当前 composition 经由每段的 public
  wrapper 和 mutation-only custom-op，重复执行含 device checks 的 ABI validation。这是下一次
  优化的单一、可证伪假设，不能把 profiler 的绝对 kernel 时间直接当作 benchmark 结果。
- 下一步保持对外 `build_batch_cell_csr_hip_into` 的 validation/error contract，新增仅由已验证
  结构显式选择的私有 direct-extension composition，避免同一次调用的重复 validator device work。
  必须以现有 8 workload 的 direct/public/Torch CSR parity 和重复 timing 证明收益；否则拒绝
  fast path，再研究 global-cell clear、scan 或 fill 算法。仍不接 runtime、dispatcher、AOT、
  `auto` 或 `hip` capability。详见 `reports/g2-batch-cell-build-breakdown-profile.md`。

## 57. T1 阶段二 native HIP trusted build fast path（2026-09-20）

- 小点 33 新增私有 `_build_batch_cell_csr_hip_into_trusted`，只对已经由上层验证的精确
  buffer 集合直接调用已有 key/count、rocPRIM scan 和 atomic fill extension。公开
  `build_batch_cell_csr_hip_into` 的完整 validation/error contract 不变，trusted helper 不
  导出、不接 dispatcher、AOT、`auto` 或 `hip` capability，也不改变 CSR/PBC/dtype 语义。
- HCU 0/BW200/gfx936、DTK 26.04、PyTorch HIP `6.3.26093`，FP32、mixed PBC/non-orthogonal
  cell，`46/92 × 32/64`、uniform/clustered 8 workload，3 warm-up/5 samples。direct/public/
  Torch 的 keys/counts/starts、cell membership、cursor 和 capacity tail parity 全部通过；
  trusted/public HIP-event factor `0.0285x--0.0318x`，trusted/Torch `0.0398x--0.0423x`，
  API wall-clock 同方向。CPU focused suite `26 passed`。
- 该结果支持“重复 public/custom-op validation 是显著开销”的假设，但只是 isolated build
  evidence，不是完整 neighbor/MD/MACE 或默认 runtime 加速。未覆盖 FP64 performance、
  rebuild/skin、JIT/分配成本、autograd/compile、target/pair outputs、DomainParallel。
- 下一步评估可安全复用的 prevalidated plan/lifecycle，并重新测含 query/materialization/
  geometry 的完整流程；若无法清晰表达 validation ownership，则保留 public path，转向
  clear/scan/fill 算法优化。完整记录见 `reports/g2-batch-cell-build-trusted-fastpath.md`。

## 58. T1 阶段二 native HIP prevalidated build plan（2026-09-20）

- 小点 34 增加私有 `_TrustedBatchCellBuildPlan`。`initialize` 先执行一次完整 checked public
  build，然后保存精确的 tensor/workspace 引用；后续 `run` 直接执行 trusted composition。
  plan 有明确生命周期：storage、shape、dtype、device、alias、cell metadata、capacity、
  workspace 和 offset 变化时必须重新初始化；positions 数值可在同一 storage 内更新。
- HCU 0/BW200/gfx936、FP32、`46/92 × 32/64`、uniform/clustered 的 8 workload 全部通过
  plan/public/Torch CSR parity。trusted/public HIP-event factor `0.0281x--0.0313x`，
  trusted/Torch `0.0394x--0.0413x`；CPU focused `21 passed`。API wall 只作辅助证据，
  46×64 clustered 存在一个异常样本。
- 该 plan 仍是 isolated build candidate，不接 dispatcher、AOT、`auto` 或 `hip` capability，
  也不代表完整 neighbor/MD/MACE 支持。下一步将其接入已有完整 build→query→topology→geometry
  benchmark，先保留 Torch geometry 梯度路径，再决定 runtime 接线。详见
  `reports/g2-batch-cell-build-trusted-plan.md`。

## 59. T1 阶段二 trusted plan 完整邻居 pipeline benchmark（2026-09-20）

- 小点 35 将 `_TrustedBatchCellBuildPlan` 接入现有完整隔离流程：cell-list build、native
  query candidate、composite-key topology 和 Torch geometry。public native、trusted plan 和
  Torch reference 三条训练兼容路径均保留；public/trusted geometry output 使用独立 buffer。
- HCU 0/BW200/gfx936、FP32、capacity 256、mixed PBC/non-orthogonal cell，`46/92 × 32/64`、
  uniform/clustered、3 warm-up/5 samples 的 8 workload 全部通过完整 output parity。public
  native/Torch full HIP-event factor 为 `0.0249x--0.0501x`，trusted plan/Torch 为
  `0.0121x--0.0257x`，trusted/public native 为 `0.472x--0.544x`；API wall 与 event 同方向。
- 该结果证明 trusted build 的局部优化可传递到隔离 full forward pipeline，但不代表默认
  neighbor、MACE/FIRE2、变胞、rebuild/skin 或生产 runtime 支持。Torch geometry 继续承担
  训练/力梯度路径；native geometry 仍为 forward-only。未覆盖 FP64 performance、half-list、
  target/pair outputs、compile/opcheck 和 DomainParallel。
- 下一步设计 runtime-facing capability/错误合同，先将 fixed-cell Torch reference 接线和
  回归闭合，再决定是否允许显式 HIP 选择。完整记录见
  `reports/g2-batch-cell-build-query-materialization-trusted-plan.md`。

## 60. T1 阶段二 Point 36 runtime boundary slice（2026-09-20）

- Point 36 的第一步已把已登记的 fixed-cell Torch reference cell-list 接到 framework
  的真实消费链：periodic `NeighborListHook(backend="torch_reference", method="cell_list")`
  写入 Batch 的 neighbor matrix/image shifts，随后由 Torch reference LJ wrapper 计算
  energy/force。CPU framework focused suite 为 `33 passed, 1 warning`，ops registry/cell-list
  suite 为 `31 passed`。
- 同时补充显式 `backend="hip"`, `method="cell_list"` 的失败合同：由于 native HIP
  full pipeline 仍是 isolated candidate，framework 必须抛 `BackendUnavailableError`，不得
  静默 fallback 到 Torch。`backend=None`/Warp 和 `auto` 行为未改变。
- 该点不登记 HIP capability，也不覆盖 MACE/FIRE2、native geometry autograd、variable-cell、
  rebuild/skin、compile/opcheck、FP64 performance 或 DomainParallel。完整证据见
  `reports/g2-fixed-cell-torch-reference-runtime-boundary.md`。
- 下一步继续 Point 36：把该 runtime-facing capability/error contract 收敛为可审查的
  fixed-cell 选择记录，并冻结未来 native HIP wrapper 的最小 ABI；在此之前不让
  `backend="hip"` 进入 registry/`auto`。

## 61. T1 阶段二 Point 36.2 selection contract（2026-09-20）

- `compute_neighbors` 与 `NeighborListHook` 现在共同使用
  `nvalchemi._backend.resolve_neighbor_list_backend` 构造邻居 topology selection：
  PBC/no-PBC、full/half、MATRIX/COO、method strategy、device 和 dtype 统一进入中央
  `BackendSelection`。这避免未来 native HIP wrapper 在两个 framework 入口复制 capability
  判断。
- helper 只描述 topology capability，不把邻居 topology selection 推断成 geometry、force
  gradient 或完整 native pipeline capability。`backend=None`/Warp、`auto` 和显式 unsupported
  backend 的既有行为保持不变。
- CPU framework focused suite 为 `34 passed, 1 warning`；`py_compile` 和
  `git diff --check` 通过。该点没有新增 HCU kernel 或性能证据，native HIP 仍未注册。
- 下一步为 Point 36.3：定义 native HIP runtime wrapper 的最小 ABI 和 admission gate，覆盖
  workspace/lifetime、query candidate、public topology、Torch geometry 及 unsupported
  gradient/rebuild/variable-cell 请求。

## 62. T1 阶段二 Point 36.3 native HIP runtime ABI/admission gate（2026-09-20）

- 新增 `nvalchemiops._hip_batch_neighbor_runtime`，冻结最小候选流水线：checked build
  initialization、trusted reusable CSR workspace、native unordered candidate query、
  composite-key canonical full topology、Torch reference distance/vector geometry。
  workspace/lifetime 和 query-order 边界写入 ABI record；Torch geometry 保留 0/1/2 阶
  连续梯度语义，native topology 仍是离散 forward-only stage。
- gate 只接受 Batch、fixed-cell、full-list、float32/float64 和 Torch geometry；显式拒绝
  CPU、float16、half-list、variable-cell、skin/rebuild、target/pair outputs、
  DomainParallel 和 native geometry，并统一抛 `BackendUnavailableError`，禁止静默
  fallback。该 gate 只做 semantic admission，不检查实际设备可用性。
- CPU contract suite 为 `15 passed`，`git diff --check` 通过。本点未新增 HCU kernel 或
  device execution；`backend="hip"` 仍未注册，MACE/FIRE2、compile/opcheck 和生产性能
  仍未验证。完整记录见 `reports/g2-native-hip-neighbor-runtime-abi.md`。
- 下一步是把该 gate 接到不改变 `backend=None`/Warp、Torch reference 和 `auto` 默认行为的
  native wrapper 骨架/显式调用边界；在此之前不宣称 native HIP runtime 已支持。

## 63. T1 阶段二 Point 37 native HIP runtime wrapper（2026-09-20）

- 新增显式 `nvalchemiops.run_native_hip_batch_neighbor_into`：调用方提供已 checked
  initialization 的 `_TrustedBatchCellBuildPlan`、candidate/public/geometry outputs 和
  composite topology workspace；wrapper 按 `trusted build -> native query -> composite
  topology -> Torch geometry` 执行，不引入新的 backend-ID 分派表。
- admission gate 在 native mutation 前执行，wrapper 只覆盖 Batch、fixed-cell、full-list、
  float32/float64 和 Torch geometry；native topology 保持离散，Torch geometry 保留位置
  一阶/二阶梯度。`backend="hip"` 仍未注册，默认 Torch/Warp/`auto` 不变。
- Point37 期间发现并修复 composite workspace size-query 的高 Batch 类型/容量问题：不能把
  int32 candidate storage reinterpret 为 int64 key storage，size query 现在使用正确类型的
  临时 scratch。CPU ops suite `109 passed, 1 warning`；gfx936/BW200 HCU 0 的 `46×32`、
  `92×64` parity 和 `46×32` 一阶/二阶位置梯度均通过。完整记录见
  `reports/g2-native-hip-neighbor-runtime-wrapper.md`。
- 该点仍未接 framework registry/executor、MACE/FIRE2、native geometry backward、
  half-list、skin/rebuild、变胞、DomainParallel、compile/opcheck 或性能准入。下一步为
  Point 38：接入 framework 的显式 HIP selection/executor 边界，同时保持默认路径不变。

## 64. T1 阶段二 Point 38 framework native HIP neighbor executor（2026-09-20）

- 新增 `hip.neighbor.cell_list-v1` registry implementation 和
  `nvalchemiops._hip_neighbor_executor.neighbor_list`，通过通用 executor binding 将公开
  `dispatch_neighbor_list` ABI 接到 Point37 native pipeline；没有新增 implementation-ID
  分派表，也没有改变 `backend=None`/Warp、Torch reference 或 `auto` 默认行为。
- 当前登记宽度固定为 periodic、Batch、fixed-cell、full-list、MATRIX、FP32/FP64；adapter
  负责 Batch/cell metadata、native workspace、capacity retry 和 public output。未提供
  `max_neighbors` 时从体系规模起步并在保守上界内增长，capacity overflow 不截断邻居。
- Point38 明确不登记 no-PBC、half-list、COO、target/pair outputs、variable-cell、
  skin/rebuild、DomainParallel、native geometry backward 或 `auto` 性能选择。NeighborListHook
  对显式 HIP + skin/rebuild 请求直接抛错，避免把 lifecycle 支持误报为已完成。
- ops focused suite `38 passed`；framework focused suite `11 passed, 1 warning`；gfx936/BW200
  `HIP_VISIBLE_DEVICES=4` 上 `46×32` 和 `92×64` framework selection/executor/native
  build-query-topology 对照独立 Torch 分层 cell-list ABI 通过。完整记录见
  `reports/g2-framework-hip-neighbor-executor.md`。
- 这只是 explicit functional boundary，不代表 MACE/FIRE2 端到端、native geometry backward、
  完整 Hook workspace reuse、compile/opcheck 或性能准入。下一步为 Point39：在第五张 DCU
  上测 build-inclusive API wall-clock，并分别记录冷启动、稳态和端到端成本。

## 65. T1 阶段二 Point 39 framework HIP neighbor wall-clock（2026-09-20）

- Point39 新增 `probes/framework_neighbor_hip_wallclock.py`，测量公开
  `compute_neighbors` 的完整 topology-only API wall-clock：framework selection、ops
  dispatcher、executor allocation、cell-list build、native query、composite topology
  materialization 和 Batch write-back 均在范围内；距离/向量 geometry 不在本点范围。
- 期间修复了 Torch reference 大 Batch mixed-PBC 的 cell allocation/build 上限不一致，
  并将 HIP executor 的 `candidate_counts` 在 workspace allocation 前显式清零，保证重复
  调用不依赖未初始化 device memory。Torch cell-list CPU focused suite `14 passed`。
- 在 BW200/gfx936、`HIP_VISIBLE_DEVICES=4`、FP32、fixed periodic/full/MATRIX、capacity
  256 下，46×32 warm median 为 HIP `22.389 ms`、Torch reference `238.222 ms`，factor
  `0.09398x`；92×64 为 HIP `39.399 ms`、Torch reference `502.571 ms`，factor `0.07839x`。
  两组 HIP public output 均与 framework Torch reference 的 matrix/counts/shifts 对照通过。
  cold 值单独记录且包含首次 runtime/extension load 与分配，不把源文件编译时间混入性能
  结论。Hook `skin=0` 通过，`skin>0` 明确拒绝；性能原始样本在对应 `artifacts/` 目录。
- 该结果是固定 capability 的窄范围描述性证据，不进入 `auto`，不代表 MACE/FIRE2、
  lifecycle reuse、half/COO/no-PBC、变胞、native geometry backward、compile/opcheck 或
  DomainParallel 已支持。Point40 应先补 explicit HIP executor 的 capacity/empty/dtype/
  format/unsupported-request 回归矩阵，再决定是否改造 workspace/lifecycle。

## 66. T1 阶段二 Point 40 framework HIP neighbor contract matrix（2026-09-20）

- 新增 `probes/framework_neighbor_hip_contract.py`，覆盖 Point39 后的显式 HIP contract：
  FP32/FP64 正常路径、empty periodic input、显式容量溢出、no-PBC/half/COO rejection，
  以及 `NeighborListHook(skin=0)` 连续调用。
- CPU ops registry tests `7 passed`，framework neighbor tests `11 passed, 1 warning`；
  HCU `HIP_VISIBLE_DEVICES=4`、gfx936/BW200 上 46×32 FP32/FP64 matrix/counts/shifts
  与 Torch reference parity 通过。empty 输入返回 `(0,256)` matrix、空 counts/shifts；
  `max_neighbors=1` close-pair 明确报 `native HIP Batch query capacity overflow`；
  unsupported requests 均由 capability registry 明确拒绝；Hook 两次结果一致。
- Point40 不改变 `hip.neighbor.cell_list-v1` 的 capability 宽度、不进入 `auto`，也不将
  contract evidence 扩大成 skin/rebuild、变胞、target/pair、native geometry backward、
  compile/opcheck、DomainParallel 或 MACE/FIRE2 支持。完整记录见
  `reports/g2-framework-hip-neighbor-contract.md`。
- 下一点为 Point41：纳入稳定 framework/ops compatibility gate，并补
  `max_neighbors=None` 自动容量增长/overflow 上界验证，再评估 Hook workspace/lifecycle
  reuse。

## 67. T1 阶段二 Point 41 framework HIP neighbor compatibility gate（2026-09-20）

- Point41 将 `probes/framework_neighbor_hip_contract.py` 固化为当前显式 HIP 邻居的
  compatibility gate，并新增 `max_neighbors=None` 自动容量增长/有限上界检查；没有新增
  kernel、backend capability、`auto` 策略或 lifecycle reuse。
- CPU ops `7 passed`，framework `11 passed, 1 warning`，probe `py_compile` 通过；在
  gfx936/BW200、`HIP_VISIBLE_DEVICES=4` 的 46×32 HCU gate 中，单原子 `cell=2I` 周期
  image fixture 的容量从初始 `1` 增长到 `8`，最大实际邻居数 `6`，保守上界 `343`，与
  Torch reference parity 通过；Point40 的 FP32/FP64、empty、capacity overflow、unsupported
  request 和 repeated Hook 检查继续通过。测试结束后无 KFD PIDs。
- 阶段二在 periodic、fixed-cell、Batch、full-list、MATRIX、FP32/FP64、`skin=0` 和
  自动容量增长的窄范围内收口。该结论不扩大为完整上游 neighbor、MACE/FIRE2、lifecycle
  reuse、skin/rebuild、变胞、native geometry backward、compile/opcheck、DomainParallel 或
  `auto` 支持。当前 HIP `max_neighbors=None` 使用本地 doubling policy，不承诺上游
  `estimate_max_neighbors` 的最小 16/16 对齐 padded capacity 形状；完整证据见
  `reports/g2-framework-hip-neighbor-compatibility-gate.md`。
