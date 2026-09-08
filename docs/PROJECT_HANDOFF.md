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

## 11. 本轮暂停交接（2026-09-06）

### 已完成

- 根仓库分支为 `codex/g0-initialization`；上游锁定不变：framework
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

## 12. 统一 benchmark 进展（2026-09-08）

- 已补齐 unified reference benchmark 的输入元数据：每个 case 记录 CIF 路径、原始
  `source_pbc`、实际 `effective_pbc`、原子数和晶胞体积，避免将同一 CIF 的 periodic
  与 no-PBC 算法边界测量混淆。详见 `reports/g2-unified-reference-benchmark-cpu.md`。
- CPU neighbor/E2E smoke 已通过；HCU 小范围 smoke 也已通过：periodic/no-PBC full/half
  的 46/92 原子及 batch=1 契约检查退出码 `0`，设备为 `BW200, UBB BW1000`。详见
  `reports/g2-unified-reference-benchmark-hcu-smoke.md`。
- HCU 单体系规模和 `perf_92` batch 阶梯均已完成；下一步运行 periodic `[46,92]`
  MACE/FIRE2 固定晶胞 100 步，完整结果前不启动 cell-list 实现。
- HCU 单体系规模阶梯现已完成：periodic full、no-PBC full/half 的 46/92/184/368
  全部通过；periodic steady 为 `3.243/3.702/5.753/13.969 ms`。
- `perf_92` 的 HCU batch=1/4/8/16/32 阶梯现已完成；periodic steady 为
  `3.768/9.789/17.819/33.571/65.577 ms`，batch=32 与 Tier-1 `54,760` 边基线连续。
  下一小步运行 periodic `[46,92]` MACE/FIRE2 固定晶胞 100 步端到端，记录总耗时、
  steady step 和 stage timing，再决定 cell-list 的切入位置。
