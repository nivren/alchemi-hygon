# 海光 DCU 开发者指南

本文面向要修改 Triton、HIP、Torch reference，移植新算子，或增加上层功能的开发者与
开发 agent。它补充而不替代：

- [AGENTS.md](../AGENTS.md)：必须遵守的项目级规则；
- [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)：架构背景、阶段目标和历史交接；
- [FEATURE_COMPATIBILITY.yaml](FEATURE_COMPATIBILITY.yaml)：逐特性契约和状态；
- [UPSTREAM.md](UPSTREAM.md) / [UPSTREAM_LOCK.yaml](UPSTREAM_LOCK.yaml)：上游来源和锁定 SHA；
- [DEVELOPMENT_ENVIRONMENT.md](DEVELOPMENT_ENVIRONMENT.md)：环境部署和 HCU 运行方法。

当前代码的核心策略是：上层 API 和科学语义尽量继承，Torch reference 先定义可验证语义，
Triton/HIP 逐算子替换并以实测决定。当前 ops registry 只注册了 torch_reference；
triton、hip 名称已被识别但尚未作为该 slice 的可用后端注册。默认 Warp 路径仍保留，
因此“能导入”或“reference 通过”都不能写成完整生产后端已支持。

## 1. 开始前：定位工作范围

从项目根目录开始，先读规则和状态，再做只读检查：

~~~bash
git status --short --branch
git log -8 --oneline --decorate
rg -n "目标符号或功能名" packages/framework packages/ops packages tests probes
~~~

不要从 external/复制一份文件后直接改。先确认锁定上游源码、产品侧已有补丁和对应
测试；需要了解上游行为时，在 external/中查阅，修改只能进入 packages/。

一次开发任务应尽量只拥有一个清楚的边界，例如：

| 任务 | 主要代码位置 | 必要配套 |
|---|---|---|
| ops reference/dispatcher | packages/ops/nvalchemiops/ | ops tests、framework compatibility、probe |
| framework 后端接线 | packages/framework/nvalchemi/ | 上游行为测试、reference contract |
| 新 Triton kernel | ops 对应算子族目录 | capability gate、CPU/HCU 数值和性能 |
| 新 HIP kernel | ops 对应算子族目录及其构建入口 | gfx936 编译、stream/atomic/错误路径测试 |
| 新模型/动力学功能 | framework models/、dynamics/、hooks/ | Batch、梯度、生命周期和端到端测试 |
| 跨包行为 | tests/compatibility/ | 两包安装/import 和纵向链 |
| 设备/性能探针 | probes/ | reports/摘要与 artifacts/原始输出 |

不要顺手重排邻近代码、全库格式化或修改无关的 STATUS.md 历史段落；这些是最常见的
合并冲突来源。

## 2. 先写契约，再写实现

新增或移植的算子必须在代码、测试或特性契约中明确以下字段：

| 维度 | 必须回答的问题 |
|---|---|
| shape/索引 | 输入输出形状是什么？原子、体系、边和 padding 如何对应？batch_idx 与 batch_ptr 是否一致？ |
| dtype/layout | 支持 float32/float64 和哪些整数类型？矩阵/COO、连续性和 stride 有什么要求？ |
| 物理语义 | 单位、符号、cutoff、PBC、image shift、full/half neighbor、自相互作用如何定义？ |
| 空/异常 | 空 batch、空行、零邻居、重合点、非法 cell、容量不足、未支持参数如何处理？ |
| mutation/alias | 哪些输入原地写？输出是否拥有独立存储？是否允许与输入 alias？ |
| stream/device | 所有张量是否在同一设备？调用使用哪个 stream？是否需要同步？是否跨 rank？ |
| 顺序/确定性 | 是否承诺邻居排序、体系顺序、重复 pair 和 deterministic reduce？ |
| 梯度 | forward-only、一阶、二阶分别支持什么？拓扑不可微时连续坐标路径如何保留？ |
| 选择/回退 | None、auto、显式 backend 的行为是什么？失败是否明确而不是静默 fallback？ |

先用小而可解释的 CPU Torch reference 验证集合、解析公式和梯度；再把同一契约带到 HCU
和优化后端。reference 也不能把 CPU 作为默认回退：如果目标输入在 HCU，reference 应
尽量在 HCU Torch 上运行并在报告中写明后端。

## 3. 后端选择与分层

### 3.1 共同规则

ops 的 backend.py 提供 BackendName、BackendSelection、BackendUnavailableError 和选择记录；
torch_backend.py 负责将公共参数交给具体实现。新增 backend 时优先扩展这条集中路径，不要在
每个组件中各自维护一套互相矛盾的字符串判断。

选择后端不能只看 device.type == "cuda" 或设备名中是否含 cuda：海光 HIP Torch 也使用
torch.cuda 命名空间。至少结合构建信息、torch.version.hip、设备可用性、目标架构、输入规模、
dtype、梯度等级和实际 benchmark。

当前约定：

- backend=None：保持上游默认（通常是 Warp）；不要为了 HCU smoke 改掉它；
- backend="torch_reference"：显式选择正确性优先 reference；
- backend="triton" / "hip"：未注册时必须抛 BackendUnavailableError；实现和测试完成后才
  能登记；
- backend="auto"：只在能力过滤、后端注册和证据齐全时选择优化后端；当前 slice 只会选
  已注册的 reference，并记录原因；
- backend="warp"：NVIDIA 上游参考路径，不能被 HCU 修改偷偷替换。

### 3.2 Torch reference

Torch reference 是语义基线，不是“临时能跑”的简化版本。实现时：

- 保持 tensor 运算和目标设备；避免逐元素 .item()、.tolist()、Python append 写回，除非
  契约明确这是小输入 reference 的非性能路径，并记录代价；
- 保留 Batch 的体系边界和全局索引，禁止跨体系 pair；
- 对未支持的 half_list、switching、virial/stress、方法选择、容量等显式报错；不忽略参数；
- 先覆盖 float64 解析对照，再覆盖 float32/HCU；需要 force 时检查 F=-dE/dx、总力守恒和
  可能的二阶路径；
- 若通过 torch.library.custom_op 暴露，补 register_fake；自定义 backward 时补
  register_autograd，并用实际版本的 opcheck/gradcheck/gradgradcheck 验证；
- torch.compile 只能在 eager 正确后尝试。动态容量、输入校验和数据依赖分支可能产生 graph
  break，必须在报告中说明，不要为了 fullgraph 删除检查。

当前周期邻居 reference 的例子在 packages/ops/nvalchemiops/torch_reference.py：周期
full-list 使用 r_ij = r_i - r_j - shift @ cell，MATRIX/COO、批边界、padding 和 overflow
是契约的一部分。其 Tier 1 scatter 结果见 reports/g2-neighbor-tier1-scatter-reference.md，
但 no-PBC 完整优化、cell-list、half-list 和生产性能仍是独立任务。

### 3.3 Triton

Triton 适合有清晰 tile、规则索引、融合和归约的计算；不要把所有 Warp kernel 逐行翻译为
Triton。开发顺序为：

1. 先有 Torch reference 和输入规模/边长分布的基线；
2. 在 probes/写最小 kernel（向量、索引、归约或目标算子片段），确认海光 Triton 编译器
   能生成并执行目标 gfx936；
3. 将 kernel 放入对应 ops 算子族，保持一个真源，不在根目录复制一份实现；
4. 通过 dispatcher 注册，并按 dtype、shape、梯度和设备做 capability gate；
5. 对比集合/能量/力/梯度和错误路径，再测 cold/JIT、warm steady、kernel 和端到端成本；
6. 没有证据时不让 auto 选择它，也不要以 reference 作为无记录的运行时 fallback。

Triton 的基础 vector-add 和双卡通信探针已经通过，但这只证明工具链和原语可用，不证明
neighbor、LJ、MLIP 或 DomainParallel 的生产 kernel 已完成。

### 3.4 HIP

HIP 适合 cell-list、复杂邻居/迁移、原子操作、通信打包和需要显式线程控制的路径，但应
用 profile 证明收益。开发时特别检查：

- 编译目标固定为当前目标 gfx936；不要把过程试探的 gfx928 当成目标证据；
- 输入输出的 device pointer、dtype、stride、stream 和生命周期；
- 越界、空输入、容量溢出、重复 pair、atomic race 和多 rank ownership；
- Torch/HIP 互操作和构建隔离；不要全局替换 cuda 字符串，也不要强行改上游 Python API；
- kernel 有了之后仍需 Torch reference 数值、HCU smoke、梯度/归约和性能回归；
- HIP 源码放在 ops 对应的单一构建入口中，避免同一个算法同时存在无法同步的 C++/HIP 副本。

HIP 适配不等于自动获得 DomainParallel 支持。ghost、halo、ownership、通信归约和多层
消息传递必须单独做双卡/多卡测试。

## 4. 新算子或算子移植的标准流程

以 neighbors 或 LJ 为例，按下面顺序推进；其它算子族也遵循同样的纵向闭环。

### 4.1 盘点上游来源

从 UPSTREAM_LOCK.yaml 的 framework/ops SHA 出发，定位：

- 上游实现文件、公共符号、调用者和 docstring；
- 上游测试、fixture、示例和配置；
- Warp/JAX/第三方依赖的实际用途；
- framework 到 ops 的导入关系和默认参数。

在 FEATURE_COMPATIBILITY.yaml 登记稳定 feature ID。导入文件的任何修改同时更新
UPSTREAM.md 的本地补丁清单，写清默认路径影响和 upstream candidate；不要只写“适配海光”。

### 4.2 写 reference 和 contract tests

先实现最小可验证 slice，测试至少包括：

- 解析/独立 FP64 结果和边集合；
- full/half（若契约支持）、self、PBC shift、三斜胞、空输入和异构 Batch；
- capacity/overflow、非法参数、未知 backend 和不支持的组合；
- output shape/dtype/layout、global index、batch_ptr、alias/mutation；
- energy-force 一致性、总力守恒，以及需要的 grad/gradgrad；
- CPU 与 HCU 同一输入，先小规模再真实数据。

ops 原测试位于 packages/ops/test/，framework 行为测试位于 packages/framework/test/。
跨包语义放 tests/compatibility/；探针只负责真实设备、模型、时序和性能证据，不替代单元
测试。不要把测试整体 skip/xfail 来掩盖缺实现；必要的 strict xfail 必须关联原因、解除条件
和跟踪条目。

### 4.3 接 dispatcher 和 framework

ops 层建议保持以下关系：

~~~text
public operation
    -> backend selection / capability gate
    -> selected implementation (torch_reference / triton / hip / warp)
    -> common output contract
~~~

framework 层再负责 Batch、model wrapper、Hook、Dynamics 和用户 API。后端参数必须沿调用链
传递，不能在中间重新选择或丢弃；默认参数保持上游语义。对于没有 Warp 的 HCU reference
导入，使用按需导入隔离 NVIDIA-only 模块；不能删掉高级类名或改变默认 Warp 构造。

LJ reference 的实际接线可参阅 nvalchemi.models._ops.lj、nvalchemi.models.lj 和
nvalchemi.neighbors：邻居拓扑、pair geometry、energy/force、switching、virial/stress
是不同特性，不能因 energy/force 通过就一起标记完成。

### 4.4 真实设备和性能证据

每个后端至少分开记录：

- cold/JIT 首次调用；
- warmup 后 steady-state；
- 单 kernel 与端到端（包含邻居、模型、Hook/同步）；
- 同形 batch 与异构 batch；
- 设备、架构、可见卡、线程、dtype、输入尺寸、边数和共享负载。

统一使用 timeout、显式同步和固定 OMP_NUM_THREADS。原始输出进 artifacts/，摘要进
reports/；报告写命令、源码 SHA、退出码、结果、失败和证据边界。shared HCU 数字只能支撑
本轮相对观察，不能不加说明写成发布门槛。

## 5. 新功能的接入方法

### 5.1 上层功能

新 dynamics、model、Hook、storage 或 training 能力先保持上游公共 API：参数名、默认值、
返回字段、状态字段和生命周期不随移植随意改变。新增 backend 是显式扩展，不把 HCU 结果
硬编码为默认值。

对 FusedStage、inflight、Hooks 和 checkpoint 尤其注意：

- 不同体系可以处于不同阶段，stage 状态和 system ID 不能按 batch 粗暴覆盖；
- 已收敛体系的共享前向、临时 pre-update、输出恢复和一次性 sink 写出必须可追踪；
- Hook 的 stage、频率、顺序、异常传播和资源释放属于 API 行为；
- 轨迹快照不是 checkpoint/restart，需分别验证模型/优化器/模拟状态、随机数、体系 ID、
  sampler 和阶段状态；
- DistributedPipeline、DDP 和 DomainParallel 是三种不同语义，不能互相代替验收。

### 5.2 兼容性条目

每个新功能在 YAML 中分别填写：

- category：A（上层语义）、B（内部后端替换）或 C（明确延期/实验）；
- status：planned、implemented、verified、blocked 或 deferred；
- source、contract、dependencies、backends、gradient level；
- upstream tests、DCU tests、environment、evidence、limitations、release gate。

分类和验证状态是两条轴。没有真实 HCU/CPU/多卡证据时不能写 verified；只有 import 成功
不能写功能通过；窄 reference slice 不能扩大成完整 feature。

## 6. 测试分层与常用命令

先加载环境，然后按改动范围运行最小集合：

~~~bash
source scripts/activate_hygon_env.sh project

# ops：Torch reference 单元测试
PYTHONPATH=packages/ops .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_reference_backend.py

# framework：后端接线和 model/Hooks contract
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py \
  packages/framework/test/models/test_lj_torch_reference.py \
  packages/framework/test/compatibility/test_public_dynamics_reference.py

# 已登记的上游 reference 子集（通过环境变量选择，不改变产品默认）
NVALCHEMI_TEST_BACKEND=torch_reference \
  PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_ops.py

# 静态检查/编译（工具已安装时）
.venv/bin/python -m compileall -q packages/framework/nvalchemi packages/ops/nvalchemiops
.venv/bin/ruff check packages/framework/nvalchemi packages/ops/nvalchemiops
git diff --check
~~~

上游测试文件中默认未设置 NVALCHEMI_TEST_BACKEND 时仍走 Warp/上游默认；显式 reference
测试变量是本地测试载体，不得带进产品默认配置。framework 与 ops 测试应分进程运行，防止
顶层 test 包冲突。

涉及梯度时追加与算子契约匹配的 gradcheck/gradgradcheck；涉及 custom op 时按当前
Torch 版本检查 fake/meta、opcheck 和 compile。不要把一个 gradcheck 当成力损失混合二阶
路径已经通过。

## 7. 探针和报告写法

新增探针放 probes/，保持独立、可重复、不会修改产品状态。探针入口至少包含设备、输入
路径、随机种子、dtype、warmup/steady、超时和结果输出参数；不依赖交互式 heredoc，不在脚本
中隐藏本机模型/数据路径。

报告最小模板：

~~~text
# 范围和 feature ID
日期 / 代码 SHA / 上游 framework 与 ops SHA
环境 / DTK / 设备 / 架构 / 可见卡
可重跑命令
输入、dtype、seed、同步和计时口径
CPU 结果、HCU 结果、退出码
数值/梯度/性能结果
未验证项、失败模式、资源限制
原始 artifacts 路径
~~~

实现、测试、数值结果、性能结果和未验证假设分开写。尤其不要把“编译成功”“import
成功”“一个模型 smoke 通过”合并为“后端支持”。

## 8. Git 协作与降低冲突

### 8.1 分支和提交

每位开发者从明确的共享基线（当前为 develop 或 g2-development）创建自己的
`<开发者>/<类型>-<主题>` 分支。推荐示例为 `alice/feature-triton-neighbor`、
`bob/fix-pbc-shifts`、`carol/docs-onboarding`。推荐提交粒度：

~~~text
docs: update development handoff
test(ops): add periodic neighbor contract
feat(ops): register triton neighbor backend
feat(framework): pass backend through neighbor hook
perf(ops): vectorize periodic scatter
~~~

每个提交只解决一个可审查问题，使用 git commit -s 保留 DCO。典型顺序是：

1. 上游审计/契约和测试；
2. Torch reference 或框架接线；
3. Triton/HIP 实现；
4. HCU 探针、报告和性能记录；
5. 文档、兼容矩阵和状态交接。

共享基线分支只接受审核后的合并，不直接在其上开发。用户明确要求“当前状态检查点”时才
一次性提交既有工作树；普通开发不要用 git add -A。

### 8.2 开工、暂存和交接

~~~bash
git status --short --branch
git diff -- path/to/file
git add path/to/your/file path/to/your/test
git diff --cached --check
git diff --cached --stat
git commit -s -m "feat(ops): ..."
git status --short --branch
~~~

工作树已有他人改动时，按路径暂存自己的文件，不要运行 git reset、git checkout -- 或
用脚本覆盖整文件。发现目标文件同时被别人修改，先保存事实和 diff，再与负责人协调；不要
猜测性重写。格式化只针对自己修改的文件，优先使用项目既有版本的 Ruff/clang-format。

### 8.3 同步基线和处理冲突

与远端同步需要用户/仓库权限明确后执行；不要自动 push。常规流程是：

~~~bash
git fetch local-origin
git rebase local-origin/<目标基线>
# 解决后逐个检查：
git status
git diff --check
git add <已解决文件>
git rebase --continue
~~~

上游 subtree 同步必须使用单独分支，先更新 lock、依赖和补丁清单，再做兼容回归。不要把
external/当产品源，不要 squash 掉需要追踪的上游历史。若冲突涉及同一个上游导入文件，
优先保留上游变化和本地补丁的明确边界，重新核对 UPSTREAM.md，不能简单以本地版本覆盖。

降低日常冲突的约定：

- 不全库重排 import、重命名或格式化；
- 不把性能数据、状态历史和新功能代码混在一个大提交；
- STATUS.md 采用追加式交接，避免重写历史段落；
- FEATURE_COMPATIBILITY.yaml 只修改目标条目或在末尾追加，不做无关格式化；
- UPSTREAM.md 的补丁表按 ID 追加，ID 不复用；
- 每个后端/算子指定单一 owner 和路径，另一个人修改前先在提交或任务中说明范围；
- 共享提交一旦被使用不再 amend/rebase 改写，修复另建提交。

### 8.4 合并前清单

- [ ] git status 中只有本任务文件，未意外加入模型、轨迹、缓存、.venv 或 external/。
- [ ] git diff --check 通过，修改没有无关格式化。
- [ ] 上游文件、符号、SHA 和本地补丁已登记。
- [ ] CPU/reference 单测通过；有资源时 HCU smoke、数值和性能证据已运行。
- [ ] custom op 的 fake/autograd/compile 边界已按支持范围验证。
- [ ] Batch、体系 ID、索引、PBC、full/half、overflow、空输入和梯度没有被跳过。
- [ ] FEATURE_COMPATIBILITY.yaml 的分类、status、evidence 和 limitation 一致。
- [ ] STATUS.md 写清下一步，且交接中分别列出实现、测试、数值和未验证项。
- [ ] 提交信息清楚并带 Signed-off-by；没有自动推送。
