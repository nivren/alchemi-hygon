# nvalchemi-toolkit 海光 DCU 移植：技术路线、开发计划与工程规范

> 调研基线：2026-09-05。本文以 NVIDIA `nvalchemi-toolkit` 主线（当前 0.2 系列）和 `nvalchemi-toolkit-ops` 主线（当前 0.4 系列）的公开源码、文档为依据。海光卡型、DTK、PyTorch 和 Triton 必须按实际交付环境重新锁定；文中的阶段工期按 6–8 名核心研发估算。

## 1. 结论先行

1. **上层架构值得继承，GPU 后端不应照搬。** 应保留 `AtomicData/Batch → Model Wrapper → Dynamics/Training → Hooks → Distributed` 的分层、`toolkit` 与 `ops` 的边界、统一模型输出约定和透明的 `DomainParallel` 用户体验；不要把 Warp、PhysicsNeMo、CUDA extras 等 NVIDIA 绑定继续扩散到新工程。
2. **PyTorch 代码大部分可复用，但“可 import”不等于“可生产”。** 数据、配置、训练循环、Hook、模型适配器中的纯 PyTorch/Python 代码通常可直接迁移；MACE/UMA/AIMNet2 的依赖图、`torch.compile`、自定义算子、分布式通信和数值精度必须逐项验收。
3. **不建议现在把生产路线押在 ROCm Warp 上。** NVIDIA Warp 上游公开支持 CPU/CUDA；AMD 已有活跃的 ROCm Warp 移植，但其公开要求 ROCm 7.x，当前列出的支持卡仅 gfx942/MI325X。公开海光部署资料常见的是 DTK 25.04.2、rock/ROCm 6.3 系列。这条路线值得做短期技术验证，但不宜成为唯一依赖。
4. **推荐“PyTorch + Triton + HIP”三层后端。** PyTorch 是语义基线和自动微分/分布式基座；Triton 用于规则的块化逐元素、归约、segment、mesh 插值与融合；HIP C++ 用于邻居表、空间分桶、粒子迁移、动态循环、复杂原子操作以及需要精细控制 wavefront/共享内存的关键路径；FFT、随机数、通信优先调用 DTK 随附库，而不是自研。
5. **不用 Warp 不会必然损失性能。** Warp 是生产力、JIT、互操作和自动微分工具，不是某种不可替代的硬件加速。DCU 上原生 HIP 的性能上限通常最高；真正风险是移植初期缺少自动生成的反向核、动态 kernel 组合能力和成熟调优。Triton 对规则块计算很合适，但不能假设它对邻居表等不规则工作负载总是优于 HIP。
6. **首个用户版本不要追求功能全量。** 第一版应先闭环“批量体系 → MLIP 推理 → NVE/NVT MD 或 FIRE 弛豫 → 轨迹/检查点”，并同时提供 DDP 训练。空间域分解要在项目第一个月做可行性原型，但在单卡数值和算子稳定后再进入正式发布。

## 2. 建议的产品边界

产品应同时覆盖三种不同的并行问题，不能用一个“多卡”概念混在一起：

- **多任务吞吐**：大量互不相关的小/中体系，做动态批处理、长度分桶、in-flight 补位；核心指标是 trajectories/hour 与 GPU 占用率。
- **训练扩展**：MLIP 数据并行（DDP/FSDP），核心指标是 samples/s、扩展效率、断点重启和确定性。
- **单一大体系扩展**：空间域分解、halo/ghost、粒子迁移、跨 rank 归约；核心指标是 atoms·steps/s、单步时延、强/弱扩展效率。

建议首年只承诺 PyTorch 前端。JAX 保留 API 插槽，但不要在资源有限时同步维护两套前端。

## 3. 目标架构

```text
用户 API / CLI / Python 配置
        │
        ├── Data: AtomicData, Batch, Zarr, transforms, samplers
        ├── Models: BaseModel + MACE/AIMNet2/UMA/自定义 MLIP 适配器
        ├── Workflows: MD, relaxation, training, evaluation, checkpoint
        ├── Hooks: logging, safety, sampling, profiling, convergence
        └── Distributed: batch sharding, DDP/FSDP, domain decomposition
                         │
                 稳定的 Ops 语义层
          schema / shape / dtype / unit / autograd contract
                         │
            ┌────────────┼────────────┐
            │            │            │
       Torch reference  Triton       HIP C++
       正确性与回退      规则融合核    不规则/极致性能核
            │            │            │
            └────── DTK runtime/libraries ──────┘
              hipFFT/hipRAND/rocPRIM/RCCL/ROCTX
```

### 3.1 仓库组织

建议前 12–18 个月采用 **monorepo、两个可独立发布的 Python 包**：

```text
repo/
  packages/framework/        # 上层框架，只依赖公开 ops API
  packages/ops/              # schema + torch/triton/hip 实现
  cpp/                       # HIP/C++ 扩展、CMake
  tests/{unit,integration,numerics,distributed}/
  benchmarks/{micro,workflow,distributed}/
  examples/
  docs/
  containers/
  adr/                       # 架构决策记录
```

这样既保留原项目正确的包边界，又避免早期两个仓库联调、版本同步和 CI 重复成本。稳定后若维护团队和发布节奏明显分化，再拆仓。

### 3.2 后端接口

每个算子先定义与硬件无关的契约，再挂实现：

- 输入/输出 shape、dtype、layout、单位、PBC 语义、空体系行为；
- 是否支持 batch、动态 shape、`torch.compile`、autograd、二阶梯度、确定性；
- mutation/aliasing 契约；
- capability registry，例如 `supports_fp64_atomic`、`supports_dynamic_shape`、`supports_gradgrad`；
- 实现优先级 `hip > triton > torch_reference`，用户可强制选择，失败必须给出明确原因而不是静默换精度。

不要用字符串中是否含 `cuda` 判断 NVIDIA。海光/ROCm PyTorch 仍复用 `torch.cuda` 和 `torch.device("cuda")` 接口；应结合 `torch.version.hip`、包构建元数据和运行时 capability 探测。

## 4. Warp 的实际作用与替代策略

### 4.1 Warp 在原项目中的作用

`nvalchemi-toolkit-ops` 把 Warp 当成低层领域 DSL 和运行时，承担：

- Python 中编写并 JIT 编译 CPU/CUDA kernel；
- 一线程一原子/一候选 pair、动态循环、gather/scatter、atomic 等不规则计算；
- 向 PyTorch/JAX 做零拷贝互操作和框架绑定；
- 从 kernel 源码变换生成 adjoint，并通过 Tape 或框架自定义 autograd 接回训练图；
- 模板化/动态生成 pair 函数和 dtype 特化 kernel；
- 在邻居表、MD 积分器、FIRE/FIRE2、D3、DSF/Ewald/PME、B-spline、球谐/GTO、segment ops 中复用统一的数据类型和 launcher。

Warp 确实比直接写 HIP/CUDA 更容易表达不规则 kernel，但“容易表达”不等于自动高效：线程分歧、非合并访存、原子冲突、邻居数长尾仍需算法和布局优化。

### 4.2 自动微分边界

Warp 默认可生成 forward/backward kernel，并支持自定义 gradient、Jacobian 与有限差分检查；但有限制：某些 in-place 更新、动态循环中间值、重复写入、原子归约和多次状态复用需要特殊处理。原 `-ops` 也没有把所有输出都交给 Warp 自动微分：部分直接力/virial 是分析公式，segment ops 明确提供一阶/二阶 backward，邻居拓扑通常视为不可微的离散量。

新后端应采用以下梯度政策：

1. 纯 Torch 参考实现使用原生 autograd；
2. Triton/HIP 实现注册正式的 `torch.library` custom op、FakeTensor/meta 和 `register_autograd`；
3. backward 可以由 Torch 组合实现，性能不足时再写 Triton/HIP backward；
4. 训练能量+力时需要对模型参数通过力损失求导，因此关键算子必须明确支持 **二阶梯度**，并运行 `gradgradcheck`；
5. 邻居索引、分桶、排序等拓扑量 `detach` 并声明不可微；距离、位移、能量对 position/cell/charge 的连续路径保持可微；
6. 分开发布 `inference_only`、`first_order`、`second_order` 能力，不允许缺梯度时静默返回零。

### 4.3 Triton 与 HIP 的职责边界

| 工作负载 | 首选 | 原因 |
|---|---|---|
| elementwise、规则 reduction、segment sum、固定宽 neighbor matrix 消费 | Triton | 易融合、迭代快、可和 `torch.compile` 组合 |
| B-spline spread/gather、规则 mesh stencil | Triton 起步，HIP 兜底 | 块化清晰，但 scatter 原子冲突需实测 |
| naive O(N²) 小体系 pair kernel | Triton 或 HIP 双实现 | 规则 tile 可发挥 Triton；需要按体系大小自动调度 |
| cell list 构建、排序、prefix-sum、动态邻居输出 | HIP + rocPRIM/hipCUB | 动态长度、原子和 scan/sort 更适合底层控制 |
| cluster-pair tiled neighbor | HIP | 强依赖 wavefront、共享内存和目标架构细节 |
| Velocity Verlet/Langevin/FIRE 的逐原子更新 | Torch/Triton | 算法简单，先保证组合与编译；性能不足再融合 |
| Ewald real-space、DSF、D3 pair 部分 | HIP/Triton 双实现 | 取决于邻居布局和特殊函数吞吐 |
| PME FFT | Torch FFT/hipFFT + Triton/HIP spread/gather | FFT 必须复用成熟库 |
| halo pack/unpack、粒子迁移 | HIP + PyTorch distributed/RCCL | 不规则索引与通信重叠需要精细控制 |

结论：**Triton 不是 HIP 的替代物，而是减少 HIP 数量的高生产率层。** 目标不是统一语言，而是统一算子契约、测试和调度。

### 4.4 ROCm Warp 技术预研

安排 2–3 人、2–4 周的 time-boxed spike：

- 在目标 DCU/DTK 上编译 ROCm Warp；
- 跑最小 kernel、PyTorch 零拷贝、atomic、动态 loop、Tape backward、gradcheck；
- 跑原项目的 naive neighbor、cell-list、segment sum、Velocity Verlet 四个代表算子；
- 记录需要修改的 LLVM target、wavefront size、runtime API、图捕获、设备识别和 HSACO 生成点；
- 只有在“可维护补丁量、二阶梯度、稳定 CI、性能”均过闸后，才允许把它作为可选后端。

无论 spike 成败，都不能让上层 API 再直接依赖 `warp.Array`；Warp 必须被封装在 backend 内。

## 5. 模块继承与迁移矩阵

### 5.1 `nvalchemi-toolkit` 上层框架

| 模块 | 决策 | 主要工作 | 优先级 |
|---|---|---|---|
| `_typing.py`、序列化、optional dependency | 直接继承思想并清理 vendor 名称 | 固化 Tensor shape/单位/输出协议；可选依赖错误要清晰 | P0 |
| `data/AtomicData`、`Batch` | 高比例复用 | 保持 batch/ptr/index 不变量；设备迁移、空 batch、异构批次测试 | P0 |
| `data/datapipes`、transforms、Zarr/level storage | 复用，解除 PhysicsNeMo 强依赖 | 建立 PyTorch DataLoader/DataPipe 适配，GPU prefetch 与流同步 | P0/P1 |
| `models/base`、pipeline、neighbor filter | 继承接口 | 定义 energy/forces/stress/virial 及 cutoff、distribution spec | P0 |
| Demo/LJ wrapper | 完整保留为 oracle | 用于所有单卡、多卡和梯度的最小端到端验证 | P0 |
| MACE wrapper | 首个真实 MLIP | 验证 e3nn、cuequivariance、scatter、compile；提供纯 PyTorch 回退 | P0 |
| AIMNet2/UMA | 按用户需求逐个适配 | 依赖版本相互冲突，使用独立 extra/容器 | P1/P2 |
| DFTD3/Ewald/PME model wrapper | 保留 API，重接新 ops | 必须先完成底层算子与能量导数契约 | P1/P2 |
| `dynamics/base`、stage 组合、sampler/sinks | 高比例复用 | 保持 `+`/`|` 组合、restart、in-flight batching | P0 |
| Integrators/optimizers | 保留数学与用户 API | 底层更新核重做；先 NVE/Langevin/FIRE，再 NHC/NPT/NPH/FIRE2 | P0/P1 |
| `hooks/` | 直接复用思想 | 邻居重建、安全、轨迹、收敛、日志；NVIDIA profiler 改为 torch profiler/ROCTX | P0 |
| `training/` | 高比例复用 | 去 PhysicsNeMo 强绑定；DDP、AMP、EMA、checkpoint、validation | P0/P1 |
| `distributed/` runtime | 以 PyTorch 重构 | 用 `torch.distributed`/DeviceMesh；仍可能使用 backend 名 `nccl`，底层由 RCCL 提供 | P0 |
| `distributed/` domain decomposition | 继承设计，不建议逐行搬运 | ShardedBatch、partition、halo、migration、output consolidation、global reductions 全部做 DCU 验证 | P1 |
| compile bridge / graph padding | 延后开启 | 首先 eager 正确；再做静态容量、FakeTensor、自定义 op 和编译缓存 | P1/P2 |

### 5.2 `nvalchemi-toolkit-ops`

| 模块/算子族 | 保留内容 | DCU 实现建议 | 梯度要求 | 优先级 |
|---|---|---|---|---|
| batch utils、output schema、types、dispatch | API/语义全保留 | Python + Torch；建立 capability/auto-dispatch | 按调用方 | P0 |
| `segment_ops` | sum/mean/max 等契约 | Torch reference → Triton → 必要时 HIP | 一阶+二阶 | P0 |
| naive / batch naive neighbor | matrix 与 COO、双 cutoff | Torch oracle；Triton tile 与 HIP 比赛 | 拓扑不可微，geometry 可微 | P0 |
| cell list / batch cell list | cell build/query/rebuild | HIP + rocPRIM scan/sort；pair/atom-centric 双策略 | 同上 | P0 |
| cluster tile neighbor | 算法和缓存/skin 思想 | DCU wavefront-aware HIP 重写 | 同上 | P2 |
| neighbor rebuild detection、matrix↔COO | 语义保留 | Torch/Triton | 通常不可微 | P0 |
| Velocity Verlet、Langevin | 数学保留 | Torch/Triton 融合 | 通常 forward-only | P0 |
| NHC、NPT/NPH、velocity rescale | 数学保留 | Torch/Triton，global reduction 接 distributed coordinator | 按需求 | P1 |
| FIRE/FIRE2 | 数学保留 | Torch/Triton；全局 dot/max 在多卡下归约 | 通常 forward-only | P0/P1 |
| LJ、switching | 全保留 | Torch oracle + Triton/HIP | position/cell 一阶；测试二阶 | P0 |
| DFT-D3(BJ) | 算法/参数/API | pair 主体 HIP/Triton；参数表保留来源和许可证 | energy→position/cell，按训练需求二阶 | P1 |
| DSF/direct Coulomb | 算法/API | HIP/Triton | energy 对 position/charge/cell | P1 |
| Ewald | real/k-space/self/slab 分组件 | real-space HIP；k-space Torch/Triton；先点电荷后 multipole | 一阶，训练路径二阶 | P1/P2 |
| PME | B-spline、FFT、Green function、gather | hipFFT/torch.fft + Triton/HIP spread/gather | position/charge/cell；二阶是高风险项 | P1/P2 |
| multipole Ewald/PME、quadrupole second backward | 保留接口但最后迁移 | HIP + 明确的自定义 backward/gradgrad | 二阶 | P2 |
| math: spline、球谐、GTO | 数学与测试向量保留 | Torch reference + Triton/HIP | 视上游调用 | P1 |
| `torch/` binding | 重构而非照搬 Warp helper | `torch.library` schema、fake/meta、autograd、autocast、vmap（需要时） | 明确登记 | P0 |
| `jax/` binding | 暂不交付 | 只保留设计位，PyTorch 稳定后再做 | 后续 | P3 |

## 6. 分阶段开发计划

### Phase 0：基线与技术闸门（第 0–4 周）

交付：架构决策、环境矩阵、可复现实验容器、上游 golden fixtures、风险清单。

- 固定上游 tag/commit，不跟随 main 漂移；建立 upstream merge log。
- 锁定目标卡型、每节点卡数、互联、CPU/NUMA、OS、驱动、DTK、PyTorch、Triton、RCCL 版本。
- 跑依赖探针：TensorDict/Zarr/Pydantic、MACE/e3nn、AIMNet2、UMA、`torch.compile`、FFT、DDP/RCCL。
- 完成 ROCm Warp spike 和 Triton/HIP 微基准：gather/scatter、atomic add、scan/sort、segment、pair tile。
- 在 NVIDIA/CPU 参考环境生成小型确定性 golden 数据：邻居表、能量、力、stress、1–100 步轨迹。
- 通过/退出标准：若 ROCm Warp 未通过目标 DCU 基本测试，则正式采用 Torch/Triton/HIP 路线，不再等待。

### Phase 1：可维护骨架与参考实现（第 3–8 周）

交付：两个可安装 wheel、统一 backend、CPU/Torch 单测、文档站和 CI。

- 迁移 AtomicData/Batch、模型输出协议、单位与符号约定、optional dependency。
- 建立 ops schema、Torch reference、device/capability registry。
- 建立 `torch.library` custom op 模板，包含 fake/meta、autograd、opcheck、gradcheck。
- 替换 PhysicsNeMo 强依赖：实现轻量 DistributedContext，PhysicsNeMo 仅作为 NVIDIA 可选适配器。
- CI：静态检查/许可证/文档/CPU test 每 PR；单 DCU smoke 每 PR 或合并队列。

### Phase 2：单卡 MVP（第 6–14 周）

交付 `0.1`：批量 MD/弛豫可供早期科学用户试用。

- P0 邻居表：naive、batch naive、cell list、rebuild/skin、matrix 和 COO。
- P0 动力学：NVE、Langevin NVT、FIRE；轨迹 sink、检查点、异常力/温度安全 Hook。
- P0 模型：LJ oracle、通用 BaseModel、至少一个真实 MLIP（优先 MACE，若依赖不通则选择在 DCU 上最先通过的模型）。
- 支持异构批次、动态补位、固定容量可选模式。
- 验收：能量/力/邻居正确，NVE 漂移达标，FIRE 收敛，1/10/100/1000 个体系批处理稳定。

### Phase 3：训练与高吞吐（第 10–20 周）

交付 `0.2`：可配置 MLIP 训练/微调和多卡 DDP。

- Zarr/in-memory dataset、分桶 sampler、prefetch、断点重启；
- energy/force/stress loss，EMA、validation、AMP；
- segment ops 二阶梯度、关键模型依赖适配；
- 1/2/4/8 卡 DDP/RCCL，单机先行，多机随后；
- `torch.compile` 只在 eager 数值通过后开启，建立 eager/compile 等价测试。

### Phase 4：长程物理与高级系综（第 16–30 周）

交付 `0.3`：面向带电、周期和晶胞优化场景。

- 先 DSF 与 D3，再 Ewald 点电荷，最后 PME；
- 先 energy-only 可微路径，再 forces/charge/cell/stress，再 force-loss 二阶梯度；
- NHC、velocity rescale、NPT/NPH、FIRE2 与 cell relaxation；
- slab correction 和 multipole 放在点电荷路径稳定之后。

### Phase 5：单一大体系多卡（第 4 周开始预研，第 20–40 周产品化）

交付 `0.4`：空间域分解的 LJ 和至少一个 MLIP。

- 第一个月即用 LJ 做 2 卡 halo exchange 原型，尽早暴露通信与 ownership 问题；
- 正式实现 spatial partition、owned/ghost、粒子迁移、neighbor rebuild、pack/unpack、output consolidation；
- per-atom 操作本地执行；energy、temperature、FIRE dot、barostat、convergence 做正确的全局归约；
- 通信与计算重叠，使用独立 stream，并按实际拓扑/NUMA 做 rank 绑定；
- MLIP 必须声明 `distribution_spec`：cutoff、消息传递层的 halo 需求、输出归约与 ghost correction；
- 验收：1/2/4/8 卡结果与单卡在规定容差内一致；无粒子丢失/重复；强、弱扩展报告公开。

### Phase 6：稳定性与生态（第 32–52 周）

交付 `1.0`：API 稳定、生产运维和兼容矩阵。

- 多节点、作业调度模板、故障诊断、checkpoint/resume；
- 性能自动回归、内存泄漏和长时间稳定性；
- ASE 适配与常用格式导入导出；
- 按用户需求增加第二、第三 MLIP；
- 评估 JAX 前端和 ROCm Warp 可选后端，不影响 PyTorch 主线。

## 7. 首发功能排序

### 必须首发（P0）

- AtomicData/Batch、单位和 energy/force/stress 约定；
- naive + cell-list neighbor、skin/rebuild、matrix/COO；
- BaseModel、LJ、一个真实 MLIP；
- NVE、Langevin NVT、FIRE；
- 批量调度、轨迹、日志、安全检查、checkpoint/restart；
- energy/force loss、单卡训练和 DDP；
- Torch reference 与 DCU 高性能实现的数值对照。

### 第二批（P1）

- D3、DSF、Ewald point charge、PME；
- stress/cell gradient、NHC、NPT/NPH、FIRE2；
- 单体系 domain decomposition；
- `torch.compile` 稳态路径和性能自动调度。

### 延后（P2/P3）

- multipole/quadrupole、PME dispersion、二阶 Hessian/phonon；
- JAX 全前端；
- 为追求 API 数量而复制所有实验功能。

## 8. 数值、性能与发布闸门

### 8.1 数值正确性

每个算子至少有四层 oracle：解析小例、CPU double reference、原 NVIDIA 实现 fixture、端到端物理不变量。

- neighbor：pair 集合完全一致；PBC shift、half/full、self interaction、triclinic、空体系、极端密度；
- energy/force/stress：finite difference、`gradcheck`、关键训练算子 `gradgradcheck`；
- MD：NVE 总能漂移、动量、温控分布、重启连续性；
- relaxation：最终 max force、能量单调性/允许的非单调边界、cell 约束；
- distributed：owned+ghost 守恒、跨边界 pair 不丢不重、rank-count 等价；
- 精度策略：FP64/FP32/BF16 分开给容差，禁止一个全局宽松阈值。

### 8.2 性能基准

固定 benchmark 矩阵：

- 单体系 N = 32、128、1k、10k、100k；
- 同尺寸 batch 与宽分布异构 batch；
- 真空、3D PBC、triclinic、高/低邻居密度；
- 1/2/4/8 DCU；训练和推理分开；
- 记录 cold/JIT 与 steady-state，不混为一个数字。

指标：kernel time、MD step time、atoms·steps/s、trajectories/hour、训练 samples/s、峰值显存、neighbor rebuild 占比、通信占比、强/弱扩展效率。PR 只要求“相对上一稳定基线无显著回退”（建议默认阈值 5%，噪声较大的分布式基准另定）；版本发布必须给完整报告，不承诺未经同卡实测的 NVIDIA/DCU 比值。

### 8.3 CI 分层

- PR-fast：格式、lint、类型、许可证、文档、CPU/Torch reference、schema/opcheck；
- PR-DCU：单 DCU 核心 smoke、数值 golden、gradcheck 小样；
- nightly：全量单卡、多个 dtype/shape、compile、压力和性能；
- weekly：2/4/8 卡、多节点、长 MD、性能趋势；
- release：干净容器安装、wheel、SBOM、许可证、示例、升级/回滚、所有受支持卡型。

## 9. 软件工程规范

- **API**：公开 API 遵循 SemVer；实验 API 放 `experimental`；所有 public symbol 明确 `__all__`；弃用至少跨一个 minor 版本。
- **ADR**：后端选择、数据布局、梯度契约、单位/符号、domain ownership、确定性各写一份 ADR。
- **代码风格**：Python 3.11+；Ruff format/check、类型检查、NumPy 风格 docstring；HIP 用 clang-format/clang-tidy。
- **提交与评审**：Conventional Commits、DCO、CODEOWNERS；physics、distributed、autograd、HIP 核心分别至少一名 owner；任何性能优化必须同时给正确性测试和基准。
- **依赖**：锁定 DTK/PyTorch/Triton 组合；模型 extras 分环境；容器镜像不可只写 `latest`；维护公开兼容矩阵。
- **可复现**：每次运行记录 git SHA、模型哈希、数据版本、卡型、驱动/DTK/torch、dtype、随机种子和 backend dispatch 结果。
- **错误处理**：不支持二阶梯度、FP64 atomic、compile、dynamic shape 时立即报 capability error；不允许静默 CPU fallback 或静默降精度。
- **上游同步**：保留 Apache-2.0 版权/NOTICE/SPDX；维护 `UPSTREAM.md`，记录移植来源、patch 和定期同步窗口；不要长期复制后不跟踪安全与数值修复。
- **文档**：每个功能必须包含概念、最小示例、单位/符号、性能适用区间、梯度等级和已知限制。

## 10. 团队与里程碑治理

建议 6–8 人最小团队：

- 2 名 GPU kernel/HIP/Triton；
- 2 名 PyTorch/MLIP/自动微分；
- 1–2 名 distributed/HPC；
- 1 名计算化学数值负责人；
- 1 名工程/CI/发布（可与其他角色兼任）。

每两周迭代，但用“能力闸门”而非功能数考核：

- G0 环境与四个代表 kernel；
- G1 单卡 LJ 完整闭环；
- G2 一个真实 MLIP 的批量 MD/弛豫；
- G3 energy+force 训练与 DDP；
- G4 Ewald/PME 数值与梯度；
- G5 2→4→8 卡 domain decomposition；
- G6 长时间稳定与 1.0。

## 11. 主要风险及应对

| 风险 | 早期信号 | 应对 |
|---|---|---|
| 海光 Torch/Triton/模型依赖版本互斥 | wheel 无法共存、compile graph break | 每个模型独立 extra/容器；只承诺测试矩阵 |
| 把 Triton 用在所有 kernel | cell-list/atomic 比 Torch 还慢 | HIP 兜底；以 benchmark 决策而非语言偏好 |
| 一阶结果对、force-loss 训练错 | gradcheck 过而 gradgradcheck 失败 | 梯度等级写入 capability；二阶测试为发布闸门 |
| 多卡“能跑”但重复/漏算 halo | rank 数变化结果漂移 | LJ 可解析 oracle、pair accounting、跨边界随机测试 |
| 上层继续耦合 NVIDIA 组件 | import 时拉 CUDA wheel/读取 NVIDIA env | PhysicsNeMo/Warp 均变可选 backend；依赖审计 |
| 过早追求全功能 | 许多算子半成品、无用户闭环 | 以 P0 场景发布，P1/P2 只有过数值/性能闸门才合并 |
| 只测平均体系 | 稀疏/高密度/邻居长尾 OOM | shape/density/property-based test 与 max-neighbor 压测 |

## 12. 前四周具体任务清单

1. 确认目标 DCU 卡型、节点拓扑和 DTK/PyTorch 镜像，输出 `compatibility.yaml`。
2. 固定两个 NVIDIA 上游 commit，导出 API 清单、测试清单和许可证清单。
3. 建立 monorepo、两个 package、CI、容器、ADR 模板、benchmark JSON schema。
4. 写 AtomicData/Batch 和 `ModelOutputs` 的兼容性测试。
5. 完成 Torch reference 的 LJ、naive neighbor、segment sum、Velocity Verlet。
6. 完成 Triton 与 HIP 的 naive neighbor/segment 微基准。
7. 完成 ROCm Warp spike，按预定退出标准做 go/no-go。
8. 完成 2 卡 LJ halo 原型以及 RCCL all-reduce/P2P 带宽基线。
9. 探测 MACE/AIMNet2/UMA，选出首个 MLIP；不要求三个同时支持。
10. 召开第一次架构评审，只批准已形成 benchmark 和维护成本证据的技术路线。

## 13. 调研依据

- NVIDIA ALCHEMI Toolkit repository: https://github.com/NVIDIA/nvalchemi-toolkit
- NVIDIA ALCHEMI Toolkit-Ops repository: https://github.com/NVIDIA/nvalchemi-toolkit-ops
- Toolkit distributed guide: https://github.com/NVIDIA/nvalchemi-toolkit/blob/main/docs/userguide/distributed.md
- Toolkit-Ops neighbor-list guide: https://github.com/NVIDIA/nvalchemi-toolkit-ops/blob/main/docs/userguide/components/neighborlist.md
- Toolkit-Ops electrostatics guide: https://nvidia.github.io/nvalchemi-toolkit-ops/main/userguide/components/electrostatics.html
- NVIDIA Warp differentiability: https://nvidia.github.io/warp/latest/user_guide/differentiability.html
- NVIDIA Warp compatibility/FAQ: https://nvidia.github.io/warp/latest/user_guide/faq.html
- ROCm Warp fork: https://github.com/AMD-Ecosystem/warp
- PyTorch HIP semantics: https://docs.pytorch.org/docs/main/notes/hip.html
- PyTorch custom operator and autograd registration: https://docs.pytorch.org/docs/main/library.html
- PyTorch user-defined Triton kernels: https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
- ROCm HIP programming model and libraries: https://rocm.docs.amd.com/projects/HIP/en/develop/understand/programming_model.html
- ROCm Triton installation/overview: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html
- ROCm RCCL usage: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- 海光开发者文档入口: https://developer.sourcefind.cn/document
- OpenCloudOS 海光 DCU/DTK 部署实践: https://docs.opencloudos.org/OC9/ai-deployment/GPU-optimization-practice/hygon-deployment/

