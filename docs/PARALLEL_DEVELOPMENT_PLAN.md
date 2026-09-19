# 并行开发工作计划

日期：2026-09-10

适用基线：当前产品 `develop`；B1 executor binding 已完成并已同步两个产品远端。

初始代码基线：`b1c930f`（本文件所在提交之后，以远端 `develop` 最新提交为准）。

本文是团队进入具体开发时的操作入口，目的是让开发者从同一个 `develop` 创建个人分支，
在明确的文件边界内并行工作。它不替代 `AGENTS.md`、
`docs/BACKEND_PLATFORM_PIPELINE_PLAN.md`、`docs/PROJECT_HANDOFF.md` 和
`docs/FEATURE_COMPATIBILITY.yaml`；这些文件发生冲突时，以项目级指令和权威技术计划为准。

## 1. 总体判断

B1 已将实现绑定从 framework/ops dispatcher 中的 implementation-ID 分支收敛为 catalog
entrypoint + generic executor。新增 reference implementation 时，主要工作可以留在自己的
executor、测试、probe 和 report 中，不应再修改通用 executor 或大面积修改 framework。

因此现在适合并行开展固定晶胞 Torch reference 功能。并行的边界不是“每个人随意认领一个
目录”，而是每条线拥有一个 operation contract 和一组明确文件；共享的 catalog 汇总、CPU gate、
`STATUS.md` 与兼容矩阵由集成负责人最后串行收口。

## 2. 立即启动的任务

### 第一批：T1 核心功能

这三项是当前权威计划已经批准的首批并行任务。

| ID | 功能 | 优先级 | framework 影响 | 交付边界 |
|---|---|---:|---:|---|
| `TORCH-NVT-LANGEVIN` | 固定晶胞 BAOAB Langevin NVT | P0 | 低—中 | Torch reference、局部 dispatcher/wrapper 接线、CPU/HCU contract |
| `TORCH-NEIGHBOR-PBC-CELL` | 周期 full-list cell-list neighbor reference | P0 | 很低 | ops reference、neighbor catalog、pair/image/capacity 回归 |
| `TORCH-NVT-NHC` | 固定晶胞 Nose-Hoover chain NVT | P1 | 中 | chain state、质量、Yoshida 更新、Batch/inflight 回归 |

三项都必须先完成 CPU contract，再申请 HCU 批验证；任何一项通过前都不能登记更宽的
strategy/capability。它们各自的第一版不包括 NPT/NPH、变胞、域分解、生产 Triton/HIP 或
M2 planner；FIRE2 变胞弛豫作为下面的独立扩展任务实施。

### T1 当前状态（2026-09-19）

- `TORCH-NVT-LANGEVIN` 已在 `develop` 合入 `1fafd7b`，完成固定晶胞 BAOAB Torch
  reference、registry/catalog、generic executor binding、公共 `NVTLangevin` 显式
  `torch_reference` 接线、CPU contract 和 HCU smoke；随后补充的短谐势统计 oracle 已在
  CPU/HCU 通过，当前分支另补了普通 Batch 的最小 integrator continuation state。
  这些交付仍是可审查的窄 reference slice。
- 该状态不等于完整 Langevin/NVT 支持。上游完整统计/行为套件、checkpoint/restart、
  `atom_ptr`、`_out`、inflight refill、分布式 ownership、torch.compile 和 Triton/HIP
  生产实现仍未完成；本轮 restart 只保存积分器续跑元数据，完整 checkpoint/restart
  仍应另建任务，不回填本分支。
- 本里程碑满足第 6 节的“完成一个可审查里程碑后暂停”条件；后续开发前重新确认优先级，
  不自动启动 M2、NPT/NPH、DomainParallel 或生产优化。

### 第二批：低耦合扩展功能

如果有额外开发者，可以同时开展下面三项。它们不改变 B1 的架构方向，但新增的
兼容条目应由集成负责人统一落盘。

| ID | 功能 | 优先级 | framework 影响 | 说明 |
|---|---|---:|---:|---|
| `TORCH-THERMOSTAT-UTILS` | Maxwell-Boltzmann 速度初始化、去 COM、velocity rescale | P1 | 低 | 补齐实际 NVT 初始化链路；需先冻结 seed、温度和 Batch 契约 |
| `TORCH-LJ-SWITCHING` | LJ cutoff switching 的能量、力和连续性 | P1 | 很低 | 主要是 ops reference 和 focused tests；framework 只取消明确拒绝 |
| `TORCH-FIRE2-VARIABLE-CELL` | FIRE2 原子/晶胞联合结构弛豫 | P1 | 低—中 | 先做 stress→cell-force reference，再做 coupled FIRE2 step；不接 NPT/NPH |
| `TORCH-BFGS-ASE-COMPAT` | ASE 3.29 无 line-search BFGS 数值兼容 | P1 | 中 | 先交付固定晶胞；变胞严格采用 ASE `UnitCellFilter` 语义，另设依赖里程碑 |

`LJ virial/stress` 暂不与 switching 分成两个同时修改同一实现文件的分支；它应在 switching
完成后单独排队，或由同一 owner 负责连续交付。

`TORCH-BFGS-ASE-COMPAT` 的“兼容”有严格含义：项目 `.venv` 当前锁定的 ASE 3.29
`ase.optimize.BFGS` 是数值 oracle，不是仅借用 BFGS 名称。首版不得把标准逆 Hessian
近似、L-BFGS、line search、曲率跳过/阻尼或不同的晶胞参数化伪称为该兼容路径。

## 3. 每条开发线的工作定义

### 3.1 `TORCH-NVT-LANGEVIN`

建议分支：`<developer>/feature-torch-nvt-langevin`

主要文件边界：

- 新增 `packages/framework/nvalchemi/_dynamics_reference/langevin.py`；
- 局部修改 `packages/framework/nvalchemi/dynamics/_ops/langevin.py` 和
  `dynamics/integrators/nvt_langevin.py`；
- 新增独立 compatibility test、HCU probe 和 report；
- 在 dynamics catalog 中登记 implementation，但不修改 generic executor。

最小验收：

- BAOAB `B-A-O-A-B` 顺序正确；`friction=0` 与 velocity Verlet 对照；
- 异构 Batch 的 `dt/kT/friction` 按 system 正确映射；
- 同一设备、输入和 seed 可复现，换 seed 通常产生不同轨迹；
- 普通 Batch 的 state/restart、空输入、非法参数和 mutation 语义明确；完整 checkpoint、
  inflight state 和分布式 ownership 不在首版范围；
- CPU 通过后再运行 HCU smoke；没有 HCU 证据只能标为 CPU verified。

不在本任务内：NPT/NPH、域分解、跨设备逐位随机一致、Triton/HIP 优化。

### 3.2 `TORCH-NEIGHBOR-PBC-CELL`

建议分支：`<developer>/feature-torch-neighbor-pbc-cell`

主要文件边界：

- `packages/ops/nvalchemiops/torch_reference_cell_list.py` 或同一 operation 的 reference 模块；
- `packages/ops/test/torch/` 下的 cell-list contract tests；
- `packages/ops/nvalchemiops/_backend_catalog/neighbors.py`；
- 专用 PBC probe 和 report；尽量不修改 framework runtime。

最小验收：

- 第一版只承诺 periodic full-list；覆盖正交/三斜晶胞、混合 PBC、image shift、Batch、空输入；
- 与 dense CPU FP64 对照 pair 集合、排序、距离、向量和有效邻居数；
- 分开检查 `num_neighbors`、active slice 和 capacity，不能用分配宽度代替有效邻居数；
- capacity overflow 必须明确失败或按契约扩容；不能截断、静默 fallback 或漏算；
- 新路径不得把 host `.cpu().tolist()` 或 Python 逐原子循环当作默认设备内实现。

不在本任务内：periodic half-list、target rows、动态 skin rebuild、自动 strategy 选择和生产
性能结论。

### 3.3 `TORCH-NVT-NHC`

建议分支：`<developer>/feature-torch-nvt-nhc`

主要文件边界：

- 新增 `packages/framework/nvalchemi/_dynamics_reference/nose_hoover.py`；
- 局部修改 NHC dispatcher 和 `nvt_nose_hoover.py`；
- 独立覆盖 chain state、Batch/inflight 和 extended-energy tests；
- 不接 NPT/NPH 的 barostat 代码。

最小验收：

- chain length、`Q`、`eta`、`eta_dot` 和 Yoshida order 的契约稳定；
- 异构 Batch 每个 system 独立使用温度、时间常数和自由度；
- state 初始化、补位、清理和 restart 不串状态；
- 固定晶胞短轨迹的温控统计、extended energy 和确定性回归可解释；
- 明确 NHC 与分布式全局 kinetic energy 的边界，不能提前宣称 DomainParallel 支持。

### 3.4 `TORCH-THERMOSTAT-UTILS`

建议分支：`<developer>/feature-torch-thermostat-utils`

先完成半天以内的契约审计，再实现：

- `initialize_velocities`：Maxwell-Boltzmann、seed、每 system 温度；
- `remove_com_motion`：质量加权的每 system 动量归零；
- `velocity_rescale`：按 system 目标温度缩放；
- 与现有 kinetics reference 统一单位，不重复实现另一套温度归约。

验收必须包括异构 Batch、不同质量、零/异常参数、COM 残差、温度统计和同设备 seed 行为。
它可以先作为独立 ops/reference operation 交付，公共 dynamics 初始化接线随后单独提交。

### 3.5 `TORCH-LJ-SWITCHING`

建议分支：`<developer>/feature-torch-lj-switching`

主要工作是把当前明确拒绝的非零 `switch_width` 变成可验证的 Torch reference：

- 从上游 switching 定义建立能量/力解析 oracle；
- 检查 switching 起点和 cutoff 处的连续性；
- 覆盖 full/half、PBC、Batch reduction、autograd force 和必要的混合二阶路径；
- 通过后才考虑 Triton/HIP，不凭 reference 结果写性能结论。

### 3.6 `TORCH-FIRE2-VARIABLE-CELL`

建议分支：`<developer>/feature-torch-fire2-variable-cell`

现有公共 `FIRE2VariableCell`、state、dispatcher ABI 和 `variable_cell` capability 解析已经存在；
本任务只补齐缺失的 Torch reference 与窄范围纵向验证。由一个 owner 按两个顺序里程碑交付：

1. `TORCH-CELL-STRESS-FORCE`：实现并验证
   `F_cell = -V * stress * inverse(cell).T`、`keep_aligned` 和 Batch 语义；
2. `TORCH-FIRE2-VARIABLE-CELL`：实现原子/晶胞 DOF 的共同归约、mix、clamp 和 affine update，
   再接现有公共 wrapper。

主要文件边界：

- `packages/framework/nvalchemi/_dynamics_reference/fire.py`；
- `packages/framework/nvalchemi/dynamics/_ops/fire.py`；
- `packages/framework/nvalchemi/dynamics/_ops/npt_nph.py` 中仅限 `stress_to_cell_force` 的
  reference binding，不迁移其他 NPT/NPH op；
- `packages/framework/nvalchemi/dynamics/optimizers/fire2.py` 的局部接线；
- 独立 compatibility test、variable-cell probe 和 report。

最小验收：

- 用 CPU FP64 有限应变或独立解析 oracle 验证 stress 符号、volume、cell inverse 和单位；
- 覆盖正交胞、三斜胞、`keep_aligned`、单体系、异构 Batch、空输入与奇异/非法 cell；
- 原子与晶胞状态的 `vf/vv/ff`、`maxstep`、`dt`、`alpha`、uphill/downhill 更新可独立对照；
- cell 改变后周期邻居重建不漏 pair；正确性阶段可使用现有 periodic dense reference，
  不依赖周期 cell-list 任务先完成；
- CPU contract 通过后，再用能够输出可信 stress 的解析模型或已验证模型运行 HCU smoke。

不在本任务内：普通 `FIREVariableCell`、NPT/NPH barostat、DomainParallel replicated cell state、
LJ virial/stress 的完整实现、生产 Triton/HIP 和性能结论。没有可信 stress/HCU 证据时只能登记
对应的 CPU reference slice。

### 3.7 `TORCH-BFGS-ASE-COMPAT`

建议分支：`<developer>/feature-torch-bfgs-ase-compat`

本任务为此前分子晶体弛豫工作提供可审计的 ASE 3.29 无 line-search BFGS 对齐路径。先冻结
固定晶胞的公开 API 和状态契约，再独立实现；变胞部分不与固定胞首版捆绑合入。

固定晶胞首版必须逐式对齐项目 `.venv` 的
`ase._4.optimize.bfgs.BFGSMethod` 与 `ase.optimize.BFGS`：

- 初始 Hessian 为 `alpha * I`（ASE 默认 `alpha=70`）；
- 以上一位置/梯度完成 Hessian BFGS 更新；
- 每步执行实对称 `eigh(H)`，以 `abs(eigenvalues)` 计算下降方向；
- 使用 ASE 的全局最大原子步长缩放，默认 `maxstep=0.2 Å`；
- 无 line search，且 ASE-compatible 模式不私自加入曲率拒绝、阻尼、reset 或 trust-region
  等改变轨迹的安全规则；
- checkpoint/restart 至少保存 Hessian、上一步位置、上一步力和 `maxstep` 的等价状态。

主要文件边界：

- 新增 `packages/framework/nvalchemi/_dynamics_reference/bfgs.py`；
- 局部新增 BFGS dispatcher/wrapper、catalog implementation 与独立 compatibility tests；
- 新增 ASE FP64 oracle test、HCU probe 和 report；
- 不修改 generic executor、公共 registry 或 M2 planner；不得把 BFGS 的内部线性代数需求提前
  扩张为公共通用 `eigh` operation。

固定晶胞最小验收：

- 用解析二次势和预设的 `position/gradient` 序列，逐步对照 ASE 3.29 的 Hessian、未裁剪方向、
  裁剪后位移与 restart；比较步向量而非符号不唯一的特征向量；
- CPU FP64 对照覆盖首次步、负特征值取绝对值、零位移 restart、`alpha`、`maxstep`、非法/非有限
  输入和收敛 Hook；明确首版只支持单个固定晶胞，异构 Batch、inflight 和 DomainParallel 必须
  显式拒绝，不能靠 padding 或共享 Hessian 静默伪支持；
- HCU 先以 `torch.linalg.eigh` 完成正确性 smoke 与基准；没有 HCU 证据只能登记 CPU verified。

变胞严格兼容是 `TORCH-BFGS-ASE-UNITCELL` 子里程碑，依赖固定胞验收和可信 stress：

- 采用 ASE `UnitCellFilter` 的原始胞 deformation-gradient 参数化，即 `3N+9` 自由度、
  `cell_factor`（默认原子数）、mask、hydrostatic/constant-volume/scalar-pressure 语义；
- 该路径与当前 `TORCH-FIRE2-VARIABLE-CELL`、`cell_filter` 的上三角 `3N+6` 原生表示不同。
  后者可继续发展，但不得标记为 ASE-compatible BFGS；
- 变胞 ASE 对齐不得在未确认此前工作实际使用的 filter（裸 `BFGS`、`UnitCellFilter`、
  `FrechetCellFilter` 或其他）前宣称轨迹一致。

`HIP-BFGS-EIGH` 不是首版承诺，而是有数据的条件任务：在目标晶体的 `D=3N` 与 `D=3N+9`、
FP64、预热后条件下分别记录 `eigh`、Hessian update、模型力计算与端到端每步时间。只有
`eigh` 确认为稳定瓶颈，才评估 HIP 小型实对称 eigensolver；先比较 Torch/hipSOLVER 路径，
不能仅因库名或设备名预设 Triton/HIP 更快。任何 HIP 路径仍必须以 ASE CPU FP64 步级 oracle
验证数学等价，且不能因性能回退到不同算法。

## 4. 共享文件与冲突控制

每条分支只直接拥有自己的 executor、测试、probe 和 report。以下文件是共享热点：

- `packages/ops/nvalchemiops/_backend_catalog/dynamics.py`；
- `packages/ops/nvalchemiops/_backend_catalog/__init__.py`；
- `docs/FEATURE_COMPATIBILITY.yaml`；
- `docs/STATUS.md`；
- `scripts/check_cpu_reference.sh`。

约定如下：

1. 不修改 `packages/ops/nvalchemiops/executor.py`、公共 registry 或 generic adapter；
2. catalog 只增加自己的 metadata，保持追加式修改，不重排既有登记顺序；
3. 每个开发者维护自己的 test/report；兼容矩阵和 STATUS 由集成负责人按实际证据串行更新；
4. CPU gate 脚本由集成负责人统一加入测试入口，避免多人同时改同一 shell 文件；
5. 任何跨任务 ABI、seed、state、planner 或 capability 变化，先写契约再改代码；
6. 不在个人分支合并其他人的分支，不改写共享历史。

## 5. 标准开工流程

从远端最新 `develop` 开始，不从旧的个人分支或候选指针开始：

```bash
git status --short --branch
git fetch --prune <product-remote> develop
git switch develop
git merge --ff-only <product-remote>/develop
git switch -c <developer>/<type>-<topic>
```

开工前在任务记录中写清：

- operation ID、上游 SHA/符号和本次不支持范围；
- 输入输出、shape/dtype/layout、单位、PBC/full-half、空输入和错误契约；
- mutation/alias、stream、确定性、梯度等级和 Batch/inflight 语义；
- 计划修改的文件、不会修改的共享文件和对应验收测试。

提交顺序保持可审查：

1. contract/test；
2. Torch reference；
3. dispatcher/catalog 接线；
4. HCU probe、report 和性能记录；
5. 兼容矩阵、STATUS 和交接文档。

每个提交使用 `git commit -s`；提交前至少运行：

```bash
git diff --check
scripts/check_cpu_reference.sh
```

有分配的 HCU 后，再运行对应的 HCU probe。未通过 HCU 时，报告必须写明 `HCU pending`，不能
把 CPU 或 import 成功写成 DCU verified。

## 6. 合入顺序与暂停点

建议由集成负责人按以下顺序审核：

1. `TORCH-NEIGHBOR-PBC-CELL` 与 `TORCH-NVT-LANGEVIN` 可并行开发，先完成各自 CPU contract；
2. `TORCH-NVT-NHC` 可并行开始，但其 framework state 接线应独立审查；
3. `TORCH-FIRE2-VARIABLE-CELL` 可由独立 owner 并行开始，但必须先完成
   `TORCH-CELL-STRESS-FORCE`，不能借机迁移 NPT/NPH；
4. `TORCH-BFGS-ASE-COMPAT` 的固定胞 contract/reference 可与上述任务并行；其变胞
   `UnitCellFilter` 里程碑必须等固定胞验收与可信 stress，且不阻塞 FIRE2 的原生 `3N+6` 路径；
5. 低耦合的 thermostat utilities / LJ switching 在额外人力充足时并行；
6. 每个 operation 单独通过 CPU gate，再安排 HCU 批验证和 review；
7. 至少两个 operation 形成多个已验证实现、或确实出现可复现实验策略需求后，才重新评估 M2。

本批暂停条件：三个 T1 或指定的低耦合任务完成一个可审查里程碑后，更新
`docs/STATUS.md`、对应 report 和兼容性条目，等待下一轮确认。期间不启动 M2、NPT/NPH、
DomainParallel 或生产 Triton/HIP kernel。

## 7. 这个协作方式为什么成立

它保留了 Git 最简单、最可恢复的协作模型：一个共享 `develop`、每人一个短分支、每项一个
operation、短提交、人工 review。文档只负责稳定边界，不承担 issue tracker、审批系统或实时
任务状态的职责。

如果团队规模很小，最简单的执行方式是只启动前三项：一人负责 neighbor，一人负责 Langevin，
一人负责 NHC/集成。若增加第四人，材料结构弛豫优先时可选择 BFGS fixed-cell ASE compatibility
或 FIRE2 variable-cell；前者不等待变胞 FIRE2，但不得提前宣称变胞 ASE 对齐。其余扩展项只有在
不争用共享文件且有人能完成完整 CPU/HCU 证据时才启动。
