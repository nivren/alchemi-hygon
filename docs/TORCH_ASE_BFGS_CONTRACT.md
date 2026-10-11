# TORCH-BFGS-ASE-COMPAT：数值与接口合同

M4 扩展见[Batch合同](TORCH_ASE_BFGS_BATCH_CONTRACT.md)，覆盖同一数值合同的
异构分组更新及Framework生命周期。为拒绝实际出现的 info=0/有限错误谱，当前
单体系和Batch step均增加设备内分解有效性守卫；不修改谱或ASE更新公式。

2026-10-09；本地新增功能，**不是 NVIDIA BFGS 算子的移植**。实现独立表达
数学公式，不复制 ASE 实现代码；ASE 仅为测试依赖，不是 Ops 运行依赖。

## 来源与版本边界

- 产品基线：Framework 0.2.0 (`4dfe3723def34df3fadb245981081ccf8c94c257`)，
  Ops 0.4.1 (`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`)；不修改版本锁。
- 架构参考：Framework 0.3.0-rc (`0353b85f489b5f6863952a0c4441a0dd04e62ee8`)
  的 `nvalchemi/dynamics/optimizers/lbfgs.py`；Ops 0.5.0-rc
  (`5fffc0726b1e71bb8b68e10b5f87942064759f5b`) 的 `nvalchemiops/torch/lbfgs.py`
  L-BFGS 显式状态、prepare/step
  分层。未导入候选版本的 grouped/checkpoint 依赖，也未实现有限历史算法。
- 数学 Oracle：项目环境 `ase==3.29.0` 的公开 `ase.optimize.BFGS`
  (`initialize/prepare_step/update/determine_step`)；ASE 许可证 LGPL-2.1-or-later。

冻结源码 SHA256：

| 文件 | SHA256 |
|---|---|
| `ase/optimize/bfgs.py` | `12dd0cb6d324b9cc61c69b5f751b7482d55c18f34dfac85cb0585557fd6f9166` |
| `ase/_4/optimize/bfgs.py` | `a9326832e6fa760c98d0ae6ceeab0a4db400ee3e8fc43a34e93fb1a23e3c647e` |
| `ase/filters.py` | `9ca69004713e3595b9c784e6771756c73bad60329b5e7ef96b4e0c3575f83470` |
| `ase/_4/optimize/cellutil.py` | `e33bfc63cc9104eaa9917f58e7f39a0a33e4ba3dcaacb2b950e198dba271d276` |

## M1 适用域

单体系固定晶胞，`q, f: [N,3]`，`N>=1`，连续 strided FP64 Tensor，CPU 或
PyTorch CUDA 设备（HIP 构建也使用此标签）。输入力为负能量梯度；坐标 Å，
力 eV/Å，Hessian eV/Å²。不处理 PBC、邻居、约束、模型计算或收敛停止；调用方
提供一致的坐标/力。不得将多个体系拼接冒充单体系。

M1 不接 Framework、不支持 variable-cell/Batch/inflight/DomainParallel；M3 才加入
ASE UnitCellFilter `3N+9` 适配。空间 O((3N)²)、特征分解通常 O((3N)³)，无隐式
容量截断。空体系、FP32、非连续 layout、不同 device、requires_grad 输入报错。
功能仅为 eager、forward-only；不声称 autograd/compile 支持。

## 数学合同

默认 `alpha=70.0, maxstep=0.2`，均须有限且严格正。首步 `H=alpha*I`。
后续令 `s=q-q_prev, y=-(f-f_prev)`，当 `max(abs(s))<1e-7` 时保持 H；否则
以旧 `Hs` 计算 `H += yyᵀ/(sᵀy) - (Hs)(Hs)ᵀ/(sᵀHs)`。

`torch.linalg.eigh(H, UPLO='L')` 得到 λ,V；方向
`p=V ((fᵀV)/abs(λ))`，还原 `[N,3]`。令 `L=max_i norm(p_i)`，
若 `L>=maxstep` 则所有行统一乘 `maxstep/L`，否则乘 1。更新 `q+=p`，
保存**更新前**坐标及当前力。不引入阻尼、谱截断、曲率跳过或线搜索；允许负
特征值，用绝对值分母。零力给出零步长。

只在精确零分母、精确零特征值或非有限结果时报 `BFGSNumericalError`，不改变
坐标、H 或历史。ASE 可能对此传播 NaN/Inf；这是明确的失败边界，有限有效轨迹
仍逐步对照。不得通过近零阈值跳过合法更新。对 eigenvector 不做逐列比较；
符号和简并子空间不唯一，以 H、谱、方向及坐标为验收对象。

## API 与状态

入口：`nvalchemiops.bfgs_ase`，不经会初始化 Warp 的 `nvalchemiops.torch`。

- `bfgs_ase_prepare_state(num_scalar_dofs, *, alpha=70, maxstep=.2,
  dtype=torch.float64, device='cpu', system_id=None, backend='torch_reference',
  selection=None) -> ASEBFGSState`：自由度为正的 3 的倍数。
- `bfgs_ase_step(positions, forces, state, *, system_id=None,
  backend='torch_reference', selection=None) -> ASEBFGSStep`：原位更新 positions
  及 state，forces 保持不变。参数由 state 持有。返回独立诊断张量
  `eigenvalues/raw_step/applied_step/scale/max_step_length`。
- State 保存 H、历史坐标、历史力、initialized、step_count、alpha、maxstep、
  system_id；dof/dtype/device 从张量得到。单体系 ID 可为整数或 None；指定调用 ID
  须与状态匹配。未来 Batch 将持有多个独立 State，而非共享 H。
- `state_dict()` 返回克隆张量及 schema/algorithm 元数据；
  `ASEBFGSState.from_state_dict(data, device=None)` 验证并独立恢复。可通过
  `torch.save/torch.load` 保存；显式指定 device 才搬移，不降精度；不承诺 ASE JSON
  互读。非法快照不得生成可运行状态。

q、f、H、两份历史不得共享 storage（包括同 storage 的不相交视图）。返回值与
可变状态无 alias。输入和状态验证、候选数值计算均在提交原位更新前完成；不保证
硬件故障或外部并发修改时的事务性。调用方不得跨 stream 并发操作同一 State。

最小 Ops 调用（这里只用解析力；模型和停止规则在 M2 接入）：

```python
import torch
from nvalchemiops.bfgs_ase import (
    ASEBFGSState, bfgs_ase_prepare_state, bfgs_ase_step,
)

q = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float64, device="cpu")
state = bfgs_ase_prepare_state(q.numel(), device=q.device, system_id=7)
forces = -2.0 * q  # harmonic potential, eV/Å
diagnostics = bfgs_ase_step(q, forces, state, system_id=7)
restored = ASEBFGSState.from_state_dict(state.state_dict())
```

HCU 调用由调用者显式创建 HCU 输入和 state；无需换数学实现。

## Backend、设备与 stream

注册 `torch_reference.bfgs-ase-v1`，operation=`bfgs_ase_compat`；metadata 声明
Ops executor `nvalchemiops.bfgs_ase_reference` 两个 prepare/step entrypoint。
使用现有通用 executor；不新增公共 eigensolver 或通用 registry/state manager。

该新增 API 默认显式 `torch_reference`。`None/warp` 保留项目 legacy 意义，而
本功能没有 legacy BFGS，故明确报错；`auto` 未准入、其他未注册实现报错。
传入已解析 selection 时仍验证 operation/device/dtype/features，不重新选择
或绕过能力边界。CUDA 标签仅表示 Torch API 设备类型，不证明 NVIDIA 已验证。

所有数值张量运算在输入设备/current Torch stream 执行，不调用 `.cpu()`、NumPy、
ASE、Warp 或隐藏 dtype 转换。Python 元数据与有限性/分母检查允许标量同步；
不是无同步实现。数学库内部执行策略以实际 profiler 证据为准。不同设备/数学库
不承诺 bitwise 一致。reference 永久保留；优化后端以实际设备证据和同一 Oracle
为准，不改变算法。

## 验收与证据

先运行 ASE CPU FP64 逐步 Oracle、边界错误、restart、registry/executor/import
隔离测试；再运行既有 CPU reference gate。有空闲设备时执行限时单 HCU smoke，
覆盖 H/谱/方向/坐标、restart、非默认 stream 和特征分解残差。
解析良态案例 CPU 比较 `rtol=1e-10, atol=1e-12`；HCU
`rtol=1e-8, atol=1e-10`，分别记录物理量的实际误差。病态/简并案例采用相应
不变量验证，不通过全局放宽容差掩盖问题。

M1 实测：CPU gate 493 passed/3 skipped，BFGS 专项包含 68 passed；HCU4 三规模
各八步与 stream/restart/device guard 通过，GPU profiler 记录求解器 kernel 活动。
实际命令、退出码、数量和误差记录于 `reports/g2-ase-bfgs-m1.md`；完整功能
仍须 M2–M4，Ops 验收不能描述成 Framework/CSP 已支持。

## M2：已确认的 Framework 接入合同

2026-10-09 用户确认 M2。新增 `nvalchemi.dynamics.ASECompatibleBFGS`，复用
BaseDynamics 的模型输出、Hooks 和 run 循环；优化更新只委托 M1 Ops。默认
`alpha=70, maxstep=.2, fmax=.05, backend='torch_reference'`。BackendSelection
初始化时解析一次，prepare/step 复用。没有上游 legacy BFGS，None/warp/auto
等不支持请求继续明确失败。

- 输入为含一个非空体系的 Batch，positions/forces 为连续 FP64 `[N,3]`，CPU/HCU
  同设备，force/energy 输出槽按已有 Dynamics 要求存在。固定 cell、原子身份、
  system_id、自由度和设备；变胞、约束、sampler/refill、跨 rank 和异构 Batch
  本阶段未支持，拒绝会混淆历史的输入变化。
- 初次活动入口经 BEFORE_COMPUTE/AFTER_COMPUTE Hooks 求初始模型力，先检查
  收敛，再进行 pre_update（BFGS）→ compute（新位置模型力）→ post_update
  （无更新）的原有 Hooks 序列。默认使用 ConvergenceHook 的 custom_op，以
  `max(norm(forces, dim=-1)) < fmax` 实现 ASE 严格判据，不修改公共 `<=`。
  已收敛初始结构无优化步；普通 run 每个成功更新只增加一次模型求值。
  可显式指定自定义 convergence_hook，此时停止合同以用户的 hook 为准。
- 单体系 masked pre/post/update 的 False mask 和退出体系不运行 BFGS；完整
  Hessian、历史、初始化位和 Ops step_count 都冻结。BaseDynamics 的 step_count
  表示调度次数，Ops step_count 表示实际更新次数，二者分别记录。屏蔽入口不
  声称 M4 的混合体系/FusedStage 共享求值、动态补位已验收。
- 本地 state_dict/load_state_dict 保存数值状态、Framework 调度次数和 fmax；
  配套 Batch 坐标/身份/cell 单独保存。恢复时验证历史所属体系和固定几何元数据，
  重新求当前模型力，随后轨迹等于不中断运行。不是 ASE JSON restart 格式。
  两类计数各自非负，没有大小顺序约束（masked 更新可以没有独立调度次数）。
  default/custom 收敛模式必须相同，任意 custom hook 及模型由调用方另行重建。
- forward-only 优化状态；模型力照原有 compute 使用 autograd，训练/优化器
  可微/compile 不在合同中。禁止静默 CPU 回退、改 dtype 或修改 M1 数学。

M2 验收包括 CPU ASE 逐步坐标/Hessian/方向/停止对照、初始收敛、重启、Hooks、
屏蔽/退出状态及现有 CPU gate；真实 MACE 小体系 CPU/HCU 弛豫记录实际退出码、
模型求值/更新次数、误差、收敛和逐项计时。HCU eigensolver 独立评估
`n=138,199,200,201,276,552; B=1,4,16`，只读参考外部 rocSOLVER 工程并隔离构建
候选用于探针，不注册生产后端。记录 API/solver/numerical 三层结果及库内 host
工作、传输和同步证据；Batch eigensolver 探针不是 Batched BFGS 支持。若证实
瓶颈，提出独立 E1 阶段，待用户确认后再适配 eigensolver。

M2 实际发现：Torch n=138 的库内 LAPACKE 数学及三对角 D2H/特征向量 H2D 已
实测证实。因此 M1 的 device guard 只能证明 Python Torch 张量路径，没有证明
库内数学全在 HCU；M2 不具备全 HCU 求解验收。隔离候选使用 ctypes 直接调用当前
DTK C ABI（参考工程扩展为 Python3.10，与当前3.12不兼容），未复制/修改/安装
外部工程、未引入新产品依赖。E1 必须完成生产适配及误差/stream/失败合同。
解析势 CPU/HCU 重启位相等；真实 MACE CPU 位相等，HCU 同几何重算力有舍入差，
按独立物理量容差验证并记录实际误差。详见[M2报告](../reports/g2-ase-bfgs-m2.md)。

最小 Framework 用法（model 为已配置的 FP64 BaseModelMixin，batch 为单体系且
已有同设备 FP64 forces/energy 输出槽；有邻居需求的模型按上文配置 compute Hook）：

```python
from nvalchemi.dynamics import ASECompatibleBFGS

optimizer = ASECompatibleBFGS(model, hooks=hooks, n_steps=100)
optimizer.run(batch)
paired_batch = batch.clone()
snapshot = optimizer.state_dict()
resumed = ASECompatibleBFGS(model, hooks=restart_hooks, n_steps=100)
resumed.load_state_dict(snapshot)  # 显式 device= 可移动整个 snapshot，不改 dtype
resumed.run(paired_batch)
```


## E1：已确认的显式 HCU eigensolver

2026-10-09 用户确认并完成 E1 窄验收。新增 `backend="hip"`，登记
`hip.bfgs-ase-rocsolver-v1`，沿既有 executor metadata 接入；Framework 无需重写。
`torch_reference` 仍为默认/永久数值基线，None/warp/auto 继续明确失败。
两后端共用 Hessian/abs谱/步长/历史与原子 commit 数学，只有 eigh 被替换；
M2→E1 AST 对照证明除 solver callee 外数学 body 相同。

求解器私有于 BFGS，FP64 `[n,n]`/等维 `[B,n,n]`、L/U、完整升序谱/列向量；
不改变输入，当前 Torch stream/device guard，RAII 每线程/设备 handle，
API/solver info/非有限结果分层检查。Jacobi n<200，DC n>=200，按本机 trace
而非外部 n<=200 规则；无算法 fallback、夹紧或额外曲率条件。
初期仅准入已验证 HCU Torch/HIP/rocSOLVER 构建，详见
[HIP-BFGS-EIGH 合同](HIP_BFGS_EIGH_CONTRACT.md)。

CPU gate550 passed/3既有skip，最终registry18；原生HCU31项（包含真实info=1
非收敛、冻结和原子失败、schema/fake opcheck，不推导compile）；解析势3/9/36
自由度逐步ASE及位相等restart、真实MPA-0固定晶胞11步收敛。当前构建追踪
n138/199/200/201/552求解器与9/138/201/552自由度完整BFGS，均无受监测主机
LAPACK或矩阵回传，只有诊断标量及库索引元数据传输；Torch对照仍有host路径。
数值、命令、实际退出码与性能分别见[E1报告](../reports/g2-ase-bfgs-e1.md)。
等维solver batch已验收，不代表异构BFGS完成；M3 Filter/M4 Batch/refill/CSP
尚未交付，NVIDIA/其他构建/多卡/线程压力/compile/训练/生产性能未验收。

```python
# 仅在已准入 HCU 构建：初次显式 HIP 调用 lazy JIT，需要已安装 DTK/toolchain。
optimizer = ASECompatibleBFGS(model, hooks=hooks, backend="hip", n_steps=100)
# 恢复时调用方重建同样的 backend/model/Hooks，并配套恢复 Batch。
```

## M3 UnitCellFilter 扩展（2026-10-09）

用户已确认 M3。新增 Ops `ASEUnitCellFilter` 与 Framework
`ASECompatibleBFGSVariableCell`，完整 `3N+9` 广义坐标/力，全部 ASE Filter 参数及
默认值，参考晶胞与物理几何配对重启；详情见
[UnitCellFilter 合同](TORCH_ASE_UNITCELL_FILTER_CONTRACT.md)。仍为单体系 FP64，
复用同一个数值 body 和显式 Torch/HIP，原生 solver 文件不变。State prepare 新增
`coordinate_mode='cartesian'|'ase_unitcell'`；Cartesian 快照仍 schema1，无新增字段；
UnitCell 快照 schema2 明确 mode。两套 Framework 快照用 kind/键集合隔离。
默认收敛和 maxstep 使用 N+3 广义行范数，Batch energy 仍保存势能，
`get_enthalpy` 返回 E+pV。冻结/缓存失效/Hooks 均按单体系既有生命周期。

```python
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell

# model active_outputs 包括 energy/forces/stress；Batch 预置相应 FP64 输出槽和晶胞。
# hooks 包括 BEFORE_COMPUTE NeighborListHook(skin=0)；不加入 native AlignCellHook。
optimizer = ASECompatibleBFGSVariableCell(
    model, hooks=hooks, backend="hip", fmax=0.05,
    mask=None, cell_factor=None, hydrostatic_strain=False,
    constant_volume=False, orig_cell=None, scalar_pressure=0.0,
)
optimizer.run(batch, n_steps=100)
restart = optimizer.state_dict()  # 另存配套 batch/model，并重建 hooks。
```

数值证据、命令/退出码及性能边界见[M3 报告](../reports/g2-ase-bfgs-m3.md)。
M4 异构 Batch/inflight/CSP 仍为必交付；不因单体系 Filter 或 solver batch 验收
扩大完整特性支持状态。没有 NVIDIA 硬件验收，也不声明 compile/训练或生产性能。
