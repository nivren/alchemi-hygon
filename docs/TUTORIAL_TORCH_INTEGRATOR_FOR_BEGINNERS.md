# 手把手教程：为海光 DCU 移植一个 Torch 积分器

本文面向第一次参与本项目、对 PyTorch 后端和原子模拟都不熟悉的开发者。

实际练习案例选择固定晶胞的 `NVTLangevin`（BAOAB Langevin，NVT 系综）。它是基础版本
之后的独立任务 `TORCH-NVT-LANGEVIN`，同时包含确定性更新、异构 Batch 和随机数三个典型
问题。已经完成的 VV 会作为对照模板。

本文的目标是：在 `backend="torch_reference"` 下使用设备内 Torch 张量完成固定晶胞
Langevin。它不是替换上游默认行为，也不是一次完成 Triton/HIP 优化。

## 0. 先知道我们要改什么

```text
NVTLangevin
    │  初始化时解析一次 BackendSelection
    ▼
framework dispatcher: dynamics/_ops/langevin.py
    │  提供 Langevin ABI 和局部 legacy handler
    ▼
generic executor binding
    ├── catalog entrypoint → _dynamics_reference/langevin.py
    └── legacy handler     → 原有 Warp custom op
```

Torch reference 是用 Torch 张量在当前目标设备上执行的正确性基线，不是把数据搬到 CPU
的临时代码。海光 Torch 可能仍显示设备类型 `cuda`；这不表示可以导入 NVIDIA Warp。

第一阶段只承诺：

| 项目 | 范围 |
|---|---|
| 算法 | 固定晶胞、BAOAB Langevin NVT |
| 输入 | `positions/velocities/forces: [N, 3]`，`masses: [N]`，`batch_idx: [N]` |
| 参数 | 每个体系一个 `dt`、`kT`、`friction`，形状 `[M]` |
| dtype | `float32`、`float64`；索引为 `int32` 或 `int64` |
| 设备 | CPU Torch reference；有分配设备后验证 Hygon DCU Torch |
| 梯度 | 先保证状态更新 forward 语义；不宣称积分器支持训练高阶梯度 |
| 随机数 | 同一设备、同一输入、同一 seed 可复现；不要求 CPU/HCU 逐位相同 |
| 不包含 | 变胞、NPT/NPH、域分解、Triton/HIP kernel、性能 auto 选择 |

## 1. 开始前的准备

### 1.1 检查仓库

从项目根目录执行：

```bash
pwd
git status --short --branch
git log -5 --oneline --decorate
```

应该看到干净的候选基础版本分支，或者负责人指定的阶段分支。如果 `git status` 已有
别人的改动，不要用 `reset --hard`、`git checkout --` 或覆盖式复制，先确认文件归属。

先读：`AGENTS.md`、`docs/PROJECT_HANDOFF.md`、`docs/STATUS.md`、
`docs/TEAM_DEVELOPMENT_BASELINE.md`、`docs/ADD_TORCH_OPERATION.md`。

### 1.2 创建自己的分支

假设开发者名为 `alice`，任务名为 `torch-nvt-langevin`：

```bash
git switch team/dev-baseline-v0.1
git switch -c alice/feature-torch-nvt-langevin
git status --short --branch
```

如果基础版本已经合入 `develop`，则从 `develop` 创建。个人姓名只放在分支名，不写进运行
时代码。

### 1.3 加载环境

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -c 'import torch; print(torch.__version__); print(torch.version.hip)'
```

不要为解决导入错误直接 `pip install torch` 或 `uv sync --all-extras`。海光 Torch 是受保护
的项目依赖，环境问题先看 `docs/DEVELOPMENT_ENVIRONMENT.md`。

### 1.4 认识目录职责

| 目录 | 本任务用途 |
|---|---|
| `packages/framework` | 公共 `NVTLangevin`、Torch reference、framework dispatcher 和测试 |
| `packages/ops` | 统一 `ImplementationRegistry` 和 backend selection metadata |
| `external/` | 只读查看锁定上游，不能修改、不能作为产品安装源 |
| `probes/` | 在真实 CPU/HCU 环境验证并保存原始结果 |
| `reports/` | 提交脱敏验证摘要 |

当前 VV、kinetics、FIRE reference 都放在 framework 的 `_dynamics_reference`，所以第一版
Langevin reference 也放在那里，不要复制一份算法到 `packages/ops`。

## 2. 先读懂上游算法

### 2.1 找到关键文件

```bash
rg -n "NVTLangevin|langevin_half_step|langevin_finalize" \
  external/nvalchemi-toolkit \
  external/nvalchemi-toolkit-ops \
  packages/framework
```

锁定上游参考文件：

- wrapper：`external/nvalchemi-toolkit/nvalchemi/dynamics/integrators/nvt_langevin.py`；
- framework adapter：`external/nvalchemi-toolkit/nvalchemi/dynamics/_ops/langevin.py`；
- ops kernel：`external/nvalchemi-toolkit-ops/nvalchemiops/dynamics/integrators/langevin.py`；
- 版本以 `docs/UPSTREAM_LOCK.yaml` 为准。

产品侧已有对应文件，但当前是 Warp 路径：

- `packages/framework/nvalchemi/dynamics/integrators/nvt_langevin.py`；
- `packages/framework/nvalchemi/dynamics/_ops/langevin.py`；
- `packages/ops/nvalchemiops/_backend_catalog/dynamics.py` 尚未登记 Langevin。

### 2.2 BAOAB 公式

每个 step 拆成 `B-A-O-A-B`。代码把它分成模型计算前后的两段：

```text
pre_update:  B 旧力半步踢速度 → A 位置半步漂移
             O 随机热浴更新速度 → A 位置再半步漂移
model:       在新位置计算新力
post_update: B 用新力半步踢速度
```

对每个原子，设 `a=F/m`、`h=dt/2`：

```text
v1 = v + h*a
r1 = r + h*v1
c1 = exp(-friction*dt)
c2 = sqrt(kT*(1-c1*c1)/m)
v2 = c1*v1 + c2*xi       xi ~ Normal(0, 1)
r_new = r1 + h*v2
v_new = v2 + h*(F_new/m)
```

三个易错点：

1. 上游 wrapper 初始化时把 `temperature` 乘以 `KB_EV`，因此 state 中的值实际是能量单位
   的 `kT`，不是 Kelvin。reference 不要重复乘一次。
2. `dt`、`kT`、`friction` 是 `[M]`，必须按 `batch_idx` 映射成每原子参数。
3. 随机算法要求同设备复现，不要求 CPU/HCU 产生相同逐位轨迹；跨设备比较统计量、有限
   性和物理范围。

## 3. 第一步：写契约，不写实现

建议先在任务说明或临时笔记中写下契约，稳定后同步到
`docs/FEATURE_COMPATIBILITY.yaml` 和 report。

```text
langevin_half_step(
    positions, velocities, forces, masses,
    dt, kT, friction, random_seed, batch_idx
) -> None

langevin_finalize(
    velocities, forces_new, masses, dt, batch_idx
) -> None
```

| 字段 | 约定 |
|---|---|
| positions | `[N,3]`，固定晶胞 Cartesian 坐标，原地修改 |
| velocities | `[N,3]`，与 positions 同 dtype/device，原地修改 |
| forces | `[N,3]`，旧力/新力输入，不修改 |
| masses | `[N]`；若需要 `[N,1]` 兼容，沿用现有 VV 约定并写测试 |
| dt/kT/friction | `[M]`，每个体系一个值，由 dynamics state 提供 |
| batch_idx | `[N]`，每个原子对应体系 ID，不允许跨体系取参数 |
| random_seed | Python `int`，由 `random_seed + step_count` 形成每步 seed |
| 返回值 | `None`，通过 mutation 更新 state |
| 梯度 | 初版只声明状态更新 forward；不通过 detach 掩盖梯度问题 |

还要明确：同 device、同输入、同 seed 复现；空输入、单体系、异构体系；device/dtype/shape
不匹配；batch 越界；非正质量；负 `dt`、`friction`、`kT`；变胞和不支持特征。

所有行为先查锁定上游。若要改变上游行为，单独写理由和测试，不要让 `sqrt`/`exp` 自己
产生难懂的 NaN。

选择语义必须保持：

```text
backend=None              → 上游 Warp legacy
backend="torch_reference" → 新的 torch_reference.langevin-v1
backend="auto"            → 没有 M2 profile 时不凭空选择新实现
backend="triton"/"hip"/未知 → 无 capability 时明确失败
```

## 4. 第二步：先写 CPU 测试

新增：

```text
packages/framework/test/compatibility/test_dynamics_reference_langevin.py
```

### 4.1 状态更新和随机复现

先覆盖最小形状和 mutation：

```python
positions = torch.zeros((4, 3), dtype=torch.float64)
velocities = torch.ones_like(positions)
forces = torch.ones_like(positions)
masses = torch.ones(4, dtype=torch.float64)
dt = torch.tensor([0.1], dtype=torch.float64)
kT = torch.tensor([KB_EV * 300.0], dtype=torch.float64)
friction = torch.tensor([0.1], dtype=torch.float64)
batch_idx = torch.zeros(4, dtype=torch.int32)
```

保存 old positions/velocities，调用 half step，检查 shape 不变且 state 改变。这个测试只
证明“发生了更新”，不能证明公式正确。

再准备两份完全独立的相同输入，用相同 seed 调用两次，检查 positions/velocities 相等；
换 seed 后检查通常不同。不要共享第一次调用已经被修改的 tensor。

### 4.2 用 VV 检查确定性极限

当 `friction=0` 时，`c1=1`、`c2=0`，随机项消失，BAOAB 退化为 velocity Verlet。
准备两份相同状态：

```text
一份：Langevin half step + Langevin finalize
另一份：VV position update + VV velocity finalize
```

用 float64 紧容差比较 positions 和 velocities。这会同时检查 B-A-O-A-B 顺序和半步系数，
比“结果不相等”有用得多。

### 4.3 异构 Batch 和 finalize

使用 `batch_idx=[0,0,1,1,1]`，参数为 `dt=[0.1,0.2]`、不同 `kT` 和 `friction`。检查：

- system 0 只使用参数下标 0；system 1 只使用下标 1；
- 修改 system 0 参数不会改变 system 1 结果；
- batch 越界、负数、长度不一致时明确失败。

`finalize` 只修改 velocities，不修改 positions 或 forces，并满足：

```text
v_expected = v + 0.5 * dt_atom * forces_new / mass_atom
```

### 4.4 先运行测试

```bash
source scripts/activate_hygon_env.sh project
export PYTHONPATH=packages/framework:packages/ops
.venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_langevin.py
```

此时测试失败是正常的；失败应该是函数不存在、模块不存在或断言失败，不应是环境错误、
Warp 初始化或错误 import path。

## 5. 第三步：实现 Torch reference

### 5.1 新建 reference 文件

建议文件：

```text
packages/framework/nvalchemi/_dynamics_reference/langevin.py
```

它应满足：

- 顶层不能 `import warp`；
- 顶层不能导入 `nvalchemi.dynamics`，避免 namespace import cycle；
- 只接收 Torch tensor 和 Python integer；
- 维持设备，不使用 `.cpu()`、`.numpy()`、逐元素 Python 循环或 `.item()` 作为默认路径；
- 维持上游 mutation 语义；
- 先做输入检查，再做张量计算；
- 如果使用 `torch.library.custom_op`，补 fake/meta 行为，并按当前 Torch 版本验证。

可以参照但不要机械复制：

```text
packages/framework/nvalchemi/_dynamics_reference/velocity_verlet.py
packages/framework/nvalchemi/_dynamics_reference/kinetics.py
packages/framework/nvalchemi/_dynamics_reference/fire.py
```

### 5.2 用 Torch 张量实现公式

核心计算可以按下面的伪代码组织：

```python
batch = batch_idx.to(torch.int64)
dt_atom = dt.index_select(0, batch).unsqueeze(-1)
kT_atom = kT.index_select(0, batch).unsqueeze(-1)
gamma_atom = friction.index_select(0, batch).unsqueeze(-1)
mass_atom = masses.reshape(-1, 1)

inv_mass = torch.where(
    mass_atom > 0, 1.0 / mass_atom, torch.zeros_like(mass_atom)
)
half_dt = 0.5 * dt_atom

# B-A
velocity_half = velocities + half_dt * forces * inv_mass
position_half = positions + half_dt * velocity_half

# O
c1 = torch.exp(-gamma_atom * dt_atom)
c2 = torch.sqrt(kT_atom * (1.0 - c1.square()) * inv_mass)
xi = device_local_normal_random_numbers(...)
velocity_thermostat = c1 * velocity_half + c2 * xi

# A
positions_new = position_half + half_dt * velocity_thermostat
```

实际 mutation 应在正确的边界完成。例如 custom op 可以先计算临时 tensor，最后在
`torch.no_grad()` 中写回 positions/velocities；不要在半途写回后才检查输入错误。

### 5.3 随机数必须在目标设备生成

先做一个小探针，确认当前项目 Torch 是否支持目标 device 上的：

```python
generator = torch.Generator(device=positions.device)
generator.manual_seed(random_seed)
noise = torch.randn(
    positions.shape,
    dtype=positions.dtype,
    device=positions.device,
    generator=generator,
)
```

确认：CPU/HCU 都能执行；同设备同 seed 可复现；不会偷偷在 CPU 生成再 `.to(device)`；每个
原子得到三维 noise；不同 step 使用 `random_seed + step_count`。

如果 Hygon Torch 的 generator 或 device random API 不可用，不要把随机数临时放到 CPU 作为
“兼容方案”。记录平台阻塞，再决定设备内随机 API、stateless counter-based generator，或
暂缓 capability 登记。

初版不需要复现 Warp 的 `wp.rand_init`/`wp.randn` 逐位序列。跨设备要比较有限性、均值/方差、
温度分布和短轨迹行为。

### 5.4 custom op 的最小注意事项

现有 VV reference 使用类似结构：

```python
@torch.library.custom_op(
    "nvalchemi::reference_langevin_half_step",
    mutates_args={"positions", "velocities"},
)
def reference_langevin_half_step(...):
    ...

@reference_langevin_half_step.register_fake
def _reference_langevin_half_step_fake(...):
    return None
```

上面只是结构示意，名称和签名必须按当前源码设计。第一阶段顺序应是：

```text
eager CPU 正确 → eager HCU 正确 → fake/meta → 必要的梯度 → 单独评估 compile
```

不要因为 compile 失败就删除输入检查或随机数逻辑。compile 是独立能力，不是 eager 正确
性的替代品。

## 6. 第四步：接入 framework dispatcher

### 6.1 先看已完成的 VV dispatcher

```bash
sed -n '1,240p' packages/framework/nvalchemi/dynamics/_ops/velocity_verlet.py
```

要复用的模式：

1. Warp custom op 放在私有/延迟边界；
2. 对外 dispatcher 保留旧参数，并额外接受可选 `selection`；
3. 有 selection 时检查 `selection.operation == "langevin"`；
4. 将固定的 operation entrypoint 名称和 framework 局部 legacy handler 交给通用
   `execute_selected(selection, entrypoint_name, legacy_fn, ...)`；
5. binding 按 registry 声明的 executor 模块和 entrypoint lazy import/call；它不包含 Langevin
   或其他 operation 的分派表；
6. legacy selection 走原有 Warp handler，未知或未登记 capability 仍明确报错。

不要让 dispatcher 根据原始字符串再解析一次。framework 应在第一个 concrete batch 到来
时解析并缓存 selection，pre/post 两个阶段复用同一个 selection。

### 6.2 保持旧调用兼容

上游测试可能直接调用：

```python
langevin_half_step(
    positions, velocities, forces, masses,
    dt, temperature, friction, 42, batch_idx,
)
```

新参数不要插入已有位置参数中间，推荐放在末尾：

```python
def langevin_half_step(
    positions, velocities, forces, masses, dt,
    temperature, friction, random_seed, batch_idx,
    *, selection: BackendSelection | None = None,
) -> None:
    ...
```

如果还需要 `backend` 兼容参数，像 VV/FIRE 一样统一交给 framework resolver，不在 dispatcher
内部维护另一套 backend 字符串集合。

### 6.3 dispatcher 的安全检查

调用 reference 前检查：

```text
selection.operation == "langevin"
selection.device/dtype/features 与实际调用一致
entrypoint 名称和参数签名与 Langevin dispatcher ABI 一致
```

selection 错误时抛出清楚的 `BackendUnavailableError` 或 `ValueError`。不要看到 selection
不认识就回退 Warp，也不要看到 Torch 失败就回退 CPU。

## 7. 第五步：登记 registry capability

在：

```text
packages/ops/nvalchemiops/_backend_catalog/dynamics.py
```

新增 metadata，结构类似：

```python
Implementation(
    implementation_id="torch_reference.langevin-v1",
    operation="langevin",
    family="torch_reference",
    executor="nvalchemi._dynamics_reference.langevin",
    entrypoints=("langevin_half_step", "langevin_finalize"),
    executor_owner="framework",
    features=frozenset({"fixed_cell", "stochastic", "per_graph"}),
    max_gradient_order=0,
    evidence="T1 Torch-reference Langevin contract",
    default_strategy=True,
)
```

具体字段以当前 `Implementation` 定义为准。记住：

- catalog 只能保存 metadata、模块路径、entrypoint 元组和 owner，不能导入 reference executor；
- implementation ID 必须唯一；
- `operation="langevin"` 是统一 contract，half/final 是同一 selection 下的两个阶段；
- `fixed_cell`、`stochastic`、`per_graph` 是能力要求，不是装饰性标签；
- HCU 没跑过前，evidence 不能写成 DCU verified；
- 不要把新实现直接加入 `auto` 全局偏好，当前没有 M2 profile。

增加 registry 测试：

```python
selection = resolve_backend(
    "torch_reference",
    operation="langevin",
    device="cpu",
    dtype=torch.float64,
    features={"fixed_cell", "stochastic", "per_graph"},
)
assert selection.implementation_id == "torch_reference.langevin-v1"
```

同时测试 `variable_cell` 不满足时明确失败，以及未知 backend 明确失败。

## 8. 第六步：接入公共 `NVTLangevin`

### 8.1 在构造器中增加 backend

参照 `packages/framework/nvalchemi/dynamics/integrators/nve.py`：

```text
__init__(..., backend: str | None = None, ...)
    self.backend = backend
    self._backend_selection = None
```

不要把默认值改成 `"torch_reference"`。项目约定是默认 `None` 保持上游 Warp。

### 8.2 第一个 concrete batch 到来时解析一次

在 `_init_state` 完成 device、dtype、体系数和 state 初始化后解析：

```python
self._backend_selection = resolve_compute_backend(
    self.backend,
    operation="langevin",
    device=batch.positions.device,
    dtype=batch.positions.dtype,
    gradient_order=0,
    features={"fixed_cell", "stochastic", "per_graph"},
)
```

这样做是因为此时才知道真实 device、dtype、M，也符合“解析一次、后续复用”的架构。

### 8.3 pre/post 传同一个 selection

```python
langevin_half_step(
    batch.positions,
    batch.velocities,
    batch.forces,
    batch.atomic_masses,
    self._state.dt,
    self._state.temperature,
    self._state.friction,
    self._random_seed + self.step_count,
    self._get_batch_int32(batch),
    selection=self._backend_selection,
)
```

`post_update` 也传同一个 selection。不要每个 step 重新解析，也不要为 half/final 生成两个
不同 selection。

保留已有的：

- `temperature * KB_EV` 单位转换；
- `_make_new_state` 对 inflight 新体系的状态形状；
- `_get_batch_int32` 的 topology 变化刷新逻辑；
- `random_seed + step_count` 的步级 seed 语义。

## 9. 第七步：接 framework 集成测试

### 9.1 公共 wrapper 测试

用已有 DemoModel 验证三步 reference workflow：

```python
model = DemoModelWrapper(DemoModel())
model.eval()
data = make_data_with_positions_velocities_and_masses(...)
batch = Batch.from_data_list([data])

dynamics = NVTLangevin(
    model=model,
    dt=0.1,
    temperature=300.0,
    friction=0.1,
    random_seed=42,
    backend="torch_reference",
)
for _ in range(3):
    dynamics.step(batch)

assert torch.isfinite(batch.positions).all()
assert torch.isfinite(batch.velocities).all()
```

这证明公共 workflow 能走通，但不能单独证明物理正确。

### 9.2 selection 只解析一次

参照：

```text
packages/framework/test/compatibility/test_backend_selection_propagation.py
```

用 monkeypatch 或调用计数器验证：

- `_init_state` 解析一次；
- 多次 `step()` 不增加解析次数；
- pre_update 和 post_update 收到同一个 `BackendSelection` 对象；
- selection 的 operation 是 `langevin`，不是 `velocity_verlet` 或通用字符串。

### 9.3 保留默认 Warp 反向守护

```text
NVTLangevin(..., backend=None)
    仍表示 legacy Warp；没有 Warp/设备时可以明确失败，但不能偷偷变成 Torch。

NVTLangevin(..., backend="torch_reference")
    才进入新 reference 路径。
```

如果默认 Warp 测试在无 Warp 环境失败，不要为了 CPU gate 通过而改默认值；把它记录为
legacy boundary 或环境限制。

## 10. 第八步：正确运行测试

先运行新测试：

```bash
source scripts/activate_hygon_env.sh project
export PYTHONPATH=packages/framework:packages/ops
export NVALCHEMI_TEST_BACKEND=torch_reference

.venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_langevin.py
```

再运行现有相关测试：

```bash
.venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_import_boundary.py \
  packages/framework/test/compatibility/test_backend_selection_propagation.py \
  packages/framework/test/dynamics/test_state_management.py -k 'nve or fire'
```

最后运行基础 gate：

```bash
scripts/check_cpu_reference.sh
```

该脚本会分进程运行 framework 和 ops。不要把两包测试强行合并到一个 pytest 进程，因为
上游都存在顶层 `test` 包名，可能出现 `ImportPathMismatchError`。

新增 Langevin 测试后，要把它显式加入 `scripts/check_cpu_reference.sh` 的 framework 列表，
或者确认它会被脚本已有的 compatibility 集合收集。

## 11. 第九步：HCU 验证

### 11.1 先检查设备可见性

HCU 测试只能在已经分配给你的设备上运行：

```bash
test -e /dev/kfd
test -e /dev/dri
echo "$HIP_VISIBLE_DEVICES"
```

不要因为 `torch.cuda.is_available()` 的命名就假设这是 NVIDIA，也不要在共享服务器上猜
设备或占满所有卡。

### 11.2 写专用 probe

建议新增：

```text
probes/dynamics_reference_langevin.py
```

probe 至少支持：

```bash
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python probes/dynamics_reference_langevin.py --device cpu

HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 90 \
  .venv/bin/python probes/dynamics_reference_langevin.py --device cuda
```

输出 JSON 或清晰文本，至少记录：实际 Torch/HIP 版本、设备名称、dtype、atom/system 数、
step 数、seed、finite 检查、same-seed 结果、`friction=0` 与 VV oracle 的最大误差、多体系
温度范围和退出码。不要只打印“success”。

### 11.3 如何判断 HCU 结果

可以写入 DCU 窄 slice 证据的条件：

- reference eager 在显式 HCU 设备上退出码为 0；
- shape、dtype、mutation、batch 边界和错误路径通过；
- `friction=0` 与 Torch CPU FP64 oracle 在定义容差内一致；
- 随机温控统计有限且处于预期范围。

不能写成完整支持的情况：

- 只有一个原子或一次调用通过；
- 只有 `torch.randn` 能运行，没有验证公共 workflow；
- 只有 CPU 通过；
- 只和 Warp 的随机轨迹逐位比较；
- 没有设备、环境、命令和退出码；
- HCU 失败后自动改到 CPU 并报告“通过”。

HCU 完成后更新 `docs/FEATURE_COMPATIBILITY.yaml`、`docs/STATUS.md` 和
`reports/g2-torch-nvt-langevin.md`。`implemented`、`CPU verified`、`HCU verified` 是不同
概念，必须分别写。

## 12. 常见错误

| 现象 | 常见原因 | 正确处理 |
|---|---|---|
| import reference 时加载 Warp | reference 或 namespace 有 eager import | 移到 lazy dispatcher，增加 `sys.modules` 边界测试 |
| HCU 出现 CPU/device mismatch | 参数或随机数在 CPU 创建 | 所有 tensor 在输入 device 创建，不做 CPU fallback |
| 两体系结果互相影响 | 直接使用 scalar 参数或 batch 映射错误 | 用 `index_select(batch_idx)` 做 `[M] → [N]` 映射 |
| 温度差约 `KB_EV` 倍 | Kelvin 与内部 kT 重复或漏转换 | wrapper 只转换一次，reference 消费 state 中的 kT |
| 同 seed 两次结果不同 | 使用全局 RNG 状态或共享 generator 未重置 | 每次调用按 step seed 使用确定的设备内 RNG |
| 旧测试参数错位 | 新关键字参数插入旧位置参数 | 新参数放末尾并使用 keyword-only |
| 新实现悄悄走错路径 | dispatcher 搬入中央 operation 分派表 | entrypoint 由 catalog 声明，binding 通用化；未知 capability 明确报错 |
| catalog 导入 Warp | metadata 文件顶层导入 executor | 只写 `Implementation` 和字符串路径 |
| 默认值被改成 Torch | 混淆显式 reference 与默认 legacy | 恢复 `None` legacy，单测显式传 reference |
| CPU 通过、HCU 失败仍报告通过 | 把 reference 当 CPU fallback | 写 HCU pending/blocked，不能改写结果 |

## 13. 应提交哪些文件

一个聚焦的 Langevin reference 变更通常会涉及：

```text
packages/framework/nvalchemi/_dynamics_reference/langevin.py
packages/framework/nvalchemi/dynamics/_ops/langevin.py
packages/framework/nvalchemi/dynamics/integrators/nvt_langevin.py
packages/ops/nvalchemiops/_backend_catalog/dynamics.py
packages/framework/test/compatibility/test_dynamics_reference_langevin.py
packages/framework/test/compatibility/test_backend_selection_propagation.py
probes/dynamics_reference_langevin.py
scripts/check_cpu_reference.sh
docs/FEATURE_COMPATIBILITY.yaml
docs/STATUS.md
reports/g2-torch-nvt-langevin.md
```

不一定每次都修改全部文件：如果只完成 reference，就只提交 reference 和测试；不要把未完成
的 framework 接线和未验证 capability 一起提交成“已支持”。

推荐拆成三个小提交：

```text
feat(framework): add Torch Langevin reference contract
refactor(framework): route NVTLangevin through backend selection
docs(test): record Torch Langevin validation
```

每个提交前执行：

```bash
git diff --check
git status --short
git diff --stat
git add <只属于你的明确路径>
git commit -s -m "..."
```

不要用 `git add -A` 把其他人的改动带进来，不要自动 push。

## 14. 完成定义

只有全部满足下面条件，才可以交给负责人审阅：

- [ ] 上游来源、算法公式和单位已写清楚；
- [ ] CPU Torch reference 覆盖 mutation、解析极限、随机 seed、异构 Batch、错误输入；
- [ ] reference 不导入 Warp，不发生静默 CPU fallback；
- [ ] registry 有唯一 implementation ID 和正确 capability；
- [ ] framework dispatcher 接受并复用 `BackendSelection`，不二次解析；
- [ ] 公共 `NVTLangevin(backend="torch_reference")` 完成短 workflow；
- [ ] `backend=None` legacy 行为未改变；
- [ ] `scripts/check_cpu_reference.sh` 通过；
- [ ] HCU probe 已运行，或 report 明确写 pending/阻塞原因；
- [ ] compile、训练梯度、分布式、性能和变胞范围已明确列为已验证或未验证；
- [ ] `FEATURE_COMPATIBILITY.yaml`、`STATUS.md`、report 和必要的 upstream/ADR 已同步；
- [ ] `git diff --check` 通过，提交范围单一且没有修改 `external/`。

## 15. 卡住时怎么做

1. 复制完整 traceback，不要只写“跑不通”；
2. 判断问题属于 import、shape/dtype、数值、随机数、dispatcher、环境还是 HCU 设备层；
3. 用最小输入把问题缩小到 reference 或 framework；
4. 检查锁定上游同一符号，不根据其他版本猜 API；
5. 如果要改变公共 API、单位、随机数语义或 capability 范围，先停下并报告影响；
6. 交接时分别报告实现、测试、数值结果、未验证假设和下一步。

完成本教程后，下一位开发者应能从你的 branch 和 report 继续工作，而不需要猜测哪些是真实
实现、哪些只有 CPU 证据、哪些还没有 HCU 证据。
