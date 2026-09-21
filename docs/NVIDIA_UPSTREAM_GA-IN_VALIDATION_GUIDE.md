# NVIDIA 上游 Ga-In 来源样例派生应用验证指南

> 文件名：`docs/NVIDIA_UPSTREAM_GA-IN_VALIDATION_GUIDE.md`
>
> 目的：在开始 Hygon DomainParallel 移植前，先在 NVIDIA GPU 上证明“只使用上游已有能力”的改造版 workload 能成立。
>
> 上游 framework：`NVIDIA/nvalchemi-toolkit@4dfe3723def34df3fadb245981081ccf8c94c257`
>
> 上游 ops：`NVIDIA/nvalchemi-toolkit-ops@26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`
>
> 更新时间：2026-09-21

---

## 1. 为什么必须先做这个验证

当前 Hygon 项目的主要目标是“移植上游已有能力”。

原 GPUMD Ga-In 应用包含：

```text
NEP4-ZBL
heat_lan
anisotropic NPT
anisotropic NPH-MTTK
388800 atoms
4 GPU
```

其中：

- `DomainParallel`、MACE、NVT、NPT、NPH、变胞和 global thermo：上游已有；
- NEP4-ZBL：上游没有；
- `heat_lan` grouped thermostat：上游没有。

所以在海光开发前必须先回答：

> 不增加 NEP、不增加 heat_lan，只使用 NVIDIA 上游已有的 MACE + DomainParallel + NVT/NPT/NPH，能不能构造一个足够接近原应用计算特征的 workload？

如果 NVIDIA 上游自己都不能完成设想的改造版 workload，就不应把这个问题交给 Hygon port 去解决。

硬边界：本验证只组合和测试锁定上游已经存在的 API、算法与模型适配。若某项能力在锁定上游
不存在或不能成立，结果是 upstream blocker 或范围缩小；不得先给上游新增 feature，再把它
包装成 Hygon 移植前提。

---

# 2. 验证边界

本验证必须使用**锁定的上游源码**，不要使用当前 Hygon 仓库已经修改过的 `packages/framework` / `packages/ops` 来得出“上游支持”结论。

目标环境：

```text
NVIDIA GPU
Linux
Python 3.12
PyTorch CUDA build
>= 2 GPUs
4 GPUs for representative application acceptance
```

核心测试：

```text
MACE + DomainParallel + NVT
MACE + DomainParallel + NPT
MACE + DomainParallel + NPH
anisotropic cell
representative periodic system
2 GPU correctness
4 GPU representative run
```

只有 2 GPU 时可以完成 correctness gate，但最终结论最多为 `CONDITIONAL_GO`；4 GPU
representative run 是完整 `GO` 的硬条件。

不测试：

```text
NEP
heat_lan
Hygon
HIP
GPUMD input compatibility
melting-point scientific equivalence
```

## 2.1 代表性输入事实与传递方式

源样例位于 Hygon 服务器：

```text
/home/wangleping/codes/atoms_388800
```

该路径不是 NVIDIA 机器可依赖的共享产品路径。向同事传递数据时应提供 manifest 和经批准的
大文件传输位置，并在 NVIDIA 侧复核 SHA256。当前关键快照为：

```text
run.in                    e0e2f94cc15bb7600550b924aafb854726d70184e52f5869379e962e09373108
model.xyz                 03b4f28fc411a2c3a3c56dc6c2d7ed1fa002adff0fc8e4b88d988f035da5a8d5
nep.txt                   e68f065784724a6242f44b080f3347fa50116e351cf763316a27f2d89853b34c
output/restart.xyz        24a84b5216d0d04e2743a74c4e541fc6399947884affd619c5937e01b1b77a76
output/output.log         bf5daeadcf3fefbd54970efdc16ae69ba6f3380cdd0ece5eb6d09b9c1d7805bb
output/error.log          91ccc2883b657af96c6dc264c13c142a963b8666f05f0d1dc33452dabfe0f3f1
```

必须保留的事实边界：

- `nep.txt` 声明 Ga/In 两种类型；
- 当前 `model.xyz` 和 `output/restart.xyz` 实际均为 388,800 个 Ga、0 个 In；
- group 0/1 各 194,400 个原子，是空间控温分组，不是元素类型；
- `restart.xyz` 是 1,500,000 步 NPH 后的末态，不是 `heat_lan` 刚结束时的 NPH 起点；
- 原始 `error.log` 非空，记录过一个 kernel launch-bounds 诊断；虽然主日志完成全部步骤，仍应
  在来源说明中保留，不能把原始运行描述为无告警基线。

因此本指南中的“Ga-In 派生”指来源于 Ga-In 势/应用工作流，不表示当前输入覆盖混合 Ga-In
组成。若必须验证 In，需要另提供含 In 的输入，这不是本次移植临时生成的新功能。

---

# 3. 建议环境准备

## 3.1 获取锁定源码

```bash
git clone https://github.com/NVIDIA/nvalchemi-toolkit-ops.git
cd nvalchemi-toolkit-ops
git checkout 26dbceb61e30cca80e1a5805eebeb51d7dc68fd1
cd ..

git clone https://github.com/NVIDIA/nvalchemi-toolkit.git
cd nvalchemi-toolkit
git checkout 4dfe3723def34df3fadb245981081ccf8c94c257
cd ..
```

验证：

```bash
git -C nvalchemi-toolkit rev-parse HEAD
git -C nvalchemi-toolkit-ops rev-parse HEAD
```

必须与上面的锁定 SHA 一致。

## 3.2 使用 uv 建环境

示例：

```bash
cd nvalchemi-toolkit
uv venv --python 3.12 .venv
source .venv/bin/activate
```

上游提供：

```text
cu12
cu13
mace
```

extras。

根据机器实际 CUDA/PyTorch 环境选择 `cu12` 或 `cu13`。

为了确保 ops 也使用锁定源码，建议先安装本地 ops，再安装 framework：

```bash
uv pip install -e '../nvalchemi-toolkit-ops[torch-cu12]'
uv pip install -e '.[cu12,mace]'
```

上述命令是环境构造示例，不等于依赖已经锁定。执行者必须优先检查上游 `uv.lock` 与目标 CUDA
环境是否兼容；若不能直接 `uv sync --locked`，至少保存完整的 `uv pip freeze`、安装命令和
解析后的 PhysicsNeMo/MACE/Torch 版本。不得只记录顶层 extra 名称。

如果机器是 CUDA 13，对应换成：

```text
torch-cu13
cu13
```

安装后必须记录：

```bash
python - <<'PY'
import torch
import nvalchemi
import nvalchemiops

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
print("nvalchemi:", nvalchemi.__file__)
print("nvalchemiops:", nvalchemiops.__file__)
PY
```

重点确认 `nvalchemiops.__file__` 指向锁定源码目录，而不是意外装入其他版本。

还必须确认两个源码工作树干净，并记录：

```bash
git -C nvalchemi-toolkit status --short
git -C nvalchemi-toolkit-ops status --short
uv pip freeze
```

派生验证脚本放在独立 validation/probe 目录，不直接修改锁定上游源码。最终把脚本、命令、
退出码和脱敏结果交回项目；只交终端截图不算可复现证据。

---

# 4. U0：硬件和基础环境确认

记录：

```bash
nvidia-smi
```

以及：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

最低要求：

```text
2 GPU
torch.cuda.is_available() == True
```

4 GPU 用于最终 representative validation。

---

# 5. U1：先运行上游自带 distributed 示例

不要先写代表性输入脚本；先证明上游自带示例在锁定环境可运行。

## 5.1 MACE + NVTLangevin

上游已有：

```text
examples/distributed/03_mace_nvt_distributed.py
```

先：

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/distributed/03_mace_nvt_distributed.py
```

具体参数以该脚本 `--help` 为准：

```bash
python examples/distributed/03_mace_nvt_distributed.py --help
```

目标：

- DistributedManager 正常；
- DeviceMesh 正常；
- MACE 正常；
- neighbor 正常；
- DomainParallel 正常；
- Langevin 正常；
- trajectory gather 正常。

## 5.2 MACE + NPT

上游已有：

```text
examples/distributed/06_mace_npt_distributed.py
```

先 2 GPU：

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/distributed/06_mace_npt_distributed.py
```

确认：

```text
energy
forces
stress
cell change
global pressure
```

均正常。

再按脚本参数提高到 4 GPU 做 smoke。

### U1 验收

如果 upstream 原始示例都失败：

```text
UPSTREAM_BASELINE = FAIL
```

先解决 NVIDIA upstream 环境问题，不开始 Hygon port。

---

# 6. U2：验证 anisotropic NPT

上游自带 MACE NPT 示例默认使用 isotropic，因此需要做一个**最小派生脚本**，但不修改 framework。

从：

```text
06_mace_npt_distributed.py
```

复制成项目外验证脚本，例如：

```text
validation/mace_npt_aniso_distributed.py
```

只修改 workload 参数：

```python
pressure_coupling="anisotropic"
```

原 GPUMD：

```text
aniso 1e-4 1e-4
```

目标压力 `1e-4 GPa` 约为：

```text
6.241509e-7 eV/Å^3
```

nvalchemi 的 NPT/NPH pressure 单位为：

```text
eV/Å^3
```

推荐显式传三个相同方向压力，而不是依赖隐式广播。

时间步长沿用：

```text
1 fs
```

对于 thermostat/barostat time：

- 当前源日志分别为 100 timesteps 和 1000 timesteps；在 `dt=1 fs` 下可先映射为
  `thermostat_time=100 fs`、`barostat_time=1000 fs`；
- 必须在日志中记录实际值；
- 本验证不宣称和 GPUMD MTTK 的积分 splitting 完全等价。

原 NPH 阶段还有 `correct_velocity 50`。若锁定上游没有对应公共能力，本验证不实现它，也不把
缺少该命令判作 DomainParallel 失败。

### U2 测试

先：

```text
small periodic system
2 GPU
20~100 steps
```

再：

```text
medium periodic system
2 GPU
1000 steps
```

检查：

```text
cell on all ranks identical
stress finite
pressure finite
no NaN
no deadlock
```

### U2 验收

```text
ANISO_NPT_UPSTREAM = PASS
```

---

# 7. U3：验证 NPH under DomainParallel

上游没有像 NPT 那样的独立 MACE NPH worked example，但 API 和 DomainParallel coordinator 明确支持 NPH。

因此从 NPT example 派生一个最小验证脚本。

核心修改：

```python
from nvalchemi.dynamics import NPH
```

创建：

```python
integrator = NPH(
    model=wrapper,
    dt=1.0,
    pressure=...,
    barostat_time=...,
    pressure_coupling="anisotropic",
    hooks=[nl_hook],
    n_steps=...,
)
```

同时 MACE 必须：

```python
wrapper.set_config(
    "active_outputs",
    {"energy", "forces", "stress"},
)
```

仍然使用：

```text
DomainParallel
DomainConfig
NeighborListHook
```

### U3 测试

```text
1 GPU/reference
2 GPU/DD
```

先 20~100 steps。

比较：

```text
energy
forces
stress
cell
```

再运行：

```text
1000+ steps
```

观察稳定性。

### U3 验收

```text
ANISO_NPH_UPSTREAM = PASS
```

若失败，必须明确是：

```text
upstream bug
environment issue
script/parameter issue
```

在原因不清楚前不要进入 Hygon port。

---

# 8. U4：选择 MACE checkpoint 与当前输入兼容性 smoke

NEP 在原应用中的逻辑是提供：

```text
Ga/In-capable local MLIP interface
→ energy
→ forces
→ virial/stress
```

在改造 workload 中，MACE 只需要承担相同的**计算接口角色**，不要求科学精度与 NEP 等价。

## 8.1 首选

优先使用上游 examples 已使用、已经被 MACEWrapper 验证过的 checkpoint 系列。

不要一开始自训练模型。

模型名只用于首次解析。正式比较前必须把 checkpoint 固定为本地文件或不可变 artifact，并记录：

```text
checkpoint SHA256
MACE version
atomic number table
cutoff
dtype
plain e3nn / cuEquivariance state
active_outputs
```

NVIDIA 与 Hygon 必须使用同一 checkpoint 内容；两个环境各自重新按模型名下载但不比对 hash，
不能作为跨平台 oracle。

## 8.2 用当前全 Ga 构型做最小 smoke

从原 `model.xyz` 或 `output/restart.xyz` 中取：

- 小超胞；
- 或按原晶胞的周期重复关系构造 cell-aligned 子超胞。

不得直接从大体系中任意裁一块后继续沿用原周期边界；缩小后的结构必须没有切断周期连接、原子
重叠或明显真空层，并记录构造方法。该步骤只生成测试输入，不新增上游功能。

验证：

```text
Ga species accepted
energy finite
forces finite
stress finite
```

当前输入没有 In，因此只能额外记录 checkpoint 的元素表是否声明支持 In，不能据此声称已完成
In 的数值验证。若后续提供含 In 的真实输入，再单列 mixed-composition smoke。

随后做：

```text
10~100 step NVE
```

再做：

```text
short NPT
```

### 结果分类

#### PASS

当前全 Ga 输入在选定 MACE 下数值稳定：

```text
REPRESENTATIVE_MACE = PASS
```

可以继续使用当前 388,800 原子代表结构。

#### FAIL

如果：

```text
species unsupported
明显异常力
结构快速爆炸
stress 异常
```

则：

```text
REPRESENTATIVE_MACE = NOT_SUITABLE_FOR_VALIDATION
```

这不是 DomainParallel failure。

不要在当前任务内：

```text
实现 NEP
微调 MACE
训练新势
```

而是切换到同样计算特征的 MACE-stable periodic material。

---

# 9. U5-A：使用现有 `output/restart.xyz`（默认）

这是当前数据包已经具备、无需重新运行 GPUMD 的默认路线。

## 9.1 输入

使用经 SHA256 核验的：

```text
output/restart.xyz
388800 Ga atoms
positions + mass + velocities + group
periodic cell
```

它是原 GPUMD 完成 1,500,000 步 NPH 后的最终 restart，不是 `heat_lan` 结束时的 NPH 起点。
因此它用于验证代表性大体系计算形态，不用于逐步 continuation 或原熔点结果复现。

理想 snapshot 包含：

```text
species
positions
velocities
cell
PBC
```

GPUMD restart 中 velocity 单位为 Å/fs。转换器必须显式验证 nvalchemi 内部单位并写 round-trip
测试；不能未经换算直接假定单位相同。group 字段应保留为 metadata，即使本次 nvalchemi 不使用。

## 9.2 验证流程

```text
locked restart.xyz
        ↓
1 GPU MACE smoke
        ↓
1 GPU short NPH
        ↓
2 GPU DomainParallel NPH
        ↓
4 GPU DomainParallel NPH
```

`heat_lan` 完全跳过。这里的“solid-liquid representative”只沿用现有空间密度/有序特征的工程
判断，不替代 RDF/Q6 或论文级相态验证。

如果需要额外 sanity check，可以在 NPH 前做很短的 NPT，但不是为了重新形成两相。

## 9.3 验收

不比较：

```text
MACE energy vs NEP energy
melting temperature
interface lifetime vs paper
```

只检查：

```text
single vs multi GPU consistency
NPH variable-cell works
stress works
memory sharding
no deadlock
no atom loss
stable trajectory
```

---

# 10. U5-B：`restart.xyz` 不适合所选 MACE 时的 fallback

有两种做法。

## 10.1 B1：能力等价 workflow

用原 `model.xyz` 或其他稳定周期结构：

```text
initial structure
     ↓
anisotropic NPT
     ↓
global NVTLangevin
     ↓
anisotropic NPH
```

例如：

```text
NPT:       30 K
Langevin:  300 K
NPH:       target pressure
```

具体步数不需要复制 GPUMD：

```text
correctness: 20~100
stability:   1000~10000
```

这里的 global Langevin 只是在不增加 `heat_lan` 的条件下覆盖：

```text
NPT → thermostat → NPH
```

workflow。

它**不是两相法**。

报告必须明确：

> 此路线验证框架能力，不验证 solid-liquid coexistence 或 melting point。

## 10.2 B2：如果必须使用 `heat_lan` 结束时的两相初态

优先向应用方索取已经存在的 `heat_lan` 结束 snapshot；当前数据包没有 `dump.xyz`，不能从现有
文件恢复这个精确时点。

允许：

- 使用已有历史 trajectory；
- 应用方另行提供已有 snapshot。

然后执行 U5-A。

不得要求为本次 port 重跑前 78,000 步，也不要给 nvalchemi 新增 grouped Langevin。

---

# 11. U6：规模递增

不要直接从几百原子跳到 388,800。

建议：

```text
small     ~10^2-10^3
medium    ~10^4
large     ~10^5
target    ~388800
```

每级先单 GPU smoke，再 2 GPU。

最终 4 GPU 做 target。

如果 1 GPU target OOM，这是允许的：

```text
1 GPU = OOM
2 GPU = PASS
4 GPU = PASS
```

反而可以直接证明 DomainParallel 的容量价值。

---

# 12. U7：必须记录的 correctness 指标

对于可以在 1 GPU 跑的规模：

```text
1 GPU
vs
2 GPU
vs
4 GPU
```

记录：

```text
energy abs/rel diff
force max abs diff
force RMS diff
stress max abs diff
cell diff
atom count
```

额外记录：

```text
n_owned/rank
n_halo/rank
migration count
```

不得只看“跑完了”。

比较顺序必须分层：

1. 同一冻结构型的一次 neighbor/model forward；
2. 按稳定 global atom ID gather、排序后的 per-atom force；
3. 单步 positions/velocities/cell/controller state；
4. 少量步骤的误差增长；
5. 中长轨迹的守恒量/ensemble 统计和稳定性。

混沌 MD 长轨迹不要求逐点重合。NPH 的 conserved quantity 按锁定上游实现和上游测试定义，
不得未经推导只用普通 `E + PV` 作严格判据。

---

# 13. U8：显存和性能

原应用多卡的重要动机是显存分摊，所以必须记录：

```text
peak GPU memory / rank
```

以及：

```text
ms/step
steps/s
```

建议表：

| atoms | GPUs | peak mem/rank | ms/step | n_owned min/max | n_halo min/max |
|---:|---:|---:|---:|---:|---:|

最终重点：

```text
~388800 atoms
4 GPU
```

---

# 14. 最终 GO / NO-GO

## GO

如果以下成立：

```text
upstream MACE NVT DD PASS
upstream MACE NPT DD PASS
anisotropic NPT PASS
anisotropic NPH PASS
2 GPU correctness PASS
4 GPU representative run PASS
```

则：

```text
UPSTREAM_DP_WORKLOAD = GO
```

可以开始 Hygon port。

## CONDITIONAL GO

如果 DomainParallel/NPT/NPH 上游成立，但当前全 Ga 输入 + MACE 本身数值不稳定：

```text
UPSTREAM_FRAMEWORK = GO
REPRESENTATIVE_MACE = UNSUITABLE
UPSTREAM_DP_WORKLOAD = CONDITIONAL_GO
```

改用同规模的 MACE-stable periodic material，仍可开始 Hygon port。

## NO-GO

如果：

```text
官方 upstream MACE DD example 不能工作
或
anisotropic NPT/NPH under DD 本身不能成立
```

则先解决/确认 upstream 问题，不把它带入 Hygon 分支。

只有 2 GPU correctness、但没有完成 4 GPU representative run 时，也只能写：

```text
UPSTREAM_DP_WORKLOAD = CONDITIONAL_GO
```

---

# 15. 建议最终报告模板

```markdown
# NVIDIA Upstream Ga-In-derived Validation

## Environment
- framework SHA:
- ops SHA:
- PyTorch:
- CUDA:
- GPU:
- world size:

## U1 Official Examples
- NVT DD:
- NPT DD:

## U2 Anisotropic NPT
- status:
- parameters:
- energy/force/stress:
- cell:

## U3 Anisotropic NPH
- status:
- parameters:
- short parity:
- long stability:

## U4 MACE on representative input
- checkpoint:
- input composition:
- Ga supported:
- In declared by checkpoint (not exercised by current input):
- NVE:
- NPT:
- decision:

## U5 Application Route
- prepared two-phase: yes/no
- Route A/B:
- atoms:
- GPUs:
- status:

## Memory / Performance
...

## Final decision
UPSTREAM_DP_WORKLOAD = GO / CONDITIONAL_GO / NO_GO
```

---

# 16. 本验证不能证明什么

即使全部 PASS，也只说明：

> 上游能力足以组成一个与原 GPUMD 应用具有相近计算特征的 MACE + DomainParallel + NPT/NPH 大体系 workload。

不能说明：

- MACE 对 Ga-In 熔点和 NEP 一样准确；
- MACE 势能等于 NEP；
- nvalchemi MTK 与 GPUMD MTTK 逐步一致；
- 已经复现原论文；
- 当前全 Ga 输入已经覆盖 In 元素或混合 Ga-In 成分。
