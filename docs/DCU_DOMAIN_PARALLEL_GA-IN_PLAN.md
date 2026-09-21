# DCU DomainParallel Ga-In 来源样例派生应用开发计划

> 文件名：`docs/DCU_DOMAIN_PARALLEL_GA-IN_PLAN.md`
>
> 仓库：`nivren/alchemi-hygon`
>
> 基线分支：`develop`
>
> 基线提交：`bd26569b48cd438a2fd2e945285c3b9fb6635eb1`
>
> 当前上游锁定：`nvalchemi-toolkit@4dfe3723def34df3fadb245981081ccf8c94c257`（0.2.0），`nvalchemi-toolkit-ops@26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`（0.4.1）
>
> 状态：修订稿；执行前需确认 owner，并完成 Phase 0 双前置门
>
> 更新时间：2026-09-21

---

## 1. 目标与背景

本任务负责把 NVIDIA `nvalchemi-toolkit` 已有的**单大体系 DomainParallel 能力**移植并验证到海光 DCU，并使用一个由现有 Ga-In GPUMD 算例派生出的代表性大体系 workload 做最终工程验收。

事实来源是服务器上的只读样例目录：

```text
/home/wangleping/codes/atoms_388800
```

原始应用的关键特征是：

- 单一周期体系，388,800 个原子；
- 采用支持 Ga/In 两种类型的 NEP4-ZBL 势文件；
- 当前 `model.xyz` 与 `output/restart.xyz` 实际均为 388,800 个 Ga 原子、0 个 In 原子，
  因而本样例不能作为混合 Ga-In 成分覆盖证据；
- GPUMD 4.8；
- NEP4-ZBL 势；
- 4 GPU 运行，多卡的首要目的之一是**分摊大体系显存占用**；
- 1 fs 时间步长；
- 24,000 步 30 K 各向异性 NPT；
- 54,000 步 `heat_lan`，source/sink 温度约 570 K / 30 K；
- 1,500,000 步各向异性 NPH-MTTK；
- 输出温度、势能、应力、晶胞和轨迹，用于判断热力学平台与固液界面。

本任务**不以逐命令复现 GPUMD 科学计算为第一目标**。当前阶段的核心目标是验证：

> 上游已经具备的 DomainParallel、HALO、原子迁移、MACE、NVE/NVT/NPT/NPH、变胞和 stress/virial 路径，能否在海光 DCU 上正确运行，并能在约 388,800 原子的单体系上实现真实的多卡分片和显存分摊。

如果后续验收必须覆盖 In 元素或真实混合 Ga-In 成分，必须另行提供含 In 的输入；不得把
“势文件声明支持 In”写成“本次 388,800 原子输入已覆盖 In”。

---

## 2. 核心开发原则

### 2.1 先移植上游已有能力，不先扩充上游功能

原 GPUMD 应用包含两个当前上游没有的关键能力：

- NEP4-ZBL 模型；
- `heat_lan` 式同一体系内按 atom group 施加不同温度的 Langevin 控温。

它们**不作为本次 DomainParallel 海光移植的 blocker**。

本任务不得为了复现原 `run.in`，先在上游增加 NEP 或 grouped Langevin，再开始海光移植。否则无法清晰区分：

```text
上游新增 feature 问题
vs.
DomainParallel 本身问题
vs.
Hygon/DTK 移植问题
```

### 2.2 保留应用“计算特征”，允许替换具体模型和准备流程

本任务采用：

```text
NEP4-ZBL
    ↓ 替换
MACE（上游已支持）
```

并将：

```text
heat_lan
```

视为**原应用的两相初态制备步骤**，而不是本次 DomainParallel 移植必须实现的运行时能力。

当前应用包已经包含可用的 `restart.xyz`，它具有：

```text
388800 atoms
PBC = T T T
cell = 387.331 × 136.777 × 136.723 Å
species + positions + mass + velocities + group
```

该文件当前只包含 Ga。`nep.txt` 虽定义 Ga/In 两种类型，但原始 GPUMD 日志记录的实际组成是：

```text
388800 Ga
0 In
```

并且现有轻量分析表明：

- group 0 / group 1 各 194,400 个原子；
- 两组在 X 方向仍基本保持左右分区；
- 200-bin X 向原子数分布中，group 0 区域明显更平滑；
- group 1 区域保留强烈、规则的空间周期起伏；
- 这些现象与“左侧 liquid-like、右侧 crystal-like”的固液共存结构相容。

这不是论文级相态判定，但对**框架/系统代表性 workload** 已经足够。

因此当前路线调整为：

- **默认路线**：直接使用现有最终 `restart.xyz` 作为代表性两相大体系输入，跳过 `heat_lan`；
- **可选参考路线**：如果应用方恰好保留 `heat_lan` 结束、进入 NPH 前的 snapshot，可额外用于更接近原始 NPH 起点的验证，但它不是 blocker；
- **fallback 路线**：只有当 `restart.xyz` 在所选 MACE 下不可用时，才退回 `model.xyz → anisotropic NPT → global NVTLangevin → anisotropic NPH` 的能力验证路线；`model.xyz` 同样是全 Ga 输入。

因此本任务验证的是**系统和框架能力**，不是 NEP 对 Ga-In 熔点的科学精度。

### 2.3 MACE 是 reference workload model，不是 NEP 的科学替代结论

MACE 在本任务中承担：

```text
positions/species
    ↓
local MLIP
    ↓
energy + forces + stress
```

用于验证：

- neighbor；
- HALO；
- message passing；
- force/energy/stress；
- DomainParallel；
- NPT/NPH；
- 显存分片。

不得把 MACE 结果和原 NEP 的绝对势能、熔点、固液平衡温度直接作科学等价比较。

---

## 3. 当前仓库基线

截至 `develop@bd26569...`，当前项目已经具备：

### 3.1 已验证的海光侧基础

- Torch reference neighbor/cell-list：已有窄 reference；
- 显式 HIP neighbor/cell-list：已验证 periodic、fixed-cell、Batch、full-list、MATRIX、FP32/FP64、`skin=0` 的窄能力；
- MACEWrapper：真实 checkpoint、短 energy/force 链路已验证；
- fixed-cell NVE / FIRE / FIRE2：已有 reference；
- fixed-cell Langevin BAOAB：已有窄 Torch reference；
- backend registry / executor binding：已收口。

### 3.2 当前仍未完成

- DomainParallel 海光适配；
- DomainParallel + neighbor；
- distributed MACE；
- NPT/NPH 海光 reference/ops；
- variable-cell 海光生产路径；
- HIP neighbor 的 DomainParallel；
- HIP neighbor 的 variable-cell；
- distributed Langevin；
- 长时间大体系运行。

### 3.3 上游已有但当前海光尚未验证的关键能力

上游 NVIDIA framework 已经包含：

```text
DomainParallel
ShardedBatch
SpatialPartitioner
HaloStrategy
particle_halo
atom migration
DistributedModel
MLIPSpec
DynamicsDistributionCoordinator
NPT
NPH
NVTLangevin
MACE distribution_spec
```

上游还已有分布式 MACE NVT 和 MACE NPT 示例。

因此本任务的默认策略是：

> **复用上游算法和接口，只适配 Hygon/DTK 不兼容点。**

但“复用上游算法”不表示当前 DCU 已具备 NPT/NPH 运行路径。当前 NPT/NPH bridge 仍为
Warp-backed，必须先建立独立的 Torch reference 和单 DCU gate，再进入 distributed NPT/NPH。

---

# 4. 本任务模块边界

## 4.1 本任务必须完成

### A. DCU 分布式运行时

- `torchrun` 多进程；
- rank → DCU 映射；
- ProcessGroup；
- DeviceMesh；
- collective；
- P2P；
- async collective；
- subgroup。

### B. 单体系空间域分解

- `ShardedBatch`；
- spatial partition；
- ownership；
- HALO/ghost；
- halo reverse；
- atom migration；
- gather。

### C. DomainParallel 正确性

- Torch reference cell-list neighbor；
- toy/LJ oracle；
- plain MACE / non-cuEquivariance；
- NVE；
- anisotropic NPT；
- anisotropic NPH。

### D. 代表性大体系应用

- 周期单体系；
- 目标规模约 388,800 atoms；
- 2 DCU correctness；
- **4 DCU 为首版实际应用验收配置**；
- 8 DCU 做扩展/scaling 验证；
- 重点测 peak memory per rank、稳定性和性能。

## 4.2 本任务首版明确不做

以下不是首版 blocker：

```text
NEP / NEP4 / NEP4-ZBL
直接读取 nep.txt
GPUMD run.in parser
heat_lan / grouped Langevin
GPUMD correct_velocity command 的逐步等价
GPUMD MTTK 逐步数值等价
Ga-In 熔点科学复现
DPA3 / DPA4 / DeePMD
cuEquivariance
torch.compile distributed
Triton distributed
dynamic load balancing
multi-node production
no-PBC isolated-cluster production
```

如后续产品必须忠实复现 GPUMD 应用，应另立 `GPUMD_APPLICATION_COMPATIBILITY` 任务。

---

# 5. 三份文档之间的工作流

本计划依赖两份可以并行执行、最终汇合的前置验证文档：

```text
同事：NVIDIA_UPSTREAM_GA-IN_VALIDATION_GUIDE.md ─┐
                                                 │
                                                 ├─> Phase 0 汇总结论
                                                 │         │
本人：HYGON_DCU_DISTRIBUTED_PREFLIGHT_GUIDE.md ──┘         ▼
                                              实际 Hygon DomainParallel port
```

NVIDIA 验证与 Hygon preflight 互不依赖，不应人为串行等待。两项均完成前，不应把“上游
workload 不成立”或“底层 collective 不可用”的问题埋进 DomainParallel 代码中解决。

当前权威任务队列仍将 NPT/NPH 与 DomainParallel 标为暂停项。本文件不自行改变该状态；正式
开工前由统筹者确认 owner、分支和优先级，再由集成者更新 `PARALLEL_DEVELOPMENT_PLAN.md`。

---

# 6. Phase 0：两项前置结论

## 6.1 NVIDIA upstream validation

要求先按照：

```text
docs/NVIDIA_UPSTREAM_GA-IN_VALIDATION_GUIDE.md
```

在 NVIDIA 环境使用锁定上游 SHA 验证：

- MACE + DomainParallel + NVT；
- MACE + DomainParallel + NPT；
- NPH API / DomainParallel；
- anisotropic cell path；
- 代表性小/中体系；
- 2 GPU；
- 4 GPU；
- 两相结构存在/不存在两种改造方案。

要求得到统一结果字段：

```text
UPSTREAM_DP_WORKLOAD = GO
```

也允许 `CONDITIONAL_GO` 或 `NO_GO`，但必须明确 upstream、模型/输入和资源三类 blocker。

## 6.2 Hygon distributed preflight

按照：

```text
docs/HYGON_DCU_DISTRIBUTED_PREFLIGHT_GUIDE.md
```

验证：

- 2 / 4 / 8 DCU process launch；
- device collective；
- P2P；
- DeviceMesh；
- async communication；
- repeated stability。

要求得到：

```text
HYGON_DISTRIBUTED_BASE = GO
```

或者 `CONDITIONAL_GO` 并列出 Phase 1 必须修复的问题。

## 6.3 输入与模型锁定

进入实现前生成一份可提交的 manifest，至少记录：

```text
source directory
relative path
file size
SHA256
atom count / species count
cell / PBC
velocity unit
MACE checkpoint local path or immutable artifact id
MACE checkpoint SHA256
MACE / e3nn / torch versions
cutoff / dtype / active_outputs
```

当前样例的关键 SHA256（2026-09-21 快照）为：

```text
run.in                    e0e2f94cc15bb7600550b924aafb854726d70184e52f5869379e962e09373108
model.xyz                 03b4f28fc411a2c3a3c56dc6c2d7ed1fa002adff0fc8e4b88d988f035da5a8d5
nep.txt                   e68f065784724a6242f44b080f3347fa50116e351cf763316a27f2d89853b34c
output/restart.xyz        24a84b5216d0d04e2743a74c4e541fc6399947884affd619c5937e01b1b77a76
output/output.log         bf5daeadcf3fefbd54970efdc16ae69ba6f3380cdd0ece5eb6d09b9c1d7805bb
output/error.log          91ccc2883b657af96c6dc264c13c142a963b8666f05f0d1dc33452dabfe0f3f1
```

样例目录不属于产品仓库，不能成为运行时隐式依赖。NVIDIA 与 Hygon 两侧必须使用同一份经
SHA256 核验的输入和 checkpoint；大文件仍可留在外部数据区，仓库只提交 manifest、转换器和
脱敏报告。

---

# 7. Phase 1：DCU Runtime Compatibility

## 7.1 目标

让上游 distributed runtime 能在 Hygon DTK 环境工作，不修改 DomainParallel 算法。

本任务只允许为上游已有符号和语义增加 Hygon/DTK compatibility。若锁定上游在 NVIDIA 上未
实现或不能成立，则缩小本阶段范围或登记 upstream blocker；不得在 Hygon 分支先发明新算法、
新 ensemble 或新用户接口。

初始审计至少覆盖：

```text
packages/framework/nvalchemi/distributed/_runtime.py
packages/framework/nvalchemi/distributed/config.py
packages/framework/nvalchemi/distributed/helpers.py
packages/framework/nvalchemi/distributed/sharded_batch.py
packages/framework/nvalchemi/distributed/_core/gather_primitives.py
packages/framework/nvalchemi/distributed/_core/particle_halo.py
packages/framework/nvalchemi/distributed/_core/reshard.py
packages/framework/nvalchemi/distributed/_core/_st_backend.py
packages/framework/nvalchemi/distributed/validate/
packages/framework/nvalchemi/models/mace.py
```

这不是预先批准全部文件修改。先形成平台假设清单和失败复现，再只修改实际不兼容点。

重点审计：

```text
physicsnemo.distributed
backend == "nccl"
torch.cuda.*
torch.backends.cuda.*
torch.backends.cudnn.*
DeviceMesh device_type
indexed_all_to_all_v_wrapper
functional all_to_all_single + wait_tensor
variable/zero split sizes
ShardTensor backend/version assumptions
```

注意：

> Hygon PyTorch 可能仍复用 `torch.cuda` API 名称。不得仅凭名字把所有 `cuda` 替换成 `hip`。

所有修改必须以 Preflight Probe 实际结果为依据。

## 7.2 测试

至少：

```text
2 DCU:
- import nvalchemi.distributed
- resolve world size/rank
- correct local device
- DeviceMesh
- subgroup all_reduce

4 DCU:
same

8 DCU:
same
```

同时运行已有 CPU/Gloo distributed regression。

## 7.3 验收

- 不隐式 CPU fallback；
- 不破坏 NVIDIA 路径；
- world-size 1 行为不变；
- 2/4/8 DCU runtime smoke 均通过。

---

# 8. Phase 2：ShardedBatch / Partition / HALO / Migration

## 8.1 分层顺序

```text
global Batch
   ↓
ShardedBatch
   ↓
SpatialPartitioner
   ↓
owned atoms
   ↓
HALO forward
   ↓
HALO reverse
   ↓
migration
   ↓
gather
```

不得跳过这些层直接调 MACE。

## 8.2 优先复用的上游测试

包括但不限于：

```text
test/distributed/test_spatial_partitioner.py
test/distributed/test_deferred_migration_halo.py
test/distributed/test_domain_parallel.py

test/distributed/_core/test_gather_primitives.py
test/distributed/_core/test_halo_autograd.py
test/distributed/_core/test_halo_primitives_roundtrip.py
test/distributed/_core/test_neighbor_p2p.py
test/distributed/_core/test_particle_halo.py
test/distributed/_core/test_reshard.py
test/distributed/_core/test_shard_tensor.py
test/distributed/_core/test_subgroup_communication.py
```

## 8.3 测试体系

先用人工周期盒：

```text
32 atoms
128 atoms
1024 atoms
```

再用随机均匀周期体系。

必须检查：

```text
每个 atom 唯一 owner
scatter/gather round-trip
halo sender/receiver
halo reverse
migration
periodic boundary
global atom id continuity
```

`global atom id` 必须是显式、稳定、可 gather/sort 的字段。所有 per-atom force、ownership 和
migration 对照都先按该 ID 排序，不能依赖不同 rank 的局部行顺序。

## 8.4 验收

2 / 4 / 8 DCU：

- 0 atom lost；
- 0 duplicate owner；
- halo forward/reverse 正确；
- 100+ 人工迁移事件正确；
- 连续 1000 次 halo exchange 无 deadlock。

---

# 9. Phase 3：Distributed Neighbor

## 9.1 第一版只使用 Torch reference cell-list

当前 HIP neighbor 尚未支持：

- DomainParallel；
- variable-cell。

所以 correctness 阶段禁止同时引入 HIP neighbor。

首选：

```text
backend="torch_reference"
method="cell_list"
```

具体公共调用以当前 framework API 为准。

## 9.2 正确性 oracle

对同一体系：

```text
single-device neighbor
vs.
union(distributed owned-receiver neighbor)
```

比较 global atom pair。

比较前把局部 pair 映射回 global atom ID，并明确有向 full-list 或无向 pair 的归一化规则；
不能直接拼接局部矩阵索引后比较。

重点验证跨 domain pair：

```text
A owned by rank 0
B owned by rank 1
distance(A,B) < cutoff
```

不能漏 pair，也不能因 halo 产生物理 double counting。

## 9.3 验收

2 / 4 / 8 DCU：

- topology parity；
- periodic image shift parity；
- boundary pair parity；
- migration 后 neighbor 重建正确。

---

# 10. Phase 4：Toy/LJ End-to-End

MACE 前必须先使用简单 oracle。

测试：

```text
1 device
2 DCU
4 DCU
8 DCU
```

比较：

- energy；
- per-atom force；
- boundary atom force；
- stress/virial。

stress/virial 是后续 NPT/NPH 的硬前置，不能作为可选检查。若现有 LJ wrapper 没有 stress
输出，则新增一个仅用于验证的解析 pair-potential oracle 或独立 virial 计算器；它不成为新的
产品模型能力，也不扩大上游公共 API。

验收：

```text
distributed ≈ single-device
```

必须使用非退化 HALO partition。

同时固定并记录 `grid_dims`、`cutoff`、`skin`、ghost width 和
`require_nondegenerate=True`。至少包含一个跨周期边界 pair 和一个跨 domain boundary pair。

---

# 11. Phase 5：MACE + NVE Golden Path

## 11.1 目的

这是 DomainParallel 移植的第一个真实 MLIP 闭环。

范围：

```text
plain MACE
non-cuEquivariance
periodic
fixed-cell
Torch reference cell-list
NVE
```

## 11.2 测试

先小体系：

```text
1 DCU vs 2 DCU
energy
forces
```

再：

```text
2 vs 4 vs 8 DCU
```

最后短 NVE：

```text
100~1000 steps
```

记录：

- total energy；
- force error；
- atom count；
- migration count；
- halo size；
- peak memory；
- step time。

## 11.3 验收

2 DCU 必须首先 PASS。

之后：

- 4 DCU PASS；
- 8 DCU PASS 或明确记录为扩展项。

## 11.4 Phase 5A：单 DCU 变胞/NPT/NPH 前置门

进入 distributed NPT/NPH 前，必须先在同一 Hygon 软件栈上完成上游已有 NPT/NPH 语义的
单设备适配。该阶段不是新增 ensemble，而是把锁定上游已经存在、且 NVIDIA Phase 0 已通过的
能力移植到 DCU。

最低顺序：

```text
CPU Torch oracle for existing upstream ops
→ single DCU pressure/stress/cell-update step
→ single DCU anisotropic NPH short run
→ single DCU anisotropic NPT short run
→ same-input CPU/NVIDIA/Hygon bounded comparison
```

必须锁定并测试：

- stress 正负号与 `eV/Å^3` 单位；
- pressure shape：scalar、`[M,3]`，以及上游确已支持时的 `[M,3,3]`；
- cell layout、体积、逆矩阵和原位 mutation；
- barostat/NHC state shape、dtype 和 continuation state；
- float32/float64；
- 空输入和非法 cell/pressure 的显式错误；
- 不导入 Warp、不静默 CPU fallback、不静默换 dtype。

如果锁定上游 NVIDIA gate 没有证明某个模式成立，本阶段不得自行补齐该模式，只能缩小到已
证明的上游范围。单 DCU gate 未通过前，不进入 Phase 6/7。

---

# 12. Phase 6：MACE + anisotropic NPT

## 12.1 目的

验证和原应用最相关的第一个变胞 ensemble：

```text
30 K
anisotropic NPT
```

但不要求和 GPUMD MTTK 逐步等价。

## 12.2 要求

模型必须输出：

```text
energy
forces
stress
```

DomainParallel 必须正确处理：

```text
global KE
global DOF
global pressure tensor
replicated barostat state
replicated cell
partitioner.update_cell()
HALO geometry refresh
migration bounds refresh
```

## 12.3 参数映射

原 GPUMD 目标压力：

```text
1e-4 GPa
```

对应约：

```text
6.241509e-7 eV/Å^3
```

对于 `pressure_coupling="anisotropic"`，建议显式提供三个方向相同的目标压力，避免依赖隐式广播语义。

时间步长保留：

```text
dt = 1 fs
```

thermostat/barostat coupling time 只作为 workload 参数记录，不宣称与 GPUMD MTTK 完全等价。

当前 GPUMD 日志中的源参数是 thermostat period `100` timesteps、barostat period `1000`
timesteps；在 `dt=1 fs` 下分别对应 `100 fs` 和 `1000 fs`。这些值可作为初始 workload 参数，
但仍须按上游 NPT 定义解释。原 NPH 阶段的 `correct_velocity 50` 不属于本次必须复现的能力。

## 12.4 验收

- 1 DCU vs 2 DCU 短轨迹一致性；
- 4 DCU 运行；
- cell 在所有 rank 保持一致；
- pressure/stress 有限；
- 无 rank drift；
- migration 在变胞条件下正确；
- 1000+ steps 稳定。

---

# 13. Phase 7：MACE + anisotropic NPH

## 13.1 目的

模拟原应用最终生产阶段的**计算形态**：

```text
MLIP
+
stress
+
anisotropic variable cell
+
NPH
+
DomainParallel
```

不要求复现原 NEP 熔点。

## 13.2 验收

小体系：

```text
1 DCU vs 2 DCU
```

比较：

- energy；
- force；
- stress；
- cell evolution；
- enthalpy behavior。

中体系：

```text
4 DCU
1000~10000 steps
```

检查：

- 无 NaN/Inf；
- 无 deadlock；
- cell 同步；
- global atom count 恒定；
- peak memory 不持续增长。

---

# 14. 当前可用的代表性应用输入

当前应用包没有 `dump.xyz`，但已有：

```text
model.xyz
restart.xyz
thermo.out
compute.out
```

已确认：

### `model.xyz`

```text
388800 atoms
cell = 398.8841994 × 136.76370624 × 136.6651059 Å
Properties = species + pos + spacegroup_kinds + group
PBC = T T T
```

group 空间分区：

```text
group 0: 194400 atoms, X = 0.000 ~ 197.226 Å
group 1: 194400 atoms, X = 199.442 ~ 396.668 Å
```

说明原应用预先沿长轴 X 将体系基本均分成两块，对应 `heat_lan` 的 source/sink group。

元素组成：

```text
Ga: 388800
In: 0
```

这里的 group 是控温空间分组，不是化学元素类型。

### `restart.xyz`

```text
388800 atoms
cell = 387.331 × 136.777 × 136.723 Å
Properties = species + pos + mass + vel + group
PBC = T T T
```

GPUMD 官方定义 `restart.xyz` 与 simulation model 使用相同格式，可作为 restart 输入；其 model/restart velocity 单位为 Å/fs。

现有 40-bin / 200-bin X 向统计还显示：

- 两个 group 在长时间 NPH 后仍基本保持空间分区；
- group 0 区域原子数分布显著更平滑；
- group 1 区域存在清楚的周期性密度起伏。

工程上将其定义为：

> **388,800 原子的 solid-liquid representative input candidate**

注意：这里的“solid-liquid”是基于空间密度/有序特征的工程判断，不替代 RDF、Q6 等论文级相态分析。

---

# 15. Phase 8：Ga/NEP/GPUMD 派生代表性 workload

本阶段不再要求先取得 `heat_lan` 中间 snapshot。

当前默认输入为应用包中已有的最终 `restart.xyz`。

---

## 15.1 Route A：现有 `restart.xyz`（默认）

这是当前首选路线。

### 输入事实

`restart.xyz` 已包含：

```text
species
positions
mass
velocities
group
cell
PBC
```

因此不需要重新初始化速度即可构造连续 MD 初始状态。

但是要注意：

> 这是原 GPUMD 长时间 NPH 后的 restart，不是 `heat_lan` 刚结束时的原始 NPH 起点。

所以本路线验证的是**代表性两相大体系 + DomainParallel + variable-cell NPH**，不是原 GPUMD 轨迹的逐步 continuation。

### 进入 nvalchemi 前必须检查

转换器至少要确认：

```text
atom count = 388800
species mapping
species count: Ga=388800, In=0
positions unit = Å
cell unit = Å
PBC
velocity unit conversion
group preserved as metadata（如果 framework 暂不使用 group，也不得误读）
```

GPUMD model/restart 的 velocity 单位为：

```text
Å/fs
```

如果 nvalchemi 内部采用不同单位，必须显式换算并写测试。

### 推荐流程

```text
restart.xyz
    ↓
parse / convert to nvalchemi Batch
    ↓
single-device MACE energy/force/stress smoke
    ↓
single-device short NVE or NPH
    ↓
2-GPU/DCU DomainParallel correctness
    ↓
4-GPU/DCU representative NPH
    ↓
8-GPU/DCU scaling（可选）
```

### 验收重点

不比较：

```text
MACE energy vs NEP energy
melting point
GPUMD trajectory point-by-point
```

重点比较：

```text
same-input 1 vs 2 vs 4 device consistency
global atom count
owned atoms/rank
halo atoms/rank
peak memory/rank
energy/force/stress finite
cell replicated state consistency
NPH stability
no deadlock
no atom loss
```

如果单卡 target OOM，但 2/4 卡能够运行并且每 rank 显存明显下降，这是符合本项目目标的有效结果。

---

## 15.2 Route B：`heat_lan` 末态 snapshot（可选增强）

如果应用方已经保留：

```text
NPT 24000
+
heat_lan 54000
```

结束后、刚进入 NPH 前的 snapshot，可以作为额外参考输入。

它的优势是：

- 更接近原 GPUMD NPH 的真实起点；
- 更容易解释为“跳过两相制备、直接验证生产 NPH”。

但：

> **不得要求应用方为了本次 port 专门重跑前 78,000 步。**

没有这个文件不影响首版验收。

流程与 Route A 相同。

---

## 15.3 Route C：`restart.xyz` 在 MACE 下不可用时的 fallback

只有当：

```text
Ga species unsupported by the selected MACE checkpoint
MACE force/stress 明显异常
restart state 在所选 MACE 下立即爆炸
```

时才使用 fallback。

### C1：仍保留当前全 Ga geometry

如果只是 velocity continuation 不合适：

```text
model.xyz or restart positions
        ↓
重新初始化 velocity
        ↓
short NVE/NPT smoke
```

### C2：最小能力等价 workflow

```text
initial periodic structure
        ↓
anisotropic NPT
        ↓
global NVTLangevin
        ↓
anisotropic NPH
```

该路线仅覆盖：

```text
NPT → thermostat → NPH
```

不声称制造 GPUMD `heat_lan` 的真实冷热两区。

### C3：当前输入 + MACE 本身不可用

切换到：

```text
同数量级
periodic
dense
MACE-stable
variable-cell
```

的材料体系。

这不改变 DomainParallel port 的主目标。

---

# 16. MACE 对当前输入的使用原则

在 NVIDIA upstream validation 阶段先验证选定 MACE checkpoint：

```text
Ga species accepted
In support recorded separately and not claimed from this input
energy finite
forces finite
stress finite
short NVE stable
short NPT stable
```

如果 Ga 输入失败：

> 不实现 NEP，不在本任务内训练/微调新势。

改用：

```text
同数量级、周期、dense、MACE 已验证稳定的材料体系
```

复制到约 388,800 atoms，保留：

```text
大体系
periodic
stress
variable cell
DomainParallel
memory pressure
```

即可完成移植验收。

---

# 17. 4 DCU 是首版代表性应用验收

原应用使用 4 GPU，因此本任务定义：

```text
2 DCU
= correctness / debugging target

4 DCU
= 第一版 representative application acceptance

8 DCU
= scaling / extension target
```

不要求 8 DCU 全部完成才能宣布第一阶段 feature 可用。

---

# 18. 显存是核心验收指标

原应用多卡的重要动机是分摊 388,800 原子体系的显存。

因此必须测：

```text
1 DCU peak memory
2 DCU peak memory/rank
4 DCU peak memory/rank
8 DCU peak memory/rank
```

还要记录：

```text
n_owned/rank
n_halo/rank
max/min owned imbalance
max/min halo imbalance
```

正确的空间 DomainParallel 应表现为：

- 主体 per-atom storage 随 rank 数分摊；
- 每个 rank 只持有 owned + 必要 halo；
- 不应在所有 rank 永久复制完整 388,800 原子主数据。

不设死板的 1/P 显存阈值，因为模型参数、workspace、halo 和 allocator overhead 不随原子数线性缩放。

---

# 19. 性能测试

正确性优先于性能。

固定代表体系测试：

```text
1 → 2 → 4 → 8 DCU
```

记录：

```text
ms/step
steps/s
speedup
parallel efficiency
peak memory/rank
neighbor time
halo exchange time
model forward time
global reduction time
migration time
other time
```

如果单卡无法承载 388,800 原子，可以：

- 用较小体系做 strong scaling；
- 对 388,800 原子只记录 2/4/8 卡；
- 明确写 `1-DCU OOM`，这本身也是 DomainParallel 的价值证据。

---

# 20. HIP Neighbor 的接入顺序

当前 HIP neighbor 只验证了窄范围：

```text
periodic
fixed-cell
Batch
full-list
MATRIX
skin=0
```

尚未支持：

```text
DomainParallel
variable-cell
```

所以首版代表性 workload 的 correctness 可以使用 Torch reference cell-list。

HIP neighbor 只在以下条件满足后接入：

```text
DomainParallel + Torch neighbor PASS
MACE + NVE PASS
MACE + NPT/NPH PASS
4-DCU representative workload PASS
```

HIP 接入分两阶段：

### H1：fixed-cell DomainParallel

先验证：

```text
owned + halo padded input
HIP topology == Torch topology
```

### H2：variable-cell

再实现/验证：

```text
cell change
→ metadata rebuild/update
→ neighbor rebuild
→ topology parity
```

HIP variable-cell 是后续性能优化，不应阻塞第一轮 DomainParallel correctness。

---

# 21. 测试矩阵

| 层级 | 体系 | 设备 | 关键检查 |
|---|---|---|---|
| runtime | tensor | 2/4/8 DCU | collective/P2P/DeviceMesh |
| partition | 32~1024 atoms | 2/4/8 | ownership/gather |
| halo | 32~1024 | 2/4/8 | forward/reverse/autograd |
| migration | artificial | 2/4 | atom continuity |
| neighbor | small periodic | 1/2/4 | topology parity |
| LJ/toy | small periodic | 1/2/4 | energy/force |
| MACE NVE | small/medium | 1/2/4/8 | energy/force/trajectory |
| MACE NPT | small/medium | 1/2/4 | stress/cell |
| MACE NPH | small/medium | 1/2/4 | stress/cell/enthalpy behavior |
| representative | ~388,800 | 2/4 | memory/stability |
| scaling | representative | 2/4/8 | speed/memory |

---

# 22. 数值验收原则

每一级都必须有单设备 oracle。

比较分四层，不能把长轨迹逐点重合作为唯一正确性标准：

1. **冻结构型 forward**：同一 positions/cell 下比较 neighbor、energy、按 global atom ID 排序
   的 per-atom force 和 stress；这是最硬的 1/2/4 卡数值门。
2. **单步 state transition**：比较一步 pre-update、model、post-update 后的 positions、velocities、
   cell 和 controller state。
3. **短轨迹**：固定输入、dtype、精度设置和确定性配置，比较少量步骤；记录误差增长，不因
   浮点归约顺序不同要求逐位一致。
4. **中长轨迹**：检查守恒量或 ensemble 统计、无 NaN/Inf、无 atom loss/deadlock/memory growth，
   不要求混沌 MD 轨迹长期逐点重合。

至少记录：

```text
energy abs/rel error
force max abs error
force RMS error
boundary-atom force error
stress max abs error
cell max abs error
atom count
global atom id set equality
```

不允许：

- 只检查程序不报错；
- 只比较 total energy；
- 只用 degenerate halo；
- 为了过测试随意放宽 tolerance；
- 用 MACE 结果和 NEP absolute energy 比较后判断 DomainParallel 对错。

tolerance 由 dtype、模型和单卡重复性证据决定，不在计划中提前硬编码。

NPH 的“enthalpy behavior”必须按锁定上游实现实际定义的 extended conserved quantity 或上游
测试 oracle 解释；不得未经推导就把普通 `E + PV` 当作各向异性 MTK 路径的严格守恒量。

---

# 23. 长时间稳定性

首版不需要照搬原 GPUMD 1,578,000 步。

分成：

### Correctness

```text
20~100 steps
```

### Stability

```text
1000~10000 steps
```

### Representative performance

```text
50~500 measured steps
```

只有后续真正做科学应用时，才考虑 100k~1.5M step 级运行。

稳定性测试至少记录：

- NaN/Inf；
- deadlock；
- atom count；
- memory growth；
- cell determinant；
- temperature；
- energy；
- stress；
- migration count。

---

# 24. 输出与 checkpoint

首版 DomainParallel 只要求足以做 correctness 和稳定性判断的输出：

```text
energy
temperature
stress/pressure
cell
step time
memory
n_owned/n_halo
migration count
periodic snapshot
```

完整 GPUMD 式：

```text
restart.xyz
compute.out
thermo.out
dump.xyz
```

格式兼容不是首版要求。

完整 checkpoint/restart 属于应用产品化 follow-up，不阻塞首个 DomainParallel feature。

因此首版完成后只能声明“短/中轨迹基础 DomainParallel MD 能力”。在补齐完整 checkpoint、
长时间恢复和失败重启前，不能声明已经具备百万步生产 MD 能力。

---

# 25. 建议提交拆分

```text
DP-GA0-UPSTREAM
  record locked NVIDIA upstream workload evidence

DP-GA0-PREFLIGHT
  record Hygon distributed preflight evidence

DP-GA0-MANIFEST
  lock input and MACE checkpoint artifacts

DP-GA1-RUNTIME
  Hygon distributed runtime compatibility

DP-GA2-SHARD
  ShardedBatch / partition DCU gates

DP-GA3-HALO
  halo + reverse + migration

DP-GA4-NEIGHBOR
  Torch reference cell-list under DD

DP-GA5-TOY
  analytic energy/force/stress parity

DP-GA6-MACE-NVE
  plain MACE NVE distributed

DP-GA7-SINGLE-DCU-NPT-NPH
  port existing upstream variable-cell NPT/NPH semantics to one DCU

DP-GA8-MACE-NPT
  anisotropic NPT distributed

DP-GA9-MACE-NPH
  anisotropic NPH distributed

DP-GA10-REPRESENTATIVE
  ~388800 atom / 4 DCU workload

DP-GA11-HIP-NEIGHBOR
  optional performance follow-up
```

---

# 26. 第一阶段 Definition of Done

第一阶段 feature 完成必须满足：

## 系统底座

```text
2/4 DCU process group stable
collective stable
P2P stable
DeviceMesh stable
variable/zero-split all_to_all stable
functional collective exact path PASS
PhysicsNeMo exact communication path PASS（若锁定上游运行时依赖该路径）
```

8 DCU 应尽量验证，但不是第一阶段 4-DCU 应用验收的硬 blocker。

## DomainParallel

```text
partition PASS
halo PASS
halo reverse PASS
migration PASS
neighbor parity PASS
analytic stress/virial parity PASS
```

## MLIP

```text
plain MACE
1 vs 2 DCU energy/force parity
4 DCU representative run
stress path works
```

## Dynamics

```text
NVE PASS
single-DCU anisotropic NPT/NPH PASS
anisotropic NPT PASS
anisotropic NPH PASS
```

## 代表性应用

```text
single periodic system
~388800 atoms preferred
4 DCU
real sharding confirmed
memory/rank recorded
no deadlock
no atom loss
stable short/medium trajectory
```

## 工程质量

```text
CPU/upstream regression not broken
NVIDIA path not broken
no silent CPU fallback
limitations documented
reports committed
```

---

# 27. 不应混淆的结论

本任务成功后可以说：

> 海光 DCU 上已经具备 nvalchemi 单大体系 DomainParallel 基础能力，并以 MACE + anisotropic NPT/NPH 和约 388,800 原子的代表性 workload 完成 4-DCU 端到端验证。

不能直接说：

> 已复现 Ga-In NEP 熔点应用。

也不能说：

> 当前 388,800 原子输入已经验证了 In 元素或混合 Ga-In 成分。

后者还需要：

```text
NEP4-ZBL
heat_lan/group thermostat
GPUMD MTTK exact semantics
long production run
scientific validation
```

另行完成。
