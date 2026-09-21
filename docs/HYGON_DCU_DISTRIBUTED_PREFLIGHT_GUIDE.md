# Hygon DCU Distributed Preflight Probe 指南

> 文件名：`docs/HYGON_DCU_DISTRIBUTED_PREFLIGHT_GUIDE.md`
>
> 目的：在修改 nvalchemi DomainParallel 前，独立确认当前海光 DCU 软硬件环境具备单大体系多卡所需的通信和运行时基础。
>
> 更新时间：2026-09-21

---

## 1. 原则

这是一项**系统先导 Probe**，不是 nvalchemi 功能开发。

Probe 阶段尽量只依赖：

```text
Python
DTK PyTorch
torch.distributed
操作系统/驱动/通信库
```

不要引入：

```text
MACE
nvalchemi DomainParallel
HALO
neighbor
NPT/NPH
```

目标是先回答：

> 当前海光单节点 8-DCU 环境，是否能够稳定完成 DomainParallel 所需的多进程启动、device collective、P2P、DeviceMesh、async communication 和长时间重复通信？

如果底层 Probe 不通过，禁止在 DomainParallel 代码里加入 workaround 掩盖系统问题。

本指南只验证上游现有 DomainParallel 所需的底座，不以 probe 名义开发上游不存在的功能。
发现某项原语在锁定上游从未使用或上游语义本身未成立时，只记录事实，不扩大产品范围。

---

## 1.1 与代表性应用输入的边界

当前 Ga-In 应用包已经确认存在一个可用的：

```text
restart.xyz
388800 atoms
positions + mass + velocities + group
periodic cell
```

该 restart 实际包含 388,800 个 Ga、0 个 In；这里的应用名称只描述来源工作流。组成事实不影响
本 preflight，因为本阶段不读取该文件。

并且它将作为后续 NVIDIA/Hygon DomainParallel 的首选代表性输入。

但是：

> **本 preflight 不读取、不运行、不依赖这个文件。**

原因是本阶段只验证：

```text
torch.distributed
collective
P2P
async
DeviceMesh
memory/runtime
```

如果 preflight 需要 MACE、neighbor、Ga-In 或 `restart.xyz` 才能 PASS，说明测试层次已经混淆。

因此应用输入的 parser、velocity unit、MACE 和 NPH 验证全部留在后续阶段。

---

# 2. 首版测试范围

默认硬件目标：

```text
single node
2 DCU
4 DCU
8 DCU
```

含义：

> 同一台物理服务器内部，分别使用 2、4、8 张 DCU。

首版不把跨多台服务器作为硬验收条件。

multi-node 可以额外探测，但结果单列。

所有多卡 probe 必须按 `2 → 4 → 8` 递增。每一级先只读确认设备占用和拓扑，再使用 timeout
限时运行；出现 device reset、XID/驱动错误、通信 hang 或首个不可恢复错误时立即停止扩大卡数，
保存日志，不循环重试、不杀其他用户进程。

---

# 3. 建议目录

在项目中建议新增：

```text
probes/distributed/
├── README.md
├── probe_env.py
├── probe_device.py
├── probe_process_group.py
├── probe_collectives.py
├── probe_all_to_all_v.py
├── probe_p2p.py
├── probe_async.py
├── probe_devicemesh.py
├── probe_functional_collectives.py
├── probe_physicsnemo_compat.py
├── probe_bandwidth.py
├── probe_stability.py
└── run_preflight.sh
```

原始输出：

```text
artifacts/distributed-preflight/<timestamp>/
```

脱敏总结：

```text
reports/dcu-distributed-preflight.md
```

本指南只定义 probe contract；具体脚本可以按此另行实现。

---

# 4. P0：环境画像

## 4.1 必须记录

```text
hostname
OS
kernel
Python
PyTorch
torch.version.hip
torch.version.cuda
DTK version
driver/runtime
visible devices
device count
device names
device total memory
HIP_VISIBLE_DEVICES
torch.distributed.is_available()
dist.is_nccl_available()
dist.is_gloo_available()
PhysicsNeMo 是否已安装（仅记录，不要求）
MPI 是否已安装
通信库版本（可发现时）
```

同时保存可用厂商工具输出，例如系统实际提供的：

```text
device status
topology
PCIe/UBB/NUMA relation
NIC relation
```

不强制某一个特定工具名；以当前机器实际安装工具为准。

## 4.2 Python 基础信息示例

```bash
python - <<'PY'
import os
import torch
import torch.distributed as dist

print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("torch.version.cuda:", torch.version.cuda)
print("cuda_api_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory)

print("dist_available:", dist.is_available())
print("nccl_available:", dist.is_nccl_available())
print("gloo_available:", dist.is_gloo_available())
print("HIP_VISIBLE_DEVICES:", os.getenv("HIP_VISIBLE_DEVICES"))
PY
```

注意：

> Hygon DTK PyTorch 可能沿用 `torch.cuda` API 名称。`torch.cuda` 这个 Python 命名本身不能用于判断设备是不是 NVIDIA。

必须结合：

```text
torch.version.hip
真实 device name
DTK/runtime
```

判断。

## 4.3 P0 PASS

- 至少 2 DCU 可枚举；
- 每张卡可以分配 tensor；
- 简单 elementwise / matmul 可以运行；
- 无隐式 CPU fallback。

---

# 5. P1：单卡 device tensor smoke

每张可见 DCU 逐张测试：

```text
allocation
H2D
D2H
elementwise
matmul
synchronize
memory free/reuse
```

建议至少：

```text
float32
float64
int64
```

并记录：

```text
allocated memory
reserved memory
```

### PASS

每张卡单独运行正常。

---

# 6. P2：torchrun 和 rank → DCU 映射

## 6.1 运行规模

```bash
HIP_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 probes/distributed/probe_process_group.py
```

再：

```text
4 ranks
8 ranks
```

## 6.2 每 rank 输出

```text
RANK
LOCAL_RANK
WORLD_SIZE
selected logical device
device name
tensor.device
PID
```

每个 rank 做一次独立 device matmul。

## 6.3 PASS

- 2/4/8 ranks 都能启动和干净退出；
- local rank 与可见 DCU 一一对应；
- 不出现所有进程落在同一张卡；
- 不依赖人工逐 rank 修改脚本。

---

# 7. P3：确定实际可用的 ProcessGroup backend

禁止先假设 backend 字符串一定是：

```text
nccl
rccl
```

应以当前 DTK PyTorch 实测为准。

Probe 流程：

1. 打印 backend capability；
2. 使用当前平台官方/实际支持的 device backend 初始化；
3. 打印 `dist.get_backend()`；
4. 对 DCU tensor 运行 collective。

如果 DTK 通过 NCCL-compatible API 暴露 RCCL，记录事实即可，不要因为字符串叫 `nccl` 就写成 NVIDIA NCCL。

### PASS

ProcessGroup 可以直接处理 DCU tensor。

---

# 8. P4：Collective correctness

DomainParallel 会依赖多种 collective。

必须测试：

```text
barrier
broadcast
all_reduce
all_gather
all_gather_into_tensor
all_to_all_single
reduce_scatter / reduce_scatter_tensor（当前 PyTorch 支持时）
```

`all_to_all_single` 必须同时覆盖等分和变长 split。HALO/migration 的关键形态不是普通等分
all-to-all，而是每个 rank 发送/接收行数不同，且允许某些 peer 为 0：

```text
input_split_sizes != output_split_sizes
different split vector on every rank
zero send to one or more peers
zero receive from one or more peers
one heavily imbalanced destination
forward exchange followed by reverse exchange
```

## 8.1 dtype

至少：

```text
float32
float64
int64
```

## 8.2 payload

至少：

```text
4 KiB
1 MiB
64 MiB
```

性能 probe 再扩大。

## 8.3 correctness pattern

例如 rank `r` 输入：

```text
tensor filled with r + 1
```

all_reduce 后必须能精确预测结果。

all_gather/all_to_all 同样使用可验证 pattern。

## 8.4 PASS

关键硬门槛：

```text
all_reduce PASS
all_gather PASS
all_to_all_single PASS
variable/zero-split all_to_all_single PASS
```

且直接作用于 DCU tensor。

如果必须：

```text
DCU → CPU → collective → DCU
```

才工作，则标记：

```text
FUNCTIONAL = PARTIAL
PRODUCTION_READY = NO
```

---

# 9. P5：P2P correctness

HALO 和 migration 对点对点通信非常敏感。

必须测试：

```text
dist.isend
dist.irecv
dist.batch_isend_irecv
```

拓扑：

```text
rank 0 <-> rank 1
ring: rank i -> rank (i+1)
bidirectional pairs
```

先根据 P14 拓扑选择一对已知相邻/同 NUMA 卡做 2-rank 测试，再扩到 ring 和 4/8 rank。每轮
命令必须有 timeout，并在前后保存设备健康状态。直接 P2P 导致设备 reset 或驱动错误时立即停止
该拓扑组合，不用环境变量反复试错掩盖问题。

payload：

```text
float32 positions/features
float64 positions/features
int64 indices/routing
```

覆盖：

```text
fixed size
different size per rank
zero-length where API permits
asymmetric send/recv
```

### PASS

- data exact；
- request.wait() 正常；
- 2/4/8 rank 无 hang；
- 1000 次连续小 P2P 无错误。

---

# 10. P6：Async collective

上游 migration 使用 async consensus，并希望隐藏通信延迟。

必须验证：

```python
work = dist.all_reduce(..., async_op=True)
# do independent work
work.wait()
```

测试：

- handle 生命周期；
- 多次 outstanding operation；
- wait 顺序；
- tensor 在 wait 前后是否符合语义；
- 退出前清理。

### PASS

连续重复无 hang、无 use-after-free、无异常同步错误。

---

# 11. P7：DeviceMesh 和 subgroup

上游 DomainParallel 使用：

```text
torch.distributed.device_mesh.DeviceMesh
```

测试：

```text
1D mesh
subgroup
mesh group
rank lookup
subgroup all_reduce
```

8 卡时建议额外测试：

```text
2D mesh / subgroup
```

即使首版 DomainParallel 用 1D，也能提前发现 DeviceMesh backend 假设问题。

### PASS

- DeviceMesh 可绑定当前 DCU device API；
- group 正确；
- subgroup collective 正确。

---

# 12. P8：functional collectives

当前上游 distributed `_core` 会使用：

```text
torch.distributed._functional_collectives
```

因此不能只测普通 `torch.distributed`。

Probe 应至少确认当前 DTK PyTorch 中：

- 模块可导入；
- 相关 primitive 可以在 DCU tensor 上执行；
- world size 2/4 正确；
- 如当前实现需要 compile/autograd，记录支持程度。

必须按上游实际调用形态验证：

```text
torch.distributed._functional_collectives.all_to_all_single
explicit send/recv split lists
wait_tensor
forward + reverse exchange
world size 2/4
zero and imbalanced splits
```

普通 `dist.all_to_all_single` 通过不能替代本项。

如果功能缺失：

```text
P8 = CONDITIONAL
```

并作为正式 DomainParallel port 的明确 runtime gap。

不要静默用普通 collective 替换后宣称 upstream-compatible。

## 12.1 P8-B：PhysicsNeMo exact compatibility（条件门）

纯 PyTorch preflight 通过后，如果目标 Hygon 环境已经能够安装锁定上游要求的 PhysicsNeMo，
额外直接验证：

```text
physicsnemo.distributed.DistributedManager
physicsnemo.distributed.utils.indexed_all_to_all_v_wrapper
```

输入必须覆盖不等长、0-size、int64 routing 和 FP32/FP64 payload。PhysicsNeMo 在当前环境不可
安装时，本项记为 `RUNTIME_GAP`，不伪造替代结果；它不推翻纯 PyTorch 平台底座结论，但必须在
正式 Phase 1 完成前解决或由已有上游兼容实现显式替代。

---

# 13. P9：Autograd-sensitive communication smoke

HALO 路径需要保证跨 rank 数据参与模型计算后，owner 可以得到正确贡献。

Preflight 不实现 nvalchemi HALO，但可以做最小 autograd smoke：

```text
local tensor
→ communication/gather
→ differentiable local computation
→ backward
```

若当前 PyTorch collective 本身不是 autograd-aware，不直接判定平台失败；应记录：

```text
native autograd collective supported: yes/no
```

DomainParallel 后续可能使用自己的 halo/autograd adapter。

这一项是信息项，不是首要 hard gate。

---

# 14. P10：带宽 / latency 画像

测试：

```text
all_reduce
all_gather
all_to_all_single
P2P ping-pong
```

world size：

```text
2
4
8
```

payload：

```text
1 KiB
4 KiB
64 KiB
1 MiB
16 MiB
64 MiB
256 MiB（资源允许）
```

每种至少：

```text
warmup
20~100 measured iterations
```

计时必须在 warmup 后进行，并在区间边界做明确同步。GPU/DCU 异步 API 的 Python 提交时间不
等于通信完成时间；优先使用当前 PyTorch/DTK 可用的 device event，另记录同步后的 wall-clock。

输出：

```text
median latency
p90 latency
effective bandwidth
min/max
```

这一项不设绝对 GB/s PASS 阈值。

主要找：

- 某一 rank pair 异常；
- 4/8 卡突然退化；
- host staging；
- 拓扑瓶颈。

---

# 15. P11：通信 + 计算并行 smoke

真实 DomainParallel step 同时包含：

```text
communication
+
model compute
```

建议测试：

```text
async all_reduce
+
independent matmul/kernel
+
wait
```

记录：

```text
serial time
overlap time
```

`async_op=True` 只证明返回了 handle，不自动证明通信与计算发生重叠。比较 serial/overlap 前后
都要同步，并记录 matmul 是否使用独立 stream；否则该项只记为 async correctness，不给 overlap
结论。

目的不是追求优化数字，而是发现：

- async 实际全同步；
- stream/synchronization 异常；
- runtime deadlock。

---

# 16. P12：长时间稳定性

大体系 MD 不是一次 collective。

最低建议：

```text
10000 small collectives
1000 medium collectives
100 large collectives
```

混合：

```text
all_reduce
all_gather
all_to_all
P2P
async all_reduce
```

记录：

```text
hang
timeout
driver/runtime error
process exit
memory growth
```

### PASS

- 无 hang；
- 无 rank 丢失；
- 无持续显存增长；
- 重复结果正确。

---

# 17. P13：显存行为

每个 rank 记录：

```text
memory_allocated
max_memory_allocated
memory_reserved
max_memory_reserved
```

在大 collective 前后比较。

PyTorch allocator 数字不包含所有通信库、driver 和上下文分配。资源允许时同时采样厂商设备
监控工具的进程显存峰值，并把 allocator peak 与设备侧 peak 分列，不能相互替代。

重点识别：

```text
all_gather/all_to_all 临时 buffer
是否造成异常倍增
是否释放
```

这会影响后续 388,800 atom DomainParallel 的真实容量。

---

# 18. P14：单节点拓扑记录

保存：

```text
DCU <-> DCU
DCU <-> CPU NUMA
DCU <-> NIC
```

的拓扑信息。

如果机器存在不同 DCU 互联路径，带宽测试结果必须和物理拓扑对应。

不要仅给平均值。

---

# 19. Multi-node 可选 Probe

首版不是硬要求。

如果已有两台同型节点，可以额外测试：

```text
2 nodes × 1 DCU
2 nodes × 2 DCU
```

重点：

```text
跨节点 all_reduce
P2P
IB/RDMA
GPU/DCU-aware MPI/RCCL
NIC topology
```

必须单独标记：

```text
MULTI_NODE_EXPERIMENTAL
```

不和首版单节点验收混在一起。

---

# 20. 建议总执行顺序

严格：

```text
P0 environment
 ↓
P1 single device
 ↓
P14 topology record
 ↓
P2 torchrun mapping
 ↓
P3 backend
 ↓
P4 collectives
 ↓
P5 P2P
 ↓
P6 async
 ↓
P7 DeviceMesh
 ↓
P8 functional collectives
 ↓
P8-B PhysicsNeMo exact compatibility（可安装时）
 ↓
P9 autograd-sensitive smoke
 ↓
P10 performance
 ↓
P11 communication + compute
 ↓
P12 stability
 ↓
P13 memory
```

P4/P5 不通过时，不要继续讨论 nvalchemi HALO。

---

# 21. 结果分类

## GO

以下必须通过：

```text
2/4 DCU process launch
device all_reduce
device all_gather
device all_to_all_single
device variable/zero-split all_to_all_single
device P2P
async all_reduce
DeviceMesh
functional all_to_all + wait_tensor
long-repeat stability
```

8 DCU 也应测试；若只有 8 卡特定问题，可以先判：

```text
CONDITIONAL_GO
```

并限制首个 DomainParallel milestone 到 4 DCU。

## CONDITIONAL_GO

例如：

```text
collective/P2P 正常
但 DeviceMesh 有 Python compatibility issue
```

或：

```text
普通 collectives 正常
functional_collectives 某路径不支持
```

这些可以进入 Phase 1 runtime compatibility，但必须有明确 issue。

## NO_GO

如果：

```text
device all_reduce 不可靠
P2P 经常 hang
4-rank process group 不稳定
长期测试持续 runtime error
```

则先解决 DTK/驱动/通信栈。

不开始 DomainParallel 正式代码修改。

---

# 22. 推荐报告格式

```markdown
# Hygon DCU Distributed Preflight

## Hardware
- node:
- DCU:
- count:
- topology:

## Software
- OS:
- kernel:
- DTK:
- PyTorch:
- torch.version.hip:
- backend:

## P0-P14 Results
| Probe | 2 DCU | 4 DCU | 8 DCU | Notes |
|---|---|---|---|---|

## Commands and raw evidence
- exact command:
- visible devices:
- timeout:
- exit code:
- artifact path:
- pre/post device health:

## Collective Performance
...

## P2P
...

## Stability
...

## Known Issues
...

## Runtime gaps
- PhysicsNeMo:
- indexed all-to-all-v:
- functional collectives:

## Decision
HYGON_DISTRIBUTED_BASE = GO / CONDITIONAL_GO / NO_GO
```

---

# 23. 对 DomainParallel 开发者的交付内容

Probe 完成后必须把以下信息交给 DomainParallel 开发者：

```text
实际 process-group backend
实际 device type/API
rank-device mapping
DeviceMesh 是否原生可用
all_to_all_single 是否稳定
P2P API 是否稳定
functional_collectives 支持情况
variable/zero-split all_to_all 行为
PhysicsNeMo exact primitive 支持情况或 runtime gap
async collective 行为
2/4/8 卡 topology/bandwidth
已知 runtime workaround
```

这样开发者不需要再在 `particle_halo.py` 里猜环境问题。

---

# 24. 本 Probe 不验证什么

即使全部 PASS，也不能说明：

```text
DomainParallel 已经支持 Hygon
MACE 已经支持多 DCU
HALO 已经正确
neighbor 已经正确
NPT/NPH 已经正确
```

它只证明：

> 当前 Hygon DCU 软硬件系统具备实现这些功能所需的基础分布式通信能力。
