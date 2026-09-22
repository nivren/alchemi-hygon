# Hygon DCU Distributed Preflight

日期：2026-09-21--2026-09-22 UTC
测试分支：`codex/test-distributed-md-preflight`
测试基线：`e9796d4` (`develop`)
依据：[`docs/HYGON_DCU_DISTRIBUTED_PREFLIGHT_GUIDE.md`](../docs/HYGON_DCU_DISTRIBUTED_PREFLIGHT_GUIDE.md)

## 范围

本次只做系统级 distributed preflight，不读取 `restart.xyz`，不引入 MACE、neighbor、HALO
或 DomainParallel，也没有修改产品实现。目标是先确认多进程启动是否能进入 rank 脚本。

## P0/P14 静态环境与资源

| 项目 | 观测 |
|---|---|
| host / OS / kernel | `dcu2` / Ubuntu 22.04.5 / Linux 5.15.0-136-generic x86_64 |
| DCU | 8 × BW200 / UBB BW1000 |
| VRAM | 每卡 65520 MiB |
| driver | `6.3.30-V1.4.1a` |
| DCU topology | 初始 `hy-smi --showtopotype`：任意卡对均为 `HSW` |
| 初始资源 | 初始 `hy-smi --showmeminfo vram`：8 卡各约 2 MiB used |
| installer hlink topology | `/var/log/rock-kernel/install-2.log`：`7 HSW + 8 HCU -> topo 10`；最终选择 `OAM_7HSW_8HCU` |
| active HFM topology | `/etc/hfm/hys.cfg`：`config = OAM_7HSW_8HCU`，preset 为 `OAM_7HSW_8HCU_0_runpkg_link_cfg_preset_jx_md5_937fd.ini` |
| project Torch | `2.9.0`, `torch.version.hip=6.3.26093` |
| project distributed | `is_available=True`, `nccl_available=True`, `gloo_available=True`（沙箱导入信息） |

沙箱内的 `/dev/kfd`/`/dev/dri` 不可见，因此沙箱中的 `device_count=0` 不作为 DCU 能力结论。
本次设备信息和健康检查均通过设备可见的主机权限命令获取。

## P1 single-device tensor smoke

新增 [`probes/distributed/probe_device.py`](../probes/distributed/probe_device.py)，在设备可见
主机上逐张隔离运行：

```bash
source /opt/dtk-26.04/env.sh
for card in 0 1 2 3 4 5 6 7; do
  HIP_VISIBLE_DEVICES=$card OMP_NUM_THREADS=1 timeout 90 \
    .venv/bin/python probes/distributed/probe_device.py --device 0
done
```

8 张物理卡均返回 `RC=0`，设备名均为 `BW1000 64G`。每张卡的 `P1_RESULT` 均报告：

```text
fp32/fp64/int64: allocation, H2D, elementwise, D2H = PASS
fp32/fp64 matmul = PASS
synchronize = PASS
allocator release/reuse = PASS
```

每张卡的运行后 allocator 快照一致：`allocated=33554432`、`reserved=33554432`，
复用后没有继续增长；该数值是当前 DTK PyTorch allocator 的运行时保留基线，不将其解释为
长期 soak 或生产容量证据。当前 P1 单卡矩阵已完成并通过，覆盖本指南要求的单卡基础路径。

## P2 2-rank process launch

使用仓库已有的最小 probe `probes/distributed_probe.py`，未进入 DomainParallel：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  timeout 90 .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed_probe.py
```

关键原始结果：

```text
Could not open /var/log/hylog/.
Fatal Python error: Segmentation fault
  ... torch/cuda/__init__.py, line 182 in is_available
  ... torch/distributed/launcher/api.py, line 112 in __post_init__
  ... torch/distributed/run.py, line 847 in config_from_args
timeout: the monitored command dumped core
PROBE_RC=139
```

段错误发生在 `torchrun` launcher 调用 `torch.cuda.is_available()` 时，尚未产生 rank 输出，
也未确认 `init_process_group`、device collective 或 P2P。故障后没有发现残留 `torchrun`/
`distributed_probe` 进程。

## 恢复后的分层诊断

随后在 `hy-smi` 恢复正常的主机状态下重新验证：

| 层次 | 结果 |
|---|---|
| 直接 Python `torch.cuda.is_available()` | `HIP_VISIBLE_DEVICES=0` 和 `0,1` 均 `True`；device count 分别为 1、2；退出码 0 |
| `torchrun --nproc-per-node=1` + 单卡 Torch probe | 通过，退出码 0；FP64 二阶梯度、FFT 和张量检查通过 |
| `torchrun --nproc-per-node=2 --no-python /bin/echo` | 两子进程启动并退出 0 |
| 2-rank process group | NCCL/RCCL communicator 初始化完成 |
| 2-rank 1-element all-reduce | 两 rank 均在 45 秒 watchdog 后超时；`DISTRIBUTED_PROBE_RC=124` |

静态审查 Torch 2.9 launcher 源码确认：显式 `--nproc-per-node=2` 本身不需要通过 device count
推导进程数，但 `LaunchConfig.__post_init__` 在未指定 `--numa-binding` 且
`torch.cuda.is_available()` 为真、`device_count() == nproc_per_node` 时，会再次调用
`torch.cuda.is_available()` 并自动推导 NUMA 选项。首次段错误的 traceback 正落在该调用；
因此 `--numa-binding node` 是恢复后可验证的诊断变量，不是当前已验证的 workaround。

all-reduce 超时后的一次受控诊断设置为 `NCCL_P2P_DISABLE=1`、默认 SHM，但两个子进程在
`torch.cuda.set_device()` 即报 `No HIP GPUs are available`，随后 `hy-smi` 再次不可初始化。
由于设备在 all-reduce 超时后已经处于异常状态，该 SHM 对照被标记为**受污染**，不能用于
判断 P2P transport 是否是根因。

## 故障后健康检查

故障后只读检查得到：

```text
hy-smi --showproductname --showdriverversion --showmeminfo vram --showtopotype
Error: No device available, no device found or initialization failed, exiting.
```

`/dev/kfd` 和 `/dev/dri/card0..card8` 节点仍存在；本次 `dmesg` 关键词筛选未发现可归因的
`HCU`/`XID`/`reset`/`VMFault` 行。没有执行设备 reset、驱动操作、环境变量试错或自动重试。

## P0-P14 结果矩阵

| Probe | 2 DCU | 4 DCU | 8 DCU | 说明 |
|---|---|---|---|---|
| P0 environment | PASS | — | — | 静态环境、8 卡枚举和 P1 基础 device tensor smoke 均通过 |
| P1 single device | PASS (HCU0--7) | — | — | FP32/FP64/INT64、H2D/D2H、elementwise、FP32/FP64 matmul、同步、释放复用均通过 |
| P14 topology | PASS | PASS | PASS | 2026-09-22 重新记录 HSW、NUMA、PCIe/NIC；非压力测试 |
| P2 torchrun mapping | PASS | PASS* | PASS* | 2-rank 专项与后续 P5--P13 的 4/8-rank torchrun 均干净退出并完成 rank/device 绑定；初次 launcher 段错误保留为历史未归因记录 |
| P3 backend | PASS | PASS | PASS | 实测 backend 为 `nccl`（DTK RCCL-compatible API）；2/4/8-rank communicator 与 DCU tensor collective 均完成 |
| P4 collectives | PASS | PASS* | PASS* | 2-rank 完整 P4 矩阵通过；4/8-rank 的 collective 正确性由 P5--P13 分项矩阵覆盖 |
| P5 P2P | PASS | PASS | PASS | 2/4/8 rank 全部 P2P case 通过；覆盖 4 KiB/1 MiB、fp32/fp64/int64、zero/variable/repeated |
| P6 Async collective | PASS | PASS | PASS | async all-reduce、独立设备计算、多个 outstanding、正序/逆序 wait、100 次重复均通过 |
| P7 DeviceMesh/subgroup | PASS | PASS | PASS | 1D mesh、rank lookup、mesh group、subgroup；4/8 rank 另测 2D row/col subgroup |
| P8 Functional collectives | PASS | PASS | PASS | `_functional_collectives`、functional `all_to_all_single` + `wait_tensor`；显式 zero/imbalanced split 的 forward/reverse 通过 |
| P9 Autograd-sensitive communication | PASS | PASS | PASS | direct `dist.all_gather` 前向通过但 native autograd=no；functional autograd gather 前向/反向通过 |
| P10 Performance baseline | PASS | PASS | PASS | 4 类通信、6 个 payload、10 warmup + 20 samples；P12 stability、P13 memory 另列 |
| P11 Communication/compute overlap | PASS | PASS | PASS | async/stream correctness 与同步完成通过；未观察到稳定的 all-rank overlap 收益 |
| P12 Stability | PASS | PASS | PASS | 10,000/1,000/100 mixed operations；阶段内显存 trace 稳定 |
| P13 Memory behavior | PASS | PASS | PASS | 64 MiB FP32；all_gather 瞬时 peak 随 world size 增长，释放后无 persistent allocated growth |

`*` 表示后续对应 rank 的专项探针提供了启动、rank/device 绑定和 DCU tensor 执行证据，
不是声称原始 P2/P4 脚本在每个规模上重复执行了完全相同的矩阵。

## Decision

```text
HYGON_DISTRIBUTED_BASE = GO (current post-reboot preflight)
```

该结论针对当前重启后的单节点分布式 preflight，不是 DomainParallel 生产验收结论。重启前
曾发生的 launcher/RCCL 异常保留在下方历史诊断章节，不参与当前 `GO` 决策，也不能据此归因
于新驱动；用户已确认安装新驱动前重启后的测试就已恢复。

## 当前收口

P0--P14 的单节点通信先导范围已经完成：P1 已补齐 8 张卡的完整单卡 smoke，P3/P4 的早期
保守标签已按 post-reboot 和后续 2/4/8-rank 证据修正。剩余未验证范围是 PhysicsNeMo exact
compatibility、HALO/owner contribution、DomainParallel 端到端语义、生产规模和多节点；这些
不属于本次 preflight 分支的补测内容。

历史文档中 2026-09-06 的双卡 all-reduce/P2P 通过记录仅作背景，不替代本次当前运行证据。

## Historical pre-reboot direct HIP/RCCL diagnostic

在 `hy-smi` 恢复可用后，使用 host DTK 26.04 环境编译并运行了两个不依赖 PyTorch、
`torchrun` 或 MPI 的 probe：

- [`probes/distributed/hip_runtime_probe.cpp`](../probes/distributed/hip_runtime_probe.cpp)：覆盖
  `hipGetDeviceCount`、`hipSetDevice`、分配、H2D/D2H、同步和释放。
- [`probes/distributed/rccl_allreduce_probe.cpp`](../probes/distributed/rccl_allreduce_probe.cpp)：
  两个普通进程通过临时文件交换 RCCL unique ID，直接调用 `ncclCommInitRank` 和
  `ncclAllReduce`。
- [`probes/distributed/README.md`](../probes/distributed/README.md)：编译、限时运行、分层判定和厂家取证步骤。

编译结果：HIP probe 和 RCCL probe 均返回 `0`，目标为 `gfx936`。

### HIP runtime

分别用 `HIP_VISIBLE_DEVICES=0` 和 `HIP_VISIBLE_DEVICES=1` 运行，均返回 `0`：

```text
hip_device_count=1
requested_device=0
device_name=BW200, UBB BW1000
round_trip_value=1
status=passed
```

这证明本次主机状态下，至少 HCU0/HCU1 的基础 HIP 设备枚举、显存分配、H2D/D2H 和同步
路径可以完成；它不证明跨卡通信可用。

### Direct RCCL two-rank all-reduce

使用 `HIP_VISIBLE_DEVICES=0,1`、`WORLD_SIZE=2`、每个 rank 一个普通进程运行。两 rank
均完成 `ncclCommInitRank`，但在进入单元素 `ncclAllReduce` 后没有返回：

```text
rank=0 communicator=initialized
rank=0 before_ncclAllReduce
rank=1 communicator=initialized
rank=1 before_ncclAllReduce
# 两个日志均没有 after_ncclAllReduce
rank0_rc=124 rank1_rc=124
```

RCCL INFO 日志显示该路径选择了 `via P2P/IPC`，拓扑日志记录了 XGMI link；因此这是一个
脱离 PyTorch/ProcessGroup/torchrun 后仍能复现的 RCCL collective hang。超时后本次
`hy-smi` 仍然可用，故本次 direct repro 没有观察到先前 PyTorch 测试中的后续设备丢失。
原始 rank 日志位于临时目录 `/tmp/alchemi-rccl-probe.aoOFjv/`，不是仓库证据文件。

日志同时给出两个需要系统管理员/厂家确认的环境警告：未检测到内核命令行
`iommu=pt`，以及未设置 `HSA_FORCE_FINE_GRAIN_PCIE=1`。本次没有修改这些系统参数或环境变量，
所以不能据此断言根因，也不能把它们当作已验证 workaround。

### Layered conclusion

```text
HIP runtime on HCU0/HCU1       = PASS
RCCL communicator initialization = PASS
RCCL 1-element all-reduce       = HANG / TIMEOUT (124)
PyTorch/torchrun                 = not the only reproduction layer
HYGON_DISTRIBUTED_BASE          = NO_GO (historical pre-reboot snapshot; not current decision)
```

这已经足以提供厂家一个最小、非 PyTorch 的复现方式，重点排查 RCCL P2P/IPC/XGMI 数据路径、
HIP runtime、驱动/固件以及主机 IOMMU/PCIe 配置。它仍不是单独的“硬件故障已定论”：
communicator 初始化通过，且当前权限无法读取 `dmesg`（`Operation not permitted`），所以还
缺少驱动/固件错误码和硬件复位记录。

### HCU6/HCU7 confirmation

随后将相同探针切换到物理 HCU6/HCU7。注意 `HIP_VISIBLE_DEVICES=6` 或 `7` 后，程序内
逻辑设备号必须传 `0`；修正该参数后，两张卡的纯 HIP probe 均返回 `0`，枚举、分配、
H2D/D2H 和同步通过。

独立 RCCL 双进程运行使用 `HIP_VISIBLE_DEVICES=6,7`：

```text
rank=0 communicator=initialized
rank=0 before_ncclAllReduce
rank=1 communicator=initialized
rank=1 before_ncclAllReduce
# 两个日志均没有 after_ncclAllReduce
```

RCCL 日志确认 rank0/rank1 分别映射到物理 `nvmlDev 6`/`7`，并选择 `via P2P/IPC` 与
`XGMI`。探针未能正常完成超时收尾，清理本次探针后 `hy-smi` 出现：

```text
Open mkfd failed
Expected integer value from monitor, but got ""
Error: No device available, no device found or initialization failed, exiting.
```

在故障前的只读检查中，HCU6/HCU7 各约 448 MiB 使用量；故障后监控输出为 `N/A`。这使
HCU0/HCU1 与 HCU6/HCU7 两个不同双卡组合都在 direct RCCL 的最小 all-reduce 数据路径上
复现，且伴随设备管理接口丢失。该结果显著提高了驱动/固件、RCCL P2P/IPC/XGMI 或共享
硬件路径问题的可能性，但仍不能仅凭用户态日志完成硬件归因；当前应由管理员恢复设备并
从宿主机权限采集 `dmesg`/`journalctl -k`、驱动和固件信息。原始日志目录为
`/tmp/alchemi-rccl-probe-67.C6QvIq/`。

## Installer topology confirmation

本机只读确认到 DMI 为 `Suma / X7950H0`、主板 `62DC24X_U8`。安装日志记录：

```text
HSW(7 devices)
HCU(8 devices)
selected 7+8->10
user select (OAM_7HSW_8HCU) ... 10:OAM_7HSW_8HCU ... 11:OAM508_7HSW_8HCU ... ->10
```

因此本节点的正确显式参数是 `-t 10`；`-t 11`（OAM508 专用变体）不适用。由于安装器
已经自动识别并落盘为 `OAM_7HSW_8HCU`，当前 RCCL/设备故障没有证据表明是“安装时未显式
指定 `-t`”造成的。后续若在同一硬件上重新部署，可显式传 `-t 10`，但在当前设备不可用
状态下不应直接重装或更新固件。

## Post-reboot torchrun revalidation

重启后、安装新驱动之前，用户报告同一 probe 已能正确返回；该时间线不是本次独立采集的
旧驱动 A/B 实验，因此记录为用户提供的因果线索。随后在当前重启后的环境中重新执行：

```text
Driver Version: 6.3.31-V1.5.6
Card Series: BW1000 64G
direct torch: is_available=True, device_count=2
torchrun: TORCHRUN_RC=0
rank 0: all_reduce=3.0, received=8.0, status=passed
rank 1: all_reduce=3.0, received=7.0, status=passed
hy-smi after: HY_SMI_AFTER_RC=0; all 8 cards about 2 MiB used
```

原始输出目录：`/tmp/torchrun-reboot-preflight.21hGna/`。本次结果证明重启后的当前状态可以
完成 Torch ProcessGroup、RCCL all-reduce 和 P2P send/recv；结合“旧驱动、重启后、安装新
驱动前已通过”的用户时间线，当前最稳健的结论是：新驱动不是恢复成功的必要条件，故障与
重启前后的设备/驱动运行时状态有关，不能宣称由驱动升级修复。

## Post-reboot P4 collective correctness gate

为补齐指南要求的 P4 contract，新增 [`probes/distributed/probe_collectives.py`](../probes/distributed/probe_collectives.py)，
不读取应用输入，也不依赖 MACE、neighbor 或 DomainParallel。探针在 DCU tensor 上直接验证：

```text
broadcast
all_reduce
all_gather
all_gather_into_tensor
all_to_all_single (equal split)
all_to_all_single (variable/zero split)
reduce_scatter_tensor
```

本次使用重启后的 dcu2，`HIP_VISIBLE_DEVICES=0,1`，两 rank，未设置 RCCL transport workaround。
三种 dtype 和三档 payload 的所有项目均通过：

```text
dtype: fp32, fp64, int64
payload: 4 KiB, 1 MiB, 64 MiB
torchrun rc: 0
backend requested/actual: nccl/nccl
device: BW1000 64G
hy-smi before/after: rc=0, 8 卡均正常
```

原始输出目录：`/tmp/p4-collectives.It4SoS/`。这是当前节点 post-reboot 的 **P4/2-rank
narrow PASS**；该独立 P4 证据不能覆盖重启前 Torch/RCCL hang，也不能单独推出 4/8-rank、P2P
1000 次、async、DeviceMesh、长稳或生产 DomainParallel 已通过；后续 P5/P6/P7/P8 结果见本报告后续章节。

## Post-reboot P5 P2P gate: first bounded run

新增 [`probes/distributed/probe_p2p.py`](../probes/distributed/probe_p2p.py) 后，按默认配置在
重启后的 dcu2 上执行一次 2-rank、180 秒有界测试，未设置 transport workaround：

```text
payloads: 4 KiB, 1 MiB
dtype: fp32, fp64, int64
torchrun rc: 124 (timeout)
hy-smi before/after: rc=0, 8 卡均正常
```

日志显示 ProcessGroup 初始化完成，并在 rank 0/1 出现两次 unbatched P2P communicator 初始化
提示，随后直到超时，没有 rank summary 或 numerical PASS。由于本版首次运行没有逐 case 起止
标记，不能仅凭日志把失败精确归因到某个具体函数内部；代码已补充 flushed `case_start`/
`case_pass` 事件以及显式 `device_id`，以便下一次在设备健康状态重新确认后做单次定位运行。

随后将探针增加 `--case` 选择，可单独运行 `pair_isend_irecv`、`ring_isend_irecv`、
`ring_variable_asymmetric`、`bidirectional_pair_batch`、`zero_length_ring` 或
`repeated_small`。本轮不使用该选项重试，因为前一轮已经发生通信超时。

原始输出目录：`/tmp/p5-p2p.qaNp3Y/`。本次只说明当前节点的 P5/2-rank gate 未通过；前后
`hy-smi` 正常，不能直接推出设备 reset 或硬件故障，也没有扩大到 4/8 卡或切换 transport。

## P5 pair isend/irecv focused revalidation

根据 RCCL P2P operation contract，修正 `pair_isend_irecv` 的操作序列：低 rank 先发送、高 rank
接收，随后反向发送；两个方向之间使用 barrier，避免两个 rank 以同一顺序提交对向 unbatched
操作。修正后只运行这一 case：

```text
world size: 2
payload: 4 KiB
dtype: fp32, fp64, int64
backend requested/actual: nccl/nccl
torchrun rc: 0
rank 0/1: all case_start -> case_pass
hy-smi before/after: rc=0, 8 卡均正常
```

原始输出目录：`/tmp/p5-pair-focused2.m2vG0r/`。因此当前已确认 **P5 pair unbatched
isend/irecv correctness = PASS（2-rank、4 KiB、三种 dtype）**。此前对称提交模式的超时保留
为测试顺序/ProcessGroup 序列化诊断，不作为 RCCL 或硬件故障证据。P5 仍未完成，因为
`batch_isend_irecv`、ring、变长/非对称、zero-length 和 1000 次重复尚未在修正后的 harness
下运行。

## P5 pair batch_isend_irecv focused validation

随后只运行 `pair_batch_isend_irecv`，配置为 2 ranks、4 KiB、`fp32/fp64/int64`。两 rank
均完成 `case_start -> case_pass`，最终 `torchrun rc=0`；前后 `hy-smi` 均返回 0，8 卡保持
正常。原始输出目录：`/tmp/p5-batch-focused.BsnOzN/`。

当前 P5 pair 层结论为：

```text
unbatched isend/irecv: PASS
batch_isend_irecv: PASS
```

这仍只覆盖 2-rank、4 KiB 的 pair correctness，不代表 ring、变长/非对称、zero-length、
1000 次重复或 4/8 rank 已通过。

## P5 ring isend/irecv focused validation

按照 P2P dependency-cycle contract，将 ring 的 unbatched `isend/irecv` 分成偶数 rank 和奇数
rank 两个有序阶段，保留真实 ring peer 关系。2-rank、4 KiB 测试中，`fp32/fp64/int64` 三种
dtype 均完成 `case_start -> case_pass`，最终 `torchrun rc=0`，前后 `hy-smi` 均正常。

原始输出目录：`/tmp/p5-ring-focused.A6IjzG/`。当前已确认：

```text
pair isend/irecv: PASS
pair batch_isend_irecv: PASS
ring isend/irecv: PASS
```

这仍是 2-rank、4 KiB 的 narrow correctness evidence；ring 的 4/8 rank、变长/非对称、
zero-length 和 1000 次重复尚未完成。

## P5 remaining four focused validations

在上述 pair/ring 基础上，继续逐项运行剩余四个 case，均使用 2 ranks、4 KiB 基准、前后
`hy-smi` 健康检查和 90 秒 timeout，未设置 transport workaround：

| Case | dtype | 结果 | 原始输出 |
|---|---|---|---|
| `bidirectional_pair_batch` | fp32/fp64/int64 | PASS | `/tmp/p5-bidir-focused.dQo935/` |
| `ring_variable_asymmetric` | fp32/fp64/int64 | PASS | `/tmp/p5-variable-focused.o7ig2i/` |
| `zero_length_ring` | fp32/fp64/int64 | PASS | `/tmp/p5-zero-focused.klVP3g/` |
| `repeated_small_1000` | fp32，16 bytes | PASS | `/tmp/p5-repeat-focused.ktyAzs/` |

四项均为两 rank `case_start -> case_pass`，`torchrun rc=0`，前后 `hy-smi rc=0`。因此当前
P5 已完成 2-rank、4 KiB narrow correctness matrix；仍未完成 4/8-rank 扩展、较大 payload
扩展和多卡长稳验收。

## Comparison host: dcu1

在另一台节点 `dcu1` 使用同一个 `min_rccl_repro.sh 0 1` 完成对照测试。该运行的
退出码和数值结果均通过：

```text
rank0_rc=0 rank1_rc=0
rank=0 after_ncclAllReduce value=3
rank=1 after_ncclAllReduce value=3
```

RCCL 仍选择 `via P2P/IPC`，拓扑为 `XGMI[168.0]`，并完成 `Connected all rings`、
`Connected all trees`。dcu1 日志中也出现了 `NUMA auto balancing`、缺少 `iommu=pt`、
缺少 `HSA_FORCE_FINE_GRAIN_PCIE=1` 和 `net_ib.cc ... error code 2` 等提示，但最小单节点
P2P all-reduce 仍成功。因此这些提示在 dcu1 上不是该 probe 失败的充分条件，不能单独作为
dcu2 故障根因；应将 dcu1 与 dcu2 的驱动/固件、HFM/Hlink 配置、内核启动参数和设备健康
日志逐项对比。此次输出没有包含测试后的 `hy-smi` 实际结果，故只记为 direct RCCL PASS，
不扩大为 dcu1 完整节点健康结论。
## P5 4-rank and 8-rank full matrix

在完成 P5 探针中的依赖顺序修正后，继续执行完整默认矩阵：

```text
--case all --payloads 4K,1M --repeats 1000
```

矩阵覆盖 `pair_isend_irecv`、`pair_batch_isend_irecv`、`ring_isend_irecv`、`ring_variable_asymmetric`、`bidirectional_pair_batch`、`zero_length_ring` 和 `repeated_small_1000`，数据类型覆盖 fp32/fp64/int64。

| world size | 物理卡 | torchrun | hy-smi 前/后 | 原始输出 |
|---:|---|---:|---:|---|
| 4 | 0,1,2,3 | 0 | 0 / 0 | `/tmp/p5-world4.olb8ZN/` |
| 8 | 0,1,2,3,4,5,6,7 | 0 | 0 / 0 | `/tmp/p5-world8.aPcTJG/` |

两次运行中各 rank 均完成 `case_start -> case_pass`，最终 JSON 状态为 `PASS`，RCCL backend 为 `nccl/nccl`，设备识别为 `BW1000 64G`。运行后的 hy-smi 显示 8 张 HCU 均可见且为 Normal。

结合此前的 2-rank focused/full 结果，P5 的 2/4/8-rank correctness matrix 现已通过：4 KiB/1 MiB payload、fp32/fp64/int64，以及 pair、batch、ring、variable/asymmetric、bidirectional、zero-length 和 1000 次小消息重复路径均有实测证据。

本结果是 P2P 正确性与有限重复门，不等同于性能结论、长时间稳定性结论、64 MiB 大消息结论或多节点结论；这些属于后续独立验收范围。当时 P10+ 及 DomainParallel 尚未开始。

## P6 async collective matrix

新增 [`probes/distributed/probe_async.py`](../probes/distributed/probe_async.py)，按 RCCL
async contract 在每次数值检查前执行 `work.wait()` 和 device synchronize。探针覆盖：

- 单个 `all_reduce(async_op=True)` 及 wait 后数值检查；
- enqueue async collective 后执行独立设备计算，再 wait 并分别校验两个结果；
- 3 个同时 outstanding 的 all-reduce，正序和逆序 wait；
- fp32 4 KiB 消息连续 100 次 async all-reduce，验证 handle 生命周期和退出前清理。

完整矩阵命令为：

```text
--case all --payloads 4K,1M --repeats 100
```

| world size | 物理卡 | torchrun | hy-smi 前/后 | 原始输出 |
|---:|---|---:|---:|---|
| 2 | 0,1 | 0 | 0 / 0 | `/tmp/p6-world2.zXpJp4/` |
| 4 | 0,1,2,3 | 0 | 0 / 0 | `/tmp/p6-world4.sNgpti/` |
| 8 | 0,1,2,3,4,5,6,7 | 0 | 0 / 0 | `/tmp/p6-world8.JGAisz/` |

三次运行均使用 `nccl` backend、设备 `BW1000 64G`，各 rank 的最终 JSON 为 `PASS`；8-rank
运行记录了 200 个 `case_start` 和 200 个 `case_pass` 事件。运行后 8 张 HCU 均可见且为
`Normal`。

因此 P6 async correctness 在 2/4/8 ranks、4 KiB/1 MiB、fp32/fp64/int64 范围内通过。
本结果不证明通信与计算的真实 overlap 或吞吐性能；独立 stream overlap 属于 P11，长时间
稳定性属于 P12；P8 functional collectives 和 P10 performance 结果见下文。

## P7 DeviceMesh and subgroup matrix

新增 [`probes/distributed/probe_devicemesh.py`](../probes/distributed/probe_devicemesh.py)，
使用当前 DTK PyTorch 2.9 的 `DeviceMesh` API，显式构造 CPU rank mesh，并在已绑定的 DCU
device 上执行 group collective。探针验证：

- 1D mesh 的 shape、rank lookup、coordinate、local rank 和 mesh group；
- 1D `mesh["world"]` submesh 及 subgroup all-reduce；
- 4/8 rank 的 2D mesh：4 rank 使用 2×2，8 rank 使用 2×4；
- 2D row/col subgroup 的 rank mapping、group size/local rank 和精确 all-reduce。

完整矩阵使用：

```text
--payloads 4K,1M
```

| world size | mesh | 物理卡 | torchrun | hy-smi 前/后 | 原始输出 |
|---:|---|---|---:|---:|---|
| 2 | 1D `[2]` | 0,1 | 0 | 0 / 0 | `/tmp/p7-world2.uKPSBp/` |
| 4 | 1D `[4]` + 2D `[2,2]` | 0,1,2,3 | 0 | 0 / 0 | `/tmp/p7-world4.GjwdYW/` |
| 8 | 1D `[8]` + 2D `[2,4]` | 0,1,2,3,4,5,6,7 | 0 | 0 / 0 | `/tmp/p7-world8.dDIheI/` |

三次运行均使用 `nccl` backend、设备 `BW1000 64G`，所有 rank 最终 JSON 为 `PASS`。事件
计数分别为 2-rank `26/26`、4-rank `104/104`、8-rank `208/208`（`case_start/case_pass`），
运行后 8 张 HCU 均为 `Normal`。

因此 P7 DeviceMesh/subgroup correctness 在 2/4/8 ranks、4 KiB/1 MiB、fp32/fp64/int64 范围内
通过。该结果不等于 DomainParallel 已支持，也不覆盖 autograd collective、通信计算 overlap、
性能或长时间稳定性；P8/P9 functional 与 autograd、P10 performance 结果见下文。

## P8 functional collectives matrix

新增 [`probes/distributed/probe_functional_collectives.py`](../probes/distributed/probe_functional_collectives.py)，
直接导入当前 DTK PyTorch 的 `torch.distributed._functional_collectives`，调用 functional
`all_to_all_single` 与 `wait_tensor`。探针不以普通 `dist.all_to_all_single` 替代 functional API，
并对每个 case 显式提供 input/output split lists，覆盖 zero split、非均衡 split、forward/reverse
exchange、`fp32/fp64/int64` 以及 4 KiB/1 MiB payload。

完整矩阵命令为：

```text
--payloads 4K,1M
```

| world size | 物理卡 | torchrun | hy-smi 前/后 | 原始输出 |
|---:|---|---:|---:|---|
| 2 | 0,1 | 0 | 0 / 0 | `/tmp/p8-world2.YvSIS1/` |
| 4 | 0,1,2,3 | 0 | 0 / 0 | `/tmp/p8-world4.AyUjTw/` |
| 8 | 0,1,2,3,4,5,6,7 | 0 | 0 / 0 | `/tmp/p8-world8.aGlhbD/` |

三次运行均使用 `nccl` backend、设备 `BW1000 64G`；2-rank、4-rank、8-rank 分别完成
24/24、48/48、96/96 个 `case_start/case_pass` 事件，所有 rank 最终 JSON 为 `PASS`，
运行后 8 张 HCU 均为 `Normal`。因此 P8 functional collective correctness 在 2/4/8 ranks、
4 KiB/1 MiB、fp32/fp64/int64 范围内通过；其中 8-rank 是在指南要求的 2/4-rank gate 之外的
单节点扩展验证。

该结果不等于 P8-B PhysicsNeMo exact compatibility，不覆盖通信计算 overlap、吞吐性能、长时间
稳定性或生产 DomainParallel 支持；P9 autograd-sensitive 结果见下文。

## P9 autograd-sensitive communication matrix

新增 [`probes/distributed/probe_autograd_collective.py`](../probes/distributed/probe_autograd_collective.py)，
按指南的最小链路验证：

```text
local tensor requires_grad
→ communication/gather
→ differentiable local loss
→ backward
```

每个 case 同时运行 direct `dist.all_gather` 和显式
`torch.distributed._functional_collectives.all_gather_tensor_autograd`。前者用于记录当前
PyTorch c10d collective 的事实：数据前向精确正确，但输出没有 autograd 路径，
`native_autograd_collective_supported=false`。后者显式支持跨 rank 反向；由于每个 rank 都对
完整 gathered loss 反向，local gradient 的 oracle 为 `world_size * 2 * local`，并在每个 rank
精确校验通过。

矩阵使用 `fp32/fp64` 和 4 KiB/1 MiB payload；P9 不使用 int64，因为该项验证反向梯度。

| world size | 物理卡 | torchrun | hy-smi 前/后 | case start/pass | 原始输出 |
|---:|---|---:|---:|---:|---|
| 2 | 0,1 | 0 | 0 / 0 | 8 / 8 | `/tmp/p9-world2.7037Zm/` |
| 4 | 0,1,2,3 | 0 | 0 / 0 | 16 / 16 | `/tmp/p9-world4.E9EBRO/` |
| 8 | 0,1,2,3,4,5,6,7 | 0 | 0 / 0 | 32 / 32 | `/tmp/p9-world8.hXLbgq/` |

三次运行均使用 `nccl` backend、设备 `BW1000 64G`，所有 rank 最终 JSON 为 `PASS`，
`functional_autograd_supported=true`，`native_autograd_collective_supported=false`，运行后
8 张 HCU 均为 `Normal`。

因此 P9 autograd-sensitive runtime smoke 在 2/4/8 ranks、4 KiB/1 MiB、fp32/fp64 范围内通过。
这是 PyTorch functional autograd runtime 证据，不等于 nvalchemi HALO、owner contribution
语义、DomainParallel 生产路径或 P12/P13 已完成；P11 overlap 结果见下文。

## P10 communication performance baseline

新增 [`probes/distributed/probe_performance.py`](../probes/distributed/probe_performance.py)，按指南
在无强制 transport、algorithm、protocol、channel 或 topology file 的 baseline 环境运行：

```text
all_reduce
all_gather
all_to_all_single
P2P ping-pong（物理 rank pair：0↔1、2↔3、...）
```

覆盖 1 KiB、4 KiB、64 KiB、1 MiB、16 MiB 和 64 MiB；每个 case 使用 10 次 warmup 和 20 次
steady-state samples。优先使用 DTK PyTorch `cuda_event`，本次三轮均为 `cuda_event`；每个 rank
的原始 JSON 保留完整 samples、median/p90/min/max、mean、stdev、relative stdev 和 correctness。
`rccl-tests` 的 `all_reduce_perf` 等工具在本节点未安装，因此没有混入另一套 benchmark 口径。

effective GiB/s 是透明的 descriptive metric，logical bytes 定义为：

```text
all_reduce:        2 * (world_size - 1) / world_size * payload
all_gather:          (world_size - 1) * payload
all_to_all_single:   (world_size - 1) / world_size * payload
P2P ping-pong:       2 * payload
```

| world size | 物理卡 | torchrun | hy-smi 前/后 | case start/pass | 原始输出 |
|---:|---|---:|---:|---:|---|
| 2 | 0,1 | 0 | 0 / 0 | 48 / 48 | `/tmp/p10-world2.GOn3ef/` |
| 4 | 0,1,2,3 | 0 | 0 / 0 | 96 / 96 | `/tmp/p10-world4.AZD8hP/` |
| 8 | 0,1,2,3,4,5,6,7 | 0 | 0 / 0 | 192 / 192 | `/tmp/p10-world8.Dgiuxh/` |

下表是每轮 rank 0 的代表性结果；所有 rank 的 JSON 和 20 个原始样本均在对应 artifact 中。单位为
`median/p90 ms` 和按上述公式计算的 `effective GiB/s`：

| world | payload | all_reduce | all_gather | all_to_all_single | P2P ping-pong |
|---:|---:|---|---|---|---|
| 2 | 1 MiB | 0.100880/0.113920; 9.680437 | 0.121440/0.147041; 8.041523 | 0.086080/0.100960; 5.672412 | 0.206881/0.212640; 9.440814 |
| 2 | 64 MiB | 0.674561/0.685281; 92.652851 | 1.004001/1.013601; 62.250934 | 0.833201/0.847681; 37.505956 | 1.105921/1.108321; 113.027965 |
| 4 | 1 MiB | 0.132800/0.141280; 11.030450 | 0.153040/0.165120; 19.143280 | 0.089360/0.104480; 8.196306 | 0.230160/0.241441; 8.485945 |
| 4 | 64 MiB | 0.911601/0.915841; 102.841046 | 2.293201/2.310081; 81.763439 | 0.552640/0.559680; 84.820136 | 1.233761/1.242081; 101.316219 |
| 8 | 1 MiB | 0.151200/0.161600; 11.302807 | 0.229200/0.248960; 29.825207 | 0.128080/0.141920; 6.671550 | 0.246321/0.252320; 7.929186 |
| 8 | 64 MiB | 1.045281/1.053600; 104.636935 | 4.902961/4.924321; 89.231793 | 0.660880/0.671840; 82.749516 | 1.238800/1.244000; 100.904101 |

这些数字用于画像和发现 rank/拓扑/规模异常，不设绝对 GB/s PASS 阈值，也不构成跨机器、跨 DTK
版本或生产 MD 端到端性能结论。小消息部分存在较高相对离散度，后续如需调优应先按 rank、拓扑
和 transport 继续拆分；本次没有做 transport workaround 或性能优化。

## P11 communication/compute overlap smoke

新增 [`probes/distributed/probe_overlap.py`](../probes/distributed/probe_overlap.py)，验证与真实
DomainParallel step 相近的最小顺序：

```text
serial:  all_reduce -> default stream 上的 1024x1024 FP32 matmul
overlap: all_reduce(async_op=True) + 独立 CUDA stream 上的 matmul -> wait
```

两条路径都对 all-reduce 结果和矩阵结果做 exact correctness check。overlap 路径显式等待
`Work`，随后同步设备和独立 stream，计时使用同步后的 host wall-clock；所以本项没有把
`async_op=True` 的 handle 返回误判为真实 overlap。每个 payload、每条路径均使用 5 次 warmup
和 10 次 steady-state samples，未设置 transport、algorithm、protocol、channel 或 topology
覆盖。

正式运行命令（2/4/8-rank 分别替换 visible devices 和 `--nproc-per-node`）：

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 300 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_overlap.py --payloads 1M,16M \
  --matrix-size 1024 --warmup 5 --iterations 10
```

结果矩阵如下；每个 case 的 `case_start`/`case_pass` 均完整，所有 rank 最终 correctness 和
completion 均通过：

| world size | 物理卡 | torchrun | hy-smi 前/后 | case start/pass | 原始输出 |
|---:|---|---:|---:|---:|---|
| 2 | 0,1 | 0 | 0 / 0 | 4 / 4 | `/tmp/p11-world2-20260922-075713/` |
| 4 | 0,1,2,3 | 0 | 0 / 0 | 8 / 8 | `/tmp/p11-world4-20260922-075815/` |
| 8 | 0,1,2,3,4,5,6,7 | 0 | 0 / 0 | 16 / 16 | `/tmp/p11-world8-20260922-075914/` |

下表为 rank 0 的代表性中位数，单位 ms；括号内是 `(overlap - serial) / serial` 的描述性变化：

| world size | payload | serial median | overlap median | 描述性变化 |
|---:|---:|---:|---:|---:|
| 2 | 1 MiB | 0.178535 | 0.209309 | +17.24% |
| 2 | 16 MiB | 0.356979 | 0.346699 | -2.88% |
| 4 | 1 MiB | 0.200060 | 0.224604 | +12.27% |
| 4 | 16 MiB | 0.409994 | 0.402904 | -1.73% |
| 8 | 1 MiB | 0.196435 | 0.254100 | +29.36% |
| 8 | 16 MiB | 0.473264 | 0.475314 | +0.43% |

1 MiB 在各 rank/规模整体表现为 overlap 更慢。16 MiB 只有部分 rank 出现约 5--9% 的局部
下降，另一些 rank 持平或略慢；按 all-rank completion 不能确认稳定的通信计算收益。8-rank 的
1 MiB case 还保留了一个 rank 2 的 4.035 ms 单样本离群值（该 rank 中位数为 0.267784 ms），
说明小消息和多 rank 下仍有调度/系统噪声。该数据用于发现同步、stream 和 rank 异常，不设
overlap 性能 PASS 阈值，也不宣称生产 DomainParallel 已具备通信计算 overlap。

## P12 mixed communication stability

新增 [`probes/distributed/probe_stability.py`](../probes/distributed/probe_stability.py)，按指南
完成最低 operation-count 矩阵。每个阶段都复用预分配的 FP32 device buffers，五种操作按固定
顺序循环，并在每次操作完成后同步和 exact-check：

```text
all_reduce
all_gather
all_to_all_single
batched P2P ring
async all_reduce + wait
```

默认阶段为 4 KiB/10,000 次、1 MiB/1,000 次和 16 MiB/100 次；另有 2 个 warmup cycles，
不计入正式 operation-count。探针逐阶段输出 flushed progress、各操作计数、allocator
`memory_allocated`/`max_memory_allocated`/`memory_reserved`/`max_memory_reserved` 快照，
遇到首个错误即停止，不自动重试。

| world size | 物理卡 | torchrun | hy-smi 前/后输出 | phase start/pass | 原始输出 |
|---:|---|---:|---:|---:|---|
| 2 | 0,1 | 0 | captured / captured | 3 / 3 | `/tmp/p12-world2-20260922-080859/` |
| 4 | 0,1,2,3 | 0 | captured / captured | 3 / 3 | `/tmp/p12-world4-20260922-080953/` |
| 8 | 0,1,2,3,4,5,6,7 | 0 | captured / captured | 3 / 3 | `/tmp/p12-world8-20260922-081047/` |

三轮均完成 11,100 个正式 mixed operations；每个 phase 的五类 operation 各占 1/5，所有 rank
最终 JSON 为 `PASS`，没有 hang、timeout、rank 丢失或 correctness failure。阶段内 memory trace
保持不变：以 rank 0 为例，2-rank 的 allocated 在 small/medium/large 阶段分别稳定在约
68 KiB/17.0 MiB/272.0 MiB，4-rank 约为 84 KiB/21.0 MiB/336.0 MiB，8-rank 约为
116 KiB/29.0 MiB/464.0 MiB。阶段之间 reserved 随首次分配更大 buffer 增长，这是 allocator
缓存和 payload 尺寸变化，不是循环内持续增长；阶段结束释放 buffer 后 allocated 均回到约
512 B，而 reserved 保留缓存。

因此 P12 的 operation-count stability gate 在 2/4/8 ranks 上通过。该证据是几秒级、以操作
数量为定义的长稳测试，不是小时级 soak；P13 的 allocator 与设备侧显存证据见下文，不能用
本项替代。

## P13 allocator and device-memory behavior

新增 [`probes/distributed/probe_memory.py`](../probes/distributed/probe_memory.py)，按指南分别
测试大 payload 的 `all_gather` 和 `all_to_all_single`。每个 rank 使用 64 MiB FP32 payload，
每个 stage 先做 2 次 warmup，再做 5 次正式重复；每次重复都进行 exact correctness check。
探针在大 buffer 创建前、stage 准备后、重复期间、释放后和 `empty_cache()` 后记录：

```text
memory_allocated
max_memory_allocated
memory_reserved
max_memory_reserved
```

| world size | 物理卡 | torchrun | hy-smi / showpids 前后 rc | stage start/pass | 原始输出 |
|---:|---|---:|---:|---:|---|
| 2 | 0,1 | 0 | 0 / 0；0 / 0 | 4 / 4 | `/tmp/p13-world2-20260922-082933/` |
| 4 | 0,1,2,3 | 0 | 0 / 0；0 / 0 | 8 / 8 | `/tmp/p13-world4-20260922-083011/` |
| 8 | 0,1,2,3,4,5,6,7 | 0 | 0 / 0；0 / 0 | 16 / 16 | `/tmp/p13-world8-20260922-083050/` |

表中前一组 rc 是概览 `hy-smi`，后一组 rc 是 `hy-smi --showpids`；运行中另保存了
`hy-smi --showpids` 采样。三轮所有 rank 最终 JSON 为 `PASS`，无 hang、timeout、rank 丢失、
correctness failure 或错误标记。以每个 rank 的 allocator peak 增量计：

| world size | all_gather peak 增量 | all_to_all_single peak 增量 | 释放后 allocated | empty_cache 后 reserved |
|---:|---:|---:|---:|---:|
| 2 | 128 MiB | 0 MiB | 512 B | 2 MiB |
| 4 | 256 MiB | 0 MiB | 512 B | 2 MiB |
| 8 | 512 MiB | 0 MiB | 512 B | 2 MiB |

`all_gather` 的瞬时 peak 随 `world_size × payload` 增长，说明单体系多卡容量预算不能只看
用户预分配 tensor；本轮没有观察到释放后的 persistent allocated growth。`all_to_all_single`
在相同设置下没有额外 allocator peak。`hy-smi --showpids` 已保存设备侧辅助证据，但部分
PID 的 HCU 映射字段为空、VRAM 百分比为 `inf`，因此不把它当作可精确归因的每进程峰值，仍以
PyTorch allocator trace 作为本项主要证据。

因此 P13 memory behavior gate 在 2/4/8 ranks 上通过；它验证的是本轮 64 MiB FP32 大 collective
场景，不替代更大 payload 或小时级 soak，也不等于 DomainParallel 已具备生产容量承诺。

## P14 refreshed single-node topology record

本轮使用 [`dcu-topology`](../.agents/skills/dcu-toolkit-skills/dcu-topology/SKILL.md) 规定的只读采集
方式，原始输出保存在 `/tmp/p14-topology-20260922-083837/`。collector、所有 `hy-smi` 拓扑查询、
`lspci`、`rdma link` 和 `ibdev2netdev` 均返回 `rc=0`。

### DCU ↔ CPU NUMA ↔ OAM ↔ NIC

`hy-smi --showbus` 提供 HCU 顺序和 PCI/OAM，`hy-smi --showtoponuma --json` 提供同一 HCU 顺序
的 NUMA 亲和性；两者按 HCU index 直接配对。PCIe 树确认 HCU 与 NIC 位于同一 PCIe 分支：

| HCU | DCU PCI | OAM ID | NUMA | 同分支 NIC PCI | mlx5 / netdev |
|---:|---|---:|---:|---|---|
| 0 | `9f:00.0` | 1 | 3 | `9b:00.0` | `mlx5_0` / `ibs66` |
| 1 | `57:00.0` | 0 | 1 | `53:00.0` | `mlx5_1` / `ibs69` |
| 2 | `5e:00.0` | 4 | 1 | `5f:00.0` | `mlx5_2` / `ibs64` |
| 3 | `05:00.0` | 5 | 0 | `06:00.0` | `mlx5_3` / `ibs61` |
| 4 | `e9:00.0` | 3 | 7 | `ea:00.0` | `mlx5_4` / `ibs82` |
| 5 | `c1:00.0` | 2 | 5 | `c2:00.0` | `mlx5_5` / `ibs80` |
| 6 | `cc:00.0` | 7 | 5 | `c8:00.0` | `mlx5_6` / `ibs75` |
| 7 | `b1:00.0` | 6 | 4 | `ad:00.0` | `mlx5_7` / `ibs72` |

CPU NUMA 节点为 8 个，每节点 16 个逻辑 CPU：

```text
node0: 0-15       node1: 16-31      node2: 32-47      node3: 48-63
node4: 64-79      node5: 80-95      node6: 96-111     node7: 112-127
```

NUMA distance 矩阵如下，保留非平均值：

```text
node0: 10 15 15 15 25 27 27 27
node1: 15 10 15 15 27 25 27 27
node2: 15 15 10 15 27 27 25 27
node3: 15 15 15 10 27 27 27 25
node4: 25 27 27 27 10 15 15 15
node5: 27 25 27 27 15 10 15 15
node6: 27 27 25 27 15 15 10 15
node7: 27 27 27 25 15 15 15 10
```

### DCU ↔ DCU

当前 `hy-smi` 拓扑矩阵显示：

- 任意非自身 HCU 对的 link type 都是 `HSW`；
- 8×8 access 矩阵全部为 `TRUE`；
- 任意非自身 HCU 对的 hop 都是 `1`；
- `showtopoweight` 返回矩阵全部为 `0`，该字段本轮不提供额外区分度。

这与 P5/P10/P12/P13 使用的 0--7 物理卡顺序一致；本轮没有发现不同 HCU 对之间的拓扑路径类别。

### 证据边界

`collect_topology.sh -j` 本身成功，但其输出将 lexicographic PCI slot 列表与 hy-smi card 列表
按位置 zip；本机两种顺序不同。因此本报告没有把 collector JSON 的 `dcu_devices` 列表直接当作
最终 HCU↔NUMA 映射，而是采用原始 `showbus` + `showtoponuma` 的 HCU-index 配对，再用 PCIe 树
核对 NIC。P14 只完成拓扑记录，不启用自动绑核、不修改 HFM/hlink 配置，也不等于 DomainParallel
生产支持。

因此 P14 topology-record gate 完成，P0--P14 的单节点拓扑与通信先导证据已收口；可选 multi-node
probe 仍不属于本次首版验收范围。
