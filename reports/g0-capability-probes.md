# G0 能力探针摘要

日期：2026-09-06 UTC  
代码基线：`c25662e`（探针执行前）  
目标设备：BW200 / UBB BW1000，适配架构 gfx936  
环境：项目 `.venv`，Torch `2.9.0+das.opt1.dtk2604`，Triton `3.3.0+das.opt1.dtk2604.torch290`，DTK 26.04

## 单卡 Triton（P03）

主机设备节点可见时运行：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  TRITON_CACHE_DIR=$PWD/artifacts/triton-cache timeout 60 \
  .venv/bin/python probes/triton_probe.py
```

退出码：`0`。

```json
{
  "status": "passed",
  "triton": "3.3.0",
  "device": "BW200, UBB BW1000",
  "cold_seconds": 0.5193737840745598,
  "warm_host_loop_seconds_per_call": 2.4572492111474276e-05,
  "max_error": 2.384185791015625e-07
}
```

探针覆盖 1025 个 FP32 元素、tail mask、5 次预热和 20 次稳态调用。它证明 gfx936 上 Triton 的基础编译和执行路径可用，但不证明任何生产算子已经满足 registry 门槛，也不提供邻居、梯度、动态容量或端到端性能证据。

## 双卡 RCCL/NCCL（P07）

主机设备节点可见且两张卡空闲时运行：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 90 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed_probe.py
```

退出码：`0`。PyTorch `nccl` backend 在 DTK 环境中使用 RCCL；初始化日志报告 NCCL version `2.22.3`。

```json
{"rank": 0, "all_reduce": 3.0, "received": 8.0, "status": "passed"}
{"rank": 1, "all_reduce": 3.0, "received": 7.0, "status": "passed"}
```

这只证明 all-reduce 和双向点对点通信原语可用，不证明 DomainParallel、LJ ownership、跨卡原子归属或多卡轨迹守恒。

## 环境边界

在沙箱内直接运行相同探针时，`/dev/kfd` 不可见，Torch 报 `No HIP GPUs are available`，退出码为 `1`。这是执行环境设备隔离，不作为 Triton 或 HCU 能力失败证据。项目激活脚本现已按 shell 选择 DTK 环境：bash 使用 `/opt/dtk-26.04/env.sh`，zsh 使用 `/opt/dtk-26.04/env.zsh`；zsh 路径已验证 `DTKROOT=/opt/dtk-26.04`。
