# 环境盘点与更新（2026-09-06 UTC）

- Ubuntu 22.04.5 LTS，Linux 5.15.0-136-generic，x86_64。
- 沙箱外 hy-smi：8 张 BW200 / UBB BW1000，每卡 65520 MiB；驱动 6.3.30-V1.4.1a；任意两卡 Link Type 为 HSW。
- 13:11 UTC 首次资源盘点 8 卡均 100% HCU 利用率、43% VRAM；稍后型号盘点显存用量约 7.2–7.4 GiB，说明资源随时间变化，不能据此抢占。未取得专用卡分配，不启动计算。
- 沙箱内 /dev/dri、/dev/kfd 不可见，hy-smi 报无设备；沙箱外只读命令确认实际有卡。不是驱动缺失结论。
- PATH 中 hipcc 为 /opt/dtk-26.04/bin/hipcc；dcc 25.10.0-0、clang 17.0.0。/opt 还存在多套 DTK，不切换/升级系统。
- /opt/dtk-26.04/lib 存在 libamdhip64、librccl、libhipfft、librocfft、hipBLAS 等；文件存在不证明运行可用，RCCL 运行版本/性能待测。
- 系统 Python 3.10.12；uv 已安装 CPython 3.12.13，用该解释器创建项目 .venv，无系统 site-packages。
- Torch/Triton 使用用户提供的 cp312 海光 wheel；ABI 约束见 `configs/probe-constraints.txt`，完整 Hygon reference 冻结见 `configs/hygon-reference-lock.txt`；导入和 CPU 结果见 `reports/g0-validation.md`。不安装产品两包、不启用 CUDA extras。
- 默认 uv cache 指向 `/data/envs/uv-cache`；所有后续 `uv`/`pip` 安装默认使用 `https://mirrors.bfsu.edu.cn/pypi/web/simple`，必要时显式记录例外，不修改用户全局配置。
- 早期安装曾遇到其他镜像 HTTP 403；当前不再把其他源作为默认。本地 Torch/Triton wheel 来源不变。
- 为解除 framework 导入缺口，项目 `.venv` 已用 `/data/envs/uv-cache` 和北外镜像补齐 MACE 最小依赖（`mace-torch==0.3.15`、`e3nn==0.4.4`、`ase==3.29.0`、`matscipy==1.1.1` 等）以及 `jaxtyping`、`periodictable`、`tensordict` 等基础包；Torch/Triton 仍为海光版本。项目回归工具已固定为 `pytest==8.4.2`、`pytest-asyncio==1.4.0`。PhysicsNeMo 未安装，单进程 Torch/HCU 路径通过延迟导入保持可用；域并行和 PhysicsNeMo profiling 仍是显式未启用能力。

原始探针输出放 artifacts/g0 和 artifacts/g1，摘要放 reports。用户缓存中的 `MACE-OFF23_small.model` 已在探索环境和项目 `.venv` 的 CPU/HCU 运行；向量化 Torch reference 邻居后，项目 `.venv` 的两个 perf_46 结构 HCU wrapper batching 也已通过。本轮又在 Tier 1 装配优化前采集了真实 perf_46/92/184/368 的 CPU/HCU 邻居基线（含 46/92 原子 batch=32 和 184 原子 batch=16），详见 `reports/g2-neighbor-baseline-reference.md`；这些 HCU 数据来自共享负载，不是发布性能结论。生产 cell-list/Triton/HIP、完整 dynamics、域并行和 PhysicsNeMo profiling 仍未验证。

用户随后明确 BW200/BW1000 适配目标为 gfx936；所有目标 HIP 编译使用 gfx936。此前 gfx928 编译仅为过程试探，不作为目标能力证据。

本轮已从 `/home/wangleping/codes/nvalchemi-toolkit/.venv` 复用已验证的 NumPy 1.26.4，安装到项目 `.venv`；未重新下载 Torch/Triton。当前 Torch 的 NumPy 互操作已通过，详见 G0 验证摘要。

常用环境加载：

```bash
source scripts/activate_hygon_env.sh project       # 项目 .venv
source scripts/activate_hygon_env.sh exploration   # 已验证的探索环境
```

`activate_hygon_env.sh` 会根据当前 shell 自动选择 DTK 脚本：bash 使用 `/opt/dtk-26.04/env.sh`，zsh 使用 `/opt/dtk-26.04/env.zsh`。也可以直接加载对应脚本，但不能在 zsh 中直接 source 只适合 bash 的 `env.sh`。脚本还会激活选定 Python 环境，并设置项目 `PYTHONPATH`、北外 PyPI 镜像和 `/data/envs/uv-cache`；GPU 可见性、线程数和超时仍需由探针或作业命令显式指定。

环境脚本已在 bash 和 zsh 中实测：`DTKROOT=/opt/dtk-26.04`，项目 `.venv`
中的海光 Torch `torch.cuda.is_available()=True`。HCU 运行前使用仓库内的基础
探针确认当前 shell，而不要用 heredoc 临时拼接多行 Python：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 60 \
  .venv/bin/python -u probes/torch_probe.py --device cuda
```

在本机设备节点可见的主机权限终端，该命令于 2026-09-06 退出码为 `0`；若在
受限沙箱中得到 `No HIP GPUs are available`，通常是 `/dev/kfd`/`/dev/dri` 被隔离，
不能据此判断 DTK 未加载或 HCU 不可用。所有 HCU 证据均应记录运行终端的 DTK
入口和设备可见性。

2026-09-06 主机能力探针已确认：单卡 Triton 基础向量 kernel 和双卡 RCCL/NCCL all-reduce/P2P 均通过；这不等于生产 Triton/HIP kernel、LJ ownership 或 DomainParallel 已完成。沙箱内 `/dev/kfd` 不可见，GPU 探针必须在具备设备节点访问权限的主机终端运行。

项目环境的可重建规格：

- 输入依赖：[`configs/hygon-reference.in`](../configs/hygon-reference.in)；
- 当前主机精确冻结：[`configs/hygon-reference-lock.txt`](../configs/hygon-reference-lock.txt)；
- 海光本地 wheel 名称和 SHA256：[`configs/hygon-wheel-manifest.txt`](../configs/hygon-wheel-manifest.txt)；
- 生成冻结：`UV_CACHE_DIR=/data/envs/uv-cache bash scripts/freeze_hygon_env.sh project`；
- 运行时冻结摘要：[`reports/probe-environment-freeze.txt`](../reports/probe-environment-freeze.txt)。

精确冻结中的 Torch/Triton 是 `/data/envs` 下的本地主机 wheel URI；换机器时需先提供同版本海光 wheel，再按输入规格或锁文件同步。`packages/framework/uv.lock` 和 `packages/ops/uv.lock` 仍是各上游包的解析锁，不替代这个 Hygon reference 环境锁，也不要求安装 NVIDIA PhysicsNeMo。
