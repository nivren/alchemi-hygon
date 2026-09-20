# 海光 DCU 开发环境部署指南

本文面向第一次加入本项目的开发者。它描述的是当前已验证的 Hygon Torch reference
开发环境和可复现的检查方法，不是把所有上游 NVIDIA CUDA/Warp 功能一次性装齐的安装手册。
先建立一个能运行 CPU/reference 和 HCU smoke 的环境，再按具体任务增加依赖。

当前主机的事实以 [ENVIRONMENT.md](ENVIRONMENT.md)、
[configs/hygon-reference-lock.txt](../configs/hygon-reference-lock.txt) 和
[reports/probe-environment-freeze.txt](../reports/probe-environment-freeze.txt) 为准。
锁文件中的 /data/envs/*.whl 是主机本地海光 wheel；换机器时必须重新确认 wheel、DTK
和 ABI，不能把这些路径当成公共下载地址。

## 1. 环境分层

| 层 | 当前参考值 | 用途 |
|---|---|---|
| 操作系统/架构 | Ubuntu 22.04.5、x86_64 | 当前服务器基线 |
| DCU/DTK | BW200 / UBB BW1000、gfx936、DTK 26.04 | HCU 运行和 HIP/Triton 编译 |
| Python | CPython 3.12（当前 uv 提供 3.12.13） | framework/ops 声明版本的交集 |
| Torch | 2.9.0+das.opt1.dtk2604，海光本地 wheel | 模型、reference 和训练前端 |
| Triton | 3.3.0+das.opt1.dtk2604.torch290，海光本地 wheel | 基础能力已探针通过，生产算子仍未完成 |
| Python 环境 | 项目根目录 .venv | 正式开发环境；被 .gitignore 忽略 |
| 依赖源 | 北外 PyPI 镜像、/data/envs/uv-cache | 只用于 Python 依赖；Torch/Triton 仍从本地 wheel 来 |

当前已验证的 HCU 是共享服务器资源。ENVIRONMENT.md 和报告中的绝对耗时不能直接作为
发布性能承诺；新的 benchmark 必须注明是否独占设备、cold/JIT 还是 steady-state。

## 2. 前置检查

以下命令只读检查当前 shell，不安装软件：

~~~bash
cd /path/to/alchemi-hygon
command -v git
command -v uv
python3 --version
test -f /opt/dtk-26.04/env.sh
test -f /opt/dtk-26.04/env.zsh
test -f /data/envs/torch-2.9.0+das.opt1.dtk2604-cp312-cp312-manylinux_2_28_x86_64.whl
test -f /data/envs/triton-3.3.0+das.opt1.dtk2604.torch290-cp312-cp312-manylinux_2_28_x86_64.whl
~~~

如果 DTK 或 wheel 不存在，不要自行升级系统驱动、切换 /opt 下的 DTK 或从 PyPI 安装
CUDA 版 Torch。记录缺失项并使用 CPU/reference 路径继续工作，或向环境管理员申请对应
资源。

核对本地 wheel 的完整性：

~~~bash
sha256sum /data/envs/torch-2.9.0+das.opt1.dtk2604-cp312-cp312-manylinux_2_28_x86_64.whl
sha256sum /data/envs/triton-3.3.0+das.opt1.dtk2604.torch290-cp312-cp312-manylinux_2_28_x86_64.whl
~~~

输出应分别与 [configs/hygon-wheel-manifest.txt](../configs/hygon-wheel-manifest.txt) 一致。

## 3. 创建或复用项目环境

项目环境不存在时创建 CPython 3.12 环境。已有 .venv 时不要为了开始任务重建它；先用
uv pip freeze 和当前锁文件检查差异。

~~~bash
cd /path/to/alchemi-hygon
UV_CACHE_DIR=/tmp/alchemi-uv-cache uv venv --python 3.12 .venv
~~~

当前主机的精确 reference 环境可以用锁文件同步：

~~~bash
cd /path/to/alchemi-hygon
UV_CACHE_DIR=/data/envs/uv-cache \
  uv pip sync \
  --python .venv/bin/python \
  --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple \
  configs/hygon-reference-lock.txt
~~~

该命令会安装锁定的 MACE/e3nn/ASE 和 framework 基础依赖，并使用锁文件中的本地 Torch/
Triton wheel。新机器若本地 wheel 路径不同，不要直接修改并提交共享锁文件；应由管理员在
机器上提供同版本 wheel，或生成只留在机器上的私有 lock，并把差异记录在环境报告中。

输入规格 [configs/hygon-reference.in](../configs/hygon-reference.in) 用于理解这套环境的
最小依赖和不可替换的 Torch/Triton 来源。packages/framework/uv.lock 与
packages/ops/uv.lock 是各自上游包的解析锁，不替代 Hygon reference 环境锁。

### 开发工具

reference 锁文件刻意保持较小，未承诺包含完整 upstream dev group。需要 lint、pre-commit
或 coverage 时按当前网络和权限单独安装，并把实际版本写入报告：

~~~bash
UV_CACHE_DIR=/data/envs/uv-cache \
  uv pip install --python .venv/bin/python \
  --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple \
  pre-commit ruff==0.11.13 pytest-timeout pytest-cov
~~~

不要运行 uv sync --all-extras。framework 的 cu12/cu13、MACE、UMA 等 extra 存在
互斥依赖，而且 CUDA extras 可能替换海光 Torch；本项目 reference 环境不安装
nvidia-physicsnemo，也不把 Warp 当作 HCU 默认路径。

## 4. 加载环境

从项目根目录执行：

~~~bash
source scripts/activate_hygon_env.sh project
~~~

脚本会：

- bash 加载 /opt/dtk-26.04/env.sh，zsh 加载 /opt/dtk-26.04/env.zsh；
- 激活项目 .venv；
- 设置 HYGON_PROJECT_ROOT、HYGON_PYTHON_ENV、PYTHONPATH；
- 设置 UV/PIP 默认镜像和 UV_CACHE_DIR。

不要在 zsh 中手动 source 只适用于 bash 的 env.sh。GPU 可见性、线程数和超时由每条
探针/作业命令显式设置，不要把 HIP_VISIBLE_DEVICES 固定写进全局环境脚本。

exploration 模式只用于复现历史探索环境：

~~~bash
source scripts/activate_hygon_env.sh exploration
~~~

它指向外部探索工程的旧 .venv，不是产品安装目标；不要在其中写入项目代码、冻结新依赖
或用它替代项目 .venv 的回归结果。

## 5. 三步验证

### 5.1 先确认 Torch、设备和梯度

CPU 验证不需要设备节点：

~~~bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -u probes/torch_probe.py --device cpu
~~~

HCU 验证必须在 /dev/kfd、/dev/dri 可见的主机权限终端执行：

~~~bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 60 \
  .venv/bin/python -u probes/torch_probe.py --device cuda
~~~

HIP 版 PyTorch 仍使用 torch.cuda 命名空间，因此不要仅根据字符串 cuda 判断 NVIDIA。
应同时检查 torch.version.hip、torch.cuda.is_available() 和设备名称。受限容器中出现
No HIP GPUs are available 通常只说明设备节点没有映射，不足以推断 DTK 或 HCU 不可用。

### 5.2 运行最小 reference 回归

两个包都包含顶层 test 包名，不能在同一个 pytest 进程中混跑，避免
ImportPathMismatchError。先分开运行：

~~~bash
source scripts/activate_hygon_env.sh project

PYTHONPATH=packages/ops \
  .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_reference_backend.py

PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py \
  packages/framework/test/models/test_lj_torch_reference.py
~~~

HCU 运行时给第二条命令加 HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 120。测试
通过只说明对应 reference slice 通过，不代表完整上游测试、生产 cell-list、Triton/HIP
kernel 或 DomainParallel 已通过。

### 5.3 检查真实模型（可选）

真实 MACE 探针需要管理员或用户提供可信的本地 checkpoint 和数据，不从脚本中自动下载：

~~~bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops \
  HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 120 \
  .venv/bin/python -u probes/mace_probe.py \
  --device cuda \
  --checkpoint /path/to/MACE-OFF23_small.model
~~~

checkpoint 路径必须是真实文件。报告中记录路径的脱敏标识或 SHA256，不要提交模型、数据、
token 或完整环境变量。当前已验证的本地模型和 HCU 结果见 `docs/STATUS.md` 与
[`reports/README.md`](../reports/README.md)。

## 6. HCU 作业注意事项

- 先用 hy-smi 或管理员提供的作业工具查看资源；共享服务器上不要自行占满所有卡、杀掉
  他人任务或启动无时间上限的 benchmark。
- 探针使用 HIP_VISIBLE_DEVICES=<id> 限定卡号，使用 timeout，并设置合理的
  OMP_NUM_THREADS；双卡探针只有在明确获得两张空闲卡后运行。
- HCU 结果必须写明 DTK 入口、设备型号、架构、可见卡、dtype、随机种子、warmup/同步方式、
  退出码以及是否存在共享负载。
- 原始日志写入被忽略的 artifacts/；可审查、脱敏的结论写入 reports/。不要把大轨迹、
  checkpoint 或机器私有路径加入 Git。

## 7. 常见失败处理

| 现象 | 首先检查 | 处理 |
|---|---|---|
| DTK environment not found | 当前 shell、HYGON_DTK_ENV、DTK 文件路径 | 选择正确的 bash/zsh 入口；不要切换系统 DTK |
| Python environment not found | .venv/bin/activate | 按第 3 节创建环境；不要覆盖已有环境 |
| Torch 变成 CUDA/PyPI 版本 | python -c 'import torch; print(torch.__version__, torch.version.hip)'、uv pip freeze | 停止回归，恢复/重建隔离环境并记录；不要继续用错误环境产证据 |
| No HIP GPUs are available | /dev/kfd、/dev/dri、是否在受限沙箱 | 换设备节点可见的主机作业环境；CPU/reference 可继续 |
| No module named warp | 是否显式请求 Warp、是否误导入 NVIDIA-only 模块 | reference 路径按需导入；不要为通过 HCU smoke 强行安装 CUDA Warp |
| ImportPathMismatchError | 是否同时收集 framework/ops 两个 test 树 | 分两个 pytest 进程执行 |
| mirror 403/超时 | UV_INDEX_URL、缓存和网络策略 | 使用项目脚本指定的镜像；把例外写入报告，不改全局配置 |

## 8. 交接记录

部署完成后至少保留以下信息：

~~~text
主机/作业环境：
DTK 与设备：
Python/Torch/Triton 版本：
环境创建/同步命令：
wheel 来源与 SHA256：
探针命令与退出码：
pytest 命令与结果：
未安装或未验证的依赖：
~~~

环境变更影响支持范围时同步 docs/ENVIRONMENT.md、docs/STATUS.md 和对应
FEATURE_COMPATIBILITY.yaml 条目。不要只在聊天中保留关键环境事实。
