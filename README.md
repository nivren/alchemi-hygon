# alchemi-hygon

海光 DCU 原子模拟框架的开发主仓库。项目以 NVIDIA nvalchemi-toolkit 和
nvalchemi-toolkit-ops 为完整上游基线，保留 framework/ops 两个包边界，并在
Torch reference、Triton、HIP 后端边界内逐步适配 Hygon DCU。

这是面向开发者的入口，不是用户安装包。支持范围和实测证据以
[docs/STATUS.md](docs/STATUS.md) 与 reports/ 为准。

## 先读这些文档

1. [AGENTS.md](AGENTS.md)：所有开发者 agent 必须遵守的项目规则。
2. [docs/DEVELOPMENT_ENVIRONMENT.md](docs/DEVELOPMENT_ENVIRONMENT.md)：开发环境部署、DTK、海光 Torch/Triton 和验证命令。
3. [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)：后端、算子、功能、测试和 Git 协作方法。
4. [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md)：架构背景、阶段目标和历史交接。
5. [docs/UPSTREAM_LOCK.yaml](docs/UPSTREAM_LOCK.yaml)：两个上游工程的准确来源 URL 和锁定 SHA。
6. [docs/DEVELOPER_ARCHITECTURE.md](docs/DEVELOPER_ARCHITECTURE.md)：开发者视角的系统结构图、数据流和后端边界。

## 目录边界

- packages/framework：framework 正式开发代码，导入命名空间仍按上游保留。
- packages/ops：ops 正式开发代码和 backend dispatcher。
- external/：本地上游参考 clone，根仓库忽略，不是产品安装源。
- probes/：隔离的环境、kernel、梯度、模型和多卡探针。
- reports/：可提交的脱敏验证摘要。
- artifacts/：原始日志、轨迹和性能输出，默认不入库。
- docs/、adr/：兼容契约、环境、状态和重要决策。

不要在 packages/ 内创建嵌套 Git 仓库，不要修改 external/，不要把 external/直接
git add 成 submodule 或普通产品源码。

## 快速开始

从项目根目录加载当前 Hygon reference 环境：

~~~bash
source scripts/activate_hygon_env.sh project
~~~

先运行团队基础版本的 CPU gate：

~~~bash
scripts/check_cpu_reference.sh
~~~

HCU 批验证只在已分配的设备、DTK 已加载且 `/dev/kfd`、`/dev/dri` 可见的主机权限终端执行。
必须显式指定设备；该结果才可作为新的 DCU 验证证据：

~~~bash
HIP_VISIBLE_DEVICES=0 scripts/check_hcu_reference_smoke.sh
~~~

两个包都含顶层 test 包名，pytest 应分进程运行：

~~~bash
PYTHONPATH=packages/ops \
  .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_reference_backend.py

PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py \
  packages/framework/test/models/test_lj_torch_reference.py
~~~

若 .venv 不存在或本机 wheel 路径不同，先读
[开发环境部署指南](docs/DEVELOPMENT_ENVIRONMENT.md)，不要从 PyPI 安装 CUDA
版 Torch/Triton，也不要运行 uv sync --all-extras。

## 建立锁定版本的 external 参考 clone

external/ 被根 .gitignore 整目录忽略是有意设计：它是每台开发机本地保存的完整
上游参考源码，不进入产品仓库，不参与产品安装，也不应产生合并冲突。产品代码已经
通过不 squash 的 subtree 导入到 packages/；external/只用于查阅、版本审计和后续同步。

当前锁定值位于 docs/UPSTREAM_LOCK.yaml：

- framework：NVIDIA/nvalchemi-toolkit，SHA
  4dfe3723def34df3fadb245981081ccf8c94c257；
- ops：NVIDIA/nvalchemi-toolkit-ops，SHA
  26dbceb61e30cca80e1a5805eebeb51d7dc68fd1。

在一个新目录中可执行以下完整 clone 和固定版本流程：

~~~bash
mkdir -p external

git clone --no-single-branch \
  https://github.com/NVIDIA/nvalchemi-toolkit.git \
  external/nvalchemi-toolkit
git clone --no-single-branch \
  https://github.com/NVIDIA/nvalchemi-toolkit-ops.git \
  external/nvalchemi-toolkit-ops

FRAMEWORK_SHA=4dfe3723def34df3fadb245981081ccf8c94c257
OPS_SHA=26dbceb61e30cca80e1a5805eebeb51d7dc68fd1

git -C external/nvalchemi-toolkit fetch --no-tags origin "$FRAMEWORK_SHA"
git -C external/nvalchemi-toolkit checkout --detach "$FRAMEWORK_SHA"

git -C external/nvalchemi-toolkit-ops fetch --no-tags origin "$OPS_SHA"
git -C external/nvalchemi-toolkit-ops checkout --detach "$OPS_SHA"

test "$(git -C external/nvalchemi-toolkit rev-parse HEAD)" = "$FRAMEWORK_SHA"
test "$(git -C external/nvalchemi-toolkit-ops rev-parse HEAD)" = "$OPS_SHA"
git -C external/nvalchemi-toolkit status --short
git -C external/nvalchemi-toolkit-ops status --short
~~~

如果 external/ 已存在，先检查 remote、工作树和当前 HEAD，再 fetch/checkout；不要
重复 clone、清空目录或覆盖其中的本地工作。若已有 shallow clone，先补齐完整历史：

~~~bash
git -C external/nvalchemi-toolkit fetch --unshallow origin
git -C external/nvalchemi-toolkit-ops fetch --unshallow origin
~~~

服务器不能直连 GitHub 时，可使用团队认可的完整镜像或 Git bundle，但必须在
docs/UPSTREAM.md 中记录官方 URL、实际来源、SHA、完整历史状态和许可证信息。固定
版本后，根仓库导入/同步仍按 [docs/START_HERE.md](docs/START_HERE.md) 的 subtree
流程执行；普通开发者通常只需查阅 external，不需要重新导入 packages/。

## 当前后端边界

- Torch reference 是当前正确性基线，已注册并有 CPU/HCU 窄 slice 证据。
- Triton 和 HIP 工具链基础探针已通过，但生产 neighbor/LJ/MLIP kernel 尚未因此完成。
- Warp 默认路径和上游公共 API 保留；不要把 HCU reference 改成默认静默回退。
- 新后端必须先进入 dispatcher/capability gate，再由 framework 显式传递。
- 具体算子必须分别说明 forward、一阶梯度、二阶梯度、PBC、full/half neighbor、
  空输入、容量、错误和 stream 语义。

## 开发循环

~~~text
读 AGENTS/STATUS 和锁定上游
  -> 写 feature/operator contract
  -> Torch reference 与最小测试
  -> Triton/HIP 实现和 capability gate
  -> CPU/HCU/必要的多卡验证
  -> reports、FEATURE_COMPATIBILITY、STATUS 和必要 ADR
  -> 小而独立的 git commit
~~~

每位开发者从 develop 或当前阶段分支创建自己的 `<开发者>/<类型>-<主题>` 分支，例如
`alice/feature-triton-neighbor`。提交前只暂存自己的路径，运行
git diff --check、相关测试和可获得的 HCU 探针，并使用 git commit -s。不要修改
共享提交历史，不自动 push；需要同步上游时另建分支并保留 subtree 历史。

详细规则见 [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)。
团队协作入口、已收口的 golden paths 与下一个任务队列见
[docs/TEAM_DEVELOPMENT_BASELINE.md](docs/TEAM_DEVELOPMENT_BASELINE.md)；新增 Torch operation
按 [docs/ADD_TORCH_OPERATION.md](docs/ADD_TORCH_OPERATION.md) 走最小闭环；新手积分器教程见
[docs/TUTORIAL_TORCH_INTEGRATOR_FOR_BEGINNERS.md](docs/TUTORIAL_TORCH_INTEGRATOR_FOR_BEGINNERS.md)。
