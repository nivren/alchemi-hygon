# 海光服务器部署与 Codex 启动

新开发者请先读 [开发环境部署指南](DEVELOPMENT_ENVIRONMENT.md) 和
[开发者指南](DEVELOPMENT_GUIDE.md)。本文件保留首轮服务器初始化和 Codex 启动流程；已有
工程不要重复初始化、覆盖文件或重复 clone。

## 1. 这份交接包包含什么

- `AGENTS.md`：放在项目根目录，供 Codex 自动读取，包含目标、约束、源码组织和首轮任务。
- `docs/PROJECT_HANDOFF.md`：完整开发背景、Feature Compatibility Contract、模块迁移、探针和验收路线。
- `docs/STATUS.md`：初始状态、后续交接和已取得实测证据；支持范围以当前快照和报告边界为准。
- `docs/DEVELOPMENT_ENVIRONMENT.md`：新开发者部署项目 reference 环境并运行最小验证的指南。
- `docs/DEVELOPMENT_GUIDE.md`：后端、算子、功能、测试和 Git 协作开发指南。
- `docs/海光DCU移植开发路线与工程规范.md`：此前调研原文，作为历史背景保留；修订意见在 PROJECT_HANDOFF 中。
- `.gitignore`：排除上游参考 clone、本地环境和大体积计算输出。

主文档是 AGENTS.md，建议整包复制，确保详细引用可读。这里只提供文件和命令说明，未在用户服务器上创建目录、安装软件或 clone 仓库。

## 2. 在海光服务器创建项目目录

以下命令在 Linux Bash 执行，路径只是示例。先将交接 ZIP 上传到服务器自己的目录，再替换两处路径。目标应为新的、专用于此项目的目录；如果已有内容，先检查并合并，不能覆盖现有 AGENTS 或 Git 配置。

```bash
# 修改为你的真实路径
PROJECT_DIR="/data/your-user/alchemi-dcu"
HANDOFF_ZIP="/data/your-user/uploads/hygon-project-starter.zip"

mkdir -p "$PROJECT_DIR"
unzip -n "$HANDOFF_ZIP" -d "$PROJECT_DIR"
cd "$PROJECT_DIR"

# AGENTS.md 应直接出现在项目根目录，不在多一层子目录中
ls -la
git init

# 只创建参考目录；不要先创建 packages/framework 和 packages/ops
mkdir -p external
git clone https://github.com/NVIDIA/nvalchemi-toolkit.git external/nvalchemi-toolkit
git clone https://github.com/NVIDIA/nvalchemi-toolkit-ops.git external/nvalchemi-toolkit-ops
```

使用完整 clone，不加 `--depth 1`。如服务器不能直连 GitHub，可用团队认可的完整镜像或 Git bundle 导入，保留真实来源 URL、完整历史和对象 SHA。已经存在的 clone 只检查，不重复执行 clone、不清空目录。

此时的两个 clone 是上游参考。不要在它们里面开始写 DCU 产品补丁。它们也不应出现在根仓库的待提交源码中：

```bash
git status --short
git -C external/nvalchemi-toolkit remote -v
git -C external/nvalchemi-toolkit rev-parse HEAD
git -C external/nvalchemi-toolkit-ops remote -v
git -C external/nvalchemi-toolkit-ops rev-parse HEAD
```

此刻打印出的 HEAD 只是实际 clone 到的版本，不自动等于后续锁定基线。下一步需检查两项目版本依赖、Python 和 Torch 等约束。

## 3. 启动 Codex CLI

安装与登录完成的 Codex CLI 从根目录运行：

```bash
cd "$PROJECT_DIR"
codex
```

项目已提供 AGENTS.md，通常不需要再运行 `/init`。`/init` 是交互命令，用于生成项目指令骨架，并不是让已有 AGENTS.md 生效的必要步骤。不要假定 shell 中存在 `codex init` 子命令。

第一次可以直接发送以下提示词：

```text
请读取根目录 AGENTS.md、docs/PROJECT_HANDOFF.md、docs/STATUS.md，
并按 AGENTS.md 中“第一次执行的具体任务”开始工作。

这是一项基于 NVIDIA nvalchemi-toolkit 与 nvalchemi-toolkit-ops 的海光 DCU 移植。
两个原始仓库已经 clone 到 external/。请先确认现状与指令来源，再检查本机环境、
源码依赖和上游版本兼容性，建立真实的功能清单、环境记录与可重跑探针。
确认兼容基线后，完整保留历史导入 packages/framework 和 packages/ops。

请保留上游 API、AtomicData/Batch、FusedStage/inflight、Hooks、训练和分布式设计，
采用 PyTorch reference、适配版 Triton 与 HIP 的后端路线。
不修改系统驱动/DTK，不替换系统中的海光 PyTorch，不影响他人 GPU 作业。
本轮请完成首轮任务范围内可执行的工作，并把真实结果和下一步写入 STATUS.md。
如果缺少依赖、网络或多卡资源，请明确记录，继续独立可完成的部分。
```

如果之前已运行 Codex，再复制了本包，请从项目根重新启动会话。可先让 Codex“列出已加载的项目指令，并复述项目目标与首个任务”，验证入口生效。若加载内容不符，检查根路径、全局指令和 `AGENTS.override.md`，不要直接删掉个人配置。

Codex 官方文档说明项目指令沿根目录到当前目录加载，默认累计大小限制为 32 KiB。本包根 AGENTS.md 控制在该范围内，详情以显式读取文档的方式展开。始终从项目根启动可减少进入 external 嵌套仓库后使用错误项目边界的风险。

参考：[AGENTS.md 加载规则](https://developers.openai.com/codex/guides/agents-md)、[/init 命令](https://developers.openai.com/codex/cli/slash-commands)。

## 4. 正式源码导入方式（由 Codex 完成也可以）

不要在版本兼容审计前直接运行这一节。条件是：已选出两份完整 SHA、编写 UPSTREAM_LOCK、根仓库没有未提交修改、目标 packages 子目录不存在，并确认 Git 支持 subtree。

根仓库需要首个提交。只提交实际检查过的交接与锁定文档，不用 `git add .` 意外收进环境或数据。若没有用户 Git 身份，应让用户配置真实身份，不能伪造作者。

```bash
git add AGENTS.md .gitignore docs
git commit -m "docs: initialize DCU port project guidance"
```

使用下面的命令结构，SHA 必须替换为审核后写入 lock 的真实提交。保留 `<...>` 占位符时不能运行：

```bash
FRAMEWORK_SHA="<兼容审计选定的 framework 完整 SHA>"
OPS_SHA="<兼容审计选定的 ops 完整 SHA>"

# 根仓库从本地参考 clone 获取完整对象，避免重复网络下载
git remote add upstream-framework "$PROJECT_DIR/external/nvalchemi-toolkit"
git remote add upstream-ops "$PROJECT_DIR/external/nvalchemi-toolkit-ops"
git fetch upstream-framework
git fetch upstream-ops

# 不使用 --squash，导入各自完整源码树并保留历史
git subtree add --prefix=packages/framework "$FRAMEWORK_SHA"
git subtree add --prefix=packages/ops "$OPS_SHA"
```

remote 已存在时检查而不是重复添加。若环境无 subtree，不要安装系统工具或改用丢失历史的随意复制；记录工具缺口，可使用团队已有 Git 环境完成导入，或先在 ADR 中说明可维护的替代流程。

这里的 `upstream-framework`、`upstream-ops` 是主仓库指向本地参考 clone 的取数远程。官方源 URL 记录在 external 的 origin 和 UPSTREAM_LOCK 中。项目移动路径后需检查根 remote 的路径。后续更新先在 external fetch 官方源，再在根 fetch 本地参考源，评估新 SHA；不自动切换产品版本。

两个 subtree 导入提交之后，再创建后端适配提交。不要为了方便直接删除许可证、测试、文档或原始构建文件。

## 5. 环境与开发入口

源码导入不等于可以直接执行上游安装命令。首先检查两个 pyproject 及 extras，避免拉取官方 CUDA wheel 或覆盖海光供应 PyTorch。用团队提供的镜像、虚拟环境和包源，记录当前 Python/Torch 构建信息，然后建立隔离的探针环境。

正式开发只使用 `packages/framework`、`packages/ops` 的安装源。NVIDIA 基线或 external 原版运行使用另一隔离环境，避免相同 import 命名空间覆盖。在报告中打印实际导入包的 `__file__`，保证测试对象确为当前移植代码。

第一次交付应能回答：本机有什么可用资源、哪个上游组合被锁定、哪些关键依赖可用、哪些探针真实通过、第一条闭环从哪里实现。不要把交接包里的计划表当成已完成的功能矩阵。
