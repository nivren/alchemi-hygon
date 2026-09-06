# 上游锁定与导入依据

2026-09-05，按用户指定的两个完整 SHA 锁定，见 UPSTREAM_LOCK.yaml。
指令来源为本次用户指令、根 AGENTS.md、PROJECT_HANDOFF.md 和 START_HERE.md；历史路线文档与外部探索仅供参考。
初始主仓库 master 无提交，仅有未跟踪的交接文档和 .gitignore，无 packages 产品代码。已有真实 Git 身份 Wang, Leping / wangleping@dcu2。

两个 external origin 均为 NVIDIA 官方 GitHub URL，工作树干净，HEAD 正好等于锁定 SHA，is-shallow-repository=false。两仓库 `git fsck --full --no-dangling` 均退出 0；完整保留本地所含的可达历史，不声称核验远端所有 refs。根仓库从本地 clone 获取对象，使用不 squash subtree 独立导入提交；不修改 external。

framework 0.2.0 要求 Python >=3.11,<3.14、torch>=2.8、ops>=0.4.1；ops 0.4.1 要求 Python >=3.11,<3.15，torch extras >=2.8，基础依赖 warp-lang>=1.13.0、numpy。Python 3.12 / 海光 Torch 2.9 满足声明版本交集。AST 审计的 97 条 framework→ops 显式 from-import 均可从指定 ops 的模块/符号静态解析。脚本及完整 API、导入、测试、示例、配置和许可证清单见 probes/audit_upstream.py 与 reports/upstream_inventory.json。

这是**源码导入兼容基线**，不是 DCU 运行验证：静态符号解析不证明调用参数、动态导出、数值或二阶梯度正确。原始 Warp/NVIDIA 依赖尚未适配，未安装两包到探针环境。framework uv 的 dependency-metadata 对 ops 0.4.1 与本次源码依赖一致，但注释中的 Torch >=2.11 不适用于这两个锁定 pyproject 的实际 >=2.8 约束。不运行原包的 CUDA extras / uv sync。

两个根 LICENSE 均为 Apache-2.0；完整导入还保留 .licenses、内嵌第三方来源、SPDX、测试、CI 和文档。许可证清单记录于 lock/inventory，不将根许可证推广为所有第三方文件的许可证。

依赖风险：Hooks 直接使用 physicsnemo.utils.profiling；分布式还含 vendored upstream 与 shard 包装；不能整体删除 PhysicsNeMo 能力。当前 HCU 项目环境不安装该 NVIDIA/Warp 绑定包，单进程 Torch/HCU 通过可选导入路径，域并行与 profiling 保留为显式能力。MACE extra 固定 mace-torch==0.3.15，与 UMA 的 e3nn 版本要求存在冲突；当前项目环境使用 `mace-torch 0.3.15/e3nn 0.4.4`，探索环境使用 `mace-torch 0.3.16/e3nn 0.4.4`，cuEquivariance/UMA/compile 单列 C。保留 upstream tests；未运行的测试均不算通过。

旧探索 /home/wangleping/codes/nvalchemi-toolkit 和 /home/wangleping/codes/hyalchemi-ops 只读查阅了文件清单及后者 PROGRESS.md 的邻居/PBC/LJ 记录。其性能与数值记录未重跑、不纳入当前验证。后续按算子提取思路与测试案例，逐项检查来源、许可证和当前 SHA 语义后才移植。

## 实际导入记录

主分支 `codex/g0-initialization`；初始审计提交 `bd4c612`；framework subtree 提交 `52ffa2d`；ops subtree 提交 `88aa209`。两次导入均不 squash，锁定提交是对应导入提交的第二父提交，全部可达历史保留。

普通 fetch 仅获取 external 本地 main，而 external HEAD 锁定在另一个提交，首次 subtree 因对象不存在而退出 1、未导入目录。随后显式 `git fetch upstream-framework 4dfe3723def34df3fadb245981081ccf8c94c257` 和 `git fetch upstream-ops 26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`，再执行 subtree，均成功。不得将默认 main 当作锁定 SHA。

`probes/verify_import.py` 退出 0：两个 packages 子树 tree hash 与上游完全相同，两个 SHA 均是 HEAD 祖先，无嵌套 .git。证据 reports/import-verification.json。external 保持干净。嵌入的原 CI 文件仅作为来源保留，不会自动作为根 CI 生效。
