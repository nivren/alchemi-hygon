---
name: dcu-toolkit-skills
description: "DCU 测试调优工具包总入口（包根）。海光 DCU 相关测试、调优、项目支持（标前/验收/POC/FAE）的可移植 skill 集合。本文件是包级入口：路由表、安装（agent 执行）、安全说明。各子 skill 在同级目录平铺。"
license: MIT
metadata:
  version: 1.1.1
  author: DCU Engineer
  hermes:
    tags: [dcu, hygon, testing, tuning, benchmark, infrastructure, docker, container]
    related_skills: [dcu-ssh, dcu-env, dcu-docker, dcu-driver, dcu-dtk, dcu-topology, dcu-hy-smi, dcu-cpu-affinity, dcu-rccl-test, dcu-ontrack-experience]
---

# DCU Toolkit Skills — 包级总入口

本目录是 **dcu-toolkit-skills** 工具包的仓库根，是**顶层路由器 + 共享数据层**，根据用户意图把 DCU 任务分发到同级平铺的各子 skill。兼容 Hermes Agent、Claude Code、Cursor 等智能体环境。

## 路由表（按用户意图分发）

| 用户意图关键词 | 分发目标 | 状态 |
|---|---|---|
| SSH 连接 / 远程 / 登录设备 | dcu-ssh | ✅ |
| 环境检测 / 设备信息 | dcu-env | ✅ |
| 驱动安装 / 驱动升级 / rock.run / hymgr / 驱动加载验证 / lsmod / hy-smi 设备可见 | dcu-driver | ✅ |
| DTK 安装 / DTK 激活 / /opt/dtk / source env.sh / hipcc / DTK 版本选型 / 框架编译前置 | dcu-dtk | ✅ |
| 拓扑 / NUMA / CPU 亲和性采集 / 卡↔NUMA | dcu-topology | ✅ 采集已实现（多机型验证） |
| 绑核 / numactl / CPU 亲和方案生成 | dcu-cpu-affinity | ✅ 方案生成（默认 dry-run） |
| DCU 监控 / hy-smi / 温度 / 功耗 / 利用率 / 显存 / 进程 / 按容器反查 | dcu-hy-smi | ✅ |
| DCU 拓扑查询 / 卡↔NUMA 亲和性 / 卡间链接 | dcu-hy-smi | ✅ |
| 绑核 / numactl / CPU 亲和 | dcu-cpu-affinity | ✅ 方案生成（默认 dry-run） |
| 集合通信 / RCCL / allreduce / alltoall / 多卡互联带宽 | dcu-rccl-test | ✅ |
| 故障排查 / 经验反查 / 报错关键词速查 / 驱动·容器·vLLM·RCCL·编译踩坑 | dcu-ontrack-experience | ✅ |
| 性能基准 / benchmark / 烤机 / 精度 / 量化 / 标前 / 验收 / 售后排查 / POC | （对应规划中子 skill） | 规划中 |

若目标子 skill 尚未创建，告知用户该功能在规划中，并建议优先创建。

## 子 skill 清单（共 10 个）

| 子 skill | 职责 | 状态 |
|---|---|---|
| dcu-ssh | SSH 连接 / 远程登录设备 | ✅ |
| dcu-env | 环境检测 / 设备信息 | ✅ |
| dcu-docker | DCU 容器启动 / 容器间 SSH 免密 / 测评执行方式 | ✅ |
| dcu-driver | 驱动安装 / 升级 / 加载验证 / 兼容矩阵 | ✅ |
| dcu-dtk | DTK 安装 / 激活 / 版本选型 / 框架编译前置 | ✅ |
| dcu-topology | 拓扑 / NUMA / CPU 亲和性采集 / 卡↔NUMA | ✅ |
| dcu-hy-smi | DCU 监控 / 温度 / 功耗 / 利用率 / 显存 / 按容器反查 | ✅ |
| dcu-cpu-affinity | 绑核 / numactl / CPU 亲和方案生成 | ✅ |
| dcu-rccl-test | 集合通信 RCCL 测试 / 多卡多机带宽 | ✅ |
| dcu-ontrack-experience | 故障排查经验速查（驱动/容器/vLLM/RCCL/编译，按现象反查根因） | ✅ |

> 版本：1.1.1（1.1.0 → 1.1.1：文档对齐——根路由表厘清 dcu-topology 与 dcu-cpu-affinity 的绑核职责边界；补 dcu-hy-smi / dcu-ontrack-experience 的 version 字段；related_skills 补 dcu-dtk。无新增子 skill、无破坏性变更）。
> 工作流类方法论（如工单 CSV → 排查 skill）不纳入本发布包，存于个人 Hermes skills 目录。

## 发布前文档一致性自检（每次改完 skills 必跑）

文档漂移是本包最高频的回归点。每次改完任意子 skill 或根文档，**提交推送前**按此清单核对一遍（用 grep/目录遍历，不要凭记忆）：

1. **根 README 结构树 vs 实际目录**：逐一比对 `README.md` 的 ```` ``` ```` 树块与实际 `ls` 一级条目。常见漏项：
   - 根级脚本/文件（如 `install.sh`）被遗忘在树外；
   - `scripts/`、`data/`、`references/` 目录形式不统一（有的画成 `scripts/README.md` 文件、有的画成目录）。
2. **根 SKILL.md frontmatter `metadata.hermes.related_skills`**：必须与 `dcu-*` 实际子目录**逐项对齐**（含 `dcu-dtk`！曾漏过）。缺一个就补，多一个就删。
3. **子 skill `metadata.version` 字段**：每个子 skill 的 SKILL.md frontmatter 都应带版本号（统一 `1.0.0` 起步）。漏带就补。
4. **根文档计数与版本**：根 README 与根 SKILL.md 的「共 N 个」「version」必须一致；改动落地后按语义升版本（纯文档对齐升补丁位，如 1.1.0→1.1.1）。
5. **路由表职责边界**：路由表/子 skill 清单里每条意图→分发目标的「状态」标记必须与子 skill 自身声明一致。曾出现根路由表把「绑核」同时归给 dcu-topology 并标「已实现」，而 dcu-topology 自己声明绑核「未实现」——正确拆分：拓扑/卡↔NUMA 采集→dcu-topology（已实现），绑核方案生成→dcu-cpu-affinity（已实现，dry-run）。
6. **references 可见性**：子 skill 的 `references/*.md` 不能只躺在目录里。子 skill 自己的 `README.md` 要列一张 references 速查表（文件名+用途）；若该参考是核心能力（如 dcu-ontrack-experience 的 `xid_sxid_errors.md` XID/SXID 错误码速查），**根 README 结构树对应子 skill 那一行也要点出**，否则用户/agent 读 README 时完全不知道有这个能力。

> 注意：`pack-enrich-maintenance.md` 规定「未新增子 skill 时根文档计数/版本不改动」——这条指**数量与版本号**，不等于「根 README 可以不提新并入的 references 能力」。把已有参考**显式暴露**到 README 是文档补全，不是 enrich 扩容，照常做、不冲突。

## 安装（agent 执行）

用户把仓库地址发给 agent（如「帮我安装一下这些skills技能 <repo_url>」）后，agent 执行包内安装脚本即可完成：自动探测 skills 路径、从仓库 URL clone（已存在则 pull 更新）、校验包完整、幂等生成凭证骨架。

- 脚本位置：`install.sh`（仓库根，与本文件同级）
- 用法：`bash install.sh <repo_url>`；`<repo_url>` 缺省则使用内置默认仓库地址
- 脚本幂等、带 `set -euo pipefail`、明确状态输出（OK/SKIP/ERR）、失败非 0 退出；真实密码只在凭证骨架留 `<password>` 占位，绝不写死

## 安全说明（技术知识）

- 包内零密码、零密钥：真实凭证在 `~/.config/dcu-toolkit/credentials.json`（chmod 600），包内仅有 `credentials-template.json` 模板
- 远程密码认证用 `sshpass -f <文件>` 或 `sshpass -e`（从环境变量读），**不要**用 `sshpass -p`（密码会出现在进程列表）
- 所有脚本用标准命令（ssh/scp/bash/python），不绑定特定智能体平台 API
