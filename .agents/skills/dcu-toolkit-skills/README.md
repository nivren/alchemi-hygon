# dcu-toolkit-skills

海光 DCU（深度计算单元）测试调优工具包 —— 可移植的 Hermes Agent / Claude Code / Cursor 等智能体 skill 集合。

覆盖：远程连接、环境检测、驱动安装、容器操作、拓扑采集、DCU 监控（hy-smi）、
NUMA 绑核优化、集合通信（RCCL）通测。设计为"知识+指导"，不绑定特定平台 API。

## 包结构

```
dcu-toolkit-skills/
├── README.md                包总览（本文件）
├── SKILL.md                 包级入口 + 路由表 + 文档边界（agent 看）
├── .gitignore               排除 vendor 工具 / 测试产物 / 真实凭证
├── install.sh               一键安装脚本（agent/人工安装入口）
├── data/                    共享数据层（devices.json / 凭证模板 / 模板 / 知识库）
├── references/              跨 skill 参考（型号映射 / 多机型验证 / git 推送）
├── scripts/                 共享脚本索引（scripts/README.md）
├── dcu-ssh/                SSH 远程连接管理（凭证隔离、免密规范）
├── dcu-env/                远程环境检测（OS/kernel/驱动/拓扑）
├── dcu-docker/             DCU 容器 Docker 操作（免密 SSH、容器内执行）
├── dcu-driver/             DCU 驱动安装 / 兼容矩阵 / 加载验证
├── dcu-dtk/                DTK 开发工具包下载 / 安装 / 激活（仅物理机）
├── dcu-topology/           拓扑采集（CPU/NUMA/卡↔NUMA，hy-smi 权威源）
├── dcu-hy-smi/             海光 DCU SMI 监控 / 拓扑查询 / 占用进程按容器反查
├── dcu-cpu-affinity/       NUMA 绑核方案生成（默认 dry-run）
├── dcu-rccl-test/          集合通信（RCCL）通测
└── dcu-ontrack-experience/ 故障排查经验速查（驱动/容器/vLLM/RCCL/编译；含 XID/SXID 错误码速查）
```

每个子目录内有独立的 `README.md`（用户向）与 `SKILL.md`（agent 向），详见各路径。

## 设计原则

1. **知识+指导，不绑平台**：用标准命令（ssh/scp/bash/python）和标准格式（JSON/YAML），
   放到任何支持 skills 的智能体环境都能用
2. **凭证物理隔离**：真实密码在 `~/.config/dcu-toolkit/credentials.json`（chmod 600），不进包；
   包内仅有 `credentials-template.json` 模板
3. **自包含、可复用**：每个子 skill 内聚了自身的踩坑（Pitfalls）、命令模板与实测结论，clone 下来即可用，不依赖外部私有文档
4. **默认不启用高危操作**：绑核（dcu-cpu-affinity）默认 dry-run；驱动/设置类需显式确认

## 使用

将本仓库目录（保持名 `dcu-toolkit-skills`）放到 agent 的 skills 搜索路径下即自动加载：
- Hermes：`~/.hermes/skills/dcu-toolkit-skills/`
- Claude Code：`~/.agents/skills/dcu-toolkit-skills/` 或项目内 `.claude/skills/dcu-toolkit-skills/`

加载后 agent 按意图自动路由到对应子 skill（路由表见根 `SKILL.md`）。

**agent 自行安装**：直接把下面这句发给 agent 即可，它会自动 clone 并安装整个包（含全部 10 个子 skill），无需人工逐步操作：

> 帮我安装一下这些skills技能 https://developer.sourcefind.cn/codes/jixx/dcu-toolkit-skills

## 版本

- dcu-toolkit-skills 1.1.1
- 子 skill（共 10 个）：dcu-ssh / dcu-env / dcu-docker / dcu-driver / dcu-dtk / dcu-topology / dcu-hy-smi / dcu-cpu-affinity / dcu-rccl-test / dcu-ontrack-experience
- 变更：1.1.0 → 1.1.1 文档对齐——根路由表厘清 dcu-topology（拓扑采集已实现）与 dcu-cpu-affinity（绑核方案生成）的职责边界；补 dcu-hy-smi / dcu-ontrack-experience 的 version 字段；related_skills 补 dcu-dtk。无新增子 skill、无破坏性变更。
- 历史：1.0.0 → 1.1.0 新增 `dcu-ontrack-experience`（故障排查经验速查）；工作流类方法论（如工单 CSV 转排查 skill）不纳入本包，存于个人 Hermes skills 目录
