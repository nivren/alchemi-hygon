# DCU 工具包维护细则：存量 enrich（不新建子 skill）

> 本文件承载 dcu-toolkit-skills 包的"维护操作流程"，因包根 SKILL.md 为包级入口不便频繁改，且本 enrich 实例就发生在 dcu-ontrack-experience，故作为参考放此。涉及"新增子 skill"的扩容流程见包根 SKILL.md。

## 背景（用户 2026-07-24 决策）
把一份大型/专项参考（如 HYGON 官方 XID/SXID 错误手册 PDF）并入包时，用户明确选择：**不新建子 skill**，而是作为 `references/<topic>.md` 并入最相关的**现有**子 skill，且**分开存放**以防数据膨胀、便于针对性查询。

## 存量 enrich 规范（不新建子 skill）

1. **references 分离存放**：专项内容写成 `<子skill>/references/<topic>.md`，与主 SKILL.md 分离。目的——防止主文数据量膨胀后混在一起，出问题能针对性定位到对应 references 文件查。
2. **主文只指路不抄内容**：在子 skill SKILL.md 的触发条件/关联段加一句指向 `references/<topic>.md` 的索引（如"XID 错误码 → 查 references/xid_sxid_errors.md"），**不要**把参考内容抄进主文，避免双源维护。
3. **不动根计数与版本号**：未新增子 skill 时，包根 SKILL.md 的"子 skill 清单（共 N 个）"、版本号、README 包结构树**均不改动**（数量没变）。仅 enrich 内部 references 不算版本变更。
4. **脱敏自检**（落地前必跑）：grep 确认 references 与改动无 `harbor.sourcefind.cn` / `image.sourcefind.cn` / 真实 IP / `密码` / `提取码` / 用户名（107/242/jixx 等）；内网域统一记 `<internal_registry>`。

## 实例
XID/SXID 错误手册（源自 HYGON 官方 PDF，双引擎交叉核对解析）作为 `references/xid_sxid_errors.md` 并入 `dcu-ontrack-experience`：
- SKILL.md 仅加：触发条件 2 条（XID/SXID 错误码场景）+ 关联段 1 条索引（指向 references）
- 根计数/版本不变（仍为 10 个子 skill、1.1.0）
- Git：按实际改动提交（commit 本地成功；push 需 sourcefind.cn 写凭证，本环境未配置 → 待用户提供 PAT/SSH）

## 关联
- 包根 SKILL.md：新增子 skill 的扩容同步规则（路由表+related_skills+计数+版本升位）
- skill-package-hygiene：发布包卫生（脱敏、README/SKILL.md 边界）
- document-extraction：PDF 多引擎交叉验证提取方法论（本实例的错误码手册即按此法解析）
