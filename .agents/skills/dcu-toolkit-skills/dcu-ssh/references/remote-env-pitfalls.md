# DCU SSH 远程环境 — 实测陷阱与技巧

> 本文件记录 dcu-ssh 在实际 DCU 服务器上验证时发现的陷阱与可复用技巧。
> 这些是从真实会话中提炼的，优先于 SKILL.md 中的通用假设。

## 1. DTK 环境变量路径（重要）

**SKILL.md 原文假设 `source /opt/dtk/env.sh` 是错误的，至少对以下环境不成立。**

实测机器（老内核 Kylin V10 机型）：
- `/opt/dtk/env.sh` **不存在**
- hy-smi 实际路径：`/opt/hyhal/bin/hy-smi`，且**已在 PATH 中**，无需 source

**正确做法**：
```bash
# 先确认 hy-smi 是否已在 PATH（大多数预装环境如此，直接可用）
ssh root@<HOST> "which hy-smi && hy-smi --version"
# 实测输出: /opt/hyhal/bin/hy-smi  Version 1.24.0

# 仅当 hy-smi 不在 PATH 时，才探测 env 文件再 source
ssh root@<HOST> "ls /opt/dtk*/env.sh /opt/hyhal*/env.sh 2>/dev/null; find / -name 'env.sh' -path '*dtk*' 2>/dev/null | head"
```

**不要无脑在每条远程命令前加 `source /opt/dtk/env.sh`** —— 文件不存在时整条命令报
`没有那个文件或目录` 并失败。

## 2. sourcefind.cn 仓库克隆（获取 DCU 工具/参考代码）

DCU 相关开源仓库托管在 `https://developer.sourcefind.cn/codes/tsoc/<repo>`。

**正确获取方式：直接 git clone（可用）**
```bash
git clone --depth 1 https://developer.sourcefind.cn/codes/tsoc/dcuprofiletools.git
```
实测可成功克隆（无需登录、无反爬拦截）。

**注意**：`wechat-article-spider` skill 仅用于抓取微信公众号文章，**不适用于 sourcefind.cn**。
不要用它来抓 DCU 工具仓库——直接 git clone 即可。

## 3. devices.json 的 auth_method 字段语义（数据契约）

`auth_method` 字段填的是**认证方式**，不是密码本身：
- `"password"` — 用密码（从外部 credentials.json 读取）
- `"key"` — 用密钥（credentials.json 里填 `key_path`）
- `"config"` — 用 SSH config 别名（无需密码/密钥路径）

**密码/密钥永远不进 devices.json**，只进包外 `~/.config/dcu-toolkit/credentials.json`（chmod 600）。
（用户初次录入时曾把密码直接填进 auth_method，已纠正。）
