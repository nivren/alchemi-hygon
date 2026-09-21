---
name: dcu-ssh
description: "Use when connecting to DCU servers via SSH for remote command execution. Defines device inventory, credential isolation, security rules (sshpass -f not -p), and execution patterns for short/long commands and file transfer. Foundation skill for all dcu-toolkit sub-skills."
license: MIT
metadata:
  version: 1.0.0
  author: DCU Engineer
  hermes:
    tags: [dcu, ssh, remote, security, infrastructure]
    related_skills: [dcu-env, dcu-docker, dcu-driver, dcu-topology, dcu-hy-smi, dcu-cpu-affinity, dcu-rccl-test]
---

# DCU SSH 远程连接管理

## Overview

指导智能体如何安全地通过SSH连接海光DCU服务器并执行远程操作。所有其他dcu-toolkit子skill在执行远程命令前，均需遵循本skill的规范。

## When to Use

- 需要SSH到DCU服务器执行任何远程命令
- 需要在DCU服务器上部署脚本、运行测试、收集结果
- 任何dcu-toolkit子skill执行远程操作前的连接准备

Don't use for: 本地操作；非DCU服务器的SSH连接（但规范通用）。

## Security Rules (强制，不可违反)

1. **永远不要在skill包内存储密码或密钥**
2. 凭证文件路径：`~/.config/dcu-toolkit/credentials.json`（必须 `chmod 600`）
3. 密码认证必须用 `sshpass -f` 或 `sshpass -e`，**禁止 `sshpass -p`**（密码会出现在进程列表）
4. 临时密码文件用完立即删除
5. 密钥文件权限必须 `chmod 600`
6. 如果检测到SSH config别名存在，优先使用（无需密码）
7. SSH连接加 `-o StrictHostKeyChecking=no -o ConnectTimeout=10` 避免交互式卡住
8. **用户若在对话中直接贴明文令牌/密码**：仅用于当次操作（克隆/登录），用完即弃，**绝不写进 skill 包、不写进 credentials.json 之外的文件、不回显到输出**；并提示用户去对应平台吊销/重发该令牌（对话/日志会留存明文，视为已泄露）。参考 dcu-toolkit 的"私有仓库令牌克隆"段。

## Device Inventory

设备信息存储在 `~/.hermes/skills/dcu/data/devices.json`（仅非敏感信息）：

```json
{
  "<device_name>": {
    "host": "<device_ip>",
    "port": 22,
    "user": "<user>",
    "auth_method": "password",
    "dcu_type": "<dcu_type>",
    "dcu_count": 8,
    "cpu": "<cpu_model>",
    "memory": "<memory_size>",
    "note": "示例设备（请替换为真实设备，且勿提交进 git）"
  }
}
```

字段说明：
- `auth_method`: "password" 或 "key" 或 "config"
- `dcu_type`: DCU卡型号（Z100/Z100L/K100-AI/K100等）
- `dcu_count`: DCU卡数量

## Credential Reading

凭证文件位于skill包外部：`~/.config/dcu-toolkit/credentials.json`（chmod 600）

```json
{
  "<device_name>": { "password": "YOUR_PASSWORD_HERE" },
  "dcu-node-02": { "key_path": "~/.ssh/dcu_keys/node02.key" }
}
```

执行步骤：
1. 读取 `data/devices.json` 获取目标设备连接信息
2. 读取 `~/.config/dcu-toolkit/credentials.json` 获取凭证
3. 根据auth_method选择连接方式（见下文）

如果credentials.json不存在，提示用户创建：
```bash
mkdir -p ~/.config/dcu-toolkit
cat > ~/.config/dcu-toolkit/credentials.json << 'EOF'
{
  "<device_name>": { "password": "YOUR_PASSWORD_HERE" }
}
EOF
chmod 600 ~/.config/dcu-toolkit/credentials.json
```

## Connection Methods

### Method 1: Password Auth (sshpass -f)

```bash
# 1. 从凭证文件读取密码，写入临时文件
PASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['<device_name>']['password'])")
echo "$PASS" > /tmp/.dcu_pass_$$ && chmod 600 /tmp/.dcu_pass_$$

# 2. 执行SSH
sshpass -f /tmp/.dcu_pass_$$ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@<device_ip> "hy-smi"

# 3. 清理临时文件
rm -f /tmp/.dcu_pass_$$
```

### Method 1b: Password Auth (sshpass -e, 更简洁)

`sshpass -e` 从环境变量 `SSHPASS` 读密码，免去临时文件创建/清理：

```bash
export SSHPASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['<device_name>']['password'])")
sshpass -e ssh -o StrictHostKeyChecking=no root@<device_ip> "hy-smi"
# 批量循环时尤其方便（for hp in ...; do export SSHPASS=...; sshpass -e ssh ...; done）
# 注意：SSHPASS 仅存在于当前 shell 进程环境，不落盘、不留进程参数，比 -p 安全
```

### Method 2: Key Auth

```bash
ssh -i ~/.ssh/dcu_keys/node01.key -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@<device_ip> "hy-smi"
```

### Method 3: SSH Config Alias (推荐，最简洁)

先配置 `~/.ssh/config`：
```
Host dcu-node-01
  HostName <device_ip>
  User root
  Port 22
  IdentityFile ~/.ssh/dcu_keys/node01.key
  StrictHostKeyChecking no
```

然后直接：
```bash
ssh dcu-node-01 "hy-smi"
```

不需要知道IP、密码、密钥路径，全部委托给SSH config。

## Command Execution Patterns

### Short Command (< 30s, 直接执行)

```bash
sshpass -f /tmp/.dcu_pass ssh -o StrictHostKeyChecking=no root@<device_ip> "hy-smi"
```

直接取stdout作为结果。适用于：hy-smi、lscpu、free、cat配置文件等快速命令。

### Long Command (> 30s, 后台执行+轮询)

适用于benchmark、烤机、模型推理测试等长时间任务。

```bash
# 1. 上传脚本到服务器
scp -o StrictHostKeyChecking=no run_bench.sh root@<device_ip>:/tmp/

# 2. 后台启动，记录PID
PID=$(sshpass -f /tmp/.dcu_pass ssh root@<device_ip> "nohup bash /tmp/run_bench.sh > /tmp/dcu_task.log 2>&1 & echo \$!")

# 3. 轮询日志（每隔30-60秒）
sshpass -f /tmp/.dcu_pass ssh root@<device_ip> "tail -20 /tmp/dcu_task.log"

# 4. 检查进程是否结束
sshpass -f /tmp/.dcu_pass ssh root@<device_ip> "ps -p $PID"
# 返回非0 → 进程已结束，任务完成

# 5. 下载结果文件
scp -o StrictHostKeyChecking=no root@<device_ip>:/tmp/result.json ./
```

### File Transfer

```bash
# 上传单个文件
scp -o StrictHostKeyChecking=no local_script.sh root@<device_ip>:/tmp/

# 下载单个文件
scp -o StrictHostKeyChecking=no root@<device_ip>:/tmp/result.json ./

# 批量传输（rsync更高效）
rsync -avz -e "ssh -o StrictHostKeyChecking=no" ./scripts/ root@<device_ip>:/tmp/dcu_scripts/
```

## Connection Verification

首次连接设备时，先验证连接：

```bash
ssh root@<device_ip> "echo CONNECTION_OK && hostname && whoami"
```

输出包含 `CONNECTION_OK` 则连接正常。

## DTK Environment Loading

DCU服务器上的DTK工具需要加载环境变量。远程执行时在命令前加source：

```bash
ssh root@<device_ip> "source /opt/dtk/env.sh && hy-smi"
```

如果DTK路径不确定，先检测：
```bash
ssh root@<device_ip> "ls /opt/dtk*/env.sh 2>/dev/null || find /opt -name 'env.sh' -path '*dtk*' 2>/dev/null"
```

## Migration to Key Auth (推荐)

密码认证能跑但安全性不如密钥认证。建议一次性迁移：

```bash
# 1. 生成密钥
ssh-keygen -t ed25519 -f ~/.ssh/dcu_keys/node01.key -N ""

# 2. 部署到服务器（需密码，一次性操作）
sshpass -f /tmp/.dcu_pass ssh-copy-id -i ~/.ssh/dcu_keys/node01.key root@<device_ip>

# 3. 配置SSH config
cat >> ~/.ssh/config << 'EOF'
Host dcu-node-01
  HostName <device_ip>
  User root
  Port 22
  IdentityFile ~/.ssh/dcu_keys/node01.key
  StrictHostKeyChecking no
EOF

# 4. 测试
ssh dcu-node-01 "echo OK"

# 5. 之后所有操作直接用别名，无需密码
```

## Common Pitfalls

1. **用 sshpass -p 传密码** — 密码出现在 `ps aux` 进程列表中。必须用 `-f` 或 `-e`
2. **忘记清理临时密码文件** — 用完必须 `rm -f /tmp/.dcu_pass_*`
3. **长命令阻塞SSH** — 超过30秒的命令不用直接执行模式，用后台+轮询
4. **忘记source DTK环境** — DTK 工具链（dtk 编译器/hipcc/roc* 等）命令找不到。远程命令前加 `source /opt/dtk/env.sh`（路径不确定时先 `ls /opt/dtk*/env.sh`）。
   **例外：hy-smi 不需要 source DTK** — 它是 DCU 驱动自带的，装驱动后就在 `/opt/hyhal/bin/hy-smi`（宿主机与容器内均有，PATH 可能不含）。直接调绝对路径即可：`/opt/hyhal/bin/hy-smi`。若绝对路径不存在，再考虑 `docker exec <dcu容器> hy-smi`（见 dcu-topology / references/hy-smi.md）。
5. **StrictHostKeyChecking弹交互** — 首次连接卡住。必须加 `-o StrictHostKeyChecking=no`
6. **credentials.json权限不对** — 必须 `chmod 600`，否则有泄露风险
7. **sshpass未安装** — 检查并安装：`apt install sshpass` 或 `yum install sshpass`
8. **用户在对话贴明文凭证** — 用户可能直接把 root 密码、gitee/gitlab 私人令牌贴进对话。处理：当次用、不落盘、不回显，用完提示吊销令牌；密码如需持久化，只写进 `~/.config/dcu-toolkit/credentials.json`（chmod 600），不进 skill 包

9. **credentials.json 格式错误 → SSH Permission denied（连不上）** — 实测 2026-07-22：credentials.json 被存成错误结构（`devices` 字段是 IP 字符串数组，而非"顶层 key=设备名 + `password` 字段"的字典），导致 `c['<设备名>']['password']` 取空密码、sshpass 传空、服务器回 Permission denied。连设备前务必先验格式。正确格式（顶层 key 必须与 devices.json 的设备名完全一致）：
   ```json
   {
     "_comment": "DCU 设备凭证 — chmod 600",
     "<device_name_1>": { "password": "<from-secure-store>" },
     "<device_name_2>": { "password": "<from-secure-store>" }
   }
   ```
   **连设备前的格式自检/自愈**（只打印长度，不泄露明文）：
   ```python
   import json, os, sys
   p = os.path.expanduser('~/.config/dcu-toolkit/credentials.json')
   c = json.load(open(p)); dev = '<设备名>'   # 取自 devices.json
   if dev not in c or not isinstance(c.get(dev), dict) or 'password' not in c.get(dev, {}):
       print(f"WARN: {dev} 凭证格式错误/缺失，请修正 credentials.json"); sys.exit(2)
   print("password len:", len(c[dev]['password']))
   ```
   修正/重建用 python `json.dump` 写回并 `chmod 600`，**不要**用 echo/cat heredoc 写含密码文件。含密码的 credentials.json 绝不提交进任何 git 仓库（`~/.config` 本就不在 skills git 内）。

10. **不要假设同机房/同批次设备共用同一密码** — 实测发现同网段、同一批交付的设备，实际密码也可能不同。把 A 设备的密码套到 B 设备会导致 `Permission denied`。连一台失败时报 Permission denied，**先核对这台设备自己的凭证**，不要跨设备套用，也不要假设"同一批贴的就一定相同"。每台独立存、独立试。

11. **scp 指定端口用大写 `-P`，不是 `-p`** — 实测 2026-07-22：scp 里 `-p` 是"保留时间戳"，端口参数是大写 `-P`。写成 `scp -p 22 ...` 会被当成保留时间戳、忽略端口，接着因找不到默认22以外的路径或源文件而报 `No such file or directory`（误导性报错）。正确：`scp -o StrictHostKeyChecking=no -P 22 local_file root@host:/tmp/`。ssh/sshpass 的端口参数才是小写 `-p`。注意区分：ssh `-p 22`、scp `-P 22`、sshpass 包 ssh 时透传的是 ssh 的小写 `-p`。

12. **多机测试需免密通信** — 双机/多机 rccl-test（mpirun）依赖节点间 root SSH 免密。物理机免密：关 firewalld/iptables → 各机 `ssh-keygen -t rsa` → `/etc/hosts` 配管理网 IP 与主机名 → `ssh-copy-id` 互信（用管理网 IP，非 IB/ROCE 网）。容器免密见 dcu-docker。来自内部测试手册"基本环境准备"。

## Verification Checklist

- [ ] devices.json中目标设备信息完整（host/port/user/auth_method）
- [ ] credentials.json存在且权限为600，且格式正确（顶层 key=设备名、`password` 字段存在，非 IP 字符串数组）—— 见 pitfall 9
- [ ] SSH连接验证通过（输出CONNECTION_OK）
- [ ] 临时密码文件已清理（用 `-e` 则无需）
- [ ] DTK环境变量已加载（如需要；hy-smi 不需要）
- [ ] 长命令使用后台+轮询模式
- [ ] 结果文件已下载到本地（如需要）
- [ ] **跨系统验证**：DCU 服务器混用 Ubuntu(5.15) 与 Kylin V10(4.19) 老内核，命令行为差异大（nproc/numactl/lspci 等）。脚本改动需在两种系统都验证，双机 harness 见 dcu-toolkit `../references/dual-machine-verification.md`
