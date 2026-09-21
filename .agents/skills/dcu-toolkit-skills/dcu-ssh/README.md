# dcu-ssh — 海光 DCU 远程连接管理

SSH 远程连接管理 skill。定义 DCU 服务器的连接规范、凭证隔离策略、免密登录配置，
以及配合 sshpass 的安全用法。所有对 DCU 设备的远程操作都应以本 skill 的规范为入口。

## 核心内容

- **凭证隔离**：真实密码在 `~/.config/dcu-toolkit/credentials.json`（chmod 600），
  包内只有 `data/credentials-template.json` 模板
- **连接方式**：`sshpass -f <cred_file> ssh -o StrictHostKeyChecking=no root@<host>`
- **免密 SSH**：推荐 SSH key + config 别名（见 `references/remote-env-pitfalls.md`）
- **多设备批量**：遍历 devices.json 批量执行

## 目录

```
dcu-ssh/
├── SKILL.md                       连接规范 + 安全纪律
└── references/
    └── remote-env-pitfalls.md     远程操作踩坑（含凭证模板、sshpass 用法、白名单IP）
```

## 快速用法

```bash
# 从凭证文件读密码免密登录
sshpass -f ~/.config/dcu-toolkit/credentials.json ssh -o StrictHostKeyChecking=no root@<device_ip> "hostname"

# 多机批量执行（示例）
for h in <device_ip_1> <device_ip_2>; do
  sshpass -f ~/.config/dcu-toolkit/credentials.json ssh root@$h "uptime"
done
```

详见 `SKILL.md` 与 `references/remote-env-pitfalls.md`。
