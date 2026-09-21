#!/usr/bin/env bash
# ===== dcu-toolkit-skills 一键安装（从 git 仓库 URL，幂等）=====
# 用法: bash install.sh <repo_url>
#   repo_url 缺省则用下面 DEFAULT_REPO（用户发来的仓库地址）
# 行为: 探测 skills 路径 → git clone(已存在则 pull) → 校验包完整 → 幂等生成凭证骨架
set -euo pipefail

DEFAULT_REPO="https://developer.sourcefind.cn/codes/jixx/dcu-toolkit-skills"
REPO="${1:-$DEFAULT_REPO}"
NAME="dcu-toolkit-skills"

# 0. 前置检查：git 是否可用
command -v git >/dev/null 2>&1 || { echo "ERR: 未安装 git，请先安装 git 后重试" >&2; exit 1; }

# 1. 探测 skills 安装目标路径（Hermes 优先，其次 Claude Code）
if [ -d "$HOME/.hermes/skills" ]; then
  DEST="$HOME/.hermes/skills/$NAME"
elif [ -d "$HOME/.agents" ] || [ -w "$HOME/.agents" ] 2>/dev/null; then
  DEST="$HOME/.agents/skills/$NAME"
else
  DEST="$HOME/.hermes/skills/$NAME"     # 默认兜底到 Hermes
fi
echo "目标路径: $DEST"

# 2. clone / 更新（已存在则拉最新，不覆盖本地凭证等用户修改）
if [ -e "$DEST/.git" ]; then
  echo "SKIP: 已 clone，拉取最新..."
  git -C "$DEST" pull --ff-only
else
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 "$REPO" "$DEST"
  echo "OK: 已 clone 到 $DEST"
fi

# 3. 校验包完整（根 SKILL.md + 全部子 skill 的 SKILL.md）
[ -f "$DEST/SKILL.md" ] || { echo "ERR: 根 SKILL.md 缺失，仓库内容异常" >&2; exit 1; }
SUBS=(dcu-ssh dcu-env dcu-docker dcu-driver dcu-dtk dcu-topology dcu-hy-smi dcu-cpu-affinity dcu-rccl-test dcu-ontrack-experience)
miss=0
for s in "${SUBS[@]}"; do
  [ -f "$DEST/$s/SKILL.md" ] || { echo "ERR: 子 skill 缺失: $s" >&2; miss=1; }
done
[ "$miss" -eq 0 ] && echo "OK: 包完整，10 个子 skill 齐全"

# 4. 凭证骨架（不进包，真实密码留占位符由用户/环境后续填入）
CONF_DIR="$HOME/.config/dcu-toolkit"
CRED="$CONF_DIR/credentials.json"
if [ -f "$CRED" ]; then
  echo "SKIP: $CRED 已存在，保留现有凭证"
else
  mkdir -p "$CONF_DIR"
  cat > "$CRED" <<'JSON'
{
  "_comment": "实际凭证文件。真实密码请由用户/环境填入，切勿提交进 skill 包。",
  "dcu-node-01": { "password": "YOUR_PASSWORD_HERE" }
}
JSON
  chmod 600 "$CRED"
  echo "OK: 已生成凭证骨架 $CRED (chmod 600)，请填入真实密码"
fi

# 4b. 设备清单骨架（不进包，真实设备数据留占位符由用户/环境后续填入）
DATA_DIR="$DEST/data"
DEV="$DATA_DIR/devices.json"
if [ -f "$DEV" ]; then
  echo "SKIP: $DEV 已存在，保留现有设备清单"
else
  mkdir -p "$DATA_DIR"
  cat > "$DEV" <<'JSON'
{
  "_comment": "设备清单骨架（非敏感占位）。真实设备数据（IP/机型/凭证映射）由使用者填入本文件；切勿把真实设备信息提交进 skill 包 git 仓库。",
  "_fields": {
    "host": "服务器IP地址（占位示例：<device_ip>）",
    "port": "SSH端口，默认22",
    "user": "SSH用户名（占位示例：<user>）",
    "auth_method": "认证方式: password | key | config",
    "dcu_type": "DCU卡型号（占位示例：<dcu_type>）",
    "dcu_count": "DCU卡数量",
    "cpu": "CPU型号",
    "memory": "内存大小",
    "note": "设备用途说明"
  },
  "devices": {
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
}
JSON
  echo "OK: 已生成设备清单骨架 $DEV（占位符，请填入真实设备信息）"
fi

# 5. 完成
echo "DONE: 全部安装完成。agent 请读取 $DEST/SKILL.md 的路由表，按用户意图分发到子 skill。"
