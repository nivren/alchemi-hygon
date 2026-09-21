#!/bin/bash
# check_driver.sh — 海光DCU驱动加载验证（兼容 6.2 hydcu / 6.3 hycu）
# 用途：安装驱动后确认模块加载、hymgr服务、设备节点、hy-smi可用性
# 用法：bash check_driver.sh        （本地执行）
#       ssh root@host "bash -s" < check_driver.sh   （远程，配合 dcu-ssh）
set -uo pipefail

echo "==================================================================="
echo " 海光 DCU 驱动加载验证"
echo "==================================================================="

# 1. 内核模块（兼容 hydcu / hycu）
echo "--- 1. 内核模块 (lsmod | grep -E hydcu|hycu) ---"
MODS=$(lsmod 2>/dev/null | grep -E "hydcu|hycu" || true)
if [[ -n "$MODS" ]]; then
  MOD_NAME=$(echo "$MODS" | head -1 | awk '{print $1}')
  echo "$MODS"
  if [[ "$MOD_NAME" == hycu* ]]; then
    echo "  ✅ 驱动已加载 (6.3.x 模块名: hycu)"
  else
    echo "  ✅ 驱动已加载 (≤6.2 模块名: hydcu)"
  fi
else
  echo "  ❌ 未检测到 hydcu/hycu 模块 — 驱动可能未加载"
fi

# 2. hymgr 服务
echo "--- 2. hymgr 服务 ---"
if command -v systemctl >/dev/null 2>&1; then
  if systemctl is-active --quiet hymgr 2>/dev/null; then
    echo "  ✅ hymgr active"
  else
    echo "  ⚠️  hymgr 未运行 — 试: systemctl restart hymgr"
  fi
else
  echo "  (无 systemctl，跳过)"
fi

# 3. 设备节点
echo "--- 3. 设备节点 (/dev/kfd, /dev/dri) ---"
[[ -e /dev/kfd ]] && echo "  ✅ /dev/kfd 存在" || echo "  ❌ /dev/kfd 缺失"
[[ -d /dev/dri ]] && echo "  ✅ /dev/dri 存在" || echo "  ❌ /dev/dri 缺失"

# 4. hy-smi 可用性与设备数
echo "--- 4. hy-smi 设备概览 ---"
HY=""
[[ -x /opt/hyhal/bin/hy-smi ]] && HY=/opt/hyhal/bin/hy-smi
command -v hy-smi >/dev/null 2>&1 && HY=hy-smi
if [[ -n "$HY" ]]; then
  COUNT=$($HY 2>/dev/null | grep -cE "^[ ]*[0-9]+[ ]" || true)
  echo "  hy-smi: $HY"
  $HY 2>/dev/null | head -12
  echo "  (设备行数≈$COUNT，具体以表格为准)"
else
  echo "  ⚠️  hy-smi 未找到（应位于 /opt/hyhal/bin/hy-smi，驱动自带）"
fi

# 5. 容器内调试挂载提示（仅信息）
echo "--- 5. 容器调试挂载检查 (/sys/kernel/debug) ---"
if [[ -d /sys/kernel/debug ]]; then
  echo "  ✅ /sys/kernel/debug 可访问（6.3.x 容器内获取DCU进程信息需挂载此路径）"
else
  echo "  ⚠️  /sys/kernel/debug 不可访问"
fi

echo "==================================================================="
