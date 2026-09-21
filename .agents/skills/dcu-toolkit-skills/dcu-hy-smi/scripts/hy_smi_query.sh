#!/bin/bash
# hy_smi_query.sh — 海光 DCU hy-smi 便捷封装
# 自动探测 hy-smi 路径（绝对路径 / PATH / docker 容器兜底），免去手敲 docker exec。
#
# 用法:
#   bash hy_smi_query.sh topo          # 卡↔NUMA 亲和性 + 卡间跳数(hops)，JSON 输出
#   bash hy_smi_query.sh monitor       # 温度/功耗/利用率概览（默认表格）
#   bash hy_smi_query.sh <hy-smi参数>  # 透传，如 --showpids --json / -a --json
#
# 设计依据（实测确认，见 dcu-topology Pitfall #7）:
#   hy-smi 是 DCU 驱动自带，位于 /opt/hyhal/bin/hy-smi（宿主机与容器内均有），
#   不一定在 PATH。探测顺序: 绝对路径 → PATH → docker exec 第一个能跑的容器。

set -uo pipefail

# ---------- 探测 hy-smi 执行入口 ----------
detect_hy_smi() {
  if [[ -x /opt/hyhal/bin/hy-smi ]]; then
    echo "/opt/hyhal/bin/hy-smi"
    return 0
  elif command -v hy-smi >/dev/null 2>&1; then
    command -v hy-smi
    return 0
  elif command -v docker >/dev/null 2>&1; then
    for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
      if docker exec "$c" hy-smi --showtoponuma --json >/dev/null 2>&1; then
        echo "docker exec $c hy-smi"
        return 0
      fi
    done
  fi
  return 1
}

HY=$(detect_hy_smi) || { echo "ERROR: hy-smi 未找到（宿主机/opt/hyhal/bin 无、PATH 无、容器内也探测不到）" >&2; exit 1; }

# ---------- 子命令分发 ----------
case "${1:-monitor}" in
  topo)
    # 卡↔NUMA 亲和性 + 卡间跳数，统一 JSON 输出便于解析
    echo "=== 卡↔NUMA 亲和性 (--showtoponuma --json) ==="
    $HY --showtoponuma --json 2>/dev/null
    echo "=== 卡间跳数 (--showtopohops --json) ==="
    $HY --showtopohops --json 2>/dev/null
    ;;
  monitor)
    # 温度/功耗/利用率概览（默认表格）
    $HY -t -P -u --showhcuutil --showcuutil 2>/dev/null
    ;;
  *)
    # 透传所有参数给 hy-smi
    $HY "$@"
    ;;
esac
