#!/usr/bin/env bash
# =============================================================================
# collect_topology.sh — DCU 服务器 NUMA/CPU/DCU 拓扑采集
# =============================================================================
# 设计思路参考 autobinder（https://gitee.com/shilei-wang/autobinder）的
# inspect 拓扑采集逻辑，但改为纯标准 Linux 命令实现，不依赖 autobinder 二进制。
#
# autobinder inspect 本质读取：
#   /sys/devices/system/cpu/cpuN/topology/{core_id,physical_package_id,thread_siblings_list}
#   /sys/devices/system/node/nodeN/{meminfo,distance,memory_tier}
#   /proc/cpuinfo  (厂商判定，仅支持 Hygon)
#   /sys/devices/system/cpu/cpuN/cache/index3/id  (L3/CCX 归属)
# 这些全部是标准接口，用 lscpu / numactl / lspci 等价获取即可。
#
# 本脚本额外补上 autobinder 没有的：DCU 卡 ↔ NUMA 节点亲和性（DCU 场景关键）。
#
# 用法：
#   bash collect_topology.sh            # 输出人类可读
#   bash collect_topology.sh -j          # 输出 JSON（推荐，便于后续处理）
#
# 需 root 或能读 /sys /proc；hy-smi 可选（有则采集 DCU 信息）。
# =============================================================================

set -euo pipefail

OUT_JSON=0
[[ "${1:-}" == "-j" ]] && OUT_JSON=1

# ---------- 1. 基础信息 ----------
HOST=$(hostname)
KERNEL=$(uname -r)
CPU_VENDOR=$(grep -m1 vendor_id /proc/cpuinfo | awk -F: '{print $2}' | tr -d ' ')
# Hygon C86 的 vendor_id 通常为 "HygonGenuine"；autobinder 仅支持 Hygon
# 注意：model name 字段内含冒号(如 "Hygon C86-4G (OPN:7490)")，不能用 -F: 切，用 sub 去掉首个冒号前缀
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo | sed 's/^[^:]*:[[:space:]]*//')
# total_cpu 优先用 lscpu 的 CPU(s)（nproc 在部分老内核(Kylin)下会返回 255 等异常值）
TOTAL_CPU=$(LANG=C lscpu 2>/dev/null | awk -F: '/^CPU\(s\)/{gsub(/^[ \t]+/,"",$2);print $2}')
TOTAL_CPU=${TOTAL_CPU:-$(nproc 2>/dev/null || echo 0)}
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---------- 2. NUMA 节点与 CPU 分布 ----------
# 优先从 lscpu 的 NUMA node(s) 读节点数（numactl 可能未安装，如 Kylin V10）
NUM_NODES=$(LANG=C lscpu 2>/dev/null | awk -F: '/^NUMA node\(s\)/{gsub(/^[ \t]+/,"",$2);print $2}')
NUM_NODES=${NUM_NODES:-1}
declare -A NUMA_CPUS=()
# 从 /sys 读各节点 CPU 列表（不依赖 numactl）
for nd in /sys/devices/system/node/node[0-9]*; do
  n=$(basename "$nd" | tr -dc '0-9')
  if [[ -r "$nd/cpulist" ]]; then
    NUMA_CPUS[$n]=$(cat "$nd/cpulist" 2>/dev/null)
  fi
done
# 兜底：若 /sys 无数据且 numactl 可用，用 numactl -H 解析
if [[ ${#NUMA_CPUS[@]} -eq 0 ]] && command -v numactl >/dev/null 2>&1; then
  while read -r line; do
    if [[ "$line" =~ node\ ([0-9]+)\ cpus:\ (.*) ]]; then
      NUMA_CPUS[${BASH_REMATCH[1]}]="${BASH_REMATCH[2]}"
    fi
  done < <(numactl -H 2>/dev/null)
fi
[[ ${#NUMA_CPUS[@]} -eq 0 ]] && NUM_NODES=1

# NUMA 距离矩阵
declare -A NUMA_DIST=()
if [[ -r /sys/devices/system/node/ ]]; then
  for nd in /sys/devices/system/node/node[0-9]*; do
    n=$(basename "$nd" | tr -dc '0-9')
    if [[ -r "$nd/distance" ]]; then
      NUMA_DIST[$n]=$(cat "$nd/distance" 2>/dev/null)
    fi
  done
fi

# ---------- 3. CPU Socket / 核 / 超线程 拓扑 ----------
# 用 lscpu 解析（等价 autobinder 读 /sys/devices/system/cpu/...）
# 注意：强制 LANG=C 避免中文 locale 下字段名变化（架构/套接字 vs Architecture/Socket）
LSCPU_OUT=$(LANG=C LC_ALL=C lscpu 2>/dev/null || true)
SOCKETS=$(echo "$LSCPU_OUT" | awk -F: '/^Socket\(s\)/{gsub(/^[ \t]+/,"",$2);print $2}')
CORES_PER_SOCKET=$(echo "$LSCPU_OUT" | awk -F: '/^Core\(s\) per socket/{gsub(/^[ \t]+/,"",$2);print $2}')
THREADS_PER_CORE=$(echo "$LSCPU_OUT" | awk -F: '/^Thread\(s\) per core/{gsub(/^[ \t]+/,"",$2);print $2}')
L1=$(echo "$LSCPU_OUT" | awk -F: '/^L1d cache/{gsub(/^[ \t]+/,"",$2);print $2}')
L2=$(echo "$LSCPU_OUT" | awk -F: '/^L2 cache/{gsub(/^[ \t]+/,"",$2);print $2}')
L3=$(echo "$LSCPU_OUT" | awk -F: '/^L3 cache/{gsub(/^[ \t]+/,"",$2);print $2}')
SOCKETS=${SOCKETS:-1}
CORES_PER_SOCKET=${CORES_PER_SOCKET:-0}
THREADS_PER_CORE=${THREADS_PER_CORE:-1}

# ---------- 4. DCU 卡 ↔ NUMA 亲和性（DCU 场景关键，autobinder 无） ----------
# 数据来源优先级（实测确认）：
#   1) hy-smi --showtoponuma --json  ← 海光官方 SMI 工具，给真实卡↔NUMA 亲和性
#      （解决 lspci -vvv 在海光上所有设备都报 NUMA node 0 的失真问题，见 Pitfall #6）
#   2) lspci -vvv 的 "NUMA node: N"  ← 仅作 hy-smi 不可用时的兜底，标记不可靠
# hy-smi 是 DCU 驱动自带的（安装驱动后位于 /opt/hyhal/bin/hy-smi，宿主机与容器内
#   均有），优先用绝对路径；PATH 内有则直接用；最后才兜底 docker exec 探测容器。
# 脚本自动探测执行入口。
DCU_DEVS=()
DCU_SOURCE="none"

# --- 4a. 探测 hy-smi 执行入口 ---
HY_SMI_CMD=""
if [[ -x /opt/hyhal/bin/hy-smi ]]; then
  HY_SMI_CMD="/opt/hyhal/bin/hy-smi"
elif command -v hy-smi >/dev/null 2>&1; then
  HY_SMI_CMD="hy-smi"
elif command -v docker >/dev/null 2>&1; then
  # 找第一个能跑 hy-smi 的 DCU 容器
  for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
    if docker exec "$c" hy-smi --showtoponuma --json >/dev/null 2>&1; then
      HY_SMI_CMD="docker exec $c hy-smi"
      break
    fi
  done
fi

# --- 4b. 用 hy-smi 拿卡↔NUMA（可信源） ---
if [[ -n "$HY_SMI_CMD" ]]; then
  set +e
  HY_JSON=$($HY_SMI_CMD --showtoponuma --json 2>/dev/null)
  if [[ -n "$HY_JSON" ]]; then
    # 解析 cardN -> Numa Node，PCI slot 用 lspci 反查（hy-smi 不报 PCI 地址）
    # 先用 lspci 拿到 card 顺序对应的 co-processor slot 列表（仅要 slot，不要 NUMA）
    SLOT_LIST=()
    if command -v lspci >/dev/null 2>&1; then
      SLOT_LIST=($(lspci -nn 2>/dev/null | grep -iE "co-processor" | awk '{print $1}' | sort -u))
    fi
    # JSON 形如 {"card0": {"(Topology) Numa Node": "0", ...}, ...}
    idx=0
    while read -r card nn; do
      [[ -z "$card" || -z "$nn" ]] && continue
      slot="${SLOT_LIST[$idx]:-?}"
      DCU_DEVS+=("${slot}|${nn}")
      idx=$((idx+1))
    done < <(echo "$HY_JSON" | grep -oE 'card[0-9]+|"\(Topology\) Numa Node": "[0-9]+"' | \
             awk '/card[0-9]+/{c=$0} /Numa Node/{match($0,/"([0-9]+)"/,a); print c, a[1]}')
    [[ ${#DCU_DEVS[@]} -gt 0 ]] && DCU_SOURCE="hy-smi"
  fi
  set -e
fi

# --- 4c. 兜底：hy-smi 不可用时用 lspci -vvv（不可靠，标记） ---
if [[ ${#DCU_DEVS[@]} -eq 0 && -n "$(command -v lspci)" ]]; then
  set +e
  RAW_SLOTS=$(lspci -nn 2>/dev/null | grep -iE "co-processor" | awk '{print $1}' | sort -u)
  LSPCI_VVV=$(lspci -vvv 2>/dev/null)
  while read -r slot; do
    [[ -z "$slot" ]] && continue
    nnode=$(echo "$LSPCI_VVV" | awk -v s="$slot" '
      $0 ~ "^"s" " {found=1; next}
      found && /NUMA node:/ {match($0,/NUMA node: *([0-9]+)/,a); if(a[1]!=""){print a[1]; exit}}
      found && /^$/ {exit}
    ')
    [[ -z "$nnode" ]] && nnode="?"
    DCU_DEVS+=("${slot}|${nnode}")
  done < <(echo "$RAW_SLOTS")
  set -e
  [[ ${#DCU_DEVS[@]} -gt 0 ]] && DCU_SOURCE="lspci-unreliable"
fi
DCU_COUNT=${#DCU_DEVS[@]}

# 跨 NUMA 检测：若 DCU 卡分布在多个 NUMA 节点，标记（影响集合通信亲和性策略）
DCU_NUMA_SET=$(printf "%s\n" "${DCU_DEVS[@]//*|/}" | sort -u | grep -v '?' | tr '\n' ',' )
DCU_NUMA_SET=${DCU_NUMA_SET%,}

# ---------- 5. 输出 ----------
if [[ "$OUT_JSON" -eq 1 ]]; then
  {
    echo "{"
    echo "  \"host\": \"$HOST\","
    echo "  \"timestamp\": \"$TIMESTAMP\","
    echo "  \"kernel\": \"$KERNEL\","
    echo "  \"cpu_vendor\": \"$CPU_VENDOR\","
    echo "  \"cpu_model\": \"$CPU_MODEL\","
    echo "  \"total_cpu\": $TOTAL_CPU,"
    echo "  \"sockets\": ${SOCKETS:-1},"
    echo "  \"cores_per_socket\": ${CORES_PER_SOCKET:-0},"
    echo "  \"threads_per_core\": ${THREADS_PER_CORE:-1},"
    echo "  \"cache\": {\"L1d\": \"${L1:-?}\", \"L2\": \"${L2:-?}\", \"L3\": \"${L3:-?}\"},"
    echo "  \"numa_nodes\": $NUM_NODES,"
    printf "  \"numa_cpus\": {"
    first=1
    for n in $(echo "${!NUMA_CPUS[@]}" | tr ' ' '\n' | sort -n); do
      [[ $first -eq 0 ]] && printf ", "
      printf "\"%s\": \"%s\"" "$n" "${NUMA_CPUS[$n]}"
      first=0
    done
    printf "},\n"
    printf "  \"numa_distance\": {"
    first=1
    for n in $(echo "${!NUMA_DIST[@]}" | tr ' ' '\n' | sort -n); do
      [[ $first -eq 0 ]] && printf ", "
      printf "\"%s\": \"%s\"" "$n" "${NUMA_DIST[$n]}"
      first=0
    done
    printf "},\n"
    printf "  \"dcu_devices\": ["
    first=1
    for d in "${DCU_DEVS[@]}"; do
      slot="${d%%|*}"; nn="${d##*|}"
      [[ $first -eq 0 ]] && printf ", "
      printf "\n    {\"pci_slot\": \"%s\", \"numa_node\": \"%s\"}" "$slot" "$nn"
      first=0
    done
    printf "\n  ],\n"
    echo "  \"dcu_count\": $DCU_COUNT,"
    echo "  \"dcu_numa_span\": \"$DCU_NUMA_SET\","
    echo "  \"dcu_source\": \"$DCU_SOURCE\","
    echo "  \"cross_numa\": $( [[ "$DCU_NUMA_SET" == *","* || "$DCU_NUMA_SET" == "?" ]] && echo true || echo false )"
    echo "}"
  }
else
  echo "==================================================================="
  echo " Host         : $HOST"
  echo " Kernel       : $KERNEL"
  echo " CPU Vendor   : $CPU_VENDOR  (autobinder 仅支持 HygonGenuine)"
  echo " CPU Model    : $CPU_MODEL"
  echo " Total CPU    : $TOTAL_CPU  (Sockets=$SOCKETS Cores/Sock=$CORES_PER_SOCKET Thr/Core=$THREADS_PER_CORE)"
  echo " Cache        : L1d=$L1 L2=$L2 L3=$L3"
  echo " NUMA nodes   : $NUM_NODES"
  for n in $(echo "${!NUMA_CPUS[@]}" | tr ' ' '\n' | sort -n); do
    echo "   node $n cpus: ${NUMA_CPUS[$n]}"
    [[ -n "${NUMA_DIST[$n]:-}" ]] && echo "   node $n distance: ${NUMA_DIST[$n]}"
  done
  echo "-------------------------------------------------------------------"
  echo " DCU devices  : $DCU_COUNT"
  for d in "${DCU_DEVS[@]}"; do
    slot="${d%%|*}"; nn="${d##*|}"
    echo "   $slot  -> NUMA node $nn"
  done
  echo " DCU NUMA span: $DCU_NUMA_SET"
  echo " DCU data src : $DCU_SOURCE"
  if [[ "$DCU_NUMA_SET" == *","* || "$DCU_NUMA_SET" == "?" ]]; then
    echo "   ⚠️ cross_numa=true : DCU 卡跨多个 NUMA 节点，集合通信需按卡↔NUMA 亲和性绑核"
  else
    echo "   cross_numa=false: DCU 卡均在单一 NUMA 节点（理想）"
  fi
  echo "==================================================================="
fi
