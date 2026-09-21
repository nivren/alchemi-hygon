#!/bin/bash
# check_iommu_acs.sh — DCU 测试前置：检查 BIOS/内核是否已关闭 IOMMU 与 ACS
#
# 为什么做：rccl-test 通信测试对 PCIe 拓扑敏感，IOMMU / ACS 开启会把 DCU 卡
# 隔离进独立 iommu group、限制 P2P 直连，导致 allreduce/alltoall 带宽不达标。
# 文档《rccl-test 通信测试》明确要求"BIOS 下已关闭 acs 和 iommu 选项"。
#
# 判定：
#   IOMMU 关闭 = 内核未启用 iommu（/proc/cmdline 无 iommu=on / intel_iommu=on /
#              amd_iommu=on，且 /sys/kernel/iommu_groups 为空或仅少量）
#   ACS   关闭 = 未做 pcie_acs_override 且各 DCU 设备 ACS 控制位为 Disabled（内核默认不启用）
#
# 输出：JSON（stdout）+ 人类可读提示（stderr）。两者都未关闭时 exit=1 并提示用户。

set -u

CMD_LINE=$(cat /proc/cmdline 2>/dev/null || echo "")

# ---------- IOMMU 检测 ----------
iommu_status="unknown"
iommu_reason=""
if echo "$CMD_LINE" | grep -qE 'iommu=on|intel_iommu=on|amd_iommu=on'; then
  iommu_status="enabled"
  iommu_reason="内核启动参数启用了 iommu"
elif echo "$CMD_LINE" | grep -qE 'iommu=off|intel_iommu=off|amd_iommu=off'; then
  iommu_status="disabled"
  iommu_reason="内核启动参数明确 iommu=off"
elif [ -d /sys/kernel/iommu_groups ] && [ "$(ls -A /sys/kernel/iommu_groups 2>/dev/null)" = "" ]; then
  iommu_status="disabled"
  iommu_reason="/sys/kernel/iommu_groups 为空（未启用）"
else
  # 兜底：有 iommu_groups 且非空，视为启用（除非显式 off）
  if [ -d /sys/kernel/iommu_groups ] && [ -n "$(ls -A /sys/kernel/iommu_groups 2>/dev/null)" ]; then
    iommu_status="enabled"
    iommu_reason="/sys/kernel/iommu_groups 非空（已启用）"
  else
    iommu_status="disabled"
    iommu_reason="无 iommu 证据"
  fi
fi

# ---------- ACS 检测 ----------
# 权威判定：用 `lspci -vvv` 查 DCU 设备（1d94:）的 ACS 能力行（"ACS: ... Enabled/Disabled"）。
# 不硬编码寄存器偏移（ACS 是 PCIe 扩展能力，偏移因设备而异，setpci 固定地址不可靠）。
# 若无法 lspci，则退化为检查内核 pcie_acs_override（绕过即视为关闭）。
acs_status="unknown"
acs_reason=""
dcu_devs=$(lspci -nn 2>/dev/null | grep -iE '1d94:' | awk '{print $1}' | head -8)
if [ -z "$dcu_devs" ]; then
  acs_status="no_dcu_found"
  acs_reason="未发现 1d94: DCU 设备（可能无法 lspci）"
else
  # 取 DCU 设备的 lspci -vvv，找 ACS 行
  acs_lines=$(lspci -vvv -s "$(echo "$dcu_devs" | head -1)" 2>/dev/null | grep -iE '^\s*ACS:' || true)
  if [ -z "$acs_lines" ]; then
    # 该设备无 ACS 能力行（内核未编译 ACS 支持 / 硬件不支持）→ 视为关闭
    acs_status="disabled"
    acs_reason="DCU 设备无 ACS 能力行（lspci -vvv 无 'ACS:'，视为未启用）"
  elif echo "$acs_lines" | grep -qi 'Enabled'; then
    acs_status="enabled"
    acs_reason="DCU 设备 ACS 能力 Enabled（lspci -vvv）"
  else
    acs_status="disabled"
    acs_reason="DCU 设备 ACS 能力 Disabled（lspci -vvv）"
  fi
  # 内核显式绕过也视为关闭（覆盖上面判定）
  if echo "$CMD_LINE" | grep -qE 'pcie_acs_override='; then
    acs_status="disabled"
    acs_reason="内核含 pcie_acs_override（ACS 被绕过，视为关闭）"
  fi
fi

# ---------- 汇总判定 ----------
all_disabled="true"
warn_msg=""
if [ "$iommu_status" != "disabled" ]; then
  all_disabled="false"
  warn_msg="${warn_msg}IOMMU 未关闭(${iommu_status}:${iommu_reason}); "
fi
# ACS override 视为关闭；enabled 才算未关
if [ "$acs_status" = "enabled" ]; then
  all_disabled="false"
  warn_msg="${warn_msg}ACS 未关闭(${acs_reason}); "
fi

# ---------- 输出 ----------
cat <<JSON
{
  "iommu": {"status": "${iommu_status}", "reason": "${iommu_reason}"},
  "acs":   {"status": "${acs_status}",   "reason": "${acs_reason}"},
  "all_disabled": ${all_disabled}
}
JSON

if [ "$all_disabled" = "false" ]; then
  echo "" >&2
  echo "⚠️  [环境检查] IOMMU / ACS 未全部关闭，建议在 BIOS 关闭后再跑 rccl-test，否则 PCIe P2P 受限、通信带宽不达标。" >&2
  echo "    推荐：BIOS 中关闭 ACS 和 IOMMU 选项；或内核启动参数加 iommu=off amd_iommu=off（海光平台 amd_iommu）。" >&2
  echo "    详情：${warn_msg}" >&2
  exit 1
else
  echo "" >&2
  echo "✅ [环境检查] IOMMU 与 ACS 均已关闭，可以跑通信测试。" >&2
  exit 0
fi
