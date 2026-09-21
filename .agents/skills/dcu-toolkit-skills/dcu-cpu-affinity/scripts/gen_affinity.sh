#!/bin/bash
# gen_affinity.sh — 海光DCU NUMA 亲和性绑核方案生成器
#
# 输入：dcu-topology 采集的拓扑 JSON（--topo-json），或 --topo-json 缺省时现场调 hy-smi
# 输出：每张卡 -> numactl --cpunodebind=N --membind=N 绑定方案（dry-run 默认只打印）
#
# 安全：默认 dry-run（不执行 numactl）。--apply 仅生成命令/映射文件，不 kill/重启进程。
#
# 用法：
#   bash gen_affinity.sh --topo-json /tmp/topology.json
#   bash gen_affinity.sh --topo-json /tmp/topology.json --cmd "python train.py --device 0"
#   bash gen_affinity.sh --topo-json /tmp/topology.json --json
#   bash gen_affinity.sh --topo-json /tmp/topology.json --apply
#
# 设计依据（实测确认）：见 dcu-cpu-affinity SKILL.md Pitfalls

set -uo pipefail

TOPO_JSON=""
CMD=""
OUT_JSON=0
APPLY=0
FORCE=0

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --topo-json) TOPO_JSON="$2"; shift 2;;
    --cmd)       CMD="$2"; shift 2;;
    --json)      OUT_JSON=1; shift;;
    --apply)     APPLY=1; shift;;
    --force)     FORCE=1; shift;;
    -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "未知参数: $1" >&2; exit 1;;
  esac
done

# ---------- 取拓扑 JSON ----------
if [[ -n "$TOPO_JSON" && -r "$TOPO_JSON" ]]; then
  RAW=$(cat "$TOPO_JSON")
elif command -v hy-smi >/dev/null 2>&1 || [[ -x /opt/hyhal/bin/hy-smi ]]; then
  # 现场采：优先 dcu-hy-smi 封装，退化直接 hy-smi
  HY=/opt/hyhal/bin/hy-smi; command -v hy-smi >/dev/null 2>&1 && HY=hy-smi
  NUMA_JSON=$($HY --showtoponuma --json 2>/dev/null)
  # 仅取 card->Numa，numa_cpus 从 /sys 补
  SYS_CPUS=""
  for nd in /sys/devices/system/node/node[0-9]*; do
    n=$(basename "$nd" | tr -dc '0-9')
    SYS_CPUS="$SYS_CPUS\"$n\":\"$(cat "$nd/cpulist" 2>/dev/null)\","
  done
  SYS_CPUS="{${SYS_CPUS%,}}"
  RAW=$(echo "$NUMA_JSON" | python3 -c "
import sys,json
d=json.load(sys.stdin)
devs=[{\"pci_slot\":\"?\",\"numa_node\":v.get('(Topology) Numa Node','?')} for k,v in sorted(d.items())]
print(json.dumps({'dcu_devices':devs,'numa_cpus':$SYS_CPUS,'dcu_source':'hy-smi'},ensure_ascii=False))
" 2>/dev/null)
else
  echo "ERROR: 未提供 --topo-json 且本机无 hy-smi，无法采集拓扑" >&2
  exit 1
fi

[[ -z "$RAW" ]] && { echo "ERROR: 拓扑数据为空" >&2; exit 1; }

# ---------- 校验 dcu_source（除非 --force） ----------
SRC=$(echo "$RAW" | python3 -c "import sys,json;print(json.load(sys.stdin).get('dcu_source','?'))" 2>/dev/null)
if [[ "$SRC" != "hy-smi" && "$FORCE" -eq 0 ]]; then
  echo "⚠️  拓扑来源不可信 (dcu_source=$SRC，期望 hy-smi)。lspci 失真会导致绑定方案错误。" >&2
  echo "    若确认要用，请加 --force。" >&2
  exit 2
fi

# ---------- 生成映射 ----------
MAP_JSON=$(echo "$RAW" | python3 -c "
import sys,json
d=json.load(sys.stdin)
devs=d.get('dcu_devices',[])
cpus=d.get('numa_cpus',{})
out=[]
for i,dev in enumerate(devs):
    nn=dev.get('numa_node','?')
    slot=dev.get('pci_slot','?')
    cpulist=cpus.get(str(nn),'?') if nn!='?' else '?'
    out.append({'gpu_id':i,'pci_slot':slot,'numa_node':nn,'numa_cpus':cpulist})
print(json.dumps(out,ensure_ascii=False,indent=2))
" 2>/dev/null)

[[ -z "$MAP_JSON" ]] && { echo "ERROR: 映射生成失败" >&2; exit 1; }

# ---------- 输出 ----------
if [[ "$OUT_JSON" -eq 1 ]]; then
  echo "$MAP_JSON"
  exit 0
fi

echo "==================================================================="
echo " DCU NUMA 亲和性绑定方案（节点级 numactl）"
echo " 拓扑来源: $SRC"
echo "==================================================================="
echo "$MAP_JSON" | python3 -c "
import sys,json
m=json.load(sys.stdin)
nodes=set(str(x['numa_node']) for x in m if x['numa_node']!='?')
print(' 卡总数: %d   涉及NUMA节点: %s' % (len(m), ','.join(sorted(nodes)) if nodes else '?(未知)'))
print('-------------------------------------------------------------------')
for x in m:
    print(' GPU%-2d (PCI %s) -> NUMA %-2s  cpus: %s' % (x['gpu_id'],x['pci_slot'],x['numa_node'],x['numa_cpus']))
print('-------------------------------------------------------------------')
if len(nodes)>1:
    print(' ⚠️  跨NUMA：必须按卡↔NUMA分组绑定，切勿混绑到其他节点！')
else:
    print(' 单NUMA：绑定主要防止OS调度漂走，卡间通信收益有限（保险式绑定）')
"

# ---------- 生成可执行命令 ----------
echo ""
echo "-------------------- 可复制执行命令（dry-run，未执行） --------------------"
echo "$MAP_JSON" | python3 -c "
import sys,json
m=json.load(sys.stdin)
base_cmd='$CMD' if True else ''
for x in m:
    nn=x['numa_node']
    if nn=='?':
        print('# GPU%d: NUMA未知，跳过' % x['gpu_id']); continue
    if base_cmd.strip():
        # 替换 device 占位（若原命令含 --device N 或 =N）
        per_cmd=__import__('re').sub(r'(--device\s+)(\d+)', r'\g<1>%d'%x['gpu_id'], base_cmd)
        per_cmd=__import__('re').sub(r'(CUDA_VISIBLE_DEVICES=)(\d+)', r'\g<1>%d'%x['gpu_id'], per_cmd)
        print('numactl --cpunodebind=%s --membind=%s %s' % (nn,nn,per_cmd))
    else:
        print('# GPU%d: numactl --cpunodebind=%s --membind=%s <启动命令>' % (x['gpu_id'],nn,nn))
"
echo "-------------------------------------------------------------------"

# ---------- --apply：写映射文件（不执行进程操作） ----------
if [[ "$APPLY" -eq 1 ]]; then
  MAPFILE_OUT=/tmp/dcu_affinity_map.json
  echo "$MAP_JSON" > "$MAPFILE_OUT"
  echo ""
  echo "✅ --apply: 映射已写入 $MAPFILE_OUT（仅文件，未重启/绑定任何进程）"
  echo "   请用户自行用上方命令启动进程（容器内需在容器内执行 numactl，需 --privileged/SYS_NICE）"
  # 检测 numactl 是否可用
  if ! command -v numactl >/dev/null 2>&1; then
    echo "⚠️  本机未安装 numactl，执行绑定命令前需先装：yum/apt install numactl"
  fi
else
  echo ""
  echo "（默认 dry-run，仅打印。加 --apply 写映射文件；真正绑定请用上方命令启动进程）"
fi
