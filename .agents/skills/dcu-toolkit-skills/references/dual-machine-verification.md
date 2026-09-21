# 多机型验证 Harness（dcu-toolkit 通用）

> 用途：任何脚本/命令在写进 skill 前，必须在真实 DCU 硬件上验证。本文件是
> 可复用的"推脚本→跑→解析→对比"模板，避免每次手敲。

## 设备与凭证

- 设备清单（非敏感）：`data/devices.json`（IP / 端口 / 用户 / 硬件配置）。
- 真实凭证（密码）：**绝不进 skill 包**，只存 `~/.config/dcu-toolkit/credentials.json`（chmod 600）。
- 脚本示例里密码一律从环境变量 `SSHPASS` 读（配合 `sshpass -e`），不内联明文。

## 标准验证循环（复制即用）

```bash
# 1. 把本地脚本推到各验证机（密码从外部凭证读，不内联）
#    凭证文件格式见 data/credentials-template.json；用 sshpass -f 或 -e 注入
export SSHPASS="$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json')))['<device>']['password'])")"
for host in <device_ip_1> <device_ip_2>; do
  sshpass -e scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    /root/.hermes/skills/dcu/<skill>/scripts/<script>.sh root@$host:/tmp/ 2>&1 \
    | grep -v -E "Permanently|Authorized"
  # 2. 跑 + 结构化解析（JSON 优先）
  sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$host \
    "bash /tmp/<script>.sh -j" 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('  dcu_source=%s dcu_count=%d numa_span=%s cross_numa=%s' %
      (d['dcu_source'], d['dcu_count'], d['dcu_numa_span'], d['cross_numa']))
"
done
```

要点：
- **多机型都要跑**：Ubuntu 与 Kylin 老内核行为差异巨大（见下），单台验证会漏坑。
- **JSON 输出优先**：脚本必须支持 `-j`，便于 python 解析断言，不靠肉眼读表格。
- **`2>/dev/null` 吞 sshpass 的 "Authorized users" 横幅**，但需保留脚本本身的 stderr 排错。

## 机型差异速查（实测踩出的）

| 现象 | Ubuntu 机型 | Kylin V10（4.19 老内核）机型 | 正确做法 |
|------|------------|------------------------------|----------|
| `nproc` | 正常 | 返回 255 异常 | 用 `lscpu\|CPU(s)` |
| `numactl` | 有 | 未安装 | 从 `/sys/devices/system/node/nodeN/cpulist` + `lscpu` 读 |
| `lspci -vvv` NUMA | 碰巧对(全0) | **失真**(全报0,实际4) | 用 `hy-smi --showtoponuma` 为准 |
| `hy-smi` PATH | 不在 PATH | 不在 PATH | 用绝对路径 `/opt/hyhal/bin/hy-smi` |
| `cpu_model` 冒号 | 正常 | 含 `OPN:7490` 截断风险 | `sed 's/^[^:]*:[[:space:]]*//'` |

## 脚本开发纪律（与 Working Discipline 联动）

1. 写完/改完脚本，**同一轮**推多机型验证，不靠"应该没问题"。
2. 验证发现 bug → 修脚本 → 重推 → 通过 → **立刻把 pitfall 写回对应 skill**。
3. skill 里 `[规划中]`/`占位` 一旦实现，当轮翻成 ✅，路由表状态同步。
4. `set -e` + 远程命令：远程端用 `set +e` 包裹可能失败的探测段（lspci/numactl 缺失时不应终止）。
5. **patch 工具对 bash 里的双重转义 `\\\"` 会判定"新旧相同"**——修 SKILL.sh 里 JSON 行的引号转义错乱时，用 `python3 - <<'PYEOF'` 直接重写该行，或用 `sed -i '行号s|...|...|'`（注意 `&`/`/` 需转义）。execute_code 被 cron 安全策略拦截，别用。

## 拓扑采集权威结论（dcu-topology / dcu-hy-smi 共用）

- **卡↔NUMA 亲和性：以 `hy-smi --showtoponuma --json` 为唯一可信源**。
  `lspci -vvv` 在海光上所有 PCI 设备都报 `NUMA node: 0`，失真。
- **hop 矩阵印证拓扑**：跨 NUMA 机型组内 2 hop / 跨 NUMA 组 3 hop（真跨 socket）；
  单 NUMA 机型组内 1 hop / 跨组 2 hop（全同 NUMA 但物理分两组）。
- 仅 `--showtoponuma` 支持 `--json`；`--showtopohops/access/weight/type` 只出表格。
- 跨 NUMA 机型实测：card0-3→NUMA0，card4-7→NUMA4；单 NUMA 机型实测：8卡全 NUMA0。
