---
name: dcu-hy-smi
description: 海光DCU系统管理接口(hy-smi)查询与监控。对应AMD ROCm的rocm-smi。用于查看DCU卡的温度/功耗/利用率/显存/时钟/拓扑(NUMA亲和性、卡间链接)/进程。DCU环境巡检、故障排查、集合通信绑核前拓扑采集的必备工具。
metadata:
  version: 1.0.0
---

# dcu-hy-smi — 海光 DCU 系统管理接口

> 触发词：海光DCU、DCU、HCU、hy-smi、DCU监控、DCU拓扑、卡间亲和性、rocm-smi对应

`hy-smi` 是海光 DCU（C-3000 系列 HCU）的 SMI 工具，**对应 AMD ROCm 生态的 `rocm-smi`**。
它是 **DCU 驱动自带的**（安装驱动后自动位于 `/opt/hyhal/bin/hy-smi`，宿主机与容器内均有），
不一定在 PATH 里，但一定在那个绝对路径。

## 工具定位

- **巡检/监控**：温度、功耗、利用率、显存、时钟、进程
- **拓扑采集（绑核/集合通信关键）**：卡↔NUMA 亲和性、卡间链接类型、可达性、跳数、权重
- **故障排查**：RAS 错误、XGMI 错误计数、异常信息
- 危险设置类操作（超频/电压/卸载驱动）见原 skill 安全警告，**默认不做**

## 执行入口（实测确认）

```bash
# 宿主机（驱动装好即有，不一定在 PATH）
/opt/hyhal/bin/hy-smi --showtoponuma --json

# 容器内（若宿主机无 hy-smi 在 PATH）
docker exec <dcu容器> hy-smi <options>

# 远程执行（配合 dcu-ssh 的 sshpass 封装）
sshpass -f ~/.config/dcu-toolkit/credentials.json ssh root@<device_ip> \
  "/opt/hyhal/bin/hy-smi -a --json"
```

**注意**：宿主机（如 Ubuntu 系统）PATH 无 hy-smi，但 `/opt/hyhal/bin/hy-smi` 存在，直接绝对路径调用即可，
不必非走 docker exec（原 rookie-J/hy-smi README 的 SSH 示例默认走 docker exec，但实际宿主机直跑更省事）。

## 快速常用命令

| 用途 | 命令 |
|------|------|
| 设备概览 | `hy-smi` 或 `hy-smi -a` |
| 设备ID/序列号 | `hy-smi -i --showserial --showuniqueid` |
| 温度/功耗 | `hy-smi -t -P` |
| 利用率(HCU/CU) | `hy-smi -u --showhcuutil --showcuutil` |
| 显存使用 | `hy-smi --showmemuse --showmemavailable` |
| 时钟频率 | `hy-smi -c -g --showclkfrq` |
| 运行进程 | `hy-smi --showpids` |
| JSON 输出 | `hy-smi -a --json` |

## 拓扑与连接（集合通信绑核关键 — 优先用这些）

| 选项 | 含义 |
|------|------|
| `--showtopo` | 硬件拓扑总览 |
| `--showtoponuma` | **卡↔NUMA 节点亲和性**（最常用，解决 lspci 失真问题） |
| `--showtopoaccess` | HCU 间链接可达性 |
| `--showtopoweight` | HCU 间相对权重 |
| `--showtopohops` | HCU 间跳数 |
| `--showtopotype` | 链接类型（XGMI / PCIe） |
| `--shownodeid` | 节点 ID |
| `--showxhclstatus` | xhcl 链接状态 |
| `--showxgmierr` | XGMI 错误计数 |

> ⚠️ **`--showtoponuma` 是获取卡↔NUMA 亲和性的唯一可信源**。`lspci -vvv` 在海光上所有 PCI
> 设备都报 `NUMA node: 0`，失真（实测跨 NUMA 机型的 card4-7 实际在 NUMA4，lspci 却报 0）。
> 详见 dcu-topology skill 的 Pitfall #6/#7。dcu-topology 的 `dcu-topology/scripts/collect_topology.sh` 已内置
> 用 hy-smi 采集，无需手动跑。

## 带宽相关

- `-b, --showbw` - PCIe 带宽
- `--showdfbw/--showmembw` - DF/显存读写带宽
- `--showumcbw` - UMC 读写带宽
- `--showxhclbw` - xhcl 带宽
- `--channel/--link/--direction/--delay/--count/--bwdetail` - 带宽细查参数

## 健康与异常

- `--healthcheck` - 健康状态检查
- `--showrasinfo BLOCK` - RAS 使能/错误计数
- `--showexceptioninfo` - 异常信息
- `--showxgmierr` - XGMI 错误计数

## 输出格式选项

- `--json` - JSON 格式（脚本解析首选）
- `--csv` - CSV 格式
- `--loglevel LEVEL` - debug/info/warning/error/critical

## 实测参考（多机型验证）

```json
// 跨 NUMA 机型（Kylin）: hy-smi --showtoponuma --json
{"card0":{"(Topology) Numa Node":"0","(Topology) Numa Affinity":"0"},
 "card1":{...:"0"}, "card2":{...:"0"}, "card3":{...:"0"},
 "card4":{"(Topology) Numa Node":"4","(Topology) Numa Affinity":"4"},
 "card5":{...:"4"}, "card6":{...:"4"}, "card7":{...:"4"}}
// → 跨 NUMA 机型：卡0-3在NUMA0，卡4-7在NUMA4（跨socket，每socket 4卡）

// 单 NUMA 机型（Ubuntu）: 8卡全在NUMA0
```

解析 tip（bash）：
```bash
hy-smi --showtoponuma --json | grep -oE 'card[0-9]+|"\(Topology\) Numa Node": "[0-9]+"' \
  | awk '/card[0-9]+/{c=$0} /Numa Node/{match($0,/"([0-9]+)"/,a); print c, a[1]}'
```

## 占用进程定位与按容器反查（巡检最实用 ⭐）

DCU 上任务基本跑在 docker 容器里。光看概览只能知道"哪张卡有占用"，要定位"**谁**在占卡"必须查进程并反查容器。

```bash
# 1. 查占用进程（PID / VRAM 占用）
/opt/hyhal/bin/hy-smi --showpids
# 输出每个占用 DCU 的 PID，带 HCU Index / GPUID / VRAM USED(MiB) / VRAM USED(%)。
# 注意 VRAM USED(%) 可能为 inf（相对卡总显存算出的异常值），以 VRAM USED(MiB) 为准。

# 2. 按容器反查占用者：从 hy-smi 拿 PID → 查 cgroup 里的 docker 容器 ID → docker inspect 容器名
/opt/hyhal/bin/hy-smi --showpids 2>/dev/null \
  | awk '/^PID:/ {print $2}' \
  | xargs -I {} sh -c "cat /proc/{}/cgroup 2>/dev/null | grep -oP '(docker/\K[^/]+)|(docker-\K[^.]+)' | head -1" \
  | while read cid; do
      echo "${cid:0:5} -> $(docker inspect --format='{{.Name}}' $cid 2>/dev/null | sed 's/^//')"
    done
```

**实测（2026-07-22）**：直接定位到占用卡 0 的是容器 `<container>`（PID 2083159，52703 MiB ≈ 80% 显存），比光看概览精确得多。

> 注意：`--showpids` 可能列出多个 PID（含仅占用 1 MiB 的附属进程），同一容器会重复出现，认 VRAM USED(MiB) 大的那个 PID 才是真正占显存的。

### 典型判断流程（"设备有没有人用"）

```bash
# 步骤1：概览看哪张卡有占用
/opt/hyhal/bin/hy-smi
# 步骤2：谁在线
who            # 看登录会话来源 IP / 时间
w              # 看挂机时长 + 当前命令
# 步骤3：占卡的具体容器（上面反查指令）
```

- `VRAM%` 高但 `HCU%`=0、`AvgPwr`≈60W：进程常驻占显存但当前没计算（如遗留的推理服务）
- 卡 1~7 `VRAM%`=0：完全空闲，可直接抢占

## 封装脚本（本 skill 提供）

`scripts/hy_smi_query.sh` — 自动探测 hy-smi 路径（绝对路径→PATH→docker容器兜底），
一行出结构化拓扑/监控，避免每次手敲 docker exec：
```bash
bash scripts/hy_smi_query.sh topo      # 卡↔NUMA + 卡间跳数/hops（JSON）
bash scripts/hy_smi_query.sh monitor   # 温度/功耗/利用率概览
bash scripts/hy_smi_query.sh <任意hy-smi参数>   # 透传，如 --showpids --json
```

## 环境要求

- 驱动版本 6.2.22+（支持 `--finegrain` 等选项）
- 需要 root 权限（部分操作）；容器内需 `--privileged` 或映射 `/dev/kfd`、`/dev/dri`
- 驱动加载检查：`lsmod | grep -E "hydcu|hycu"`

## Pitfalls（实测确认 — 多机型）

1. **hy-smi 不在 PATH 但驱动自带**：宿主机（如 Ubuntu 系统）PATH 无 `hy-smi`，但 `/opt/hyhal/bin/hy-smi` 存在。优先绝对路径，不必非走 docker exec。
2. **`--showtoponuma` 是唯一可信卡↔NUMA 源**：`lspci -vvv` 在海光上所有 PCI 设备都报 `NUMA node: 0`（失真）。跨 NUMA 机型实测 card4-7 实际在 NUMA4，lspci 却报 0。**绑核/集合通信亲和性必须以 hy-smi 为准**（dcu-topology 已内置）。
3. **仅 `--showtoponuma` 支持 `--json`**：`--showtopohops/--showtopoaccess/--showtopoweight/--showtopotype` 不支持 `--json`（hy-smi 报 `This command does not support to set print format`），输出为表格。解析这些需按表格提取（如 hops 矩阵）。
4. **hop 矩阵印证拓扑**：单 NUMA 机型组内 1 hop、跨组 2 hop（全同 NUMA 但物理分两组）；跨 NUMA 机型组内 2 hop、跨 NUMA 组 3 hop（真跨 socket）。hop 数差异直接反映 NUMA 跨 socket 代价，是 RCCL 绑核策略依据（见 dcu-rccl-test）。
5. **容器内执行需 `--privileged` 或映射 `/dev/kfd` `/dev/dri`**：否则 `Permission denied`。
6. **`--showpids` 的 VRAM USED(%) 可能为 `inf`**——以 `VRAM USED(MiB)` 判断真实占用量。
7. **PID 反查容器依赖 docker**——若任务跑在裸机（非容器），`docker inspect` 会找不到，退回到 `ps -p <pid> -o cmd` 看命令行。
8. **多 PID 同容器重复出现正常**——认显存占用大的那个 PID。
9. **⚠️ PID→容器反查指令只能在物理机（宿主机）执行**：容器内看不到宿主的 docker 层级，`/proc/<pid>/cgroup` 里没有宿主 docker ID，且容器内通常没有 `docker` 命令，这条会失效。正确做法：SSH 到**宿主机**跑 `hy-smi --showpids` + 反查；若当前在容器里，先 `exit` 回宿主。（实测：宿主机直接跑成功定位到 `<container>`。）

> 参考源：https://github.com/rookie-J/hy-smi （本 skill 命令参考基于此仓库，已补充多机型实测与本包集成封装）

## References

- 原 skill 仓库：https://github.com/rookie-J/hy-smi （含完整选项参考 + examples.sh，本 skill 的 `scripts/examples.sh` 即来源于此）
- 关联：dcu-topology（拓扑采集已内置 hy-smi，`dcu-topology/scripts/collect_topology.sh` 用 --showtoponuma 为权威源）、dcu-rccl-test（集合通信绑核，hops/亲和性为 NCCL 映射依据）
