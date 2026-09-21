---
name: dcu-topology
description: "Use when optimizing DCU multi-card/multi-node topology — NUMA affinity, PCIe/XGMI topology inspection, CPU core binding for multi-card communication performance. Topology data collection is IMPLEMENTED and verified on multiple machine types (scripts/collect_topology.sh); auto-binding (autobinder) remains a candidate, disabled by default. Do not enable/use the auto-binding approach until explicitly activated."
license: MIT
metadata:
  version: 1.0.0
  author: DCU Engineer
  hermes:
    tags: [dcu, topology, numa, cpu-binding, affinity, multi-card]
    related_skills: [dcu-env, dcu-rccl-test, dcu-ssh]
---

# DCU Topology 拓扑与绑核

## Overview

本 skill 目标：DCU 多卡 / 多机场景下的拓扑分析与绑核优化——包括 NUMA 亲和性、PCIe/XGMI 拓扑、CPU 核绑定，以最大化多卡通信（RCCL）与计算性能。

**当前状态：拓扑数据采集已实现并多机型验证（含单 NUMA 与跨 NUMA 两种拓扑）；CPU 自动绑核（autobinder 方案）仍为候选、默认不启用。**

## 候选参考（默认不启用）

- **autobinder**（https://gitee.com/shilei-wang/autobinder）
  - 定位：**海光(Hygon) C86 CPU 的 NUMA 感知进程调度 / CPU 亲和性优化工具**（单二进制 Go CLI，开箱即用）
  - 运行模式：`daemon`（持续自动绑核）、`inspect`（查看系统拓扑和绑核状态）、`clear`（清除策略恢复默认）
  - 核心能力：**进程级 CPU 绑核**——单 CCX 独占优先 → 粘滞优先 → NUMA 距离感知 → SMT 感知；自适应内存绑定防 OOM；一键 `clear` 回滚
  - 系统要求：海光 C86 CPU + Linux 3.10+ + `CAP_SYS_NICE`（设亲和性）
  - 与 DCU 的关系：**autobinder 只优化 CPU 侧调度，不涉及 DCU 卡间通信（PCIe/XGMI）**。DCU 多卡场景真正影响集合通信的是"卡↔NUMA 亲和性"（进程绑到 DCU 卡所在 NUMA 的 CPU），autobinder 的 `inspect` 拓扑采集思路可参考，但**绑核策略必须适配 DCU 拓扑，不能直接套用**
  - **默认不启用**：autobinder 是通用进程绑核工具，DCU 通信优化需专门验证；在 dcu-topology 显式激活并确认适配前，不要直接对 DCU 训练/推理进程跑 autobinder daemon

> 源码已 clone 参考（/tmp/autobinder，含 benchmarks/、docs/、CLAUDE.md）。仅作设计参考，不纳入 skill 包、不内置凭证。

## 已实现的基础能力（来自其他 skill，可复用）

- **dcu-env**：已能采集 NUMA 节点、PCIe 设备拓扑（`lspci -nn | grep -i co-processor` + `lspci -tvvv`）、跨 NUMA 标记（cross_numa）。拓扑采集基础已具备。
- **dcu-rccl-test**：双机测试前需修正 `topo_mapping_default.xml` 的 HCA 数（见其速查表），属于拓扑相关调优的一部分。

## When to Use（拓扑采集已实现；绑核策略规划中）

- 多卡通信性能不佳，需排查 NUMA / 跨 NUMA / PCIe 拓扑问题
- 需要为 DCU 进程做 CPU 亲和性绑定以优化多卡通信
- 多机训练前拓扑确认（配合 dcu-rccl-test）

Don't use for：单机单卡；纯推理且性能达标无需调优。

## NUMA/CPU 拓扑采集（已实现 — 基础能力）

> 思路来源：autobinder 的 `inspect` 拓扑采集（读 `/sys/devices/system/cpu/...`、`/sys/devices/system/node/...`、`/proc/cpuinfo`）。
> 本 skill 不依赖 autobinder 二进制，改用标准命令（`lscpu` / `numactl` / `lspci`）等价实现，并**额外补上 autobinder 没有的「DCU 卡 ↔ NUMA 节点亲和性」**——这是 DCU 多卡通信的关键。

### 采集脚本
`scripts/collect_topology.sh`（纯标准命令，无需 autobinder）：
```bash
bash collect_topology.sh        # 人类可读
bash collect_topology.sh -j     # JSON 输出（推荐，便于后续解析）
```

采集内容：
- **CPU 拓扑**：厂商/型号、socket 数、每 socket 核数、超线程、各级缓存（LANG=C 强制，避免中文 locale 字段名变化）
- **NUMA 拓扑**：节点数、各节点 CPU 列表、NUMA 距离矩阵（来自 `lscpu` 的 `NUMA node(s)` + `/sys/devices/system/node/nodeN/cpulist` + `/sys/.../distance`；**不依赖 numactl**，Kylin V10 未装 numactl 实测确认）
- **DCU 卡 ↔ NUMA 亲和性**（DCU 关键）：优先用 `hy-smi --showtoponuma --json`（海光官方 SMI，对应 rocm-smi）拿真实卡↔NUMA 亲和性；`lspci -vvv` 在海光上所有设备都报 `NUMA node: 0`，失真不可用（见 Pitfall #6）。hy-smi 探测顺序：`/opt/hyhal/bin/hy-smi`（驱动自带）→ PATH `hy-smi` → 兜底 `docker exec <容器> hy-smi`。输出 `dcu_source` 字段标记数据来源。
- **跨 NUMA 检测**：若 DCU 卡分布在多个 NUMA 节点，`cross_numa=true`，提示集合通信需按卡↔NUMA 亲和性绑核

### 在 DCU 设备上运行
```bash
# 从 Hermes 侧推到设备并执行（示例设备，密码经 sshpass -f 从外部凭证读，不写进 skill）
sshpass -f ~/.config/dcu-toolkit/credentials.json scp collect_topology.sh root@<device_ip>:/tmp/
sshpass -f ~/.config/dcu-toolkit/credentials.json ssh root@<device_ip> "bash /tmp/collect_topology.sh -j"
```
### Pitfalls（多机型实测确认）

1. **DCU 设备识别过宽**：`lspci ... | grep 1d94` 会把海光 Host bridge / PCI bridge 全算进去（实测得到 230 个假 DCU）。**必须** `lspci -nn | grep -i co-processor` 只抓协处理器类（DCU 卡）。
2. **numa_node sysfs 路径不存在**：海光 DTK 内核下 `/sys/bus/pci/devices/<slot>/numa_node` 文件不存在（多机型实测）。NUMA 节点号改从 `lspci -vvv -s <slot>` 的 `NUMA node: N` 行提取。
3. **`nproc` 在 Kylin V10（kernel 4.19.90）返回 255 异常**：`total_cpu` 改用 `lscpu | awk '/^CPU\(s\)/'` 取真实值（跨 NUMA 机型实测 256，nproc 却返 255）。
4. **`numactl` 在 Kylin V10 未安装**：NUMA 节点数与各节点 CPU 列表改从 `lscpu` 的 `NUMA node(s)` + `/sys/devices/system/node/nodeN/cpulist` 读，不依赖 numactl（跨 NUMA 机型（Kylin V10）实测 numactl 命令不存在，原版退化为单节点错误）。
5. **`cpu_model` 截断**：`/proc/cpuinfo` 的 `model name` 字段内含冒号（如 `Hygon C86-4G (OPN:7490)`），用 `awk -F:` 切会丢 `7490)`。改用 `sed 's/^[^:]*:[[:space:]]*//'` 去首个冒号前缀。
6. **8 卡 NUMA 亲和性必须用 hy-smi（lspci 失真）**：`lspci -vvv` 在海光上所有 PCI 设备都报 `NUMA node: 0`，与真实卡↔NUMA 不符。**正解用 `hy-smi --showtoponuma --json`**（海光官方 SMI，对应 rocm-smi）。实测：单 NUMA 机型的 8 卡确实全在 node0（lspci 碰巧对）；跨 NUMA 机型的 card0-3→node0、card4-7→node4（lspci 全报 0 是错的）。脚本现以 hy-smi 为准，`dcu_source` 字段标记数据来源（hy-smi / lspci-unreliable）。
7. **hy-smi 是 DCU 驱动自带**，装驱动后位于 `/opt/hyhal/bin/hy-smi`（宿主机与容器内均有），不一定在 PATH。脚本探测顺序：`/opt/hyhal/bin/hy-smi` → PATH 的 `hy-smi` → 最后兜底 `docker exec <容器> hy-smi`。参考 skill：https://github.com/rookie-J/hy-smi （含完整选项参考，拓扑类 `--showtopo/--showtoponuma/--showtopohops/--showtopoaccess` 等）。


## 待办（绑核优化，未实现，默认不启用 autobinder）

1. **CPU 绑核策略**：参考 autobinder 的 daemon 思路（CCX 独占 → NUMA 距离感知），但**必须适配 DCU**——按 `dcu-topology/scripts/collect_topology.sh` 输出的「卡↔NUMA」做进程↔NUMA 亲和性绑定，而非通用进程绑核
2. **与 dcu-rccl-test 串联**：跨 NUMA 场景下用 `NCCL_TOPO_FILE` / topo_tools 生成 mapping 文件（见 dcu-rccl-test 双机章节）
3. **激活条件**：在 dcu-topology 显式确认适配 DCU 前，`autobinder daemon` 不应对 DCU 训练/推理进程使用
4. **卡↔NUMA 精确亲和性（已解决）**：已由 `hy-smi --showtoponuma --json` 实现（替换失真的 lspci）。后续可选增强：用 `hy-smi --showtopohops/--showtopoaccess` 拿卡间跳数/可达性，生成更精确的 NCCL_TOPO_FILE（见 references/hy-smi.md）

> ⚠️ 任何"自动绑核"动作都视为未授权，直到本 skill 绑核策略实现并显式激活。优先用 dcu-rccl-test 已有 topo_tools 方案。

## References

- autobinder: https://gitee.com/shilei-wang/autobinder （Hygon CPU 绑核，候选参考，默认不启用；源码已 clone 参考，未纳入 skill 包）
- 关联：dcu-env（拓扑采集）、dcu-rccl-test（topo 优化/双机测试）
- 采集脚本：scripts/collect_topology.sh（多机型已验证）
- hy-smi 命令参考（拓扑/监控/危险操作红线）：references/hy-smi.md
