---
name: dcu-cpu-affinity
description: 海光DCU CPU绑核优化（NUMA亲和性）。基于 dcu-topology / dcu-hy-smi 采集的真实卡↔NUMA拓扑，为每张DCU卡生成 numactl 进程↔NUMA 绑定方案（节点级 cpunodebind+membind）。默认只生成方案+dry-run打印，不执行；绑核属高危操作，需显式确认才落地。
license: MIT
metadata:
  version: 1.0.0
  author: DCU Engineer
  hermes:
    tags: [dcu, hygon, numa, affinity, bind, numactl, tuning, topology]
    related_skills: [dcu-topology, dcu-hy-smi, dcu-rccl-test]
---

# dcu-cpu-affinity — 海光 DCU CPU 绑核优化（NUMA 亲和性）

## 目标

基于**真实拓扑**（dcu-topology / dcu-hy-smi 采集的 `cardN → NUMA node X`），
为每张 DCU 卡生成 NUMA 亲和性绑定方案：**让每张卡的 Rank/进程就近其所在的 NUMA 节点**，
避免进程跨 NUMA 节点访问内存（跨 socket 内存访问延迟高、带宽低，直接拖慢集合通信与推理）。

## 采用方式：**节点级（轻量）**

- 每张卡 N 绑定到其 NUMA 节点 X：`numactl --cpunodebind=X --membind=X <启动命令>`
- RCCL / 推理框架自身会读 NUMA 拓扑做通信调度，我们只需保证**进程不跨 NUMA 抢内存**
- **绑定粒度 = 进程↔NUMA 节点（粗粒度）**，不用细绑到具体 core
  - 理由：海光 NPS 交织模式下 NUMA node 内含跨 socket 的核（如跨 NUMA 机型的 node0 = 0-15,128-143），
    细绑 core 反而易踩交织坑；节点级已满足"不跨 NUMA"目标

## ⚠️ 安全纪律（必须遵守）

> 绑核是**高危操作**（影响进程调度、可能降低而非提升性能若绑错），沿用 autobinder 的"默认不启用"纪律：

1. **默认 dry-run**：脚本默认只打印绑定方案与可执行命令，**不修改任何进程、不执行 numactl**
2. **`--apply` 才落地**：需用户显式传 `--apply`；且脚本只**生成命令/写映射文件**，真正的进程启动由用户用生成的命令执行（脚本不替用户 `kill`/重启线上进程）
3. **绝不自动绑线上进程**：任何对正在运行的 DCU 训练/推理进程自动绑核，视为未授权
4. 生成方案前必须确认拓扑来源可信（`dcu_source=hy-smi`，非 `lspci-unreliable`）

## 输入（拓扑数据来源）

两种取数路径，脚本自动选：

1. **直接读 dcu-topology 的采集 JSON**（推荐）：`dcu-topology/scripts/collect_topology.sh -j` 输出，字段：
   - `dcu_devices[].{pci_slot, numa_node}` — 卡↔NUMA
   - `numa_cpus.{node: "cpulist"}` — 各 NUMA 节点的 CPU 列表
   - `dcu_source` — 数据来源（必须 `hy-smi` 才可信）
2. **直接用 hy-smi 现场采集**（无 JSON 时）：`dcu-hy-smi` 的 `dcu-hy-smi/scripts/hy_smi_query.sh topo` 拿 `--showtoponuma`，
   NUMA cpulist 从 `/sys/devices/system/node/nodeX/cpulist` 读

## 输出（绑定方案）

对每张卡生成：

```bash
# 卡0 在 NUMA0 → 绑 node0 的 CPU 与内存
numactl --cpunodebind=0 --membind=0 <原启动命令>

# 卡4 在 NUMA4 → 绑 node4
numactl --cpunodebind=4 --membind=4 <原启动命令>
```

并输出映射表（JSON + 可读）：
```
GPU0 (PCI 07:00.0) -> NUMA 0  cpus: 0-15,128-143
GPU1 (PCI 0a:00.0) -> NUMA 0  cpus: 0-15,128-143
...
GPU4 (PCI 87:00.0) -> NUMA 4  cpus: 64-79,192-207
```

同时给出**分组提示**：
- 若 `cross_numa=true`（如跨 NUMA 机型：卡分散在多个 NUMA 节点），明确分两组，绝不混绑
- 若 `cross_numa=false`（如单 NUMA 机型：8卡全在同一 NUMA 节点），提示"全在单 NUMA，绑定收益有限，主要防 OS 调度漂走"

## 用法

```bash
# 1. 先用 dcu-topology 采拓扑（或 dcu-hy-smi 现场采）
bash dcu-topology/scripts/collect_topology.sh -j > /tmp/topology.json

# 2. 生成绑定方案（默认 dry-run，只打印）
bash dcu-cpu-affinity/scripts/gen_affinity.sh --topo-json /tmp/topology.json

# 3. 带原启动命令，生成可直接复制的 numactl 命令
bash dcu-cpu-affinity/scripts/gen_affinity.sh --topo-json /tmp/topology.json \
  --cmd "python train.py --device 0"   # 会为每张卡替换 device 并加 numactl 前缀

# 4. 仅输出 JSON 映射（供其他脚本消费）
bash dcu-cpu-affinity/scripts/gen_affinity.sh --topo-json /tmp/topology.json --json

# 5. 真正落地（写映射文件 + 打印执行命令，不自动重启进程）
bash dcu-cpu-affinity/scripts/gen_affinity.sh --topo-json /tmp/topology.json --apply
```

> `--apply` 只生成命令和可选的映射文件（如 `/tmp/dcu_affinity_map.json`），**不替用户 kill/重启进程**。
> 用户拿到命令自行执行。这是安全护栏。

## Pitfalls（实测确认 — 多机型）

1. **跨 NUMA 必须分组绑**：跨 NUMA 机型实测卡分散在多个 NUMA 节点（如 card0-3→N0、card4-7→N4）。若把 card4 绑到 node0 的 CPU，
   其显存在 node4，内存访问跨 socket（hops=3），性能反而下降。必须按 `card→NUMA` 严格分组。
2. **NPS 交织下 NUMA 节点含双 socket 核**：跨 NUMA 机型的 node0 = `0-15,128-143`（两个 socket 各 16 核并入 node0）。
   节点级绑定会把进程放到这些核上，但这些都是 node0 本地内存，没问题；**不要进一步细绑到某一 socket 的核**，
   否则可能和另一 socket 的卡抢 node0 内存带宽。粗绑节点即可。
3. **全单 NUMA 时绑定收益有限**：单 NUMA 机型的 8 卡全在同一 NUMA 节点，绑该节点主要防止 OS 调度把进程漂到其他 NUMA 节点，
   对卡间通信本身无改善（hops 已由硬件拓扑决定）。预期收益小，属"保险式绑定"。
4. **`dcu_source` 必须 hy-smi**：若拓扑来自 `lspci-unreliable`，卡↔NUMA 可能失真（lspci 全报 0），
   生成的绑定方案会全绑 node0，对跨 NUMA 机型是错的。脚本会校验并拒绝（除非 `--force`）。
5. **numactl 可能未装**：老内核（Kylin V10 / 4.19）机型实测 `numactl` 命令不存在。绑定命令需要 numactl，
   目标机若无则需先装（`yum/apt install numactl`）；脚本生成命令时会检测并告警。
6. **不解决卡间 hops 路径优化**：hops（单 NUMA 机型组内1/跨组2，跨 NUMA 机型组内2/跨组3）是硬件拓扑决定的，
   numactl 绑 NUMA 只保证内存就近，不重排卡间通信路径。要优化通信路径需 NCCL_TOPO_FILE 强制拓扑，本 skill 不做。
7. **容器内需在容器内执行 numactl**：若进程跑在 docker 容器内，numactl 必须在容器内执行
   （`docker exec <c> numactl ...`），且容器需 `--privileged` 或 `--cap-add SYS_NICE` 才能改亲和性。

## 与 dcu-rccl-test 的衔接（可选，本 skill 不做实测）

绑核后若有 dcu-rccl-test 的 allreduce 带宽测试，可对比绑前/绑后带宽验证收益。
跨 NUMA 机型场景预期收益明显（消除跨 socket 内存访问），单 NUMA 机型预期收益小。
本 skill 默认不含验证步骤，仅在 Pitfalls #3/#6 标注预期。

## References

- 关联：dcu-topology（`dcu-topology/scripts/collect_topology.sh` 采拓扑，本 skill 的输入）、dcu-hy-smi（`--showtoponuma` 拓扑源）、
  dcu-rccl-test（可选验证带宽）
- 设计参考：autobinder（Hygon CPU 绑核思路，本 skill 仅取"NUMA 距离感知 + 默认不启用"纪律，不做 daemon 自动绑）
