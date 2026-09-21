# dcu-topology — 海光 DCU 拓扑采集

DCU 拓扑采集 skill。采集目标机的 CPU/NUMA 拓扑、DCU 卡↔NUMA 亲和性，是绑核优化
（dcu-cpu-affinity）和集合通信调优（dcu-rccl-test）的数据基础。

## 核心内容

- **卡↔NUMA 权威源**：`hy-smi --showtoponuma --json`（lspci 在海光上失真，全报 0）
- **CPU/NUMA 拓扑**：从 `lscpu` + `/sys/devices/system/node/` 读（不依赖 numactl）
- **采集脚本**：`scripts/collect_topology.sh`（纯标准命令，支持 `-j` JSON 输出）

## 目录

```
dcu-topology/
├── SKILL.md                 采集规范 + Pitfalls（多机型实测）
├── scripts/
│   └── collect_topology.sh  拓扑采集脚本（多机型验证）
└── references/
    └── hy-smi.md           hy-smi 拓扑选项参考
```

## 快速用法

```bash
# 本地/远程采拓扑（JSON）
bash scripts/collect_topology.sh -j > /tmp/topology.json

# 远程（配合 dcu-ssh）
sshpass -f ~/.config/dcu-toolkit/credentials.json scp scripts/collect_topology.sh root@<device_ip>:/tmp/
sshpass -f ~/.config/dcu-toolkit/credentials.json ssh root@<device_ip> "bash /tmp/collect_topology.sh -j"
```

输出含：`cpu_model`、`total_cpu`、`numa_nodes`、`numa_cpus`、`dcu_devices[].numa_node`、
`dcu_source`（标记 hy-smi / lspci-unreliable）、`cross_numa`。

详见 `SKILL.md` Pitfalls（lspci 失真、nproc 异常、numactl 缺失等）。
