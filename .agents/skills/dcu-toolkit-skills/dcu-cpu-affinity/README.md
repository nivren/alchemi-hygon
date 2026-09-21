# dcu-cpu-affinity — 海光 DCU NUMA 绑核优化

NUMA 亲和性绑核方案生成 skill。基于 dcu-topology / dcu-hy-smi 的真实卡↔NUMA 拓扑，
为每张 DCU 卡生成 `numactl --cpunodebind=N --membind=N` 绑定方案，避免进程跨 NUMA
访问内存（跨 socket 延迟高、带宽低，拖慢集合通信与推理）。

## 采用方式

**节点级（轻量）**：每张卡绑到其 NUMA 节点，`numactl --cpunodebind=X --membind=X`。
RCCL/推理框架自身读 NUMA 拓扑调度，我们只需保证进程不跨 NUMA 抢内存。不做细绑 core
（NPS 交织下 node 含跨 socket 核，细绑易踩坑）。

## ⚠️ 安全纪律

仿 autobinder "默认不启用"：脚本**默认 dry-run 只打印**，不执行 numactl；
`--apply` 仅写映射文件，不 kill/重启进程。绑核高危，需显式确认才落地。

## 目录

```
dcu-cpu-affinity/
├── SKILL.md                 设计 + 安全纪律 + Pitfalls（多机型实测）
└── scripts/
    └── gen_affinity.sh      绑定方案生成器（吃拓扑JSON，--cmd/--json/--apply）
```

## 快速用法

```bash
# 先用 dcu-topology 采拓扑
bash ../dcu-topology/scripts/collect_topology.sh -j > /tmp/topology.json

# 生成方案（dry-run）
bash scripts/gen_affinity.sh --topo-json /tmp/topology.json \
  --cmd "python train.py --device 0"

# 写映射文件（不执行）
bash scripts/gen_affinity.sh --topo-json /tmp/topology.json --apply
```

## 实测要点

- **跨 NUMA 机型**：卡分散在多个 NUMA 节点（如 card0-3→N0、card4-7→N4），须分组绑，混绑反而降速
- **单 NUMA 机型**：8 卡全在同一 NUMA 节点，全绑该节点即可，主要防 OS 调度漂移，收益有限（保险式）
- 校验 `dcu_source=hy-smi`，否则拒绝（lspci 失真会导致绑错）

详见 `SKILL.md` Pitfalls。
