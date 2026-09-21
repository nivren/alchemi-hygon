# dcu-hy-smi — 海光 DCU 系统管理接口

海光 DCU（C-3000 HCU）的 SMI 工具查询 skill，对应 AMD ROCm 的 `rocm-smi`。
用于监控设备状态（温度/功耗/利用率/显存）、拓扑（卡↔NUMA 亲和性、卡间链接）、
带宽、RAS 错误，故障排查。是 dcu-topology 的拓扑数据源，也是日常巡检工具。

## 核心内容

- **工具定位**：`hy-smi` 是 DCU 驱动自带，位于 `/opt/hyhal/bin/hy-smi`（宿主机/容器内均有，
  不一定在 PATH），优先绝对路径，不必非走 docker exec
- **拓扑关键选项**：
  - `--showtoponuma`：卡↔NUMA 亲和性（**唯一可信源**，lspci 失真）
  - `--showtopohops`：卡间跳数（仅表格，不支持 --json）
  - `--showtopoaccess/--showtopoweight/--showtopotype`：链接可达性/权重/类型
- **监控选项**：`-t -P`（温度/功耗）、`-u --showhcuutil`（利用率）、`--showmemuse`（显存）

## 目录

```
dcu-hy-smi/
├── SKILL.md                 命令参考（基于 rookie-J/hy-smi）+ 多机型实测 Pitfalls
└── scripts/
    ├── hy_smi_query.sh      封装：自动探测 hy-smi 路径，topo/monitor 子命令 + 透传
    └── examples.sh          原仓库示例脚本（参考）
```

## 快速用法

```bash
# 自动探测路径，出拓扑（卡↔NUMA + 卡间hops）
bash scripts/hy_smi_query.sh topo

# 监控概览
bash scripts/hy_smi_query.sh monitor

# 透传任意参数
bash scripts/hy_smi_query.sh --showpids --json
```

详见 `SKILL.md` Pitfalls（hy-smi 路径、仅 --showtoponuma 支持 json、容器内权限）。
