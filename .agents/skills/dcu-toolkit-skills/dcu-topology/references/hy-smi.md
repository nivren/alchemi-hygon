# hy-smi 海光 DCU 管理工具参考（知识库）

> 来源：https://github.com/rookie-J/hy-smi （Hermes Agent skill，含完整命令参考）
> 对应 AMD ROCm 生态的 `rocm-smi`。支持 C-3000 系列 HCU（DCU 卡）。

## 关键事实（实测确认）

- **hy-smi 是 DCU 驱动自带的**，安装 DCU 驱动后位于 `/opt/hyhal/bin/hy-smi`（宿主机与容器内均有），不一定在 PATH。
- 调用方式：`/opt/hyhal/bin/hy-smi <opts>`，或容器内 `docker exec <dcu容器> hy-smi <opts>`。
- 结构化输出：`--json`（推荐解析）、`--csv`。
- 需要 root 权限（部分操作）；容器内需 `--privileged` 或映射 `/dev/kfd` + `/dev/dri`。
- 驱动版本要求：6.2.22+（支持 `--finegrain` 等）。

## 拓扑 / 连接类（DCU 调优最常用）

| 命令 | 作用 |
|------|------|
| `hy-smi --showtoponuma` | **卡↔NUMA 节点亲和性**（最准，替代失真的 lspci） |
| `hy-smi --showtopo` | 硬件拓扑总览 |
| `hy-smi --showtopoaccess` | HCU 间链接可达性矩阵 |
| `hy-smi --showtopoweight` | HCU 间相对权重 |
| `hy-smi --showtopohops` | HCU 间跳数（生成 NCCL_TOPO_FILE 用） |
| `hy-smi --showtopotype` | 链接类型（XGMI / PCIe） |
| `hy-smi --shownodeid` | 节点 ID |
| `hy-smi --showxhclstatus` | xhcl 链接状态 |
| `hy-smi --showxgmierr` | XGMI 错误计数 |

> 用 `--showtoponuma --json` + 解析，可直接喂给 dcu-topology 的 `dcu-topology/scripts/collect_topology.sh`（已集成）。
> 用 `--showtopohops/--showtopoaccess` 可进一步生成卡间通信映射，配合 `NCCL_TOPO_FILE` 优化多卡集合通信（见 dcu-rccl-test）。

## 监控类（快速巡检）

| 用途 | 命令 |
|------|------|
| 概览 | `hy-smi -a` |
| 温度/功耗 | `hy-smi -t -P` |
| 利用率(HCU/CU/波前) | `hy-smi -u --showhcuutil --showcuutil --showwaveutil` |
| 显存使用 | `hy-smi --showmemuse --showmemavailable` |
| 时钟频率 | `hy-smi -c -g --showclkfrq` |
| 运行进程 | `hy-smi --showpids` |
| 设备ID/序列号 | `hy-smi -i --showserial --showuniqueid` |
| 设备/总线 | `hy-smi --showbus`（PCI 地址，可与 lspci slot 对应） |
| 健康/RAS | `hy-smi --healthcheck` / `--showrasinfo BLOCK` / `--showexceptioninfo` |

## 危险操作红线（设置类，默认不自动执行）

- 🔴 `--setmemiovol`（烧显存）、`--setpoweroverdrive`（超功耗）、`--rasinject`（注入 poison）、`--loaddriver/--unloaddriver`（卸载驱动会中断任务）
- 🟡 `--setsclk/--setmclk/--setsocclk`（超频，先 `--setperflevel manual`）、`--enablelowpower`、`--setboost`、`--rasenable/disable`
- 🟢 低风险：`--setperflevel`、`--setprofile`、`--setmaxprocess`、`--save/--load`

> ⚠️ 任何写操作（设频率/功耗/电压）在 DCU 上视为未授权，需用户显式确认。优先用查询类。

## 排错

- 驱动未加载：`lsmod | grep -E "hydcu|hycu"`
- 权限拒绝：加 sudo 或 `--privileged`
- 设备未找到：查 HCU 是否在线
- 内核日志：`dmesg -T | grep hycu`
- 异常详情：`--showexceptioninfo`
