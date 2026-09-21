# dcu-rccl-test — 海光 DCU 集合通信（RCCL）通测

集合通信（RCCL）通测 skill。对多卡/多机的 DCU 集群做 allreduce / allgather 等集合
通信性能测试，验证拓扑与绑核优化效果（与 dcu-topology、dcu-cpu-affinity 衔接）。

## 核心内容

- 多机 RCCL 测试流程（NCCL_TOPO_FILE / topo_tools 生成 mapping，跨 NUMA 场景）
- 单机 8 卡 allreduce 带宽基线
- 与 dcu-cpu-affinity 串联：绑前 vs 绑后带宽对比，量化绑核收益

## 状态

✅ 框架已搭（路由可达）。具体测试脚本与双机带宽基线数据在持续补充中。
详见 `SKILL.md`。

## 衔接

- 拓扑采集：dcu-topology（提供卡↔NUMA）
- 绑核方案：dcu-cpu-affinity（提供 numactl 绑定命令）
- 验证收益：本 skill 跑 allreduce，对比绑前/绑后

> 注：本包当前版本 dcu-cpu-affinity 未内置验证步骤，
> 若需量化绑核收益可在此 skill 跑对比。
