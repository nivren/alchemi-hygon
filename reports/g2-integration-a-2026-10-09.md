# G2 集成基线验收 A（2026-10-09）

## 源码与范围

验收源码：`2e211ab93a4f1ad0c8c720eecb22129d846c7792`，即已集成 HIP Neighbor M1–M9 的
`develop`。本阶段在 `wangleping/test-g2-integration-a` 增加流程、状态和证据记录；
不修改 framework/ops 产品源码、探针、上游锁、Torch/DTK 或数值容差。
已有 M1–M9 集成采用用户授权的 fast-forward 并已推送三处远端，本阶段不补造历史 PR。

原始输出：`artifacts/g2-integration-a-2026-10-09/`（忽略，不入库）。
该目录保存 `cpu-status.json`、`hcu-status.json`、逐任务日志、资源快照、环境、
源码/夹具/native 缓存哈希和各真实模型 summary。命令、起止记录、限时与进程退出码
随任务写入 status JSON，不根据日志中间的 passed 判定整组通过。

## 环境与夹具

| 项目 | 本轮实际配置 |
|---|---|
| 设备 | HCU4，BW1000 64G/gfx936；`HIP_VISIBLE_DEVICES=4` |
| 运行时 | 项目 `.venv`，Python 3.12.13，Torch `2.9.0+das.opt1.dtk2604`，HIP `6.3.26093` |
| DTK/驱动 | `/opt/dtk-26.04/env.sh`；驱动 `6.3.31-V1.5.6`；dcc `25.10.0-0`/clang 17.0.0 |
| 模型 | MACE-MPA-0 medium，mace-torch 0.3.15/e3nn 0.4.4；`enable_cueq=False`、`compile_model=False` |
| 其他依赖 | NumPy 1.26.4、ASE 3.29.0、pytest 8.4.2、tensordict 0.14.1 |
| 上游锁 | Framework `4dfe3723def34df3fadb245981081ccf8c94c257`；Ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` |
| 执行控制 | `OMP_NUM_THREADS=1`、`MAX_JOBS=1`、`PYTORCH_ROCM_ARCH=gfx936`；GPU 作业顺序执行 |

checkpoint SHA256：`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
较大 Batch 复用 `artifacts/hip-real-batch-m7/sources.json` 的 101 个 CIF 清单，逐文件核验
哈希；`--atoms 1000/2000` 分别得到实际 1029 原子/39 结构、2030 原子/92 结构，
生命周期另加入 8 个真实 CIF，分别验证 47/100 个结构的预算退出与补位。
小模型入口使用既有三个 Li/Na CIF，共 18 原子。CIF 和 checkpoint 未写入 Git。
清单 SHA256：`59cbe1ebb189f8d9fe18746c7636a0bd0228b6079836c250672660f8d116172b`。

开始检查 HCU4 显存占用 2 MiB、利用率 0、Normal，无 KFD PIDs；每个 GPU 任务前
再次采集并检查 HCU4 状态/实际显存，发现外部占用则停止后续任务，不等待或换卡。
native 复用 M1 的独立缓存目录，由现有 JIT loader 对当前源码检查/构建；保存源码和
binary 哈希。冷启动/JIT 和探针自身计时均保留，本阶段不作新的性能准入判断。

## CPU gate

| 任务 | 结果 | 退出码 | 限时 | 日志 |
|---|---|---:|---:|---|
| 既有完整指定 reference gate | 425 passed、3 skipped | 0 | 360 s | `cpu-reference.log` |
| HIP 相关 ops，独立进程 | 67 passed、34 skipped | 0 | 180 s | `cpu-hip-ops.log` |
| HIP 相关 framework，独立进程 | 49 passed、28 skipped、1 xfailed | 0 | 240 s | `cpu-hip-framework.log` |
| FP32 输出/状态分层比较器 | 3 passed | 0 | 120 s | `cpu-comparator.log` |

reference gate 为 `HIP_VISIBLE_DEVICES='' bash scripts/check_cpu_reference.sh`；五个 pytest
子进程分别通过 156、1、222、22、24 项，之后 compileall 与 diff 检查通过。
该脚本是项目指定 reference gate，不是全部上游原测试。其 3 个 skip 为实际 Warp 对照缺失；
补充用例的 34/28 个 skip 为未分配 HIP 设备。xfail 为既有 `WARP-DEFAULT-001`，
需在 NVIDIA/Warp 环境执行；没有增加 skip/xfail 或放宽容差。
各 gate 有交叉用例，计数描述各次运行，不声称全部是不同测试。

补充 ops 清单：`test_hip_batch_cell_query_boundary.py`、`test_hip_batch_metadata.py`、
`test_hip_cooperative_query.py`、`test_hip_current_cell_neighbors.py`、
`test_hip_neighbor_executor.py`、`test_hip_batch_neighbor_runtime.py`，均位于
`packages/ops/test/torch/`。

补充 framework 清单：`test/compatibility/` 下 `test_hip_coo_neighbors.py`、
`test_hip_current_cell_neighbors.py`、`test_hip_variable_cell_workflow.py`、
`test_backend_default_boundary.py`；`test/models/test_neighbors_torch_reference.py`；
`test/hooks/test_neighbor_list_torch_reference.py`。

## HCU gate

所有 GPU 任务先激活项目环境，再设置 HCU4/单线程，并加载
`artifacts/hip-current-cell-neighbor-m1/native_env.sh`。

| 任务 | 结果 | 退出码 | 限时 | 日志/summary |
|---|---|---:|---:|---|
| `probes/torch_probe.py --device cuda` | HIP、FP64 一二阶/segment/FFT smoke 通过 | 0 | 60 s | `hcu-torch-probe.log` |
| 同 CPU ops 清单 | 101 passed | 0 | 300 s | `hcu-hip-ops.log` |
| 同 CPU framework 清单 | 77 passed、1 xfailed | 0 | 240 s | `hcu-hip-framework.log` |
| `probes/framework_neighbor_hip_contract.py` | 46×32 两精度、空输入、容量、重复 Hook 和 unsupported 请求通过 | 0 | 180 s | `hcu-public-fixed-cell.log` |
| `probes/hip_coo_mace_snapshot.py` | 两精度各 initial/shear/wrap、Hook E/F/stress 与 FP64 有限差分通过 | 0 | 240 s | `coo-mpa0.json` |
| `probes/hip_variable_cell_mace_workflow.py` | 两精度×两优化器，各五步与三结构预算补位通过 | 0 | 240 s | `mpa0-short-workflow.json` |

HCU ops/framework 包含 CPU oracle 与实际 HIP 参数项，不是全库 HCU gate。
公共 gate 容量 `1→8`、最大有效邻居数 6、保守上界 343；no-PBC、periodic half 和 skin
请求明确拒绝。当前晶胞收缩/剪切、容量增长、MATRIX/COO、排序、非默认 stream、
重叠/溢出/整数上界失败和低层 geometry 一二阶路径由指定用例覆盖。
低层 mixed-PBC/half/geometry 证据不扩大公共 HIP capability。

## 较大真实 Batch

使用未改动的 `probes/hip_real_batch_validation.py`，两后端分别为显式
`torch_reference` cell-list 与 `hip` cell-list；优化器均为现有 Torch reference。
每组 3 轮×3 步，每轮 priming 与各步比较共 12 次 trajectory/state 检查；另比较 initial/step3
同几何输出，并执行真实预算补位。模型 E/F/stress 按原物理量容差，坐标/晶胞/速度和私有
状态按原状态容差；拓扑、Batch ptr、ID、状态及计数精确比较。每组限时 300 s。

| 实际原子/结构 | dtype | 优化器 | 整组结果 | 移动晶胞/生命周期结构 |
|---|---|---|---|---|
| 1029/39 | FP64 | FIRE | passed，exit 0 | 47/47 |
| 1029/39 | FP64 | FIRE2 | passed，exit 0 | 47/47 |
| 2030/92 | FP64 | FIRE | passed，exit 0 | 100/100 |
| 2030/92 | FP64 | FIRE2 | passed，exit 0 | 100/100 |
| 1029/39 | FP32 | FIRE | passed，exit 0 | 47/47 |
| 1029/39 | FP32 | FIRE2 | passed，exit 0 | 46/47 |
| 2030/92 | FP32 | FIRE | passed，exit 0 | 100/100 |
| 2030/92 | FP32 | FIRE2 | passed，exit 0 | 99/100 |

两组 FP32 FIRE2 的未移动项仍为 source 36：`mp-1013795_Li3Ge3(MoO6)2.cif`，
分别对应补位生命周期的 system ID 27/42。
其 HIP 和 Torch reference 的最终晶胞都与初态完全相同，cross-backend 晶胞差为 0；
全部输出与状态比较通过。保持 M9 的 `moving > 0` 条件，并逐结构保存诊断。

按四组/精度的 summary 中记录的物理量与状态比较汇总最大绝对差：

| dtype | 能量 eV | 力 eV/Å | 应力 eV/Å³ | 坐标 Å | 晶胞 Å |
|---|---:|---:|---:|---:|---:|
| FP64 | 6.82e-13 | 6.98e-14 | 3.28e-15 | 1.78e-15 | 1.78e-15 |
| FP32 | 4.88e-4 | 6.76e-5 | 1.68e-6 | 9.54e-7 | 4.77e-7 |

沿用原 gate：FP64 物理量 rtol `2e-8`，E/F/stress atol `2e-7/2e-7/2e-9`；
FP32 物理量 rtol `2e-4`，E/F/stress atol `2e-3/2e-4/2e-6`。
通用浮点状态 FP64 rtol/atol `2e-8/2e-9`，FP32 `2e-4/2e-5`；
离散状态保持精确比较。这里报告实际记录值，不把不同用途的容差混用。
小模型 FP64 finite difference 另录最大 force/strain-stress 误差
`2.43e-10 eV/Å`、`2.73e-12 eV/Å³`，所测拓扑未改变。
完整各字段数值见本轮 `validation-summary.json` 和各组 `summary.json`。

复现一组的入口（先按上述 HCU 环境激活；参数替换得到其余组合）：

```bash
timeout 300 .venv/bin/python -u probes/hip_real_batch_validation.py \
  --checkpoint /home/wangleping/mace-mpa-0-medium.model \
  --data-root /data/csp_data/catl_data/li_mp_data \
  --manifest artifacts/hip-real-batch-m7/sources.json \
  --output-dir artifacts/g2-integration-a-2026-10-09/mpa0-1000-float64-fire \
  --atoms 1000 --dtype float64 --optimizer fire --lifecycle
```

## 源码审查与限制

核对 `fb05b61..2e211ab` 的集成范围，重点检查 Hook padding/容量、registry、当前晶胞 metadata
和协作 query。仅裁去 padding，counts 超容量明确失败；HIP `auto_enabled=False`，
每次按当前晶胞建新 plan，旧 plan 变胞复用未准入；模型 COO 仍由设备内 Torch 打包。
保留已有容量计算/错误检查的 host readback 与同步，不将其描述为全设备异步路径。

Warp 默认选择/失败边界有 CPU/reference 检查；实际 Warp/NVIDIA 数值仍未执行。
部分 Hook Dynamo 包装触发 `recompile_limit` 提示并转为 eager；日志保留，不能据此宣称
完整 compile 已验证。预算退出/有限步移动不等于物理收敛。
没有新增长收敛、全目录/其他模型、20k 真实 MLIP 吞吐、全库 HCU、native dynamics、
skin/selective rebuild、no-PBC/half 公共准入、native geometry backward、DomainParallel 或 auto。
本阶段不更新历史性能结论；各原阶段报告保留当时成功/失败及已知限制。

## 阶段状态

A 的指定集成验收已完成：4 个 CPU 任务、14 个 HCU 任务均退出 0；
其中 8/8 较大真实 Batch summary 为 passed。收尾设备检查 HCU4 Normal、利用率 0、
显存 2 MiB，无 KFD PIDs。源码、锁、探针与夹具的 912 项记录哈希复核未改变。
长期流程已写入 [UPSTREAM_SYNC.md](../docs/UPSTREAM_SYNC.md)，并由 AGENTS/开发指南链接。
状态/交接/任务队列和相关特性证据已更新；历史阶段报告保留原结论，当前能力限制不扩大。
用户于 2026-10-09 确认 A 验收及下一阶段 B；A 文档由本阶段提交收口，保留完整测试记录。
本轮未推送或再次合并 develop。
B 的差异工具、候选记录和首次报告已获启动授权，由独立阶段实施。
