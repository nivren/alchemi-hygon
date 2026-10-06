# HIP 邻居变胞 FIRE/FIRE2 工作流 M3

日期：2026-10-05。起点 M2 实现 `3e394d7`、文档 `4136f0f`；分支
`codex/test-hip-variable-cell-workflow`。M3 已获用户确认，当前完成窄验证，已收口实现 `6b1680a`，未合并或推送。
2026-10-06 用户确认 M4，按约定收口本阶段。

## 目的与实现

以现有公共 `FIRE2VariableCell` / `FIREVariableCell`、`NeighborListHook`、`FusedStage`
组合真实 MPA-0 弛豫短步与 Batch 生命周期。优化器/stress/align 为 Torch reference，
邻居显式 HIP cell-list，native MATRIX 后设备内 Torch 打包 COO；full-PBC/full-list/skin=0。

本阶段无需修改产品算子或优化器。新增契约、参数化工作流测试与可重复真实模型探针：

- `docs/HIP_VARIABLE_CELL_WORKFLOW_CONTRACT.md`。
- `packages/framework/test/compatibility/test_hip_variable_cell_workflow.py`：CPU/reference 和
  HCU/HIP，两优化器×两精度×四类测试。每次解析模型调用独立重建 reference 拓扑，
  精确核对 `(i,j,shift)` 多重集合和体系归属；解析模型本身不使用邻居求能量。
- `probes/hip_variable_cell_mace_workflow.py`：真实 MPA-0 两邻居分支各五步，逐步比较
  r/H、速度、E/F/stress 和全部私有状态；另外在 HIP 当前几何独立重算 reference。
  三结构容量2的 FusedStage 预算退出/补位，记录来源ID、退出状态、ptr和首次 kick 前输出。

接线遵守现有 API：NeighborListHook 必须显式登记 `BEFORE_COMPUTE`；FusedStage 共享
计算由其自身邻居 Hook 重建，sub-stage 保留 AlignCellHook/优化器和退出 Hook。
SizeAwareSampler 返回 host 数据，测试/探针向 `run(batch=...)` 显式传入 GPU Batch。
FIRE2 初始计算前显式执行对齐预处理，随后六项下三角；FIRE 保留 fork 完整矩阵，
本探针 `dt=0.02, dt_min=0`，不改变公共默认值。

## 验证环境与执行

project `.venv`，DTK26.04，Torch2.9.0/HIP6.3.26093，HCU4 BW1000 64G/gfx936；
普通 MACE/e3nn eager，cueq/模型compile关闭，冻结权重，位置/应变 autograd 保持开启。
运行前 hy-smi 无 KFD 计算进程；只占 HCU4，`OMP_NUM_THREADS=1`，每项180/240秒限时。
复用 M1 native build cache，无新增 kernel 编译或系统修改。

执行前 `source scripts/activate_hygon_env.sh project`；HIP gate 另
`source artifacts/hip-current-cell-neighbor-m1/native_env.sh`。

| gate | 命令主体 | 结果 |
|---|---|---|
| CPU 新工作流 | `HIP_VISIBLE_DEVICES='' timeout 180 .venv/bin/python -m pytest -q packages/framework/test/compatibility/test_hip_variable_cell_workflow.py` | 16 passed, 16 skipped |
| CPU 相关回归 | 同环境，`test_variable_cell_fire_reference.py test_variable_cell_workflow.py test_hip_coo_neighbors.py test_hip_current_cell_neighbors.py` | 48 passed, 12 skipped |
| HCU4 新工作流 | `HIP_VISIBLE_DEVICES=4 timeout 180 .venv/bin/python -m pytest -q packages/framework/test/compatibility/test_hip_variable_cell_workflow.py -k cuda` | 16 passed, 16 deselected |
| HCU4 真实模型 | `HIP_VISIBLE_DEVICES=4 timeout 240 .venv/bin/python probes/hip_variable_cell_mace_workflow.py --checkpoint /home/wangleping/mace-mpa-0-medium.model --data-root /data/csp_data/catl_data/li_mp_data --output artifacts/hip-variable-cell-workflow-m3/mpa0-workflow.json` | 两优化器×两精度，五步轨迹及三步预算补位全部通过 |

以上最终命令退出码均0。CPU skips 为未分配 HIP 的参数项；deselected 为 CPU 参数项。
compileall 与 `git diff --check` 通过；未运行全 CPU/HCU suite 或其他 ops gate，本阶段产品代码未改。
NeighborListHook 原有 Dynamo 包装出现 recompile-limit 提示后执行 eager；没有 CPU
计算回退，所有连续数据/输出均核对 dtype/device。此处不作为 compile 支持证据。

首次 CPU 因夹具漏登记 Hook stage 失败，首次 HCU Fused 夹具因 host Batch 失败；
均修正测试接线并重跑最终版本，失败日志保留于 artifacts，不计作成功 gate。

## 数值与生命周期

checkpoint SHA256：`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
样本 `mp-1222651_Li2MgGe.cif`(4)、`mp-998591_LiSnCl3.cif`(10)、
`mp-1096278_NaLiTl2.cif`(4)，CIF SHA256 和逐步数据见 JSON。轨迹 Batch ptr `[0,4,14,18]`。

下表为五步+初始几何和补位首次 kick 前的同几何 reference 对照最大绝对误差：

| 精度 | E (eV) | F (eV/Å) | stress (eV/Å³) | 两分支位置 (Å) | 两分支晶胞 (Å) |
|---|---:|---:|---:|---:|---:|
| FP64 | 1.42e-14 | 1.50e-14 | 2.06e-16 | 1.23e-18 | 2.17e-18 |
| FP32 | 3.81e-6 | 7.80e-6 | 8.29e-8 | 5.73e-10 | 2.22e-9 |

整数状态与拓扑精确一致；浮点私有状态按契约逐字段检查，不将上表 r/H 误差当作
全部状态误差。所有晶胞正体积、输出有限，r/H 均实际改变。没有放宽容差。

解析势四类测试覆盖12步对照、inactive 字段/私有状态精确冻结、幸存者体系状态重排、
arrival 新状态逐字段比较、补位当前拓扑与输出，以及联合力/绝对应力退出与预算退出区分。
真实模型补位每个精度/优化器均三结构全部毕业，source0/1/2分别预算1/3/2次更新，
step_count=3，体系ID无丢失或重复，状态全部2（预算）；ptr由 `[0,4,14]` 变
`[0,10,14]`。arrival 首次 kick 前核对当前邻居、E/F/stress；真实模型未宣称物理收敛。

## 边界与下一阶段

本阶段为明确配置下 HIP 邻居 + Torch reference dynamics 的短步工作流准入。
不等于 native HIP 优化器、全目录/长弛豫、完整上游/HCU gate、其他 MLIP、训练力损失、
checkpoint、DD、NPT/NPH、compile/opcheck 或性能准入。skin/rebuild、旧 plan metadata
复用、native COO、query 并行、workspace复用和 auto 仍未新增。

建议 M4：先实测当前邻居 rebuild/metadata/CSR/query/COO 与端到端开销，区分冷启动、
预热和模型耗时；据结果实现保持当前晶胞完整重建语义的 workspace容量复用，保留
N/B/ptr/晶胞/容量改变回归。query 并行策略由实测热点决定，另阶段确认。
M4 已于2026-10-06确认，后续证据独立报告。

原始证据：`artifacts/hip-variable-cell-workflow-m3/{cpu-workflow.log,cpu-regression.log,hcu-workflow.log,mpa0-workflow.log,mpa0-workflow.json}`。
