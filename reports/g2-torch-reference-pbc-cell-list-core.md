# G2 Torch reference PBC cell-list core

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段一。

## 实现

- 将 `torch_reference_cell_list.py` 从 no-PBC 27-cell 窄实现扩展为独立于 Warp 的
  build/query 分层实现，覆盖周期、非周期和 mixed-PBC 的正交/三斜晶胞。
- 支持单体系与连续 Batch、full/half、MATRIX/COO、image shift、distance/vector、显式
  capacity overflow、预分配 buffer 和按 system 的 selective rebuild。
- cell 搜索半径由 `cell^{-1}` 的 reciprocal norm 保守计算，不假定 cutoff 只覆盖相邻
  27 个 cell；最终仍以严格 `distance < cutoff` 过滤。
- topology 不可微；由 positions/cell 重新计算的 vector/distance 保留一阶和二阶 Torch
  梯度。`target_indices`、`pair_fn`、pair-centric query 和 compile/custom-op 仍显式失败或延期。
- registry 只扩宽显式 `backend="torch_reference", method="cell_list"` capability；
  `backend=None`/Warp、`auto` 选择和未知后端失败行为不变。

## 测试与数值证据

项目 CPU gate `scripts/check_cpu_reference.sh` 退出码为 0：ops `40 passed`，framework 主集
`135 passed, 1 deselected`，state `22 passed, 34 deselected`，dynamics ops
`24 passed, 73 deselected`；另有 framework import `1 passed`。

focused pytest 另在 DTK 26.04、`HIP_VISIBLE_DEVICES=0` 的主机权限进程中运行：

- ops cell-list/registry：`10 passed, 17 deselected`；
- framework one-shot/NeighborListHook：`2 passed, 22 deselected`；

这两组测试自身构造 CPU tensor，因此只计 CPU contract 回归，不计 HCU kernel 证据。真实
HCU 证据来自同一环境、设备 `BW200, UBB BW1000` 上的
`probes/pbc_cell_list_reference.py --device cuda`：退出码 0，mixed/triclinic Batch 的
  `(source,target,shift)` 与 dense reference 一致，full/half 边数为 `8/4`，分层 build/query
  counts 为 `[1,1]`，二阶梯度有限。

更新后的 `HIP_VISIBLE_DEVICES=0 scripts/check_hcu_reference_smoke.sh` 六个 probe 全部完成且
脚本退出码为 0；除新增 PBC cell-list 外，既有 neighbor/LJ、PBC neighbor、Velocity Verlet、
kinetics 与 FIRE/FIRE2 golden paths 均保持通过。

额外以固定随机种子生成 20 组 FP64 triclinic/mixed-PBC 输入，cell-list 与 dense 的
`(source,target,shift)` 集合 `20/20` 一致；unit cell cutoff `1.1/2.1/3.1` 的多 image/self-image
案例分别得到 `16/152/480` 条边并与 dense 精确一致。

## 阶段一性能基线

同一空闲 HCU 0、MACE-OFF23-small、3 次 warm-up、10 个 steady host-wall samples：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python probes/unified_reference_benchmark.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --neighbor-method dense --scale-sizes 46 92 --batch-root /data/csp_data/perf_92 \
  --batch-sizes 1 4 --warmup 3 --steady 10 --skip-e2e
# 第二次运行仅将 --neighbor-method dense 改为 cell_list。
```

| workload | dense median | Torch cell-list median | cell-list / dense |
|---|---:|---:|---:|
| periodic 46 atoms | 3.247 ms | 8.895 ms | 2.74x |
| periodic 92 atoms | 3.742 ms | 8.900 ms | 2.38x |
| periodic batch 4x92 | 9.778 ms | 27.797 ms | 2.84x |

20 步 periodic `[46,92]` MACE/FIRE2 + skin 的独立进程短跑分别为 dense `9.142 s`、
cell-list `8.512 s`。该测量包含首次模型/kernel 初始化且只有一条端到端样本，差异仅作描述，
不构成 20% 生产门槛通过证据。

`hipprof --hip-trace --stats` 的最小 probe 记录在
`artifacts/pbc-cell-list-stage1-hiptrace.db`（不入库）。应用发起 `2728` 次
`hipLaunchKernel`；HIPOPS 时间分散在 indexing、elementwise、radix sort、scan/reduce 等大量
细粒度 kernel，没有单个主导 kernel。阶段二的首个可证伪假设因此是：共享 ABI 下融合
build/query/count/fill 热路径、减少 launch 与中间 tensor，比单独微调某个 Torch primitive
更可能达到收益门槛。

## 阶段二 HIP JIT 可行性

按两阶段计划先在 `probes/hip_cell_list_jit/` 实现隔离的 native HIP build/binning 子模块，
并由 `probes/hip_cell_list_jit_probe.py` 通过 PyTorch JIT extension 编译。该 kernel 在当前
PyTorch stream 上一次生成 atom periodic shifts、cell coordinates 和 linear cell keys；它不被
产品包导入，也未登记为 backend。

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MAX_JOBS=4 \
  PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python probes/hip_cell_list_jit_probe.py
```

DTK 26.04 的 `/opt/dtk-26.04/bin/hipcc` 成功编译，BW200 HCU 0 实际执行；257 原子、三斜胞、
mixed-PBC 的 float32/float64 输出与 Torch oracle 完全一致。预热后用 HIP event 采集每种规模
10 组原始 device-loop 样本：

| atoms | HIP median | Torch median | Torch / HIP |
|---:|---:|---:|---:|
| 368 | 0.02127 ms | 0.24479 ms | 11.51x |
| 32768 | 0.02139 ms | 0.24637 ms | 11.52x |

这些数字只说明“fractional coordinate + wrap + key”这个融合子模块在指定输入上的可行性和
launch 减少潜力；不包含 cell sort/CSR、query、capacity、pair fill、Python dispatcher 或
端到端成本，不能写成 cell-list 11x 加速。下一步只有在共享 ABI review 后才把该实验转为
AOT/custom-op 候选，并单独验证 stream、fake/meta、错误传播和 wheel 构建。

## 边界与下一阶段

本里程碑完成的是阶段一 reference core，并验证了阶段二第一个隔离 HIP JIT build/binning
子模块；它不是上游 `pbc-cell-list` 的全部功能，也不是 production 后端。尚缺 target rows、
pair callback/energy/force、pair-centric/sorted query、fake/meta/opcheck、compile、自动
profile/dispatch、分布式 ownership，以及正式 HIP/Triton package 实现。

阶段二先冻结共享 cell metadata、scratch/output、capacity/error 和 stream ABI，再分别评估 HIP
build/query 与适合规则张量变换的 Triton 子模块。每个候选必须先过本报告的数值 oracle，再用
相同输入和重复样本比较；未达到 neighbor 2x 或端到端 20% 的预定门槛时，不进入 `auto`。
