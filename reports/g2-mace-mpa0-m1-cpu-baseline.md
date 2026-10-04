# MPA-0 弛豫里程碑1：基础收口与 CPU 单点基线

日期：2026-10-04。分支 `codex/test-mace-mpa0-baseline`，起点 `35ea133`。
CPU 探针提交 `61484db`，契约与报告独立提交。
本里程碑由用户确认后执行；只完成基础收口、来源/契约和 MPA-0 CPU 单点数值验证。
未运行 FIRE/FIRE2、未推进晶胞、未验证 MPA-0 HCU 或全目录弛豫。

## 实现与 Git 边界

- `codex/feat-cell-stress-force` 已独立收口：实现/测试 `a76107e`、文档 `35ea133`。
  新 MPA-0 探针和契约未混入这两个提交；两分支均未合并到 develop、未推送。
- 本里程碑新增 [CPU 探针](../probes/mace_mpa0_cpu_baseline.py) 和
  [弛豫契约](../docs/TORCH_MACE_MPA0_RELAXATION_CONTRACT.md)。复用现有 MACEWrapper、
  AtomicData/Batch、compute_neighbors 和 stress_to_cell_force，没有修改产品运行时源码。
- 上游 NVIDIA 两包 SHA 保持原锁定。fork 参考头为 framework `b12b54c48fc8`、
  ops `eadc3eeebaec`；只锁定相关修复来源，不整体引入更新的训练、数据或依赖代码。
- 契约分别规定 NVIDIA 原生下三角六项 FIRE2，以及指定 fork 修复后的全3×3 FIRE。
  M1 不因源码审查将这些路径登记为已实现；M2 起各里程碑须先征得用户确认。

## 模型、环境与输入

可信本地 checkpoint：`mace-mpa-0-medium.model`，SHA256：
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
ScaleShiftMACE，cutoff=6 Å，89元素（包括 Li/Na），head=`default`，含 pair repulsion；
交互块为 RealAgnosticDensityInteractionBlock 与 RealAgnosticDensityResidualInteractionBlock。
保留模型原始 E0、势函数与元素表，无自动下载。

环境：project .venv，PyTorch `2.9.0+das.opt1.dtk2604`，mace-torch `0.3.15`，
e3nn `0.4.4`，ASE `3.29.0`，NumPy `1.26.4`；CPU，1线程，ordinary e3nn eager，
`enable_cueq=False`、`compile_model=False`。显式激活 energy/forces/stress。
ASE calculator 只作单点 oracle，没有使用 ASE 优化器或 filters。

来源为用户指定 CIF 目录中的三个完整占位、三维周期非正交结构，输入原文件未改动：

| CIF | 原子数 | 内容 SHA256 |
|---|---|---|
| mp-1222651_Li2MgGe.cif | 4 | `0042a5d93e4d1d5721710dfd22282b14f7c4eb87760d4f0f88208a9f758146ad` |
| mp-998591_LiSnCl3.cif | 10 | `dd3f50b0fc5ac27db845944ba785d21cfb4af1ae6c3c460db9b7953ac043d2d6` |
| mp-1096278_NaLiTl2.cif | 4 | `96ff2c9e77a5374468409978164cb1578e380ddc24f08e6f2a80997619a86481` |

AtomicData.from_atoms 显式传 dtype，不先截断为 FP32 再提升。
两种精度均为 `batch_ptr=[0,4,14,18]`、610条 full-list 周期边、870个非零 image-shift
分量、0条跨体系边。每次差分重新生成邻居并验证 canonical atom/image 集合不变。

## CPU 结果

探针退出0、`status=passed`，含模型加载/ASE/数值导数的总墙钟37.447秒。
这是带多次前向的功能探针，不是弛豫时间、生产吞吐或冷暖分离 benchmark。

| 对照 | 能量最大绝对误差，eV | 力最大绝对误差，eV/Å | 应力最大绝对误差，eV/Å³ |
|---|---|---|---|
| FP64 Batch 对逐体系 | `1.77636e-15` | `5.13478e-15` | `1.00614e-16` |
| FP64 Batch 对 ASE calculator | `1.77636e-15` | `9.38138e-15` | `8.67362e-17` |
| FP32 Batch 对逐体系 | `4.76837e-7` | `3.15905e-6` | `5.49480e-8` |

FP64 中心差分步长 `1e-5`，force/stress/cell-force 的最大绝对误差：

- 各体系最高力原子的 XYZ，共9个原子偏导：`3.67412e-10 eV/Å`。
- 全原子确定性方向导数：`9.22178e-10 eV/Å`；不是逐个检查全部54个原子偏导。
- 每体系6个独立对称应变：`1.56310e-10 eV/Å³`。
- 固定分数坐标 cell `[1,0]`、`[2,1]`：分别 `6.54844e-10`、
  `6.00595e-10 eV/Å`；实际调用完整矩阵 reference 转换。

容差在契约和 probe 中逐物理量明确：FP64 Batch/ASE `rtol=2e-8`，
energy/force/stress 的 atol 分别 `2e-7 eV`、`2e-7 eV/Å`、`2e-9 eV/Å³`；
差分 `rtol=3e-4`，force/cell-force atol=`3e-6 eV/Å`，stress atol=`3e-7 eV/Å³`。
FP32 只验证前向、dtype/device、Batch 对照，不用临近 FP32 总能量相减作差分。

原始输入的 FP64 单点指标（不是弛豫后的结果）：

| 结构 | E，eV/structure | fmax，eV/Å | 最大绝对应力分量，GPa |
|---|---|---|---|
| Li2MgGe | -11.3909309341 | `2.24542e-15` | 0.5196373 |
| LiSnCl3（10原子） | -36.0386734640 | 0.2240747 | 1.3272173 |
| NaLiTl2 | -4.3005904121 | 0.0837534 | 0.0696459 |

Li2MgGe 的 force 很小而 stress 未达建议0.1 GPa阈值，因此单靠 fmax 会误判变胞收敛。
这些指标不说明势模型与 DFT/实验一致，也不说明后续两种优化器会得到同一极小值。

## 默认 dtype 依赖与初次失败

初次 probe 的 FP64 checks 通过，后续 FP32 的 energy dtype 检查失败。
独立诊断确认：ASE calculator 设置进程默认 dtype；MACE 0.3.15 的 ZBL 原子序数整数幂
依赖该默认值。FP32权重/输入在默认FP64进程中，raw energy 为FP64、F/stress 为FP32；
默认FP32时 energy/F/stress 均为FP32。wrapper忠实转交raw输出，不是邻居或应力公式问题。

修正只在 probe 中显式配置每个 case 的默认 dtype，并在入口 finally 恢复；没有事后 cast
输出、改依赖源码、改共享wrapper、关闭pair repulsion或放宽容差。最终整项probe重跑通过。
单一精度应用须把默认dtype、权重和输入一致配置；同进程混合精度并发模型未验收。
诊断还观察到 raw `node_energy` 在FP32情况下可为FP64，本任务未验收 atomic_energies 输出。

原始失败保留在 `cpu-model-first.log`，诊断在 `dtype-diagnostic.log`，不把第一次运行记为通过。
e3nn 的 weights_only 环境提示、cuEq 不可用及 hylog 日志目录提示出现在 stdout；
没有执行 HCU 设备任务，不据此判断 HCU 运行状态。

## 基础回归与复现

本轮基础 CPU gate 退出0：分进程 `82 passed, 3 skipped`、`1 passed`、
`168 passed, 1 deselected`、`22 passed`、`24 passed`，共297项通过。
三项跳过因 Warp 缺失；应力管线另为 `12 passed, 1 skipped, 66 deselected`，
跳过项需CUDA。NPT/NPH 24个非stress函数去除import后AST不变。
compileall、Git diff检查通过；新probe使用缓存Ruff0.11.13 check/format及py_compile通过。

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mace-mpa0-mpl timeout 240 \
  .venv/bin/python -u probes/mace_mpa0_cpu_baseline.py \
  --checkpoint /path/to/mace-mpa-0-medium.model \
  --data-root /path/to/li_mp_data \
  --output artifacts/mace-mpa0-m1/cpu.json
```

原始证据目录：`artifacts/mace-mpa0-m1/`，包括 cpu.json、cpu-model.log、cpu-gate.log、
stress-pipeline.log、cpu-model-first.log、dtype-diagnostic.log；模型和数据不提交入库。

## 验收边界与下一步

M1 已完成 CPU-only 基线；原 stress slice 的 OFF23/HCU4 证据保留原报告，不移用为MPA-0。
MPA-0 HCU、完整目录、变胞邻居生命周期、FIRE/FIRE2收敛、退出/补位、训练/二阶、
compile/cuEq、NVIDIA/Warp及生产性能均未验收。
下一步 M2 是通用晶胞工具 Torch reference，须先向用户说明目的与方案并获得确认。
