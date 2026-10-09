# FIRE2 变胞 + HIP 邻居：三样本完整弛豫补充验收

日期：2026-10-09。用户确认的集成收口前补测。结论：**本轮三样本完整联合收敛与真实补位通过**。
四个 HCU4 作业均退出 0；两精度、两种邻居后端共 12 个结构运行全部 converged，
其中 HIP 邻居路径 6 个结构运行。没有步数/墙钟退出被计为成功。

## 1. 源码、模型与边界

- 分支：`wangleping/test-hip-fire2-convergence`；独立 worktree：
  `artifacts/hip-fire2-convergence-2026-10-09/worktree`（相对主项目根目录）。
- 产品起点：`ad189987b94b8b97b3332ee8982c960a3ce6a5e6`；
  packages 与稳定 develop `2e211ab93a4f1ad0c8c720eecb22129d846c7792` 相同。
- 上游锁不变：Framework `4dfe3723def34df3fadb245981081ccf8c94c257`；
  Ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。
- [探针](../probes/hip_fire2_convergence.py) SHA256：
  `d9eb2b33f21d99ac82010e952aa88d2548a87202799cf6ffeadaaf1647465bfa`。
- checkpoint SHA256：`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
  MACE-MPA-0 medium，mace-torch0.3.15/e3nn0.4.4，ordinary eager，cueq/compile_model=False。

复用 FIRE2VariableCell、AlignCellHook、FusedStage、SizeAwareSampler、HostMemory 和
ConvergenceHook；新增探针只观测、记录和施加运行预算，没有添加科学算法或修改产品实现。
优化器/晶胞对齐为 Torch reference；邻居分别显式 hip 和 torch_reference。
HIP 为 full-PBC/full-list、native MATRIX + 设备内 Torch COO 打包、skin=0、每次当前晶胞重建。
模型/输入/默认 dtype 同精度，冻结模型参数但保留位置/应变 autograd。
完整验收规则见 [补充契约](../docs/HIP_FIRE2_CONVERGENCE_VALIDATION.md)。

## 2. 设备、输入和运行门槛

Python3.12.13、Torch distribution2.9.0+das.opt1.dtk2604（torch.__version__2.9.0）、
HIP6.3.26093，DTK26.04 env.zsh；HCU4 BW1000 64G，gfx936:sramecc+:xnack-。
HIP_VISIBLE_DEVICES=4 映射 cuda:0，OMP_NUM_THREADS=1、MAX_JOBS=1；
复用原 HIP 独立 native 缓存，进程 PYTHONPATH 明确指向本验证 worktree 的两包与 probes。
每个作业前后只读 hy-smi 均显示 HCU4 利用率0、显存占用比例0、可用65390 MiB，
对应日志无 KFD PIDs；没有更换卡、升级环境或操作他人任务。

| source_index / system_id | 输入（均为原 M1/M2/A 夹具） | 原子数 | SHA256 |
|---|---|---:|---|
| 0 / 0 | mp-1222651_Li2MgGe.cif | 4 | `0042a5d93e4d1d5721710dfd22282b14f7c4eb87760d4f0f88208a9f758146ad` |
| 1 / 1 | mp-998591_LiSnCl3.cif | 10 | `dd3f50b0fc5ac27db845944ba785d21cfb4af1ae6c3c460db9b7953ac043d2d6` |
| 2 / 2 | mp-1096278_NaLiTl2.cif | 4 | `96ff2c9e77a5374468409978164cb1578e380ddc24f08e6f2a80997619a86481` |

固定 dt=0.02、零外压、FIRE2 默认控制参数；每体系最多500次实际更新，步间预算180秒，
每个进程外层 timeout300秒。物理成功沿用 M5：
**fmax <= 0.01 eV/Å AND max(abs(stress)) <= 0.1 GPa**。

## 3. CPU 与 HCU 结果

| gate / 作业 | 结果 | 退出码 | 进程墙钟秒 |
|---|---|---:|---:|
| probe 判据/退出分类 CPU 夹具 | 4 passed | 0 | 未单独计时 |
| CPU 变胞单步/生命周期/HIP组合 reference 用例 | 59 passed、16 skipped | 0 | 27.31 |
| FP64 + Torch reference 邻居 | 3/3 converged、补位和终态核验通过 | 0 | 148.06 |
| FP64 + HIP 邻居 | 3/3 converged、补位和终态核验通过 | 0 | 54.57 |
| FP32 + Torch reference 邻居 | 3/3 converged、补位和终态核验通过 | 0 | 52.86 |
| FP32 + HIP 邻居 | 3/3 converged、补位和终态核验通过 | 0 | 54.88 |

CPU 的16项 skip 是隐藏 GPU 后既有 HIP 设备参数项；没有增加 skip/xfail。
CPU 三文件为 `test_variable_cell_fire_reference.py`、`test_variable_cell_workflow.py`、
`test_hip_variable_cell_workflow.py`；新增 [4项夹具](../probes/test_hip_fire2_convergence.py)
验证力和压缩/拉伸绝对应力联合条件、物理成功与步数/墙钟状态分类。
该门是本任务定向回归，本轮没有重跑 A 的425项指定 CPU gate或全库 HCU测试。

墙钟含 import、冷启动、模型及导出后独立核验；缓存预热没有按性能实验控制。
不根据上述进程耗时计算邻居加速比。原始 probe 的分段时间另外保留。

| HIP 邻居精度 | 结构 | 实际更新 | 终态 fmax eV/Å | max(abs(stress)) GPa | 状态 |
|---|---|---:|---:|---:|---|
| FP64 | Li2MgGe | 77 | 0.000290116 | 0.0934281 | converged |
| FP64 | LiSnCl3 | 153 | 0.00113025 | 0.0964069 | converged |
| FP64 | NaLiTl2 | 53 | 0.00902073 | 0.0570618 | converged |
| FP32 | Li2MgGe | 77 | 0.000289952 | 0.0934308 | converged |
| FP32 | LiSnCl3 | 153 | 0.00113339 | 0.0963834 | converged |
| FP32 | NaLiTl2 | 53 | 0.00901549 | 0.0570594 | converged |

Torch reference 邻居四组相应更新次数也为77/153/53，均满足双阈值；
两条长非线性轨迹独立运行，没有人为强制逐步/功率分支完全相同。

## 4. 补位、终态与数值审计

容量2结构/14原子，实际首批来源[0,1]、ptr[0,4,14]。Li2MgGe 完成后补入来源2，
新 Batch 来源[1,2]、ptr[0,10,14]；后续只保留来源1、ptr[0,10]。
四组都记录来源2在 compute call78 到达，首个 kick 前完整 E/F/stress 与 reference
重算一致；source_index 和 system_id 各恰好覆盖0/1/2一次，终态 status 均为1。
每次实际 shared compute 核对当前几何的邻居 atom/image 多重集合及无跨体系边、
dtype/device/有限性/正体积。三个终态晶胞均实际变化。

所有终态重新在**同一几何**构建 HIP/reference 邻居并计算完整 E/F/stress；
缓存输出也与新求值对照。保存12份 lossless 张量及12份 CIF，并重读实际 CIF，
另用 CPU/HCU reference 模型重建邻居、重算完整物理量与联合阈值。
输入/输出哈希、CIF 的周期坐标与下三角晶胞 roundtrip、来源及 ID 一致性核验通过。

下表为每种 dtype 的两个邻居分支、三个终态汇总最大绝对差：

| 同几何比较 | dtype | energy eV | force eV/Å | stress eV/Å³ |
|---|---|---:|---:|---:|
| HIP/reference 邻居终态重算 | FP64 | 1.42e-14 | 7.14e-15 | 6.32e-17 |
| HIP/reference 邻居终态重算 | FP32 | 3.81e-6 | 3.77e-6 | 5.61e-8 |
| 重读 CIF 的 CPU/HCU reference | FP64 | 1.42e-14 | 1.65e-14 | 2.50e-16 |
| 重读 CIF 的 CPU/HCU reference | FP32 | 3.81e-6 | 3.24e-6 | 1.10e-7 |

沿用既有逐物理量门：FP64 rtol2e-8、E/F/stress atol2e-7/2e-7/2e-9；
FP32 rtol2e-4、atol2e-3/2e-4/2e-6。没有更改任何容差或给 FP32 临时 cast。
独立长轨迹之间的终态最大位置/晶胞差为 FP64 6.22e-15/8.88e-16 Å、
FP32 1.91e-6/9.54e-7 Å；完整 E/F/stress 差另外存于汇总，不混作同几何误差。
真实补位观测不额外宣称私有状态逐元素测试；该保护直接证据继续来自既有生命周期回归。

## 5. 原始证据与复现

原始目录：主项目根目录下 `artifacts/hip-fire2-convergence-2026-10-09/`，全部忽略不入库。

- `commands-and-exits.json`：完整 argv/PYTHONPATH/native环境、起止时间、超时、退出码。
- `cpu-probe-fixtures.log`、`cpu-workflow.log`：CPU完整结果。
- `float{32,64}-{hip,torch_reference}/summary.json`：设备/环境/输入哈希、每次compute布局、
  admission与priming、各结构实际更新/退出/终态数值、导出核验和分段时间。
- 同目录 `*-final.pt`、`*-final.cif`：lossless终态与实际导出。
- 四份作业 `.log` 和 `*-before/after-{resources,pids}.log`：完整日志与资源检查。
- `validation-summary.json`、`probe-source-check.json`：独立汇总、源码哈希与不变性核对。

从主项目根目录激活环境，显式将 PYTHONPATH 指向验证 worktree，两包相同的产品源码。
加载原 `artifacts/hip-current-cell-neighbor-m1/native_env.sh` 后，复现一组：

```bash
source scripts/activate_hygon_env.sh project
source artifacts/hip-current-cell-neighbor-m1/native_env.sh
PYTHONPATH=artifacts/hip-fire2-convergence-2026-10-09/worktree/packages/framework:artifacts/hip-fire2-convergence-2026-10-09/worktree/packages/ops:artifacts/hip-fire2-convergence-2026-10-09/worktree/probes \
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout -k 10 300 .venv/bin/python -u \
  artifacts/hip-fire2-convergence-2026-10-09/worktree/probes/hip_fire2_convergence.py \
  --checkpoint /home/wangleping/mace-mpa-0-medium.model \
  --data-root /data/csp_data/catl_data/li_mp_data \
  --output-dir artifacts/hip-fire2-convergence-2026-10-09/reproduction-fp64-hip \
  --backend hip --dtype float64 --max-steps 500 --wall-seconds 180 \
  --product-commit ad189987b94b8b97b3332ee8982c960a3ce6a5e6
```

输出目录必须为新目录，不能覆盖原证据；先只读确认 HCU4 空闲。
912份原基线源码/夹具、3份环境控制文件和两套缓存引用核对不变；
packages/版本锁与稳定 develop 相同，主 worktree 保持干净，git diff --check通过。

## 6. 完成范围与下一步

新增证据仅为固定三样本的 FIRE2 完整联合收敛与真实补位；已有大Batch有限步/预算测试
继续保留原范围。不推广为任意材料收敛率、性能、完整 HCU/Warp/NVIDIA、compile、
skin缓存、restart、训练力损失、NPT/NPH 或 DomainParallel支持。
没有运行新上游候选源码，不改变 D 暂缓整体升级的决策。

用户已确认本阶段以本签名提交收口，并授权将 A–D 工程化成果和本报告整理为可审阅的
独立集成 PR。尚未合并或推送 develop；外部交付和集成按后续明确确认执行。
