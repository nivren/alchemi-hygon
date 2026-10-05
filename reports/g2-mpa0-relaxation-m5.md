# M5：真实 MACE-MPA-0 HCU 单卡变胞弛豫验收

日期：2026-10-05。分支 `codex/test-mpa0-hcu-validation`，起点 `fa74ca1`。
用户已确认 M5；完成后先总结并提出下一阶段，等待确认后才提交 M5，不自动合并或推送。
后续收口记录：用户确认较大Batch测量后，M5已提交 `4656cce`，未合并/推送；
下面原始验收范围和数据保留。后续单Batch结果见[测量报告](g2-mpa0-single-batch-timing.md)。
NVIDIA framework `4dfe372` / ops `26dbceb` 和 M1–M4 的来源锁定保持不变。
M5 窄验收完成：两种精度、两种优化器的单体系/异构补位/扩大 Batch 和导出核验通过；
物理收敛及预算结果分别报告如下，不要求所有材料在固定预算内收敛。

## 改动与验收范围

复用 M4 的 `examples/relax_mace_mpa0.py`、原生 FIRE/FIRE2、FusedStage、
SizeAwareSampler、HostMemory、多条件收敛及 Torch reference 邻居/晶胞算子。
本阶段新增 `probes/mace_mpa0_hcu_validation.py` 和验收契约/证据文档；
**framework、ops 和应用实现没有改动，没有新增科学算法或上游 bug 修复**。
探针通过现有公开 AFTER_COMPUTE Hook 检查设备、精度、有限性、体系边界和补位，
仅在验收作用域内使用派生 FusedStage 注册观测 Hook，不新增产品 API。
M4 的 UBF-015/016 私有状态和 priming 修正保持原样；退出私有状态保护仍以
M4 的 CPU/合成 HCU 测试为直接证据，真实模型 Hook 观测不冒充另一套私有状态测试。

实际模型为可信本地 `mace-mpa-0-medium.model`，SHA256：
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
MACE0.3.15/e3nn0.4.4，ordinary eager，cueq/compile_model=False。
每个作业只用一种 dtype，模型权重、输入与进程默认 dtype 一致。
GPU Tensor 保持在指定设备；证据对照和 CIF I/O 显式转主机。

## 环境与共同门

先只读 hy-smi 确认 HCU4 空闲且无 KFD PID，限制 `HIP_VISIBLE_DEVICES=4`、
OMP_NUM_THREADS=1，DTK 激活入口 `/opt/dtk-26.04/env.zsh`。
项目 `.venv`：Torch2.9.0、HIP6.3.26093；BW1000 64G / gfx936:sramecc+:xnack-。
单点外层 timeout150秒，弛豫探针每次 timeout240秒；每体系500步，应用步间墙钟180秒。
先 FP64 单体系/三结构 Batch，随后 FP32 重复，再可控扩大 Batch。
固定 dt=0.02、skin=0、零外压，两优化器保留各自的原生控制参数。

成功须同时满足 fmax<=0.01 eV/Å 与 max(abs(stress))<=0.1 GPa。
预算退出独立记录，不能计为 converged。每个输入与输出都有 SHA256；source_index
与 admission-order system_id 分开，结果和 Hook 观测映射一致且每个来源/ID 恰好一次。
每个最终 CIF 另加载 CPU/HCU 模型、重建当前几何邻居，再比较完整 E/F/stress、
排序后的邻居/周期镜像及成功阈值；核验实际导出坐标系而非写出前张量。

CPU/HCU 对照门沿用 M1 的物理量容差，没有放宽：

| dtype | rtol | energy atol (eV) | force atol (eV/Å) | stress atol (eV/Å³) |
|---|---:|---:|---:|---:|
| FP64 | 2e-8 | 2e-7 | 2e-7 | 2e-9 |
| FP32 | 2e-4 | 2e-3 | 2e-4 | 2e-6 |

不要求 CPU/HCU 非线性弛豫轨迹或两种优化器的局部极小值逐步相同。
无 NVIDIA 数值 fixture，本轮没有 Warp/NVIDIA 数值对照。

## CPU 回归与同几何单点

`scripts/check_cpu_reference.sh` 退出0：149 ops + 1 import + 230 framework +
22 state + 24 dynamics-ops = **426 passed、3 skipped**，framework/ops 分进程。
三项 skip 是既有 Warp NHC 对照；没有新增 skip。
另运行 M4 生命周期/应用 focused suite，33 passed。
定向 Ruff、py_compile 和 git diff --check 通过。

三个 M1 输入 NaLiTl2(4)、Li2MgGe(4)、LiSnCl3(10) 的同几何异构 Batch：

| dtype | 最大能量差 eV | 最大力分量差 eV/Å | 最大应力分量差 eV/Å³ | HCU warm 单点评估秒 |
|---|---:|---:|---:|---:|
| FP64 | 3.55272e-15 | 5.70378e-15 | 1.96024e-16 | 0.065911 |
| FP32 | 1.90735e-6 | 7.51019e-6 | 1.38768e-7 | 0.067000 |

邻居/镜像集合相同、无跨体系边。warm 测量为一次预热后同步三次均值，包含邻居重建、
模型、设备检查/对照数据转主机；不等于裸 kernel 或生产吞吐。CPU 同口径均值
分别1.344196/0.612587秒，模型加载计时和启动计时单独记录；不是跨平台性能保证。

## 单体系与三结构异构 Batch

NaLiTl2 单体系两精度/两优化器四次真收敛：FIRE31步，FIRE2 53步。
三结构最大活跃2、原子容量20；实际首批 [NaLiTl2(4), LiSnCl3(10)]，
NaLiTl2 退出后补入 Li2MgGe(4)。system_id 为原输入0→0、1→2、2→1，
没有用排序后的 ID 冒充原始来源索引。

以下为应用重读导出 CIF 后 HCU 重算并记入 JSONL 的结果；独立 CPU/HCU 核验另通过：

| dtype | 优化器 | 结构 | 步数 | 状态 | fmax eV/Å | max(abs(stress)) GPa |
|---|---|---|---:|---|---:|---:|
| FP64 | FIRE | NaLiTl2 | 31 | converged | 0.00287861 | 0.0412495 |
| FP64 | FIRE | Li2MgGe | 35 | converged | 0.00149852 | 0.0928081 |
| FP64 | FIRE | LiSnCl3 | 80 | converged | 0.00817191 | 0.0291459 |
| FP64 | FIRE2 | NaLiTl2 | 53 | converged | 0.00902073 | 0.0570618 |
| FP64 | FIRE2 | Li2MgGe | 77 | converged | 0.000290116 | 0.0934281 |
| FP64 | FIRE2 | LiSnCl3 | 153 | converged | 0.00113025 | 0.0964069 |
| FP32 | FIRE | NaLiTl2 | 31 | converged | 0.00286666 | 0.0412430 |
| FP32 | FIRE | Li2MgGe | 35 | converged | 0.00149961 | 0.0927844 |
| FP32 | FIRE | LiSnCl3 | 80 | converged | 0.00817580 | 0.0291664 |
| FP32 | FIRE2 | NaLiTl2 | 53 | converged | 0.00903210 | 0.0570657 |
| FP32 | FIRE2 | Li2MgGe | 77 | converged | 0.000289382 | 0.0934240 |
| FP32 | FIRE2 | LiSnCl3 | 153 | converged | 0.00113179 | 0.0963988 |

四个 Batch 作业均通过真实补位、来源/ID 唯一性、双阈值和导出后独立核验；
FIRE31/35/80步、FIRE2 53/77/153步，精度间本组步数相同，不构成逐轨迹一致要求。
FIRE2 在 M4 CPU 120秒预算内退出的两项，本次 HCU 小样本达到双阈值。
三体系应用加观测 Hook（含模型加载/邻居/优化/最终导出）约20–24秒；
额外独立 CPU/HCU 核验另计。

## 扩大 Batch 的真实样本

从用户指定的 `/data/csp_data/catl_data/li_mp_data` 选择8个不同 CIF，
总59原子，每个4–16原子；固定最大活跃4体系、
max_atoms=64。按文件名排序并复制到原始证据目录，哈希保存在每个 validation.json。
这是小结构数量和有限原子规模的扩展，不是大晶胞或全目录测试。

| 原始文件 stem | 原子数 |
|---|---:|
| mp-1001079_Li(CN)2 | 10 |
| mp-1001581_LiC6 | 7 |
| mp-1001615_LiZrSe2 | 4 |
| mp-1002069_NaLiICl | 4 |
| mp-1002572_LiMnO2 | 16 |
| mp-1096278_NaLiTl2 | 4 |
| mp-1222651_Li2MgGe | 4 |
| mp-998591_LiSnCl3 | 10 |

实际首批原子数 [4,7,10,16]；后续退出后继续补位，体系数/原子数不超显式容量。
四组均通过设备/dtype/有限性、容量、真实补位、来源/ID、输入/输出哈希和
最终 CPU/HCU E/F/stress/邻居核验。单点两组加弛豫十二组的探针退出码均为0。

| dtype | 优化器 | converged / max_steps | 应用及导出秒 | startup秒 | 探针总秒 | 应用峰值 allocated / reserved MiB |
|---|---|---|---:|---:|---:|---:|
| FP64 | FIRE | 8 / 0 | 25.914 | 19.516 | 51.025 | 626.244 / 874 |
| FP64 | FIRE2 | 7 / 1 | 41.323 | 19.662 | 66.683 | 626.240 / 874 |
| FP32 | FIRE | 8 / 0 | 35.076 | 19.673 | 57.798 | 347.035 / 434 |
| FP32 | FIRE2 | 7 / 1 | 42.454 | 19.638 | 65.223 | 347.035 / 432 |

扩大 Batch 中逐输入步数（输入顺序同上）：

| 输入 | FIRE FP64/FP32 | FIRE2 FP64/FP32 |
|---|---:|---:|
| Li(CN)2 | 88 / 364 | 500 / 500（max_steps） |
| LiC6 | 32 / 32 | 84 / 84 |
| LiZrSe2 | 49 / 49 | 126 / 126 |
| NaLiICl | 20 / 20 | 72 / 72 |
| LiMnO2 | 35 / 35 | 90 / 90 |
| NaLiTl2 | 31 / 31 | 53 / 53 |
| Li2MgGe | 35 / 35 | 77 / 77 |
| LiSnCl3 | 80 / 80 | 153 / 153 |

两份 FIRE2 Li(CN)2 均在500步用完后导出：FP64 fmax0.00707973 eV/Å、
应力0.281632 GPa；FP32 fmax0.00947517 eV/Å、应力0.264614 GPa。
力合格而应力不合格，状态均为 max_steps，final_thresholds_passed=false。
这是预算内未收敛的实测结果，没有据此归因为上游 bug，没有扩大预算或修改阈值。

**精度敏感性**：Li(CN)2 的 FIRE FP64/FP32 都满足双阈值，但结果不同：
FP64 88步、E=-72.6568610 eV、V=133.015271 Å³；FP32 364步、
E=-74.5764160 eV、V=124.967972 Å³。最终几何显著不同，本轮未判定极小值性质
或原因，也未验收重复运行的轨迹稳定性。不能据同几何单点 parity 宣称两个精度
的非线性弛豫可互换；使用 FP32 时需按应用要求验证结果稳定性。

12次弛豫共48份输出（8种不同真实输入，多配置重复）：46份 converged、2份 max_steps。
全部48份各由独立 CPU/HCU 模型重算一次，共96次最终单点评估，成功者在两个设备
均满足双阈值；两个预算结果没有被改为成功。最终同几何 CPU/HCU 全字段最大差：

| dtype | energy eV | force eV/Å | stress eV/Å³ |
|---|---:|---:|---:|
| FP64 | 2.84218e-14 | 2.95459e-14 | 3.21315e-15 |
| FP32 | 1.52588e-5 | 1.92673e-5 | 1.35409e-6 |

这些是当前小样本的数值一致性证据，未测全目录或其他晶胞/原子规模。

计时区分：startup 为探针从第三方库导入到设备初始化的墙钟（不含 Python 启动前
极短时间），relaxation 为输入清单测量 + 模型加载/工作流/观测/应用导出核验；
elapsed 另含独立 CPU/HCU 最终核验，probe_total=startup+elapsed。
峰值 allocated/reserved 分别记录应用阶段及包含独立验证的全过程，
Torch allocator 指标不是 hy-smi 整卡占用；验证 Hook 有同步/主机观测开销。
未测冷 JIT 与 warm 优化循环分项，不承诺端到端生产吞吐。

## 原始证据和边界

原始目录 `artifacts/m5-mpa0-hcu/`：resource-before/pids-before、cpu-gate.log、
cpu-workflow.log；parity-{fp64,fp32}/validation.json；
single-{fire,fire2}-{fp64,fp32}/、batch-{fire,fire2}-{fp64,fp32}/ 含 CIF、results.jsonl
及 validation.json。JSONL 保存实际参数、退出原因、步数、各物理量和模型/输入/输出哈希。
expanded-{fire,fire2}-{float64,float32}/ 为8结构四组输出，expanded-inputs/ 是原样复制的
输入；summary.json 为聚合核验。resource-after/pids-after 确认作业结束后卡空闲、无 KFD PID。
source-manifest.json 将8份测试副本映射回原始目录，重新检查原文件/副本SHA256全部一致，
复用的三个 M4 输入副本也与原始文件一致；没有人为扰动或合成上述真实 CIF。
探针验证通过只表示所有断言及预算分类正确，是否物理收敛仍看逐体系 status。

验收局限：单卡 eager Torch reference；无完整 HCU gate、NVIDIA 数值 fixture、
生产 HIP/Triton、训练/compile、DomainParallel、NPT/NPH 或 ASE filters。
普通单优化器 FusedStage/scalar refill；不支持向量 arrivals 来源参数映射，未验收
任意混合变胞算法阶段。没有通用 checkpoint/restart 或输入流式服务。
应用预读输入/HostMemory 成本随输入数增长；墙钟在步间检查，导出/核验可超预算。
小样本不能推断全目录成功率、物理预测准确度或生产吞吐。

## 使用与下一阶段建议（待用户确认）

复现本次应用的示例（目录应为上述8结构子集，output-dir 必须不存在）：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 240 .venv/bin/python examples/relax_mace_mpa0.py \
  --input-dir artifacts/m5-mpa0-hcu/expanded-inputs \
  --output-dir artifacts/my-mpa0-hcu-run \
  --checkpoint "$MACE_MPA0_CHECKPOINT" \
  --optimizer fire --device cuda:0 --dtype float64 \
  --max-batch-size 4 --max-atoms 64 --max-steps 500 --max-wall-seconds 180
```

checkpoint 环境变量指向可信本地锁定模型；该命令为现有应用的使用方式，
本轮实际探针调用其同一个 relax() 函数，并加验收 Hook，耗时不能直接等同此命令。
输入 `--limit` 只截取目录排序前若干 CIF，不代表本文三个/八个固定样本。
验收复现使用 `probes/mace_mpa0_hcu_validation.py --mode relax --inputs <明确CIF列表>`，
其余 model/device/dtype/budget 参数同上，结果包含 validation.json。

原定 M1–M5 的窄移植/验证目标已完成。下一阶段建议为应用试运行：先提交 M5，
再按原子数/化学组成选择20个真实结构、分为受限小作业，优先 FP64，检查运行预算、
结果状态和导出；另对 Li(CN)2 做重复与原生预算/参数对照，评估精度敏感性。
不新增科学算法/filters，不自动运行全目录；细化方案经用户确认后开始。

上述20结构方案为M5结束时的建议，后被用户修改为只测一个较大FP64/FIRE Batch、
估算1600项耗时；并允许改用原生晶胞力停止口径。该后续测量已完成，见上方链接，
不与本报告双阈值数据合并。
