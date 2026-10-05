# MPA-0 + 原生 FIRE：单个 FP64 Batch 耗时测量

日期：2026-10-05。分支 `codex/test-mpa0-batch-timing`，起点为 M5 收口提交
`4656cce`；用户随后确认8000原子门槛与分项测量，本次已收口提交 `7410445`，未合并或推送。
用户授权只跑一个完整 Batch，不跑全部1600结构；后续明确允许使用 FIRE 自带晶胞工具。

## 范围与实现

新增隔离探针 `probes/mace_mpa0_batch_timing.py`，复用现有 CIFDataset、StepCounter、
WallDeadline、FIREVariableCell、FusedStage、SizeAwareSampler、HostMemory、
ConvergenceHook、NeighborListHook 和 stress_to_cell_force。
没有修改 framework、ops 或产品应用入口，没有新增 ASE filter、优化算法或性能后端。
探针观测器计算原生停止指标并记录时间；不是增加一个新的晶胞参数化方案。

输入来自 `/data/csp_data/catl_data/li_mp_data`：21755份候选 CIF 按文件名排序，
`random.Random(24).sample(files, 1600)` 无放回抽样并保留返回顺序，只执行前64份。
输入未预弛豫、标准化或人为扰动；直接读取原文件。样本共2287原子，每结构3–132原子。
本地候选集合与用户 H100 测试的约34570候选不同，同seed不代表同一结构集合。

模型为 MACE MPA-0-medium，SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
FP64权重/输入/进程默认dtype一致；eval且冻结权重，位置/应变梯度仍用于力和应力。
MACE0.3.15/e3nn0.4.4，ordinary e3nn eager，CuEq和compile_model关闭。

HCU4：BW1000 64G/gfx936、Torch2.9.0/HIP6.3.26093、DTK26.04；
单进程、单worker、OMP_NUM_THREADS=1，无MPS。作业前后只读资源检查；结束后无KFD PID。
为保护共享卡，设置本进程 Torch allocator 上限为总显存80%，没有dtype/backend回退。

## 原生算法与停止口径

优化原子位置与全部9个晶胞分量，零外压、真实原子质量、晶胞质量1；
保留原生Verlet半步速度、位置/晶胞更新和联合FIRE混合顺序。
参数：dt=.1、dt_max=1、dt_min=.002、maxstep=.2、n_min=5、f_inc=1.1、
f_dec=.5、alpha_start=.1、f_alpha=.99；maxstep作用于原子实际位移和晶胞行位移。
每次求值前重建6Å邻居，COO动态容量；逐原子邻居数超过512则报错，未截断邻居。

停止指标复用 FIRE 的完整晶胞力：`F_H=-V*inv(H).T@stress`，
和原子物理力一起取所有行二范数的最大值。<=.05 eV/Å收敛，>100 eV/Å力溢出退出；
没有独立应力阈值。每结构最多500次实际更新，补位/重算不计作优化更新。
最大活跃64结构、max_atoms=2304、refill_frequency=20；全部64输入在初批放入，
以后只有退出/清空，没有新到达结构，不能据此声称测得持续动态补位收益。

该原生raw力判据与用户H100记录的参考初态/cell_factor=N的UnitCellFilter广义力判据
不同，也与M4/M5产品示例的力/应力双阈值不同。三者的收敛统计和耗时不能直接合并。
本探针仅调用现有Hook与晶胞工具，不实现UnitCellFilter或改变优化器更新方程。

初轮设置dt_min=0，在第一个完整优化步完成前被本地
`packages/ops/nvalchemiops/cell_fire_reference.py` 的 `dt_min<=0` 参数校验拒绝。
该失败记录保留，未用于外推；最终明确采用现有默认下限.002。
这属于待核对/修正的本地移植兼容性差异，本阶段没有修改算子，也不记为已修复的上游bug。

## 实测结果

| 项目 | 结果 |
|---|---:|
| 处理结构 / 原子数 | 64 / 2287 |
| converged / max_steps / force_overflow / calculation_error | 62 / 2 / 0 / 0 |
| 每结构更新步数中位数 / P95 / 最大 | 37.5 / 309.75 / 500 |
| 实际更新总数 / 原子×更新总数 | 5191 / 239900 |
| 完整fused步 / 模型求值调用 | 500 / 514 |
| 导入等固定启动墙钟 | 19.809秒 |
| CPU/HCU NumPy oracle及设备初始化 | 3.564秒 |
| 读取输入、模型与工作流准备 | 2.557秒 |
| 邻居、模型、优化、退出与清空 | 98.272秒 |
| CPU导出、CIF/原始数组/JSONL写出 | 0.218秒 |
| 应用端到端（准备+优化+导出） | **101.046秒** |
| 整个探针（启动+oracle+应用） | **124.420秒** |
| Torch峰值 allocated / reserved | **23.486 / 35.096 GiB** |

计时是墙钟，优化末尾和导出末尾均同步设备；不包含进程创建前时间、Shell启动或日志工具。
514次模型调用包含priming/reprime，不能作为单结构更新数。Torch显存指标不等于整卡
占用或全进程非Torch开销；没有OOM只证明当前2287原子样本可运行，未尝试16000原子。
本阶段不优化性能，不将冻结退出后Batch收缩的分段耗时当成满载每步吞吐。

两个预算退出结构：

| 输入stem | 原子数 | 实际步数 | 最终原生fmax eV/Å |
|---|---:|---:|---:|
| mp-1228802_Ba2Li3Y3(WO4)8 | 48 | 500 | 0.06179035 |
| mp-1213573_LiCdMoO5 | 32 | 500 | 0.18071504 |

二者的最大广义力均由原子力主导；未扩大步数/阈值，也未归因为上游bug。
“Batch完成”表示所有输入均正确分类导出，不表示64/64物理收敛。

## CPU与HCU验证边界

CPU/HCU各3个triclinic、异构原子数的NumPy独立公式案例通过，最大误差均0。
完成后在CPU上核验全部64项：来源/体系ID各恰好一次、抽样顺序、输入/输出SHA256、
FP64原始E/F/stress/r/H有限性、正体积、步数预算、收敛分类；原生力独立NumPy重算
最大差3.46945e-17 eV/Å。CIF重读的原子数/元素/PBC、晶胞Gram矩阵及周期分数坐标
与保存原始状态一致。原生行力判定使用lossless原始朝向，不以CIF规范化后的朝向代替。

本次没有对64份输出另加载模型重算E/F/stress，也没有运行完整HCU回归。
模型数值与原工作流CPU/HCU证据仍见M5（426 passed/3既有Warp skips和定向真实HCU验证），
不能把本轮NumPy/文件核验称为64结构的独立模型parity或M5双阈值验收。
本轮定向Ruff/格式、py_compile和git diff --check用于探针与文档收口。

## 1600结构的条件估算

只测得这一批。假设后续结构的原子数、邻居密度和优化步数分布接近此批，
沿用当前HCU单卡、FP64/e3nn、约2304原子容量、原生raw力判据和每结构500步预算，
1600/64=25个同等批次：

- 共享启动/模型，按每批98.272秒优化加.218秒导出串行重复：
  `19.809 + 2.557 + 25*(98.272+.218) ≈ 2485秒`，约**41.4分钟**。
  这是忽略后续输入/工作流准备成本的简化模型，不是严格下界。
- 每批保留本次全部应用准备成本、只摊一次启动：
  `19.809 + 25*101.046 ≈ 2546秒`，约**42.4分钟**。
- 每批独立进程重启、但不重复oracle：
  `25*(19.809+101.046) ≈ 3021秒`，约**50.4分钟**；
  完整探针连oracle重跑25次则约**51.8分钟**。

因此本次实测支持的是**约40–50分钟量级**的粗估，包含500步预算退出，
不是1600项全部收敛的预计耗时、统计置信区间或保证上界。
连续补位可能减少每批尾部小Batch开销，也可能受大结构/邻居数影响；本次没有测量其收益。
输入分布、算法停止口径、CuEq、容量与软件栈均不同，不能把用户提供H100约300秒
当成此次同工作量的性能基线或计算平台加速比。

## 复现与证据

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 720 .venv/bin/python \
  probes/mace_mpa0_batch_timing.py \
  --selection artifacts/mpa0-batch-timing/selection.json \
  --checkpoint "$MACE_MPA0_CHECKPOINT" \
  --output artifacts/mpa0-batch-timing/reproduce-native-batch64
```

checkpoint须指向上述可信锁定模型，输出目录须不存在。
selection.json记录候选数、seed、完整1600抽样清单（仅清单）和前64项统计；
探针消费前64项，不执行剩余1536项。
原始证据：`artifacts/mpa0-batch-timing/` 下资源检查、初轮失败日志/目录、
`fp64-native-batch64.log` 与同名目录的timing.json/results.jsonl、64份CIF、
raw-final-states.npz和verification.json；verify_results.py为本次只读输出核验。
本报告不提交私人模型、原始结构或完整文件清单。

下一阶段建议（待确认）：修正FIRE reference拒绝dt_min=0的本地参数边界，先核对
锁定上游/fork语义，再补CPU边界回归和限时HCU短步验证。目的为恢复上游参数兼容性，
不增加filter、不优化性能、不扩大批量测试。用户确认后先提交本次测量，再实施该修正。

后续记录：用户改为先测试8000原子单批及分项耗时，结果见
[容量诊断报告](g2-mpa0-batch8000-timing.md)；dt_min=0修正仍待独立确认。
