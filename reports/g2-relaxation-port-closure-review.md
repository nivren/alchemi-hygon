# MPA-0 变胞弛豫移植收口审查

日期：2026-10-05。审查 M1–M5、零下限修正、实际使用配置与验证限制，并记录应用分离。
toolkit 收口提交为 `730a848`，已由 merge `620a485` 合入本地 `develop`；应用工程首提交为
`9ae235c`，位于应用仓库本地 `develop`。`github-origin`、`local-origin`、`gitee-origin`
的推送及 refs 核验正在进行。

## 收口结论

原定 M1–M5 的窄移植目标和后续 `FIRE dt_min=0` 兼容修正均已完成。toolkit 已移植的
原生 variable-cell FIRE/FIRE2、`FusedStage` 和 `SizeAwareSampler` 可供下游独立工程
`alchemi-app` 组合成有界批量弛豫。证据包括 CPU reference gate、
合成 HCU 生命周期、真实 MPA-0 HCU4 小批量及 114 项/3959 原子的单次 FP64/FIRE 作业。

“完成”只覆盖已记录的 Torch reference 窄路径。它不代表锁定上游全部 Warp 能力、
全目录1600项成功率、完整 HCU gate、生产等变 GNN 加速、NVIDIA/H100 数值/性能等价，
或两个 optimizer/精度得到相同局部极小值。

## Git 与来源

当前功能分支 `codex/fix-fire-zero-dt-min` 的修复提交为 `0e354f4`，边界收口提交为
`730a848`；整条功能链由 merge commit `620a485` 合入本地 `develop`。remote 推送待完成。
本报告和交接文档已随集成状态提交。M4 应用入口及应用专属测试已拆到同级独立工程 `alchemi-app`；此处继续
追踪 toolkit 的移植、兼容性和集成验证证据。LOCAL-004 是 reference 校验缺陷修正，
没有登记为上游缺陷。

分离后的验证独立运行：toolkit CPU gate 为 **425 passed, 3 skipped, 1 deselected**；
`alchemi-app` 的入口回归为 **9 passed**，CLI `--help` 和 `pyproject.toml` 解析通过。
这次只移动应用文件、隔离应用测试并更新边界文档，没有修改 framework/ops 算法，故未重跑
HCU；之前各里程碑的 HCU 结果仍以对应报告描述的窄范围为准。分离提交分别为 toolkit `730a848`
和 app `9ae235c`。

## 上游保真与源码差异

按 `e6dd99c..0e354f4` 检查 framework、ops 和应用范围，累计28个文件、约4,237行新增和
329行删除；统计包含新建的 ops reference、原有测试及验证/应用文件，不能把总量视作
framework 被整体改写。两个包边界、现有 `FIREVariableCell`/`FIRE2VariableCell`、
`FusedStage`、sampler、Hook 和 sink 公共工作流仍在；新 reference 放在 ops executor 下，
公共优化器接线沿用既有后端选择。`backend=None`/Warp 默认语义保留，没有新增
UnitCellFilter、FrechetCellFilter 或独立 ASE 风格晶胞优化算法。

framework 的实质行为改动集中在已记录的上游缺陷修正和薄接线：M3 变胞更新采用指定 fork
修复，M4 对退出体系的私有 FIRE 状态做冻结并在 refill 后重算 E/F/stress，分别记录为
UBF-011–016， fork 来源与本地适配已在 `UPSTREAM_BUGFIXES.md` 分开说明。晶胞对齐 Hook 的
周期/active 子集和 cached 力、速度、应力坐标系保护也有单独归因。完整晶胞 FIRE 的
`gradient_order=0` 对应该更新 executor 的 forward-only 契约；没有声称优化轨迹可微。

`npt_nph.py` 的改动是把 Warp/ops 导入延迟到对应函数，并将已有 stress-to-cell-force 路径
转接共享转换实现；本次没有移植 NPT/NPH 的 Torch 优化器、barostat 或新的热力学语义。
因此该文件出现在源码 diff 中，不代表 NPT/NPH 已支持。LOCAL-004 则是本地 reference 参数
校验过严的修正：恢复上游允许 `dt_min=0` 的边界，另以本地问题记录，未冒称上游 bug。

新实现中的数值公式/状态语义修正均能对应到 UBF 或 LOCAL 条目和代码注释；新增工作流行为
归属 M4/LP 记录。此次审查未发现需要额外登记、但漏记的上游缺陷，也未把 fork 后的新框架
schema 或额外算法带入移植范围。

## 里程碑验收对照

| 阶段 | 已完成内容 | 证据与边界 |
|---|---|---|
| M1 | MPA-0 来源/模型与应力、力的 CPU 单点基线和差分契约 | [M1 报告](g2-mace-mpa0-m1-cpu-baseline.md)；FP64/FP32 有限真实结构；无 MPA-0 HCU 弛豫 |
| M2 | 原生晶胞对齐、pack/unpack、扩展索引/指针 Torch reference | [M2 报告](g2-cell-tools-m2.md)；CPU/HCU4 primitive；不是 optimizer/filter |
| M3 | 上游六自由度 FIRE2、指定 fork 全 3×3 FIRE 与对齐 Hook | [M3 报告](g2-variable-cell-fire-m3.md)；独立 NumPy/解析势、CPU/HCU4 合成；当时无真实目标 HCU 收敛 |
| M4 | 变胞邻居生命周期、联合阈值、状态冻结、补位 priming、CIF/JSONL 应用 | [M4 报告](g2-mpa0-relaxation-m4.md)；CPU reference/真实 MPA-0 CPU和合成 HCU4；非目标 HCU 实测 |
| M5 | 真实 MPA-0 HCU4、FIRE/FIRE2、FP64/FP32、单体系/补位/较小扩展 Batch | [M5 报告](g2-mpa0-relaxation-m5.md)；48份导出重算，46收敛、2预算退出；不是完整 HCU gate |
| Batch 测量 | 114项/3959原子 FP64 FIRE 单 Batch | [报告](g2-mpa0-batch4000-timing.md)；111收敛、3到500步预算，应用160.145秒；模型/邻居/优化器分项；不同于 M5/H100 判据 |
| LOCAL-004 | 修正本地 full-cell FIRE 将零 `dt_min` 拒绝的问题 | [修复报告](g2-fire-zero-dt-min.md)；CPU gate 434通过/3原有 Warp skip；HCU4 FP64短步；无零下限完整 Batch 收敛数据 |

M5 中 FIRE2 的 Li(CN)2 两个精度案例均用完500步仍未满足应力阈值；FIRE 的 FP64/FP32
Li(CN)2 虽均满足阈值，最终能量/体积明显不同。应按目标精度验收材料输出，不能按单点
CPU/HCU parity 推出不同精度的弛豫轨迹可互换。

## 应用入口与可复现配置

应用代码现位于独立工程 `alchemi-app/src/alchemi_app/relax_mace_mpa0.py`。它从一个目录按文件名排序读取 `*.cif`，可用 `--limit`
取排序前缀；拒绝空/非三维周期结构、奇异晶胞、非有限坐标、部分占位、模型不支持元素，
以及超过 `--max-atoms` 的结构。它校验 checkpoint SHA，运行既有单 optimizer
`FusedStage` 与 `SizeAwareSampler`，写最终 CIF 和 `results.jsonl`，并在导出 CIF 几何上
重算能量、力和应力。输出区分收敛、预算退出、非法输入和计算异常。输入与毕业结果
保存在主机内存，当前不是流式或 restart 服务。

CPU/HCU 有界试跑可按以下方式启动；HCU 上先用 `HIP_VISIBLE_DEVICES=4` 映射为
`cuda:0`。必须指定一个新输出目录：

```bash
cd /home/wangleping/codes/alchemi-app
source ../alchemi-hygon/scripts/activate_hygon_env.sh project
export PYTHONPATH="../alchemi-hygon/packages/framework:../alchemi-hygon/packages/ops:src${PYTHONPATH:+:$PYTHONPATH}"
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 python -m alchemi_app.relax_mace_mpa0 \
  --input-dir /path/to/selected-cifs \
  --output-dir artifacts/mpa0-fire-fp64-run \
  --checkpoint /path/to/mace-mpa-0-medium.model \
  --optimizer fire --device cuda:0 --dtype float64 \
  --max-batch-size 4 --max-atoms 64 --max-steps 500 \
  --max-wall-seconds 180 --dt 0.02 --fmax 0.01 \
  --stress-gpa 0.1 --skin 0
```

该示例入口的默认 optimizer 是 FIRE2，dtype 为 FP64、dt=.02、容量为2个结构/256原子、
每结构最多500步、墙钟120秒、力阈值.01 eV/Å、应力阈值.1 GPa、skin=0。
默认 FIRE2 控制为 `delaystep=60, dtgrow=1.05, dtshrink=.75, alphashrink=.985,
alpha0=.09, tmax=.08, tmin=.005, maxstep=.1`。选择 FIRE 时，`dt_max`/`dt_min`
按上游 wrapper 默认分别为 `10*dt`/`.02*dt`；其他 FIRE 默认是
`maxstep=.2, n_min=5, f_dec=.5, f_inc=1.1, alpha_start=.1, f_alpha=.99`。
应用仅提供 `--dt`，不提供其余逐 optimizer 参数的 CLI 覆盖。

入口的收敛条件是每原子力最大范数与应力最大绝对分量同时达到 `--fmax` 和
`--stress-gpa`；它没有用户 H100 设置的联合 raw 广义力阈值 `.05 eV/Å` 模式，
也没有独立的 `100 eV/Å` 溢出退出阈值。4000测量探针特意配置 native full-cell raw
force (`F_H=-V H^-T stress`) 的联合最大值判据及 `100 eV/Å` 溢出门，
使用 `dt=.1, dt_max=1, dt_min=.002`、容量4000、refill20；不能把该探针数字当作
上述应用默认参数的计时结果。

应用的 `FusedStage` 将 `refill_frequency` 固定为1，示例没有对应 CLI 参数；
4000测量和用户 H100 设置均使用20。入口虽然暴露最大 Batch 结构数和原子数，
实际达到相同补位时序需应用代码修改。

CPU/HCU应用采用普通 e3nn eager、`enable_cueq=False`、`compile_model=False` 和
Torch reference 邻居；没有接入 HCU 等变 GNN 加速算子库。LOCAL-004 验证了 reference
可接受 `dt_min=0`，但当前应用 CLI 未暴露 `dt_min`，默认 FIRE 仍会使用正下限。
应用调用 `model.eval()`；没有像4000性能探针那样显式冻结模型权重。力/应力仍对位置和
应变求导；权重是否冻结主要造成推理图和资源开销设置差异，不代表现有 parity 失效，
但限制了应用入口与 H100运行条件/性能数字的可比性。

对完整用户目录的约束：应用本身按排序目录读取，不实现用户 H100 的
`Random(24).sample(...,1600)` 清单生成；需要复现抽样时应先准备固定选中 CIF 目录，
并保存输入清单/哈希。H100 的34,570候选与本机批量测量的21,755候选也不是同一个已核实集合。
本机4000门槛下实测114项160秒；“1600项约37.8分钟”是按该一批条件外推，不是端到端
1600实测、全部收敛时间或服务吞吐。8000门槛的7937原子初始模型前向 OOM 记录仍有效。

## 后续决策建议

移植与本轮验收已达到当前窄目标；该应用入口已归独立工程维护。若开始长作业，建议在
`alchemi-app` 单独处理其配置，并先确定
停止判据。用户给定的 H100 判据按 ASE `UnitCellFilter(cell_factor=N)` 将原子力和晶胞
形变力组成广义力；4000 原子探针则直接将原子力与 `F_H=-V H^-T stress` 合并取最大范数。
二者的坐标变换/归一化不同，不能视为同一阈值或互换收敛分类。推荐保留原生 FIRE/FIRE2
作为更新器，在应用层只实现所选的停止度量；不引入 ASE filter 或新优化算法。若以 H100
设置为目标，应选 UnitCellFilter 等价的停止度量；若优先与本地4000探针一致，则选 native
raw-force 度量。

确认判据后，下一小阶段只处理薄入口：暴露原生 FIRE/FIRE2 控制项、所选停止度量与
`100 eV/Å` 溢出配置、`refill_frequency`，并在 inference 入口冻结模型参数；输入继续由
预先选定的 CIF 目录提供，不把随机抽样器加入产品入口。验收只做 CPU 配置/停止度量核对
和小型 HCU4 smoke，不重跑4000/1600结构性能作业。

若不要求复现 H100 控制和阈值，现有入口已经能运行小批量真实 MPA-0 结构；后续长作业仍需
按用户选定的结构和预算单独授权。固定晶胞 ASE BFGS、NPT/NPH、DomainParallel、训练、
checkpoint/restart、完整 HCU gate 与生产 HIP/Triton 优化仍是独立范围。
