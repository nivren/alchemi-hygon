# MPA-0 弛豫 M3：原生变胞 FIRE/FIRE2 reference

日期：2026-10-04。分支 `codex/feat-variable-cell-fire`，起点 `d5b5f3c`；
实现/测试/契约提交 `29a70f3`，未合并/推送。
本里程碑完成更新算法和公共优化器接线，批量应用/真实目标 HCU 留待 M4/M5。

## 实现与来源

- NVIDIA framework `4dfe3723def34df3fadb245981081ccf8c94c257`、ops
  `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` 保持锁定；external 只读。
- 指定 fork framework `b12b54c48fc8001adb84654c3b42f2bef21db710`、ops
  `c4c3bb56b6c6ae0d5e85b4e1ed14bd4bd4748a2a` / `eadc3eeebaeceb832f7341eb59303dc588f857bd`
  只摘取相关函数语义，不同步新版 schema/依赖。
- FIRE2 原生六项、原子数 cell-force 归一化、新 alpha/旧 dt mixing、上坡回退与
  Cartesian 位移限制保留。fork 完整 FIRE 按原子质量/晶胞质量1分别 half-kick，
  全九项共同控制，原子/cell-row 共同 cap。两种算法没有统一为 ASE filter。
- 新 ops-owned [reference](../packages/ops/nvalchemiops/cell_fire_reference.py)、
  implementation `torch_reference.fire2-cell-v1` / `torch_reference.fire-cell-v1`，
  entrypoint 由既有 generic executor 绑定。公共优化器参数/default 保持；None/warp 保留。
- AlignCellHook 摘取 fork 周期/active 子集保护，本地补齐 cached F/v/stress 协变；
  后端 keyword 和冻结 selection 是适配；已有晶胞运动时拒绝需要再次对齐的调用。
- [UBF-011–014](../docs/UPSTREAM_BUGFIXES.md) 和 LP-076–080 区分 fork 修复、本地修复、
  backend 接线及合同加强；源码加注释。不是新增 ASE、NPT/NPH 或科学参数化。

## CPU 测试

1. [ops39项](../packages/ops/test/torch/test_variable_cell_fire.py)：独立 NumPy 逐体系
   oracle，FP32/64、正/负/零/纯晶胞功率、延迟阈值、耦合仿射、实际位移限制、九项混合、
   不同原子质量、预分配/外部归约、alias/错误/空输入与非连续原地状态。
2. [公共18项](../packages/framework/test/compatibility/test_variable_cell_fire_reference.py)：
   160步解析矩阵势能降至初始的千分之一以下；分数坐标守恒；单次 selection、第二 executor、
   no-Warp import、default legacy handler guard、Hook 周期/active 子集/缓存协变/重复调用。
3. `scripts/check_cpu_reference.sh` 最终门的五个 pytest 进程、compileall 和 diff-check
   分别记录于 `artifacts/variable-cell-fire-m3/cpu-gate-final.log`；最终392 passed、3 skipped，
   退出0（分进程149/1/196/22/24 passed）。第一轮完整门389 passed、3 skipped、退出0；
   补齐 Hook active 回归并修正 context 差异后，以上最终完整门重跑通过。
   三个 skip 是现有 NHC 的实际 Warp 对照，无新增 skip/xfail。

FP64 单步 rtol/atol=`2e-12`；FP32=`3e-5`；索引/计数精确。验证不会放宽其他算子容差。
最终代码定向 Ruff、compileall、git diff-check 通过；完整 Warp 行为套件未运行。

## HCU4 合成数值证据

[探针](../probes/variable_cell_fire_reference.py)：`HIP_VISIBLE_DEVICES=4`、逻辑 cuda:0，
DTK26.04 / Torch2.9.0 / HIP6.3.26093 / BW1000 64G gfx936，180秒时限、退出0。
前后 hy-smi 八卡 Normal、HCU4 0%计算/显存、没有 KFD PID；未使用其他卡。
设备内 Torch 计算，CPU 仅作独立对照及最终证据收集，未 CPU fallback/降精度。

| 验证 | FP64 | FP32 |
|---|---|---|
| 两算法×四分支，每精度8组单步 CPU/HCU 最大绝对差 | `4.44e-16` | `1.46e-11` |
| 两公共优化器各25步解析势，分数坐标最大绝对误差 | `5.55e-16` | `1.79e-7` |

共16组单步、4组公共短运行，异构 `[2,3]`；FIRE2 用六项、FIRE 包含全部九项。
FP64 解析势初值 `[0.07565,0.161404125]`：FIRE2 25步到 `[0.0426713,0.1116209]`；
FIRE 到 `[5.38476e-5,1.14887e-4]`。不据此比较两算法材料收敛性能。
`artifacts/variable-cell-fire-m3/{cpu,hcu4}.json/.log` 保存原始结果。
没有 benchmark/warmup/JIT 性能结论；只有 eager 正确性和调用末尾同步。

## MPA-0 CPU 三步接线

[探针](../probes/mace_mpa0_variable_cell_short.py) 使用 M1 同一可信本地 checkpoint，
SHA256=`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
三个真实 CIF：Li2MgGe(4)、LiSnCl3(10)、NaLiTl2(4)，异构 Batch ptr=`[0,4,14,18]`；
源文件内容 SHA 记录于 `mpa0-cpu-short.json`。不写回输入或自动下载。
两种优化器×FP64/FP32，共4组各3步；ordinary e3nn eager，cueq/compile关闭，
显式激活 E/F/stress；MACE ZBL 默认 dtype 按 M1 配置且 finally 恢复。
已有 NeighborListHook、skin=0 每次评估重建；610 edges、无跨体系 edge。

- r/H 均更新、体积为正、E/F/stress 有限且 dtype/device 不变。
- 最终几何独立重建邻居重算 E/F/stress，与 Batch 三种输出最大差均0。
- FIRE2 输入额外旋转0.37弧度再对齐；能量不变、力/应力同帧：FP64 最大差 E
  `1.78e-15 eV`、F `2.70e-14 eV/Å`、stress `1.49e-16 eV/Å³`；FP32 分别
  `1.91e-6`、`6.44e-6`、`1.12e-7`，满足 M1 分物理量容差。
- FP64 FIRE2 E：`[-11.390930934,-36.038673464,-4.300590412]` 到
  `[-11.390935954,-36.039311231,-4.300625785]`；FIRE 到
  `[-11.390948540,-36.039365115,-4.300596549]`。仅三步，convergence_status=`not_evaluated`。

日志/JSON 在 `artifacts/variable-cell-fire-m3/mpa0-cpu-short.*`；末次退出0，约25.78秒。
早期 probe 夹具的质量维度、输出slot、Hook stage/合成零状态断言修正后重跑通过；
这些是验证夹具问题，没有为过测试改产品算法或放宽门限。
补齐 fork 掩码时曾直接引用新版 context 字段，完整门出现5项失败；改为使用当前
status/exit_status 取得等价掩码，不移植新 context/schema，公共18项和完整门重新验证。

## 验证边界与下一步

未验证 actual Warp/NVIDIA、目标 MPA-0 HCU、材料完整收敛、全目录成功率、退出/补位、
skin>0变胞缓存、完整 checkpoint、NPT/NPH、DomainParallel、轨迹梯度/compile、
生产 HIP/Triton 或性能。保留原高级接口不等于这些功能已在 DCU 准入。
FIRE2 非空 Batch 的零原子成员拒绝；M2 packing 空成员支持不扩张为优化器支持。
full-cell FIRE 不需对齐 Hook；不能把 Hook 当运行中反复重参数化工具。

下一里程碑 M4：用现有 Dynamics/Hooks 接变胞邻居/联合收敛/退出补位，
添加薄 CIF 批量入口与最终结构、逐体系状态导出；用户确认后实施。
M5 再做目标模型小样本 HCU4、FP64→FP32/单体系→异构 Batch 验收。
